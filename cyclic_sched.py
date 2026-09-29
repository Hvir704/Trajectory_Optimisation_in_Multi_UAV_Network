"""cyclic_sched.py (v2: surrogate breach handling fixed to match DynSim) -- option C4: optimised cyclic schedule, seeded by the published clustering method.

WHAT IT IS
Each UAV k owns a disjoint sensor set and a cyclic visit SEQUENCE over it. At every launch the UAV
takes the next contiguous run of its sequence that fits the energy budget, with exactly the cutting
rule of cluster_patrol (rr_planner.build_cluster_planner): skip sensors unreachable on their own,
stop at the first misfit, which opens the next sortie. The difference from cluster patrol is the
sequence:
  * sensor i has a frequency class c_i in {0..CMAX}: it appears in every 2^c_i-th sub-cycle, at
    phase phi_i. P = 2^max(c) sub-cycles per frame; each sub-cycle visits its sensors in the order
    of one master depot tour of the UAV's set (a sub-tour of a tour: shortcutting, never longer).
  * the sets are chosen by optimisation, not fixed by k-means.
All classes 0 and the k-means sets reproduce cluster patrol EXACTLY (sequence = its tour order;
tested by `python cyclic_sched.py selftest`).

SURROGATE (planner-side information only: positions, priorities, energy/flight/data parameters,
NOMINAL generation rate). Sensor sets are disjoint, so the UAVs are independent: each UAV is
simulated alone over the same window as DynSim (launch stagger k t_c/K, burn-in, horizon), with the
same launch-time dwell estimate, the same cutting rule, the same reserve check at arrival, and the
exact weighted-age integral. Not modelled: events (their rates and locations are unknown to every
planner), true per-sensor generation rates (unknown to every planner), the fleet-wide sortie-length
average (per-UAV average used instead). The fidelity check measures what these omissions cost.

SEARCH (all knobs pre-declared, see CFG): first-improvement local search from each initialisation,
moves = class up/down (both phase choices) and relocation to a neighbouring UAV (cheapest insertion
into its master tour), then a master-tour re-optimisation pass (NN + 2-opt). The schedule with the
lowest surrogate J over the initialisations is deployed. Because the k-means / class-0 start IS the
cluster-patrol schedule and only improving moves are accepted, surrogate J <= cluster patrol's
surrogate J by construction. Whether the real simulator agrees is the empirical question.
"""
from __future__ import annotations
import math, sys, time
from typing import List
import numpy as np
from rr_planner import _kmeans, tour_order

CFG = dict(CMAX=3, NRELOC=2, MAX_PASSES=6, MAX_EVALS=20000, REL_TOL=1e-4,
           INITS=("kmeans", "sector"), LS_SEED=0,
           SUR_BURN=3 * 3600.0, SUR_H=12 * 3600.0)   # surrogate design window: fixed constants,
                                                    # NOT read from DynParams (planners never see the clock)


# ------------------------------------------------------------------------------------------------
# shared cutting rule (planner AND surrogate call this, so they cannot drift apart)
# ------------------------------------------------------------------------------------------------
def cut_sortie(seq, ptr, D, dh, hov, alone, excl, E_usable, Pf_v):
    """seq: list of sensor ids (may repeat). D: (M+1)x(M+1) list-of-lists, depot = index M.
    dh[j] distance j->depot, hov[j] hover energy estimate, alone[j] = 2 fly dh + hov,
    Pf_v = Pf / v (J per metre). Returns (route, new_ptr). Identical to cluster_patrol when
    seq has no repeats; with repeats, meeting a sensor already on the route ends the sortie there."""
    n = len(seq)
    if n == 0:
        return [], ptr
    route = []; inroute = set()
    cur = len(D) - 1; E = E_usable; stop_at = None; last_pos = None
    for step in range(n):
        pos_ = (ptr + step) % n
        j = seq[pos_]
        if excl[j] or alone[j] > E_usable:
            continue
        if j in inroute:
            stop_at = pos_; break
        e_leg = Pf_v * D[cur][j] + hov[j]
        if e_leg + Pf_v * dh[j] <= E + 1e-6:
            route.append(j); inroute.add(j); E -= e_leg; cur = j; last_pos = pos_
        else:
            stop_at = pos_; break
    if stop_at is None:
        stop_at = (last_pos + 1) % n if route else ptr
    return route, stop_at


def build_sequence(master, cls, phi):
    """Frame of P = 2^max(c) sub-cycles; sub-cycle s holds i iff s mod 2^c_i == phi_i."""
    if not master:
        return []
    cm = max(cls[i] for i in master)
    P = 1 << cm
    if P == 1:
        return list(master)
    seq = []
    for s in range(P):
        for i in master:
            if s % (1 << cls[i]) == phi[i]:
                seq.append(i)
    return seq


# ------------------------------------------------------------------------------------------------
# instance context
# ------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, pos, home, w, p, K):
        self.pos = np.asarray(pos, float); self.home = np.asarray(home, float)
        self.M = len(pos); self.K = K; self.p = p
        self.w = [float(x) for x in w]
        P = np.vstack([self.pos, self.home[None, :]])
        Dm = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        self.D = Dm.tolist()
        self.dh = Dm[:self.M, self.M].tolist()
        self.Pf_v = p.Pf / p.v
        self.E_usable = (1.0 - p.rho) * p.Emax / K
        self.lam = p.lam_bits                     # NOMINAL rate: the planner does not know the true ones
        self.t_c = p.t_c
        self.burn, self.H = CFG["SUR_BURN"], CFG["SUR_H"]

    def uav_J(self, seq, members, k):
        """Weighted-age integral over [burn, H] of UAV k's sensors when it flies `seq` alone."""
        p = self.p; D = self.D; dh = self.dh; lam = self.lam; R = p.R; B = p.B_bits
        Ph = p.Ph; v = p.v; Pf_v = self.Pf_v; Eu = self.E_usable; burn, H = self.burn, self.H
        M = self.M
        tlast = {i: 0.0 for i in members}
        integ = {i: 0.0 for i in members}
        excl = _NoExcl
        t = k * self.t_c / self.K
        Ts = None; ptr = 0
        hov = [0.0] * M; alone = [0.0] * M
        n = len(seq)
        while t < H:
            if n:
                Tsx = max(Ts if Ts is not None else self.t_c, self.t_c)
                for i in members:
                    de = min((t - tlast[i] + 0.5 * Tsx) * lam, B) / R
                    hov[i] = Ph * de
                    alone[i] = 2.0 * Pf_v * dh[i] + hov[i]
                route, ptr = cut_sortie(seq, ptr, D, dh, hov, alone, excl, Eu, Pf_v)
            else:
                route = []
            t0 = t; E = Eu; cur = M; nvis = 0
            for j in route:
                d = D[cur][j]
                ta = t + d / v
                dw = min((ta - tlast[j]) * lam, B) / R
                e_step = Pf_v * d + Ph * dw
                if e_step + Pf_v * dh[j] > E:
                    # reserve breach at arrival, exactly as DynSim: the leg IS flown (time passes),
                    # the node is not served, the drone heads home from its previous position.
                    # (v1 broke out without advancing time: with a one-sensor sequence in the
                    # hover-estimate gray band this never advanced the clock -> infinite loop.)
                    t = ta
                    break
                a = max(tlast[j], burn); b = min(ta, H)
                if b > a:
                    integ[j] += 0.5 * ((b - tlast[j]) ** 2 - (a - tlast[j]) ** 2)
                tlast[j] = ta
                t = ta + dw; E -= e_step; cur = j; nvis += 1
            t += D[cur][M] / v                              # land
            dur = t - t0
            Ts = dur if Ts is None else 0.9 * Ts + 0.1 * dur
            if nvis == 0:
                t += self.t_c                               # DynSim's empty-sortie turnaround
            if not t > t0:
                raise RuntimeError(f"surrogate clock did not advance (UAV {k}, t={t})")
        J = 0.0
        for i in members:                                   # tail up to H
            a = max(tlast[i], burn)
            if H > a:
                integ[i] += 0.5 * ((H - tlast[i]) ** 2 - (a - tlast[i]) ** 2)
            J += self.w[i] * integ[i]
        return J


class _NoExclT:
    def __getitem__(self, j): return False
_NoExcl = _NoExclT()


# ------------------------------------------------------------------------------------------------
# schedule state + local search
# ------------------------------------------------------------------------------------------------
class Schedule:
    def __init__(self, ctx: Ctx, labels):
        self.ctx = ctx; K = ctx.K; M = ctx.M
        self.a = [int(x) for x in labels]
        self.c = [0] * M; self.phi = [0] * M
        self.master = []
        for k in range(K):
            idx = np.where(np.asarray(self.a) == k)[0]
            self.master.append([int(x) for x in (idx[tour_order(ctx.pos[idx], ctx.home)] if len(idx) else idx)])
        self.Jk = [self._eval(k, self.master[k]) for k in range(K)]
        self.n_evals = 0

    def _eval(self, k, master, c=None, phi=None):
        c = self.c if c is None else c; phi = self.phi if phi is None else phi
        return self.ctx.uav_J(build_sequence(master, c, phi), master, k)

    @property
    def J(self):
        return sum(self.Jk) / (self.ctx.H - self.ctx.burn)

    def sequences(self):
        return [build_sequence(self.master[k], self.c, self.phi) for k in range(self.ctx.K)]

    def _insert(self, master, i):
        """Cheapest insertion of i into the closed depot tour `master`."""
        D = self.ctx.D; M = self.ctx.M
        best, bpos = math.inf, 0
        nodes = [M] + master + [M]
        for q in range(len(nodes) - 1):
            u, v_ = nodes[q], nodes[q + 1]
            dlt = D[u][i] + D[i][v_] - D[u][v_]
            if dlt < best: best, bpos = dlt, q
        return master[:bpos] + [i] + master[bpos:]

    def search(self, cfg=CFG, log=None):
        ctx = self.ctx; K = ctx.K; M = ctx.M
        rng = np.random.default_rng(cfg["LS_SEED"])
        tol = cfg["REL_TOL"]
        for pas in range(cfg["MAX_PASSES"]):
            improved = False
            for i in rng.permutation(M):
                i = int(i); k = self.a[i]
                if self.n_evals > cfg["MAX_EVALS"]: break
                # -- class moves (same UAV)
                cands = []
                if self.c[i] < cfg["CMAX"]:
                    for ph in (self.phi[i], self.phi[i] + (1 << self.c[i])):
                        cands.append((self.c[i] + 1, ph))
                if self.c[i] > 0:
                    cands.append((self.c[i] - 1, self.phi[i] % (1 << (self.c[i] - 1))))
                best = None
                for cc, ph in cands:
                    c2 = list(self.c); p2 = list(self.phi); c2[i] = cc; p2[i] = ph
                    Jn = self._eval(k, self.master[k], c2, p2); self.n_evals += 1
                    if Jn < self.Jk[k] * (1 - tol) and (best is None or Jn < best[0]):
                        best = (Jn, cc, ph)
                if best is not None:
                    self.Jk[k] = best[0]; self.c[i] = best[1]; self.phi[i] = best[2]; improved = True
                # -- relocation to neighbouring UAVs (empty UAVs always candidates)
                dist = []
                for kk in range(K):
                    if kk == k: continue
                    if not self.master[kk]: dist.append((-1.0, kk)); continue
                    dist.append((min(ctx.D[i][j] for j in self.master[kk]), kk))
                dist.sort()
                src = [j for j in self.master[k] if j != i]
                Js = self._eval(k, src); self.n_evals += 1
                best = None
                for _, kk in dist[:cfg["NRELOC"]]:
                    dst = self._insert(self.master[kk], i)
                    Jd = self._eval(kk, dst); self.n_evals += 1
                    gain = (self.Jk[k] + self.Jk[kk]) - (Js + Jd)
                    if gain > tol * (self.Jk[k] + self.Jk[kk]) and (best is None or gain > best[0]):
                        best = (gain, kk, dst, Jd)
                if best is not None:
                    _, kk, dst, Jd = best
                    self.master[k] = src; self.Jk[k] = Js
                    self.master[kk] = dst; self.Jk[kk] = Jd; self.a[i] = kk; improved = True
            # -- master-tour re-optimisation (keep only if the surrogate improves)
            for k in range(K):
                if len(self.master[k]) < 4: continue
                idx = np.array(self.master[k])
                m2 = [int(x) for x in idx[tour_order(ctx.pos[idx], ctx.home)]]
                Jn = self._eval(k, m2); self.n_evals += 1
                if Jn < self.Jk[k] * (1 - tol):
                    self.master[k] = m2; self.Jk[k] = Jn; improved = True
            if log: log(f"    pass {pas}: J_sur={self.J:.4e} evals={self.n_evals} "
                        f"classes={np.bincount(self.c, minlength=cfg['CMAX']+1).tolist()}")
            if not improved or self.n_evals > cfg["MAX_EVALS"]:
                break
        return self


def sector_labels(pos, home, K):
    """Equal-count angular sectors about the depot (sweep partition): every UAV owns near AND far."""
    th = np.arctan2(pos[:, 1] - home[1], pos[:, 0] - home[0])
    order = np.argsort(th, kind="stable")
    lab = np.empty(len(pos), int)
    for r, i in enumerate(order):
        lab[i] = min(K - 1, r * K // len(pos))
    return lab


def optimise(pos, home, w, p, K, cfg=CFG, log=None, noopt=False, inits=None):
    """noopt=True: no search. inits=None: cfg INITS (with noopt: k-means only = cluster patrol)."""
    ctx = Ctx(pos, home, w, p, K)
    out = {}
    if inits is None:
        inits = ("kmeans",) if noopt else cfg["INITS"]
    for init in inits:
        lab = _kmeans(np.asarray(pos, float), K, seed=0) if init == "kmeans" else sector_labels(np.asarray(pos), home, K)
        S = Schedule(ctx, lab)
        J0 = S.J
        if not noopt:
            S.search(cfg, log=log)
        out[init] = (S, J0)
        if log: log(f"  init={init}: J_sur start={J0:.4e} end={S.J:.4e} ({S.J/J0-1:+.2%})")
    best = min(out, key=lambda k: out[k][0].J)
    return out[best][0], best, out


# ------------------------------------------------------------------------------------------------
# runtime planner (same bind_sim pattern as cluster_patrol)
# ------------------------------------------------------------------------------------------------
def build_cyclic_planner(noopt=False, inits=None, cfg=CFG, verbose=False):
    """cyc: full method. cyc_noopt (noopt=True): must equal cluster_patrol exactly.
    cyc_sector0 (noopt=True, inits=("sector",)): ablation -- depot-aware partition, no search."""
    st = {"seq": None, "ptr": None, "k": 0, "info": None}

    def bind_sim(sim):
        orig = sim._plan
        def _plan(k, *a, **kw):
            st["k"] = k
            return orig(k, *a, **kw)
        sim._plan = _plan
        st["K"] = sim.p.K

    def planner(req) -> List[int]:
        if req.start is not None:
            raise ValueError("cyclic schedule is launch-time only (use --replan launch)")
        p = req.p
        if st["seq"] is None:
            t = time.time()
            S, init, out = optimise(req.pos, req.home, req.weight_est, p, st["K"], cfg,
                                    log=(print if verbose else None), noopt=noopt, inits=inits)
            st["seq"] = S.sequences(); st["ptr"] = [0] * st["K"]
            st["ctx"] = S.ctx
            st["info"] = dict(init=init, J_sur=S.J, J_sur_start={k: v[1] for k, v in out.items()},
                              classes=np.bincount(S.c, minlength=cfg["CMAX"] + 1).tolist(),
                              build_s=time.time() - t)
        ctx = st["ctx"]; k = st["k"] % st["K"]
        hov = p.e_hover(req.dwell_est)
        alone = p.e_fly(2.0 * np.asarray(ctx.dh)) + hov
        excl = np.zeros(ctx.M, bool) if req.excluded is None else np.asarray(req.excluded, bool)
        route, st["ptr"][k] = cut_sortie(st["seq"][k], st["ptr"][k], ctx.D, ctx.dh,
                                         hov.tolist(), alone.tolist(), excl.tolist(), req.E_usable, ctx.Pf_v)
        return route

    planner.bind_sim = bind_sim
    planner.state = st
    return planner


# ------------------------------------------------------------------------------------------------
# self-test: class-0 k-means schedule must reproduce cluster_patrol bit-for-bit in DynSim
# ------------------------------------------------------------------------------------------------
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "selftest":
    from dyn_env import DynParams, DynSim
    from rr_planner import build_cluster_planner
    ok = True
    for lay, M, E, K, s in [("core", 100, 1.5e6, 6, 13), ("ring", 100, 3e6, 5, 14), ("paper", 50, 1.5e6, 4, 13)]:
        Js = []
        for pl in (build_cluster_planner(), build_cyclic_planner(noopt=True)):
            p = DynParams(M=M, K=K, Emax=E, layout=lay, coord_mode="exclude", T_horizon=12 * 3600.0, T_burnin=3 * 3600.0)
            sim = DynSim(p, pl, seed=s, coordinate=True); pl.bind_sim(sim); Js.append(sim.run()["J_timeavg"])
        same = Js[0] == Js[1]; ok &= same
        print(f"{lay} M={M} E={E:.1e} K={K} s={s}: cluster={Js[0]:.6e} cyc_noopt={Js[1]:.6e} identical={same}")
    print("SELFTEST", "PASS" if ok else "FAIL")
