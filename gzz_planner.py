"""gzz_planner.py -- external baseline: Gao, Zhu, Zhai, "AoI-Sensitive Data Collection in
Multi-UAV-Assisted Wireless Sensor Networks", IEEE TWC 22(8):5185-5197, 2023,
doi 10.1109/TWC.2022.3232366.

Their pipeline: (1) SCADC clusters sensors into collection points (CPs); (2) CUKK-means
(kernel k-means whose centroids include the data centre, Eq. 24) gives each UAV one CP cluster;
(3) AOTPACO (ACO with AHP-initialised pheromone and a Pareto archive over average and peak
AoI, Alg. 3) orders each UAV's CPs into endurance-feasible round trips from/to the data centre.

ADAPTATION (every point is a disclosed deviation or a filled gap; see GZZ_DEVIATIONS below):
  * One sensor per CP. Our sensors are static and data are collected by hovering at the sensor;
    the paper's Fig. 4 itself evaluates the one-to-one SN-CP association.
  * K is an input (we sweep K). The paper's own K-selection rule is gzz_k_choice() below,
    reported as a secondary result only.
  * Endurance = our energy budget E_usable (flight Pf, hover Ph), not a time T_max. Offloading
    at the depot is instantaneous (as in our simulator), so its term is zero.
  * The plan is built ONCE per simulation (offline, as in the paper) from planner-side
    information only: positions, depot, nominal generation rate p.lam_bits, buffer, link rate,
    E_usable. No priorities (the paper's AoI is unweighted). No ages, no true rates/backlogs.
  * Continuous operation: UAV k flies its planned round trips cyclically, one per sortie. At
    launch the trip is checked against the live dwell estimate (req.dwell_est): sensors
    excluded / unreachable alone are skipped, and the trip is truncated if energy runs short
    (the truncated sensors wait for the next cycle; counted in state['trunc_nodes']).
  * Nominal dwell for the offline plan (fixed on pilot seeds 13-15, registered before 1-12):
      dwell_mode="full" : every sensor's buffer full, B/R (conservative)
      dwell_mode="rev"  : dwell at the UAV's own planned cycle time, min(lam*T_k, B)/R,
                          from one fixed-point pass (plan with "full", measure T_k, replan).
Launch-time only (a route is a fixed planned trip; per-leg replanning is undefined).
"""
from __future__ import annotations
import time
from typing import List
import numpy as np
from dyn_env import SortieRequest

GZZ_DEVIATIONS = {
    "collection_points": "one sensor per CP (SCADC not run; paper Fig.4 one-to-one configuration)",
    "kernel": "linear (paper's 'new kernel function' is not given) -> k-means with depot in every centroid (Eq.24)",
    "cukk_init": "random equal-size assignment, seeded; empty cluster re-seeded with its worst-fit point",
    "endurance": "energy E_usable with Pf/Ph instead of time T_max; offload time 0",
    "heuristic_eq27": "1/(dis+1): rt and exp(-AoI) are common to all candidates at a step and cancel in Eq.26",
    "ahp": "a=1 (equal importance); distances measured from the depot; nearer and larger-data CPs weighted up",
    "pheromone": "directed tau(i->j); lam1=lam2=1, mu1=mu2=1, w1=w2=0.5, AoI in seconds; per-solution denominators in Eq.38",
    "pareto": "dominance on raw y,z (order-equivalent to exp(-y), exp(-z)); NumPos=10; FDR on relative deviations",
    "deployed_solution": "archive member with least average AoI (P1), ties by peak AoI",
    "algorithm3_trip_logic": "return to depot and open a new round trip when no CP fits the remaining endurance",
}
ACO_DEFAULTS = dict(n_ants=50, iters=200, alpha=1.0, beta=5.0, rho=0.1,   # paper Sec. VI
                    lam1=1.0, lam2=1.0, mu1=1.0, mu2=1.0, w1=0.5, w2=0.5,  # unspecified -> filled
                    num_pos=10, ahp_a=1.0)


# ---------------------------------------------------------------------------- CUKK-means
def cukk_partition(X: np.ndarray, home: np.ndarray, K: int, rng: np.random.Generator,
                   max_iter: int = 200) -> np.ndarray:
    """Alg. 2 step 1 with a linear kernel: centroid o_k = (sum_{l in k} w_l + w_0)/(n_k + 1)."""
    n = len(X); K = max(1, min(K, n))
    lab = rng.permutation(np.arange(n) % K)
    for _ in range(max_iter):
        C = np.array([(X[lab == k].sum(0) + home) / (np.sum(lab == k) + 1) for k in range(K)])
        d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = np.argmin(d2, axis=1)
        for k in range(K):
            if not np.any(new == k):
                far = int(np.argmax(d2[np.arange(n), new])); new[far] = k
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


# ---------------------------------------------------------------------------- AOTPACO
def _evaluate(trips, D, dwell, n_clu, v):
    """Paper Eqs. (9),(11),(12),(40),(41) with one sensor per CP and zero offload time.
    D: (n+1,n+1) distances, depot index n. Returns y, z, Gbar, Gp."""
    dep = D.shape[0] - 1
    y = z = 0.0; ages = []; Gp = 0.0
    for tr in trips:
        L = len(tr)
        f = [D[tr[l], tr[l + 1]] / v for l in range(L - 1)]
        phi = [dwell[j] for j in tr]
        y += sum(f)
        z += sum((l / n_clu) * phi[l] for l in range(1, L)) + sum(((l + 1) / n_clu) * f[l] for l in range(L - 1))
        back = D[tr[-1], dep] / v
        tail = back                                   # Eq.(9) backwards: own upload + rest + flight home
        for l in range(L - 1, -1, -1):
            tail += phi[l] + (f[l] if l < L - 1 else 0.0)
            ages.append(tail)
        Gp = max(Gp, ages[-1])                        # Eq.(12): first sensor of the trip
    Gbar = float(np.mean(ages)) if ages else 0.0
    return y, z, Gbar, Gp


def _dominates(a, b):
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def aotpaco(Xn: np.ndarray, home: np.ndarray, dwell: np.ndarray, E_budget: float, p,
            rng: np.random.Generator, **kw):
    """Alg. 3 for one UAV's CP cluster. Returns (trips as lists of LOCAL indices, info)."""
    c = {**ACO_DEFAULTS, **kw}
    n = len(Xn)
    if n == 0:
        return [], dict(dropped=[], Gbar=0.0, Gp=0.0)
    X = np.vstack([Xn, home[None]]); dep = n
    D = np.linalg.norm(X[:, None, :] - X[None], axis=-1)
    eh = p.e_hover(dwell); d_home = D[:n, dep]
    serv = (p.e_fly(2 * d_home) + eh) <= E_budget + 1e-6      # reachable on its own
    etab = (1.0 / (D[:, :n] + 1.0)) ** c["beta"]                # (n+1, n) heuristic, Eq.(27) reduced

    # AHP initial pheromone, Eqs.(28)-(33): criterion weights from C, scheme weights per criterion
    Cm = np.array([[1.0, c["ahp_a"]], [1.0 / c["ahp_a"], 1.0]])
    ev, evec = np.linalg.eig(Cm); wc = np.abs(np.real(evec[:, np.argmax(np.real(ev))])); wc /= wc.sum()
    w_dist = 1.0 / np.maximum(d_home, 1.0); w_dist /= w_dist.sum()     # nearer -> more important
    w_data = dwell / dwell.sum() if dwell.sum() > 0 else np.full(n, 1.0 / n)
    tau = np.tile(wc[0] * w_dist + wc[1] * w_data, (n + 1, 1))         # (n+1, n), directed

    A = c["n_ants"]
    archive = []          # entries: (y, z, Gbar, Gp, trips)
    for it in range(c["iters"]):
        cur = np.full(A, dep); E = np.full(A, float(E_budget))
        vis = np.tile(~serv, (A, 1)); trips = [[[]] for _ in range(A)]
        done = vis.all(1)
        while not done.all():
            act = np.where(~done)[0]
            need = p.e_fly(D[cur[act]][:, :n]) + eh[None] + p.e_fly(d_home)[None]
            feas = (~vis[act]) & (need <= E[act, None] + 1e-6)
            has = feas.any(1)
            mv = act[has]
            if len(mv):
                wgt = (tau[cur[mv]] ** c["alpha"]) * etab[cur[mv]] * feas[has]
                cs = np.cumsum(wgt, 1); u = rng.random(len(mv)) * cs[:, -1]
                j = np.minimum((cs < u[:, None]).sum(1), n - 1)
                E[mv] -= p.e_fly(D[cur[mv], j]) + eh[j]
                cur[mv] = j; vis[mv, j] = True
                for a_, jj in zip(mv, j):
                    trips[a_][-1].append(int(jj))
            for a_ in act[~has]:                     # nothing fits: go home, open a new round trip
                cur[a_] = dep; E[a_] = E_budget
                if trips[a_][-1]:
                    trips[a_].append([])
            done = vis.all(1)
        sols = []
        for a_ in range(A):
            tr = [t for t in trips[a_] if t]
            y, z, Gb, Gp = _evaluate(tr, D, dwell, n, p.v)
            sols.append((y, z, Gb, Gp, tr))
        nd = [s for s in sols if not any(_dominates(o[:2], s[:2]) for o in sols)]
        current = nd[int(rng.integers(len(nd)))]
        for s in nd:
            if any(_dominates(o[:2], s[:2]) or o[:2] == s[:2] for o in archive):
                continue
            archive = [o for o in archive if not _dominates(s[:2], o[:2])] + [s]
        while len(archive) > c["num_pos"]:                   # Eqs.(43)-(47), relative form
            ay = np.mean([o[0] for o in archive]); az = np.mean([o[1] for o in archive])
            fdr = [np.hypot(max(0.0, o[0] - ay) / ay, max(0.0, o[1] - az) / az) for o in archive]
            archive.pop(int(np.argmax(fdr)))
        tau *= (1.0 - c["rho"])                              # Eq.(34)
        def _deposit(sol, amt):
            den = c["mu1"] * sol[2] ** c["w1"] + c["mu2"] * sol[3] ** c["w2"]
            for tr in sol[4]:
                prev = dep
                for j in tr:
                    tau[prev, j] += amt / den; prev = j
        _deposit(current, c["lam1"])                         # Eq.(36)
        for o in archive:                                    # Eq.(38)
            _deposit(o, c["lam2"])
    best = min(archive, key=lambda o: (o[2], o[3]))
    return best[4], dict(dropped=list(np.where(~serv)[0]), Gbar=best[2], Gp=best[3],
                         n_trips=len(best[4]), archive=len(archive))


def _cycle_time(trips, X, home, dwell, v):
    T = 0.0
    for tr in trips:
        prev = home
        for j in tr:
            T += np.linalg.norm(X[j] - prev) / v + dwell[j]; prev = X[j]
        T += np.linalg.norm(home - prev) / v
    return T


def build_plan(pos, home, p, K, seed, dwell_mode="full", **aco_kw):
    """Offline plan for the whole fleet: list over UAVs of round trips (GLOBAL sensor indices)."""
    t0 = time.time()
    rng = np.random.default_rng(seed * 1_000_003 + 7919 * K + 17)
    lab = cukk_partition(pos, home, K, rng)
    full = np.full(len(pos), p.B_bits / p.R)
    plan, info = [], []
    for k in range(K):
        idx = np.where(lab == k)[0]
        dwell = full[idx].copy()
        passes = 1 if dwell_mode == "full" else 2
        for ps in range(passes):
            r = np.random.default_rng(seed * 1_000_003 + 7919 * K + 101 * k + ps)
            trips, inf = aotpaco(pos[idx], home, dwell, p.E_usable, p, r, **aco_kw)
            if dwell_mode == "rev" and ps == 0:
                T_k = _cycle_time(trips, pos[idx], home, dwell, p.v)
                dwell = np.full(len(idx), min(p.lam_bits * T_k, p.B_bits) / p.R)
        plan.append([[int(idx[j]) for j in tr] for tr in trips])
        info.append(dict(n=len(idx), n_trips=inf.get("n_trips", 0), dropped=len(inf["dropped"]),
                         T_cycle=_cycle_time(trips, pos[idx], home, dwell, p.v)))
    return plan, dict(per_uav=info, build_s=time.time() - t0, labels=lab)


# ---------------------------------------------------------------------------- planner
def build_gzz_planner(dwell_mode="full", seed=0, **aco_kw):
    """Factory -> SortiePlanner. Build fresh per DynSim; run_grid calls planner.bind_sim(sim)."""
    st = {"plan": None, "ptr": None, "k": 0, "K": None, "trunc_sorties": 0, "trunc_nodes": 0,
          "skipped": 0, "calls": 0, "info": None, "dwell_mode": dwell_mode}

    def bind_sim(sim):
        orig = sim._plan
        def _plan(k, *a, **kw):
            st["k"] = k
            return orig(k, *a, **kw)
        sim._plan = _plan
        st["K"] = sim.p.K

    def planner(req: SortieRequest) -> List[int]:
        if req.start is not None:
            raise ValueError("gzz baseline is launch-time only (use --replan launch)")
        p = req.p
        if st["K"] is None:
            raise RuntimeError("gzz planner needs bind_sim(sim) before the first launch")
        if st["plan"] is None:
            st["plan"], st["info"] = build_plan(req.pos, req.home, p, st["K"], seed, dwell_mode, **aco_kw)
            st["ptr"] = [0] * st["K"]
        k = st["k"] % st["K"]; trips = st["plan"][k]; st["calls"] += 1
        if not trips:
            return []
        trip = trips[st["ptr"][k]]; st["ptr"][k] = (st["ptr"][k] + 1) % len(trips)
        hov = p.e_hover(req.dwell_est)
        d_home = np.linalg.norm(req.pos - req.home, axis=1)
        alone = p.e_fly(2.0 * d_home) + hov
        excl = np.zeros(len(req.pos), bool) if req.excluded is None else np.asarray(req.excluded, bool)
        route, cur, E = [], req.home, req.E_usable
        for i, j in enumerate(trip):
            if excl[j] or alone[j] > req.E_usable:
                st["skipped"] += 1; continue
            e_leg = p.e_fly(float(np.linalg.norm(req.pos[j] - cur))) + hov[j]
            if e_leg + p.e_fly(d_home[j]) <= E + 1e-6:
                route.append(j); E -= e_leg; cur = req.pos[j]
            else:
                st["trunc_sorties"] += 1; st["trunc_nodes"] += len(trip) - i; break
        return route

    planner.bind_sim = bind_sim
    planner.state = st
    return planner


# ---------------------------------------------------------------------------- secondary
def gzz_k_choice(pos, home, p, K_max=60, seed=0):
    """The paper's own fleet-size rule (Alg. 2): the smallest K whose CP clusters each fit ONE
    endurance-limited round trip, under OUR shared inventory (budget (1-rho)Emax/K per UAV).
    Hamiltonian tour per cluster = NN + 2-opt (paper: 'optimal Hamiltonian trajectory'),
    full-buffer dwell. Returns (K or None, best slack ratio seen)."""
    from rr_planner import tour_order
    dwell = p.B_bits / p.R; best = np.inf
    for K in range(1, K_max + 1):
        lab = cukk_partition(pos, home, K, np.random.default_rng(seed * 1_000_003 + 7919 * K + 17))
        E_u = (1 - p.rho) * p.Emax / K; worst = 0.0
        for k in range(K):
            idx = np.where(lab == k)[0]
            if len(idx) == 0: continue
            order = idx[tour_order(pos[idx], home)]
            P = np.vstack([home, pos[order], home])
            e = p.e_fly(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) + p.e_hover(dwell) * len(idx)
            worst = max(worst, e / E_u)
        best = min(best, worst)
        if worst <= 1.0:
            return K, worst
    return None, best
