"""certify_sortie.py -- certify the sortie planner against an EXACT solver on small fields.

The per-decision problem is: choose a set S of feasible sensors and an order, maximising
sum_{j in S} w_j * a_j subject to route energy (flight + hover, start -> S -> home) <= E.
For M <= 13 we solve it exactly by Held-Karp over subsets with hover folded into node cost.
We run the real simulator on small fields, intercept every SortieRequest, and compare the
value of the route the planner chose with the exact optimum and with the greedy ratio rule.

usage: python certify_sortie.py [M=12] [seeds=1-3] [Emax=3e5] [K=2] [max_sorties=40]
"""
import sys, time, itertools, math
import numpy as np
from dyn_env import DynParams, DynSim, SortieRequest, greedy_ratio_planner
from sa_sortie import build_sa_planner, _chain_energy

def exact_sortie(req: SortieRequest, value):
    p = req.p; pos = req.pos; M = len(pos)
    start = req.home if req.start is None else req.start
    cand = [j for j in range(M) if (req.excluded is None or not req.excluded[j]) and value[j] > 0]
    n = len(cand)
    if n == 0: return 0.0, []
    P = pos[cand]; hov = np.array([p.e_hover(req.dwell_est[j]) for j in cand])
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    d0 = np.linalg.norm(P - start, axis=1); dh = np.linalg.norm(P - req.home, axis=1)
    fly = lambda x: p.e_fly(x)
    INF = float("inf"); full = 1 << n
    dp = np.full((full, n), INF)
    for j in range(n): dp[1 << j, j] = fly(d0[j]) + hov[j]
    for S in range(1, full):
        for j in range(n):
            if not (S >> j) & 1 or dp[S, j] == INF: continue
            base = dp[S, j]
            rest = (~S) & (full - 1)
            k = rest
            while k:
                b = k & -k; i = b.bit_length() - 1; k ^= b
                c = base + fly(d[j, i]) + hov[i]
                if c < dp[S | b, i]: dp[S | b, i] = c
    val = np.array([value[j] for j in cand]); best = 0.0; bestS = 0
    for S in range(1, full):
        E_min = min(dp[S, j] + fly(dh[j]) for j in range(n) if (S >> j) & 1)
        if E_min <= req.E_usable + 1e-6:
            v = sum(val[j] for j in range(n) if (S >> j) & 1)
            if v > best: best, bestS = v, S
    return best, [cand[j] for j in range(n) if (bestS >> j) & 1]

class Cert(DynSim):
    def __init__(self, *a, max_sorties=40, **k):
        super().__init__(*a, **k); self.rows = []; self.max_sorties = max_sorties
    def _plan(self, k, t, start, E_rem, elapsed_frac, iters=None, own_done=()):
        route = super()._plan(k, t, start, E_rem, elapsed_frac, iters, own_done)
        if len(self.rows) < self.max_sorties and start is None:
            req = self._last_req
            value = req.weight_est * req.age
            if req.p_live is not None: value = value * (1 + (self.p.event_gain - 1) * req.p_live)
            v_sa = float(sum(value[j] for j in route))
            g = greedy_ratio_planner(req); v_g = float(sum(value[j] for j in g))
            t0 = time.time(); v_ex, _ = exact_sortie(req, value); dt = time.time() - t0
            self.rows.append((v_sa, v_g, v_ex, dt))
        return route

# capture the request: wrap DynSim._plan's SortieRequest construction by patching the planner
def make_capturing(planner, sim_holder):
    def pl(req):
        sim_holder["sim"]._last_req = req
        return planner(req)
    pl.with_iters = getattr(planner, "with_iters", None)
    return pl

if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seeds = sys.argv[2] if len(sys.argv) > 2 else "1-3"
    Emax = float(sys.argv[3]) if len(sys.argv) > 3 else 3e5
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    maxs = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    a, b = seeds.split("-"); seeds = range(int(a), int(b) + 1)
    allrows = []
    for s in seeds:
        p = DynParams(M=M, K=K, Emax=Emax, T_horizon=6 * 3600, T_burnin=0.5 * 3600)
        holder = {}
        planner = make_capturing(build_sa_planner(iters=1200, seed_base=s), holder)
        sim = Cert(p, planner, seed=s, max_sorties=maxs); holder["sim"] = sim
        sim.run(); allrows += sim.rows
        r = np.array([x for x in sim.rows if x[2] > 0])
        print(f"seed {s}: {len(r)} sorties  SA/exact={np.mean(r[:,0]/np.maximum(r[:,2],1e-9)):.4f}  "
              f"greedy/exact={np.mean(r[:,1]/np.maximum(r[:,2],1e-9)):.4f}  SA=exact in {np.mean(r[:,0]>=r[:,2]-1e-6):.2f}  "
              f"exact solve {np.mean(r[:,3]):.2f}s", flush=True)
    r = np.array([x for x in allrows if x[2] > 0])
    print(f"\nALL: n={len(r)} sorties, M={M}, K={K}, Emax={Emax:.0e}")
    print(f"  SA     : mean ratio to exact {np.mean(r[:,0]/r[:,2]):.4f}, worst {np.min(r[:,0]/r[:,2]):.4f}, optimal in {np.mean(r[:,0]>=r[:,2]-1e-6)*100:.0f}% of sorties")
    print(f"  greedy : mean ratio to exact {np.mean(r[:,1]/r[:,2]):.4f}, worst {np.min(r[:,1]/r[:,2]):.4f}, optimal in {np.mean(r[:,1]>=r[:,2]-1e-6)*100:.0f}% of sorties")