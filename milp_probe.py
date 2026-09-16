"""milp_probe.py -- (1) validate CP-SAT against Held-Karp on small fields, (2) time it at M=40.

usage: python milp_probe.py check  [M=13] [seeds=1-3] [Emax=3e5] [K=2] [n_dec=20]
       python milp_probe.py time   [M=40] [seeds=1]   [Emax=6e5] [K=2] [n_dec=5] [tlim=120] [workers=8]

Decisions are captured from the REAL simulator running the SA planner (so ages, exclusions and
dwell estimates are realistic), frozen, then solved offline by every solver on identical input.
"""
import sys, copy, time
import numpy as np
from dyn_env import DynParams, DynSim
from sa_sortie import build_sa_planner
from milp_sortie import solve_sortie, route_energy, candidates


def capture(M, seed, Emax, K, n_dec, Th=6 * 3600, burn=0.5 * 3600, stride=1, post_burn=False):
    """Freeze launch decisions from a real SA-driven run. post_burn=True keeps only decisions
    taken after the burn-in (the first launches see near-uniform ages and are atypical);
    stride=s keeps every s-th eligible decision so the sample spreads over the run."""
    reqs = []; seen = {"n": 0}; holder = {}
    sa = build_sa_planner(iters=1200, seed_base=seed)
    def pl(req):
        r = sa(req)
        ok = req.start is None and len(reqs) < n_dec and req.age.max() > 0
        if ok and post_burn and holder["sim"]._clock < burn:
            ok = False
        if ok:
            seen["n"] += 1
            ok = (seen["n"] - 1) % stride == 0
        if ok:
            q = copy.copy(req)
            for f in ("age", "weight_est", "dwell_est", "excluded", "p_live"):
                if getattr(q, f) is not None: setattr(q, f, np.array(getattr(q, f), copy=True))
            q.pos = np.array(req.pos, copy=True)
            reqs.append((q, list(r)))
        return r
    p = DynParams(M=M, K=K, Emax=Emax, T_horizon=Th, T_burnin=burn)
    sim = DynSim(p, pl, seed=seed); holder["sim"] = sim
    sim.run()
    return reqs


def value_of(req):
    v = req.weight_est * req.age
    if req.p_live is not None:
        v = v * (1 + (req.p.event_gain - 1) * req.p_live)
    return v


def seeds_of(s):
    a, b = (s.split("-") + [s])[:2]; return range(int(a), int(b) + 1)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        from certify_sortie import exact_sortie
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 13
        seeds = seeds_of(sys.argv[3] if len(sys.argv) > 3 else "1-3")
        Emax = float(sys.argv[4]) if len(sys.argv) > 4 else 3e5
        K = int(sys.argv[5]) if len(sys.argv) > 5 else 2
        nd = int(sys.argv[6]) if len(sys.argv) > 6 else 20
        rows = []
        for s in seeds:
            for req, sa_route in capture(M, s, Emax, K, nd):
                v = value_of(req)
                hk, hk_set = exact_sortie(req, v)
                cp = solve_sortie(req, v, time_limit=60, workers=8)
                sa_v = float(sum(v[j] for j in sa_route))
                e_cp = route_energy(req, cp["route"])
                rows.append((hk, cp["value"], sa_v, cp["status"], e_cp <= req.E_usable + 1e-6, cp["secs"]))
        r = np.array([(a, b, c, f) for a, b, c, _, _, f in rows])
        rel = (r[:, 1] - r[:, 0]) / np.maximum(r[:, 0], 1e-9)
        print(f"n={len(rows)} decisions  M={M}")
        print(f"  CP-SAT vs Held-Karp: max |rel diff| {np.abs(rel).max():.2e}   "
              f"CP-SAT < HK by >1e-6 in {(rel < -1e-6).sum()}   CP-SAT > HK by >1e-6 in {(rel > 1e-6).sum()}")
        print(f"  CP-SAT status: {sorted(set(x[3] for x in rows))}   routes energy-feasible: {all(x[4] for x in rows)}")
        print(f"  SA/HK mean {np.mean(r[:,2]/r[:,0]):.4f}   SA optimal in {np.mean(r[:,2] >= r[:,0]*(1-1e-6)):.2f}")
        print(f"  CP-SAT secs: median {np.median(r[:,3]):.3f}  max {r[:,3].max():.3f}")
    elif mode == "time":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 40
        seeds = seeds_of(sys.argv[3] if len(sys.argv) > 3 else "1")
        Emax = float(sys.argv[4]) if len(sys.argv) > 4 else 6e5
        K = int(sys.argv[5]) if len(sys.argv) > 5 else 2
        nd = int(sys.argv[6]) if len(sys.argv) > 6 else 5
        tl = float(sys.argv[7]) if len(sys.argv) > 7 else 120
        wk = int(sys.argv[8]) if len(sys.argv) > 8 else 8
        from sa_sortie import sa_sortie
        for s in seeds:
            for req, sa_route in capture(M, s, Emax, K, nd, Th=4 * 3600, burn=1.5 * 3600, stride=7, post_burn=True):
                v = value_of(req)
                sa_v = float(sum(v[j] for j in sa_route))
                assert route_energy(req, sa_route) <= req.E_usable + 1e-6, "SA route infeasible?"
                # same frozen input, 10x SA iterations, 3 fresh seeds: is the gap search budget?
                sa10 = max(float(sum(v[j] for j in sa_sortie(req.pos, v, req.dwell_est, req.home, req.E_usable,
                            req.p, iters=12000, seed=1000 + r, excluded=req.excluded))) for r in range(3))
                cp = solve_sortie(req, v, time_limit=tl, workers=wk, hint=sa_route)
                feas = route_energy(req, cp["route"]) <= req.E_usable + 1e-6
                gap = (cp["bound"] - cp["value"]) / max(cp["bound"], 1e-9)
                print(f"seed {s}: n_cand={cp['n_cand']:3d} |route| cp {len(cp['route'] or []):2d} sa {len(sa_route):2d}  "
                      f"{cp['status']:8s} {cp['secs']:6.1f}s gap {gap:6.2%} feas {feas}  "
                      f"SA/cp {sa_v/max(cp['value'],1e-9):.4f}  SA10x/cp {sa10/max(cp['value'],1e-9):.4f}  "
                      f"cp/bound {cp['value']/max(cp['bound'],1e-9):.4f}", flush=True)
