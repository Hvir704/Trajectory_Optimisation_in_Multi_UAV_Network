"""paired_fix_check.py -- does the breach-accounting fix move the measured optimum?
Runs the SA planner on the same instances with legacy_breach=True (old behaviour) and False (fixed),
over each instance's usual K range, and reports per-instance argmin and J at the old argmin.
usage: python paired_fix_check.py out.csv --layouts ring paper --M 100 --Emax 1.5e6 --seeds 1-12 --procs 20
"""
import argparse, csv, os, time
from multiprocessing import Pool
from collections import defaultdict
from dyn_env import DynParams, DynSim
from sa_sortie import build_sa_planner
from run_grid import k_range, parse_seeds

def one(a):
    lay, M, E, seed, K, legacy = a
    p = DynParams(M=M, K=K, Emax=E, layout=lay, legacy_breach=legacy, legacy_clock=legacy,
                  legacy_events=legacy, legacy_eta=legacy)   # whole old simulator vs v2
    t = time.time(); m = DynSim(p, build_sa_planner(iters=1200, seed_base=seed), seed=seed).run()
    from run_grid import SIM_VERSION
    return dict(layout=lay, M=M, Emax=E, seed=seed, K=K, legacy=int(legacy), sim_version=("v1-legacy" if legacy else SIM_VERSION), J=m["J_timeavg"],
                breach_frac=m["breach_frac"], trunc_frac=m["trunc_frac"], secs=time.time() - t)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--layouts", nargs="+", default=["ring", "paper"]); ap.add_argument("--M", nargs="+", type=int, default=[100])
    ap.add_argument("--Emax", nargs="+", type=float, default=[1.5e6]); ap.add_argument("--seeds", default="1-12")
    ap.add_argument("--procs", type=int, default=os.cpu_count()); a = ap.parse_args()
    jobs = [(l, M, E, s, K, lg) for l in a.layouts for M in a.M for E in a.Emax for s in parse_seeds(a.seeds)
            for K in k_range(l, M, E, s) for lg in (True, False)]
    with Pool(a.procs) as pool, open(a.out, "w", newline="") as f:
        w = None; rows = []
        for r in pool.imap_unordered(one, jobs):
            if w is None: w = csv.DictWriter(f, fieldnames=list(r)); w.writeheader()
            w.writerow(r); f.flush(); rows.append(r)
    by = defaultdict(dict)
    for r in rows: by[(r["layout"], r["M"], r["Emax"], r["seed"], r["legacy"])][r["K"]] = r["J"]
    moved = defaultdict(list)
    for (l, M, E, s, lg), J in by.items():
        if lg: continue
        old = by[(l, M, E, s, 1)]; ko = min(old, key=old.get); kn = min(J, key=J.get)
        moved[(l, M, E)].append((ko, kn, J[ko] / old[ko] - 1))
    for c, v in sorted(moved.items()):
        n = len(v); mv = sum(1 for a_, b_, _ in v if a_ != b_)
        print(f"{c}: argmin moved in {mv}/{n}  mean old {sum(a_ for a_,_,_ in v)/n:.2f} new {sum(b_ for _,b_,_ in v)/n:.2f}  "
              f"median J change at old argmin {sorted(x for *_, x in v)[n//2]:+.1%}")
