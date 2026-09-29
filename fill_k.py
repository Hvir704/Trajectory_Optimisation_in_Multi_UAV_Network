"""fill_k.py -- put every planner on the SAME, uncensored K grid before own-optimum comparisons.

usage: python fill_k.py out.csv IN1.csv [IN2.csv ...] --planners cluster_patrol cyc [--procs 22]
       [--dry]

Per deployment (layout, M, Emax, seed) it repeats until nothing changes:
  1. UNION: every planner is run at every K that any listed planner has a row for.
     (Own optimum depends on the swept range; a planner swept on a narrower range can miss a lower
     minimum -- e.g. core M=200 3 MJ s13, where cluster's range was extended to K=24 and cyc's was not.)
  2. LOWER EDGE: if a planner's optimum is at the smallest K on the grid and that K > 1, K-1 is added.
  3. UPPER EDGE: if a planner's optimum is at the largest K on the grid, K+1 is added unless J there is
     already known to be infinite / NaN for that planner.
New rows go to out.csv (run_grid schema, run_grid.one with its defaults: coord exclude, launch-time,
no divert, q 0, prior belief, default tau, L and 12 h horizon, 1200 SA iters). Inputs are never modified.
Refuses seeds <= 12 unless --allow-eval is given (the evaluation is run once, deliberately).
"""
import argparse, csv, os, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from dyn_env import DynParams
from run_grid import one, FIELDS

KEY = ["layout", "M", "Emax", "seed"]


def job(layout, M, Emax, seed, K, planner):
    return (layout, int(M), float(Emax), int(K), int(seed), "exclude", "launch", 0, 0.0, "prior", 0.0,
            planner, DynParams.L, 12 * 3600.0, 1200)


def missing_jobs(d, planners):
    jobs = []
    for key, x in d.groupby(KEY):
        allK = set(int(k) for k in x.K.unique())
        for pl in planners:
            y = x[x.planner == pl]
            have = set(int(k) for k in y.K)
            need = allK - have
            if len(y):
                fin = y[np.isfinite(y.J)]
                if len(fin):
                    kopt = int(fin.loc[fin.J.idxmin(), "K"])
                    if kopt == min(allK) and kopt > 1:
                        need.add(kopt - 1)
                    if kopt == max(allK):
                        need.add(kopt + 1)
            for K in sorted(need - have):
                jobs.append(job(*key, K, pl))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("inputs", nargs="+")
    ap.add_argument("--planners", nargs="+", required=True)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--allow-eval", action="store_true")
    a = ap.parse_args()
    frames = [pd.read_csv(f) for f in a.inputs]
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        frames.append(pd.read_csv(a.out))
    d = pd.concat(frames, ignore_index=True)
    d = d[d.planner.isin(a.planners)].drop_duplicates(KEY + ["K", "planner"])
    if not a.allow_eval and (d.seed <= 12).any():
        sys.exit("refusing: input contains seeds <= 12 (evaluation seeds); pass --allow-eval deliberately")
    new = (not os.path.exists(a.out)) or os.path.getsize(a.out) == 0
    for rnd in range(40):
        jobs = missing_jobs(d, a.planners)
        print(f"round {rnd}: {len(jobs)} runs", flush=True)
        for j in jobs[:30]:
            print("   ", j[11], j[0], f"M={j[1]} E={j[2]:.1e} s={j[4]} K={j[3]}")
        if not jobs or a.dry:
            break
        rows = []
        with open(a.out, "a", newline="") as f, Pool(min(a.procs, len(jobs))) as pool:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader(); new = False
            for r in pool.imap_unordered(one, jobs):
                w.writerow(r); f.flush(); rows.append(r)
        d = pd.concat([d, pd.DataFrame(rows)], ignore_index=True)
    print("done; combine with the inputs, e.g. own_opt_compare.py on a concatenation of all files")
