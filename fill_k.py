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
  4. TAIL (--tail Q C, registered 29 Sep 2026 as Q=3, C=1.5): an interior optimum is not enough, because
     J(K) is non-convex on some fields (a second, lower basin further out). For every planner the grid
     is extended upward until its Q largest K values all have J > C x its current minimum, and downward
     until its Q smallest K values all do, or K = 1. KMAX caps the search.
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


KMAX = 80


def tail_ok(y, Ks, jmin, C):
    """True if every K in Ks has a row for this planner with J > C * jmin (inf counts as above)."""
    for K in Ks:
        r = y[y.K == K]
        if not len(r):
            return False
        J = float(r.J.iloc[0])
        if np.isfinite(J) and J <= C * jmin:
            return False
    return True


def missing_jobs(d, planners, tail=None):
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
                    if tail:
                        Q, C = tail; jmin = float(fin.J.min())
                        Ks = sorted(allK)
                        if not tail_ok(y, Ks[-Q:], jmin, C) and max(allK) < KMAX:
                            need.add(max(allK) + 1)
                        if min(allK) > 1 and not tail_ok(y, Ks[:Q], jmin, C):
                            need.add(min(allK) - 1)
            for K in sorted(need - have):
                jobs.append(job(*key, K, pl))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("inputs", nargs="+")
    ap.add_argument("--planners", nargs="+", required=True)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--allow-eval", action="store_true")
    ap.add_argument("--tail", nargs=2, type=float, default=None, metavar=("Q", "C"),
                    help="registered: --tail 3 1.5")
    a = ap.parse_args()
    frames = [pd.read_csv(f) for f in a.inputs]
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        frames.append(pd.read_csv(a.out))
    d = pd.concat(frames, ignore_index=True)
    d = d[d.planner.isin(a.planners)].drop_duplicates(KEY + ["K", "planner"])
    if not a.allow_eval and (d.seed <= 12).any():
        sys.exit("refusing: input contains seeds <= 12 (evaluation seeds); pass --allow-eval deliberately")
    new = (not os.path.exists(a.out)) or os.path.getsize(a.out) == 0
    for rnd in range(120):
        jobs = missing_jobs(d, a.planners, (int(a.tail[0]), a.tail[1]) if a.tail else None)
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
