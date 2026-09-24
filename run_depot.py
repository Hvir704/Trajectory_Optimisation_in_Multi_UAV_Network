"""run_depot.py -- the rule with the depot away from the field centre.

The simulator measures every distance from DynParams.home (a property fixed at the field centre); the
sensor layouts are placed about the field centre independently of the depot. Overriding that one
property therefore moves the depot while leaving the layouts, the K-range protocol (0.5-1.5 K_reach
from the depot) and every per-instance quantity (K_reach_i, K_commute_i, r_c, P_bar) consistent.
The override is set in the environment before the pool starts; spawn workers re-import this module
and apply it from there, so every process agrees on the depot.

Protocol identical to run_grid.py, plus the grid's anti-censoring rule applied automatically: if an
instance's argmin sits at the top of its swept range, K is extended by 3 and re-checked (up to 6 times).

usage: python run_depot.py out.csv --layouts paper ring --M 100 --Emax 1.5e6 3e6 --seeds 1-12 \
                           --depot 0.25 0.25 --procs 12
Output: run_grid FIELDS + depot_x, depot_y (fractions of L). Resumable; one writer per file.
"""
import os, sys, csv, argparse, math, itertools
import numpy as np

import dyn_env


def apply_depot(frac):
    fx, fy = (float(v) for v in frac.split(","))
    dyn_env.DynParams.home = property(lambda self: np.array([fx * self.L, fy * self.L]))


if os.environ.get("DEPOT_FRAC"):          # spawned workers re-import this module and inherit the env
    apply_depot(os.environ["DEPOT_FRAC"])
from run_grid import one, k_range, FIELDS, parse_seeds

OUT_FIELDS = FIELDS + ["depot_x", "depot_y"]


def job(args):
    row = one(args)
    fx, fy = (float(v) for v in os.environ["DEPOT_FRAC"].split(","))
    row["depot_x"], row["depot_y"] = fx, fy
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--layouts", nargs="+", default=["paper"])
    ap.add_argument("--M", nargs="+", type=int, default=[100]); ap.add_argument("--Emax", nargs="+", type=float, default=[1.5e6])
    ap.add_argument("--seeds", default="1-12"); ap.add_argument("--depot", nargs=2, type=float, required=True)
    ap.add_argument("--procs", type=int, default=1); ap.add_argument("--budget", type=float, default=0,
                    help="stop launching new runs after this many seconds (0 = no limit)")
    a = ap.parse_args()
    os.environ["DEPOT_FRAC"] = f"{a.depot[0]},{a.depot[1]}"   # set BEFORE any Pool is created
    apply_depot(os.environ["DEPOT_FRAC"])

    import time
    t0 = time.time()
    done = {}
    if os.path.exists(a.out) and os.path.getsize(a.out):
        for r in csv.DictReader(open(a.out)):
            done[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]), int(r["K"]))] = float(r["J"])
    new = not done
    f = open(a.out, "a", newline=""); w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
    if new: w.writeheader(); f.flush()

    def run_many(keys):
        todo = [k for k in keys if k not in done]
        args = [(lay, M, E, K, s, "exclude", "launch", 0.0, 0.0, "prior", 0.0, "sa", 12600.0, 43200.0, 1200)
                for (lay, M, E, s, K) in todo]
        if not args: return
        if a.procs > 1:
            from multiprocessing import Pool
            with Pool(a.procs) as pool:
                for row in pool.imap_unordered(job, args):
                    w.writerow(row); f.flush()
                    done[(row["layout"], int(row["M"]), float(row["Emax"]), int(row["seed"]), int(row["K"]))] = float(row["J"])
                    print(f"{row['layout']} M={row['M']} E={float(row['Emax']):.1e} K={row['K']} s={row['seed']} J={float(row['J']):.3e}", flush=True)
        else:
            for x in args:
                if a.budget and time.time() - t0 > a.budget: f.close(); sys.exit(0)
                row = job(x); w.writerow(row); f.flush()
                done[(row["layout"], int(row["M"]), float(row["Emax"]), int(row["seed"]), int(row["K"]))] = float(row["J"])
                print(f"{row['layout']} M={row['M']} E={float(row['Emax']):.1e} K={row['K']} s={row['seed']} J={float(row['J']):.3e} ({row['secs']:.0f}s)", flush=True)

    insts = list(itertools.product(a.layouts, a.M, a.Emax, parse_seeds(a.seeds)))
    run_many([(lay, M, E, s, K) for lay, M, E, s in insts for K in k_range(lay, M, E, s)])
    for _ in range(6):                                   # anti-censoring extension
        ext = []
        for lay, M, E, s in insts:
            Ks = sorted(K for (l, m, e, ss, K) in done if (l, m, e, ss) == (lay, M, E, s))
            if not Ks: continue
            am = min(Ks, key=lambda K: done[(lay, M, E, s, K)])
            if am == Ks[-1]: ext += [(lay, M, E, s, K) for K in range(Ks[-1] + 1, Ks[-1] + 4)]
        if not ext: break
        print(f"extending {len(set(k[:4] for k in ext))} censored instance(s)", flush=True)
        run_many(ext)
    f.close()


if __name__ == "__main__":
    main()
