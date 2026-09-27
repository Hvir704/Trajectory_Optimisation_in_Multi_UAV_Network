"""run_pilot.py -- sweep several planners on the PILOT seeds with automatic anti-censoring extension.
usage: python run_pilot.py pilot.csv --planners sa cluster_patrol terr_due --layouts paper ring core \
         --M 50 100 200 --Emax 1.5e6 3e6 --seeds 13-15 --procs 22
Same protocol as run_grid (12 h, burn-in 3 h, launch-time, coordinated). If a planner's argmin sits at
the top of its range for an instance, K is extended by 3 (up to 8 times). Resumable; one writer per file."""
import os, sys, csv, argparse, itertools, time
from run_grid import one, k_range, FIELDS, parse_seeds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--planners", nargs="+", required=True)
    ap.add_argument("--layouts", nargs="+", default=["paper", "ring", "core"])
    ap.add_argument("--M", nargs="+", type=int, default=[50, 100, 200]); ap.add_argument("--Emax", nargs="+", type=float, default=[1.5e6, 3e6])
    ap.add_argument("--seeds", default="13-15"); ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args()
    if any(s <= 12 for s in parse_seeds(a.seeds)):
        sys.exit("pilot seeds must be > 12: seeds 1-12 are the evaluation set")
    done = {}
    if os.path.exists(a.out) and os.path.getsize(a.out):
        for r in csv.DictReader(open(a.out)):
            done[(r["planner"], r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]), int(r["K"]))] = float(r["J"])
    new = not done
    f = open(a.out, "a", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS)
    if new: w.writeheader(); f.flush()
    def run(keys):
        todo = [k for k in keys if k not in done]
        args = [(lay, M, E, K, s, "exclude", "launch", 0, 0.0, "prior", 0.0, pl, 12600.0, 43200.0, 1200)
                for (pl, lay, M, E, s, K) in todo]
        if not args: return
        it = None
        if a.procs > 1:
            from multiprocessing import Pool
            pool = Pool(a.procs); it = pool.imap_unordered(one, args)
        else:
            it = map(one, args)
        for n, row in enumerate(it, 1):
            w.writerow(row); f.flush()
            done[(row["planner"], row["layout"], int(row["M"]), float(row["Emax"]), int(row["seed"]), int(row["K"]))] = float(row["J"])
            if n % 10 == 0 or n == len(args):
                print(f"[{n}/{len(args)}] {row['planner']} {row['layout']} M={row['M']} K={row['K']} s={row['seed']} J={float(row['J']):.3e}", flush=True)
        if a.procs > 1: pool.close(); pool.join()
    insts = list(itertools.product(a.layouts, a.M, a.Emax, parse_seeds(a.seeds)))
    run([(pl, lay, M, E, s, K) for pl in a.planners for lay, M, E, s in insts for K in k_range(lay, M, E, s)])
    for _ in range(8):
        ext = []
        for pl in a.planners:
            for lay, M, E, s in insts:
                Ks = sorted(K for (p_, l, m, e, ss, K) in done if (p_, l, m, e, ss) == (pl, lay, M, E, s))
                if Ks and min(Ks, key=lambda K: done[(pl, lay, M, E, s, K)]) == Ks[-1]:
                    ext += [(pl, lay, M, E, s, K) for K in range(Ks[-1] + 1, Ks[-1] + 4)]
        if not ext: break
        print(f"extending {len(set(k[:5] for k in ext))} censored (planner, instance) pair(s)", flush=True)
        run(ext)
    f.close()

if __name__ == "__main__":
    main()
