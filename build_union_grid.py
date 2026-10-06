"""build_union_grid.py -- the K values Liu must be run at: for each of the 216 held-out deployments
(paper/ring/core x M 50/100/200 x Emax 1.5/3 MJ x seeds 1-12), the union of every K at which ANY compared
planner (SA, C4, cluster patrol, Gao et al., Asghar et al.) was run. Writes liu_jobs.csv (layout,M,Emax,seed,K),
excluding (deployment, K) pairs already present in the given Liu result files.
usage: python build_union_grid.py <planner CSVs ...> --have m50_liu.csv liu_edge_k1.csv"""
import sys, csv
from collections import defaultdict, Counter
args = sys.argv[1:]; i = args.index("--have"); planners, have = args[:i], args[i + 1:]
U = defaultdict(set)
for f in planners:
    for r in csv.DictReader(open(f)):
        M, E, s = int(float(r["M"])), float(r["Emax"]), int(float(r["seed"]))
        if M > 200 or E not in (1.5e6, 3e6) or s > 12 or float(r.get("Th", 43200)) != 43200: continue
        if float(r.get("L", 12600)) != 12600 or r.get("coord", "exclude") != "exclude": continue
        U[(r["layout"], M, E, s)].add(int(float(r["K"])))
done = set()
for f in have:
    for r in csv.DictReader(open(f)):
        done.add((r["layout"], int(float(r["M"])), float(r["Emax"]), int(float(r["seed"])), int(float(r["K"]))))
jobs = [(*d, K) for d in sorted(U) for K in sorted(U[d]) if (*d, K) not in done]
with open("liu_jobs.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["layout", "M", "Emax", "seed", "K"]); w.writerows(jobs)
c = Counter((j[0], j[1], j[2]) for j in jobs)
print(f"deployments {len(U)}; union (deployment,K) pairs {sum(len(v) for v in U.values())}; already run {len(done)}; to run {len(jobs)}")
for cell in sorted(c): print(f"   {cell[0]:6s} M={cell[1]:3d} {cell[2]/1e6:g} MJ: {c[cell]} runs")
