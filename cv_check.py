"""cv_check.py -- CV of per-node inter-visit intervals from EXPORT_NODES npz files (§VII-B).

usage: python cv_check.py "nodes_paper_M100_*_K*_s*_exclude_launch.npz"

Drops each node's FIRST interval (left-censored: t_last_visit starts at 0, so it is measured from
t=0 or across the burn-in boundary). A node therefore needs >= 3 visits to contribute a CV.
"""
import sys, glob, re, collections
import numpy as np

pat = sys.argv[1] if len(sys.argv) > 1 else "nodes_*.npz"
files = sorted(glob.glob(pat))
if not files:
    print(f"no files match {pat}"); sys.exit(1)

byK = collections.defaultdict(list)   # K -> list of per-node CVs
kept = collections.Counter(); dropped = collections.Counter()
for f in files:
    K = int(re.search(r"_K(\d+)_", f).group(1))
    d = np.load(f)
    if "vis_int" not in d:
        print(f"{f}: no vis_int field -- exported before the dyn_env patch, re-run"); continue
    cnt = d["vis_int_counts"]
    per_node = np.split(d["vis_int"], np.cumsum(cnt)[:-1])
    for iv in per_node:
        iv = np.asarray(iv, dtype=float)[1:]          # drop the left-censored first interval
        if iv.size < 2: dropped[K] += 1; continue     # need >= 2 real intervals for a CV
        kept[K] += 1
        byK[K].append(float(iv.std(ddof=1) / iv.mean()))

print(f"{'K':>3}{'nodes used':>12}{'nodes dropped':>15}{'mean CV':>10}{'median CV':>12}{'p10':>8}{'p90':>8}")
for K in sorted(byK):
    cv = np.array(byK[K])
    print(f"{K:>3}{kept[K]:>12}{dropped[K]:>15}{cv.mean():>10.3f}{np.median(cv):>12.3f}"
          f"{np.percentile(cv,10):>8.3f}{np.percentile(cv,90):>8.3f}")
print("\nCV = 1 is memoryless (Poisson-like) revisits; CV < 1 is more regular than Poisson,")
print("CV > 1 is burstier. Report the value and say which side of 1 it falls on, and note that")
print(f"{sum(dropped.values())} of {sum(kept.values())+sum(dropped.values())} node-instances had too few visits to contribute.")