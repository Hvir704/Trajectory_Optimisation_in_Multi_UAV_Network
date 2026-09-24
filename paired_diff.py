"""paired_diff.py -- paired bootstrap for "planners agree with each other more than with the rule".
Per instance: d = 1[|K*_patrol - K*_SA| <= 1] - 1[|K*_X - rule_X| <= 1], X = SA or patrol.
Stratified by cell (instances resampled within cells), 10,000 replicates, 95% percentile interval.
usage: python paired_diff.py sa_rerun.csv rr_tour_win.csv"""
import sys, csv, math
from collections import defaultdict
import numpy as np
rng = np.random.default_rng(20260924); B = 10_000
def load(p):
    d = defaultdict(dict)
    for r in csv.DictReader(open(p)):
        if float(r["Th"]) == 43200 and r["coord"] == "exclude" and r["replan"] == "launch":
            d[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]))][int(r["K"])] = r
    return d
def rule(b):
    r0 = b[min(b)]; kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"]); law = min(kr, kc)
    return int(law) if law == kr else int(round(law))
sa, rr = load(sys.argv[1]), load(sys.argv[2])
cells = defaultdict(list)
for k in sorted(set(sa) & set(rr)):
    Ks = sorted(set(sa[k]) & set(rr[k]))
    if len(Ks) < 3: continue
    a = min(Ks, key=lambda K: float(sa[k][K]["J"])); b = min(Ks, key=lambda K: float(rr[k][K]["J"]))
    pp = abs(a - b) <= 1; sr = abs(a - rule(sa[k])) <= 1; rr_ = abs(b - rule(rr[k])) <= 1
    cells[k[:3]].append((pp - sr, pp - rr_))
G = [np.array(v, float) for v in cells.values()]; allv = np.vstack(G)
for j, lab in ((0, "patrol~SA  minus  SA~rule"), (1, "patrol~SA  minus  patrol~rule")):
    reps = [np.vstack([g[rng.integers(0, len(g), len(g))] for g in G])[:, j].mean() for _ in range(B)]
    print(f"{lab:32s} mean diff {allv[:, j].mean():+.3f}  95% [{np.quantile(reps, .025):+.3f}, {np.quantile(reps, .975):+.3f}]  n={len(allv)}")
