"""make_table8.py -- Table VIII (tab:regret) regenerated on the merged, uncensored grid.

usage: python make_table8.py grid_final.csv ../paper/apriori_regret.csv
The criterion verdict (passes) and regime flag come from apriori_regret.csv -- they depend only on
geometry and the smallest-K commute inputs, so extending the K sweeps does not change them. Regret
J(K_rule)/J(K*) - 1 is recomputed on the extended sweeps. 90th percentile uses method='higher',
the convention that reproduces the draft's unchanged rows exactly.
"""
import sys, csv, math, numpy as np
from collections import defaultdict
d = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    if float(r["L"]) == 12600: d[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]))][int(r["K"])] = r
recs = []
for r in csv.DictReader(open(sys.argv[2])):
    k = (r["lay"], int(r["M"]), float(r["E"]), int(r["seed"])); byK = d[k]; Ks = sorted(byK)
    J = {K: float(byK[K]["J"]) for K in Ks}; r0 = byK[Ks[0]]
    kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"]); law = min(kr, kc)
    li = int(law) if law == kr else int(round(law))
    recs.append((k[0], r["passes"] == "True", J[li] / min(J.values()) - 1))
def row(lab, test, R):
    x = np.array([t[2] for t in R])
    print(f"{lab} & {test} & {len(x)} & {100*np.median(x):.1f}\\% & {100*np.quantile(x,.9,method='higher'):.1f}\\% & "
          f"{100*x.max():.1f}\\% & {100*np.mean(x<0.05):.0f}\\%\\\\")
for fam in ("paper", "core"):
    row(fam, "pass", [t for t in recs if t[0] == fam and t[1]])
    row(fam, "flagged", [t for t in recs if t[0] == fam and not t[1]])
row("ring", "(commute-bound)", [t for t in recs if t[0] == "ring"])
print("\\midrule")
row("all reach-bound, test passes", "", [t for t in recs if t[0] != "ring" and t[1]])
row("all reach-bound, flagged", "", [t for t in recs if t[0] != "ring" and not t[1]])
