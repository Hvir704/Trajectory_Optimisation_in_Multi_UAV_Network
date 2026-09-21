"""make_estimators.py -- the estimator-comparison numbers of Sec. VII-A, on one merged file.
usage: python make_estimators.py grid_final.csv
Per-instance argmin vs three per-cell summaries (argmin of mean J, of median J, of mean relative
regret J(K)/J(K*_inst)), each compared with every instance's own rule. 'Disagree' counts cells
where the three summaries and the rounded mean per-instance argmin are not all equal -- a tally
that may differ from whatever definition produced the draft's earlier '8 of 24'."""
import sys, csv, math, numpy as np
from collections import defaultdict
cells = defaultdict(lambda: defaultdict(dict))
for r in csv.DictReader(open(sys.argv[1])):
    cells[(r["layout"], int(r["M"]), float(r["Emax"]), float(r["L"]))][int(r["seed"])][int(r["K"])] = r
def rule(byK):
    r0 = byK[min(byK)]; kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"])
    law = min(kr, kc); return int(law) if law == kr else int(round(law))
st = {m: [[], []] for m in ("per-instance", "mean J", "median J", "mean rel. regret")}; dis = 0
for key, inst in cells.items():
    Kc = sorted(set.intersection(*[set(b) for b in inst.values()]))
    best = {s: min(float(x["J"]) for x in b.values()) for s, b in inst.items()}
    summ = {"mean J": min(Kc, key=lambda K: np.mean([float(b[K]["J"]) for b in inst.values()])),
            "median J": min(Kc, key=lambda K: np.median([float(b[K]["J"]) for b in inst.values()])),
            "mean rel. regret": min(Kc, key=lambda K: np.mean([float(inst[s][K]["J"]) / best[s] for s in inst]))}
    ams = [min(b, key=lambda K: float(b[K]["J"])) for b in inst.values()]
    dis += len(set(summ.values()) | {int(round(np.mean(ams)))}) > 1
    for s, b in inst.items():
        am = min(b, key=lambda K: float(b[K]["J"])); l = rule(b)
        st["per-instance"][0].append(am == l); st["per-instance"][1].append(abs(am - l) <= 1)
        for m, v in summ.items(): st[m][0].append(v == l); st[m][1].append(abs(v - l) <= 1)
print(f"{len(cells)} cells, {len(st['per-instance'][0])} instances")
for m, (e, w) in st.items(): print(f"  {m:17s} exact {np.mean(e):.3f}  within-one {np.mean(w):.3f}")
print(f"  cells where the summaries disagree: {dis}")
