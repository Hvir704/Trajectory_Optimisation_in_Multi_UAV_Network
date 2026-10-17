"""horizon_v3.py -- the v3 rule as a function of the mission length (surrogate only; no simulation).

For each of the 272 instances: v3 prediction at the run horizon (12 h incl. 3 h burn-in), 24 h, 7 days, and the
long-horizon limit (no stranding) = the plain rule. Counts changed predictions and instances that strand.
usage: python horizon_v3.py results_v2/report/assembled
"""
import sys, os, csv, pickle, collections, numpy as np
import theory_v3
from theory_v3 import field, old_rule, surrogate_rule
from exante_rc import exante


def kref(byK):
    """Reference fleet size of an instance: the first K of its ORIGINAL sweep (column K_ref, written by reproduce_v2.py
    v2.2.1). Fills added for other planners (union grid) must not move the ex-ante or pilot fleet size. Older assembled
    files have no K_ref column; the smallest swept K is then the original first K."""
    for v in byK.values():
        x = v.get("K_ref") if isinstance(v, dict) else None
        if x not in (None, ""):
            k = int(float(x)); return k if k in byK else min(byK)
        break
    return min(byK)
A = sys.argv[1]
CACHE = "tour_cache_v3.pkl"
if os.path.exists(CACHE): theory_v3.TOUR_CACHE.update(pickle.load(open(CACHE, "rb")))
D = collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(A, "grid_final.csv"))):
    if r["coord"] == "exclude" and r["replan"] == "launch" and float(r["Th"]) == 43200 and r.get("planner", "sa") == "sa":
        D[(r["layout"], int(float(r["M"])), float(r["Emax"]), float(r["L"]), int(r["seed"]))][int(float(r["K"]))] = r
H = [("12 h (as run)", None), ("24 h", 24 * 3600.0), ("7 days", 7 * 86400.0)]
res = collections.defaultdict(lambda: collections.defaultdict(list))
for k, byK in sorted(D.items()):
    lay, M, E, L, s = k; p, F = field(lay, M, E, L, s); r = np.linalg.norm(F.pos - p.home, axis=1)
    rc, Pb, _ = exante(p, r, np.full(M, p.lam_bits), F.pos, kref(byK))
    old = old_rule(p, r, rc, Pb); base = None
    for lab, h in H:
        K, j = surrogate_rule(p, F, rc, Pb, key=(lay, M, L, s), horizon=h)
        if base is None: base = K
        res[(lay, lab)]["changed"].append(K != base); res[(lay, lab)]["strand"].append(j > 0); res[(lay, lab)]["eq_old"].append(K == old)
pickle.dump(theory_v3.TOUR_CACHE, open(CACHE, "wb"))
print(f"{'family':6s} {'horizon':14s} {'n':>4s} {'changed vs 12 h':>16s} {'strand':>7s} {'= plain rule':>13s}")
for lay in ("core", "paper", "ring"):
    for lab, _ in H:
        R = res[(lay, lab)]
        print(f"{lay:6s} {lab:14s} {len(R['changed']):4d} {sum(R['changed']):16d} {sum(R['strand']):7d} {sum(R['eq_old']):13d}")
