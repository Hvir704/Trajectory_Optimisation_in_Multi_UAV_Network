"""analyze_hetero.py -- reserved-sortie allocation (run_hetero.py) vs equal shares, paired per instance.
usage: python analyze_hetero.py sa_rerun.csv hetero.csv
Equal-share rows come from the same-machine grid (sa_rerun.csv). The split-fleet predictor is the reach
cap of the K-1 equally funded UAVs, set by the SECOND-farthest sensor:
    floor(1 + (U - E_0) / (2 Pf r_(2)/v + Ph B/R_u)),   E_0 = 1.02 (2 Pf r_max/v + Ph B/R_u).
It was formed for this test (no fitted constant) -- report it as an observation, not a rule."""
import sys, csv, math
from collections import defaultdict
import numpy as np
from dyn_env import DynParams, SensorField
p = DynParams()
def load(f):
    d = defaultdict(dict)
    for r in csv.DictReader(open(f)):
        if float(r["Th"]) == 43200 and r.get("planner", "sa") == "sa":
            d[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]))][int(r["K"])] = r
    return d
eq, he = load(sys.argv[1]), load(sys.argv[2])
rows = []
for k in sorted(he):
    E = {K: float(eq[k][K]["J"]) for K in eq[k]}; H = {K: float(he[k][K]["J"]) for K in he[k]}
    ae, ah = min(E, key=E.get), min(H, key=H.get)
    F = SensorField(DynParams(M=k[1], Emax=k[2], layout=k[0]), np.random.default_rng(k[3]))
    r = np.sort(np.linalg.norm(F.pos - p.home, axis=1))[::-1]; U = (1 - p.rho) * k[2]
    e0 = 1.02 * (p.e_fly(2 * r[0]) + p.Ph * p.B_bits / p.R)
    k2 = math.floor(1 + (U - e0) / (p.e_fly(2 * r[1]) + p.Ph * p.B_bits / p.R))
    rows.append((k[:3], ae, ah, E[ae], H[ah], ah == max(H), int(he[k][ah]["n_never"]), int(eq[k][ae]["n_never"]), k2))
for cell in sorted({x[0] for x in rows}):
    R = [x for x in rows if x[0] == cell]; a = lambda i: np.array([x[i] for x in R], float)
    ratio = a(4) / a(3) - 1
    print(f"{cell[0]} M={cell[1]} {cell[2]/1e6:g}MJ n={len(R)} | argmin equal {a(1).mean():.2f} reserved {a(2).mean():.2f} "
          f"(censored {int(a(5).sum())}) | stranded at optimum equal {a(7).mean():.2f} reserved {a(6).mean():.2f} | "
          f"J reserved/equal-1 median {np.median(ratio):+.1%} better in {int((ratio<0).sum())}/{len(R)} | "
          f"split-fleet cap: exact {int((a(2)==a(8)).sum())} within-one {int((abs(a(2)-a(8))<=1).sum())}")