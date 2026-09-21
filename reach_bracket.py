"""reach_bracket.py -- the reach cap is a bracket, not an iff.
usage: python reach_bracket.py grid_final.csv
K_reach (full-buffer dwell B/R_u) guarantees service from any state; K_flight = vU/(2 Pf rmax) is
the exact upper limit. Reports the bracket width, how often it contains an integer, and the point
predictor's agreement with each cap substituted."""
import sys, csv, math, numpy as np
from collections import defaultdict
from dyn_env import DynParams
p = DynParams(); inst = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    inst[(r["layout"], int(r["M"]), float(r["Emax"]), float(r["L"]), int(r["seed"]))][int(r["K"])] = r
ratio = []; band = 0; res = defaultdict(lambda: defaultdict(list))
for k, byK in inst.items():
    r0 = byK[min(byK)]; U = (1 - p.rho) * k[2]
    Kr = float(r0["K_reach_i"]); Kc = float(r0["K_commute_i"]); Kf = p.v * U / (2 * p.Pf * float(r0["r_max"]))
    ratio.append(Kf / Kr); band += math.floor(Kf) != math.floor(Kr)
    am = min(byK, key=lambda K: float(byK[K]["J"]))
    for name, cap in (("full-buffer K_reach", Kr), ("flight-only K_flight", Kf)):
        law = min(math.floor(cap), Kc); l = int(law) if law == math.floor(cap) else int(round(law))
        for g in ("all", "primary" if k[3] == 12600 else "Lsweep", k[0] if k[3] == 12600 else None):
            if g: res[name][g].append((am == l, abs(am - l) <= 1, am - l))
print(f"K_flight/K_reach in [{min(ratio):.3f}, {max(ratio):.3f}]; bracket contains an integer in {band} of {len(ratio)}")
for name, d in res.items():
    print(name)
    for g in ("all", "primary", "paper", "ring", "core", "Lsweep"):
        v = np.array(d[g]); print(f"  {g:8s} n={len(v):3d} exact {v[:,0].mean():.3f} within-one {v[:,1].mean():.3f} bias {v[:,2].mean():+.2f}")