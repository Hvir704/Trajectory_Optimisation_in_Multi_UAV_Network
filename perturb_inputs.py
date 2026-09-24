"""perturb_inputs.py -- how wrong can the 'known' inputs be before the rule's advice degrades?
The rule is recomputed with one physical input scaled by (1+e); it is then scored against the TRUE
J(K) curves already recorded for each deployment (misspecifying inputs changes only the prediction,
not the system). Predictions outside a swept range are clamped to its edge (reported).
usage: python perturb_inputs.py grid_final.csv"""
import sys, csv, math
from collections import defaultdict
import numpy as np
from dyn_env import DynParams
p = DynParams()
inst = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    if float(r["L"]) == 12600: inst[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]))][int(r["K"])] = r
base = dict(U=1.0, Pf=1.0, Ph=1.0, Ru=1.0, rmax=1.0)
def score(scale):
    ex = w1 = n = clamp = 0; regs = []
    for k, byK in inst.items():
        Ks = sorted(byK); J = {K: float(byK[K]["J"]) for K in Ks}; r0 = byK[Ks[0]]; am = min(J, key=J.get)
        U = (1 - p.rho) * k[2] * scale["U"]; Pf = p.Pf * scale["Pf"]; Ph = p.Ph * scale["Ph"]; Ru = p.R * scale["Ru"]
        rmax = float(r0["r_max"]) * scale["rmax"]; rc = float(r0["r_c"]); Pb = float(r0["P_bar"])
        Kr = p.v * U / (2 * Pf * rmax + p.v * Ph * p.B_bits / Ru); Kc = p.v * U / (2 * rc * (Pf + math.sqrt(Pf * Pb)))
        law = min(math.floor(Kr), Kc); l = int(law) if law == math.floor(Kr) else int(round(law))
        lc = min(max(l, Ks[0]), Ks[-1]); clamp += lc != l
        n += 1; ex += lc == am; w1 += abs(lc - am) <= 1; regs.append(J[lc] / J[am] - 1)
    r = np.array(regs)
    return ex / n, w1 / n, np.median(r), np.quantile(r, .9), clamp
e0 = score(base)
print(f"{'input':6s} {'error':>6s} | exact  within1  regret med  p90   | clamped")
print(f"{'none':6s} {'':>6s} | {e0[0]:.3f}  {e0[1]:.3f}   {e0[2]:6.1%} {e0[3]:6.1%} | {e0[4]}")
for name in ("U", "Pf", "Ph", "Ru", "rmax"):
    for e in (-0.2, -0.1, 0.1, 0.2):
        s = dict(base); s[name] = 1 + e; r = score(s)
        print(f"{name:6s} {e:+6.0%} | {r[0]:.3f}  {r[1]:.3f}   {r[2]:6.1%} {r[3]:6.1%} | {r[4]}")
