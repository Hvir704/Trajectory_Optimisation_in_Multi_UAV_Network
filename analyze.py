"""analyze.py -- per-instance law test, paired tests, regime table.  usage: python analyze.py grid.csv"""
import sys, csv, math
from collections import defaultdict
import numpy as np

rows = [r for r in csv.DictReader(open(sys.argv[1])) if "M" in r and r["M"] not in ("", "M")]
for r in rows:
    for k in ("M","K","seed"): r[k] = int(r[k])
    for k in ("Emax","J","K_reach_i","K_commute_i","rmax_over_rc","regime_bnd","n_never","share_never","dup_frac","trunc_frac","Th"):
        r[k] = float(r[k])

cells = defaultdict(lambda: defaultdict(dict))   # (layout,M,Emax,coord,replan,Th) -> seed -> K -> row
for r in rows: cells[(r["layout"],r["M"],r["Emax"],r["coord"],r.get("replan","launch")+" q"+str(r.get("q","0"))+" "+str(r.get("belief","prior"))+" tau"+str(r.get("tau","0")),r["Th"])][r["seed"]][r["K"]] = r

def sign_test(d):
    d = np.asarray(d); n = len(d); s = int((d > 0).sum())
    p = sum(math.comb(n, k) for k in range(max(s, n-s), n+1)) / 2**n * 2 if n else float("nan")   # two-sided approx
    return s, n, min(p, 1.0)

print(f"{'cell':44} {'inst':>4} {'argmin':>7} {'law':>5} {'agree':>6} {'agree±1':>8} {'J(law) vs J(law+1)':>20} {'J(law) vs J(law-1)':>20} {'r_max/r_c':>9} {'bnd':>5} {'regime':>7}")
for key in sorted(cells):
    inst = cells[key]; argmins = []; laws = []; hits = 0; hits1 = 0; dplus = []; dminus = []; ratios = []; bnds = []
    for seed, byK in inst.items():
        Ks = sorted(byK); J = {K: byK[K]["J"] for K in Ks}
        am = min(Ks, key=lambda K: J[K]); argmins.append(am)
        r0 = byK[Ks[0]]
        law = min(math.floor(r0["K_reach_i"]), r0["K_commute_i"])
        law_int = int(round(law)) if law != math.floor(r0["K_reach_i"]) else int(law)
        laws.append(law_int); ratios.append(r0["rmax_over_rc"]); bnds.append(r0["regime_bnd"])
        hits += (am == law_int); hits1 += (abs(am - law_int) <= 1)
        if law_int in J and law_int+1 in J: dplus.append(J[law_int+1] - J[law_int])
        if law_int in J and law_int-1 in J: dminus.append(J[law_int-1] - J[law_int])
    n = len(inst)
    sp = sign_test(dplus) if dplus else (0,0,float("nan")); sm = sign_test(dminus) if dminus else (0,0,float("nan"))
    regime = "reach" if np.mean(ratios) > np.mean(bnds) else "commute"
    lab = f"{key[0]} M={key[1]} E={key[2]:.1e} {key[3]} {key[4]} {key[5]/3600:.0f}h"
    print(f"{lab:44} {n:4d} {np.mean(argmins):7.2f} {np.mean(laws):5.2f} {hits/n:6.2f} {hits1/n:8.2f} "
          f"{sp[0]:>3}/{sp[1]:<3} p={sp[2]:.3f}   {sm[0]:>3}/{sm[1]:<3} p={sm[2]:.3f}   {np.mean(ratios):9.2f} {np.mean(bnds):5.2f} {regime:>7}")
print("\ncolumns: argmin = mean per-instance argmin J; law = mean per-instance min(floor K_reach, K_commute);")
print("agree = P(argmin == law); J(law) vs J(law+1) = #instances where J(law+1) > J(law) / n, sign-test p (same for law-1).")