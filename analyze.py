"""analyze.py -- per-instance law test, paired tests, regime table.  usage: python analyze.py grid.csv

CHANGED vs rev1 -- one defect that silently corrupted a cell, plus two robustness fixes:

  1. The cell key omitted L. Once grid.csv held more than one field size, rows differing only in L
     collided in cells[key][seed][K] and overwrote each other by file order. Worse, the surviving
     dict mixed field sizes: an L=20000 run contributes K=1 rows that an L=12600 run (K=2..8) never
     overwrites, so Ks[0] became that K=1 row and `r0` read K_reach_i from the WRONG field size.
     Measured effect on paper M=100 1.5MJ: 8 of 12 seeds took K_reach from L=20000 (floor 3) and 4
     from L=12600 (floor 4), giving a reported law of 3.33 instead of 4.00 and an agreement of 0.33
     instead of 0.75. L is now part of the key, which also splits the L sweep into its own rows --
     that is Table VI.
  2. sign_test counted ties (d == 0) as negative. Ties are now dropped, the standard convention;
     n becomes the number of non-tied pairs.
  3. q/tau are normalised to float in the key, so '0' and '0.0' from a schema rewrite cannot split
     one cell into two.

NOT changed: the law itself. `law = min(floor(K_reach_i), K_commute_i)`, and when the commute term
binds it is rounded rather than floored or ceiled. round(2.96) = 3 = ceil here, but round(3.4) = 3
while ceil(3.4) = 4 -- if the paper's definition is ceil, this needs deciding, so it is left alone
and flagged rather than silently altered.
"""
import sys, csv, math
from collections import defaultdict
import numpy as np

rows = [r for r in csv.DictReader(open(sys.argv[1])) if "M" in r and r["M"] not in ("", "M")]
for r in rows:
    for k in ("M","K","seed"): r[k] = int(float(r[k]))
    for k in ("Emax","J","K_reach_i","K_commute_i","rmax_over_rc","regime_bnd","n_never",
              "share_never","dup_frac","trunc_frac","Th"):
        r[k] = float(r[k])
    r["L"] = float(r.get("L", 12600.0) or 12600.0)

def variant(r):
    q   = float(r.get("q", 0) or 0)
    tau = float(r.get("tau", 0) or 0)
    return f"{r.get('replan','launch')} q{q:g} {r.get('belief','prior')} tau{tau:g}"

# key now includes L -- see header note 1
cells = defaultdict(lambda: defaultdict(dict))
for r in rows:
    cells[(r["layout"], r["M"], r["Emax"], r["L"], r["coord"], variant(r), r["Th"])][r["seed"]][r["K"]] = r

def sign_test(d):
    """Two-sided sign test, ties dropped."""
    d = np.asarray([x for x in d if x != 0.0]); n = len(d)
    if n == 0: return 0, 0, 1.0
    s = int((d > 0).sum())
    p = sum(math.comb(n, k) for k in range(max(s, n-s), n+1)) / 2**n * 2
    return s, n, min(p, 1.0)

print(f"{'cell':52} {'inst':>4} {'argmin':>7} {'law':>5} {'agree':>6} {'agree±1':>8} "
      f"{'J(law) vs J(law+1)':>20} {'J(law) vs J(law-1)':>20} {'r_max/r_c':>9} {'bnd':>5} {'regime':>7}")
warned = set()
for key in sorted(cells):
    inst = cells[key]; argmins = []; laws = []; hits = 0; hits1 = 0
    dplus = []; dminus = []; ratios = []; bnds = []
    for seed, byK in inst.items():
        Ks = sorted(byK); J = {K: byK[K]["J"] for K in Ks}
        am = min(Ks, key=lambda K: J[K]); argmins.append(am)
        r0 = byK[Ks[0]]
        # guard: K_reach_i is instance geometry and must not vary with K. If it does, rows from
        # different field sizes or layouts have been merged into one cell.
        spread = max(abs(byK[K]["K_reach_i"] - r0["K_reach_i"]) for K in Ks)
        if spread > 1e-6 and key not in warned:
            warned.add(key)
            print(f"  !! {key[0]} M={key[1]} E={key[2]:.1e} L={key[3]:g}: K_reach_i varies by "
                  f"{spread:.3f} within seed {seed} -- rows from different instances merged", file=sys.stderr)
        law = min(math.floor(r0["K_reach_i"]), r0["K_commute_i"])
        law_int = int(round(law)) if law != math.floor(r0["K_reach_i"]) else int(law)
        laws.append(law_int); ratios.append(r0["rmax_over_rc"]); bnds.append(r0["regime_bnd"])
        hits += (am == law_int); hits1 += (abs(am - law_int) <= 1)
        if law_int in J and law_int+1 in J: dplus.append(J[law_int+1] - J[law_int])
        if law_int in J and law_int-1 in J: dminus.append(J[law_int-1] - J[law_int])
    n = len(inst)
    sp = sign_test(dplus) if dplus else (0,0,float("nan"))
    sm = sign_test(dminus) if dminus else (0,0,float("nan"))
    regime = "reach" if np.mean(ratios) > np.mean(bnds) else "commute"
    lab = f"{key[0]} M={key[1]} E={key[2]:.1e} L={key[3]/1000:g}km {key[4]} {key[5]} {key[6]/3600:.0f}h"
    print(f"{lab:52} {n:4d} {np.mean(argmins):7.2f} {np.mean(laws):5.2f} {hits/n:6.2f} {hits1/n:8.2f} "
          f"{sp[0]:>3}/{sp[1]:<3} p={sp[2]:.3f}   {sm[0]:>3}/{sm[1]:<3} p={sm[2]:.3f}   "
          f"{np.mean(ratios):9.2f} {np.mean(bnds):5.2f} {regime:>7}")

print("\ncolumns: argmin = mean per-instance argmin J; law = mean per-instance min(floor K_reach, K_commute);")
print("agree = P(argmin == law); J(law) vs J(law+1) = #instances where J(law+1) > J(law) / n, sign-test p (ties dropped).")
print("cells are keyed by (layout, M, Emax, L, coord, variant, Th) -- L rows are the Table VI sweep.")