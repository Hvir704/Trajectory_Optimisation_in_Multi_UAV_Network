"""alpha_fit.py -- Appendix B: Table XV (fitted commute exponent) and the per-sortie energy regressions.

Table XV, per ring cell, from grid_final.csv (SA, standard settings, L = 12.6 km):
  stranding-free rows (n_never = 0); t_c = cell mean of 2 r_c / v; fit log n = log c + alpha log(U/K - Pf t_c);
  tau_s = cell mean of (T_s - t_c) / n with T_s = (T_s/t_c column) x nominal t_c; the alpha-corrected fleet size maximises
  K n(K) / (t_c + n(K) tau_s) over integers; "alpha = 1" is round(K_commute) read at the reference K (cell mean of the per-instance values);
  "measured" is the mean per-instance argmin of J.
Per-sortie regressions (needs --sorties, from sortie_regress.py), per cell, on ONE sample: post-burn-in sorties of
stranding-free runs with n >= 2: slopes of log E_hop, log E_hov and log E_p on log n, with 95% intervals from a
bootstrap over runs (sorties of one run are not independent); the hover share sum E_hov / sum E_p; and the ratio of
measured E_hov / n to Ph H / Gamma (the conservation argument), cell mean.

usage: python alpha_fit.py grid_final.csv [--sorties sortie_ring.csv]
"""
import sys, csv, math, collections
import numpy as np
from dyn_env import DynParams

args = [x for x in sys.argv[1:] if not x.startswith("--")]
sfile = sys.argv[sys.argv.index("--sorties") + 1] if "--sorties" in sys.argv else None
if sfile in args: args.remove(sfile)
p = DynParams(); Pf, Ph, v = p.Pf, p.Ph, p.v
rng = np.random.default_rng(20261010)

std = lambda r: (r["layout"] == "ring" and float(r["L"]) == 12600 and r.get("planner", "sa") == "sa"
                 and float(r["Th"]) == 43200 and r["coord"] == "exclude" and r["replan"] == "launch")
cells = collections.defaultdict(list)
for r in csv.DictReader(open(args[0])):
    if std(r): cells[(int(float(r["M"])), float(r["Emax"]))].append(r)

for restrict in (False, True):
  if not restrict:
    print("Table XV (ring): cell, measured mean optimum, alpha=1 (round K_commute), alpha-corrected, fitted alpha, 1/alpha")
  else:
    print("Sensitivity: the same fit on K >= K_ref only (original sweep and upward extensions; without the smaller K added "
          "for other planners)")
  for c in [(50, 3e6), (100, 3e6), (200, 3e6), (100, 1.5e6)]:
    x = cells.get(c, [])
    sf = [r for r in x if float(r["n_never"]) == 0
          and (not restrict or int(float(r["K"])) >= int(float(r.get("K_ref") or 0)))]
    if len(sf) < 3: print(f"  M={c[0]} {c[1]/1e6:.1f} MJ: too few stranding-free rows ({len(sf)})"); continue
    U = (1 - p.rho) * c[1]
    tc = float(np.mean([2 * float(r["r_c"]) / v for r in sf]))
    K = np.array([float(r["K"]) for r in sf]); n = np.array([float(r["mean_n"]) for r in sf])
    X = U / K - Pf * tc; ok = (X > 0) & (n > 0)
    alpha, logc = np.polyfit(np.log(X[ok]), np.log(n[ok]), 1)
    # T_s_over_t_c is the sortie duration over the NOMINAL commute time p.t_c (dyn_env metrics), not over 2 r_c / v
    tau_s = float(np.mean((np.array([float(r["T_s_over_t_c"]) for r in sf]) * p.t_c - tc) / n))
    def G(k):
        e = U / k - Pf * tc
        if e <= 0: return -1.0
        nn = math.exp(logc) * e ** alpha
        return k * nn / (tc + nn * tau_s)
    k_corr = max(range(1, 200), key=G)
    by = collections.defaultdict(dict)
    for r in x: by[r["seed"]][int(float(r["K"]))] = float(r["J"])
    meas = np.mean([min(b, key=b.get) for b in by.values()])
    # the rule's input is read at the instance's reference K (first K of its original sweep, column K_ref), so that
    # fills added for other planners do not change it
    kr = {r["seed"]: int(float(r["K_ref"])) if r.get("K_ref") not in (None, "") else min(by[r["seed"]]) for r in x}
    k1 = np.mean([round(float(r["K_commute_i"])) for r in x if int(float(r["K"])) == kr[r["seed"]]])
    print(f"  M={c[0]:3d} {c[1]/1e6:.1f} MJ  measured {meas:.2f}  alpha=1 {k1:.2f}  corrected {k_corr}  "
          f"alpha {alpha:.2f}  1/alpha {1/alpha:.2f}   ({len(sf)} stranding-free rows, {len(by)} instances)")

if not sfile:
    sys.exit(0)
print("\nPer-sortie regressions (ring, stranding-free runs, sorties with n >= 2; 95% bootstrap over runs)")
runs = collections.defaultdict(list)
for r in csv.DictReader(open(sfile)):
    if int(r["sortie"]) < 0 or float(r["n_never"]) != 0 or int(r["n"]) < 2: continue
    runs[(int(r["M"]), float(r["Emax"]), int(r["K"]), int(r["seed"]))].append(r)
ratios_all = []
for c in sorted({k[:2] for k in runs}):
    keys = [k for k in runs if k[:2] == c]
    data = {k: np.array([[float(r["n"]), float(r["E_hop"]), float(r["E_hov"]), float(r["E_p"])] for r in runs[k]]) for k in keys}
    def slopes(ks):
        a = np.vstack([data[k] for k in ks]); a = a[(a[:, 1] > 0) & (a[:, 2] > 0)]
        ln = np.log(a[:, 0])
        return [np.polyfit(ln, np.log(a[:, j]), 1)[0] for j in (1, 2, 3)]
    pt = slopes(keys)
    reps = np.array([slopes([keys[i] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(1000)])
    lo, hi = np.quantile(reps, 0.025, axis=0), np.quantile(reps, 0.975, axis=0)
    allr = np.vstack([data[k] for k in keys]); share = allr[:, 2].sum() / allr[:, 3].sum()
    rat = []
    for k in keys:
        r0 = runs[k][0]; pred = Ph * float(r0["H"]) / float(r0["Gamma"])
        rat.append(np.mean(data[k][:, 2] / data[k][:, 0]) / pred)
    ratios_all.append(np.mean(rat))
    nm = ("hop", "hover", "productive")
    print(f"  M={c[0]:3d} {c[1]/1e6:.1f} MJ  runs {len(keys)}  sorties {len(allr)}  " +
          "  ".join(f"beta_{nm[j]} {pt[j]:.2f} [{lo[j]:.2f}, {hi[j]:.2f}]" for j in range(3)) +
          f"  hover share {share:.1%}  (E_hov/n)/(Ph H/Gamma) {np.mean(rat):.3f}")
if ratios_all:
    print(f"  conservation constant across cells: {np.mean(ratios_all):.3f} +- {np.std(ratios_all, ddof=1) if len(ratios_all) > 1 else 0:.3f} (sd)")
