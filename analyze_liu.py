"""analyze_liu.py -- the registered analysis of LIU_EXTENSION_REGISTRATION.md.
usage: python analyze_liu.py --liu <Liu CSVs...> --sa sa_rerun.csv --others eval_all_v2.csv gzz_eval.csv asghar_eval.csv
          bl_fill_gzz.csv bl_fill_asghar.csv bl_fill_asghar_rule15.csv bl_fill_cluster.csv --verdicts criterion5_verdicts.csv
Every statistic is over the deployments where Liu exists (216 when the extension is complete)."""
import csv, math, argparse
from collections import defaultdict
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--liu", nargs="+", required=True); ap.add_argument("--sa", required=True)
ap.add_argument("--others", nargs="*", default=[]); ap.add_argument("--verdicts", required=True)
a = ap.parse_args()
D = defaultdict(lambda: defaultdict(dict)); RULE = {}
def take(path, force=None):
    for r in csv.DictReader(open(path)):
        M, E, s = int(float(r["M"])), float(r["Emax"]), int(float(r["seed"]))
        if M > 200 or E not in (1.5e6, 3e6) or s > 12 or float(r.get("Th", 43200)) != 43200: continue
        if float(r.get("L", 12600)) != 12600 or r.get("coord", "exclude") != "exclude": continue
        dep = (r["layout"], M, E, s); pl = force or r.get("planner", "sa")
        if pl == "cyc_sector0": continue
        D[dep][pl][int(float(r["K"]))] = float(r["J"])
        if pl == "sa":
            k = int(float(r["K"]))
            if dep not in RULE or k < RULE[dep][0]: RULE[dep] = (k, r)
take(a.sa, "sa")
for f in a.others: take(f)
for f in a.liu: take(f, "liu_mpga")
V = {(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"])): r["verdict_current"] for r in csv.DictReader(open(a.verdicts))}
def ell(dep):
    r0 = RULE[dep][1]; kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"]); law = min(kr, kc)
    return int(law) if law == kr else int(round(law))
deps = sorted(d for d in D if "liu_mpga" in D[d])
rng = np.random.default_rng(20261007)
def complete(b):
    Ks = sorted(b); m = min(b.values()); am = min(b, key=b.get)
    return am != Ks[-1] and all(b[K] > 1.5 * m for K in Ks[-3:])
inc = [d for d in deps if not complete(D[d]["liu_mpga"])]
print(f"deployments with Liu: {len(deps)}  (Liu sweeps not yet meeting the 1.5x rule / top-edge: {len(inc)})")
NAMES = {"cyc": "C4", "sa": "SA", "cluster_patrol": "cluster patrol", "gzz_rev": "Gao et al.", "asghar_rev": "Asghar et al."}
def cmp(X, at_l=False):
    out = {}
    for fam in ("paper", "core", "ring", "all"):
        rat = []
        for d in deps:
            if fam != "all" and d[0] != fam: continue
            if X not in D[d]: continue
            bX, bL = D[d][X], D[d]["liu_mpga"]
            if at_l:
                l = ell(d)
                if l not in bX or l not in bL: continue
                rat.append(bX[l] / bL[l] - 1)
            else:
                rat.append(min(bX.values()) / min(bL.values()) - 1)
        if not rat: continue
        rat = np.array(rat); bs = [np.median(rng.choice(rat, len(rat))) for _ in range(10000)]
        out[fam] = (np.median(rat), np.quantile(bs, .025), np.quantile(bs, .975), int(np.sum(rat < 0)), len(rat))
    return out
print("\nPRIMARY: C4 vs Liu, own optimum (median J*_C4/J*_Liu - 1, 95% CI, C4 lower)")
P = cmp("cyc")
for fam, (m, lo, hi, w, n) in P.items(): print(f"  {fam:5s} {m:+.1%} [{lo:+.1%}, {hi:+.1%}]  {w}/{n}")
print("  criterion (C4 lower in median in EACH family):", all(P[f][0] < 0 for f in ("paper", "core", "ring") if f in P) if all(f in P for f in ("paper","core","ring")) else "not all families present")
print("\nSECONDARY 1: C4 vs Liu at K = l");
for fam, (m, lo, hi, w, n) in cmp("cyc", True).items(): print(f"  {fam:5s} {m:+.1%} [{lo:+.1%}, {hi:+.1%}]  {w}/{n}")
print("\nSECONDARY 2: each planner X vs Liu, own optimum (median J*_X/J*_Liu - 1; negative = X lower)")
for X in ("sa", "cluster_patrol", "gzz_rev", "asghar_rev"):
    r = cmp(X); print(f"  {NAMES[X]:15s} " + "  ".join(f"{f} {r[f][0]:+.1%} [{r[f][1]:+.1%},{r[f][2]:+.1%}] {r[f][3]}/{r[f][4]}" for f in r))
print("\nSECONDARY 3-5: Liu rule agreement (exact / within one), split by the design-time test, cost of sizing with l")
def agree(sel):
    x = [(min(D[d]["liu_mpga"], key=D[d]["liu_mpga"].get), ell(d)) for d in sel]
    return (np.mean([k == l for k, l in x]), np.mean([abs(k - l) <= 1 for k, l in x]), len(x)) if x else (float("nan"),) * 3
for fam in ("all", "core", "paper", "ring"):
    sel = [d for d in deps if fam == "all" or d[0] == fam]; e, w, n = agree(sel); print(f"  {fam:5s} {e:.2f} / {w:.2f}  (n={n})")
ok = lambda d: V.get(d) in ("pass", "not_applicable")
e, w, n = agree([d for d in deps if ok(d)]); print(f"  test passes: {e:.2f} / {w:.2f} (n={n})")
e, w, n = agree([d for d in deps if not ok(d)]); print(f"  test flags:  {e:.2f} / {w:.2f} (n={n})")
cost = np.array([D[d]["liu_mpga"][ell(d)] / min(D[d]["liu_mpga"].values()) - 1 for d in deps if ell(d) in D[d]["liu_mpga"]])
print(f"  cost of sizing with l: median {np.median(cost):.1%}, 90th percentile {np.quantile(cost, .9):.1%} (n={len(cost)}; exploratory)")
seen = [d for d in deps if d[1] == 50 and d[2] == 1.5e6]
print(f"\nSECONDARY 6: already-seen deployments (M=50, 1.5 MJ): {len(seen)} of {len(deps)}")
