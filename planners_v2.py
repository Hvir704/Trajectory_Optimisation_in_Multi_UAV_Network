"""planners_v2.py -- every cross-planner number of Sec. VI-J from one set of files (union K grid, registered rule).

Deployment = (layout, M, Emax, seed), primary grid (M <= 200, 1.5/3 MJ, seeds 1-12, L = 12.6 km, 12 h).
Own optimum = argmin_K J over all of that planner's rows for the deployment. Completeness (registered rule):
the optimum is not at the largest K and the three largest K each exceed 1.5x the minimum. Incomplete sweeps are
listed in section 1 but still enter the comparisons; the numbers are final only when section 1 reports 0 incomplete.  The rule's l per deployment comes from the SA rows (smallest K), as in analyze_liu.py.

Prints
  1) completeness per planner;
  2) tab:published -- C4 vs X (median J*_C4/J*_X - 1, 95% bootstrap CI as own_opt_compare.py, C4 lower n/N,
     and at K = l);
  3) X vs cluster patrol and X vs Liu (the 'neither beats the clustering baseline' and Liu paragraphs);
  4) Table IX dagger rows and tab:split -- exact / within-one agreement with l, all / test passes / test flags,
     and agreement with SA's own optimum;
  5) cost of sizing with l (median, 90th percentile) per planner;
  6) C4 vs SA (secondary comparison).
usage: python planners_v2.py --sa SA.csv [SA_FILL.csv ...] --files F1.csv F2.csv ... --verdicts verdicts.csv
"""
import argparse, csv, math
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--sa", nargs="+", required=True); ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--verdicts", required=True)
a = ap.parse_args()

D = defaultdict(lambda: defaultdict(dict)); RULE = {}
def take(path, force=None):
    for r in csv.DictReader(open(path)):
        if not r or r.get("M") in (None, "", "M"): continue
        try: M, E, s = int(float(r["M"])), float(r["Emax"]), int(float(r["seed"]))
        except (KeyError, TypeError, ValueError): continue
        if M > 200 or E not in (1.5e6, 3e6) or s > 12 or float(r.get("Th", 43200)) != 43200: continue
        if float(r.get("L", 12600)) != 12600 or r.get("coord", "exclude") != "exclude": continue
        if r.get("replan", "launch") != "launch" or str(r.get("divert", "0")) not in ("0", "0.0"): continue
        if r.get("belief", "prior") != "prior" or float(r.get("q", 0)) != 0 or float(r.get("tau", 0)) != 0: continue
        dep = (r["layout"], M, E, s); pl = force or r.get("planner", "sa")
        try: J = float(r["J"])
        except (TypeError, ValueError): continue
        D[dep][pl][int(float(r["K"]))] = J
        if pl == "sa":
            k = int(float(r["K"]))
            if dep not in RULE or k < RULE[dep][0]: RULE[dep] = (k, r)
for f in a.sa: take(f, "sa")
for f in a.files: take(f)
V = {(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"])): r["verdict_current"] for r in csv.DictReader(open(a.verdicts))}

def ell(dep):
    r0 = RULE[dep][1]; kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"]); law = min(kr, kc)
    return int(law) if law == kr else int(round(law))
def opt(b): return min(b, key=b.get)
def complete(b):
    Ks = sorted(b); m = min(b.values())
    return opt(b) != Ks[-1] and all(b[K] > 1.5 * m for K in Ks[-3:])
def boot(v, n=5000, seed=0):                     # = own_opt_compare.boot_median
    v = np.asarray(v); rng = np.random.default_rng(seed)
    if len(v) < 2: return (float("nan"),) * 2
    m = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1); return tuple(np.percentile(m, [2.5, 97.5]))

PL = [p for p in ("sa", "cyc", "cluster_patrol", "gzz_rev", "asghar_rev", "liu_mpga", "rr_tour", "rr_sweep") if any(p in D[d] for d in D)]
NAME = {"sa": "SA", "cyc": "C4", "cluster_patrol": "cluster patrol", "gzz_rev": "Gao et al.", "asghar_rev": "Asghar et al.",
        "liu_mpga": "Liu et al.", "rr_tour": "tour patrol", "rr_sweep": "sweep patrol"}
deps = sorted(d for d in D if d in RULE)
print(f"deployments with SA rows: {len(deps)}")

print("\n1) completeness (registered rule: optimum interior, three largest K > 1.5 x min)")
for p in PL:
    have = [d for d in deps if p in D[d]]; inc = [d for d in have if not complete(D[d][p])]
    print(f"  {NAME[p]:15s} deployments {len(have):3d}  incomplete {len(inc):3d}" + (f"  e.g. {inc[:4]}" if inc else ""))

def compare(X, Y, at_l=False):
    """median of J*_X/J*_Y - 1 per family; X lower count."""
    out = {}
    for fam in ("paper", "core", "ring", "all"):
        rat = []
        for d in deps:
            if (fam != "all" and d[0] != fam) or X not in D[d] or Y not in D[d]: continue
            bX, bY = D[d][X], D[d][Y]
            if at_l:
                l = ell(d)
                if l in bX and l in bY: rat.append(bX[l] / bY[l] - 1)
            else:
                rat.append(min(bX.values()) / min(bY.values()) - 1)
        if rat:
            rat = np.array(rat); lo, hi = boot(rat)
            out[fam] = (np.median(rat), lo, hi, int((rat < 0).sum()), len(rat))
    return out

def show(title, X, Y, at_l=False):
    r = compare(X, Y, at_l)
    if not r: return
    print(f"  {title}")
    for fam, (m, lo, hi, w, n) in r.items(): print(f"     {fam:5s} {m:+7.1%} [{lo:+.1%}, {hi:+.1%}]  {NAME[X]} lower {w}/{n}")

print("\n2) tab:published: C4 vs X at each planner's own optimum, and at K = l")
for X in ("cluster_patrol", "gzz_rev", "asghar_rev", "liu_mpga"):
    if "cyc" in PL and X in PL:
        show(f"C4 vs {NAME[X]} (own optimum)", "cyc", X); show(f"C4 vs {NAME[X]} (K = l)", "cyc", X, True)

print("\n3) X vs cluster patrol and X vs Liu (own optimum; positive = X higher)")
for Y in ("cluster_patrol", "liu_mpga"):
    for X in ("sa", "cluster_patrol", "gzz_rev", "asghar_rev"):
        if X != Y and X in PL and Y in PL:
            r = compare(X, Y)
            if "all" in r: print(f"  {NAME[X]:15s} vs {NAME[Y]:15s} all {r['all'][0]:+.1%}  " +
                                  "  ".join(f"{f} {r[f][0]:+.1%}" for f in ("paper", "core", "ring") if f in r))

print("\n4) rule agreement (exact / within one) with l, split by the design-time test; agreement with SA's optimum")
ok = lambda d: V.get(d) in ("pass", "not_applicable")
def agree(p, sel):
    x = [(opt(D[d][p]), ell(d)) for d in sel if p in D[d]]
    return (np.mean([k == l for k, l in x]), np.mean([abs(k - l) <= 1 for k, l in x]), len(x)) if x else (float("nan"), float("nan"), 0)
print(f"  {'planner':15s} {'all':>16s} {'passes':>18s} {'flags':>18s} {'=SA / +-1 SA':>16s}")
for p in PL:
    A_ = agree(p, deps); P_ = agree(p, [d for d in deps if ok(d)]); F_ = agree(p, [d for d in deps if not ok(d)])
    s = [(opt(D[d][p]), opt(D[d]["sa"])) for d in deps if p in D[d] and "sa" in D[d]]
    es = f"{np.mean([x == y for x, y in s]):.2f} / {np.mean([abs(x - y) <= 1 for x, y in s]):.2f}" if p != "sa" else "--"
    print(f"  {NAME[p]:15s} {A_[0]:.2f} / {A_[1]:.2f} ({A_[2]:3d})  {P_[0]:.2f} / {P_[1]:.2f} ({P_[2]:3d})  "
          f"{F_[0]:.2f} / {F_[1]:.2f} ({F_[2]:3d})  {es:>14s}")
print(f"  test passes in {sum(1 for d in deps if ok(d))} of {len(deps)} deployments "
      f"(" + ", ".join(f"{f} {sum(1 for d in deps if ok(d) and d[0] == f)}" for f in ("core", "paper", "ring")) + ")")

print("\n5) cost of sizing with l: J(l)/J* - 1 (exploratory)")
for p in PL:
    c = np.array([D[d][p][ell(d)] / min(D[d][p].values()) - 1 for d in deps if p in D[d] and ell(d) in D[d][p]])
    if len(c): print(f"  {NAME[p]:15s} median {np.median(c):5.1%}  p90 {np.quantile(c, .9):6.1%}  (n={len(c)})")

print("\n6) C4 vs SA (secondary comparison)")
if "cyc" in PL: show("C4 vs SA (own optimum)", "cyc", "sa")
