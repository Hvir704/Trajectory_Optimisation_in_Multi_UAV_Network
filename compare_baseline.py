"""compare_baseline.py -- is argmin_K J a property of the problem or of the SA planner?

usage: python compare_baseline.py grid.csv baseline.csv [--label rr_tour] [--per-cell]

Per instance (layout, M, Emax, L, seed), on the INTERSECTION of K values present for both
planners (the banked SA grid is missing some K rows; comparing argmins over different K sets
would be a silent bias):
  am_sa, am_b          per-instance argmin J
  rule_sa              min(floor K_reach, K_commute) from the SA rows, exactly as analyze.py
                       (smallest-K row's measured r_c and P_bar)
  rule_b               the same rule evaluated from the BASELINE's own measured r_c, P_bar
  edge_b               baseline argmin sits on the boundary of the swept K range (censored)
Reported per layout family: exact / within-one agreement of am_b with rule_sa, rule_b, am_sa;
J gap at each planner's own optimum; and the baseline's regret if it were sized by SA's argmin.
Only Th = 43200 rows are used (the backfill bug).
"""
import sys, csv, math, argparse
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("sa"); ap.add_argument("base")
ap.add_argument("--label", default="baseline"); ap.add_argument("--per-cell", action="store_true")
ap.add_argument("--Th", type=float, default=43200.0)
a = ap.parse_args()


def load(path, planner=None):
    out = defaultdict(dict)   # inst -> K -> row
    for r in csv.DictReader(open(path)):
        if "M" not in r or r["M"] in ("", "M"): continue
        if float(r["Th"]) != a.Th: continue
        if r["coord"] != "exclude" or r.get("replan", "launch") != "launch": continue
        if str(r.get("divert", "0")) not in ("0", "0.0") or float(r.get("q", 0)) != 0: continue
        if r.get("belief", "prior") != "prior" or float(r.get("tau", 0)) != 0: continue
        if planner and r.get("planner", "sa") != planner: continue
        inst = (r["layout"], int(r["M"]), float(r["Emax"]), float(r.get("L", 12600)), int(r["seed"]))
        K = int(r["K"])
        if K in out[inst]: raise SystemExit(f"duplicate row {inst} K={K} in {path} -- resume-skip broke?")
        out[inst][K] = {k: r[k] for k in ("J", "K_reach_i", "K_commute_i", "n_never", "dup_frac", "trunc_frac", "r_c", "P_bar")}
    return out


def rule(byK, Ks):
    r0 = byK[min(Ks)]
    kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"])
    law = min(kr, kc)
    return int(law) if law == kr else int(round(law))


sa = load(a.sa, "sa")
bb = load(a.base)
common = sorted(set(sa) & set(bb))
if not common: raise SystemExit("no common instances")

recs = []
for inst in common:
    Ks = sorted(set(sa[inst]) & set(bb[inst]))
    if len(Ks) < 3: continue
    Js = {K: float(sa[inst][K]["J"]) for K in Ks}
    Jb = {K: float(bb[inst][K]["J"]) for K in Ks}
    am_s = min(Ks, key=Js.get); am_b = min(Ks, key=Jb.get)
    rs = rule(sa[inst], Ks); rb = rule(bb[inst], Ks)
    recs.append(dict(inst=inst, fam=inst[0], Ks=Ks, am_s=am_s, am_b=am_b, rs=rs, rb=rb,
                     in_rng=(Ks[0] <= rs <= Ks[-1]),
                     edge_b=am_b in (Ks[0], Ks[-1]), edge_s=am_s in (Ks[0], Ks[-1]),
                     gap=Jb[am_b] / Js[am_s] - 1.0,
                     gap_at_rule=(Jb[rs] / Js[rs] - 1.0) if rs in Js else float("nan"),
                     reg_b_at_sa=Jb[am_s] / Jb[am_b] - 1.0,
                     reg_b_at_rule=(Jb[rs] / Jb[am_b] - 1.0) if rs in Jb else float("nan"),
                     trunc_b=np.mean([float(bb[inst][K]["trunc_frac"]) for K in Ks]),
                     rc_ratio=float(bb[inst][min(Ks)]["r_c"]) / max(float(sa[inst][min(Ks)]["r_c"]), 1e-9)))


def agree(x, y):
    d = np.abs(np.array(x) - np.array(y)); return np.mean(d == 0), np.mean(d <= 1)


def summarise(name, R):
    if not R: return
    am_s = [r["am_s"] for r in R]; am_b = [r["am_b"] for r in R]
    rs = [r["rs"] for r in R]; rb = [r["rb"] for r in R]
    e1, w1 = agree(am_s, rs); e2, w2 = agree(am_b, rs); e3, w3 = agree(am_b, rb); e4, w4 = agree(am_b, am_s)
    signed = np.mean(np.array(am_b) - np.array(am_s))
    gap = np.array([r["gap"] for r in R]); reg = np.array([r["reg_b_at_sa"] for r in R])
    print(f"{name:26s} n={len(R):3d} | SA am vs rule {e1:.2f}/{w1:.2f} | {a.label} am vs SA-rule {e2:.2f}/{w2:.2f}"
          f" | vs own-rule {e3:.2f}/{w3:.2f} | am_b vs am_sa {e4:.2f}/{w4:.2f} bias {signed:+.2f}"
          f" | J gap med {np.median(gap):+.1%} [{np.min(gap):+.0%},{np.max(gap):+.0%}]"
          f" | reg_b@am_sa med {np.median(reg):.1%} p90 {np.quantile(reg,.9):.1%}"
          f" | edge b {np.mean([r['edge_b'] for r in R]):.2f} sa {np.mean([r['edge_s'] for r in R]):.2f}"
          f" both {np.mean([r['edge_b'] and r['edge_s'] for r in R]):.2f}")


print(f"instances compared: {len(recs)}  (exact/within-one shown as e/w)\n")
prim = [r for r in recs if r["inst"][3] == 12600.0]
for fam in ("paper", "ring", "core"):
    summarise(f"{fam} (L=12.6km)", [r for r in prim if r["fam"] == fam])
summarise("ALL primary", prim)
summarise("ALL primary, rule in range", [r for r in prim if r["in_rng"]])
summarise("L sweep (paper M100)", [r for r in recs if r["inst"][3] != 12600.0])
summarise("ALL", recs)

if a.per_cell:
    print()
    cells = defaultdict(list)
    for r in recs: cells[r["inst"][:4]].append(r)
    print(f"{'cell':34s} {'n':>3} {'am_sa':>6} {'am_b':>6} {'rule':>5} {'rule_b':>6} {'b=sa':>5} {'b~sa':>5} {'b=rule':>6} {'gap':>7} {'regb@sa':>8} {'edge_b':>6} {'edge_s':>6} {'rc_b/sa':>7}")
    for c in sorted(cells):
        R = cells[c]
        f = lambda k: np.mean([r[k] for r in R])
        e, w = agree([r["am_b"] for r in R], [r["am_s"] for r in R])
        er, _ = agree([r["am_b"] for r in R], [r["rs"] for r in R])
        lab = f"{c[0]} M={c[1]} E={c[2]/1e6:g}MJ L={c[3]/1e3:g}"
        print(f"{lab:34s} {len(R):3d} {f('am_s'):6.2f} {f('am_b'):6.2f} {f('rs'):5.2f} {f('rb'):6.2f} {e:5.2f} {w:5.2f} {er:6.2f} "
              f"{np.median([r['gap'] for r in R]):+7.1%} {np.median([r['reg_b_at_sa'] for r in R]):8.1%} {f('edge_b'):6.2f} {f('edge_s'):6.2f} {f('rc_ratio'):7.2f}")
