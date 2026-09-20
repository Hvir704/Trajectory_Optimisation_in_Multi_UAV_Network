"""baseline_table.py -- the consolidated external-baseline table for Sec. VII-J.

Every planner is compared against (a) SA's per-instance argmin and (b) the closed-form rule,
directly rather than by transitivity. The rule is evaluated per instance exactly as analyze.py
does, min(floor K_reach, round K_commute), from the smallest-K row of that planner's own rows --
K_reach is geometric and identical across planners, K_commute uses that planner's measured r_c and
P_bar, which is the honest comparison since the commute cap is defined through the realised rate
curve. `--rule-from sa` instead evaluates every planner against SA's rule number.

usage: python baseline_table.py sa_rerun.csv rr_tour_win.csv rr_sweep_win.csv m40_cp.csv
       python baseline_table.py sa_rerun.csv rr_tour_win.csv --by-family --rule-from sa
Rows are restricted to the instances present for SA and that planner, on the K intersection.
Cells whose argmin sits at the top of the swept K range are reported and can be dropped
with --drop-censored.
"""
import argparse, csv, math
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("sa"); ap.add_argument("others", nargs="+")
ap.add_argument("--Th", type=float, default=43200.0)
ap.add_argument("--by-family", action="store_true")
ap.add_argument("--rule-from", choices=["own", "sa"], default="own")
ap.add_argument("--drop-censored", action="store_true")
a = ap.parse_args()


def load(path):
    out = defaultdict(dict); planners = set()
    for r in csv.DictReader(open(path)):
        if "M" not in r or r["M"] in ("", "M"): continue
        if float(r["Th"]) != a.Th or r["coord"] != "exclude" or r.get("replan", "launch") != "launch": continue
        if str(r.get("divert", "0")) not in ("0", "0.0") or float(r.get("q", 0)) != 0: continue
        if r.get("belief", "prior") != "prior" or float(r.get("tau", 0)) != 0: continue
        planners.add(r.get("planner", "sa"))
        inst = (r["layout"], int(r["M"]), float(r["Emax"]), float(r.get("L", 12600)), int(r["seed"]))
        K = int(r["K"])
        if K in out[inst]: raise SystemExit(f"duplicate row {inst} K={K} in {path}")
        out[inst][K] = r
    if len(planners) != 1: raise SystemExit(f"{path} mixes planners: {planners}")
    return out, planners.pop()


def rule_of(byK):
    r0 = byK[min(byK)]
    kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"])
    law = min(kr, kc)
    return int(law) if law == kr else int(round(law))


sa, _ = load(a.sa)
print(f"{'planner':10s} {'fam':6s} {'n':>4} {'J vs SA':>9} {'=SA':>5} {'~SA':>5} {'=rule':>6} {'~rule':>6} {'bias':>6} {'cens':>5}")
rows_sa = defaultdict(list)

for path in [a.sa] + a.others:
    bb, name = load(path)
    recs = []
    for inst in sorted(set(sa) & set(bb)):
        Ks = sorted(set(sa[inst]) & set(bb[inst]))
        if len(Ks) < 3: continue
        Js = {K: float(sa[inst][K]["J"]) for K in Ks}; Jb = {K: float(bb[inst][K]["J"]) for K in Ks}
        am_s = min(Ks, key=Js.get); am_b = min(Ks, key=Jb.get)
        rl = rule_of(sa[inst]) if (a.rule_from == "sa" or name == "sa") else rule_of(bb[inst])
        cens = am_b == max(Ks)
        if a.drop_censored and (cens or am_s == max(Ks)): continue
        recs.append((inst[0], am_s, am_b, rl, Jb[am_b] / Js[am_s] - 1.0, cens))
    groups = {"all": recs}
    if a.by_family:
        for f in ("paper", "ring", "core"):
            groups[f] = [r for r in recs if r[0] == f]
    for g, R in groups.items():
        if not R: continue
        am_s = np.array([r[1] for r in R]); am_b = np.array([r[2] for r in R]); rl = np.array([r[3] for r in R])
        gap = np.array([r[4] for r in R])
        print(f"{name:10s} {g:6s} {len(R):4d} {np.median(gap):+8.1%} "
              f"{np.mean(am_b == am_s):5.2f} {np.mean(abs(am_b - am_s) <= 1):5.2f} "
              f"{np.mean(am_b == rl):6.2f} {np.mean(abs(am_b - rl) <= 1):6.2f} "
              f"{np.mean(am_b - am_s):+6.2f} {np.mean([r[5] for r in R]):5.2f}")
print("\n=SA/~SA: agreement of this planner's argmin with SA's. =rule/~rule: with the closed-form rule,")
print("computed directly, not by transitivity. bias: mean(argmin - SA argmin). cens: fraction whose")
print("argmin sits at the top of the swept K range (not a reliable optimum).")