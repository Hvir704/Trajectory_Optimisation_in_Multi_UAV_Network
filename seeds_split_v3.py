"""seeds_split_v3.py -- POST-REGISTRATION sensitivity check (added 2026-10-12, after the second review).

Seeds 13-15 of the held-out test are the pilot seeds of the planner comparison: the same fields had been swept with
SA, C4, the cluster patrol, Gao and Asghar under the pre-v2.2 simulator (pilot*.csv on sim-v2, 29-30 Sep and
7 Oct). Seeds 16-20 were never run before the held-out test. This script reports the held-out evaluation separately
for the two seed groups, from the raw runs and the registered predictions file (unchanged).

Regret = J(predicted K) / J(swept optimum) - 1, with the nearest swept K if the predicted K was not swept.
p90 is given both as linear interpolation (np.quantile default, as in THEORY_V3.md) and nearest rank (as in the
paper's Table VII).

usage: python seeds_split_v3.py results_v2/heldout_v3.csv [heldout_v3_predictions.csv]
"""
import sys, csv, math, collections
import numpy as np

runs = sys.argv[1]; predf = sys.argv[2] if len(sys.argv) > 2 else "heldout_v3_predictions.csv"
P = {(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"])): r for r in csv.DictReader(open(predf))}
D = collections.defaultdict(dict)
for r in csv.DictReader(open(runs)):
    if float(r["Th"]) != 43200: continue
    D[(r["planner"], r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]))][int(r["K"])] = float(r["J"])

def nr90(x):  # nearest rank: smallest observed value with at least 90% at or below
    x = np.sort(x); return x[max(0, math.ceil(0.9 * len(x)) - 1)]
def sign_p(b, w):  # two-sided exact sign test
    n = b + w
    if n == 0: return float("nan")
    k = min(b, w); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

groups = {"seeds 13-15 (pilot seeds)": range(13, 16), "seeds 16-20 (never run before)": range(16, 21), "all 13-20": range(13, 21)}
for pl in ("sa", "rr_tour", "cyc"):
    for gname, seeds in groups.items():
        for fam in ("core", "paper", "ring", "all"):
            R = collections.defaultdict(list); paired = []
            for k, pr in P.items():
                if k[3] not in seeds or (fam != "all" and k[0] != fam): continue
                c = D.get((pl,) + k)
                if not c: continue
                Ks = sorted(c); am = min(c, key=c.get)
                g = lambda K: c[min(Ks, key=lambda x: abs(x - K))] / c[am] - 1
                o, n = int(pr["old"]), int(pr["new"])
                for nm, K in (("old", o), ("new", n)):
                    R[nm + "_ex"].append(am == K); R[nm + "_w1"].append(abs(am - K) <= 1); R[nm + "_rg"].append(g(K))
                if pr["br_lo"] not in ("", None): R["br"].append(int(float(pr["br_lo"])) <= am <= int(float(pr["br_hi"])))
                if o != n: paired.append(g(n) - g(o))
            if not R["old_ex"]: continue
            line = f"{pl:8s} {gname:31s} {fam:5s} n={len(R['old_ex']):3d}"
            for nm in ("old", "new"):
                x = np.array(R[nm + "_rg"])
                line += (f" | {nm}: ex {np.mean(R[nm+'_ex']):.2f} w1 {np.mean(R[nm+'_w1']):.2f} med {100*np.median(x):.1f}%"
                         f" p90 {100*np.quantile(x, .9):.1f}% (nr {100*nr90(x):.1f}%) max {100*x.max():.1f}%")
            if R["br"]: line += f" | bracket {sum(R['br'])}/{len(R['br'])}"
            if paired:
                b = sum(d < 0 for d in paired); w = sum(d > 0 for d in paired)
                line += f" | v3 != old: {len(paired)}, better {b} worse {w} (sign p {sign_p(b, w):.3f})"
            print(line)
    print()
