"""cluster_compare.py -- the published cluster patrol (Rahimi & Shafieinejad 2024) against SA, paired.
usage: python cluster_compare.py sa_rerun.csv cluster_win.csv
Pairs by (layout, M, Emax, seed) on the common K range. Reports, overall and by family: K* agreement
with SA and with the rule; J at each planner's own optimum and at matched K; stratified-bootstrap
intervals (instances resampled within cells, 10,000 replicates) and the paired rule-agreement difference."""
import sys, csv, math
from collections import defaultdict
import numpy as np
rng = np.random.default_rng(20260924); B = 10_000
def load(f, sa=False):
    d = defaultdict(dict)
    for r in csv.DictReader(open(f)):
        if sa and not (float(r["Th"]) == 43200 and r["coord"] == "exclude" and r["replan"] == "launch" and r.get("planner", "sa") == "sa"): continue
        d[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]))][int(r["K"])] = r
    return d
def rule(b):
    r0 = b[min(b)]; kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"]); law = min(kr, kc)
    return int(law) if law == kr else int(round(law))
S, C = load(sys.argv[1], True), load(sys.argv[2])
rows = []
for k in sorted(set(S) & set(C)):
    Ks = sorted(set(S[k]) & set(C[k])); js = {K: float(S[k][K]["J"]) for K in Ks}; jc = {K: float(C[k][K]["J"]) for K in Ks}
    a, c = min(js, key=js.get), min(jc, key=jc.get)
    rows.append(dict(cell=k[:3], fam=k[0], a=a, c=c, lc=rule(C[k]), ls=rule(S[k]), own=jc[c] / js[a] - 1, same=[jc[K] / js[K] - 1 for K in Ks]))
def summary(R, lab):
    g = lambda f: np.array([f(x) for x in R], float)
    own = g(lambda x: x["own"]); same = np.concatenate([x["same"] for x in R])
    print(f"{lab:6s} n={len(R):3d} | cluster~SA exact {np.mean(g(lambda x:x['c']==x['a'])):.2f} w1 {np.mean(g(lambda x:abs(x['c']-x['a'])<=1)):.2f} "
          f"mean diff {np.mean(g(lambda x:x['c']-x['a'])):+.2f} | cluster~rule exact {np.mean(g(lambda x:x['c']==x['lc'])):.2f} "
          f"w1 {np.mean(g(lambda x:abs(x['c']-x['lc'])<=1)):.2f} | J own-optimum cluster/SA-1 median {np.median(own):+.1%} "
          f"(SA lower in {int(np.sum(own>0))}/{len(own)}) | same-K median {np.median(same):+.1%} over {len(same)} runs")
summary(rows, "all")
for f in ("paper", "ring", "core"): summary([x for x in rows if x["fam"] == f], f)
cells = defaultdict(list)
for x in rows:
    cells[x["cell"]].append((abs(x["c"] - x["lc"]) <= 1, abs(x["c"] - x["a"]) <= 1, (abs(x["c"] - x["lc"]) <= 1) - (abs(x["a"] - x["ls"]) <= 1)))
G = [np.array(v, float) for v in cells.values()]; A = np.vstack(G)
for j, lab in ((0, "cluster within one of rule"), (1, "cluster within one of SA"), (2, "paired: cluster~rule minus SA~rule")):
    reps = [np.vstack([g[rng.integers(0, len(g), len(g))] for g in G])[:, j].mean() for _ in range(B)]
    print(f"{lab:38s} {A[:, j].mean():+.3f}  95% [{np.quantile(reps, .025):+.3f}, {np.quantile(reps, .975):+.3f}]")
R = [x for x in rows if x["cell"] == ("paper", 100, 1.5e6)]
for K in sorted(set.intersection(*[set(S[k]) & set(C[k]) for k in S if k[:3] == ("paper", 100, 1.5e6) and k in C])):
    ms = np.mean([float(S[k][K]["J"]) for k in S if k[:3] == ("paper", 100, 1.5e6) and k in C])
    mc = np.mean([float(C[k][K]["J"]) for k in C if k[:3] == ("paper", 100, 1.5e6)])
    print(f"  paper M=100 1.5MJ K={K}: SA mean J {ms:.3e}, cluster {mc:.3e}, SA lower by {1 - ms / mc:+.1%}")
