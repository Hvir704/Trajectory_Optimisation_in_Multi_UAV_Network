"""singlecell_v2.py -- the single-cell tables of Sec. VI (paper layout, M=100, 1.5 MJ unless stated).

  coord    Table VII (tab:coord): mean J (x1e5) per K for coordinated launch-time SA, uncoordinated, the
           uncoordinated duplicate-visit fraction and coordinated per-leg SA; mean argmin of each; J ratio
           uncoordinated/coordinated at K=4 and at each variant's own optimum.
  perleg   per-leg vs launch-time: truncation, J change per K, optimum changes.
  horizon  Table VI (tab:horizon): J(24 h)/J(12 h) per K, paper and ring, seeds 1-6.
  info     Table XIV (tab:info): kernel (M1) and prior-rates vs event-blind (M2), paired, with sign tests;
           K* agreement under q=1 between kernel and event-blind per-leg planners.
usage: python singlecell_v2.py DIR    (DIR = folder with the results_v2 files, names as in RERUN_V2.md)
"""
import sys, os, csv, math
from collections import defaultdict
import numpy as np

DIR = sys.argv[1]
f = lambda name: os.path.join(DIR, name)

def rows(name, **flt):
    if not os.path.exists(f(name)):
        print(f"   (missing {name})"); return []
    out = []
    for r in csv.DictReader(open(f(name))):
        if all(str(r.get(k)) == str(v) for k, v in flt.items()): out.append(r)
    return out

def byKseed(rs, layout="paper", M=100, E=1.5e6, Th=43200.0):
    d = defaultdict(dict)
    for r in rs:
        if r["layout"] == layout and int(float(r["M"])) == M and float(r["Emax"]) == E and float(r["Th"]) == Th:
            d[int(float(r["K"]))][int(float(r["seed"]))] = r
    return d

def sign(x):
    nz = [v for v in x if v != 0.0]; n = len(nz); s = sum(1 for v in nz if v > 0)
    if n == 0: return 0, 0, 1.0
    return s, n, min(1.0, 2 * sum(math.comb(n, k) for k in range(max(s, n - s), n + 1)) / 2 ** n)

def argmins(d, seeds):
    out = []
    for s in seeds:
        J = {K: float(d[K][s]["J"]) for K in d if s in d[K]}
        if J: out.append(min(J, key=J.get))
    return out

KS = range(2, 9); SEEDS = range(1, 13)
SA_STD = lambda r: (r.get("planner") == "sa" and r["coord"] == "exclude" and r["replan"] == "launch" and str(r.get("divert", "0")) in ("0", "0.0")
                    and r.get("belief", "prior") == "prior" and float(r.get("q", 0)) == 0 and float(r.get("tau", 0)) == 0)
SA_ROWS = [r for r in rows("sa_grid.csv") + rows("sa_fill.csv") if SA_STD(r)]     # grid AND its fills
base = byKseed(SA_ROWS)
none = byKseed(rows("coord_none.csv"))
perl = byKseed(rows("coord_perleg.csv"))

print("== coord (tab:coord): mean J x1e5 over seeds 1-12")
for lab, d, col in (("coordinated, launch-time SA", base, "J"), ("uncoordinated", none, "J"),
                    ("duplicate-visit fraction", none, "dup_frac"), ("coordinated, per-leg SA", perl, "J")):
    vals = []
    for K in KS:
        v = [float(d[K][s][col]) for s in SEEDS if K in d and s in d[K]]
        vals.append(((np.mean(v) / (1e5 if col == "J" else 1)) if v else float("nan"), len(v)))
    print(f"   {lab:30s} " + " ".join(f"{x:6.2f}" for x, _ in vals) + "   n per K: " + " ".join(str(n) for _, n in vals))
for lab, d in (("coordinated", base), ("uncoordinated", none), ("per-leg", perl)):
    am = argmins({K: d[K] for K in KS if K in d}, SEEDS)
    if am: print(f"   mean argmin over K=2..8, {lab}: {np.mean(am):.2f} (n={len(am)})")
common = [s for s in SEEDS if all(K in base and s in base[K] and K in none and s in none[K] for K in KS)]
print(f"   seeds with every K in both coordinated and uncoordinated rows: {len(common)} of {len(SEEDS)} (ratios below use these)")
if common:
    mean = lambda d, K: np.mean([float(d[K][s]["J"]) for s in common])
    r4 = mean(none, 4) / mean(base, 4)
    mb = min(mean(base, K) for K in KS); mn = min(mean(none, K) for K in KS)
    print(f"   J ratio uncoordinated/coordinated: at K=4 {r4:.2f}x; at each variant's own optimum (cell means) {mn/mb:.2f}x")

print("\n== perleg: per-leg vs launch-time SA, same seeds")
for K in KS:
    if K in base and K in perl:
        ss = [s for s in SEEDS if s in base[K] and s in perl[K]]
        dj = np.mean([float(perl[K][s]["J"]) for s in ss]) / np.mean([float(base[K][s]["J"]) for s in ss]) - 1
        tb = np.mean([float(base[K][s]["trunc_frac"]) for s in ss]); tp = np.mean([float(perl[K][s]["trunc_frac"]) for s in ss])
        print(f"   K={K}: J {dj:+.1%}  trunc launch {tb:.1%} -> per-leg {tp:.1%}")
ab, ap_ = argmins({K: base[K] for K in KS if K in base}, SEEDS), argmins({K: perl[K] for K in KS if K in perl}, SEEDS)
if ab and ap_: print(f"   optimum unchanged in {sum(1 for x, y in zip(ab, ap_) if x == y)} of {len(ab)} instances")

print("\n== horizon (tab:horizon): J(24 h)/J(12 h), mean over seeds 1-6 of the per-seed ratio (n seeds in brackets)")
h24 = rows("horizon24.csv")
for lay in ("paper", "ring"):
    d12 = byKseed(SA_ROWS, layout=lay)
    d24 = byKseed(h24, layout=lay, Th=86400.0)
    out = []
    for K in KS:
        rr = [float(d24[K][s]["J"]) / float(d12[K][s]["J"]) for s in range(1, 7) if K in d24 and K in d12 and s in d24[K] and s in d12[K]]
        out.append(f"{np.mean(rr):.2f}({len(rr)})" if rr else "--")
    print(f"   {lay:6s} " + " ".join(f"{x:>5s}" for x in out))

print("\n== info (tab:info): vs event-blind per-leg (M2), paired over seeds 1-12, K=4")
def info(b, tag):
    return {int(float(r["seed"])): r for r in rows(f"info_{b}_{tag}.csv") if int(float(r["K"])) == 4}
for tag, lab in (("q0", "67 min, q=0"), ("q0.5", "67 min, q=0.5"), ("q1", "67 min, q=1"), ("tau15", "15 min, q=1"), ("tau5", "5 min, q=1")):
    ref = info("prior", tag)
    for b in ("kernel", "prior_rates"):
        x = info(b, tag); ss = sorted(set(x) & set(ref))
        if not ss: continue
        dJ = [float(x[s]["J"]) - float(ref[s]["J"]) for s in ss]
        rel = lambda c: np.mean([float(x[s][c]) for s in ss]) / np.mean([float(ref[s][c]) for s in ss]) - 1
        s_, n, p = sign(dJ)
        evshare = np.mean([float(ref[s]["J_event"]) / float(ref[s]["J"]) for s in ss])
        print(f"   {lab:14s} {b:11s} dJ {rel('J'):+.1%}  dJ_age {rel('J_age'):+.1%}  dJ_event {rel('J_event'):+.1%}  "
              f"dcatch {rel('catch_rate'):+.1%}  worse {s_}/{n} p={p:.3f}  (event share of M2's J {evshare:.0%})")
kq = byKseed(rows("info_kernel_q1_K.csv")); pq = byKseed(rows("info_prior_q1_K.csv"))
if kq and pq:
    a1, a2 = argmins(kq, SEEDS), argmins(pq, SEEDS)
    print(f"   K* agreement kernel vs event-blind at q=1, K=2..8: {sum(1 for x, y in zip(a1, a2) if x == y)}/{len(a1)}")
