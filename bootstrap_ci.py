"""bootstrap_ci.py -- uncertainty on the paper's headline agreement numbers.

Design: cells (layout, M, Emax, L) are fixed by the experiment; instances are independent seeds
within a cell. Aggregate intervals therefore use a STRATIFIED bootstrap -- instances resampled with
replacement within each cell, cell sizes preserved -- 10,000 replicates, 95% percentile intervals.
Proportions at 0 or 1 (e.g. 24/24) give a degenerate bootstrap, so those use the exact Clopper-Pearson
interval. Per-cell intervals (n=12) are Clopper-Pearson and are written to a CSV, not the paper.

usage: python bootstrap_ci.py grid_final.csv sa_rerun.csv rr_tour_win.csv rr_sweep_win.csv \
                              m40_sa.csv m40_rr.csv m40_cp_final.csv  [--out per_cell_ci.csv]
"""
import sys, csv, math
from collections import defaultdict
import numpy as np
from scipy.stats import beta

B = 10_000
rng = np.random.default_rng(20260924)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
out_csv = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)


def load(path):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if float(r["Th"]) != 43200 or r["coord"] != "exclude" or r.get("replan", "launch") != "launch": continue
        d[(r["layout"], int(r["M"]), float(r["Emax"]), float(r.get("L", 12600)), int(r["seed"]))][int(r["K"])] = r
    return d


def rule(byK):
    r0 = byK[min(byK)]; kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"])
    law = min(kr, kc); return int(law) if law == kr else int(round(law))


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def strat_boot(recs, stat):
    """recs: list of (cell, value-tuple); stat: function(list of value-tuples) -> float."""
    cells = defaultdict(list)
    for c, v in recs: cells[c].append(v)
    groups = [np.array(v, dtype=float) for v in cells.values()]
    point = stat(np.vstack(groups))
    reps = np.empty(B)
    for b in range(B):
        reps[b] = stat(np.vstack([g[rng.integers(0, len(g), len(g))] for g in groups]))
    return point, np.quantile(reps, 0.025), np.quantile(reps, 0.975)


def report(label, recs, col):
    n = len(recs); vals = np.array([v[col] for _, v in recs]); k = int(vals.sum())
    if k in (0, n):
        lo, hi = cp(k, n); how = "CP"
        print(f"  {label:44s} {k/n:.3f}  [{lo:.3f}, {hi:.3f}]  n={n} ({how}, all-or-none)")
        return
    p, lo, hi = strat_boot(recs, lambda a: a[:, col].mean())
    print(f"  {label:44s} {p:.3f}  [{lo:.3f}, {hi:.3f}]  n={n}")


# ---- 1. the rule against SA's measured optimum, primary grid and subsets ----
g = load(args[0])
rec = []
for k, byK in g.items():
    am = min(byK, key=lambda K: float(byK[K]["J"])); l = rule(byK)
    rec.append((k[:4], (am == l, abs(am - l) <= 1)))
prim = [r for r in rec if r[0][3] == 12600]
print("Rule vs measured optimum (stratified bootstrap by cell, 95%)")
report("primary 240: exact", prim, 0); report("primary 240: within one", prim, 1)
for fam in ("paper", "ring", "core"):
    sub = [r for r in prim if r[0][0] == fam]
    report(f"{fam}: exact", sub, 0); report(f"{fam}: within one", sub, 1)
bad = {("ring", 50, 3e6), ("ring", 100, 3e6), ("core", 100, 1.5e6), ("core", 200, 1.5e6), ("core", 200, 3e6)}
good = [r for r in prim if r[0][:3] not in bad]
report("15 cells outside the failure families: exact", good, 0)
report("15 cells outside the failure families: within one", good, 1)
report("all 272: exact", rec, 0); report("all 272: within one", rec, 1)

# per-cell Clopper-Pearson
if out_csv:
    cells = defaultdict(list)
    for c, v in rec: cells[c].append(v)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layout", "M", "Emax", "L", "n", "exact", "exact_lo", "exact_hi", "within1", "w1_lo", "w1_hi"])
        for c, vs in sorted(cells.items()):
            n = len(vs); e = sum(v[0] for v in vs); o = sum(v[1] for v in vs)
            w.writerow([*c, n, e / n, *cp(e, n), o / n, *cp(o, n)])
    print(f"  per-cell Clopper-Pearson intervals -> {out_csv}")

# ---- 2. independent planners (paired, primary grid M<=200) ----
def paired(sa_path, b_path, fams=None):
    sa, bb = load(sa_path), load(b_path); out = []
    for k in sorted(set(sa) & set(bb)):
        if fams and k[0] not in fams: continue
        Ks = sorted(set(sa[k]) & set(bb[k]))
        if len(Ks) < 3: continue
        am_s = min(Ks, key=lambda K: float(sa[k][K]["J"])); am_b = min(Ks, key=lambda K: float(bb[k][K]["J"]))
        l = rule(bb[k])
        out.append((k[:4], (am_b == am_s, abs(am_b - am_s) <= 1, am_b == l, abs(am_b - l) <= 1)))
    return out
print("\nIndependent planners (stratified bootstrap by cell, 95%)")
t = paired(args[1], args[2])
report("tour patrol = SA", t, 0); report("tour patrol within one of SA", t, 1)
report("tour patrol = rule", t, 2); report("tour patrol within one of rule", t, 3)
s = paired(args[1], args[3], fams={"paper", "ring"})
report("sweep patrol within one of SA (paper+ring)", s, 1)
report("sweep patrol within one of rule (paper+ring)", s, 3)
sa_self = paired(args[1], args[1])
report("SA = rule (same 216)", sa_self, 2); report("SA within one of rule (same 216)", sa_self, 3)

# ---- 3. M=40 controlled comparison ----
print("\nM=40, 24 instances")
m_sa = paired(args[4], args[4]); m_rr = paired(args[4], args[5]); m_cp = paired(args[4], args[6])
report("SA within one of rule", m_sa, 3)
report("tour patrol within one of SA", m_rr, 1); report("tour patrol within one of rule", m_rr, 3)
report("CP-SAT within one of SA", m_cp, 1); report("CP-SAT within one of rule", m_cp, 3)
report("CP-SAT = SA", m_cp, 0)
