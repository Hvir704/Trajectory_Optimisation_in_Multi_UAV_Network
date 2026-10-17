"""Independent check of test 4 (pandas only; no code from heldout4_v3.py or localcheck.py).
usage: python verify_heldout4.py results_v2/heldout4_v3.csv heldout4_predictions.csv"""
import sys, math, numpy as np, pandas as pd
from math import comb
D = pd.read_csv(sys.argv[1]); P = pd.read_csv(sys.argv[2])
print("rows", len(D), "versions", D.sim_version.unique().tolist(), "planners", sorted(D.planner.unique()), "seeds", sorted(D.seed.unique()))
key = ["planner","layout","M","Emax","seed","K"]; print("dups", D.duplicated(key).sum(), "nonfinite J", (~np.isfinite(D.J)).sum())
print("settings", {c: D[c].unique().tolist() for c in ["coord","replan","divert","q","belief","tau","L","Th"]})
P = P.set_index(["layout","M","Emax","seed"])
def cset(c1, c2):
    s = set()
    for c in (c1, c2):
        if pd.notna(c): s |= {int(k) for k in (c - 1, c, c + 1) if k >= 1}
    return sorted(s)
rows = []; issues = []
for (pl, lay, M, E, s), g in D.groupby(["planner","layout","M","Emax","seed"]):
    J = g.set_index("K").J.sort_index(); Ks = J.index.values; am = J.idxmin(); best = J.min(); pr = P.loc[(lay, M, E, s)]
    if not (np.diff(Ks) == 1).all(): issues.append(("gap", pl, lay, M, E, s))
    if am == Ks[-1] or (am == Ks[0] and Ks[0] > 1): issues.append(("edge", pl, lay, M, E, s))
    if Ks[0] > pr.K_lo or Ks[-1] < pr.K_hi: issues.append(("short", pl, lay, M, E, s))
    rule = "v34" if pl in ("sa", "rr_tour") else "v33"; c2 = pr.sh_c2 if rule == "v34" else pr.tr_c2
    cs = cset(int(pr[rule]), c2)
    if any(k not in J.index for k in cs): issues.append(("unswept", pl, lay, M, E, s))
    rec = dict(pl=pl, fam=lay, M=M, E=E, s=s, am=am, chk=min(J.loc[k] for k in cs) / best - 1, nchk=len(cs))
    for r in ("old", "v33", "v34"):
        K = int(pr[r]); rec[r + "_K"] = K; rec[r + "_rg"] = J.loc[K] / best - 1; rec[r + "_w1"] = abs(K - am) <= 1
    rows.append(rec)
R = pd.DataFrame(rows); print("instances", R.groupby("pl").size().to_dict(), "issues", len(issues), issues[:5])
def p90(x): x = np.sort(x); return x[math.ceil(.9 * len(x)) - 1]
thr = {"sa": (.85, 3, .15), "rr_tour": (.85, 3, .15), "cyc": (.80, 8, .25), "cluster_patrol": (.80, 8, .25)}
for pl, (ex, n10, mx) in thr.items():
    A = R[R.pl == pl]; c = A.chk.values
    print(f"{pl:15s} L1 {c.mean():.4%} {c.mean() < .01} | L2 {np.mean(c < 1e-12):.3f} {np.mean(c < 1e-12) >= ex} | L3 {(c > .1).sum()} {(c > .1).sum() <= n10} | L4 {c.max():.3%} {c.max() <= mx} | sizes {A.nchk.mean():.2f}")
for pl in ("sa", "rr_tour"):
    A = R[R.pl == pl]; G = A[A.fam == "ring"]
    print(f"{pl:15s} R1 {A.v34_w1.mean():.3f} | R2 {G.v34_w1.mean():.3f} | R3 v34 {p90(A.v34_rg.values):.4f} vs old {p90(A.old_rg.values):.4f}")
for pl in ("cyc", "cluster_patrol"):
    A = R[R.pl == pl]; print(f"{pl:15s} R4 v33 max {A.v33_rg.max():.4f} vs old {A.old_rg.max():.4f}")
