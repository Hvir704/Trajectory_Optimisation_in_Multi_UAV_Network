"""Independent check of the third held-out test: pandas only, no code from heldout3_v3.py.
usage: python verify_heldout3.py results_v2/heldout3_v3.csv heldout3_predictions.csv"""
import math, sys, numpy as np, pandas as pd
from math import comb
D = pd.read_csv(sys.argv[1]); P = pd.read_csv(sys.argv[2])
print("rows", len(D), "| versions", D.sim_version.unique(), "| planners", D.planner.unique(), "| seeds", sorted(D.seed.unique()))
key = ["planner","layout","M","Emax","seed","K"]
dup = D.duplicated(key).sum(); print("duplicate (instance,K) rows:", dup)
print("non-default settings:", {c: D[c].unique().tolist() for c in ["coord","replan","divert","q","belief","tau","L","Th"]})
print("non-finite J:", (~np.isfinite(D.J)).sum())
ikey = ["layout","M","Emax","seed"]
P = P.set_index(ikey)
out = []; issues = []
for (pl, lay, M, E, s), g in D.groupby(["planner"] + ikey):
    J = g.set_index("K").J.sort_index(); Ks = J.index.values
    if not (np.diff(Ks) == 1).all(): issues.append(("gap", pl, lay, M, E, s))
    am = J.idxmin(); pr = P.loc[(lay, M, E, s)]
    if am == Ks[-1] or (am == Ks[0] and Ks[0] > 1): issues.append(("edge", pl, lay, M, E, s, am))
    if not (J.loc[Ks[-3:]] > 1.5 * J.min()).all() and am >= Ks[-1]-1: issues.append(("tail", pl, lay, M, E, s))
    if Ks[0] > int(pr.K_lo) or Ks[-1] < int(pr.K_hi): issues.append(("short", pl, lay, M, E, s))
    rec = dict(pl=pl, fam=lay, am=am)
    for r in ["old","v33","v34"]:
        K = int(pr[r]); swept = K in J.index
        if not swept: issues.append(("unswept", pl, r, lay, M, E, s, K))
        Jk = J.loc[K] if swept else J.iloc[np.abs(Ks - K).argmin()]
        rec[r+"_K"] = K; rec[r+"_rg"] = Jk / J.min() - 1; rec[r+"_w1"] = abs(K - am) <= 1
    out.append(rec)
R = pd.DataFrame(out); print("instances per planner:", R.groupby("pl").size().to_dict())
print("issues:", len(issues), issues[:10])
def p90(x): x = np.sort(x); return x[math.ceil(.9*len(x)) - 1]
for pl in ["sa","rr_tour"]:
    A = R[R.pl == pl]; ring = A[A.fam == "ring"]
    a = (A.v34_w1.mean(), A.v33_w1.mean()); b = (ring.v34_w1.mean(), ring.v33_w1.mean()); c = (p90(A.v34_rg.values), p90(A.v33_rg.values))
    d = A[A.v34_K != A.v33_K]; bt = (d.v34_rg < d.v33_rg).sum(); wo = (d.v34_rg > d.v33_rg).sum()
    pv = sum(comb(bt+wo, i) for i in range(bt, bt+wo+1)) / 2**(bt+wo)
    print(f"{pl}: A {a[0]:.3f} vs {a[1]:.3f} {a[0]>a[1]} | B {b[0]:.3f} vs {b[1]:.3f} {b[0]>=.85 and b[0]>b[1]} | "
          f"C {c[0]:.4f} vs {c[1]:.4f} {c[0]<c[1]} | D n={len(d)} better {bt} worse {wo} p={pv:.5f} {bt>wo and pv<.05}")
    for fam in ["core","paper","ring"]:
        F = A[A.fam == fam]
        print(f"   {fam}: w1 old {F.old_w1.mean():.2f} v33 {F.v33_w1.mean():.2f} v34 {F.v34_w1.mean():.2f} | p90 v33 {p90(F.v33_rg.values):.3f} v34 {p90(F.v34_rg.values):.3f} | mean v34 {F.v34_rg.mean():.3f}")
    # where v34 differs: by cell; and what v34's worse cases look like
    d = d.assign(diff=d.v34_rg - d.v33_rg)
    print("   v34-v33 diffs by family:", d.groupby("fam").apply(lambda x: f"n={len(x)} better={(x['diff']<0).sum()} worse={(x['diff']>0).sum()} mean diff={x['diff'].mean():+.3f}").to_dict())
    print("   worse cases:\n", d[d['diff'] > 0][["fam","am","v33_K","v34_K","v33_rg","v34_rg"]].to_string())
    print("   signed error v34-am by family:", A.assign(e=A.v34_K - A.am).groupby("fam").e.describe()[["mean","min","max"]].to_dict("index"))
