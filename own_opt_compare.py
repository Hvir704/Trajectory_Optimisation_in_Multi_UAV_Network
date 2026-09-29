"""own_opt_compare.py -- planner B vs planner A at each planner's OWN optimum K, per deployment.

usage: python own_opt_compare.py A.csv B.csv --pa cluster_patrol --pb cyc [--out kstar.csv]
                                  [--ell ell.csv]   # secondary: common fleet size K = rule's l
       ell.csv columns: layout,M,Emax,seed,ell (the rule's point prediction per deployment, e.g.
       exported from reproduce.py). Deployments where either planner lacks a row at K=ell are
       listed, not silently dropped.
       (A.csv and B.csv may be the same file; rows are filtered by the planner column.)
Deployment = (layout, M, Emax, seed[, L]). Reports per family: median of J_B*/J_A* - 1 with a 95%
percentile bootstrap over deployments, B wins / n, and deployments whose optimum sits at an END of
the swept K range (censored: the true optimum may lie outside). Writes each planner's K* per
deployment to --out so the existing rule-agreement analysis can consume it.
"""
import argparse
import numpy as np
import pandas as pd

KEY = ["layout", "M", "Emax", "seed"]


def own_opt(d):
    rows = []
    for key, x in d.groupby(KEY):
        x = x.sort_values("K")
        i = x.J.idxmin()
        rows.append(dict(zip(KEY, key), K=int(x.loc[i, "K"]), J=float(x.loc[i, "J"]),
                         Kmin=int(x.K.min()), Kmax=int(x.K.max()), nK=len(x),
                         # K=1 is the physical floor: an optimum there is not censored
                         censored=bool(len(x) > 1 and (x.loc[i, "K"] == x.K.max() or
                                                       (x.loc[i, "K"] == x.K.min() and x.K.min() > 1)))))
    return pd.DataFrame(rows)


def boot_median(v, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v)
    if len(v) < 2:
        return (np.nan, np.nan)
    m = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--pa", default="cluster_patrol"); ap.add_argument("--pb", default="cyc")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ell", default=None)
    a = ap.parse_args()
    A = pd.read_csv(a.a); A = A[A.planner == a.pa]
    B = pd.read_csv(a.b); B = B[B.planner == a.pb]
    oa, ob = own_opt(A), own_opt(B)
    m = oa.merge(ob, on=KEY, suffixes=("_a", "_b"))
    missing = len(oa) + len(ob) - 2 * len(m)
    m["rel"] = m.J_b / m.J_a - 1
    print(f"{a.pb} vs {a.pa}: {len(m)} paired deployments ({missing} unpaired dropped)")
    print(f"{'family':8s} {'n':>4s} {'median J_b/J_a-1':>17s} {'95% CI':>20s} {'b wins':>8s} {'cens a/b':>9s}")
    for fam, x in list(m.groupby("layout")) + [("all", m)]:
        lo, hi = boot_median(x.rel)
        print(f"{fam:8s} {len(x):4d} {np.median(x.rel):+17.2%} [{lo:+.2%}, {hi:+.2%}] "
              f"{int((x.rel < 0).sum()):4d}/{len(x):<3d} {int(x.censored_a.sum()):4d}/{int(x.censored_b.sum()):<4d}")
    print("\nper deployment:")
    print(m[KEY + ["K_a", "J_a", "K_b", "J_b", "rel", "censored_a", "censored_b"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if a.ell:
        L = pd.read_csv(a.ell)[KEY + ["ell"]]
        rows, miss = [], []
        for _, r in L.iterrows():
            sel = lambda D: D[(D.layout == r.layout) & (D.M == r.M) & (D.Emax == r.Emax) & (D.seed == r.seed) & (D.K == r.ell)]
            xa, xb = sel(A), sel(B)
            if len(xa) and len(xb):
                rows.append(dict(layout=r.layout, rel=float(xb.J.iloc[0] / xa.J.iloc[0] - 1)))
            elif ((m.layout == r.layout) & (m.M == r.M) & (m.Emax == r.Emax) & (m.seed == r.seed)).any():
                miss.append((r.layout, r.M, r.Emax, r.seed, int(r.ell)))
        c = pd.DataFrame(rows)
        print(f"\nSECONDARY -- common fleet size K = l: {len(c)} deployments, {len(miss)} without rows at K=l")
        for fam, x in list(c.groupby("layout")) + [("all", c)]:
            lo, hi = boot_median(x.rel)
            print(f"{fam:8s} {len(x):4d} {np.median(x.rel):+17.2%} [{lo:+.2%}, {hi:+.2%}] {int((x.rel < 0).sum()):4d}/{len(x)}")
        if miss: print("  missing K=l rows:", miss[:10], "..." if len(miss) > 10 else "")
        # K* agreement with the rule (reported quantity; the rule is ex-ante, planner-independent)
        g = m.merge(L, on=KEY, how="inner")
        print(f"\nK* AGREEMENT with the rule l ({len(g)} deployments): exact / within one")
        rng = np.random.default_rng(0)
        for fam, x in list(g.groupby("layout")) + [("all", g)]:
            out = []
            for s_ in ("a", "b"):
                w1 = (abs(x[f"K_{s_}"] - x.ell) <= 1).to_numpy(float)
                ex = (x[f"K_{s_}"] == x.ell).to_numpy(float)
                out.append((ex.mean(), w1.mean()))
            dw = (abs(x.K_b - x.ell) <= 1).to_numpy(float) - (abs(x.K_a - x.ell) <= 1).to_numpy(float)
            bs = [dw[rng.integers(0, len(dw), len(dw))].mean() for _ in range(5000)] if len(dw) > 1 else [np.nan]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"{fam:8s} {len(x):4d}  {a.pa}: {out[0][0]:.2f} / {out[0][1]:.2f}   {a.pb}: {out[1][0]:.2f} / {out[1][1]:.2f}"
                  f"   within-one diff b-a {dw.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    if a.out:
        m[KEY + ["K_a", "K_b", "censored_a", "censored_b"]].to_csv(a.out, index=False)
