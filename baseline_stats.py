"""baseline_stats.py -- the registered held-out comparison for any pair of planners.

Own optimum J*_p(d) = min over K of J for planner p on deployment d (standard runs only: Th = 12 h, coord exclude,
launch replanning, no diversion, q = 0, default event lifetime, L = 12.6 km). An optimum is CENSORED unless the three
largest K values run for that deployment all have J > 1.5 x J*. Reported per family (layout) and overall:
  median of J*_a / J*_b - 1, 95% percentile bootstrap over deployments (10,000 resamples, seed 0), wins (a lower);
  same-K comparison at the rule's ell (ell_eval.csv); rule agreement of each planner's K* (exact / within one).

usage:
  python baseline_stats.py --a cyc --b cluster_patrol --csv eval_all_v2.csv eval_fill_v2.csv eval_cyc_v2.csv \
         cluster_win.csv fill_cluster.csv --ell ell_eval.csv --seeds 1-12 --Mmax 200
"""
import argparse
import numpy as np
import pandas as pd

KEY = ["layout", "M", "Emax", "seed"]


def load(paths, seeds, mmax):
    cols = ["layout", "M", "Emax", "K", "seed", "planner", "J", "Th", "coord", "replan", "divert", "q", "tau", "L"]
    d = pd.concat([pd.read_csv(p, usecols=lambda c: c in cols) for p in paths], ignore_index=True)
    std = (d.Th == 43200) & (d.coord == "exclude") & (d.replan.fillna("launch") == "launch")
    for c, v in (("divert", 0), ("q", 0.0), ("tau", 0.0)):
        if c in d: std &= d[c].fillna(v) == v
    if "L" in d: std &= d.L.fillna(12600.0).round() == 12600
    d = d[std & d.seed.isin(seeds) & (d.M <= mmax)]
    return d.drop_duplicates(KEY + ["K", "planner"], keep="last").reset_index(drop=True)


def own_optima(d, planner):
    x = d[d.planner == planner]
    rows = []
    for k, g in x.groupby(KEY):
        g = g.sort_values("K"); i = g.J.idxmin(); js = g.J.min()
        top3 = g.J.values[-3:]
        cens = not (len(g) >= 4 and (top3 > 1.5 * js).all() and g.K.values[-1] > g.loc[i, "K"])
        rows.append(dict(zip(KEY, k), J=js, K=int(g.loc[i, "K"]), censored=cens, nK=len(g)))
    return pd.DataFrame(rows).set_index(KEY)


def boot_median(r, B=10000, seed=0):
    rng = np.random.default_rng(seed); r = np.asarray(r)
    if len(r) == 0: return np.nan, np.nan, np.nan
    m = np.median(rng.choice(r, (B, len(r)), replace=True), axis=1)
    return float(np.median(r)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def parse_seeds(s):
    if "-" in s: a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--csv", nargs="+", required=True); ap.add_argument("--ell", default=None)
    ap.add_argument("--seeds", default="1-12"); ap.add_argument("--Mmax", type=int, default=200)
    ap.add_argument("--out", default=None, help="write per-deployment table here")
    a = ap.parse_args()
    d = load(a.csv, parse_seeds(a.seeds), a.Mmax)
    A, Bo = own_optima(d, a.a), own_optima(d, a.b)
    j = A.join(Bo, lsuffix="_a", rsuffix="_b", how="inner")
    j["r"] = j.J_a / j.J_b - 1
    print(f"deployments with both planners: {len(j)}   censored optima: {a.a} {int(j.censored_a.sum())}, {a.b} {int(j.censored_b.sum())}")
    print(f"\nown-optimum J*_{a.a} / J*_{a.b} - 1  (95% percentile bootstrap; wins = {a.a} lower)")
    for fam, g in list(j.groupby(level="layout")) + [("all", j)]:
        m, lo, hi = boot_median(g.r)
        print(f"  {fam:6s} median {100*m:+6.1f}%  [{100*lo:+6.1f}, {100*hi:+6.1f}]  wins {int((g.r < 0).sum())}/{len(g)}")
    if a.ell:
        ell = pd.read_csv(a.ell).set_index(KEY)["ell"]
        j = j.join(ell, how="left")
        print("\nrule agreement (exact / within one), own K* vs ell:")
        for lab, col in ((a.a, "K_a"), (a.b, "K_b")):
            ok = j.ell.notna()
            ex = (j.loc[ok, col] == j.loc[ok, "ell"]).mean(); w1 = ((j.loc[ok, col] - j.loc[ok, "ell"]).abs() <= 1).mean()
            fam = j[ok].assign(ex=(j[col] == j.ell), w1=(j[col] - j.ell).abs() <= 1).groupby(level="layout")[["ex", "w1"]].mean()
            print(f"  {lab:16s} {ex:.2f} / {w1:.2f}   " + "  ".join(f"{f} {r.ex:.2f}/{r.w1:.2f}" for f, r in fam.iterrows()))
        # same-K comparison at ell
        jk = d.set_index(KEY + ["K", "planner"]).J
        rows = []
        for idx, r in j[j.ell.notna()].iterrows():
            ka = (*idx, int(r.ell), a.a); kb = (*idx, int(r.ell), a.b)
            if ka in jk.index and kb in jk.index:
                rows.append(dict(layout=idx[0], r=jk[ka] / jk[kb] - 1))
        s = pd.DataFrame(rows)
        if len(s):
            print(f"\nsame K = ell: J_{a.a} / J_{a.b} - 1")
            for fam, g in list(s.groupby("layout")) + [("all", s)]:
                m, lo, hi = boot_median(g.r)
                print(f"  {fam:6s} median {100*m:+6.1f}%  [{100*lo:+6.1f}, {100*hi:+6.1f}]  wins {int((g.r < 0).sum())}/{len(g)}")
    if a.out:
        j.reset_index().to_csv(a.out, index=False); print("\nper-deployment table ->", a.out)
