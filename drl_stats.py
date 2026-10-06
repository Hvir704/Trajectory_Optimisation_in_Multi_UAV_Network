"""drl_stats.py -- registered DRL comparison (BASELINE_FREEZE_2026-10-05.md, section 4).

Inputs: drl_results.csv (drl_scope.py), the slotted-world comparator CSV (slot_gate.py --csv, slot 5 s), ell_eval.csv.
Only deployments whose DRL window (+ extensions) is complete are analysed. Own optima are taken on the K values
that BOTH planners have for that deployment. Primary: median J*_C4,slot / J*_DRL - 1, percentile bootstrap,
wins (C4 lower). Secondary: DRL vs every other comparator in the slotted world; DRL K* vs the rule's ell
(exact / within one; ell is the DynSim rule -- the slotted world costs 1-8% J, see the gate); comparators'
DynSim-world optima for reference.

usage: python drl_stats.py --drl ~/drl_results.csv --slot drl_comparators.csv --ell ell_eval.csv
"""
import argparse
import numpy as np
import pandas as pd

KEY = ["layout", "M", "Emax", "seed"]
SCOPE = {("paper", 1): (2, 7), ("ring", 1): (1, 5), ("paper", 2): (1, 6), ("ring", 2): (1, 6),
         ("paper", 3): (2, 7), ("ring", 3): (1, 5)}


def boot(r, B=10000, seed=0):
    r = np.asarray(r, float)
    if len(r) == 0: return (np.nan,) * 3
    m = np.median(np.random.default_rng(seed).choice(r, (B, len(r))), axis=1)
    return np.median(r), np.percentile(m, 2.5), np.percentile(m, 97.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drl", required=True); ap.add_argument("--slot", required=True); ap.add_argument("--ell", required=True)
    a = ap.parse_args()
    drl = pd.read_csv(a.drl).assign(planner="drl")
    drl = drl[drl.eval_world == 1]
    complete = []
    for k, g in drl.groupby(KEY):
        lo, hi = SCOPE[(k[0], int(k[3]))]
        if set(range(lo, hi + 1)) <= set(g.K): complete.append(k)
    print(f"complete DRL deployments: {len(complete)} of 6: {complete}")
    sl = pd.read_csv(a.slot)
    sl = sl[(sl.slot == 5.0) & (sl.rc == 200.0) & (sl.drain == "radius")].rename(columns={"J_slot": "J"})
    allp = pd.concat([drl[KEY + ["K", "planner", "J"]], sl[KEY + ["K", "planner", "J"]]], ignore_index=True)
    allp = allp[allp.set_index(KEY).index.isin(complete)]
    ell = pd.read_csv(a.ell).set_index(KEY).ell

    def opt(p, other):
        rows = []
        for k, g in allp.groupby(KEY):
            ks = set(g[g.planner == p].K) & set(g[g.planner == other].K)
            x = g[(g.planner == p) & g.K.isin(ks)]
            if len(x): rows.append(dict(zip(KEY, k), J=x.J.min(), K=int(x.loc[x.J.idxmin(), "K"])))
        return pd.DataFrame(rows).set_index(KEY)

    for other in ["cyc"] + sorted(set(sl.planner) - {"cyc"}):
        A, B = opt(other, "drl"), opt("drl", other)
        j = A.join(B, lsuffix="_o", rsuffix="_d", how="inner")
        if j.empty: continue
        r = j.J_o / j.J_d - 1
        tag = "PRIMARY " if other == "cyc" else "        "
        for fam, rr in list(r.groupby(level="layout")) + [("all", r)]:
            m, lo, hi = boot(rr)
            print(f"{tag}J*_{other} / J*_DRL - 1  [{fam:5s}] median {100*m:+6.1f}%  [{100*lo:+6.1f}, {100*hi:+6.1f}]  {other} lower in {int((rr < 0).sum())}/{len(rr)}")
    Bd = opt("drl", "drl").join(ell, how="left")
    ex = (Bd.K == Bd.ell).mean(); w1 = ((Bd.K - Bd.ell).abs() <= 1).mean()
    print(f"\nDRL own K* vs rule ell: exact {ex:.2f}, within one {w1:.2f}  (n = {len(Bd)})")
    print(Bd.assign(J=Bd.J.map("{:.4e}".format)).to_string())
