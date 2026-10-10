"""strand_planners.py -- Sec. VII-C: on the held-out deployments the design-time test flags, how often does each
independent planner's optimum lie within one UAV of the plain rule l, and of the stranding stopping point K_hat of
Prop. 6 (radial scale, corrected algebra, capped at round K_commute)? Both predictors use the annealer's smallest-K
(r_c, P_bar), so they are planner-independent.

usage: python strand_planners.py sa_rerun.csv verdicts.csv eval_all_v2.csv gzz_eval.csv asghar_eval.csv [liu_all.csv]
"""
import sys, csv, math, collections
import numpy as np
from dyn_env import DynParams, SensorField


def load(path, planner=None):
    d = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if planner and r.get("planner") != planner: continue
        if r["coord"] != "exclude" or r["replan"] != "launch" or float(r["Th"]) != 43200: continue
        if float(r.get("L", 12600) or 12600) != 12600: continue
        d[(r["layout"], int(float(r["M"])), float(r["Emax"]), int(float(r["seed"])))][int(float(r["K"]))] = r
    return d


def predictors(k, byK):
    lay, M, E, s = k
    p = DynParams(M=M, Emax=E, layout=lay); U = (1 - p.rho) * E; Tb, T = p.T_burnin, p.T_horizon - p.T_burnin
    F = SensorField(DynParams(M=M, Emax=E, layout=lay), np.random.default_rng(s))
    r = np.linalg.norm(F.pos - p.home, axis=1); w = F.wi_base; c = 2 * p.Pf * r / p.v
    r0 = byK[min(byK)]; rc = float(r0["r_c"]); Pb = float(r0["P_bar"]); tc = 2 * rc / p.v
    Phi = lambda K: (K * (U - K * p.Pf * tc) / (U + K * tc * (Pb - p.Pf))) if (U + K * tc * (Pb - p.Pf)) > 0 else -1
    reach = lambda K: p.v * (U / K - p.Ph * p.B_bits / p.R) / (2 * p.Pf)
    Kc = U / (tc * (p.Pf + (p.Pf * Pb) ** .5)); Kr = p.v * U / (2 * p.Pf * r.max() + p.v * p.Ph * p.B_bits / p.R)
    ell = math.floor(Kr) if math.floor(Kr) <= Kc else int(round(Kc))
    Kp = 1
    for K in range(1, 500):
        if Phi(K + 1) <= 0: break
        S = r <= reach(K); Sn = r <= reach(K + 1)
        QK = np.sqrt(w[S] * c[S]).sum(); QK1 = np.sqrt(w[Sn] * c[Sn]).sum()
        if QK ** 2 / (2 * Pb * Phi(K)) - QK1 ** 2 / (2 * Pb * Phi(K + 1)) > (Tb + T / 2) * w[S & ~Sn].sum(): Kp = K + 1
        else: break
    return ell, min(Kp, round(Kc))


sa = load(sys.argv[1])
flag = {(v["layout"], int(float(v["M"])), float(v["Emax"]), int(float(v["seed"]))) for v in csv.DictReader(open(sys.argv[2]))
        if v["verdict_current"] == "flag"}
pred = {k: predictors(k, sa[k]) for k in flag if k in sa}
print(f"held-out deployments the design-time test flags: {len(flag)} (with SA rows: {len(pred)})")
for path, pl, name in [(sys.argv[3], "cyc", "C4"), (sys.argv[3], "cluster_patrol", "cluster patrol"),
                       (sys.argv[4], "gzz_rev", "Gao et al."), (sys.argv[5], "asghar_rev", "Asghar et al.")] + \
                      ([(sys.argv[6], "liu_mpga", "Liu et al.")] if len(sys.argv) > 6 else []):
    d = load(path, pl); ks = [k for k in pred if k in d]
    if not ks: print(f"  {name:15s} no rows"); continue
    am = {k: min(d[k], key=lambda K: float(d[k][K]["J"])) for k in ks}
    w_l = np.mean([abs(am[k] - pred[k][0]) <= 1 for k in ks]); w_s = np.mean([abs(am[k] - pred[k][1]) <= 1 for k in ks])
    print(f"  {name:15s} n={len(ks):3d}  within one of l {w_l:.0%}   within one of K_hat (Prop. 6) {w_s:.0%}")

# ---- warning indicator (r_max/r_c above a level) vs the design-time test, as warnings against each planner's miss
# (optimum more than one UAV from l), pooled over C4, the cluster patrol, Gao et al. and Asghar et al., held-out
# deployments; l and r_max/r_c from the annealer's smallest-K row. The level is not fixed by the grid, so three are shown.
verd = {(v["layout"], int(float(v["M"])), float(v["Emax"]), int(float(v["seed"]))): v["verdict_current"] for v in csv.DictReader(open(sys.argv[2]))}
rows = []
for path, pl in [(sys.argv[3], "cyc"), (sys.argv[3], "cluster_patrol"), (sys.argv[4], "gzz_rev"), (sys.argv[5], "asghar_rev")]:
    d = load(path, pl)
    for k, v in verd.items():
        if k not in d or k not in sa: continue
        r0 = sa[k][min(sa[k])]; kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"])
        ell = kr if kr <= kc else int(round(kc)); am = min(d[k], key=lambda K: float(d[k][K]["J"]))
        rows.append((abs(am - ell) > 1, float(r0["rmax_over_rc"]), v == "flag"))
if rows:
    miss = np.array([x[0] for x in rows]); ratio = np.array([x[1] for x in rows]); test = np.array([x[2] for x in rows])
    rec = lambda f: (miss & f).sum() / max(miss.sum(), 1); pre = lambda f: (miss & f).sum() / max(f.sum(), 1)
    print(f"\nwarnings against the four planners' misses ({miss.sum()} misses in {len(rows)} planner-deployments):")
    print(f"  design-time test          recall {rec(test):.2f}  precision {pre(test):.2f}")
    for lev in (6.2, 6.8, 7.4):
        f = ratio > lev
        print(f"  r_max/r_c > {lev:<4}          recall {rec(f):.2f}  precision {pre(f):.2f}")
