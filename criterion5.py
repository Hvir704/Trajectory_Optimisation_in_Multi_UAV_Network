"""criterion5.py -- design-time criterion (5), frozen vs corrected served-set algebra.
usage: python criterion5.py grid_final.csv ../paper/apriori_regret.csv [--variant ...]
"""
import sys, csv, math, numpy as np
from collections import defaultdict
from dyn_env import DynParams, SensorField

def rows_by_inst(path):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if float(r["L"]) == 12600: d[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]))][int(r["K"])] = r
    return d

def evaluate(inst, byK, corrected, KR_mode="int"):
    lay, M, E, seed = inst
    p = DynParams(M=M, Emax=E, layout=lay); U = (1 - p.rho) * E
    F = SensorField(DynParams(M=M, Emax=E, layout=lay), np.random.default_rng(seed))
    r = np.linalg.norm(F.pos - p.home, axis=1); w = F.wi_base; q = r / r.mean()
    r0 = byK[min(byK)]
    rc = float(r0["r_c"]); Pb = float(r0["P_bar"]); tc = 2 * rc / p.v
    Ts = float(r0["T_s_over_t_c"]) * tc; n = float(r0["mean_n"]); tau = (Ts - tc) / max(n, 1e-9)
    Rstar = lambda K: (K * (U - K * p.Pf * tc) / (U + K * tc * (Pb - p.Pf))) / tau
    reach = lambda K: p.v * (U / K - p.Ph * p.B_bits / p.R) / (2 * p.Pf)
    K0 = math.floor(float(r0["K_reach_i"]))
    Kc = U / (tc * (p.Pf + (p.Pf * Pb) ** 0.5))
    KR = round(Kc) if KR_mode == "int" else Kc
    if KR <= K0: return None                       # commute-bound: criterion not applied
    H = p.T_burnin + (p.T_horizon - p.T_burnin) / 2
    S0 = r <= reach(K0); A = S0 & ~(r <= reach(K0 + 1))
    AB = lambda S: np.sqrt(w[S] * q[S]).sum() * np.sqrt(w[S] / q[S]).sum()
    lhs = H * w[A].sum()
    if corrected:
        SR = r <= reach(KR) if reach(KR) > 0 else np.zeros_like(S0)
        rhs = AB(S0) / (2 * Rstar(K0)) - AB(SR) / (2 * Rstar(KR))
    else:
        rhs = AB(S0) / 2 * (1 / Rstar(K0) - 1 / Rstar(KR))
    return lhs / rhs if rhs > 0 else float("inf")

if __name__ == "__main__":
    d = rows_by_inst(sys.argv[1])
    banked = {(x["lay"], int(x["M"]), float(x["E"]), int(x["seed"])): x for x in csv.DictReader(open(sys.argv[2]))}
    for mode in ("int", "cont"):
        errs = []; agree = 0; n = 0
        for k, b in banked.items():
            if b["reach_bound"] != "True": continue
            v = evaluate(k, d[k], corrected=False, KR_mode=mode)
            if v is None: continue
            n += 1; old = float(b["ratio"]); errs.append(abs(v - old) / old); agree += (v > 1) == (b["passes"] == "True")
        print(f"reproduction (frozen algebra, K_R {mode}): n={n} verdict match {agree}/{n}  median rel. diff in ratio {np.median(errs):.2e}  max {max(errs):.2e}")