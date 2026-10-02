"""analyze_mass.py -- battery mass: does the optimum move, and does the fixed-point rule follow it?
Mass model (run_mass.py defaults): m(K) = 2.0 + (U/K)/(180 Wh/kg); Pf = 300 (m/2.5)^1.0, Ph = 400 (m/2.5)^1.5.
Rules: constant-power (the paper's: K_reach, K_commute with Pf=300, Ph=400, Pbar measured) vs
mass-aware: reach cap = fixed point K = vU/(2Pf(K) rmax + v Ph(K) B/Ru); commute cap = integer argmax of
Gamma(K) = K(U - K Pf(K) t_c)/(U + K t_c (Pbar(K) - Pf(K))) with Pbar(K) = Pbar_meas * Pf(K)/Pf(K_min).
Commute inputs (r_c, Pbar) from each instance's smallest-K row, as the paper's rule does.
usage: python analyze_mass.py mass.csv sa_rerun.csv"""
import sys, csv, math
from collections import defaultdict
import numpy as np
from dyn_env import DynParams
p = DynParams()
mdry, e_b, mref, gam = 2.0, 180 * 3600.0, 2.5, 1.0
def load(fn):
    d = defaultdict(dict)
    for r in csv.DictReader(open(fn)):
        if float(r["Th"]) == 43200 and r["coord"] == "exclude" and float(r.get("L", 12600)) == 12600:
            d[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]))][int(r["K"])] = r
    return d
mass, const = load(sys.argv[1]), load(sys.argv[2])
rows = []; unconverged = 0
for k, b in sorted(mass.items()):
    Ks = sorted(b); J = {K: float(b[K]["J"]) for K in Ks}; am = min(J, key=J.get)
    if am == Ks[-1] or J[Ks[-1]] < J[Ks[-2]]: unconverged += 1
    U = (1 - p.rho) * k[2]; r0 = b[Ks[0]]; rmax = float(r0["r_max"]); rc = float(r0["r_c"]); Pb0 = float(r0["P_bar"])
    m = lambda K: mdry + (U / K) / e_b
    Pf = lambda K: 300.0 * (m(K) / mref) ** gam; Ph = lambda K: 400.0 * (m(K) / mref) ** 1.5
    # constant-power rule (as in the paper)
    Kr_c = p.v * U / (2 * 300.0 * rmax + p.v * 400.0 * p.B_bits / p.R)
    tc = 2 * rc / p.v
    Kc_c = U / (tc * (300.0 + math.sqrt(300.0 * (Pb0 * 300.0 / Pf(Ks[0])))))
    law = min(math.floor(Kr_c), Kc_c); l_c = int(law) if law == math.floor(Kr_c) else int(round(law))
    # mass-aware rule
    lo, hi = 0.2 * Kr_c, 5 * Kr_c
    g = lambda K: p.v * U / (2 * Pf(K) * rmax + p.v * Ph(K) * p.B_bits / p.R)
    for _ in range(80):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if g(mid) > mid else (lo, mid)
    Kr_m = (lo + hi) / 2
    Gam = lambda K: K * (U - K * Pf(K) * tc) / (U + K * tc * (Pb0 * Pf(K) / Pf(Ks[0]) - Pf(K)))
    Kc_m = max(range(1, 200), key=lambda K: Gam(K) if U - K * Pf(K) * tc > 0 else -1)
    l_m = math.floor(Kr_m) if math.floor(Kr_m) <= Kc_m else Kc_m
    am_c = min(const[k], key=lambda K: float(const[k][K]["J"])) if k in const else None
    clamp = lambda l: min(max(l, Ks[0]), Ks[-1])
    rows.append(dict(fam=k[0], E=k[2], am=am, am_c=am_c, l_c=l_c, l_m=l_m, Kr_c=Kr_c, Kr_m=Kr_m,
                     reg_c=J[clamp(l_c)] / J[am] - 1, reg_m=J[clamp(l_m)] / J[am] - 1))
print(f"{len(rows)} mass-model instances; sweeps not converged (top edge or descending): {unconverged}")
def show(lab, R):
    a = np.array([x["am"] for x in R], float); ac = np.array([x["am_c"] for x in R], float)
    lc = np.array([x["l_c"] for x in R]); lm = np.array([x["l_m"] for x in R])
    print(f"{lab:7s} n={len(R):2d} | optimum: constant power {ac.mean():5.2f}, mass {a.mean():5.2f} (moved in {np.mean(a != ac):.0%}) "
          f"| constant-power rule: exact {np.mean(lc == a):.2f} w1 {np.mean(abs(lc - a) <= 1):.2f} regret med {np.median([x['reg_c'] for x in R]):.1%} "
          f"| fixed-point rule: exact {np.mean(lm == a):.2f} w1 {np.mean(abs(lm - a) <= 1):.2f} regret med {np.median([x['reg_m'] for x in R]):.1%}")
show("all", rows)
for f in ("paper", "ring", "core"):
    for E in (1.5e6, 3e6): show(f"{f} {E/1e6:g}", [x for x in rows if x["fam"] == f and x["E"] == E])
print(f"fixed-point K_reach / constant K_reach: median {np.median([x['Kr_m']/x['Kr_c'] for x in rows]):.3f}")
