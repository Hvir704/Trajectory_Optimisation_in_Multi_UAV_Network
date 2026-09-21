"""criterion_variants.py -- design-time window criterion (5), three self-consistent forms, all on the
same Prop.-4 radial surrogate that Prop. 6 uses:  J_s(K) = Q_K^2 / (2 Pbar Phi(K)),  Q_K over S_guar(K),
abandoned cost H * W_A(K), H = T_b + T/2, A(K) = all sensors outside S_guar(K).
  corrected_KR : H W_A(K0+1) > J_s(K0) - J_s(K_R)                     (current printed form)
  KG           : H W_A(K0+1) > J_s(K0) - min_{K>K0} J_s(K)            (reviewer's sufficient form)
  global       : F(K) > F(K0) for every K in (K0, K_max]                (exact within the surrogate)
K ranges over integers with Phi(K) > 0. Verdict 'pass' = trust K0. Regret of the point predictor is
reported conditional on each verdict, as in Table VIII."""
import sys, csv, math, numpy as np
from collections import defaultdict
from dyn_env import DynParams, SensorField
d = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    if float(r["L"]) == 12600: d[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]))][int(r["K"])] = r
ap = {(x["lay"], int(x["M"]), float(x["E"]), int(x["seed"])): x for x in csv.DictReader(open(sys.argv[2]))}
res = defaultdict(list); detail = []
for k, x in ap.items():
    byK = d[k]; Ks = sorted(byK); J = {K: float(byK[K]["J"]) for K in Ks}; r0 = byK[Ks[0]]
    kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"]); law = min(kr, kc)
    li = int(law) if law == kr else int(round(law)); reg = J[li] / min(J.values()) - 1
    if x["reach_bound"] != "True":
        for v in ("banked", "corrected_KR", "KG", "global"): res[v].append((k[0], None, reg))
        continue
    lay, M, E, seed = k; p = DynParams(M=M, Emax=E, layout=lay); U = (1 - p.rho) * E
    F = SensorField(DynParams(M=M, Emax=E, layout=lay), np.random.default_rng(seed))
    r = np.linalg.norm(F.pos - p.home, axis=1); w = F.wi_base; c = 2 * p.Pf * r / p.v
    rc = float(r0["r_c"]); Pb = float(r0["P_bar"]); tc = 2 * rc / p.v
    Phi = lambda K: K * (U - K * p.Pf * tc) / (U + K * tc * (Pb - p.Pf))
    reach = lambda K: p.v * (U / K - p.Ph * p.B_bits / p.R) / (2 * p.Pf)
    H = p.T_burnin + (p.T_horizon - p.T_burnin) / 2
    K0 = kr; Kmax = int(math.floor(U / (p.Pf * tc) - 1e-9))
    def Js(K):
        S = r <= reach(K); return (np.sqrt(w[S] * c[S]).sum()) ** 2 / (2 * Pb * Phi(K))
    def WA(K): return w[~(r <= reach(K))].sum()
    Kgrid = [K for K in range(K0 + 1, Kmax + 1) if Phi(K) > 0]
    KR = min(max(round(U / (tc * (p.Pf + (p.Pf * Pb) ** .5))), K0 + 1), Kmax)
    v_KR = bool(H * WA(K0 + 1) > Js(K0) - Js(KR))
    v_KG = bool(H * WA(K0 + 1) > Js(K0) - min(Js(K) for K in Kgrid))
    F0 = Js(K0) + H * WA(K0)
    v_gl = bool(all(Js(K) + H * WA(K) > F0 for K in Kgrid))
    res["banked"].append((k[0], x["passes"] == "True", reg))
    res["corrected_KR"].append((k[0], v_KR, reg)); res["KG"].append((k[0], v_KG, reg)); res["global"].append((k[0], v_gl, reg))
    am = min(J, key=J.get); detail.append((k, K0, KR, am))

def row(lab, R):
    x = np.array([t[2] for t in R])
    if len(x) == 0: return f"{lab:24s} n=  0"
    return (f"{lab:24s} n={len(x):3d} median {np.median(x):6.1%} p90 {np.quantile(x,.9,method='higher'):6.1%} "
            f"max {x.max():6.1%} <5% {np.mean(x<0.05):4.0%}")
for v in ("banked", "corrected_KR", "KG", "global"):
    print(f"--- {v}")
    R = res[v]
    for fam in ("paper", "core"):
        print("  " + row(f"{fam} pass", [t for t in R if t[0] == fam and t[1] is True]))
        print("  " + row(f"{fam} flagged", [t for t in R if t[0] == fam and t[1] is False]))
    for th in (0.10, 0.20, 0.30):
        big = [t for t in R if t[1] is not None and t[2] > th]
        print(f"  regret >{th:.0%}: {len(big):2d} instances, passed (missed) {sum(1 for t in big if t[1]):2d}; "
              f"max regret among passes {max([t[2] for t in R if t[1] is True], default=0):.1%}" if th==0.30 else
              f"  regret >{th:.0%}: {len(big):2d} instances, passed (missed) {sum(1 for t in big if t[1]):2d}")
ok = sum(1 for k, K0, KR, am in detail if K0 <= am <= KR)
print(f"\nResult-1 interval [K0, K_R] contains the measured argmin in {ok}/{len(detail)} reach-bound instances")

# ---- Table VIII rows for the global form (TeX) ----
if len(sys.argv) > 3 and sys.argv[3] == "--tex":
    R = res["global"]
    def tex(lab, test, S):
        x = np.array([t[2] for t in S])
        print(f"{lab} & {test} & {len(x)} & {100*np.median(x):.1f}\\% & {100*np.quantile(x,.9,method='higher'):.1f}\\% & "
              f"{100*x.max():.1f}\\% & {100*np.mean(x<0.05):.0f}\\%\\\\")
    for fam in ("paper", "core"):
        tex(fam, "pass", [t for t in R if t[0] == fam and t[1] is True])
        tex(fam, "flagged", [t for t in R if t[0] == fam and t[1] is False])
    tex("ring", "(commute-bound)", [t for t in R if t[0] == "ring"])
    print("\\midrule")
    tex("all reach-bound, test passes", "", [t for t in R if t[0] != "ring" and t[1] is True])
    tex("all reach-bound, flagged", "", [t for t in R if t[0] != "ring" and t[1] is False])