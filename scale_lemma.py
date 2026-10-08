"""scale_lemma.py -- the cost-scale lemma for criterion (5) and Proposition 6, checked on data.

The surrogate prices a visit at c_i = alpha * c_rad_i, c_rad_i = 2 Pf r_i / v (a dedicated out-and-back flight).
alpha = 1 is the radial scale used in the paper; the 'consistent' scale sets alpha = Pbar*tau_s / (2 Pf rbar / v)
so that the mean visit cost equals the measured energy per stop.  Lemma: F_alpha(K) - F_alpha(K0) =
alpha*dA(K) + dS(K) with dS >= 0, hence >= alpha*(dA + dS); a pass at alpha=1 is a pass at every alpha <= 1.
This script counts verdict pairs (radial, consistent), checks for lemma violations, and reports the regret of
instances whose verdict changes.

usage: python scale_lemma.py GRID.csv [--only apriori_regret.csv]   (--only restricts to the instances listed)
"""
import sys, csv, math, collections, numpy as np
from dyn_env import DynParams, SensorField

args = sys.argv[1:]
only = None
if "--only" in args:
    f = args[args.index("--only") + 1]
    only = {(r["lay"], int(r["M"]), float(r["E"]), int(r["seed"])) for r in csv.DictReader(open(f))}
d = collections.defaultdict(dict)
for r in csv.DictReader(open(args[0])):
    if float(r.get("L", 12600)) != 12600 or r.get("coord", "exclude") != "exclude" or r.get("replan", "launch") != "launch":
        continue
    if r.get("planner", "sa") != "sa" or float(r.get("Th", 43200)) != 43200:
        continue
    d[(r["layout"], int(float(r["M"])), float(r["Emax"]), int(float(r["seed"])))][int(float(r["K"]))] = r

cnt = collections.Counter(); viol = []; alphas = []; changed = []; n = 0
for k, byK in sorted(d.items()):
    if only is not None and k not in only:
        continue
    lay, M, E, seed = k
    p = DynParams(M=M, Emax=E, layout=lay); U = (1 - p.rho) * E; H = p.T_burnin + (p.T_horizon - p.T_burnin) / 2
    Ks = sorted(byK); r0 = byK[Ks[0]]
    kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"])
    if min(kr, kc) != kr:
        continue                                         # commute-bound: criterion does not apply
    F = SensorField(DynParams(M=M, Emax=E, layout=lay), np.random.default_rng(seed))
    r = np.linalg.norm(F.pos - p.home, axis=1); w = F.wi_base; crad = 2 * p.Pf * r / p.v
    rc = float(r0["r_c"]); Pb = float(r0["P_bar"]); tc = 2 * rc / p.v
    tau_s = (float(r0["T_s_over_t_c"]) * p.t_c - tc) / float(r0["mean_n"])
    alpha = Pb * tau_s / (2 * p.Pf * r.mean() / p.v); alphas.append(alpha)
    Phi = lambda K: K * (U - K * p.Pf * tc) / (U + K * tc * (Pb - p.Pf))
    reach = lambda K: p.v * (U / K - p.Ph * p.B_bits / p.R) / (2 * p.Pf)
    Kmax = int(math.floor(U / (p.Pf * tc) - 1e-9)); grid = [K for K in range(kr + 1, Kmax + 1) if Phi(K) > 0]
    def Fa(K, a):
        S = r <= reach(K)
        return a * np.sqrt(w[S] * crad[S]).sum() ** 2 / (2 * Pb * Phi(K)) + H * w[~S].sum()
    passes = lambda a: all(Fa(K, a) > Fa(kr, a) for K in grid)
    v1, va = passes(1.0), passes(alpha); n += 1; cnt[(v1, va)] += 1
    if v1 and not va:
        viol.append(k)
    J = {K: float(byK[K]["J"]) for K in Ks}
    reg = J[kr] / min(J.values()) - 1 if kr in J else float("nan")
    if v1 != va:
        changed.append((lay, reg))

a = np.array(alphas)
print(f"reach-bound instances: {n}")
print(f"alpha (consistent / radial): median {np.median(a):.3f}  max {a.max():.3f}  min {a.min():.3f}")
print("verdict pairs (radial, consistent):", {f"{'pass' if x else 'flag'}/{'pass' if y else 'flag'}": c for (x, y), c in sorted(cnt.items())})
print(f"lemma violations (pass at radial, flag at consistent): {len(viol)} {viol if viol else ''}")
for fam in ("paper", "core"):
    x = np.array([g for l, g in changed if l == fam and g == g])
    if len(x):
        print(f"{fam}: {len(x)} instances flagged at the radial scale but passed at the consistent scale; "
              f"regret median {np.median(x):.1%}, max {x.max():.1%}, >=10%: {int((x >= 0.10).sum())}")
