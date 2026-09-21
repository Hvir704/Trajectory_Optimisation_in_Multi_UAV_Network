"""exante_rc.py -- can the commute inputs (r_c, Pbar) be predicted before deployment?

Measured target (dyn_env.metrics): r_c = mean over post-burn-in sorties of (depot->first stop +
last stop->depot)/2, and Pbar = time-weighted productive power, both from the smallest-K row, which is
what the point predictor uses.

Ex-ante inputs only: sensor positions (site survey), nominal generation rates lambda_i, the airframe
(v, Pf, Ph, R_u, B) and the energy inventory U. No simulation, no flight.

  r_c  : order-statistic estimate of Sec. VII-I, E[r_c | n] = int_0^rmax (1 - F_r(x))^n dx, with F_r
         the empirical radial CDF of the survey.
  n    : stops per sortie from the energy budget, n = (E_u - Pf t_c) / (Pbar tau_s), with
         tau_s = tau_hop / (1 - h) and tau_hop = (mean nearest-neighbour spacing)/v.
  Pbar : from hover conservation (Prop. 3): the fleet hovers H = sum lambda_i / R_u UAV-equivalents,
         so the hover share of productive time is h = H / (K (1 - t_c/T_s)), Pbar = Pf + (Ph-Pf) h.
  The three are solved as a fixed point in (r_c, n, h) at the fleet size K where the rule is
  evaluated: the smallest swept K (to match the measured target) and, separately, the commute
  fixed point K = K_commute(r_c(K)).

Variants reported, so the error can be attributed:
  oracle_n   : order statistic with the MEASURED n  -> error of the order-statistic formula alone
  exante     : everything ex-ante                    -> the honest price of ex-ante use
usage: python exante_rc.py grid_final.csv [--out exante.csv]
"""
import sys, csv, math, argparse
import numpy as np
from collections import defaultdict
from dyn_env import DynParams, SensorField

ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("--out", default=None)
a = ap.parse_args()

inst = defaultdict(dict)
for r in csv.DictReader(open(a.grid)):
    inst[(r["layout"], int(r["M"]), float(r["Emax"]), float(r["L"]), int(r["seed"]))][int(r["K"])] = r


def order_stat_rc(radii, n):
    """E[min of n i.i.d. draws from the empirical radial distribution] = int (1-F)^n dx."""
    x = np.sort(radii); m = len(x)
    # piecewise-constant survival on [x_k, x_{k+1}): S = (m-k-1)/m
    edges = np.concatenate([[0.0], x])
    surv = np.concatenate([[1.0], (m - np.arange(1, m + 1)) / m])[:-1]
    widths = np.diff(edges)
    return float(np.sum(widths * np.power(surv, max(n, 1.0))))


def exante(p, r, lam, pos, K, n_override=None, iters=60):
    U = (1 - p.rho) * p.Emax; Eu = U / K
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1); np.fill_diagonal(d, np.inf)
    tau_hop = d.min(axis=1).mean() / p.v
    H = lam.sum() / p.R                           # fleet hover, UAV-equivalents
    rc = np.median(r); n = 10.0; Pb = p.Pf
    for _ in range(iters):
        tc = 2 * rc / p.v
        Ts = tc + max(Eu - p.Pf * tc, 0) / max(Pb, 1e-9)
        prod_frac = max(1 - tc / max(Ts, 1e-9), 1e-3)
        h = min(max(H / (K * prod_frac), 0.0), 0.95)
        Pb = p.Pf + (p.Ph - p.Pf) * h
        tau_s = tau_hop / (1 - h)
        n = n_override if n_override is not None else max((Eu - p.Pf * tc) / (Pb * tau_s), 1.0)
        rc_new = order_stat_rc(r, n)
        if abs(rc_new - rc) < 1e-3: rc = rc_new; break
        rc = 0.5 * rc + 0.5 * rc_new
    return rc, Pb, n


def Kc(p, rc, Pb):
    return (1 - p.rho) * p.Emax * p.v / (2 * rc * (p.Pf + math.sqrt(p.Pf * Pb)))


rows = []
for k, byK in sorted(inst.items()):
    lay, M, E, L, seed = k
    p = DynParams(M=M, Emax=E, layout=lay, L=L)
    F = SensorField(DynParams(M=M, Emax=E, layout=lay, L=L), np.random.default_rng(seed))
    r = np.linalg.norm(F.pos - p.home, axis=1); lam = F.lam_bits
    Ks = sorted(byK); J = {K: float(byK[K]["J"]) for K in Ks}; r0 = byK[Ks[0]]
    rc_m = float(r0["r_c"]); Pb_m = float(r0["P_bar"]); n_m = float(r0["mean_n"])
    kr = math.floor(float(r0["K_reach_i"])); am = min(J, key=J.get)

    rc_o, _, _ = exante(p, r, lam, F.pos, Ks[0], n_override=n_m)
    rc_e, Pb_e, n_e = exante(p, r, lam, F.pos, Ks[0])
    # commute fixed point K = Kc(rc(K)), started from the smallest swept K
    Kfp = float(Ks[0])
    for _ in range(40):
        rc_f, Pb_f, _ = exante(p, r, lam, F.pos, max(Kfp, 1.0))
        Knew = Kc(p, rc_f, Pb_f)
        if abs(Knew - Kfp) < 1e-3: break
        Kfp = 0.5 * Kfp + 0.5 * Knew
    def rule(kc):
        law = min(kr, kc); return int(law) if law == kr else int(round(law))
    out = dict(layout=lay, M=M, Emax=E, L=L, seed=seed, argmin=am,
               rc_meas=rc_m, rc_oracle_n=rc_o, rc_exante=rc_e, n_meas=n_m, n_exante=n_e,
               Pb_meas=Pb_m, Pb_exante=Pb_e,
               rule_meas=rule(float(r0["K_commute_i"])), rule_oracle_n=rule(Kc(p, rc_o, Pb_m)),
               rule_exante=rule(Kc(p, rc_e, Pb_e)), rule_fixedpoint=rule(Kfp),
               reg_meas=0.0, reg_exante=0.0, reg_fixedpoint=0.0, ratio_meas=float(r0["rmax_over_rc"]),
               ratio_exante=float(r.max()) / rc_e, bnd_exante=1 + math.sqrt(Pb_e / p.Pf), bnd_meas=float(r0["regime_bnd"]))
    for tag in ("meas", "exante", "fixedpoint"):
        l = out[f"rule_{tag}"]; l = min(max(l, Ks[0]), Ks[-1])
        out[f"reg_{tag}"] = J[l] / J[am] - 1 if l in J else float("nan")
    rows.append(out)

if a.out:
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def summ(R, lab):
    R = list(R)
    if not R: return
    g = lambda c: np.array([x[c] for x in R], float)
    am = g("argmin")
    print(f"{lab:22s} n={len(R):3d} | r_c ratio est/meas: oracle-n {np.median(g('rc_oracle_n')/g('rc_meas')):.2f} "
          f"ex-ante {np.median(g('rc_exante')/g('rc_meas')):.2f} [{np.quantile(g('rc_exante')/g('rc_meas'),.1):.2f},"
          f"{np.quantile(g('rc_exante')/g('rc_meas'),.9):.2f}] | Pbar ex-ante/meas {np.median(g('Pb_exante')/g('Pb_meas')):.3f} "
          f"| n ex-ante/meas {np.median(g('n_exante')/g('n_meas')):.2f}")
    for tag in ("meas", "oracle_n", "exante", "fixedpoint"):
        l = g(f"rule_{tag}")
        extra = ""
        if f"reg_{tag}" in R[0]:
            rg = g(f"reg_{tag}"); extra = f"  regret median {np.nanmedian(rg):5.1%} p90 {np.nanquantile(rg,.9):5.1%}"
        print(f"   rule with {tag:10s}: exact {np.mean(l==am):.3f} within-one {np.mean(abs(l-am)<=1):.3f}{extra}")
    reg_m = np.array([(x["ratio_meas"] > x["bnd_meas"]) for x in R]); reg_e = np.array([(x["ratio_exante"] > x["bnd_exante"]) for x in R])
    print(f"   regime classification (reach vs commute) agrees with measured-input classification: {np.mean(reg_m==reg_e):.3f}")

summ(rows, "ALL 272")
summ([x for x in rows if x["L"] == 12600], "primary 240")
for fam in ("paper", "ring", "core"):
    summ([x for x in rows if x["layout"] == fam and x["L"] == 12600], fam)
