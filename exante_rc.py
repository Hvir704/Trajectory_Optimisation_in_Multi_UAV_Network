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
usage: python exante_rc.py grid_final.csv [--out exante.csv] [--rc-model two|nearest]
"""
import sys, csv, math, argparse
import numpy as np
from collections import defaultdict
from dyn_env import DynParams, SensorField
from kref import kref

RC_MODEL = "two"     # "two": r_c = mean of the nearest and second-nearest of n draws (r_c is the mean of the first and
                     # last legs); "nearest": the old E[min of n] (biased low by construction). Set by --rc-model.


def order_stat_rc(radii, n):
    """Order-statistic estimate of the realised commute radius for n i.i.d. radial draws from the empirical
    distribution. E[r_(1)] = int S^n dx, E[r_(2)] = int S^n + n F S^(n-1) dx (S = 1 - F, piecewise constant).
    r_c is the mean of the first and last legs, so with n >= 2 stops it is (E r_(1) + E r_(2))/2; with one stop it is
    E r_(1). Between n = 1 and 2 the two are blended linearly so the estimate is continuous in n."""
    x = np.sort(radii); m = len(x)
    edges = np.concatenate([[0.0], x])
    surv = np.concatenate([[1.0], (m - np.arange(1, m + 1)) / m])[:-1]
    widths = np.diff(edges); n = max(n, 1.0)
    e1 = float(np.sum(widths * np.power(surv, n)))
    if RC_MODEL == "nearest":
        return e1
    e2 = e1 + float(np.sum(widths * n * (1 - surv) * np.power(surv, n - 1)))
    wgt = min(max(n - 1.0, 0.0), 1.0)
    return e1 + wgt * 0.5 * (e2 - e1)


_NN = {}
def exante(p, r, lam, pos, K, n_override=None, iters=60):
    U = (1 - p.rho) * p.Emax; Eu = U / K
    key = (pos.shape, float(pos.sum()), float(p.v))
    if key not in _NN:
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1); np.fill_diagonal(d, np.inf)
        _NN.clear(); _NN[key] = d.min(axis=1).mean() / p.v
    tau_hop = _NN[key]
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




if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("--out", default=None)
    ap.add_argument("--rc-model", choices=["two", "nearest"], default="two")
    a = ap.parse_args(); RC_MODEL = a.rc_model

    inst = defaultdict(dict)
    for r in csv.DictReader(open(a.grid)):
        inst[(r["layout"], int(r["M"]), float(r["Emax"]), float(r["L"]), int(r["seed"]))][int(r["K"])] = r


    rows = []
    for k, byK in sorted(inst.items()):
        lay, M, E, L, seed = k
        p = DynParams(M=M, Emax=E, layout=lay, L=L)
        F = SensorField(DynParams(M=M, Emax=E, layout=lay, L=L), np.random.default_rng(seed))
        r = np.linalg.norm(F.pos - p.home, axis=1); lam = np.full(M, p.lam_bits)   # nominal: true rates are unknown ex ante
        Ks = sorted(byK); J = {K: float(byK[K]["J"]) for K in Ks}; K0 = kref(byK); r0 = byK[K0]
        rc_m = float(r0["r_c"]); Pb_m = float(r0["P_bar"]); n_m = float(r0["mean_n"])
        kr = math.floor(float(r0["K_reach_i"])); am = min(J, key=J.get)

        rc_o, _, _ = exante(p, r, lam, F.pos, K0, n_override=n_m)
        rc_e, Pb_e, n_e = exante(p, r, lam, F.pos, K0)
        # commute fixed point: g(K) = Kc(rc_hat(K), Pb_hat(K)) decreases in K (fewer stops -> larger r_c), so the
        # estimator is the smallest K with g(K) <= K, found by bisection. Where g jumps across the diagonal (n_hat hits
        # its floor) no exact fixed point exists; the crossing is then flagged (fp_jump) and its residual reported.
        g = lambda K: Kc(p, *exante(p, r, lam, F.pos, max(K, 1.0))[:2])
        lo, hi = 1.0, max(2.0, float(K0))
        while g(hi) > hi: hi *= 2.0
        if g(lo) <= lo: hi = lo
        for _ in range(50):
            if hi - lo < 1e-4: break
            mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if g(mid) > mid else (lo, mid)
        Kfp = hi
        rc_f, Pb_f, _ = exante(p, r, lam, F.pos, max(Kfp, 1.0))
        fp_res = Kc(p, rc_f, Pb_f) - Kfp; fp_jump = abs(fp_res) > 0.05 * max(Kfp, 1.0)
        def rule(kc):
            law = min(kr, kc); return int(law) if law == kr else int(round(law))
        out = dict(layout=lay, M=M, Emax=E, L=L, seed=seed, argmin=am,
                   rc_meas=rc_m, rc_oracle_n=rc_o, rc_exante=rc_e, n_meas=n_m, n_exante=n_e,
                   Pb_meas=Pb_m, Pb_exante=Pb_e, rc_fp=rc_f, Pb_fp=Pb_f, K_fp=Kfp, fp_residual=fp_res, fp_jump=int(fp_jump),
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

    print(f"commute fixed point: {sum(x['fp_jump'] for x in rows)} of {len(rows)} instances have no exact fixed point "
          f"(g jumps across the diagonal); |residual| median {np.median([abs(x['fp_residual']) for x in rows]):.3f} UAVs, "
          f"max {max(abs(x['fp_residual']) for x in rows):.3f}")
    summ(rows, "ALL 272")
    summ([x for x in rows if x["L"] == 12600], "primary 240")
    for fam in ("paper", "ring", "core"):
        summ([x for x in rows if x["layout"] == fam and x["L"] == 12600], fam)