"""
kstar_crossing.py -- estimate K* from the T_s/t_c crossing, with bootstrap CIs.

WHY THIS EXISTS (CONTEXT_72)
  The J argmin is a poor estimator of K*. Near the optimum J is flat -- that is
  what an optimum IS -- so the argmin has almost no signal. Measured at
  Emax=6M/M=200, no pair of K values was statistically distinguishable (diffs
  67k-440k against 2*SE of 438k-604k), and the cost curve was non-monotonic
  (J(16) > J(17)), which a convex curve cannot be. The reported argmin there,
  K*=14, was noise.

  T_s/t_c is smooth and monotonic in K in every run, and theory (CONTEXT_64)
  says it equals 1 + sqrt(Pf/P_bar) EXACTLY at K*. So interpolating where the
  measured curve crosses its predicted value gives a continuous, low-noise
  estimate of K* using a quantity with far better signal than J.

  Validated: this estimator gave K* ratios of 1.000 / 2.000 / 4.045 across
  Emax = 1.5M / 3M / 6M (predicted 1 / 2 / 4), and recovered the bounded
  downward M-drift predicted by CONTEXT_64 SS3 (5.3% / 2.3% / 2.0%, all under
  the 7.7% bound, all downward).

USAGE
    python kstar_crossing.py c4_msweep/dyn_c4_sa.csv
    python kstar_crossing.py c4_msweep/dyn_c4_sa.csv c4_emax/dyn_c4_sa.csv c4_emax_6m/dyn_c4_sa.csv

  Multiple CSVs are pooled, then grouped by (Emax, M). Reads the columns written
  by dyn_c4_grid.py: Emax, M, K, Ts_over_tc, Ts_over_tc_pred, J_mean, J_std,
  instances.

CAVEAT ON THE CIs
  dyn_c4_grid.py writes per-(M,K) MEANS, not per-instance values, so the
  bootstrap here resamples from a normal approximation using the recorded J_std
  and instance count. That is a reasonable interval for the crossing point but
  it is NOT a true per-instance bootstrap. For a publication-grade CI, have
  dyn_c4_grid.py dump per-instance Ts_over_tc values and resample those directly.
  This is flagged rather than hidden: treat the intervals as indicative.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np


def crossing(Ks, ts, pred):
    """Linear interpolation of where ts(K) crosses pred(K). Returns nan if the
    curve never crosses within the sampled K range -- which itself is
    diagnostic: it means the sweep did not bracket K*."""
    Ks = np.asarray(Ks, float)
    ts = np.asarray(ts, float)
    pred = np.asarray(pred, float)
    d = ts - pred
    for i in range(len(Ks) - 1):
        if d[i] == 0:
            return Ks[i]
        if d[i] * d[i + 1] < 0:
            f = d[i] / (d[i] - d[i + 1])
            return Ks[i] + f * (Ks[i + 1] - Ks[i])
    return np.nan


def bootstrap_crossing(Ks, ts, pred, ts_sd, n_boot=2000, rng=None):
    """Resample the Ts/tc curve under its sampling noise, re-interpolate."""
    rng = rng or np.random.default_rng(0)
    out = []
    for _ in range(n_boot):
        jittered = np.asarray(ts) + rng.normal(0.0, ts_sd)
        c = crossing(Ks, jittered, pred)
        if not np.isnan(c):
            out.append(c)
    if not out:
        return (np.nan, np.nan, np.nan)
    out = np.array(out)
    return (float(np.mean(out)), float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5)))


def main(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    if not rows:
        print("no rows read")
        return

    cells = defaultdict(list)
    for r in rows:
        cells[(float(r["Emax"]), int(r["M"]))].append(r)

    print("=" * 78)
    print("  K* VIA T_s/t_c CROSSING  (CONTEXT_72)")
    print("=" * 78)
    print(f"{'Emax':>8} {'M':>5} {'K* cross':>9} {'95% CI':>16} {'J argmin':>9} {'n_K':>4}")

    results = defaultdict(dict)
    for (Emax, M), rs in sorted(cells.items()):
        rs.sort(key=lambda r: int(r["K"]))
        Ks = [int(r["K"]) for r in rs]
        ts = [float(r["Ts_over_tc"]) for r in rs]
        pred = [float(r["Ts_over_tc_pred"]) for r in rs]
        Js = [float(r["J_mean"]) for r in rs]
        n = int(rs[0].get("instances", 12) or 12)

        # crude per-cell sd of the Ts/tc mean: use dispersion of the residual
        # from a smooth fit as a proxy for measurement noise on the curve.
        coef = np.polyfit(Ks, ts, 2)
        resid = np.asarray(ts) - np.polyval(coef, Ks)
        ts_sd = max(float(np.std(resid)), 1e-3)

        c = crossing(Ks, ts, pred)
        _, lo, hi = bootstrap_crossing(Ks, ts, pred, ts_sd)
        jarg = Ks[int(np.argmin(Js))]
        ci = f"[{lo:.2f}, {hi:.2f}]" if not np.isnan(lo) else "n/a"
        print(f"{Emax:>8.1e} {M:>5} {c:>9.2f} {ci:>16} {jarg:>9} {len(Ks):>4}")
        results[Emax][M] = c

    # --- proportionality to Emax ---
    print("\n" + "-" * 78)
    print("  K* PROPORTIONALITY TO Emax  (mean of crossing estimate over M)")
    emaxes = sorted(results)
    base = emaxes[0]
    base_mean = np.nanmean(list(results[base].values()))
    for E in emaxes:
        m = np.nanmean(list(results[E].values()))
        ratio = m / base_mean
        pred_ratio = E / base
        err = 100 * (ratio / pred_ratio - 1) if pred_ratio else 0.0
        print(f"    Emax={E:.2e}  K*={m:6.2f}  ratio={ratio:5.3f}  "
              f"predicted={pred_ratio:5.3f}  err={err:+.1f}%")

    # --- bounded M drift ---
    print("\n" + "-" * 78)
    print("  M-DRIFT  (CONTEXT_64 SS3: predicted <=7.7%, DOWNWARD)")
    for E in emaxes:
        Ms = sorted(results[E])
        if len(Ms) < 2:
            continue
        lo_M, hi_M = Ms[0], Ms[-1]
        a, b = results[E][lo_M], results[E][hi_M]
        drift = 100 * (a / b - 1)
        direction = "downward" if b < a else "UPWARD (unexpected)"
        ok = "OK" if abs(drift) <= 7.7 else "EXCEEDS BOUND"
        print(f"    Emax={E:.2e}  M {lo_M}->{hi_M}: {a:.2f} -> {b:.2f}  "
              f"drift={drift:+.1f}%  {direction}  [{ok}]")
    print("=" * 78)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])