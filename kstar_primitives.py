"""
kstar_primitives.py  --  measure the coverage law r(e, M) = a(M)*(e - e_min).
========================================================================
The K* derivation gives

    K* = Emax / e*,     e* = sqrt( e_min^2 + p1*e_min/(c1*a) + s/(c1*a^2) )

so the constant e* (~12,500 J empirically) is PREDICTED once a(M) and e_min are
known. Eq. 32 of the math note estimates a from Beardwood-Halton-Hammersley, but
BHH is asymptotic and assumes near-optimal tours at large n; under split battery
each UAV serves ~10-25 nodes, so the BHH value is optimistic (it predicted
e* ~ 5,700 J vs the measured ~12,500). This script MEASURES a and e_min instead
of assuming them, which is what turns the fitted constant into a derived one.

METHOD (deliberately competition-free -- this is the SINGLE-UAV reach envelope):
  For each M, for each energy budget e in a log-spaced sweep, solve a ONE-UAV
  instance (K=1, Emax_each=e) with SA and record r = number of nodes served.
  Then fit the clipped-line model of Eq. 33 on the linear (unsaturated) part:
        r(e) = a * (e - e_min),  fitted where 0 < r < M_saturation_guard
  Reports per M:  a, e_min, R^2, and a/sqrt(M) (the density-law check: Eq. 32
  predicts a ∝ sqrt(M), so a/sqrt(M) should be ~constant across M).

Also reports the derived e* and K*(Emax) using the measured a, e_min, plus c1,
p1, s supplied on the command line (defaults from the note / your Phi data), so
you can compare directly against the measured K* surface in kstar_sa.csv.

Run (start small; each cell is one SA solve):
    python kstar_primitives.py --M 50 100 200 --instances 6 --iters 1500

Finer energy grid (better fit, slower):
    python kstar_primitives.py --M 50 100 200 --n-energy 14 --instances 8 --iters 2000
"""
import os, csv, argparse
import numpy as np

from compare_baseline import gen, INSTANCE_SEED
from sa_routes import sa_best_with_routes


def fit_line(e, r, M, sat_frac=0.85):
    """Least-squares r = a*(e - e_min) on the unsaturated, non-zero part.
    Returns (a, e_min, R2, n_used). Excludes r==0 (below floor) and r >= sat_frac*M
    (coverage-saturated), which are outside the linear branch of Eq. 33."""
    e = np.asarray(e, float); r = np.asarray(r, float)
    m = (r > 0.5) & (r < sat_frac * M)
    if m.sum() < 3:
        return float('nan'), float('nan'), float('nan'), int(m.sum())
    x, y = e[m], r[m]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
    a = float(slope)
    e_min = float(-intercept / slope) if abs(slope) > 1e-15 else float('nan')
    return a, e_min, r2, int(m.sum())


def plateau(e, r, tol=0.10, min_pts=2):
    """Detect the OBJECTIVE-CAP plateau of the single-UAV reach curve.

    Beyond ~20 kJ a single UAV stops adding nodes: c1*n^2 grows faster than
    p1*n, so reach saturates at r_cap << M.  Walk down from the highest energy
    and accumulate points while each stays within `tol` of the running mean.
    Returns (r_cap, e_tail, n_pts) or (nan, nan, 0) if no flat tail is found.
    """
    idx = np.argsort(np.asarray(e, float))[::-1]          # high e first
    ev = np.asarray(e, float)[idx]; rv = np.asarray(r, float)[idx]
    acc = [rv[0]]
    for k in range(1, len(rv)):
        m = float(np.mean(acc))
        if m > 1e-9 and abs(rv[k] - m) / m > tol:
            break
        acc.append(rv[k])
    if len(acc) < min_pts:
        return float('nan'), float('nan'), len(acc)
    return float(np.mean(acc)), float(ev[len(acc) - 1]), len(acc)


def c1_from_plateau(r_cap, p1):
    """An unconstrained single UAV maximises -(c1*n^2 - p1*n), so it stops at
    n = p1/(2*c1) = r_cap.  Hence c1 = p1/(2*r_cap) -- c1 MEASURED, not assumed.
    Eq. 32 predicts c1 = theta1*w_bar*tau_bar/2 with tau_bar ~ 1/a ~ 1/sqrt(M),
    so c1*sqrt(M) should come out flat; that flatness is the validity check."""
    if not np.isfinite(r_cap) or r_cap <= 0:
        return float('nan')
    return float(p1 / (2.0 * r_cap))


def e_star(e_min, a, c1, p1, s):
    """e* = sqrt(e_min^2 + p1*e_min/(c1*a) + s/(c1*a^2))  -- the derived constant.

    NOTE: when c1 is taken from the plateau, p1/c1 = 2*r_cap and p1 cancels out
    of the dominant middle term; p1 then survives only in the small s-term.
    """
    if not np.isfinite(a) or a <= 0 or not np.isfinite(c1) or c1 <= 0:
        return float('nan')
    v = e_min ** 2 + p1 * e_min / (c1 * a) + s / (c1 * a * a)
    return float(np.sqrt(v)) if v > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, nargs='+', default=[50, 100, 200])
    ap.add_argument('--e-lo', type=float, default=1500.0)
    ap.add_argument('--e-hi', type=float, default=60000.0)
    ap.add_argument('--n-energy', type=int, default=11)
    ap.add_argument('--instances', type=int, default=6)
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--restarts', type=int, default=1)
    ap.add_argument('--seed', type=int, default=INSTANCE_SEED)
    # coefficients for the e* prediction (override with measured values)
    ap.add_argument('--c1', type=float, default=0.102)
    ap.add_argument('--p1', type=float, default=5.66)
    ap.add_argument('--s', type=float, default=2.4,
                    help='deconfliction slope; 0 disables the Phi term')
    ap.add_argument('--plateau-tol', type=float, default=0.10,
                    help='relative tolerance for detecting the flat tail of r(e)')
    ap.add_argument('--no-plateau-c1', action='store_true',
                    help='report e* using --c1 only, skipping the plateau calibration')
    ap.add_argument('--Emax-check', type=float, nargs='+',
                    default=[25000, 50000, 100000])
    ap.add_argument('--out-dir', default='kstar_primitives')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    energies = np.unique(np.round(
        np.logspace(np.log10(a.e_lo), np.log10(a.e_hi), a.n_energy)))
    rng0 = np.random.default_rng(a.seed)
    seeds = [int(rng0.integers(0, 1e7)) for _ in range(a.instances)]

    print('=' * 84)
    print(f'  Single-UAV reach sweep (K=1, competition-free) | M={a.M}')
    print(f'  energies: {[int(x) for x in energies]}')
    print(f'  instances={a.instances} iters={a.iters} seed={a.seed}')
    print('=' * 84)

    raw, fits = [], []
    for M in a.M:
        print(f'\nM={M}')
        print(f'  {"e (J)":>8} {"r (nodes)":>10} {"std":>6}')
        es, rs = [], []
        for e in energies:
            counts = []
            for s in seeds:
                pos, wi, tcd = gen(M, s)
                _, trajs = sa_best_with_routes(pos, wi, tcd, 1, float(e), M,
                                               a.iters, a.restarts)
                counts.append(len(trajs[0]))
            rbar = float(np.mean(counts))
            es.append(float(e)); rs.append(rbar)
            raw.append(dict(M=M, e=float(e), r=round(rbar, 3),
                            r_std=round(float(np.std(counts)), 3),
                            instances=a.instances))
            print(f'  {e:>8.0f} {rbar:>10.2f} {np.std(counts):>6.2f}')

        aM, emin, r2, n = fit_line(es, rs, M)
        rcap, e_tail, n_tail = plateau(es, rs, tol=a.plateau_tol)
        c1p = c1_from_plateau(rcap, a.p1)
        c1_use = a.c1 if (a.no_plateau_c1 or not np.isfinite(c1p)) else c1p
        est = e_star(emin, aM, c1_use, a.p1, a.s)
        est_nom = e_star(emin, aM, a.c1, a.p1, a.s)
        fits.append(dict(M=M, a=aM, a_inv=1/aM if aM else float('nan'),
                         e_min=emin, R2=r2, n_points=n,
                         a_over_sqrtM=aM/np.sqrt(M) if np.isfinite(aM) else float('nan'),
                         r_cap=rcap, r_cap_over_sqrtM=rcap/np.sqrt(M),
                         e_tail=e_tail, n_tail=n_tail,
                         c1_plateau=c1p, c1_plateau_sqrtM=c1p*np.sqrt(M),
                         c1_used=c1_use, e_star_nominal_c1=est_nom,
                         e_star=est,
                         **{f'Kstar_E{int(E)}': (round(E/est, 2) if np.isfinite(est) else float('nan'))
                            for E in a.Emax_check}))
        print(f'  fit: a={aM:.5g} nodes/J  (1/a={1/aM if aM else float("nan"):.0f} J/node)'
              f'  e_min={emin:.0f} J  R2={r2:.4f}  [{n} pts]')
        print(f'  plateau: r_cap={rcap:.2f} nodes over {n_tail} pts (e>={e_tail:.0f} J)'
              f'   r_cap/sqrt(M)={rcap/np.sqrt(M):.3f}')
        print(f'           => c1 = p1/(2*r_cap) = {c1p:.4f}   c1*sqrt(M)={c1p*np.sqrt(M):.3f}')
        print(f'  => e* = {est:.0f} J   ' +
              '  '.join(f'K*({int(E)})={E/est:.2f}' for E in a.Emax_check))

    with open(os.path.join(a.out_dir, 'primitives_raw.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)
    with open(os.path.join(a.out_dir, 'primitives_fit.csv'), 'w', newline='') as f:
        keys = list(fits[0].keys())
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(fits)

    print('\n' + '=' * 84)
    print('  DENSITY-LAW CHECK (Eq. 32 predicts a ∝ sqrt(M), so this should be flat):')
    for r in fits:
        print(f'    M={r["M"]:>4}  a/sqrt(M) = {r["a_over_sqrtM"]:.6g}')
    print('\n  OBJECTIVE-CAP CHECK (r_cap/sqrt(M) and c1*sqrt(M) should both be flat):')
    for r in fits:
        print(f'    M={r["M"]:>4}  r_cap={r["r_cap"]:>6.2f}  r_cap/sqrt(M)={r["r_cap_over_sqrtM"]:.3f}'
              f'   c1={r["c1_plateau"]:.4f}  c1*sqrt(M)={r["c1_plateau_sqrtM"]:.3f}')
    print('    (a flat c1*sqrt(M) confirms c1 = theta1*w*tau/2 with tau ~ 1/sqrt(M).)')
    print('\n  DERIVED CONSTANT vs MEASURED (~12,500 J from kstar_sa.csv):')
    for r in fits:
        print(f'    M={r["M"]:>4}  e* = {r["e_star"]:.0f} J   '
              + '  '.join(f'K*({int(E)})={r[f"Kstar_E{int(E)}"]}' for E in a.Emax_check))
    print('\n  Compare those K* against kstar_sa.csv. If they match within +/-1, the')
    print('  law is DERIVED, not fitted -- which is the contribution reviewers asked for.')
    print(f'  wrote {a.out_dir}/primitives_raw.csv and primitives_fit.csv')
    print('=' * 84)


if __name__ == '__main__':
    main()
