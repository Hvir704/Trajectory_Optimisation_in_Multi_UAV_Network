"""
kstar_predict.py  --  close the K* derivation from already-measured data.
==========================================================================
NO SOLVER RUNS. Reads CSVs produced by kstar_primitives.py and fleet_coverage.py
and does the algebra, so it is instant and re-runnable.

WHY THIS SCRIPT EXISTS
----------------------
The eta(K) = N_meas / (K*r(Emax/K)) correction FAILED its own sanity check:
eta(1) = 0.35-0.40, not ~1. A single UAV competes with nobody, so competition
cannot be the explanation. eta was also non-monotone (peak at K=3) and exceeded
1 at M=200 -- both impossible for a competition correction. Do not fit eta.

The actual cause, visible in primitives_raw.csv: the SINGLE-UAV reach curve does
not rise to M, it PLATEAUS at r_cap ~ 17 / 25 / 34 nodes for M = 50 / 100 / 200.
That is the objective cap (c1*n^2 outgrows p1*n), i.e. Remark 5 at K=1, and it
is a branch Eq. 31 does not have. So:

    N(K) = min( A - B*K ,  K * r_cap )          <- two-branch coverage model
             ^ reach-limited   ^ objective-capped

with A = a*Emax and B = a*e_min. Fitting A, B on the reach-limited cells only
(K >= --Kfit-min, past the crossover at K ~ 2.5) recovers the primitives that
Eq. 31 needs. The single-UAV sweep gets `a` right but underestimates `e_min` by
5-10x, because under the Eq. 16 partition every UAV pays its own depot transit
whereas a lone UAV just picks the cheapest region.

The plateau additionally MEASURES c1: an unconstrained single UAV stops where
d/dn (c1*n^2 - p1*n) = 0, so c1 = p1/(2*r_cap). With that substitution
p1/c1 = 2*r_cap and p1 cancels from the dominant term of e*:

    e* = sqrt( e_min^2 + 2*r_cap*e_min/a + 2*r_cap*s/(p1*a^2) )

p1 then survives only in the last term (~10% of e*^2), and c1 is no longer an
unmeasured constant from the math note.

USAGE
-----
    python kstar_predict.py --fleet fleet_coverage/fleet_coverage.csv \
                            --raw   kstar_primitives/primitives_raw.csv

    # after the budget-invariance run:
    python kstar_predict.py --fleet fleet_coverage/fleet_coverage.csv \
                                    fleet_coverage_budgets/fleet_coverage.csv \
                            --raw   kstar_primitives/primitives_raw.csv \
                            --fit-budget 50000

--fit-budget selects which Emax the primitives are fitted on; every OTHER budget
present becomes a held-out test. That is the honest out-of-sample check and it
is the sentence reviewers will look for.
"""
import os, csv, argparse
import numpy as np


# ----------------------------------------------------------------- primitives
def plateau(e, r, tol=0.10, min_pts=2):
    """Flat tail of the single-UAV reach curve -> (r_cap, e_tail, n_pts)."""
    idx = np.argsort(np.asarray(e, float))[::-1]
    ev, rv = np.asarray(e, float)[idx], np.asarray(r, float)[idx]
    acc = [rv[0]]
    for k in range(1, len(rv)):
        m = float(np.mean(acc))
        if m > 1e-9 and abs(rv[k] - m) / m > tol:
            break
        acc.append(rv[k])
    if len(acc) < min_pts:
        return float('nan'), float('nan'), len(acc)
    return float(np.mean(acc)), float(ev[len(acc) - 1]), len(acc)


def fit_AB(K, N, Kmin):
    """N = A - B*K on the reach-limited branch. Returns (A, B, R2, n)."""
    K, N = np.asarray(K, float), np.asarray(N, float)
    m = K >= Kmin
    if m.sum() < 3:
        return (float('nan'),) * 3 + (int(m.sum()),)
    x, y = K[m], N[m]
    D = np.vstack([x, np.ones_like(x)]).T
    sl, ic = np.linalg.lstsq(D, y, rcond=None)[0]
    pred = sl * x + ic
    ss_t = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - pred) ** 2).sum()) / ss_t if ss_t > 1e-12 else float('nan')
    return float(ic), float(-sl), r2, int(m.sum())


def e_star(e_min, a, c1, p1, s):
    if not (np.isfinite(a) and a > 0 and np.isfinite(c1) and c1 > 0):
        return float('nan')
    v = e_min ** 2 + p1 * e_min / (c1 * a) + s / (c1 * a * a)
    return float(np.sqrt(v)) if v > 0 else float('nan')


# ----------------------------------------------------------------------- data
def load_fleet(paths):
    """-> {(M, Emax): [(K, N_meas, N_std), ...]}"""
    d = {}
    for p in paths:
        with open(p, newline='') as f:
            for row in csv.DictReader(f):
                key = (int(row['M']), int(float(row['Emax'])))
                d.setdefault(key, []).append((int(row['K']),
                                              float(row['N_meas']),
                                              float(row.get('N_std', 'nan') or 'nan')))
    for k in d:
        d[k].sort()
    return d


def load_raw(path):
    """-> {M: (energies, reaches)}"""
    d = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            d.setdefault(int(row['M']), []).append((float(row['e']), float(row['r'])))
    return {M: (np.array([p[0] for p in v]), np.array([p[1] for p in v]))
            for M, v in d.items()}


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fleet', nargs='+', default=['fleet_coverage/fleet_coverage.csv'])
    ap.add_argument('--raw', default='kstar_primitives/primitives_raw.csv')
    ap.add_argument('--Kfit-min', type=int, default=3,
                    help='lowest K on the reach-limited branch (crossover is ~2.5)')
    ap.add_argument('--fit-budget', type=float, default=None,
                    help='Emax to fit primitives on; others become held-out tests')
    ap.add_argument('--p1', type=float, default=5.66)
    ap.add_argument('--s', type=float, default=2.4)
    ap.add_argument('--c1-nominal', type=float, default=0.102)
    ap.add_argument('--plateau-tol', type=float, default=0.10)
    ap.add_argument('--Emax-check', type=float, nargs='+',
                    default=[25000, 50000, 100000])
    ap.add_argument('--out-dir', default='kstar_predict')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    fleet, raw = load_fleet(a.fleet), load_raw(a.raw)
    Ms = sorted(set(M for M, _ in fleet) & set(raw))
    budgets = sorted(set(E for _, E in fleet))
    fit_E = int(a.fit_budget) if a.fit_budget else budgets[0]

    L = '=' * 86
    print(L)
    print(f'  K* PREDICTOR  |  M={Ms}  budgets={budgets}  primitives fitted at Emax={fit_E}')
    print(f'  Kfit-min={a.Kfit_min}  p1={a.p1}  s={a.s}')
    print(L)

    # ---- 1. plateau / c1 ---------------------------------------------------
    print('\n[1] OBJECTIVE-CAP PLATEAU  (from the single-UAV sweep)')
    print(f'    {"M":>4} {"r_cap":>7} {"pts":>4} {"e_tail":>8} {"r_cap/sqrtM":>12} '
          f'{"c1=p1/2r_cap":>13} {"c1*sqrtM":>9}')
    cap, c1m = {}, {}
    for M in Ms:
        rc, et, nt = plateau(*raw[M], tol=a.plateau_tol)
        cap[M] = rc
        c1m[M] = a.p1 / (2 * rc) if np.isfinite(rc) and rc > 0 else float('nan')
        print(f'    {M:>4} {rc:>7.2f} {nt:>4} {et:>8.0f} {rc/np.sqrt(M):>12.3f} '
              f'{c1m[M]:>13.4f} {c1m[M]*np.sqrt(M):>9.3f}')
    print('    -> both right-hand columns should be FLAT in M (that is the check).')

    # ---- 2. two-branch fit -------------------------------------------------
    print(f'\n[2] TWO-BRANCH FIT   N(K) = min(A - B*K, K*r_cap),  A,B on K>={a.Kfit_min}')
    prim, rows = {}, []
    for M in Ms:
        for E in budgets:
            if (M, E) not in fleet:
                continue
            d = fleet[(M, E)]
            A, B, r2, n = fit_AB([x[0] for x in d], [x[1] for x in d], a.Kfit_min)
            aM, emin = A / E, (B / (A / E) if A else float('nan'))
            prim[(M, E)] = (aM, emin)
            xo = A / (B + cap[M]) if np.isfinite(cap[M]) else float('nan')
            print(f'\n  M={M} Emax={E}   A={A:.2f} B={B:.3f} R2={r2:.4f} [{n} pts]'
                  f'   -> a={aM:.5g}  e_min={emin:.0f} J   crossover K={xo:.2f}')
            print(f'    {"K":>3} {"N_meas":>8} {"std":>6} {"A-BK":>8} {"K*r_cap":>8} '
                  f'{"model":>8} {"resid":>8}')
            for K, N, sd in d:
                lin, capK = A - B * K, K * cap[M]
                mod = min(lin, capK)
                print(f'    {K:>3} {N:>8.2f} {sd:>6.2f} {lin:>8.2f} {capK:>8.2f} '
                      f'{mod:>8.2f} {N-mod:>+8.2f}')
                rows.append(dict(M=M, Emax=E, K=K, N_meas=N, N_std=sd,
                                 branch_linear=round(lin, 3), branch_cap=round(capK, 3),
                                 model=round(mod, 3), resid=round(N - mod, 3)))
    print('\n  Residuals should sit inside N_std. If K=1,2 are badly off the CAP branch,')
    print('  r_cap is mis-detected; if K>=Kfit_min are off the LINEAR branch, A/B are.')

    # ---- 3. budget invariance ---------------------------------------------
    if len(budgets) > 1:
        print('\n[3] BUDGET INVARIANCE  (a and e_min must NOT drift with Emax)')
        print(f'    {"M":>4} ' + ' '.join(f'{"a@"+str(E):>12}' for E in budgets)
              + ' | ' + ' '.join(f'{"emin@"+str(E):>12}' for E in budgets) + f' {"emin spread":>12}')
        for M in Ms:
            av = [prim.get((M, E), (np.nan, np.nan))[0] for E in budgets]
            ev = [prim.get((M, E), (np.nan, np.nan))[1] for E in budgets]
            fin = [x for x in ev if np.isfinite(x)]
            spread = (max(fin) - min(fin)) / np.mean(fin) if len(fin) > 1 else float('nan')
            print(f'    {M:>4} ' + ' '.join(f'{x:>12.5g}' for x in av) + ' | '
                  + ' '.join(f'{x:>12.0f}' for x in ev) + f' {spread*100:>11.1f}%')
        print('    -> spread under ~15% means these are PRIMITIVES and the closed form holds.')
        print('       Large drift means e_min is budget-dependent and the model is local.')
    else:
        print('\n[3] BUDGET INVARIANCE: only one budget present -- run fleet_coverage at')
        print('    Emax=25000 and 100000, then re-run with both CSVs. This is the gate.')

    # ---- 4. e* and K* ------------------------------------------------------
    print(f'\n[4] DERIVED CONSTANT AND FLEET SIZE   (primitives from Emax={fit_E})')
    print(f'    {"M":>4} {"a":>11} {"e_min":>7} {"c1":>8} {"e*":>8} '
          + ' '.join(f'{"K*("+str(int(E//1000))+"k)":>9}' for E in a.Emax_check))
    fitrows = []
    for M in Ms:
        if (M, fit_E) not in prim:
            continue
        aM, emin = prim[(M, fit_E)]
        est = e_star(emin, aM, c1m[M], a.p1, a.s)
        est_nom = e_star(emin, aM, a.c1_nominal, a.p1, a.s)
        print(f'    {M:>4} {aM:>11.5g} {emin:>7.0f} {c1m[M]:>8.4f} {est:>8.0f} '
              + ' '.join(f'{E/est:>9.2f}' for E in a.Emax_check))
        fitrows.append(dict(M=M, fit_Emax=fit_E, a=aM, e_min=emin,
                            a_over_sqrtM=aM/np.sqrt(M), r_cap=cap[M],
                            c1_plateau=c1m[M], e_star=est, e_star_nominal_c1=est_nom,
                            **{f'Kstar_E{int(E)}': round(E/est, 3) for E in a.Emax_check}))
    print(f'\n    for reference, with the nominal c1={a.c1_nominal} from the math note:')
    for r in fitrows:
        print(f'      M={r["M"]:>4}  e*={r["e_star_nominal_c1"]:.0f} J  '
              + '  '.join(f'K*({int(E//1000)}k)={E/r["e_star_nominal_c1"]:.2f}'
                          for E in a.Emax_check))
    print('\n    Compare against kstar_sa.csv. Held-out budgets (any Emax in --Emax-check')
    print(f'    other than {fit_E}) are genuine out-of-sample predictions -- say so in the paper.')

    # ---- 5. sensitivity ----------------------------------------------------
    print('\n[5] SENSITIVITY OF K*(50k)  (referee-proofing the free parameters)')
    for label, vals, kw in [('p1', [3.0, 4.5, 5.66, 7.0, 9.0], 'p1'),
                            ('s', [0.0, 1.2, 2.4, 3.6, 5.0], 's'),
                            ('Kfit_min', [3, 4], 'Kfit_min')]:
        print(f'\n    vs {label}:')
        print(f'      {label:>9} ' + ' '.join(f'{"M="+str(M):>8}' for M in Ms))
        for v in vals:
            out = []
            for M in Ms:
                if kw == 'Kfit_min':
                    d = fleet.get((M, fit_E), [])
                    A, B, _, n = fit_AB([x[0] for x in d], [x[1] for x in d], int(v))
                    aM = A / fit_E; emin = B / aM if A else float('nan')
                    c1u = c1m[M]; p1u, su = a.p1, a.s
                else:
                    aM, emin = prim[(M, fit_E)]
                    p1u = v if kw == 'p1' else a.p1
                    su = v if kw == 's' else a.s
                    c1u = p1u / (2 * cap[M])          # c1 tracks p1 by construction
                est = e_star(emin, aM, c1u, p1u, su)
                out.append(50000 / est if np.isfinite(est) else float('nan'))
            print(f'      {v:>9} ' + ' '.join(f'{x:>8.2f}' for x in out))
    print('\n    K* should barely move vs p1 (it cancels) and vs s (small term).')

    # ---- write -------------------------------------------------------------
    p1f = os.path.join(a.out_dir, 'two_branch_residuals.csv')
    p2f = os.path.join(a.out_dir, 'kstar_predicted.csv')
    with open(p1f, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(p2f, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fitrows[0].keys())); w.writeheader(); w.writerows(fitrows)
    print(f'\n  wrote {p1f}\n  wrote {p2f}')
    print(L)


if __name__ == '__main__':
    main()
