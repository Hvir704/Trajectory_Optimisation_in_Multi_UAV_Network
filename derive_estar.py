"""
derive_estar.py  --  predict e* from instance geometry. NO SOLVER RUNS.
==========================================================================
Implements DERIVATION_e_star_geometric.md:

    e_min = 2*rho*eps*L                        (depot round trip)
    mid   = 2*eps*v*theta2/theta1              (= 2*r_cap/a, M- L- and c-free)
    e*    = sqrt( e_min^2 + mid*e_min + third )
    K*    = Emax / e*

The point of this script is that NOTHING here comes from a solver. All five
inputs are read out of the instance generator and the objective definition.
If the output matches the measured 12,500 J, the fleet-sizing law is derived
from physical parameters rather than fitted.

PRE-REGISTERED FALSIFICATION TARGETS (measured, this project):
    e_min   = 3,900 +/- 350 J      (fleet slope, M = 50/100/200)
    mid     = 32,400 +/- 2,000 J   (2*r_cap/a, M = 50/100/200)
    e*      = 12,500 J             (Emax/K*, 3 budgets x 3 M)
    K*      = 2 / 4 / 8-9  at Emax = 25k / 50k / 100k

Read these five constants out of the code before running:

  --eps      J per metre flown.  If the model is E = P*d/v then eps = P/v.
  --v        cruise speed, m/s.
  --L        side of the square sensor field, m.  Non-square: use sqrt(area).
  --depot    'centre' | 'corner' | 'custom'.  Sets rho = E[d(depot,X)]/L.
             centre -> 0.38259,  corner -> 0.76519 (exact square constants).
             'custom' triggers a Monte-Carlo for rho from --depot-xy.
  --theta2 --theta1   linear and quadratic AoI weights from the objective.

Example:
    python derive_estar.py --eps 12.5 --v 15 --L 1000 --depot centre \
                           --theta2 1.0 --theta1 0.004

Diagnostics if it misses:
  * e_min off by ~exactly 2x   -> wrong --depot (centre vs corner).
  * e_min right, mid wrong     -> geometry fine, re-read theta2/theta1.
  * both off by the same ratio -> eps or L is wrong (they multiply together).
  * e_min right, mid right, e* wrong -> arithmetic bug here, not in the model.
"""
import argparse
import numpy as np

# exact mean distance from a square's centre / corner to a uniform point,
# in units of the side length L
RHO_CENTRE = (np.sqrt(2) + np.log(1 + np.sqrt(2))) / 6          # 0.382597...
RHO_CORNER = (np.sqrt(2) + np.log(1 + np.sqrt(2))) / 3          # 0.765195...

# measured anchors (see DERIVATION_e_star_geometric.md)
TARGET = dict(e_min=(3900.0, 350.0), mid=(32400.0, 2000.0), e_star=(12500.0, 0.0))


def rho_monte_carlo(dx, dy, n=2_000_000, seed=0):
    """rho = E[ d(depot, X) ] / L for X uniform on the unit square.
    depot given as (dx, dy) in units of L; may lie outside [0,1]."""
    rng = np.random.default_rng(seed)
    p = rng.random((n, 2))
    return float(np.mean(np.hypot(p[:, 0] - dx, p[:, 1] - dy)))


def band(name, value, target, tol):
    lo, hi = target - tol, target + tol
    if tol == 0:
        err = abs(value - target) / target * 100
        ok = err <= 20
        return f'{name:>8} = {value:>9.0f}   target {target:>7.0f}   ' \
               f'err {err:>5.1f}%   {"PASS" if ok else "MISS"}'
    ok = lo <= value <= hi
    err = abs(value - target) / target * 100
    return f'{name:>8} = {value:>9.0f}   target {target:>7.0f} +/- {tol:.0f}   ' \
           f'err {err:>5.1f}%   {"PASS" if ok else "MISS"}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, required=True, help='J per metre')
    ap.add_argument('--v', type=float, required=True, help='cruise speed m/s')
    ap.add_argument('--L', type=float, required=True, help='field side, m')
    ap.add_argument('--depot', choices=['centre', 'corner', 'custom'], default='centre')
    ap.add_argument('--depot-xy', type=float, nargs=2, default=None,
                    help='depot (x,y) in units of L, for --depot custom')
    ap.add_argument('--theta1', type=float, required=True, help='quadratic AoI weight')
    ap.add_argument('--theta2', type=float, required=True, help='linear AoI weight')
    ap.add_argument('--Emax', type=float, nargs='+',
                    default=[25000, 50000, 100000, 150000, 200000])
    # optional: include the deconfliction term, which needs s, c1, a and so is
    # NOT solver-free. Off by default -- it is O(M^-1/2) and 4-12% of the total.
    ap.add_argument('--s', type=float, default=None)
    ap.add_argument('--c1', type=float, default=None)
    ap.add_argument('--a', type=float, default=None)
    g = ap.parse_args()

    if g.depot == 'centre':
        rho = RHO_CENTRE
    elif g.depot == 'corner':
        rho = RHO_CORNER
    else:
        if not g.depot_xy:
            ap.error('--depot custom requires --depot-xy X Y')
        rho = rho_monte_carlo(*g.depot_xy)

    e_min = 2 * rho * g.eps * g.L
    mid = 2 * g.eps * g.v * g.theta2 / g.theta1
    third = 0.0
    if None not in (g.s, g.c1, g.a):
        third = g.s / (g.c1 * g.a * g.a)
    e_star = float(np.sqrt(e_min ** 2 + mid * e_min + third))

    L = '=' * 78
    print(L)
    print('  GEOMETRIC PREDICTION OF e*   (no solver in the loop)')
    print(L)
    print(f'\n  inputs:  eps={g.eps} J/m   v={g.v} m/s   L={g.L} m')
    print(f'           depot={g.depot} -> rho={rho:.5f}')
    print(f'           theta1={g.theta1}  theta2={g.theta2}  '
          f'(ratio {g.theta2/g.theta1:.4g})')

    print('\n  [1] TERM BY TERM')
    print(f'      e_min = 2*rho*eps*L            = {e_min:>10.0f} J')
    print(f'      mid   = 2*eps*v*theta2/theta1  = {mid:>10.0f} J')
    if third:
        print(f'      third = s/(c1*a^2)             = {third:>10.3g} '
              f'({third/(mid*e_min)*100:.1f}% of mid*e_min)')
    else:
        print('      third = (omitted; O(M^-1/2), 4-12% of the total)')
    print(f'      e*    = sqrt(e_min^2 + mid*e_min{" + third" if third else ""})'
          f'  = {e_star:>10.0f} J')

    print('\n  [2] AGAINST THE PRE-REGISTERED MEASURED VALUES')
    print('      ' + band('e_min', e_min, *TARGET['e_min']))
    print('      ' + band('mid', mid, *TARGET['mid']))
    print('      ' + band('e*', e_star, *TARGET['e_star']))

    print('\n  [3] FLEET SIZE   K* = Emax / e*')
    meas = {25000: '2 / 2 / 2', 50000: '4 / 4 / 4', 100000: '9 / 8 / 8',
            150000: '14 / 14 / 13', 200000: '>=18 / >=18 / 17'}
    print(f'      {"Emax":>8} {"K* pred":>9}   measured (M=50/100/200)')
    for E in g.Emax:
        print(f'      {int(E):>8} {E/e_star:>9.2f}   {meas.get(int(E), "-")}')
    print('\n      The law holds in the reach-limited regime (25k-100k).')
    print('      At 150k/200k the budget goes slack and K* grows superlinearly,')
    print('      so a linear predictor must under-shoot there -- that is expected.')

    print('\n  [4] READING THE RESULT')
    ok_e = abs(e_min - TARGET['e_min'][0]) <= TARGET['e_min'][1]
    ok_m = abs(mid - TARGET['mid'][0]) <= TARGET['mid'][1]
    if ok_e and ok_m:
        print('      Both primitives land. e* is derived from instance parameters.')
        print('      Remaining gate: the tour-length check (ell/n flat vs n) from')
        print('      sa_routes.py trajectories -- that is the load-bearing assumption.')
    elif not ok_e and abs(e_min / 2 - TARGET['e_min'][0]) <= 2 * TARGET['e_min'][1]:
        print('      e_min is ~2x high: --depot is probably corner, not centre.')
    elif not ok_e and abs(e_min * 2 - TARGET['e_min'][0]) <= 2 * TARGET['e_min'][1]:
        print('      e_min is ~2x low: --depot is probably centre, not corner.')
    elif ok_e and not ok_m:
        print('      Geometry checks out, objective coefficients do not.')
        print('      Re-read theta2/theta1 from the objective; check whether the')
        print('      code folds w_bar into them (it cancels in the ratio, so a')
        print('      mis-split of w_bar between theta1 and theta2 shows up here).')
    else:
        print('      Both miss. If they miss by the SAME ratio, eps*L is wrong')
        print('      (they multiply); check field units (m vs km) first.')
    print('\n' + L)


if __name__ == '__main__':
    main()
