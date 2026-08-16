"""
kstar_regime.py -- ANALYTIC fleet-sizing model. No solver, no simulator, seconds to run.

Evaluates argmin_K of the per-round objective under two AoI semantics:
  STATIC  (delivery age, A^d) : cost = theta1 * sum_k W_cum(k) * dt_k     [stage-weighted]
  DYNAMIC (field age,   A^f) : cost = theta1/T_r * age-integral over ALL M nodes

Question it answers: does K* depend on M under each semantic?
Validation gate: STATIC must reproduce banked K* ~ round(Emax/12500) within the
documented +/-1 band (M=50 is the known margin).

Usage:
  python kstar_regime.py --out-dir results_kstar_regime
  python kstar_regime.py --out-dir results_kstar_regime --Tr 60 120 240
  python kstar_regime.py --out-dir results_kstar_regime --emin 3529 3950 4229
"""
import argparse, csv, os
import numpy as np

# ---- constants, read out of compare_baseline.py / uav_aoi_solver.py ----------
PF, V, L, EPS   = 150.0, 20.0, 1000.0, 7.5
TH1, TH2, WBAR  = 0.01, 1.0, 5.5
M_LIST          = (50, 100, 200)
E_LIST          = (25_000, 50_000, 100_000)
K_MAX           = 24

# ---- the one input that is NOT settled --------------------------------------
# Banked fleet-slope e_min series. CONTEXT open item: "resolve which script/grid
# produced the rising e_min series (3529/3950/4229)". This model is most
# sensitive to exactly that number -- see the --emin sensitivity sweep.
EMIN_DEFAULT = {50: 3529.0, 100: 3950.0, 200: 4229.0}


def hop(M):
    """Mean inter-node hop under the confirmed density law a ~ sqrt(M)."""
    d = L / np.sqrt(M)
    return d, EPS * d, d / V          # metres, joules, seconds


def reach(M, e, emin, T_r=None):
    """Nodes one drone reaches on budget e, optionally capped by round length."""
    _, e_node, dtau = hop(M)
    n = max(0.0, (e - emin[M]) / e_node)
    binder = 'energy'
    if T_r is not None and T_r / dtau < n:
        n, binder = T_r / dtau, 'time'
    return n, binder


def J_static(M, Emax, K, emin, **_):
    _, _, dtau = hop(M)
    n, _ = reach(M, Emax / K, emin)
    if n <= 0:
        return np.inf, {}
    cost = TH1 * WBAR * (n ** 2 / 2.0) * dtau     # multiplier grows along the tour
    gain = TH2 * WBAR * n
    return K * (cost - gain), {'n': n, 'coverage': K * n / M}


def J_dynamic(M, Emax, K, emin, T_r=120.0, A_max=500.0):
    """Field age over ALL M nodes, per-round normalised."""
    _, _, dtau = hop(M)
    n, binder = reach(M, Emax / K, emin, T_r)
    if n < 1:
        return np.inf, {}
    served   = min(K * n, M)
    coverage = K * n / M
    # Mean age at visit = revisit period = T_r * (rounds between visits).
    # Closed form, not an iteration: served nodes per round out of M.
    abar = min(A_max, T_r * M / max(served, 1e-9))

    tau = np.arange(1, int(n) + 1) * dtau
    tau = tau[tau <= T_r]
    if len(tau) == 0:
        return np.inf, {}
    # visited: age accrues to tau, resets, then regrows for the remainder
    integral = K * (WBAR * (abar * tau + tau ** 2 / 2.0 + (T_r - tau) ** 2 / 2.0)).sum()
    # unvisited remainder ages through the whole round
    integral += max(0.0, M - served) * WBAR * (abar * T_r + T_r ** 2 / 2.0)

    cost = TH1 * integral / T_r
    gain = TH2 * WBAR * served
    return cost - gain, {'n': n, 'coverage': coverage, 'binder': binder, 'abar': abar}


def regime(cov):
    if cov is None:      return '?'
    if cov >= 1.0:       return 'OVER-SERVED'   # exclude: extra drones ~free
    if cov >= 0.6:       return 'well-served'
    return 'under-served'


def sweep(fn, label, emin, rows, **kw):
    print(f'\n{label}')
    print(f"{'M':>5} " + ''.join(f'{E//1000:>7}k' for E in E_LIST) + '    regime @ 100k')
    for M in M_LIST:
        ks, covs = [], []
        for Emax in E_LIST:
            vals, metas = [], []
            for k in range(1, K_MAX + 1):
                v, m = fn(M, Emax, k, emin, **kw)
                vals.append(v); metas.append(m)
            i = int(np.nanargmin(vals))
            ks.append(i + 1); covs.append(metas[i].get('coverage'))
            rows.append({'model': label, 'M': M, 'Emax': Emax, 'Kstar': i + 1,
                         'coverage': round(covs[-1], 3) if covs[-1] else '',
                         'regime': regime(covs[-1]), **{k2: round(v2, 2)
                         for k2, v2 in metas[i].items() if isinstance(v2, float)}})
        print(f'{M:>5} ' + ''.join(f'{k:>8}' for k in ks) + f'    {regime(covs[-1])}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='results_kstar_regime')
    ap.add_argument('--Tr', type=float, nargs='+', default=[60.0, 120.0, 240.0])
    ap.add_argument('--emin', type=float, nargs=3, default=None,
                    help='e_min for M=50 100 200 (overrides banked series)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    emin = EMIN_DEFAULT if a.emin is None else dict(zip(M_LIST, a.emin))
    rows = []

    print(f'e_min series in use: {emin}')
    print('banked static target: 25k->2  50k->4  100k->8   (M=50 gives 9; band is +/-1)')
    sweep(J_static, 'STATIC (delivery age) [VALIDATION GATE]', emin, rows)
    for T in a.Tr:
        sweep(J_dynamic, f'DYNAMIC (field age) T_r={T:.0f}s', emin, rows, T_r=T)

    path = os.path.join(a.out_dir, 'kstar_regime.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)
    print(f'\nwrote {path}  ({len(rows)} rows)')
    print('\nRead the DYNAMIC blocks only where regime != OVER-SERVED.')


if __name__ == '__main__':
    main()