r"""
measure_geometry.py  --  measure the geometric primitives directly from SA
trajectories.  v2: corrected and extended.
==========================================================================
Everything here is measured from solver output. Nothing is fitted to produce
a target value, and every comparison names both sides explicitly.

CHANGES FROM v1 (and why)
-------------------------
[D] SWEPT-AREA TEST IS NEW AND IS NOW THE DECISIVE ONE.
    v1's test [C] regressed hop length on n, which is an indirect probe of the
    clustered-tour assumption. The assumption is really a claim about the AREA
    a UAV sweeps. Measure the radius of gyration Rg of the served set:

        clustered   : n nodes drawn from a local patch of area n/delta
                      -> Rg^2 = alpha * n * L^2 / M      (log-log exponent 1)
        field-spread: nodes taken from the whole field
                      -> Rg^2 = L^2/6, constant in n     (log-log exponent 0)

    No fitting, no free constant needed to distinguish them. If the exponent
    is near 1 the derivation's foundation holds; near 0 and r(e)'s measured
    linearity has some other cause and the derivation must be rebuilt.

[C] NOW RUNS WITHIN-CELL AS WELL AS POOLED.
    v1 pooled chains across K at fixed M. That confounds n with the energy
    budget: at high K each UAV is poor and picks scattered high-weight nodes,
    at low K it can afford to fill in locally. So a negative pooled slope may
    be measuring priority-driven selection, not tour geometry. Within a fixed
    (M, K) cell the budget is constant and chains still vary in n across UAVs
    and instances -- that is the controlled comparison.

[B] CAVEAT STATED INLINE.
    1/a (the fleet coverage slope) is a MARGINAL INSERTION cost: adding a node
    between two existing ones costs d(x,new) + d(new,y) - d(x,y), which is less
    than a full hop. It is NOT comparable to tour_len/(n-1). Comparing them was
    an error in an earlier analysis; the honest comparison for the timing
    channel is tau_bar = tcd_bar + hop/V against theta2/(theta1*r_cap).

[A] TWO-TERM e_min MODEL.
    v1 tested the symmetric e_min = 2*rho*eps*L and it missed by 32-46%.
    The trajectories show why: the OUTBOUND leg is the field-mean depot
    distance (d_first ~ rho*L, confirmed to 6-14%) but the RETURN leg is much
    shorter, because chain_waoi weights the return by the chain's TOTAL
    accumulated weight and so drives the last node close to HOME. Hence

        e_min = eps * (rho*L + kappa*L/sqrt(M))

    with rho = 0.38260 derived and kappa an empirical constant carrying a
    derived sqrt(M) scaling. This script measures kappa rather than assuming
    the earlier value.

--uniform-w
    Replaces every weight with wbar, removing priority-driven node selection.
    This isolates Remark 4's homogeneous-priority approximation. If [D]'s
    exponent moves toward 1 under --uniform-w but not otherwise, the clustered
    assumption holds for homogeneous priorities and priority heterogeneity is a
    separate, characterisable correction. Run BOTH and compare.

USAGE
-----
    python measure_geometry.py --M 50 100 200 --Emax 50000 --K 2 3 4 5 6 \
                               --instances 8 --iters 1500

    python measure_geometry.py --M 50 100 200 --Emax 50000 --K 2 3 4 5 6 \
                               --instances 8 --iters 1500 --uniform-w \
                               --out-dir measure_geometry_uniform
"""
import os, csv, argparse
import numpy as np

from compare_baseline import gen, INSTANCE_SEED, HOME, PF, PH, V, AREA, wlo, whi
from sa_routes import sa_best_with_routes

EPS = PF / V                                                # J per metre flown
WBAR = (wlo + whi) / 2.0                                    # mean node weight
RHO_FIELD = (np.sqrt(2) + np.log(1 + np.sqrt(2))) / 6       # 0.38260
RG2_FIELD = AREA ** 2 / 6.0            # E|X-centroid|^2 for uniform square
C_TIMING = 0.775                       # hop constant implied by r_cap inversion
R_CAP = {50: 17.25, 100: 25.333, 200: 33.833, 400: 50.34}   # measured plateau
THETA_RATIO = 100.0                    # TH2/TH1, the AoI patience horizon (s)
TCD_BAR = 0.138                        # mean collection time, s


def chain_geometry(t, pos, tcd):
    """Decompose one chain. Mirrors chain_energy() in compare_baseline.py:
        E = eps*d_first + eps*tour_len + eps*d_last + PH*sum(tcd)
    """
    n = len(t)
    if n == 0:
        return None
    idx = np.asarray(t)
    P = pos[idx]
    d_first = float(np.linalg.norm(P[0] - HOME))
    d_last = float(np.linalg.norm(P[-1] - HOME))
    tour = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) if n > 1 else 0.0
    cen = P.mean(axis=0)
    rg2 = float(np.mean(np.sum((P - cen) ** 2, axis=1)))
    return dict(n=n,
                d_first=d_first, d_last=d_last,
                d_mean=float(np.mean(np.linalg.norm(P - HOME, axis=1))),
                d_centroid=float(np.linalg.norm(cen - HOME)),
                tour_len=tour,
                hop=tour / (n - 1) if n > 1 else float('nan'),
                Rg2=rg2,
                E_excursion=EPS * (d_first + d_last),
                E_tour=EPS * tour,
                E_hover=PH * float(np.sum(np.asarray(tcd)[idx])))


def lsq(x, y):
    """Least squares y = slope*x + icept. Returns (slope, icept, R2, n)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) < 1e-12:
        return float('nan'), float('nan'), float('nan'), len(x)
    D = np.vstack([x, np.ones_like(x)]).T
    sl, ic = np.linalg.lstsq(D, y, rcond=None)[0]
    pred = sl * x + ic
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - pred) ** 2).sum()) / ss if ss > 1e-12 else float('nan')
    return float(sl), float(ic), r2, len(x)


def loglog_exp(n, y):
    """Exponent of y ~ n^e, ignoring non-positive values."""
    n, y = np.asarray(n, float), np.asarray(y, float)
    ok = (n > 0) & (y > 0) & np.isfinite(n) & np.isfinite(y)
    if ok.sum() < 3:
        return float('nan'), float('nan'), int(ok.sum())
    e, _, r2, k = lsq(np.log(n[ok]), np.log(y[ok]))
    return e, r2, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, nargs='+', default=[50, 100, 200])
    ap.add_argument('--Emax', type=float, nargs='+', default=[50000])
    ap.add_argument('--K', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    ap.add_argument('--instances', type=int, default=8)
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--restarts', type=int, default=1)
    ap.add_argument('--seed', type=int, default=INSTANCE_SEED)
    ap.add_argument('--uniform-w', action='store_true',
                    help='replace all weights with wbar, removing priority-driven '
                         'node selection (isolates Remark 4)')
    ap.add_argument('--min-n', type=int, default=3,
                    help='minimum chain length for the geometry regressions')
    ap.add_argument('--out-dir', default='measure_geometry')
    g = ap.parse_args()

    os.makedirs(g.out_dir, exist_ok=True)
    rng0 = np.random.default_rng(g.seed)
    seeds = [int(rng0.integers(0, 1e7)) for _ in range(g.instances)]

    L = '=' * 96
    print(L)
    print(f'  TRAJECTORY GEOMETRY v2  |  M={g.M}  Emax={[int(x) for x in g.Emax]}  K={g.K}')
    print(f'  eps=PF/V={EPS} J/m   HOME={HOME}   field={AREA} m   wbar={WBAR}')
    print(f'  instances={g.instances}  iters={g.iters}  restarts={g.restarts}  '
          f'weights={"UNIFORM (priority selection OFF)" if g.uniform_w else "heterogeneous"}')
    print(L)

    chains, cells = [], []
    for M in g.M:
        print(f'\nM={M}')
        print(f'  {"Emax":>7} {"K":>3} {"n":>6} {"d_1st":>7} {"d_last":>7} {"d_cen":>7}'
              f' {"hop":>7} {"Rg":>7} {"E_exc":>7} {"E_tour":>7} {"E_hov":>6}')
        for E in g.Emax:
            for K in g.K:
                rec = []
                for s in seeds:
                    pos, wi, tcd = gen(M, s)
                    if g.uniform_w:
                        wi = np.full_like(np.asarray(wi, float), WBAR)
                    _, trajs = sa_best_with_routes(pos, wi, tcd, K, float(E) / K, M,
                                                   g.iters, g.restarts)
                    for t in trajs:
                        cg = chain_geometry(t, pos, tcd)
                        if cg and cg['n'] >= 2:
                            cg.update(M=M, Emax=int(E), K=K, seed=s,
                                      uniform_w=int(g.uniform_w))
                            rec.append(cg)
                            chains.append(cg)
                if not rec:
                    print(f'  {int(E):>7} {K:>3}   (no chains of length >= 2)')
                    continue
                mn = {k: float(np.mean([r[k] for r in rec]))
                      for k in ('n', 'd_first', 'd_last', 'd_centroid', 'hop',
                                'Rg2', 'E_excursion', 'E_tour', 'E_hover')}
                print(f'  {int(E):>7} {K:>3} {mn["n"]:>6.2f} {mn["d_first"]:>7.1f}'
                      f' {mn["d_last"]:>7.1f} {mn["d_centroid"]:>7.1f} {mn["hop"]:>7.1f}'
                      f' {np.sqrt(mn["Rg2"]):>7.1f} {mn["E_excursion"]:>7.0f}'
                      f' {mn["E_tour"]:>7.0f} {mn["E_hover"]:>6.0f}')
                cells.append(dict(M=M, Emax=int(E), K=K, n_chains=len(rec),
                                  uniform_w=int(g.uniform_w), **mn))

    if not chains:
        print('\n  no chains collected -- nothing to analyse'); return

    def sel(M):
        return [c for c in chains if c['M'] == M and c['n'] >= g.min_n]

    # ---------------------------------------------------------------- [A]
    print('\n' + L)
    print('[A] EXCURSION COST   e_min = eps*(d_first + d_last)')
    print('    Derivation predicts the OUTBOUND leg is the field-mean depot distance,')
    print(f'    rho*L = {RHO_FIELD*AREA:.0f} m.  The RETURN leg is driven short by the AoI')
    print('    penalty on the final hop (chain_waoi weights it by total chain weight).')
    print(f'\n    {"M":>4} {"d_first":>8} {"vs rho*L":>9} {"d_last":>7} '
          f'{"kappa":>7} {"E_exc":>8} {"sd":>7} {"2*rho*eps*L":>12} {"err":>8}')
    kappas = []
    for M in g.M:
        v = sel(M)
        if not v:
            continue
        df = float(np.mean([c['d_first'] for c in v]))
        dl = float(np.mean([c['d_last'] for c in v]))
        ex = [c['E_excursion'] for c in v]
        kap = dl * np.sqrt(M) / AREA
        kappas.append(kap)
        sym = 2 * RHO_FIELD * EPS * AREA
        print(f'    {M:>4} {df:>8.1f} {(df/(RHO_FIELD*AREA)-1)*100:>8.1f}% {dl:>7.1f}'
              f' {kap:>7.3f} {float(np.mean(ex)):>8.0f} {float(np.std(ex)):>7.0f}'
              f' {sym:>12.0f} {(sym/np.mean(ex)-1)*100:>7.1f}%')
    if kappas:
        kap = float(np.mean(kappas))
        print(f'\n    kappa = d_last*sqrt(M)/L = {kap:.3f} +/- {np.std(kappas):.3f} '
              f'({np.std(kappas)/kap*100:.1f}%)  -- return leg scales as node spacing')
        print(f'    two-term model  e_min = eps*(rho*L + kappa*L/sqrt(M)):')
        print(f'      {"M":>4} {"pred":>8} {"measured":>9} {"err":>8}')
        for M in g.M:
            v = sel(M)
            if not v:
                continue
            pred = EPS * (RHO_FIELD * AREA + kap * AREA / np.sqrt(M))
            meas = float(np.mean([c['E_excursion'] for c in v]))
            print(f'      {M:>4} {pred:>8.0f} {meas:>9.0f} {(pred/meas-1)*100:>7.1f}%')
        print('    NOTE: the fleet-slope e_min (from N vs K) is a DIFFERENT estimate and')
        print('    trends the opposite way in M. If they disagree, the fleet intercept is')
        print('    absorbing something beyond the excursion -- say so rather than merging.')

    # ---------------------------------------------------------------- [B]
    print('\n[B] TIMING CHANNEL vs ENERGY CHANNEL')
    print('    Compared like-for-like: tau_bar = tcd_bar + hop/V, measured from tour')
    print('    length, against theta2/(theta1*r_cap) implied by the measured plateau.')
    print('    CAVEAT: 1/a from fleet coverage is a MARGINAL INSERTION cost and is NOT')
    print('    comparable to hop -- do not use it here.')
    print(f'\n    {"M":>4} {"hop":>7} {"c=hop*sqrtM/L":>14} {"tau_meas":>9} '
          f'{"tau_rcap":>9} {"err":>7} {"r_cap pred":>11} {"r_cap meas":>11}')
    for M in g.M:
        v = sel(M)
        if not v:
            continue
        hop = float(np.mean([c['hop'] for c in v if np.isfinite(c['hop'])]))
        tau_m = TCD_BAR + hop / V
        rc = R_CAP.get(M)
        if rc:
            tau_r = THETA_RATIO / rc
            print(f'    {M:>4} {hop:>7.1f} {hop*np.sqrt(M)/AREA:>14.3f} {tau_m:>9.3f}'
                  f' {tau_r:>9.3f} {(tau_m/tau_r-1)*100:>6.1f}% '
                  f'{THETA_RATIO/tau_m:>11.1f} {rc:>11.2f}')
        else:
            print(f'    {M:>4} {hop:>7.1f} {hop*np.sqrt(M)/AREA:>14.3f} {tau_m:>9.3f}'
                  f' {"-":>9} {"-":>7} {"-":>11} {"-":>11}')
    print(f'    (timing-channel c from r_cap inversion = {C_TIMING:.3f})')

    # ---------------------------------------------------------------- [C]
    print('\n[C] HOP vs n   -- pooled AND within-cell')
    print('    Pooling across K confounds n with the energy budget, which changes which')
    print('    nodes get picked. The within-cell rows hold the budget fixed.')
    print(f'\n    {"M":>4} {"scope":>12} {"n range":>11} {"exponent":>9} {"R2":>7} {"pts":>5}')
    for M in g.M:
        v = sel(M)
        if len(v) >= 6:
            e, r2, k = loglog_exp([c['n'] for c in v], [c['hop'] for c in v])
            n = [c['n'] for c in v]
            print(f'    {M:>4} {"pooled":>12} {f"{min(n):.0f}-{max(n):.0f}":>11}'
                  f' {e:>9.3f} {r2:>7.4f} {k:>5}')
        for K in g.K:
            w = [c for c in v if c['K'] == K]
            n = [c['n'] for c in w]
            if len(w) < 6 or max(n) - min(n) < 2:
                continue          # too little n variation within the cell to regress
            e, r2, k = loglog_exp(n, [c['hop'] for c in w])
            print(f'    {"":>4} {f"K={K}":>12} {f"{min(n):.0f}-{max(n):.0f}":>11}'
                  f' {e:>9.3f} {r2:>7.4f} {k:>5}')

    # ---------------------------------------------------------------- [D]
    print('\n[D] SWEPT-AREA TEST   Rg^2 vs n      *** THE DECISIVE ONE ***')
    print('    clustered    : Rg^2 = alpha*n*L^2/M   -> log-log exponent ~ 1')
    print(f'    field-spread : Rg^2 = L^2/6 = {RG2_FIELD:.0f}, constant -> exponent ~ 0')
    print(f'\n    {"M":>4} {"scope":>12} {"n range":>11} {"exponent":>9} {"R2":>7}'
          f' {"Rg2/Rg2_field":>14} {"alpha":>7} {"verdict":>18}')
    for M in g.M:
        v = sel(M)

        def row(w, label):
            n = np.array([c['n'] for c in w], float)
            if len(w) < 6 or (label != 'pooled' and np.ptp(n) < 2):
                return            # too little n variation to regress
            e, r2, k = loglog_exp(n, [c['Rg2'] for c in w])
            rg2 = np.array([c['Rg2'] for c in w], float)
            alpha = float(np.mean(rg2 * M / (n * AREA ** 2)))
            if not np.isfinite(e):
                verdict = 'n/a'
            else:
                verdict = ('CLUSTERED' if e > 0.70 else
                           ('FIELD-SPREAD' if e < 0.30 else 'INTERMEDIATE'))
            print(f'    {M if label=="pooled" else "":>4} {label:>12}'
                  f' {f"{n.min():.0f}-{n.max():.0f}":>11} {e:>9.3f} {r2:>7.4f}'
                  f' {float(np.mean(rg2))/RG2_FIELD:>14.3f} {alpha:>7.3f} {verdict:>18}')

        row(v, 'pooled')
        for K in g.K:
            row([c for c in v if c['K'] == K], f'K={K}')
    print('\n    exponent ~ 1 -> swept area grows with n; r(e) linear; derivation holds.')
    print('    exponent ~ 0 -> nodes taken from the whole field; the measured linearity')
    print('                    of r(e) has another cause and the derivation needs rebuilding.')
    print('    Rg2/Rg2_field near 1 also indicates field-spread regardless of exponent.')
    print('    Compare this table against the --uniform-w run: if the exponent rises')
    print('    toward 1 with uniform weights, priority-driven selection (Remark 4) is')
    print('    the confound and the clustered assumption holds underneath it.')

    # ---------------------------------------------------------------- write
    f1 = os.path.join(g.out_dir, 'chain_geometry.csv')
    f2 = os.path.join(g.out_dir, 'cell_geometry.csv')
    with open(f1, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(chains[0].keys()))
        w.writeheader(); w.writerows(chains)
    with open(f2, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader(); w.writerows(cells)
    print(f'\n  wrote {f1}  ({len(chains)} chains)')
    print(f'  wrote {f2}  ({len(cells)} cells)')
    print(L)


if __name__ == '__main__':
    main()
