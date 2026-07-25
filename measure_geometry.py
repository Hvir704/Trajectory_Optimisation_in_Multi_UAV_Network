r"""
measure_geometry.py  --  measure the geometric primitives directly from SA
trajectories, so e_min stops being fitted.
==========================================================================
Closes the one remaining gap in DERIVATION_e_star_geometric.md.

WHAT IS BEING TESTED
--------------------
`chain_energy` in compare_baseline.py decomposes exactly:

    E(chain) = eps*d_first  +  eps*tour_len  +  eps*d_last  +  PH*sum(tcd)
               \_____________________________/  \__________/
                excursion (fixed, ~n-independent)   productive

with eps = PF/V = 7.5 J/m. So the model's `e_min` IS the excursion term:

    e_min = eps * (d_first + d_last)

The geometric derivation predicted e_min = 2*rho*eps*L = 5,739 J using the
field-mean depot distance rho*L = 383 m. The fleet slope measures 3,900 J,
implying an effective one-way distance of only 260 m. The hypothesis is that
UAVs preferentially serve NEAR nodes -- energy is scarce, and `chain_waoi`
weights the return leg by the chain's total accumulated weight, so ending far
from HOME is heavily penalised.

This script measures d_first and d_last directly instead of assuming either
value. It reports three things:

  [A] EXCURSION      eps*(d_first + d_last) vs the fitted e_min = 3,900 J.
                     If these agree, e_min is structurally confirmed and
                     d_served becomes the single quantity left to derive.

  [B] HOP CONSTANT   tour_len/(n-1) vs c*L/sqrt(M) with c = 0.775 (obtained
                     independently from r_cap timing). Confirms the energy
                     channel against the timing channel.

  [C] CLUSTERED-TOUR ASSUMPTION -- the load-bearing step of the whole
                     derivation. Textbook BHH says a tour through n points in
                     a FIXED area has length ~ sqrt(n*A), so hop length would
                     SHRINK as 1/sqrt(n). The derivation instead assumes an
                     energy-limited UAV serves a locally clustered set, so the
                     swept area grows with n and hop length is CONSTANT in n.
                     Regressing hop on n at fixed M distinguishes them:
                        slope ~ 0        -> clustered assumption holds
                        slope < 0, ~n^-0.5 -> BHH holds and the derivation's
                                              linearity is unexplained
                     This is the test a reviewer will demand first.

USAGE
-----
    python measure_geometry.py --M 50 100 200 --Emax 50000 --K 2 3 4 5 6 \\
                               --instances 8 --iters 1500

    # wider, if time allows -- more n values sharpens test [C]
    python measure_geometry.py --M 50 100 200 400 --Emax 25000 50000 100000 \\
                               --K 2 3 4 5 6 8 10 --instances 8 --iters 1500
"""
import os, csv, argparse
import numpy as np

from compare_baseline import gen, INSTANCE_SEED, HOME, PF, PH, V, AREA
from sa_routes import sa_best_with_routes

EPS = PF / V                      # J per metre flown
C_TIMING = 0.775                  # hop constant from r_cap inversion (independent)
RHO_FIELD = (np.sqrt(2) + np.log(1 + np.sqrt(2))) / 6      # 0.38260
E_MIN_FITTED = 3900.0             # fleet-slope value this script is testing


def chain_geometry(t, pos, tcd):
    """Decompose one chain into excursion / tour / hover. Mirrors chain_energy."""
    if len(t) == 0:
        return None
    P = pos[np.asarray(t)]
    d_first = float(np.linalg.norm(P[0] - HOME))
    d_last = float(np.linalg.norm(P[-1] - HOME))
    tour = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) if len(t) > 1 else 0.0
    d_mean = float(np.mean(np.linalg.norm(P - HOME, axis=1)))
    return dict(n=len(t), d_first=d_first, d_last=d_last, d_mean=d_mean,
                tour_len=tour, hop=tour / (len(t) - 1) if len(t) > 1 else float('nan'),
                E_excursion=EPS * (d_first + d_last),
                E_tour=EPS * tour,
                E_hover=PH * float(np.sum(np.asarray(tcd)[np.asarray(t)])))


def lsq(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) < 1e-9:
        return float('nan'), float('nan'), float('nan')
    D = np.vstack([x, np.ones_like(x)]).T
    sl, ic = np.linalg.lstsq(D, y, rcond=None)[0]
    pred = sl * x + ic
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - pred) ** 2).sum()) / ss if ss > 1e-12 else float('nan')
    return float(sl), float(ic), r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, nargs='+', default=[50, 100, 200])
    ap.add_argument('--Emax', type=float, nargs='+', default=[50000])
    ap.add_argument('--K', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    ap.add_argument('--instances', type=int, default=8)
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--restarts', type=int, default=1)
    ap.add_argument('--seed', type=int, default=INSTANCE_SEED)
    ap.add_argument('--out-dir', default='measure_geometry')
    g = ap.parse_args()

    os.makedirs(g.out_dir, exist_ok=True)
    rng0 = np.random.default_rng(g.seed)
    seeds = [int(rng0.integers(0, 1e7)) for _ in range(g.instances)]

    L = '=' * 94
    print(L)
    print(f'  TRAJECTORY GEOMETRY  |  M={g.M}  Emax={[int(x) for x in g.Emax]}  K={g.K}')
    print(f'  eps=PF/V={EPS} J/m   HOME={HOME}   field={AREA} m   '
          f'instances={g.instances} iters={g.iters}')
    print(L)

    chains, cells = [], []
    for M in g.M:
        print(f'\nM={M}')
        print(f'  {"Emax":>7} {"K":>3} {"n":>6} {"d_first":>8} {"d_last":>8} {"d_mean":>8}'
              f' {"hop":>7} {"E_exc":>8} {"E_tour":>8} {"E_hov":>7}')
        for E in g.Emax:
            for K in g.K:
                rec = []
                for s in seeds:
                    pos, wi, tcd = gen(M, s)
                    _, trajs = sa_best_with_routes(pos, wi, tcd, K, float(E) / K, M,
                                                   g.iters, g.restarts)
                    for t in trajs:
                        cg = chain_geometry(t, pos, tcd)
                        if cg and cg['n'] >= 2:
                            cg.update(M=M, Emax=int(E), K=K, seed=s)
                            rec.append(cg); chains.append(cg)
                if not rec:
                    continue
                mean = {k: float(np.mean([r[k] for r in rec]))
                        for k in ('n', 'd_first', 'd_last', 'd_mean', 'hop',
                                  'E_excursion', 'E_tour', 'E_hover')}
                print(f'  {int(E):>7} {K:>3} {mean["n"]:>6.2f} {mean["d_first"]:>8.1f}'
                      f' {mean["d_last"]:>8.1f} {mean["d_mean"]:>8.1f} {mean["hop"]:>7.1f}'
                      f' {mean["E_excursion"]:>8.0f} {mean["E_tour"]:>8.0f}'
                      f' {mean["E_hover"]:>7.0f}')
                cells.append(dict(M=M, Emax=int(E), K=K, n_chains=len(rec), **mean))

    # ---- [A] excursion vs fitted e_min ------------------------------------
    print('\n' + L)
    print('[A] EXCURSION COST vs THE FITTED e_min')
    print(f'    {"M":>4} {"E_exc (J)":>10} {"sd":>7} {"d_1way (m)":>11}'
          f' {"vs fitted 3900":>15} {"vs 2*rho*eps*L":>15}')
    for M in g.M:
        v = [c['E_excursion'] for c in chains if c['M'] == M]
        if not v:
            continue
        m = float(np.mean(v))
        print(f'    {M:>4} {m:>10.0f} {float(np.std(v)):>7.0f} {m/(2*EPS):>11.0f}'
              f' {(m/E_MIN_FITTED-1)*100:>14.1f}% {(m/(2*RHO_FIELD*EPS*AREA)-1)*100:>14.1f}%')
    print(f'    field-mean prediction 2*rho*eps*L = {2*RHO_FIELD*EPS*AREA:.0f} J'
          f'  (one-way {RHO_FIELD*AREA:.0f} m)')
    print('    If E_exc ~ 3,900 J the fitted e_min IS the excursion term: structure')
    print('    confirmed, and d_1way is then the single geometric quantity to derive.')

    # ---- [B] hop constant --------------------------------------------------
    print('\n[B] HOP CONSTANT   hop = c*L/sqrt(M),  c from timing = %.3f' % C_TIMING)
    print(f'    {"M":>4} {"hop meas":>9} {"hop pred":>9} {"err":>7} {"c implied":>10}')
    for M in g.M:
        v = [c['hop'] for c in chains if c['M'] == M and np.isfinite(c['hop'])]
        if not v:
            continue
        h = float(np.mean(v)); hp = C_TIMING * AREA / np.sqrt(M)
        print(f'    {M:>4} {h:>9.1f} {hp:>9.1f} {(h/hp-1)*100:>6.1f}%'
              f' {h*np.sqrt(M)/AREA:>10.3f}')
    print('    Timing channel (r_cap) and energy channel (tour length) should agree.')

    # ---- [C] clustered-tour test ------------------------------------------
    print('\n[C] CLUSTERED-TOUR TEST   hop vs n at fixed M   *** THE LOAD-BEARING STEP ***')
    print(f'    {"M":>4} {"n range":>12} {"slope d(hop)/dn":>16} {"R2":>7}'
          f' {"log-log exp":>12} {"verdict":>22}')
    for M in g.M:
        v = [(c['n'], c['hop']) for c in chains
             if c['M'] == M and np.isfinite(c['hop']) and c['n'] >= 3]
        if len(v) < 6:
            print(f'    {M:>4}  too few chains'); continue
        n = np.array([x[0] for x in v], float); h = np.array([x[1] for x in v], float)
        sl, ic, r2 = lsq(n, h)
        ex, _, _ = lsq(np.log(n), np.log(h))
        rel = sl * np.mean(n) / np.mean(h)
        verdict = ('CLUSTERED (hop flat)' if abs(ex) < 0.15
                   else ('BHH-like (hop ~ n^-0.5)' if ex < -0.35 else 'AMBIGUOUS'))
        print(f'    {M:>4} {f"{n.min():.0f}-{n.max():.0f}":>12} {sl:>16.3f} {r2:>7.4f}'
              f' {ex:>12.3f} {verdict:>22}')
    print('    exponent ~  0.0  -> swept area grows with n; r(e) linear; derivation holds')
    print('    exponent ~ -0.5  -> textbook BHH; the measured linearity is unexplained')

    # ---- write -------------------------------------------------------------
    f1 = os.path.join(g.out_dir, 'chain_geometry.csv')
    f2 = os.path.join(g.out_dir, 'cell_geometry.csv')
    with open(f1, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(chains[0].keys())); w.writeheader(); w.writerows(chains)
    with open(f2, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys())); w.writeheader(); w.writerows(cells)
    print(f'\n  wrote {f1}  ({len(chains)} chains)')
    print(f'  wrote {f2}  ({len(cells)} cells)')
    print(L)


if __name__ == '__main__':
    main()
