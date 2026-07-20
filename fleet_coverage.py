"""
fleet_coverage.py  --  is fleet coverage really K * r(Emax/K)?
===============================================================
The K* derivation assumes N(K) = min(M, K * r(Emax/K)) (Eq. 31): K UAVs each
achieve the SINGLE-UAV reach envelope r(.). Using measured primitives
(a, e_min from kstar_primitives.py) that predictor gives K*(50k) = 6/7/10 for
M = 50/100/200, whereas the SA sweep measures K* = 4 flat. So Eq. 31 is wrong in
a way that matters -- almost certainly because UAVs COMPETE for the same nodes
under the partition constraint (Eq. 16): each extra UAV serves progressively
worse marginal nodes, so fleet coverage is BELOW K * r(e).

This script measures the truth and quantifies the gap:

    N_meas(K)   nodes actually served by the SA fleet solution
    N_pred(K)   = min(M, K * a*(Emax/K - e_min))   using measured a, e_min
    eta(K)      = N_meas / N_pred        <-- the competition-efficiency factor

If eta(K) is a clean decreasing function of K (e.g. eta ~ K^-gamma), substitute
    N(K) = eta(K) * K * r(Emax/K)
back into J(K) = c1 N^2/K - p1 N + s(K-1) and re-derive e*. That converts the
failed predictor into a corrected one -- and the correction itself is a modeling
contribution ("fleet coverage is not K times single-UAV coverage").

The script fits both a power law eta = A*K^-gamma and a linear-in-K form, and
reports which describes the data better, plus the implied K* under the corrected
model (compare against kstar_sa.csv).

Primitives default to the measured values from kstar_primitives.py; override if
you re-measure.

Run (uses the same shared-seed instances as every other table):
    python fleet_coverage.py --M 50 100 200 --Emax 50000 --K 1 2 3 4 5 6 ^
        --instances 8 --iters 1500

Wider (to see eta across energy budgets too):
    python fleet_coverage.py --M 100 --Emax 25000 50000 100000 --K 1 2 3 4 5 6 8 10
"""
import os, csv, argparse
import numpy as np

from compare_baseline import gen, INSTANCE_SEED
from sa_routes import sa_best_with_routes

# measured single-UAV primitives (kstar_primitives.py, saturation-corrected fit)
PRIM = {50:  dict(a=0.00094956, e_min=730.0),
        100: dict(a=0.00136285, e_min=928.0),
        200: dict(a=0.00171,    e_min=514.0)}


def r_single(e, a, e_min):
    """Single-UAV reach envelope, Eq. 33 (clipped at 0)."""
    return max(0.0, a * (e - e_min))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, nargs='+', default=[50, 100, 200])
    ap.add_argument('--Emax', type=float, nargs='+', default=[50000.0])
    ap.add_argument('--K', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--instances', type=int, default=8)
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--restarts', type=int, default=1)
    ap.add_argument('--seed', type=int, default=INSTANCE_SEED)
    ap.add_argument('--out-dir', default='fleet_coverage')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    rng0 = np.random.default_rng(a.seed)
    seeds = [int(rng0.integers(0, 1e7)) for _ in range(a.instances)]

    print('=' * 88)
    print(f'  Fleet coverage vs K*r(e)  |  M={a.M}  Emax={[int(x) for x in a.Emax]}  K={a.K}')
    print(f'  instances={a.instances} iters={a.iters} seed={a.seed}')
    print('=' * 88)

    rows = []
    for M in a.M:
        prim = PRIM.get(M)
        if prim is None:
            print(f'  !! no measured primitives for M={M}; skipping')
            continue
        for E in a.Emax:
            print(f'\nM={M} Emax={int(E)}   (a={prim["a"]:.5g}, e_min={prim["e_min"]:.0f})')
            print(f'  {"K":>2} {"Ee":>8} {"N_meas":>8} {"N_pred":>8} {"eta":>6} {"obj":>10}')
            for K in a.K:
                Ee = E / K
                Ns, objs = [], []
                for s in seeds:
                    pos, wi, tcd = gen(M, s)
                    J, trajs = sa_best_with_routes(pos, wi, tcd, K, Ee, M,
                                                   a.iters, a.restarts)
                    Ns.append(sum(len(t) for t in trajs))
                    objs.append(J)
                N_meas = float(np.mean(Ns))
                N_pred = min(float(M), K * r_single(Ee, prim['a'], prim['e_min']))
                eta = N_meas / N_pred if N_pred > 1e-9 else float('nan')
                rows.append(dict(M=M, Emax=int(E), K=K, Ee=round(Ee, 1),
                                 N_meas=round(N_meas, 3),
                                 N_std=round(float(np.std(Ns)), 3),
                                 N_pred=round(N_pred, 3), eta=round(eta, 4),
                                 obj=round(float(np.mean(objs)), 3),
                                 instances=a.instances))
                print(f'  {K:>2} {Ee:>8.0f} {N_meas:>8.2f} {N_pred:>8.2f} '
                      f'{eta:>6.3f} {np.mean(objs):>10.2f}')

    out = os.path.join(a.out_dir, 'fleet_coverage.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ── fit the competition correction eta(K) ────────────────────────────────
    print('\n' + '=' * 88)
    print('  COMPETITION CORRECTION  eta(K) = N_meas / (K * r(Emax/K))')
    print('  (fit only where N_pred < M, i.e. not coverage-clipped)')
    for M in a.M:
        for E in a.Emax:
            sub = [r for r in rows if r['M'] == M and r['Emax'] == int(E)
                   and r['N_pred'] < M - 1e-6 and r['eta'] == r['eta']]
            if len(sub) < 3:
                continue
            K = np.array([r['K'] for r in sub], float)
            et = np.array([r['eta'] for r in sub], float)
            # power law: log eta = log A - gamma log K
            cp = np.polyfit(np.log(K), np.log(et), 1)
            gamma, logA = -cp[0], cp[1]
            pred_p = np.exp(logA) * K ** (-gamma)
            r2p = 1 - ((et - pred_p) ** 2).sum() / max(((et - et.mean()) ** 2).sum(), 1e-12)
            # linear: eta = b0 + b1*K
            cl = np.polyfit(K, et, 1)
            pred_l = np.polyval(cl, K)
            r2l = 1 - ((et - pred_l) ** 2).sum() / max(((et - et.mean()) ** 2).sum(), 1e-12)
            better = 'power' if r2p >= r2l else 'linear'
            print(f'\n  M={M} Emax={int(E)}  [{len(sub)} unclipped pts]')
            print(f'    power : eta = {np.exp(logA):.3f} * K^-{gamma:.3f}   R2={r2p:.4f}')
            print(f'    linear: eta = {cl[1]:.3f} + {cl[0]:.4f}*K          R2={r2l:.4f}')
            print(f'    -> {better} fits better')
            if better == 'power' and r2p > 0.8:
                # corrected coverage N = A K^(1-gamma) * a (E/K - e_min)
                # => effective per-UAV energy constant shifts; report implied K*
                print(f'    corrected model: N(K) = {np.exp(logA):.3f} K^(1-{gamma:.3f}) '
                      f'* a(Emax/K - e_min)')

    print(f'\n  wrote {out}')
    print('  NEXT: if eta(K) fits cleanly, substitute N(K)=eta(K)*K*r(Emax/K) into')
    print('        J(K)=c1 N^2/K - p1 N + s(K-1), re-solve dJ/dK=0, and compare the')
    print('        new K* against kstar_sa.csv (2/4/8 at 25k/50k/100k).')
    print('=' * 88)


if __name__ == '__main__':
    main()
