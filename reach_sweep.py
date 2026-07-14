"""
reach_sweep.py  —  single-UAV reach envelope r(e, M) over a WIDE budget range.
=============================================================================
Characterises how many nodes ONE UAV can reach as a function of its energy
budget e, across e in [1k, 1M] J. This is competition-free and policy-free
(a nearest-neighbour tour under the budget), so it is Emax-general and cheap:
no fleet training, no torch, no saved models.

It writes reach_curve.csv (e, M, r) into --out-dir and fits the density-law
envelope  r(e,M) = clip(kappa*M^beta*(e - emin), 0, M), reporting kappa, beta,
emin. The near-constant a(M)/sqrt(M) is the evidence that the coverage term
generalises across M and Emax.

Run:
    python reach_sweep.py --out-dir kstar_ME_runs
"""

import os, csv, argparse
import numpy as np

AREA, V, PH, PF = 1000.0, 20.0, 200.0, 150.0
Ps, k0, s2, Wbw = 0.1, 1e-3, 1e-14, 1e6
Dlo, Dhi = 0.5e6, 5.0e6
R = Wbw * np.log2(1 + k0 * Ps / (100.0**2 * s2))
HOME = np.array([500.0, 500.0])


def gen(M, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, AREA, (M, 2)), rng.uniform(Dlo, Dhi, M) / R


def nn_reach(pos, tcd, budget):
    M = len(pos); vis = np.zeros(M, bool); cur = HOME.copy(); E = budget; n = 0
    while True:
        d = np.linalg.norm(pos - cur, axis=1); dh = np.linalg.norm(pos - HOME, axis=1)
        need = PF * (d + dh) / V + PH * tcd
        feas = (~vis) & (need <= E)
        if not feas.any():
            break
        dd = d.copy(); dd[~feas] = np.inf; j = int(np.argmin(dd))
        E -= PF * d[j] / V + PH * tcd[j]; cur = pos[j].copy(); vis[j] = True; n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, nargs='+', default=[50, 100, 200, 400])
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--out-dir', default='kstar_ME_runs')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    budgets = [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]

    rows, tab = [], {}
    print(f'{"e(J)":>9} ' + ' '.join(f'M{m}'.rjust(7) for m in a.M))
    for e in budgets:
        line = []
        for M in a.M:
            r = float(np.mean([nn_reach(*gen(M, s), e) for s in range(a.seeds)]))
            tab[(e, M)] = r; line.append(r)
            rows.append(dict(e=e, M=M, r=round(r, 3)))
        print(f'{e:9.0f} ' + ' '.join(f'{v:7.1f}' for v in line))

    out = os.path.join(a.out_dir, 'reach_curve.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['e', 'M', 'r']); w.writeheader(); w.writerows(rows)

    print('\ndensity-law envelope r(e,M)=clip(kappa*M^beta*(e-emin),0,M):')
    slopes = []
    for M in a.M:
        es = np.array([2e3, 5e3, 1e4, 2e4]); rs = np.array([tab[(e, M)] for e in es])
        (a_, b_) = np.polyfit(es, rs, 1); emin = -b_ / a_
        slopes.append((M, a_, emin))
        print(f'  M={M:>4}: a={a_:.6f}  emin={emin:6.0f}  a/sqrt(M)={a_/np.sqrt(M):.6f}')
    Ms = np.array([m for m, _, _ in slopes]); aa = np.array([s for _, s, _ in slopes])
    beta, lk = np.polyfit(np.log(Ms), np.log(aa), 1)
    print(f'  => kappa={np.exp(lk):.3e}, beta={beta:.3f}, '
          f'emin~{np.median([e for _,_,e in slopes]):.0f} J  (beta~0.5 confirms sqrt-density law)')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
