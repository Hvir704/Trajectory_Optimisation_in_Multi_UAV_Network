"""
sa_repair.py  --  unfreeze SA at cap-packed cells, then test whether K* MOVES.
==============================================================================
THE PROBLEM (from CONTEXT_SA_freeze.md + the exact_dp measurements):

  `greedy_init` packs chains to 98-100% of Ee. `compare_baseline.sa` tests
  `feasible(nt,...)` BEFORE scoring, so at a cap-packed solution:
    op0 insert / op2 relocate / op4 swap-in  -> raise energy -> rejected
    op3 segment reversal                     -> energy-neutral, usually no gain
    op1 remove                               -> always feasible, always uphill
  The accessible neighbourhood is empty or uphill. SA returns greedy VERBATIM
  (30/30 at M=100,K=4; 12/12 at M=200,K=4).

WHY IT MATTERS (measured by exact_dp.py, certified against exact optima):

  | fill        | greedy alone | SA    |
  |-------------|--------------|-------|
  | <0.85       | 0-5%         | 0-0.7%|
  | 0.90-0.96   | 2-10%        | 0-4.6%|
  | 0.983 (M=16)| **10.9%**    | 1.26% |
  K=2 partitioning, Ee=9000: greedy 7.92%, SA **0.00%** (exact).

  So greedy degrades with fill and SA recovers it -- but only where SA actually
  searches. At M>=100 it does not, and that is the HIGHEST-fill regime.

THE FIX. One new move: **energy-repair swap** = remove-then-insert as a SINGLE
atomic move, so the intermediate infeasible state is never tested. Two variants:
  op5  eject-insert : drop a served node, insert an unserved one elsewhere
  op6  ejection-chain: move node A out of chain k, node B from another chain into k
Both can lower energy first and raise value after, which is precisely what the
current move set cannot express.

WHAT THIS SCRIPT DOES. It does NOT replace the banked pipeline. It runs
baseline-SA and repaired-SA side by side over a K sweep at M>=100 and reports
whether the ARGMIN MOVES. That is the only thing that would invalidate K*.

  - argmin unchanged -> report the freeze as a documented limitation WITH
    evidence it does not affect K*. Every banked number stands.
  - argmin moves     -> important finding; re-run the grid with repaired SA.

Run (start small -- this is the decision experiment, not a grid):
    python sa_repair.py --M 100 --K 3 4 5 --Emax 50000 --instances 8 --iters 4000
    python sa_repair.py --M 200 --K 3 4 5 6 --Emax 50000 --instances 8 --iters 6000
"""
import os, csv, time, argparse
import numpy as np

from compare_baseline import (gen, greedy_init, fleet_obj, feasible,
                              chain_energy, EMAX, INSTANCE_SEED)
from compare_baseline import sa as sa_baseline


def sa_repaired(pos, wi, tcd, K, Ee, M, iters, seed, repair_p=0.35):
    """compare_baseline.sa + two atomic energy-repair moves (op5, op6).

    Identical structure, schedule, and acceptance rule to the baseline; the ONLY
    change is two extra move types that combine a removal with an insertion so
    the infeasible intermediate is never feasibility-tested. `repair_p` is the
    probability of drawing a repair move instead of the original five."""
    rng = np.random.default_rng(seed)
    trajs, served = greedy_init(pos, wi, tcd, K, Ee, M)
    cur = fleet_obj(trajs, pos, wi, tcd)
    best = cur; best_tr = [t[:] for t in trajs]
    T0, T1 = abs(cur) * 0.05 + 1e-3, 1e-4

    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / max(iters - 1, 1))
        nt = [t[:] for t in trajs]
        uns = [j for j in range(M) if j not in served]

        if rng.random() < repair_p and uns and any(nt):
            if rng.random() < 0.5:
                # op5 eject-insert: drop one served node, insert an unserved one
                k = int(rng.choice([i for i in range(K) if nt[i]]))
                i = int(rng.integers(0, len(nt[k])))
                del nt[k][i]
                j = int(rng.choice(uns))
                k2 = int(rng.choice([i for i in range(K)]))
                p = int(rng.integers(0, len(nt[k2]) + 1))
                nt[k2] = nt[k2][:p] + [j] + nt[k2][p:]
            else:
                # op6 ejection chain: swap one node out of k for one out of k2
                nonempty = [i for i in range(K) if nt[i]]
                if len(nonempty) < 2:
                    continue
                k, k2 = rng.choice(nonempty, 2, replace=False)
                k, k2 = int(k), int(k2)
                i1 = int(rng.integers(0, len(nt[k])))
                i2 = int(rng.integers(0, len(nt[k2])))
                a, b = nt[k][i1], nt[k2][i2]
                del nt[k][i1]
                del nt[k2][i2]
                p1 = int(rng.integers(0, len(nt[k]) + 1))
                p2 = int(rng.integers(0, len(nt[k2]) + 1))
                nt[k] = nt[k][:p1] + [b] + nt[k][p1:]
                nt[k2] = nt[k2][:p2] + [a] + nt[k2][p2:]
        else:
            op = rng.integers(0, 5)
            if op == 0 and uns:
                j = int(rng.choice(uns)); k = int(rng.integers(0, K))
                p = int(rng.integers(0, len(nt[k]) + 1))
                nt[k] = nt[k][:p] + [j] + nt[k][p:]
            elif op == 1 and any(nt):
                k = int(rng.choice([i for i in range(K) if nt[i]]))
                i = int(rng.integers(0, len(nt[k]))); del nt[k][i]
            elif op == 2 and any(nt):
                k = int(rng.choice([i for i in range(K) if nt[i]]))
                i = int(rng.integers(0, len(nt[k]))); j = nt[k][i]; del nt[k][i]
                k2 = int(rng.integers(0, K)); p = int(rng.integers(0, len(nt[k2]) + 1))
                nt[k2] = nt[k2][:p] + [j] + nt[k2][p:]
            elif op == 3 and any(len(t) >= 2 for t in nt):
                k = int(rng.choice([i for i in range(K) if len(nt[i]) >= 2]))
                a2, b2 = sorted(rng.choice(len(nt[k]), 2, replace=False))
                nt[k][a2:b2 + 1] = nt[k][a2:b2 + 1][::-1]
            elif op == 4 and uns and any(nt):
                k = int(rng.choice([i for i in range(K) if nt[i]]))
                i = int(rng.integers(0, len(nt[k])))
                nt[k][i] = int(rng.choice(uns))

        if not feasible(nt, pos, tcd, Ee):
            continue
        o = fleet_obj(nt, pos, wi, tcd); d = o - cur
        if d < 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            trajs = nt; cur = o; served = set(x for t in nt for x in t)
            if o < best:
                best = o; best_tr = [t[:] for t in nt]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--K', type=int, nargs='+', default=[3, 4, 5])
    ap.add_argument('--Emax', type=float, default=EMAX)
    ap.add_argument('--instances', type=int, default=8)
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--restarts', type=int, default=2)
    ap.add_argument('--repair-p', type=float, default=0.35)
    ap.add_argument('--seed', type=int, default=INSTANCE_SEED)
    ap.add_argument('--out-dir', default='sa_repair')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    seeds = [int(rng.integers(0, 10_000_000)) for _ in range(a.instances)]
    M = a.M

    print('=' * 96)
    print(f'  SA repair vs baseline | M={M} Emax={a.Emax:.0f} K={a.K}')
    print(f'  instances={a.instances} (meta-seed {a.seed}) iters={a.iters} '
          f'restarts={a.restarts} repair_p={a.repair_p}')
    print('=' * 96)
    print(f'{"K":>2} {"Ee":>8} {"greedy":>10} {"SA_base":>10} {"SA_rep":>10} '
          f'{"base-frz":>9} {"rep-frz":>8} {"gain%":>7} {"sec":>6}')

    rows = []
    for K in a.K:
        Ee = a.Emax / K
        g, sb, sr, fz_b, fz_r = [], [], [], 0, 0
        t0 = time.time()
        for si, s in enumerate(seeds):
            pos, wi, tcd = gen(M, s)
            gtr, _ = greedy_init(pos, wi, tcd, K, Ee, M)
            Jg = fleet_obj(gtr, pos, wi, tcd)
            Jb = min(sa_baseline(pos, wi, tcd, K, Ee, M, a.iters, r)
                     for r in range(a.restarts))
            Jr = min(sa_repaired(pos, wi, tcd, K, Ee, M, a.iters, r, a.repair_p)
                     for r in range(a.restarts))
            g.append(Jg); sb.append(Jb); sr.append(Jr)
            fz_b += int(abs(Jb - Jg) < 1e-9)
            fz_r += int(abs(Jr - Jg) < 1e-9)
        dt = time.time() - t0
        G, B, R = float(np.mean(g)), float(np.mean(sb)), float(np.mean(sr))
        gain = 100 * (B - R) / abs(B) if abs(B) > 1e-9 else 0.0
        rows.append(dict(M=M, K=K, Emax=a.Emax, Ee=round(Ee, 1),
                         greedy=round(G, 4), SA_base=round(B, 4),
                         SA_repair=round(R, 4),
                         base_frozen=fz_b, repair_frozen=fz_r,
                         repair_gain_pct=round(gain, 3),
                         instances=a.instances, iters=a.iters, sec=round(dt, 1)))
        print(f'{K:>2} {Ee:>8.0f} {G:>10.3f} {B:>10.3f} {R:>10.3f} '
              f'{fz_b:>4}/{a.instances:<4} {fz_r:>3}/{a.instances:<4} '
              f'{gain:>7.2f} {dt:>6.1f}')

    out = os.path.join(a.out_dir, f'sa_repair_M{M}_E{int(a.Emax)}.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    kb = min(rows, key=lambda r: r['SA_base'])['K']
    kr = min(rows, key=lambda r: r['SA_repair'])['K']
    print('\n' + '=' * 96)
    print(f'  argmin K  baseline = {kb}   repaired = {kr}')
    if kb == kr:
        print('  => K* DOES NOT MOVE. Report the freeze as a documented limitation')
        print('     WITH this evidence that it does not affect the argmin.')
        print('     Every banked number stands. This is the good outcome.')
    else:
        print('  => K* MOVES. Important finding. The banked grid was computed with a')
        print('     frozen solver at these cells; re-run the K* grid with repaired SA')
        print('     before claiming the law. Do this NOW, not near the deadline.')
    print(f'\n  freeze counts: baseline {sum(r["base_frozen"] for r in rows)}'
          f'/{len(rows)*a.instances}, repaired '
          f'{sum(r["repair_frozen"] for r in rows)}/{len(rows)*a.instances}')
    print('  (if repaired is still ~all frozen, the repair moves are not firing --')
    print('   raise --repair-p or --iters before drawing any conclusion)')
    print(f'  wrote {out}')
    print('=' * 96)


if __name__ == '__main__':
    main()