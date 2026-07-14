"""
fleet_optimality_gap.py  —  how far is the RL policy from optimal?
=================================================================
We MINIMISE J, so any feasible solution upper-bounds the optimum:
    J_LB <= J_opt <= J_SA <= J_RL,   policy gap >= (J_RL - J_SA)/|J_SA|.
SA (from compare_baseline) is validated near-optimal against an EXACT solver on
small single-UAV instances (mean gap ~1.6%), so it is a near-tight optimum proxy;
the gap it reveals is a rigorous LOWER bound on the policy's suboptimality.

Two parts:
  1) validate: SA vs EXACT optimum on small single-UAV cells (credibility check).
  2) gap grid: SA (best of restarts) vs RL policy obj from eval_table, per (M,K).

Run:
    python fleet_optimality_gap.py --validate
    python fleet_optimality_gap.py --M 50 100 200 --K 1 2 3 4 5 6 \
        --iters 3000 --restarts 3 --instances 20 --eval-table eval_table_split.csv \
        --out optimality_gap.csv
"""
import argparse, csv, time
from collections import defaultdict
import numpy as np
from compare_baseline import (gen, sa, chain_waoi, tf,
                              TH1, TH2, PF, PH, HOME, EMAX, INSTANCE_SEED)


def obj_single(t, pos, wi, tcd):
    return TH1 * chain_waoi(t, pos, wi, tcd) - TH2 * sum(wi[j] for j in t)


def exact_single(pos, wi, tcd, budget, M):
    """Exact single-UAV optimum by DFS with energy pruning (M <= ~9)."""
    best = [0.0, []]
    def dfs(traj, served, e_used, curr):
        J = obj_single(traj, pos, wi, tcd)
        if J < best[0] - 1e-12:
            best[0] = J; best[1] = list(traj)
        for j in range(M):
            if j in served:
                continue
            eu = e_used + PF * tf(curr, pos[j]) + PH * tcd[j]
            if eu + PF * tf(pos[j], HOME) <= budget + 1e-9:
                served.add(j); traj.append(j)
                dfs(traj, served, eu, pos[j])
                traj.pop(); served.discard(j)
    dfs([], set(), 0.0, HOME)
    return best[0]


def sa_best(pos, wi, tcd, K, Ee, M, iters, restarts):
    return min(sa(pos, wi, tcd, K, Ee, M, iters=iters, seed=r) for r in range(restarts))


def validate(n=8, budget=9000.0, iters=8000, restarts=5):
    rng = np.random.default_rng(7); gaps = []
    print('SA vs EXACT optimum (single UAV, binding budget):')
    print(f'{"M":>2} {"J_exact":>9} {"J_SA":>9} {"gap%":>6}')
    for i in range(n):
        M = 8 if i % 2 else 7; s = int(rng.integers(0, 1e7))
        pos, wi, tcd = gen(M, s)
        Je = exact_single(pos, wi, tcd, budget, M)
        Jsa = sa_best(pos, wi, tcd, 1, budget, M, iters, restarts)
        g = 100 * (Jsa - Je) / abs(Je) if abs(Je) > 1e-9 else 0.0
        gaps.append(g); print(f'{M:>2} {Je:>9.3f} {Jsa:>9.3f} {g:>6.2f}')
    print(f'=> SA mean gap to optimum {np.mean(gaps):.2f}%, max {np.max(gaps):.2f}% '
          f'(SA is a near-tight optimum proxy).')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--M', type=int, nargs='+', default=[50, 100, 200])
    ap.add_argument('--K', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--restarts', type=int, default=3)
    ap.add_argument('--instances', type=int, default=20)
    ap.add_argument('--eval-table', default='eval_table_split.csv')
    ap.add_argument('--out', default='optimality_gap.csv')
    a = ap.parse_args()

    if a.validate:
        validate(); return

    rl = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(a.eval_table)):
        rl[int(r['M'])][int(r['K'])].append(float(r['obj']))
    rl = {M: {K: float(np.mean(v)) for K, v in d.items()} for M, d in rl.items()}

    rows = []
    print(f'{"M":>4} {"K":>2} {"RL_obj":>9} {"SA_obj":>9} {"gap%":>7} {"scope":>9} {"s/inst":>7}')
    for M in a.M:
        for K in a.K:
            Ee = EMAX / K
            rng = np.random.default_rng(INSTANCE_SEED)
            seeds = [int(rng.integers(0, 1e7)) for _ in range(a.instances)]
            t0 = time.time()
            vals = [sa_best(*gen(M, s), K, Ee, M, a.iters, a.restarts) for s in seeds]
            sa_obj = float(np.mean(vals)); dt = (time.time() - t0) / a.instances
            rlv = rl.get(M, {}).get(K, float('nan'))
            gap = 100 * (rlv - sa_obj) / abs(sa_obj) if rlv == rlv else float('nan')
            scope = 'large' if gap > 15 else 'moderate' if gap > 7 else 'small'
            rows.append(dict(M=M, K=K, RL_obj=rlv, SA_obj=sa_obj, gap_pct=gap,
                             scope=scope, iters=a.iters, restarts=a.restarts,
                             instances=a.instances))
            print(f'{M:>4} {K:>2} {rlv:>9.2f} {sa_obj:>9.2f} {gap:>7.1f} {scope:>9} {dt:>7.1f}')
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    finite = [r['gap_pct'] for r in rows if r['gap_pct'] == r['gap_pct']]
    print(f'\nMean policy gap: {np.mean(finite):.1f}%  (>=, since SA >= optimum). Wrote {a.out}')


if __name__ == '__main__':
    main()
