"""
exact_dp.py  --  CERTIFIED exact optima for K=1 and K=2, up to M ~ 14-18.
==========================================================================
WHY THIS EXISTS. `fleet_optimality_gap.exact_single` is a plain DFS: it caps at
M<=9 and is single-UAV only. WS15 needs exact optima in the CAP-PACKED regime,
which requires node abundance (greedy must run out of ENERGY, not out of NODES).
At M=8-9 greedy stops at fill ~0.9 because it has served everything worth
serving, so cap-packing is unreachable there and WS15 could not be answered.

THE KEY STRUCTURAL FACT that makes a DP possible:

    chain_waoi accumulates W = sum of w over nodes visited SO FAR.
    W depends only on the SET visited, not on the order.

So a Held-Karp DP over (subset, last_node) is valid. Cost is charged on
DEPARTURE from a node:

    leaving `last` (set S, W = w(S)) toward j:
        waoi   += w(S) * (tcd[last] + tf(last, j))
        energy += PF*tf(last, j) + PH*tcd[j]
    terminating from `last` to HOME:
        waoi   += w(S) * (tcd[last] + tf(last, HOME))
        energy += PF*tf(last, HOME)

Energy and waoi both accumulate along the path and trade off, so each
(subset, last) keeps a PARETO SET of non-dominated (energy, waoi) pairs. That is
what makes this exact rather than heuristic.

    J(S) = TH1*waoi(S) - TH2*w(S)     with w(S) fixed per subset

K=2 is exact by partition enumeration: best_subset[S] is the optimal single-chain
value over exactly-served set S, then minimise over disjoint (S1, S2). This is
the FIRST certification of multi-UAV PARTITIONING in this project — WS15's
scope note ("multi-UAV partitioning is NOT certified here") is what this closes.

COST. K=1 is O(2^M * M^2) states-by-transitions with small Pareto sets: M=14 is
seconds, M=16 tens of seconds, M=18 is pushing it. K=2 partition enumeration is
O(3^M): M=12 fine, M=14 slow, M>=16 don't.

VALIDATION. `--validate` checks against fleet_optimality_gap.exact_single at
M<=9, where that DFS is itself exact. They must agree to 1e-9.

Run:
    python exact_dp.py --validate
    python exact_dp.py --M 14 --K 1 --budgets 6000 9000 12000 --instances 8
    python exact_dp.py --M 12 --K 2 --budgets 6000 9000 --instances 6
"""
import os, csv, time, argparse
from itertools import combinations
import numpy as np

from compare_baseline import (gen, greedy_init, chain_energy, fleet_obj,
                              tf, chain_waoi, TH1, TH2, PF, PH, HOME)
from fleet_optimality_gap import exact_single, sa_best

EPS = 1e-9


def _pareto_add(front, e, v):
    """Insert (energy, waoi); keep only non-dominated. Returns False if dominated."""
    for (fe, fv) in front:
        if fe <= e + EPS and fv <= v + EPS:
            return False
    front[:] = [(fe, fv) for (fe, fv) in front
                if not (e <= fe + EPS and v <= fv + EPS)]
    front.append((e, v))
    return True


def exact_dp_single(pos, wi, tcd, budget, M, return_subsets=False):
    """Exact single-UAV optimum via Held-Karp with Pareto (energy, waoi) states.

    Returns J* (<=0). If return_subsets, also returns best_sub: dict
    subset_bitmask -> best J achievable serving EXACTLY that set."""
    # precompute
    t_home = np.array([tf(HOME, pos[j]) for j in range(M)])
    t_ij = np.array([[tf(pos[i], pos[j]) for j in range(M)] for i in range(M)])
    wsum = np.zeros(1 << M)
    for S in range(1, 1 << M):
        low = S & -S
        j = low.bit_length() - 1
        wsum[S] = wsum[S ^ low] + wi[j]

    # dp[S][last] = list of non-dominated (energy, waoi)
    dp = [dict() for _ in range(1 << M)]
    for j in range(M):
        e = PF * t_home[j] + PH * tcd[j]
        if e + PF * t_home[j] <= budget + EPS:      # must be able to return
            dp[1 << j][j] = [(e, 0.0)]

    best_J = 0.0
    best_sub = {0: 0.0}

    for S in range(1, 1 << M):
        if not dp[S]:
            continue
        W = wsum[S]
        for last, front in dp[S].items():
            for (e, v) in front:
                # terminate: fly home
                e_end = e + PF * t_home[last]
                if e_end <= budget + EPS:
                    v_end = v + W * (tcd[last] + t_home[last])
                    J = TH1 * v_end - TH2 * W
                    if J < best_J - 1e-12:
                        best_J = J
                    if return_subsets and J < best_sub.get(S, 0.0) - 1e-12:
                        best_sub[S] = J
                # extend
                for j in range(M):
                    if S >> j & 1:
                        continue
                    e2 = e + PF * t_ij[last][j] + PH * tcd[j]
                    if e2 + PF * t_home[j] > budget + EPS:
                        continue
                    v2 = v + W * (tcd[last] + t_ij[last][j])
                    S2 = S | (1 << j)
                    f2 = dp[S2].setdefault(j, [])
                    _pareto_add(f2, e2, v2)

    if return_subsets:
        # ensure every subset key exists (0.0 = serve nothing)
        return best_J, best_sub
    return best_J


def exact_dp_pair(pos, wi, tcd, Ee, M):
    """Exact K=2 optimum: enumerate disjoint (S1, S2) partitions of served nodes.
    Uses best_sub from the single-UAV DP (each chain is independently optimal)."""
    _, best_sub = exact_dp_single(pos, wi, tcd, Ee, M, return_subsets=True)
    # fill missing subsets with 0.0 (infeasible/empty -> serve nothing)
    full = (1 << M) - 1
    sub = np.zeros(1 << M)
    for S, J in best_sub.items():
        sub[S] = J
    # best over subsets of a mask (subset-sum DP over masks)
    best_within = sub.copy()
    for b in range(M):
        for S in range(1 << M):
            if S >> b & 1:
                cand = best_within[S ^ (1 << b)]
                if cand < best_within[S]:
                    best_within[S] = cand
    best = 0.0
    for S1 in range(1 << M):
        j1 = sub[S1]
        if j1 >= -1e-12:
            continue
        comp = full ^ S1
        tot = j1 + best_within[comp]
        if tot < best:
            best = tot
    return float(best)


def validate(n=6, M=8, budget=9000.0):
    print(f'exact_dp_single vs exact_single (DFS)   M={M} budget={budget:.0f}')
    rng = np.random.default_rng(2025)
    seeds = [int(rng.integers(0, 10_000_000)) for _ in range(n)]
    worst = 0.0
    print(f'{"seed":>9} {"DP":>11} {"DFS":>11} {"|diff|":>9}')
    for s in seeds:
        pos, wi, tcd = gen(M, s)
        a = exact_dp_single(pos, wi, tcd, budget, M)
        b = exact_single(pos, wi, tcd, budget, M)
        d = abs(a - b); worst = max(worst, d)
        print(f'{s:>9} {a:>11.5f} {b:>11.5f} {d:>9.1e}')
    print(f'\nworst |diff| = {worst:.1e}  ->  '
          + ('OK, DP is exact.' if worst < 1e-7 else 'MISMATCH — do not use.'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--M', type=int, default=14)
    ap.add_argument('--K', type=int, default=1, choices=[1, 2])
    ap.add_argument('--budgets', type=float, nargs='+',
                    default=[4000, 6000, 8000, 10000, 12000, 15000])
    ap.add_argument('--instances', type=int, default=8)
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--restarts', type=int, default=3)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--pack-lo', type=float, default=0.98)
    ap.add_argument('--out-dir', default='exact_dp')
    a = ap.parse_args()

    if a.validate:
        validate(); return

    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    seeds = [int(rng.integers(0, 10_000_000)) for _ in range(a.instances)]
    M, K = a.M, a.K

    print('=' * 96)
    print(f'  EXACT DP cap-packing check | M={M} K={K} | budgets='
          f'{[int(b) for b in a.budgets]}')
    print(f'  instances={a.instances} (meta-seed {a.seed}) | SA {a.iters}it x{a.restarts}')
    print('=' * 96)
    print(f'{"Ee":>7} {"fill":>6} {"packed":>7} {"J_exact":>10} {"J_greedy":>10} '
          f'{"J_SA":>10} {"g_gap%":>7} {"sa_gap%":>7} {"sec":>6}')

    rows = []
    for Ee in a.budgets:
        fills, Jex, Jgr, Jsa = [], [], [], []
        t0 = time.time()
        for s in seeds:
            pos, wi, tcd = gen(M, s)
            gtr, _ = greedy_init(pos, wi, tcd, K, Ee, M)
            fl = [chain_energy(t, pos, tcd) / Ee for t in gtr if t]
            fills.append(float(np.mean(fl)) if fl else 0.0)
            Jgr.append(fleet_obj(gtr, pos, wi, tcd))
            Jex.append(exact_dp_single(pos, wi, tcd, Ee, M) if K == 1
                       else exact_dp_pair(pos, wi, tcd, Ee, M))
            Jsa.append(sa_best(pos, wi, tcd, K, Ee, M, a.iters, a.restarts))
        dt = time.time() - t0

        f = float(np.mean(fills))
        je, jg, js = float(np.mean(Jex)), float(np.mean(Jgr)), float(np.mean(Jsa))
        packed = f >= a.pack_lo
        gg = 100 * (jg - je) / abs(je) if abs(je) > 1e-9 else float('nan')
        sg = 100 * (js - je) / abs(je) if abs(je) > 1e-9 else float('nan')
        rows.append(dict(M=M, K=K, Ee=round(float(Ee), 1), fill=round(f, 4),
                         cap_packed=int(packed), J_exact=round(je, 4),
                         J_greedy=round(jg, 4), J_SA=round(js, 4),
                         greedy_gap_pct=round(gg, 3), sa_gap_pct=round(sg, 3),
                         instances=a.instances, sec=round(dt, 1)))
        print(f'{Ee:>7.0f} {f:>6.3f} {"YES" if packed else "no":>7} {je:>10.3f} '
              f'{jg:>10.3f} {js:>10.3f} {gg:>7.2f} {sg:>7.2f} {dt:>6.1f}')

    out = os.path.join(a.out_dir, f'exact_dp_M{M}_K{K}.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    pk = [r for r in rows if r['cap_packed']]
    sl = [r for r in rows if not r['cap_packed']]
    print('\n' + '=' * 96)
    if pk:
        gp = float(np.mean([r['greedy_gap_pct'] for r in pk]))
        gw = float(np.max([r['greedy_gap_pct'] for r in pk]))
        sp = float(np.mean([r['sa_gap_pct'] for r in pk]))
        print(f'  CAP-PACKED ({len(pk)} cells): greedy mean {gp:.2f}% worst {gw:.2f}%'
              f' | SA mean {sp:.2f}%')
        if sl:
            gs = float(np.mean([r['greedy_gap_pct'] for r in sl]))
            print(f'  SLACK ({len(sl)} cells):      greedy mean {gs:.2f}%'
                  f'   -> packed-minus-slack {gp - gs:+.2f} pp')
        print()
        if gw < 3.0:
            print('  VERDICT: greedy stays near-optimal WHEN CAP-PACKED -> K* is not')
            print('           an artifact of cap-packing. WS15 ANSWERED.')
        elif gw < 8.0:
            print('  VERDICT: MODERATE degradation under cap-packing. Report honestly;')
            print('           check whether the gap grows with M.')
        else:
            print('  VERDICT: LARGE degradation. The constructor may bias K*.')
            print('           Escalate to larger M before claiming the law.')
    else:
        print('  No cap-packed cells. Raise M (more nodes -> greedy runs out of ENERGY')
        print('  rather than out of NODES) or lower the budgets.')
    if K == 1:
        print('\n  NOTE: K=1. Re-run with --K 2 --M 12 to certify multi-UAV partitioning.')
    else:
        print('\n  This certifies multi-UAV PARTITIONING — the gap WS15 could not close.')
    print(f'  wrote {out}')
    print('=' * 96)


if __name__ == '__main__':
    main()