"""
sa_routes.py  --  SA that returns TRAJECTORIES, not just the objective.
=======================================================================
compare_baseline.sa() returns only `best` (the objective float), so the routes
that achieved it are discarded -- which makes it impossible to run
deconfliction_schedule on SA solutions and therefore impossible to measure Phi.

This is a verbatim copy of compare_baseline.sa() with ONE change: it returns
(best_obj, best_trajs) instead of best_obj. The search itself is untouched --
same RNG draws, same move sequence, same acceptance -- so for a given seed it
produces exactly the same objective as the original. Verified by _verify() below.

Why a copy and not an edit: leaves compare_baseline.py (and every result already
produced by it) untouched.

Run the identity check:
    python sa_routes.py
"""
import numpy as np

from compare_baseline import (gen, greedy_init, fleet_obj, feasible,
                              sa as sa_orig, EMAX, INSTANCE_SEED)


def sa_with_routes(pos, wi, tcd, K, Ee, M, iters, seed):
    """Verbatim compare_baseline.sa(), but returns (best_obj, best_trajs)."""
    rng = np.random.default_rng(seed)
    trajs, served = greedy_init(pos, wi, tcd, K, Ee, M)
    cur = fleet_obj(trajs, pos, wi, tcd)
    best = cur
    best_tr = [t[:] for t in trajs]
    T0, T1 = abs(cur) * 0.05 + 1e-3, 1e-4
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / max(iters - 1, 1))
        nt = [t[:] for t in trajs]
        op = rng.integers(0, 5)
        uns = [j for j in range(M) if j not in served]
        if op == 0 and uns:
            j = int(rng.choice(uns)); k = int(rng.integers(0, K))
            p = int(rng.integers(0, len(nt[k]) + 1)); nt[k] = nt[k][:p] + [j] + nt[k][p:]
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
            a, b = sorted(rng.choice(len(nt[k]), 2, replace=False))
            nt[k][a:b + 1] = nt[k][a:b + 1][::-1]
        elif op == 4 and uns and any(nt):
            k = int(rng.choice([i for i in range(K) if nt[i]]))
            i = int(rng.integers(0, len(nt[k]))); nt[k][i] = int(rng.choice(uns))
        if not feasible(nt, pos, tcd, Ee):
            continue
        o = fleet_obj(nt, pos, wi, tcd); d = o - cur
        if d < 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            trajs = nt; cur = o; served = set(x for t in nt for x in t)
            if o < best:
                best = o; best_tr = [t[:] for t in nt]
    return best, best_tr


def sa_best_with_routes(pos, wi, tcd, K, Ee, M, iters, restarts):
    """Best-of-restarts, mirroring fleet_optimality_gap.sa_best."""
    out = [sa_with_routes(pos, wi, tcd, K, Ee, M, iters, r) for r in range(restarts)]
    return min(out, key=lambda t: t[0])


def _verify(n=8, M=100, K=4, iters=1500):
    """Same seed -> same objective as the original sa(). Also checks the returned
    routes actually evaluate to the returned objective (no stale best_tr)."""
    Ee = EMAX / K
    rng = np.random.default_rng(INSTANCE_SEED)
    seeds = [int(rng.integers(0, 1e7)) for _ in range(n)]
    print(f'sa_with_routes vs compare_baseline.sa   M={M} K={K} iters={iters}')
    print(f'{"seed":>9} {"NEW obj":>11} {"ORIG obj":>11} {"|diff|":>9} {"obj(trajs)":>11} {"ok":>4}')
    worst = 0.0; bad = 0
    for s in seeds:
        pos, wi, tcd = gen(M, s)
        o_new, tr = sa_with_routes(pos, wi, tcd, K, Ee, M, iters, seed=0)
        o_ref = sa_orig(pos, wi, tcd, K, Ee, M, iters, seed=0)
        recomputed = fleet_obj(tr, pos, wi, tcd)
        d = abs(o_new - o_ref); worst = max(worst, d)
        ok = (d < 1e-9) and (abs(recomputed - o_new) < 1e-9)
        bad += (0 if ok else 1)
        print(f'{s:>9} {o_new:>11.4f} {o_ref:>11.4f} {d:>9.1e} {recomputed:>11.4f} '
              f'{"OK" if ok else "BAD":>4}')
    print(f'\nworst |diff| = {worst:.1e}   failures: {bad}/{n}')
    print('OK: identical search, routes consistent.' if bad == 0 else
          'MISMATCH -- do not use until fixed.')


if __name__ == '__main__':
    _verify()
