"""
sa_anchor_sweep.py  --  K*(M, Emax) via SA (the near-optimal solver of record).
================================================================================
Replaces RL training for locating the optimal fleet size. SA is validated ~1.6%
from the exact optimum (see fleet_optimality_gap.py --validate), so the K* it
locates is the near-optimal K*, and where two K are within that margin we report
a BAND, i.e. rigorous bounds rather than a false-precision point.

WHY THIS IS Emax-GENERAL (your requirement)
--------------------------------------------
* Emax is a swept knob. Per-UAV split budget is Ee = Emax / K. Nothing here is
  tied to 50000 J -- the 50k grid is just one slice of the surface.
* We record E_per_node = Emax / M for every cell. If K* collapses onto a function
  of E_per_node (or another dimensionless group), K* generalises to ANY (M, Emax)
  via that scaling instead of a lookup over an arbitrary grid -- the strong,
  publishable form of the result. Check this in kstar_sa.csv before fitting a
  raw 2-D surface.
* Instances are shared-seed and drawn identically to fleet_optimality_gap.py, so
  at Emax=50000 the per-cell mean_obj REPRODUCES optimality_gap.csv's SA_obj -- a
  built-in cross-check that the pipeline is consistent.

OUTPUTS (into --out-dir, incremental + resumable)
    anchor_sa_grid.csv : one row per (M, Emax, K): Ee, mean_obj, std, sem, ...
    kstar_sa.csv       : one row per (M, Emax): Kstar, band (Kstar_lo..Kstar_hi),
                         E_per_node, per-K objectives. THIS is the frozen K*.

RESUME: re-running skips (M,Emax,K) cells already in anchor_sa_grid.csv, so you
can run in chunks, kill, and continue. kstar_sa.csv is recomputed from the full
grid after every cell, so it is always consistent with whatever has completed.

CONVENTION: K* is located on the base fleet objective that SA optimises
(routing/WAoI + priority, Phi=0 -- the 'intra' convention). The deconfliction
penalty is small and solver-invariant; to fold it in you must expose SA's routes
(sa() currently returns only the objective) and run deconfliction_schedule on
them -- ask and I'll wire that as an augmented-objective pass.

Run (PowerShell, one line). Start SMALL, widen once the surface shape is clear:
    python sa_anchor_sweep.py --M 50 100 200 --Emax 25000 50000 100000 ^
        --K 1 2 3 4 5 6 --iters 2000 --restarts 2 --instances 12

Full grid (WARNING: hours-to-days at M=200; run in chunks, it resumes):
    python sa_anchor_sweep.py --M 50 60 80 100 120 150 200 ^
        --Emax 10000 25000 50000 100000 200000 --K 1 2 3 4 5 6 8
"""
import os, csv, time, argparse
from collections import defaultdict
import numpy as np

from compare_baseline import gen, INSTANCE_SEED, EMAX
from fleet_optimality_gap import sa_best


def sa_cell(M, Emax, K, seeds, iters, restarts):
    Ee = Emax / K
    vals = np.array([sa_best(*gen(M, s), K, Ee, M, iters, restarts) for s in seeds])
    n = len(vals)
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    return Ee, float(vals.mean()), std, float(sem)


def compute_kstar(grid_rows, tol):
    """K* and its band per (M,Emax). Band = all K whose mean objective is within
    max(tol*|min|, 1 SEM) of the minimum -- i.e. statistically / near-optimally
    indistinguishable from best. That set IS the bound on K*."""
    by = defaultdict(dict)
    for r in grid_rows:
        by[(int(r['M']), int(r['Emax']))][int(r['K'])] = (
            float(r['mean_obj']), float(r['sem_obj']))
    out = []
    for (M, Emax), d in sorted(by.items()):
        Ks = sorted(d)
        means = {K: d[K][0] for K in Ks}
        Kmin = min(Ks, key=lambda K: means[K])        # most-negative obj = best
        mn = means[Kmin]
        margin = max(tol * abs(mn), d[Kmin][1])
        band = [K for K in Ks if means[K] <= mn + margin]
        row = dict(M=M, Emax=Emax, E_per_node=round(Emax / M, 2),
                   Kstar=Kmin, Kstar_lo=min(band), Kstar_hi=max(band),
                   band=f'{min(band)}-{max(band)}' if len(band) > 1 else str(Kmin),
                   n_in_band=len(band), min_obj=round(mn, 3), margin=round(margin, 3))
        for K in Ks:
            row[f'obj_K{K}'] = round(means[K], 2)
        out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    # union of keys (K columns can vary across (M,Emax))
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval='')
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, nargs='+', default=[50, 100, 200])
    ap.add_argument('--Emax', type=int, nargs='+', default=[25000, 50000, 100000])
    ap.add_argument('--K', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--restarts', type=int, default=2)
    ap.add_argument('--instances', type=int, default=12)
    ap.add_argument('--seed', type=int, default=INSTANCE_SEED)
    ap.add_argument('--tol', type=float, default=0.016,
                    help="rel. margin for the K* band (default = SA's ~1.6%% optimum gap)")
    ap.add_argument('--out-dir', default='kstar_sa')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    grid_path = os.path.join(a.out_dir, 'anchor_sa_grid.csv')
    kstar_path = os.path.join(a.out_dir, 'kstar_sa.csv')

    rng = np.random.default_rng(a.seed)
    seeds = [int(rng.integers(0, 1e7)) for _ in range(a.instances)]

    # resume: load any completed cells
    grid_rows = []
    done = set()
    if os.path.exists(grid_path):
        for r in csv.DictReader(open(grid_path)):
            grid_rows.append(r)
            done.add((int(r['M']), int(r['Emax']), int(r['K'])))
        print(f'resume: {len(done)} cells already done in {grid_path}')

    cells = [(M, E, K) for M in a.M for E in a.Emax for K in a.K
             if (M, E, K) not in done]
    total = len(cells)
    print('=' * 82)
    print(f'  SA anchor sweep | M={a.M} Emax={a.Emax} K={a.K}')
    print(f'  iters={a.iters} restarts={a.restarts} instances={a.instances} '
          f'seed={a.seed} | {total} cells to run')
    print(f'  band tol={a.tol} | cross-check: Emax=50000 mean_obj == optimality_gap.csv')
    print('=' * 82)

    t_all = time.time(); done_ct = 0
    for (M, E, K) in cells:
        t0 = time.time()
        Ee, mean_obj, std, sem = sa_cell(M, E, K, seeds, a.iters, a.restarts)
        dt = time.time() - t0
        grid_rows.append(dict(M=M, Emax=E, K=K, Ee=round(Ee, 1),
                              mean_obj=round(mean_obj, 4), std_obj=round(std, 4),
                              sem_obj=round(sem, 4), E_per_node=round(E / M, 2),
                              instances=a.instances, iters=a.iters, restarts=a.restarts))
        # rewrite both CSVs each cell so a kill loses nothing and kstar stays fresh
        write_csv(grid_path, grid_rows)
        write_csv(kstar_path, compute_kstar(grid_rows, a.tol))

        done_ct += 1
        eta = (time.time() - t_all) / done_ct * (total - done_ct)
        xchk = '  [50k xcheck cell]' if E == 50000 else ''
        print(f'  M={M:>3} E={E:>6} K={K}  Ee={Ee:>8.0f}  obj={mean_obj:>9.2f} '
              f'+/-{sem:>5.2f}  {dt:>5.1f}s  ETA {eta/60:>5.1f}m{xchk}')

    # final K* surface summary
    ks = compute_kstar(grid_rows, a.tol)
    print('\n' + '=' * 82)
    print('  K*(M, Emax)   [band = bounds where K are within SA margin of best]')
    print(f'  {"M":>4} {"Emax":>7} {"E/node":>7} {"K*":>3} {"band":>6} {"min_obj":>9}')
    for r in ks:
        print(f'  {r["M"]:>4} {r["Emax"]:>7} {r["E_per_node"]:>7.1f} '
              f'{r["Kstar"]:>3} {r["band"]:>6} {r["min_obj"]:>9.2f}')
    print(f'\n  wrote {grid_path}\n  wrote {kstar_path}')
    print('  NEXT: check if K* is a clean function of E_per_node (a scaling law) ->')
    print('        generalises to random (M,Emax); else feed kstar_sa.csv to the predictor.')
    print('=' * 82)


if __name__ == '__main__':
    main()