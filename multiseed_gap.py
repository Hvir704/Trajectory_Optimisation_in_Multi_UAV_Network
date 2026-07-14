"""
multiseed_gap.py  --  error bars for the paper: MLP objective/gap across the
three trained model seeds (7, 42, 123), vs the fixed near-optimal SA reference.
================================================================================
Reviewers flag single-seed results. This reports, per (M, K):

    SA_obj                      : the fixed near-optimal reference (computed once
                                  per cell; SA's randomness is per-restart and is
                                  already averaged, so SA is the fixed baseline).
    MLP rollout+pp  mean +/- std  across model seeds  (the deployable objective)
    gap%            mean +/- std  across model seeds  (how far above SA)
    MLP rollout-only mean +/- std across model seeds  (the fast/degraded point)

The MLP number varies across the 3 trained models; SA does not -> the +/- is on
the MLP side, which is exactly the robustness claim: "the policy achieves within
X +/- Y% of optimal regardless of training seed."

Shared-seed instances (INSTANCE_SEED, default 2025) so SA reproduces your existing
tables at Emax=50000. Uses the (now delta-accelerated) fleet_post_process. New
output dir; resumable (skips (M,K) already in the CSV).

Run (PowerShell, one line):
    python multiseed_gap.py --M 50 100 200 --K 1 2 3 4 5 6 --iters 2000 ^
        --restarts 2 --instances 12 --model-seeds 7 42 123
"""
import os, csv, argparse, warnings
from statistics import mean, pstdev
import numpy as np
import torch

from uav_aoi_solver import P, Env
from multi_uav_solver import MP, MultiUAVPolicy, fleet_rollout, fleet_post_process
from compare_baseline import gen, INSTANCE_SEED
from fleet_optimality_gap import sa_best

CKPT_TMPL = 'models_multi_uav/fleet_M{M}_K{K}_split_seed{seed}.pt'


def load_policy(path, device):
    ck = torch.load(path, map_location=device)
    idim = int(ck.get('input_dim', MP.INPUT_DIM)) if isinstance(ck, dict) else MP.INPUT_DIM
    state = ck['policy'] if (isinstance(ck, dict) and 'policy' in ck) else ck
    pol = MultiUAVPolicy(hidden=256, input_dim=idim).to(device)
    pol.load_state_dict(state); pol.eval()
    return pol


def mlp_cell(path, M, K, Ee, seeds, device):
    """Mean rollout+pp and rollout-only objective for one trained model."""
    pol = load_policy(path, device)
    pp, ro = [], []
    for s in seeds:
        env = Env(M=M, seed=s)
        f = fleet_rollout(pol, env, K, device, Emax_each=Ee, greedy=True)
        ro.append(f.fleet_objective())          # BEFORE pp = rollout-only
        f = fleet_post_process(env, f)
        pp.append(f.fleet_objective())           # AFTER pp = deployable
    return float(mean(pp)), float(mean(ro))


def sa_ref(M, K, Ee, seeds, iters, restarts):
    return float(mean(sa_best(*gen(M, s), K, Ee, M, iters, restarts) for s in seeds))


def _msd(xs):
    return (float(mean(xs)), float(pstdev(xs)) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, nargs='+', default=[50, 100, 200])
    ap.add_argument('--K', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--model-seeds', type=int, nargs='+', default=[7, 42, 123])
    ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--restarts', type=int, default=2)
    ap.add_argument('--instances', type=int, default=12)
    ap.add_argument('--seed', type=int, default=INSTANCE_SEED)
    ap.add_argument('--Emax', type=float, default=P.Emax)   # total; split Emax/K
    ap.add_argument('--out-dir', default='multiseed_gap')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, 'multiseed_gap.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rng = np.random.default_rng(a.seed)
    inst_seeds = [int(rng.integers(0, 1e7)) for _ in range(a.instances)]

    rows, done = [], set()
    if os.path.exists(out):
        for r in csv.DictReader(open(out)):
            rows.append(r); done.add((int(r['M']), int(r['K'])))
        print(f'resume: {len(done)} cells already done')

    print('=' * 90)
    print(f'  multiseed gap | model-seeds={a.model_seeds} | Emax={a.Emax:.0f} (split /K)')
    print(f'  SA iters={a.iters} restarts={a.restarts} | instances={a.instances} '
          f'seed={a.seed} | device={device}')
    print('=' * 90)

    for M in a.M:
        for K in a.K:
            if (M, K) in done:
                continue
            Ee = a.Emax / K
            sa = sa_ref(M, K, Ee, inst_seeds, a.iters, a.restarts)

            pp_vals, ro_vals, gaps, used = [], [], [], []
            for sd in a.model_seeds:
                path = CKPT_TMPL.format(M=M, K=K, seed=sd)
                if not os.path.exists(path):
                    warnings.warn(f'missing model M={M} K={K} seed={sd}: {path}')
                    continue
                pp, ro = mlp_cell(path, M, K, Ee, inst_seeds, device)
                pp_vals.append(pp); ro_vals.append(ro)
                gaps.append(100 * (pp - sa) / abs(sa)); used.append(sd)

            if not pp_vals:
                warnings.warn(f'no models for M={M} K={K}; skipping cell')
                continue

            pp_m, pp_s = _msd(pp_vals)
            ro_m, ro_s = _msd(ro_vals)
            g_m, g_s = _msd(gaps)
            row = dict(M=M, K=K, Ee=round(Ee, 1), instances=a.instances,
                       n_seeds=len(used), seeds_used='|'.join(map(str, used)),
                       SA_obj=round(sa, 3),
                       mlp_pp_mean=round(pp_m, 3), mlp_pp_std=round(pp_s, 3),
                       gap_pct_mean=round(g_m, 2), gap_pct_std=round(g_s, 2),
                       mlp_rollout_mean=round(ro_m, 3), mlp_rollout_std=round(ro_s, 3))
            for sd, pp, g in zip(used, pp_vals, gaps):
                row[f'pp_seed{sd}'] = round(pp, 3)
                row[f'gap_seed{sd}'] = round(g, 2)
            rows.append(row)

            # write every cell (resume-safe)
            keys = []
            for rr in rows:
                for k in rr:
                    if k not in keys: keys.append(k)
            with open(out, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys, restval='')
                w.writeheader(); w.writerows(rows)

            print(f'  M={M:>3} K={K}  SA={sa:>9.2f}  MLP+pp={pp_m:>9.2f}+/-{pp_s:>5.2f}  '
                  f'gap={g_m:>5.1f}+/-{g_s:>4.1f}%  (n={len(used)})')

    print('\n' + '=' * 90)
    print('  gap_pct_mean +/- gap_pct_std  is the headline robustness number per cell.')
    print(f'  wrote {out}')
    print('  paper phrasing: "within X +/- Y% of optimal across 3 training seeds",')
    print('  quoting the K>=4 rows (K=1 is the degenerate single-UAV case).')
    print('=' * 90)


if __name__ == '__main__':
    main()
