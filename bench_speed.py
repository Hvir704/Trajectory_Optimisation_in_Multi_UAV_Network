"""
bench_speed.py  --  speed of the policy, and the anytime-SA Pareto baseline
===========================================================================
Two things in one run on ONE machine (so the hardware cancels in the ratio):

  1. How fast is the policy?  Times three things separately, warmup discarded,
     CUDA-synchronised, model-load and instance-gen EXCLUDED:
        rollout only        (the learned construction)
        post-process only   (2-opt + insertion)
        rollout + pp         (what your quality numbers actually use)
     Reported as ms/instance on the detected device.

  2. The honest "nuanced baseline you beat": SA is not beaten on quality by a
     mediocre policy -- but it IS beaten in the LOW-TIME regime. So we sweep SA
     over a grid of iteration budgets and record (time, objective) for each,
     tracing SA's ANYTIME curve. The policy is a single point on that plane.
     The script then reports:
        speedup vs full-budget SA          = t_SA(ref) / t_policy
        parity budget                       = smallest SA budget whose obj
                                              reaches the policy's obj
        speedup-to-parity                   = t_SA(parity) / t_policy
     If SA already beats the policy at the SMALLEST budget, that is reported
     honestly as "no low-time dominance" -- better to see it here than in review.

Device-dependence is handled the right way: raw ms are printed for context, but
the HEADLINE numbers are RATIOS measured back-to-back on the same device, which
are what transfer across hardware. SA settings (restarts) are pinned and printed,
since SA time is meaningless without them.

Quality note: policy runs on Env(M,seed); SA runs on gen(M,seed). If those differ
per seed the obj comparison is DISTRIBUTION-matched, not instance-matched (same
assumption as fleet_optimality_gap). The budget-check line flags it.

New output dir only -- nothing overwritten.

Run (PowerShell, one line):
    python bench_speed.py --M 50 100 200 --K 1 4 6 --instances 12 ^
        --sa-iters 100 500 2500 --restarts 2
"""
import os, csv, time, argparse, warnings
import numpy as np
import torch

from uav_aoi_solver import P, Env
from multi_uav_solver import (MP, MultiUAVPolicy, fleet_rollout,
                              fleet_post_process)
from compare_baseline import gen, INSTANCE_SEED
from fleet_optimality_gap import sa_best


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def load_policy(path, device):
    ck = torch.load(path, map_location=device)
    idim = int(ck.get('input_dim', MP.INPUT_DIM)) if isinstance(ck, dict) else MP.INPUT_DIM
    state = ck['policy'] if (isinstance(ck, dict) and 'policy' in ck) else ck
    pol = MultiUAVPolicy(hidden=256, input_dim=idim).to(device)
    pol.load_state_dict(state)
    pol.eval()
    return pol


def time_policy(pol, envs, K, Ee, device, warmup):
    """Return (rollout_ms, pp_ms, total_ms, mean_obj) per instance."""
    # warmup: first CUDA calls compile kernels; never time them
    for _ in range(warmup):
        f = fleet_rollout(pol, envs[0], K, device, Emax_each=Ee, greedy=True)
        fleet_post_process(envs[0], f)
    sync(device)

    # rollout only
    fleets = []
    sync(device); t0 = time.perf_counter()
    for env in envs:
        fleets.append(fleet_rollout(pol, env, K, device, Emax_each=Ee, greedy=True))
    sync(device); roll_s = (time.perf_counter() - t0) / len(envs)

    # rollout-only objective (the FAST-end operating point, no pp) -- untimed
    roll_obj = float(np.mean([f.fleet_objective() for f in fleets]))

    # post-process only (on the already-built fleets)
    objs = []
    sync(device); t0 = time.perf_counter()
    for env, f in zip(envs, fleets):
        fp = fleet_post_process(env, f)
        objs.append(fp.fleet_objective())
    sync(device); pp_s = (time.perf_counter() - t0) / len(envs)

    return (roll_s * 1e3, pp_s * 1e3, (roll_s + pp_s) * 1e3,
            roll_obj, float(np.mean(objs)))


def time_sa_curve(sa_instances, K, Ee, M, iters_grid, restarts):
    """For each iteration budget: (mean s/instance, mean obj). SA's anytime curve."""
    curve = []
    for it in iters_grid:
        t0 = time.perf_counter()
        vals = [sa_best(pos, wi, tcd, K, Ee, M, it, restarts)
                for (pos, wi, tcd) in sa_instances]
        dt = (time.perf_counter() - t0) / len(sa_instances)
        curve.append((it, dt, float(np.mean(vals))))
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, nargs='+', default=[50, 100, 200])
    ap.add_argument('--K', type=int, nargs='+', default=[1, 4, 6])
    ap.add_argument('--instances', type=int, default=12)
    ap.add_argument('--seed', type=int, default=INSTANCE_SEED)
    ap.add_argument('--sa-iters', type=int, nargs='+', default=[100, 500, 2500],
                    help='SA iteration budgets to trace the anytime curve')
    ap.add_argument('--restarts', type=int, default=2)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--ckpt-template',
                    default='models_multi_uav/fleet_M{M}_K{K}_split_seed{seed}.pt')
    ap.add_argument('--full-battery', action='store_true')
    ap.add_argument('--out-dir', default='bench_speed')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iters_grid = sorted(a.sa_iters)
    ref_iters = iters_grid[-1]                       # "full budget" reference

    print('=' * 84)
    print(f'  bench_speed | device={device} | instances={a.instances} | seed={a.seed}')
    print(f'  SA anytime budgets={iters_grid} | restarts={a.restarts} (pinned)')
    print(f'  headline numbers are RATIOS (device-portable); raw ms are context only')
    print('=' * 84)

    rng = np.random.default_rng(a.seed)
    seeds = [int(rng.integers(0, 1e7)) for _ in range(a.instances)]

    rows = []
    for M in a.M:
        for K in a.K:
            Ee = P.Emax if a.full_battery else P.Emax / K
            ckpt = a.ckpt_template.format(M=M, K=K, seed=a.seed)
            if not os.path.exists(ckpt):
                warnings.warn(f'skip {M},{K}: no checkpoint at {ckpt}')
                continue

            pol   = load_policy(ckpt, device)
            envs  = [Env(M=M, seed=s) for s in seeds]
            sa_in = [gen(M, s) for s in seeds]

            roll_ms, pp_ms, tot_ms, roll_obj, rl_obj = time_policy(
                pol, envs, K, Ee, device, a.warmup)
            curve = time_sa_curve(sa_in, K, Ee, M, iters_grid, a.restarts)

            # fast-end point = rollout-only: does SA beat it within its own runtime?
            roll_s = roll_ms / 1e3
            sa_at_roll_time = [ob for (it, dt, ob) in curve if dt <= roll_s]
            roll_verdict = ('SA cannot answer in rollout time (fast-end is policy-only)'
                            if not sa_at_roll_time
                            else ('rollout-only WINS at matched time'
                                  if min(sa_at_roll_time) > roll_obj
                                  else 'SA beats rollout-only at matched time'))

            # anytime analysis: does the policy dominate SA in the low-time regime?
            # more-negative obj = better, so SA "reaches" policy when sa_obj <= rl_obj.
            parity = next(((it, dt, ob) for (it, dt, ob) in curve if ob <= rl_obj), None)
            ref = next(c for c in curve if c[0] == ref_iters)
            rl_tot_s = tot_ms / 1e3
            speedup_ref = ref[1] / rl_tot_s if rl_tot_s > 0 else float('nan')
            if parity is None:
                dominance = 'DOMINATES (SA never reaches policy quality in grid)'
                speedup_parity = float('nan')
            elif parity[0] == iters_grid[0] and curve[0][2] <= rl_obj:
                dominance = 'NO low-time dominance (SA beats policy even at min budget)'
                speedup_parity = parity[1] / rl_tot_s
            else:
                dominance = f'dominates below ~{parity[0]} SA iters'
                speedup_parity = parity[1] / rl_tot_s

            row = dict(device=device, M=M, K=K, instances=a.instances,
                       rl_rollout_ms=round(roll_ms, 3), rl_pp_ms=round(pp_ms, 3),
                       rl_total_ms=round(tot_ms, 3),
                       rl_rollout_obj=round(roll_obj, 3), rl_rollout_verdict=roll_verdict,
                       rl_obj=round(rl_obj, 3),
                       sa_ref_iters=ref_iters, sa_ref_s=round(ref[1], 3),
                       sa_ref_obj=round(ref[2], 3),
                       speedup_vs_ref=round(speedup_ref, 1),
                       speedup_to_parity=(round(speedup_parity, 1)
                                          if speedup_parity == speedup_parity else 'n/a'),
                       dominance=dominance)
            for (it, dt, ob) in curve:
                row[f'sa{it}_s'] = round(dt, 3)
                row[f'sa{it}_obj'] = round(ob, 3)
            rows.append(row)

            print(f'\nM={M} K={K}  [device={device}, Emax_each={Ee:.0f}]')
            print(f'  rollout-only {roll_ms:8.2f} ms   obj={roll_obj:.2f}   [{roll_verdict}]')
            print(f'  rollout+pp   {tot_ms:8.2f} ms   obj={rl_obj:.2f}   '
                  f'(pp alone {pp_ms:.1f} ms)')
            for (it, dt, ob) in curve:
                mark = '  <- policy-quality reached here' if (parity and it == parity[0]) else ''
                print(f'  SA it={it:>5}  {dt*1e3:9.1f} ms/inst   obj={ob:.2f}{mark}')
            print(f'  => {speedup_ref:.0f}x faster than SA@{ref_iters}it   |   {dominance}')

    if not rows:
        print('\nNo cells ran -- check --ckpt-template path.')
        return

    out = os.path.join(a.out_dir, 'bench_speed.csv')
    keys = sorted({k for r in rows for k in r})
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print('\n' + '=' * 84)
    print('  HEADLINE = speedup_vs_ref (device-portable ratio).')
    print('  Pareto claim = policy obj vs the SA anytime curve: if the policy obj is')
    print('  better than SA at low iters, you dominate the low-latency regime.')
    print(f'  wrote {out}')
    print('=' * 84)


if __name__ == '__main__':
    main()