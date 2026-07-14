"""
build_anchor.py  —  generate a fleet ANCHOR dataset at a chosen TOTAL energy Emax.
=================================================================================
The existing multi_uav_solver CLI hard-wires split battery to P.Emax/K (50 kJ).
This driver lets you set an arbitrary TOTAL Emax so the K*(M,Emax) predictor can
be calibrated at several energy budgets. It trains (or reuses) each fleet cell,
evaluates routing metrics AND the deconfliction penalty, and writes two tables
into a NEW output directory. Nothing existing is overwritten.

Split battery: each UAV gets Emax_each = Emax / K   (Emax = the TOTAL knob).

Outputs (into --out-dir):
    models/fleet_M{M}_K{K}_E{Emax}_seed{seed}.pt   (trained models)
    eval_E{Emax}.csv       (M,K,seed,nodes,waoi,priority,obj,...)  -> predictor eval table
    compare_E{Emax}.csv    (M,K,beam_routing,beam_penalty,augmented_final,...) -> predictor cmp table

Run one line at a time (uav_env active, from repo root):
    python build_anchor.py --Emax 10000  --out-dir kstar_ME_runs
    python build_anchor.py --Emax 200000 --out-dir kstar_ME_runs
    python build_anchor.py --Emax 50000  --out-dir kstar_ME_runs --reuse-legacy
"""

import os, csv, argparse
import numpy as np
import torch

from uav_aoi_solver import P, Env
from multi_uav_solver import (MP, MultiUAVPolicy, train_fleet,
                              fleet_rollout, fleet_post_process, deconfliction_schedule)

INSTANCE_SEED = 2025   # shared across all eval scripts (stitchable)


def model_path(out_models, M, K, Emax, seed):
    return os.path.join(out_models, f'fleet_M{M}_K{K}_E{int(Emax)}_seed{seed}.pt')

def legacy_path(M, K, seed):
    return os.path.join('models_multi_uav', f'fleet_M{M}_K{K}_split_seed{seed}.pt')

def load_policy(path, device):
    ck = torch.load(path, map_location=device)
    pol = MultiUAVPolicy(hidden=256, input_dim=MP.INPUT_DIM).to(device)
    pol.load_state_dict(ck['policy']); pol.eval()
    return pol


def eval_cell(pol, M, K, Emax_each, n, device):
    """Routing metrics + deconfliction penalty over n shared-seed instances."""
    rng = np.random.default_rng(INSTANCE_SEED)
    obj = nodes = waoi = pri = lstd = pen = fin = None
    acc = {k: [] for k in ('obj', 'nodes', 'waoi', 'pri', 'lstd', 'pen', 'fin')}
    with torch.no_grad():
        for _ in range(n):
            s = int(rng.integers(0, 10_000_000)); env = Env(M=M, seed=s)
            f = fleet_rollout(pol, env, K, device, Emax_each=Emax_each, greedy=True)
            f = fleet_post_process(env, f)
            ro = f.fleet_objective()
            if K >= 2:
                sched = deconfliction_schedule(f, delta=25.0, dt=0.25,
                                               optimize_order=True, verify=False)
                pn = float(sched['aoi_penalty'])
            else:
                pn = 0.0
            acc['obj'].append(ro); acc['nodes'].append(f.fleet_nodes())
            acc['waoi'].append(P.theta1 * f.fleet_waoi()); acc['pri'].append(f.fleet_priority())
            acc['lstd'].append(float(np.std([len(t) for t in f.trajs])))
            acc['pen'].append(pn); acc['fin'].append(ro + pn)
    m = lambda a: float(np.mean(a))
    return {k: m(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Emax', type=float, required=True, help='TOTAL energy budget (J)')
    ap.add_argument('--M', type=int, nargs='+', default=[100, 200])
    ap.add_argument('--K', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--instances', type=int, default=100)
    ap.add_argument('--out-dir', default='kstar_ME_runs')
    ap.add_argument('--reuse-legacy', action='store_true',
                    help='for Emax=50000, reuse existing models_multi_uav/*_split_* models')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_models = os.path.join(args.out_dir, 'models')
    os.makedirs(out_models, exist_ok=True)
    tag = int(args.Emax)

    print('=' * 72)
    print(f'  Anchor build | TOTAL Emax={tag} J | M={args.M} K={args.K} | '
          f'seed={args.seed} | n={args.instances} | device={device}')
    print(f'  out-dir={args.out_dir}  (models/, eval_E{tag}.csv, compare_E{tag}.csv)')
    print('=' * 72)

    eval_rows, cmp_rows = [], []
    for M in args.M:
        for K in args.K:
            Ee = args.Emax / K
            mp = model_path(out_models, M, K, args.Emax, args.seed)
            if os.path.exists(mp):
                pol = load_policy(mp, device); print(f'[reuse ] M={M:>3} K={K} E={tag}')
            elif args.reuse_legacy and abs(args.Emax - 50000) < 1 and os.path.exists(legacy_path(M, K, args.seed)):
                pol = load_policy(legacy_path(M, K, args.seed), device); print(f'[legacy] M={M:>3} K={K}')
            else:
                print(f'[train ] M={M:>3} K={K} E={tag} Emax_each={Ee:.0f} ...')
                train_fleet(M=M, K=K, n_epochs=args.epochs, eps_per_epoch=64, device=device,
                            save_path=mp, Emax_each=Ee, seed=args.seed)
                pol = load_policy(mp, device)

            r = eval_cell(pol, M, K, Ee, args.instances, device)
            eval_rows.append(dict(M=M, K=K, seed=args.seed, battery='split',
                                  Emax_each=round(Ee, 3), obj=r['obj'], nodes=r['nodes'],
                                  waoi=r['waoi'], priority=r['pri'], load_std=r['lstd'],
                                  n=args.instances))
            cmp_rows.append(dict(M=M, K=K, instances=args.instances,
                                 greedy_routing=r['obj'], beam_routing=r['obj'], d_routing=0.0,
                                 greedy_penalty=r['pen'], beam_penalty=r['pen'], d_penalty=0.0,
                                 greedy_final=r['fin'], beam_final=r['fin'], d_final=0.0,
                                 augmented_final=r['fin'], augmented_method='greedy+deconflict'))
            print(f'         -> obj={r["obj"]:8.2f} nodes={r["nodes"]:6.1f} '
                  f'pen={r["pen"]:6.2f} final={r["fin"]:8.2f}')

    ev = os.path.join(args.out_dir, f'eval_E{tag}.csv')
    cm = os.path.join(args.out_dir, f'compare_E{tag}.csv')
    with open(ev, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys())); w.writeheader(); w.writerows(eval_rows)
    with open(cm, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys())); w.writeheader(); w.writerows(cmp_rows)
    print(f'\nWrote {ev}\nWrote {cm}')
    print(f'\nNext: feed all anchors to the predictor (see kstar_ME_predictor.py --anchor).')


if __name__ == '__main__':
    main()
