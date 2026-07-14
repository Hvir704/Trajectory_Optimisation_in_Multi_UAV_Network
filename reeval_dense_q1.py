"""
reeval_dense_q1.py  --  close Q1: did the dense reward fix priority-blindness?
==============================================================================
The DENSE checkpoints store `best_final_obj` but NOT the priority-per-node
diagnostic, and their FINAL metric is greedy/no-pp -- so it is NOT comparable to
the eval_table-based gap in optimality_gap.csv (that used post-processed RL). This
script fixes both:

  1. Loads any number of fleet checkpoints (auto-reads M, K, Emax_each, convention
     from the ckpt; falls back to filename / CLI for older checkpoints).
  2. Runs the GREEDY, no-post-process rollout on a fixed held-out instance set
     (default seed = INSTANCE_SEED = 2025, stitchable with your existing tables)
     and reports the decomposition that answers Q1:
        priority-per-node (pooled)  <-- the ~5.78 -> ~7.0 number
        nodes served, total priority, WAoI, objective
        population mean weight       <-- random-selection reference (~5.50)
        per-UAV priority/node spread (K>1) -- is one UAV carrying the blindness?
  3. Runs SA (compare_baseline, via sa_best) on the SAME seeds and reports a
     MATCHED-METRIC gap: greedy/no-pp RL vs SA. This isolates the POLICY from the
     2-opt/insertion heuristic -- the honest gap for Q1/Q3.

Sanity anchors printed automatically:
  * SA_obj should reproduce optimality_gap.csv (same seeds/iters/restarts).
  * RL greedy/no-pp obj should be close to the ckpt's best_final_obj (intra;
    common_t0's FINAL folds in the deconfliction penalty, so expect a small gap).

Everything is written to a NEW directory (reeval_dense_q1/) -- nothing is
overwritten.

Run (PowerShell, one line):
    python reeval_dense_q1.py --instances 12 --iters 2500 --restarts 2

Priority/node only (fast, no SA):
    python reeval_dense_q1.py --no-sa

Add the OLD terminal-reward checkpoint to get the before/after on priority/node:
    python reeval_dense_q1.py --checkpoints ^
        models_multi_uav/fleet_M100_K1_split_seed42.pt ^
        fleet_M100_K1_split_seed42_DENSE.pt
"""
import os, re, csv, argparse, warnings
import numpy as np
import torch

from uav_aoi_solver import P, Env
from multi_uav_solver import MultiUAVPolicy, fleet_rollout, MP
from compare_baseline import gen, INSTANCE_SEED, EMAX
from fleet_optimality_gap import sa_best

DEFAULT_CKPTS = [
    'fleet_M100_K1_split_seed42_DENSE.pt',
    'fleet_M100_K4_split_seed42_DENSE.pt',
    'fleet_M100_K4_split_seed42_DENSE_CT0.pt',
]

_NAME_RE = re.compile(r'M(\d+)_K(\d+)_(split|full)')


def resolve_config(ckpt, path):
    """Pull M, K, Emax_each, convention from ckpt; fall back to filename / defaults."""
    name = os.path.basename(path)
    m = _NAME_RE.search(name)
    fn_M = int(m.group(1)) if m else None
    fn_K = int(m.group(2)) if m else None
    fn_split = (m.group(3) == 'split') if m else True

    M = int(ckpt.get('M', fn_M) if isinstance(ckpt, dict) else fn_M)
    K = int(ckpt.get('K', fn_K) if isinstance(ckpt, dict) else fn_K)
    if isinstance(ckpt, dict) and 'Emax_each' in ckpt:
        Ee = float(ckpt['Emax_each'])
    else:
        Ee = (P.Emax / K) if fn_split else P.Emax
    conv = ckpt.get('convention', '?') if isinstance(ckpt, dict) else '?'
    bfo = ckpt.get('best_final_obj', None) if isinstance(ckpt, dict) else None
    idim = int(ckpt.get('input_dim', MP.INPUT_DIM)) if isinstance(ckpt, dict) else MP.INPUT_DIM
    return M, K, Ee, conv, bfo, idim


def load_policy(path, device):
    ckpt = torch.load(path, map_location=device)
    M, K, Ee, conv, bfo, idim = resolve_config(ckpt, path)
    state = ckpt['policy'] if (isinstance(ckpt, dict) and 'policy' in ckpt) else ckpt
    pol = MultiUAVPolicy(input_dim=idim).to(device)
    pol.load_state_dict(state)
    pol.eval()
    return pol, M, K, Ee, conv, bfo


def per_uav_ppn(fleet):
    """(mean, std) of chain priority/chain nodes over non-empty UAVs; None if <2."""
    vals = []
    for t in fleet.trajs:
        if t:
            vals.append(sum(fleet.env.wi[j] for j in t) / len(t))
    if len(vals) < 2:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def eval_checkpoint(path, seeds, device, run_sa, iters, restarts):
    pol, M, K, Ee, conv, bfo = load_policy(path, device)

    obj, nodes, prio, waoi, ppn = [], [], [], [], []
    popw, uav_mu, uav_sd = [], [], []
    sa_vals = []

    for s in seeds:
        env = Env(M=M, seed=s)
        fleet = fleet_rollout(pol, env, K, device, Emax_each=Ee, greedy=True)
        n = fleet.fleet_nodes()
        p = fleet.fleet_priority()
        obj.append(fleet.fleet_objective())
        nodes.append(n)
        prio.append(p)
        waoi.append(fleet.fleet_waoi())
        ppn.append(p / max(n, 1))
        popw.append(float(env.wi.mean()))
        mu, sd = per_uav_ppn(fleet)
        if mu is not None:
            uav_mu.append(mu); uav_sd.append(sd)

        if run_sa:
            pos, wi, tcd = gen(M, s)
            sa_vals.append(sa_best(pos, wi, tcd, K, Ee, M, iters, restarts))

    row = dict(
        ckpt=os.path.basename(path), conv=conv, M=M, K=K, Emax_each=Ee,
        instances=len(seeds),
        nodes=float(np.mean(nodes)),
        priority=float(np.mean(prio)),
        ppn=float(np.mean(ppn)),                 # per-instance priority/node, meaned
        pop_mean_w=float(np.mean(popw)),         # random-selection reference
        uav_ppn_mean=float(np.mean(uav_mu)) if uav_mu else float('nan'),
        uav_ppn_std=float(np.mean(uav_sd)) if uav_sd else float('nan'),
        waoi=float(np.mean(waoi)),
        RL_obj_greedy_nopp=float(np.mean(obj)),
        best_final_obj=bfo if bfo is not None else float('nan'),
    )
    if run_sa:
        sa_obj = float(np.mean(sa_vals))
        rl = row['RL_obj_greedy_nopp']
        row['SA_obj'] = sa_obj
        row['gap_pct_matched'] = 100 * (rl - sa_obj) / abs(sa_obj)
        row['Ee_vs_EMAX_over_K'] = f'{Ee:.0f} vs {EMAX / K:.0f}'
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', nargs='+', default=DEFAULT_CKPTS)
    ap.add_argument('--instances', type=int, default=12)
    ap.add_argument('--seed', type=int, default=INSTANCE_SEED)   # 2025, matches tables
    ap.add_argument('--iters', type=int, default=2500)
    ap.add_argument('--restarts', type=int, default=2)
    ap.add_argument('--no-sa', dest='sa', action='store_false')
    ap.add_argument('--out-dir', default='reeval_dense_q1')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # identical seed draw to fleet_optimality_gap => SA reproduces optimality_gap.csv
    rng = np.random.default_rng(a.seed)
    seeds = [int(rng.integers(0, 1e7)) for _ in range(a.instances)]

    print('=' * 78)
    print(f'  reeval_dense_q1  device={device}  instances={a.instances}  seed={a.seed}')
    print(f'  SA: {"on" if a.sa else "OFF"}'
          + (f'  (iters={a.iters}, restarts={a.restarts})' if a.sa else ''))
    print('=' * 78)

    rows = []
    for path in a.checkpoints:
        if not os.path.exists(path):
            warnings.warn(f'skip missing checkpoint: {path}')
            continue
        row = eval_checkpoint(path, seeds, device, a.sa, a.iters, a.restarts)
        rows.append(row)

        print(f'\n{row["ckpt"]}   [conv={row["conv"]}  M={row["M"]}  K={row["K"]}'
              f'  Emax_each={row["Emax_each"]:.0f}]')
        print(f'  priority/node (pooled)     : {row["ppn"]:.3f}'
              f'   (pop-mean weight {row["pop_mean_w"]:.2f}; target ~7.0)')
        if row['K'] > 1 and row['uav_ppn_mean'] == row['uav_ppn_mean']:
            print(f'  per-UAV priority/node      : {row["uav_ppn_mean"]:.3f}'
                  f'  +/- {row["uav_ppn_std"]:.3f}  (spread across UAVs)')
        print(f'  nodes served / priority    : {row["nodes"]:.1f} / {row["priority"]:.1f}')
        print(f'  WAoI                       : {row["waoi"]:.2f}')
        print(f'  RL obj (greedy, no-pp)     : {row["RL_obj_greedy_nopp"]:.2f}', end='')
        if row['best_final_obj'] == row['best_final_obj']:
            print(f'   (ckpt best_final_obj {row["best_final_obj"]:.2f})')
        else:
            print()
        if a.sa:
            print(f'  SA obj (matched seeds)     : {row["SA_obj"]:.2f}'
                  f'   [budget {row["Ee_vs_EMAX_over_K"]}]')
            print(f'  MATCHED gap to SA          : {row["gap_pct_matched"]:.1f}%'
                  f'   (policy only, no pp/beam)')

    # write CSV
    out_csv = os.path.join(a.out_dir, 'reeval_dense_q1.csv')
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print('\n' + '=' * 78)
    print('  Q1 READ:  priority/node moved off pop-mean (~5.50) toward ~7.0  ->')
    print('            dense reward was the lever (scale it).')
    print('            still ~5.5-6.0 and matched gap still large -> architecture')
    print('            (Fix 2), not reward.')
    print(f'  wrote {out_csv}')
    print('=' * 78)


if __name__ == '__main__':
    main()
