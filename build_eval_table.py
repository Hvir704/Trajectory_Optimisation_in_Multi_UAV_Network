"""
build_eval_table.py  —  Assemble the K*(M) evaluation table from SAVED models.
==============================================================================
Runs eval_fleet() on every trained fleet checkpoint in models_multi_uav/ and
writes one CSV row per (M, K, seed). This CSV is the single source of truth for
kstar_predictor.py and for the sweet-spot figure.

GUARDRAILS (from the handoff):
  * Objectives come ONLY from eval_fleet on the saved model — never a figure title.
  * Split battery MUST use Emax_each = P.Emax / K for both policy and (later) baselines.
  * A shared instance seed (INSTANCE_SEED=2025) keeps this table stitchable with
    beam_eval / deconflict results.

Run (from the repo root, uav_env active), one line at a time:
    python build_eval_table.py                       # split battery, all found models
    python build_eval_table.py --instances 200       # match your headline n
    python build_eval_table.py --battery full         # ablation table
    python build_eval_table.py --out eval_table_split.csv --out-dir kstar_out
"""

import os, re, csv, glob, argparse
import torch

from uav_aoi_solver import P
from multi_uav_solver import MP, MultiUAVPolicy, eval_fleet

# Shared across ALL eval scripts so cells are directly comparable / stitchable.
INSTANCE_SEED = 2025

# The K grid the K* study intends to cover (used only to report what's missing).
INTENDED_K = [1, 2, 3, 4, 5, 6, 8]

_FNAME = re.compile(r'fleet_M(\d+)_K(\d+)_(split|full)_seed(\d+)\.pt$')


def parse_name(path):
    m = _FNAME.search(os.path.basename(path))
    if not m:
        return None
    M, K, bat, seed = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
    return dict(M=M, K=K, battery=bat, seed=seed, path=path)


def budget_for(K, battery):
    """Split: each UAV gets Emax/K. Full: each UAV gets the single-UAV Emax."""
    return (P.Emax / K) if battery == 'split' else MP.Emax_each


def load_policy(path, device):
    ckpt = torch.load(path, map_location=device)
    pol = MultiUAVPolicy(hidden=256, input_dim=MP.INPUT_DIM).to(device)
    pol.load_state_dict(ckpt['policy'])
    pol.eval()
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', default='models_multi_uav')
    ap.add_argument('--M', type=int, nargs='+', default=[],
                    help='Only tabulate these M values (default: all found). '
                         'Use to skip small-M B&B/exact-anchor leftovers.')
    ap.add_argument('--battery', choices=['split', 'full'], default='split',
                    help='Which battery mode to tabulate (split is the primary narrative).')
    ap.add_argument('--instances', type=int, default=200,
                    help='n instances per eval_fleet call (match your headline n).')
    ap.add_argument('--seed', type=int, default=INSTANCE_SEED,
                    help='Instance seed passed to eval_fleet (keep 2025 for stitchability).')
    ap.add_argument('--out', default='eval_table.csv')
    ap.add_argument('--out-dir', default='.', help='Directory to write the CSV into.')
    ap.add_argument('--no-postprocess', action='store_true',
                    help='Disable fleet_post_process (default keeps it ON, matching eval_fleet).')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out)

    paths = sorted(glob.glob(os.path.join(args.models_dir, 'fleet_M*_K*_*_seed*.pt')))
    found = [p for p in (parse_name(p) for p in paths) if p and p['battery'] == args.battery]
    if args.M:
        keep = set(args.M)
        skipped = sorted({d['M'] for d in found if d['M'] not in keep})
        found = [d for d in found if d['M'] in keep]
        if skipped:
            print(f'Skipping M not in --M filter: {skipped}')

    if not found:
        print(f'No {args.battery}-battery checkpoints found in {args.models_dir}/.')
        print('Expected filenames like: fleet_M100_K3_split_seed42.pt')
        return

    print('=' * 70)
    print(f'  Building eval table | battery={args.battery} | n={args.instances} '
          f'| instance_seed={args.seed} | device={device}')
    print(f'  post_process={"OFF" if args.no_postprocess else "ON"}')
    print('=' * 70)

    rows = []
    for i, meta in enumerate(sorted(found, key=lambda d: (d['M'], d['K'], d['seed'])), 1):
        M, K, seed = meta['M'], meta['K'], meta['seed']
        e_each = budget_for(K, args.battery)
        pol = load_policy(meta['path'], device)
        res = eval_fleet(pol, M, K, n=args.instances, seed=args.seed, device=device,
                         Emax_each=e_each, use_postprocess=not args.no_postprocess)
        row = dict(M=M, K=K, seed=seed, battery=args.battery,
                   Emax_each=round(e_each, 3),
                   obj=res['obj'], nodes=res['nodes'], waoi=res['waoi'],
                   priority=res['priority'], load_std=res['load_std'],
                   n=args.instances)
        rows.append(row)
        # Sanity: obj should equal waoi - theta2*priority (theta2 = 1.0).
        recon = res['waoi'] - P.theta2 * res['priority']
        flag = '' if abs(recon - res['obj']) < 1e-3 else '  <-- obj/waoi/priority mismatch!'
        print(f'[{i:>3}/{len(found)}] M={M:>3} K={K} seed={seed:<3} '
              f'obj={res["obj"]:+9.3f} nodes={res["nodes"]:6.1f} '
              f'waoi={res["waoi"]:7.2f} pri={res["priority"]:7.2f} '
              f'load_std={res["load_std"]:.2f}{flag}')

    fields = ['M', 'K', 'seed', 'battery', 'Emax_each', 'obj', 'nodes',
              'waoi', 'priority', 'load_std', 'n']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'\nWrote {len(rows)} rows -> {out_path}')

    # ---- Coverage report: which (M, K) cells and how many seeds each ----------
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r['M'], r['K']), set()).add(r['seed'])
    Ms = sorted({r['M'] for r in rows})
    print('\nSeed coverage per (M, K)  [target: 3 seeds; intended K =',
          INTENDED_K, ']')
    header = 'M \\ K  ' + ' '.join(f'{k:>4}' for k in INTENDED_K)
    print(header)
    for M in Ms:
        cells = []
        for k in INTENDED_K:
            s = by_cell.get((M, k))
            cells.append(f'{len(s):>4}' if s else '   .')
        print(f'{M:>5}  ' + ' '.join(cells))
    print('\n(".": cell absent; a number < 3 means seeds still missing for a CI band.)')


if __name__ == '__main__':
    main()