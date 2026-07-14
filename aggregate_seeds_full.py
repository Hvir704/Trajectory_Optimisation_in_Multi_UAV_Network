"""
aggregate_seeds_full.py
================================================================================
Collapse the 3 seed models {42, 123, 7} into ONE representative result and ONE
representative checkpoint per (M, K) cell -- at full quality, on your machine.

Unlike the quick container pass, this evaluates the POST-PROCESSED objective at
the full instance count for every seed, so the mean +/- std and the best-of-3
selection are both computed on the number you actually report (MLP+pp), not on a
greedy proxy.

WHAT IT WRITES (into --out-dir, default ./agg_out)
  fleet_objective_grid.txt      human-readable M x K grids (pp mean+/-std, greedy mean+/-std)
  fleet_objective_aggregated.csv one row per (M,K): every stat + per-seed means
  fleet_models_best/            the chosen checkpoint per cell, fleet_M{M}_K{K}_best.pt
  fleet_models_best/manifest.csv which seed was chosen per cell and why
  _agg_store.jsonl              resume store (delete to recompute from scratch)

REQUIREMENTS
  * Run from a folder where  uav_aoi_solver.py  and  multi_uav_solver.py  import.
  * Point --model-dir at the folder that contains your fleet checkpoints. The
    script searches RECURSIVELY for files named
        fleet_M{M}_K{K}_split_seed{SEED}.pt
    so it works whether they sit in ./models_multi_uav/ or ./models/model/ etc.
  * torch, numpy. A CUDA GPU is used automatically if available (big speedup).

TYPICAL USAGE
  python aggregate_seeds_full.py                         # full grid, N=100, both metrics
  python aggregate_seeds_full.py --model-dir models_multi_uav
  python aggregate_seeds_full.py --M 100 150 200 --K 1 2 3 4
  python aggregate_seeds_full.py --instances 100 --select-by pp
  python aggregate_seeds_full.py --no-pp                 # greedy-only (much faster, less faithful)
  python aggregate_seeds_full.py --test-soup             # also diagnose weight-averaging (unreliable)
  python aggregate_seeds_full.py --finalize-only         # rebuild outputs from the store, no eval

Resuming: just re-run the same command. Cells already in _agg_store.jsonl are
skipped. Use --force to recompute requested cells.
"""
from __future__ import annotations
import os, sys, glob, json, copy, time, shutil, argparse, warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from uav_aoi_solver import Env, P
import multi_uav_solver as muv
from multi_uav_solver import fleet_rollout, fleet_post_process


# ─────────────────────────── config (CLI-overridable) ────────────────────────
M_LIST        = [50, 60, 80, 100, 120, 150, 200]
K_LIST        = [1, 2, 3, 4]
SEEDS         = [42, 123, 7]
N_INSTANCES   = 100
INSTANCE_SEED = 2025          # fixed -> identical instances across seeds & cells
HIDDEN        = 256
INPUT_DIM_DEF = 18


# ───────────────────────────────── helpers ───────────────────────────────────
def find_ckpt(M, K, seed, root):
    """Recursively locate fleet_M{M}_K{K}_split_seed{seed}.pt under `root`."""
    name = f'fleet_M{M}_K{K}_split_seed{seed}.pt'
    direct = os.path.join(root, name)
    if os.path.exists(direct):
        return direct
    hits = glob.glob(os.path.join(root, '**', name), recursive=True)
    return hits[0] if hits else None


def load_policy(path, K, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    pol = muv.MultiUAVPolicy(hidden=HIDDEN,
                             input_dim=ck.get('input_dim', INPUT_DIM_DEF)).to(device)
    pol.load_state_dict(ck['policy']); pol.eval()
    return pol, ck.get('Emax_each', P.Emax / K), ck['policy']


def eval_cell(pol, M, K, Emax_each, inst_seeds, device, do_pp):
    """Return (greedy_objs, pp_objs) arrays over the instance stream."""
    g_objs, pp_objs = [], []
    with torch.no_grad():
        for s in inst_seeds:
            env = Env(M=M, seed=s)
            f = fleet_rollout(pol, env, K, device, Emax_each=Emax_each, greedy=True)
            g_objs.append(f.fleet_objective())
            if do_pp:
                fp = fleet_post_process(env, f)
                pp_objs.append(fp.fleet_objective())
    g = np.array(g_objs, float)
    pp = np.array(pp_objs, float) if do_pp else g.copy()
    return g, pp


def make_soup(state_dicts):
    avg = copy.deepcopy(state_dicts[0])
    for k in avg:
        if torch.is_tensor(avg[k]) and avg[k].dtype.is_floating_point:
            avg[k] = sum(sd[k].float() for sd in state_dicts) / len(state_dicts)
    return avg


def stats(per_seed_means):
    a = np.array(per_seed_means, float)
    return dict(mean=float(a.mean()),
                std=float(a.std(ddof=1)) if a.size > 1 else 0.0,   # sample std
                best=float(a.min()), worst=float(a.max()))


# ───────────────────────────────── main ──────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='.', help='root to search for checkpoints')
    ap.add_argument('--out-dir',   default='./agg_out')
    ap.add_argument('--M', type=int, nargs='+', default=M_LIST)
    ap.add_argument('--K', type=int, nargs='+', default=K_LIST)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--instances', type=int, default=N_INSTANCES)
    ap.add_argument('--select-by', choices=['pp', 'greedy'], default='pp',
                    help='metric used to choose the best-of-3 checkpoint')
    ap.add_argument('--no-pp', action='store_true', help='greedy only (faster, less faithful)')
    ap.add_argument('--test-soup', action='store_true', help='also evaluate weight-averaged model')
    ap.add_argument('--force', action='store_true', help='recompute requested cells')
    ap.add_argument('--finalize-only', action='store_true', help='rebuild outputs from store, no eval')
    ap.add_argument('--cpu', action='store_true', help='force CPU even if CUDA present')
    args = ap.parse_args()

    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    do_pp  = not args.no_pp
    os.makedirs(args.out_dir, exist_ok=True)
    store_path = os.path.join(args.out_dir, '_agg_store.jsonl')

    # load resume store
    store = {}
    if os.path.exists(store_path):
        for line in open(store_path):
            line = line.strip()
            if line:
                d = json.loads(line); store[(d['M'], d['K'])] = d

    print(f"device={device}  instances={args.instances}  pp={'on' if do_pp else 'OFF'}  "
          f"select-by={args.select_by}  model-dir={os.path.abspath(args.model_dir)}")

    if not args.finalize_only:
        rng = np.random.default_rng(INSTANCE_SEED)
        inst_seeds = [int(rng.integers(0, 10_000_000)) for _ in range(args.instances)]
        cells = [(M, K) for M in args.M for K in args.K]
        t0 = time.perf_counter(); done = 0
        for (M, K) in cells:
            if (M, K) in store and not args.force:
                print(f"  skip M={M} K={K} (in store)"); continue
            Emax_each = None; sds = []
            per_g, per_pp, per_seed = [], [], {}
            missing = False
            for seed in args.seeds:
                path = find_ckpt(M, K, seed, args.model_dir)
                if path is None:
                    print(f"  [warn] missing checkpoint M={M} K={K} seed={seed} -> skipping cell")
                    missing = True; break
                pol, Emax_each, sd = load_policy(path, K, device)
                g, pp = eval_cell(pol, M, K, Emax_each, inst_seeds, device, do_pp)
                per_g.append(float(g.mean())); per_pp.append(float(pp.mean())); sds.append(sd)
                per_seed[str(seed)] = dict(greedy=float(g.mean()), pp=float(pp.mean()))
            if missing:
                continue

            gs, ps = stats(per_g), stats(per_pp)
            sel = per_pp if args.select_by == 'pp' else per_g
            best_idx = int(np.argmin(sel))
            best_seed = int(args.seeds[best_idx])

            soup_pp = float('nan')
            if args.test_soup and len(sds) == len(args.seeds):
                spol = muv.MultiUAVPolicy(hidden=HIDDEN, input_dim=INPUT_DIM_DEF).to(device)
                spol.load_state_dict(make_soup(sds)); spol.eval()
                sg, spp = eval_cell(spol, M, K, Emax_each, inst_seeds, device, do_pp)
                soup_pp = float(spp.mean())

            d = dict(M=M, K=K, N=args.instances, do_pp=do_pp, select_by=args.select_by,
                     greedy_mean=gs['mean'], greedy_std=gs['std'],
                     pp_mean=ps['mean'], pp_std=ps['std'],
                     pp_best=ps['best'], pp_worst=ps['worst'],
                     greedy_best=gs['best'], greedy_worst=gs['worst'],
                     best_seed=best_seed, per_seed=per_seed, soup_pp=soup_pp)
            store[(M, K)] = d
            with open(store_path, 'a') as f:
                f.write(json.dumps(d) + '\n')

            done += 1; el = time.perf_counter() - t0
            eta = el / done * (len([c for c in cells if c not in store or args.force]) - done)
            head = ps if do_pp else gs
            print(f"  M={M:>3} K={K}  {'pp' if do_pp else 'greedy'} {head['mean']:+.2f} "
                  f"+/- {head['std']:.2f}  [best seed {best_seed} -> {head['best']:+.2f}]"
                  f"   ({el:.0f}s, ~{eta:.0f}s left)")

        # dedupe/sort store
        with open(store_path, 'w') as f:
            for k in sorted(store):
                f.write(json.dumps(store[k]) + '\n')

    # ───────────────────────── build outputs from store ──────────────────────
    rows = [store[k] for k in sorted(store)]
    if not rows:
        print("no cells in store; nothing to write."); return
    Ms = sorted({r['M'] for r in rows}); Ks = sorted({r['K'] for r in rows})
    ppm = {(r['M'], r['K']): r['pp_mean'] for r in rows}
    pps = {(r['M'], r['K']): r['pp_std'] for r in rows}
    gdm = {(r['M'], r['K']): r['greedy_mean'] for r in rows}
    gds = {(r['M'], r['K']): r['greedy_std'] for r in rows}

    grid_path = os.path.join(args.out_dir, 'fleet_objective_grid.txt')
    with open(grid_path, 'w') as f:
        f.write("FINAL FLEET OBJECTIVE (lower = better), aggregated over seeds "
                f"{args.seeds}\n")
        f.write(f"Split battery (Emax_each = Emax/K).  N={rows[0]['N']} instances/seed, "
                f"identical across seeds.  mean +/- sample-std over {len(args.seeds)} seeds.\n")
        f.write("=" * 80 + "\n\n")
        f.write("[A] MLP+pp objective  (mean +/- std)\n")
        f.write("       K=" + "".join(f"{k:>16}" for k in Ks) + "\n")
        for M in Ms:
            f.write(f"M={M:>3}    " + "".join(
                f"{ppm.get((M,k),float('nan')):>9.2f}+-{pps.get((M,k),0):<5.2f}" for k in Ks) + "\n")
        f.write("\n[B] Raw greedy objective  (mean +/- std)\n")
        f.write("       K=" + "".join(f"{k:>16}" for k in Ks) + "\n")
        for M in Ms:
            f.write(f"M={M:>3}    " + "".join(
                f"{gdm.get((M,k),float('nan')):>9.2f}+-{gds.get((M,k),0):<5.2f}" for k in Ks) + "\n")
    print(f"\n  wrote {grid_path}")

    # detailed CSV
    csv_path = os.path.join(args.out_dir, 'fleet_objective_aggregated.csv')
    with open(csv_path, 'w') as f:
        cols = ["M", "K", "N", "pp_mean", "pp_std", "pp_best", "pp_worst",
                "greedy_mean", "greedy_std", "best_seed"]
        cols += [f"pp_seed{s}" for s in args.seeds] + [f"greedy_seed{s}" for s in args.seeds]
        cols += ["soup_pp"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            psd = r.get('per_seed', {})
            vals = [r['M'], r['K'], r['N'],
                    f"{r['pp_mean']:.3f}", f"{r['pp_std']:.3f}",
                    f"{r['pp_best']:.3f}", f"{r['pp_worst']:.3f}",
                    f"{r['greedy_mean']:.3f}", f"{r['greedy_std']:.3f}", r['best_seed']]
            vals += [f"{psd.get(str(s),{}).get('pp', float('nan')):.3f}" for s in args.seeds]
            vals += [f"{psd.get(str(s),{}).get('greedy', float('nan')):.3f}" for s in args.seeds]
            vals += [f"{r.get('soup_pp', float('nan')):.3f}"]
            f.write(",".join(str(v) for v in vals) + "\n")
    print(f"  wrote {csv_path}")

    # best-of-3 checkpoints + manifest
    best_dir = os.path.join(args.out_dir, 'fleet_models_best')
    os.makedirs(best_dir, exist_ok=True)
    man = os.path.join(best_dir, 'manifest.csv')
    n_copied = 0
    with open(man, 'w') as f:
        f.write("M,K,chosen_seed,select_by,chosen_obj,source_file,dest_file\n")
        for r in rows:
            M, K, bs = r['M'], r['K'], r['best_seed']
            src = find_ckpt(M, K, bs, args.model_dir)
            metric = 'pp' if r.get('do_pp', True) and r.get('select_by','pp') == 'pp' else 'greedy'
            obj = r['pp_best'] if metric == 'pp' else r['greedy_best']
            dst = os.path.join(best_dir, f'fleet_M{M}_K{K}_best.pt')
            if src:
                shutil.copy(src, dst); n_copied += 1
                f.write(f"{M},{K},{bs},{metric},{obj:.3f},{os.path.basename(src)},"
                        f"{os.path.basename(dst)}\n")
    print(f"  wrote {best_dir}/ ({n_copied} best-of-{len(args.seeds)} checkpoints + manifest.csv)")

    if args.test_soup:
        print("\n  weight-soup diagnostic (pp):")
        for r in rows:
            if not np.isnan(r.get('soup_pp', float('nan'))):
                verdict = 'usable' if r['soup_pp'] <= r['pp_mean'] + 0.05*abs(r['pp_mean']) else 'DEGRADED'
                print(f"    M={r['M']:>3} K={r['K']}  best={r['pp_best']:+.2f} "
                      f"mean={r['pp_mean']:+.2f}  SOUP={r['soup_pp']:+.2f}  {verdict}")

    print("\n  done.")


if __name__ == '__main__':
    main()
