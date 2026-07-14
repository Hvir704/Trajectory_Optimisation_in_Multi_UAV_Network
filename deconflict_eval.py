"""
deconflict_eval.py
================================================================================
Post-hoc temporal deconfliction across the trained fleet grid. For every (M, K)
it rolls out the policy (greedy -> post-process = your headline routing), runs
the launch-deconfliction scheduler, and reports the FINAL collision-avoided
objective per cell, aggregated over seeds on a common instance set.

FINAL objective (what the fleet actually achieves after deconfliction):
    final = routing_objective(pp)  +  theta1 * sum_k o_k * W_total(chain_k)
            _____ fleet_objective _____   ____ deconfliction AoI penalty ____
The penalty uses the COMMON-t=0 AoI model you adopted. Plain fleet_objective is
the routing result that your bounds / optimality gap compare against; the final
column is routing + the safe-execution overhead. Reported side by side.

OUTPUT (into --out-dir, default ./deconflict_out):
    deconflict_table.txt    per (M,K): routing, penalty, makespan, final (mean+/-std)
    deconflict.csv          machine-readable, incl. per-seed values
    _deconflict_store.jsonl resume store (delete to recompute)

REQUIREMENTS: run from the folder with uav_aoi_solver.py and the DECONFLICTION-
ENABLED multi_uav_solver.py (the one with deconfliction_schedule). Checkpoints
named fleet_M{M}_K{K}_split_seed{seed}.pt, searched recursively under --model-dir.

RUN:
    python deconflict_eval.py --model-dir models_multi_uav
    python deconflict_eval.py --M 100 200 --K 1 2 3 4 5 6 8 --instances 30 --dt 0.15
    python deconflict_eval.py --finalize-only        # rebuild table/csv from store
"""
from __future__ import annotations
import os, sys, glob, json, time, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from uav_aoi_solver import Env, P
import multi_uav_solver as muv
from multi_uav_solver import (fleet_rollout, fleet_post_process,
                              deconfliction_schedule)

SEEDS_DEFAULT = [42, 123, 7]
KS_DEFAULT    = [1, 2, 3, 4, 5, 6, 8]
MS_DEFAULT    = [50, 60, 80, 100, 120, 150, 200]
INSTANCE_SEED = 2025


def find_ckpt(M, K, seed, root):
    name = f'fleet_M{M}_K{K}_split_seed{seed}.pt'
    if os.path.exists(os.path.join(root, name)):
        return os.path.join(root, name)
    hits = glob.glob(os.path.join(root, '**', name), recursive=True)
    return hits[0] if hits else None


def recompute_Wcum(fleet, env):
    """Post-process rearranges nodes across chains; make sure each chain's
    cumulative priority W_cum (used by the deconfliction penalty) reflects the
    FINAL trajectories, not the pre-post-process state."""
    fleet.W_cum = [float(sum(env.wi[j] for j in t)) for t in fleet.trajs]
    return fleet


def eval_seed_cell(pol, M, K, Emax_each, inst_seeds, delta, dt, device):
    """Per-instance routing + deconfliction for one seed model. Returns dict of
    arrays over instances."""
    rt_g, rt_pp, pen, mk, fin, confl = [], [], [], [], [], 0
    with torch.no_grad():
        for s in inst_seeds:
            env = Env(M=M, seed=s)
            f = fleet_rollout(pol, env, K, device, Emax_each=Emax_each, greedy=True)
            rt_g.append(f.fleet_objective())
            f = fleet_post_process(env, f)
            f = recompute_Wcum(f, env)               # <- defensive: keep W_cum current
            routing = f.fleet_objective()
            rt_pp.append(routing)
            if K >= 2:
                res = deconfliction_schedule(f, delta=delta, dt=dt,
                                             optimize_order=True, verify=True)
                pen.append(res['aoi_penalty']); mk.append(res['makespan'])
                confl += int(res['conflicts_left'])
                fin.append(routing + res['aoi_penalty'])
            else:                                    # K=1: nothing to deconflict
                pen.append(0.0); mk.append(0.0); fin.append(routing)
    return dict(rt_g=np.array(rt_g), rt_pp=np.array(rt_pp), pen=np.array(pen),
                mk=np.array(mk), fin=np.array(fin), conflicts=confl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='models_multi_uav')
    ap.add_argument('--out-dir',   default='./deconflict_out')
    ap.add_argument('--M', type=int, nargs='+', default=MS_DEFAULT)
    ap.add_argument('--K', type=int, nargs='+', default=KS_DEFAULT)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS_DEFAULT)
    ap.add_argument('--instances', type=int, default=30)
    ap.add_argument('--delta', type=float, default=25.0)
    ap.add_argument('--dt', type=float, default=0.15,
                    help='scheduler time resolution; lower = tighter (less over-charged) cost')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--finalize-only', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    os.makedirs(args.out_dir, exist_ok=True)
    store_path = os.path.join(args.out_dir, '_deconflict_store.jsonl')

    store = {}
    if os.path.exists(store_path):
        for line in open(store_path):
            line = line.strip()
            if line:
                d = json.loads(line); store[(d['M'], d['K'])] = d

    print(f"device={device}  instances={args.instances}  delta={args.delta}m  dt={args.dt}s")
    print(f"model-dir={os.path.abspath(args.model_dir)}")

    if not args.finalize_only:
        rng = np.random.default_rng(INSTANCE_SEED)
        inst_seeds = [int(rng.integers(0, 10_000_000)) for _ in range(args.instances)]
        t0 = time.perf_counter()
        for M in args.M:
            for K in args.K:
                if (M, K) in store and not args.force:
                    print(f"  skip M={M} K={K} (in store)"); continue
                per_seed = []      # list of dict-of-means per seed
                tot_confl = 0
                for seed in args.seeds:
                    path = find_ckpt(M, K, seed, args.model_dir)
                    if path is None:
                        continue
                    ck = torch.load(path, map_location=device, weights_only=False)
                    pol = muv.MultiUAVPolicy(hidden=256,
                                             input_dim=ck.get('input_dim', 18)).to(device)
                    pol.load_state_dict(ck['policy']); pol.eval()
                    Ee = ck.get('Emax_each', P.Emax / K)
                    r = eval_seed_cell(pol, M, K, Ee, inst_seeds, args.delta, args.dt, device)
                    tot_confl += r['conflicts']
                    per_seed.append(dict(rt_pp=float(r['rt_pp'].mean()),
                                         pen=float(r['pen'].mean()),
                                         mk=float(r['mk'].mean()),
                                         fin=float(r['fin'].mean())))
                if not per_seed:
                    print(f"  M={M} K={K}: [no checkpoint]"); continue

                def ms(key):
                    a = np.array([p[key] for p in per_seed], float)
                    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)
                rt_m, rt_s = ms('rt_pp'); pen_m, pen_s = ms('pen')
                mk_m, _    = ms('mk');    fin_m, fin_s = ms('fin')
                pen_pct = 100.0 * pen_m / max(abs(rt_m), 1e-9)
                d = dict(M=M, K=K, n_seeds=len(per_seed), instances=args.instances,
                         routing_mean=rt_m, routing_std=rt_s,
                         penalty_mean=pen_m, penalty_std=pen_s, penalty_pct=pen_pct,
                         makespan_mean=mk_m, final_mean=fin_m, final_std=fin_s,
                         conflicts_left=tot_confl,
                         per_seed=[dict(p) for p in per_seed])
                store[(M, K)] = d
                with open(store_path, 'a') as f:
                    f.write(json.dumps(d) + '\n')
                flag = '' if tot_confl == 0 else f'  [!] {tot_confl} residual conflicts'
                print(f"  M={M:>3} K={K}  routing {rt_m:+.2f}  penalty {pen_m:+.2f} "
                      f"({pen_pct:.1f}%)  makespan {mk_m:.1f}s  -> FINAL {fin_m:+.2f}"
                      f"   ({time.perf_counter()-t0:.0f}s){flag}")
        # dedupe/sort store
        with open(store_path, 'w') as f:
            for k in sorted(store):
                f.write(json.dumps(store[k]) + '\n')

    rows = [store[k] for k in sorted(store)]
    if not rows:
        print("nothing in store."); return

    # ---- table ----
    tbl = os.path.join(args.out_dir, 'deconflict_table.txt')
    with open(tbl, 'w') as f:
        f.write("POST-HOC TEMPORAL DECONFLICTION — final collision-avoided objective\n")
        f.write(f"delta={args.delta}m, common-t=0 AoI penalty = theta1*sum(o_k*W_total_k), "
                f"theta1={P.theta1}\n")
        f.write("routing = MLP+pp fleet_objective (what bounds compare to).  "
                "FINAL = routing + deconfliction penalty.\n")
        f.write("lower = better.  +/- = std over seeds.\n")
        f.write("=" * 104 + "\n")
        f.write(f"{'M':>4}{'K':>3}  {'routing(MLP+pp)':>20}{'penalty':>14}{'pen%':>7}"
                f"{'makespan_s':>12}{'FINAL(deconflicted)':>22}{'conf':>6}\n")
        f.write("-" * 104 + "\n")
        for r in rows:
            f.write(f"{r['M']:>4}{r['K']:>3}  "
                    f"{r['routing_mean']:>11.2f}+-{r['routing_std']:<6.2f}"
                    f"{r['penalty_mean']:>9.2f}+-{r['penalty_std']:<3.1f}"
                    f"{r['penalty_pct']:>6.1f}%"
                    f"{r['makespan_mean']:>12.1f}"
                    f"{r['final_mean']:>13.2f}+-{r['final_std']:<6.2f}"
                    f"{r['conflicts_left']:>6}\n")
        f.write("-" * 104 + "\n")
    print('\n' + open(tbl).read())
    print(f"  wrote {tbl}")

    # ---- csv ----
    csv = os.path.join(args.out_dir, 'deconflict.csv')
    with open(csv, 'w') as f:
        cols = ["M", "K", "n_seeds", "instances", "routing_mean", "routing_std",
                "penalty_mean", "penalty_std", "penalty_pct", "makespan_mean",
                "final_mean", "final_std", "conflicts_left"]
        seedcols = [f"final_seed{i}" for i in range(3)]
        f.write(",".join(cols + seedcols) + "\n")
        for r in rows:
            base = [r[c] for c in cols]
            fins = [f"{p['fin']:.3f}" for p in r.get('per_seed', [])]
            fins += [""] * (3 - len(fins))
            f.write(",".join(str(x) for x in base + fins) + "\n")
    print(f"  wrote {csv}")
    print("\n  NOTE: 'conf' must be 0 everywhere (deconfliction verified). Bounds/optimality")
    print("  gap stay on the 'routing' column; 'FINAL' is routing + safe-execution overhead.")


if __name__ == '__main__':
    main()