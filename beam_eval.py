"""
beam_eval.py
================================================================================
Quantifies what fleet beam search buys over greedy, across the trained grid.
For every (M, K) it rolls out BOTH ways on a COMMON instance set and reports:

    greedy(pre)   fleet_objective of the raw greedy rollout        (current default)
    greedy+pp     ... after fleet_post_process                     (current HEADLINE routing)
    beam(pre)     fleet_objective of the raw beam rollout
    beam+pp       ... after fleet_post_process
    d_raw         beam(pre) - greedy(pre)     (raw-policy improvement; <0 = better)
    recovery%     100 * (greedy_pre - beam_pre) / (greedy_pre - greedy_pp)
                  = fraction of the post-process gain that beam alone captures
                    at the rollout stage (>100% => beam(pre) already beats greedy+pp)
    d_final       beam+pp - greedy+pp         (the decision number; <0 = beam worth it)

The instance set uses INSTANCE_SEED = 2025 and the same draw as deconflict_eval.py,
so beam(pre)/beam+pp are directly stitchable onto the deconfliction figure's
'routing' column. Bounds / optimality gap still compare to fleet_objective (the
routing column) — beam changes only which frozen-policy trajectory we evaluate,
never the objective, reward, or training, so separability and all bounds hold.

OUTPUT (into --out-dir, default ./beam_out):
    beam_table.txt      per (M,K): greedy/beam pre & +pp, d_raw, recovery%, d_final
    beam.csv            machine-readable incl. per-seed values
    _beam_store.jsonl   resume store (delete to recompute)

RUN:
    python beam_eval.py --model-dir models_multi_uav
    python beam_eval.py --M 100 200 --K 2 3 4 5 6 8 --instances 30 --beam-width 5
    python beam_eval.py --finalize-only        # rebuild table/csv from store
"""
from __future__ import annotations
import os, sys, glob, json, time, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from uav_aoi_solver import Env, P
import multi_uav_solver as muv
from multi_uav_solver import fleet_rollout, fleet_post_process
# beam lives in fleet_beam.py; allow it to also be pasted into multi_uav_solver.
try:
    from multi_uav_solver import fleet_rollout_beam
except ImportError:
    from fleet_beam import fleet_rollout_beam

SEEDS_DEFAULT = [42, 123, 7]
KS_DEFAULT    = [1, 2, 3, 4, 5, 6, 8]
MS_DEFAULT    = [50, 60, 80, 100, 120, 150, 200]
INSTANCE_SEED = 2025          # MUST match deconflict_eval.py for a stitchable set


def find_ckpt(M, K, seed, root):
    name = f'fleet_M{M}_K{K}_split_seed{seed}.pt'
    if os.path.exists(os.path.join(root, name)):
        return os.path.join(root, name)
    hits = glob.glob(os.path.join(root, '**', name), recursive=True)
    return hits[0] if hits else None


def eval_seed_cell(pol, M, K, Emax_each, inst_seeds, beam_width, device):
    """Per-instance greedy vs beam, each with and without post-process."""
    g_pre, g_pp, b_pre, b_pp = [], [], [], []
    with torch.no_grad():
        for s in inst_seeds:
            env = Env(M=M, seed=s)
            # greedy path
            gf = fleet_rollout(pol, env, K, device, Emax_each=Emax_each, greedy=True)
            g_pre.append(gf.fleet_objective())
            gf = fleet_post_process(env, gf)
            g_pp.append(gf.fleet_objective())
            # beam path (independent fresh rollout; post_process mutates its own fleet)
            bf = fleet_rollout_beam(pol, env, K, device, Emax_each=Emax_each,
                                    beam_width=beam_width, include_greedy=True)
            b_pre.append(bf.fleet_objective())
            bf = fleet_post_process(env, bf)
            b_pp.append(bf.fleet_objective())
    g_pp = np.array(g_pp); b_pp = np.array(b_pp)
    # PORTFOLIO: per-instance best of greedy+pp and beam+pp. Legit at test time
    # (fleet_objective is computable, no oracle) -> guarantees no regression.
    port_pp = np.minimum(g_pp, b_pp)
    return dict(g_pre=np.array(g_pre), g_pp=g_pp,
                b_pre=np.array(b_pre), b_pp=b_pp, port_pp=port_pp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='models_multi_uav')
    ap.add_argument('--out-dir',   default='./beam_out')
    ap.add_argument('--M', type=int, nargs='+', default=MS_DEFAULT)
    ap.add_argument('--K', type=int, nargs='+', default=KS_DEFAULT)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS_DEFAULT)
    ap.add_argument('--instances', type=int, default=30)
    ap.add_argument('--beam-width', type=int, default=5)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--finalize-only', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    os.makedirs(args.out_dir, exist_ok=True)
    store_path = os.path.join(args.out_dir, '_beam_store.jsonl')

    store = {}
    if os.path.exists(store_path):
        for line in open(store_path):
            line = line.strip()
            if line:
                d = json.loads(line); store[(d['M'], d['K'])] = d

    print(f"device={device}  instances={args.instances}  beam_width={args.beam_width}")
    print(f"model-dir={os.path.abspath(args.model_dir)}")

    if not args.finalize_only:
        rng = np.random.default_rng(INSTANCE_SEED)
        inst_seeds = [int(rng.integers(0, 10_000_000)) for _ in range(args.instances)]
        t0 = time.perf_counter()
        for M in args.M:
            for K in args.K:
                if (M, K) in store and not args.force:
                    print(f"  skip M={M} K={K} (in store)"); continue
                per_seed = []
                for seed in args.seeds:
                    path = find_ckpt(M, K, seed, args.model_dir)
                    if path is None:
                        continue
                    ck = torch.load(path, map_location=device, weights_only=False)
                    pol = muv.MultiUAVPolicy(hidden=256,
                                             input_dim=ck.get('input_dim', 18)).to(device)
                    pol.load_state_dict(ck['policy']); pol.eval()
                    Ee = ck.get('Emax_each', P.Emax / K)
                    r = eval_seed_cell(pol, M, K, Ee, inst_seeds,
                                       args.beam_width, device)
                    per_seed.append(dict(g_pre=float(r['g_pre'].mean()),
                                         g_pp=float(r['g_pp'].mean()),
                                         b_pre=float(r['b_pre'].mean()),
                                         b_pp=float(r['b_pp'].mean()),
                                         port_pp=float(r['port_pp'].mean())))
                if not per_seed:
                    print(f"  M={M} K={K}: [no checkpoint]"); continue

                def ms(key):
                    a = np.array([p[key] for p in per_seed], float)
                    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)
                gpre_m, gpre_s = ms('g_pre'); gpp_m, gpp_s = ms('g_pp')
                bpre_m, bpre_s = ms('b_pre'); bpp_m, bpp_s = ms('b_pp')
                port_m, port_s = ms('port_pp')
                d_raw   = bpre_m - gpre_m
                d_final = bpp_m - gpp_m
                d_port  = port_m - gpp_m       # <= 0 by construction (portfolio)
                pp_gain = gpre_m - gpp_m                      # >0 (pp improves)
                recovery = 100.0 * (gpre_m - bpre_m) / pp_gain if abs(pp_gain) > 1e-9 else float('nan')
                d = dict(M=M, K=K, n_seeds=len(per_seed), instances=args.instances,
                         beam_width=args.beam_width,
                         greedy_pre_mean=gpre_m, greedy_pre_std=gpre_s,
                         greedy_pp_mean=gpp_m,   greedy_pp_std=gpp_s,
                         beam_pre_mean=bpre_m,   beam_pre_std=bpre_s,
                         beam_pp_mean=bpp_m,     beam_pp_std=bpp_s,
                         port_pp_mean=port_m,    port_pp_std=port_s,
                         d_raw=d_raw, d_final=d_final, d_port=d_port,
                         recovery_pct=recovery,
                         per_seed=[dict(p) for p in per_seed])
                store[(M, K)] = d
                with open(store_path, 'a') as f:
                    f.write(json.dumps(d) + '\n')
                print(f"  M={M:>3} K={K}  g+pp {gpp_m:+.2f}  b+pp {bpp_m:+.2f}  "
                      f"portf {port_m:+.2f}  d_raw {d_raw:+.2f}  "
                      f"d_final {d_final:+.2f}  d_port {d_port:+.2f}"
                      f"   ({time.perf_counter()-t0:.0f}s)")
        with open(store_path, 'w') as f:
            for k in sorted(store):
                f.write(json.dumps(store[k]) + '\n')

    rows = [store[k] for k in sorted(store)]
    if not rows:
        print("nothing in store."); return

    # ---- table ----
    bw = rows[0].get('beam_width', args.beam_width)
    tbl = os.path.join(args.out_dir, 'beam_table.txt')
    with open(tbl, 'w') as f:
        f.write("FLEET BEAM SEARCH vs GREEDY — raw-policy routing objective\n")
        f.write(f"beam_width={bw}, common {rows[0]['instances']}-instance set "
                f"(INSTANCE_SEED={INSTANCE_SEED}), +/- = std over seeds. lower = better.\n")
        f.write("RAW: d_raw = beam(pre)-greedy(pre) (<=0 always; the clean methodological gain).\n")
        f.write("FINAL: greedy+pp = current headline.  beam+pp can regress (post-process "
                "insertion dominates at low K).\n")
        f.write("portfolio = per-instance best(greedy+pp, beam+pp).  d_port = portfolio - "
                "greedy+pp  (<=0 by construction: the safe headline).\n")
        f.write("=" * 118 + "\n")
        f.write(f"{'M':>4}{'K':>3}  {'g(pre)':>10}{'b(pre)':>10}{'d_raw':>8}  |"
                f"{'greedy+pp':>11}{'beam+pp':>11}{'portfolio':>11}{'d_final':>9}{'d_port':>9}\n")
        f.write("-" * 118 + "\n")
        for r in rows:
            port = r.get('port_pp_mean', float('nan'))
            dprt = r.get('d_port', float('nan'))
            f.write(f"{r['M']:>4}{r['K']:>3}  "
                    f"{r['greedy_pre_mean']:>10.2f}{r['beam_pre_mean']:>10.2f}"
                    f"{r['d_raw']:>8.2f}  |"
                    f"{r['greedy_pp_mean']:>11.2f}{r['beam_pp_mean']:>11.2f}"
                    f"{port:>11.2f}{r['d_final']:>9.2f}{dprt:>9.2f}\n")
        f.write("-" * 118 + "\n")
    print('\n' + open(tbl).read())
    print(f"  wrote {tbl}")

    # ---- csv ----
    csv = os.path.join(args.out_dir, 'beam.csv')
    with open(csv, 'w') as f:
        cols = ["M", "K", "n_seeds", "instances", "beam_width",
                "greedy_pre_mean", "greedy_pre_std", "greedy_pp_mean", "greedy_pp_std",
                "beam_pre_mean", "beam_pre_std", "beam_pp_mean", "beam_pp_std",
                "port_pp_mean", "port_pp_std",
                "d_raw", "d_final", "d_port", "recovery_pct"]
        seedcols = [f"port_pp_seed{i}" for i in range(3)]
        f.write(",".join(cols + seedcols) + "\n")
        for r in rows:
            base = [r.get(c, "") for c in cols]
            pp = [f"{p['port_pp']:.3f}" for p in r.get('per_seed', [])]
            pp += [""] * (3 - len(pp))
            f.write(",".join(str(x) for x in base + pp) + "\n")
    print(f"  wrote {csv}")
    print("\n  READ: d_raw < 0 everywhere = beam always improves the raw rollout (clean claim).")
    print("  d_port <= 0 by construction = the portfolio headline never regresses; its size")
    print("  is beam's real contribution to the final number. beam+pp alone can regress at")
    print("  low K where post-process insertion dominates - that's why we report portfolio.")


if __name__ == '__main__':
    main()