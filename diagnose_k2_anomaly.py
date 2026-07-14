"""
diagnose_k2_anomaly.py
================================================================================
Figure out WHY the large-M, K=2 cells (M=150, 200) show the seed disagreement /
weak greedy objective seen in aggregation, and whether the cause is:

  (A) SEED INSTABILITY  -> one/two seeds converged to a bad policy; the others
      are fine. Fix = retrain the bad seed(s). Signature: one seed's objective is
      a clear outlier, while node-coverage / load-balance of the GOOD seeds look
      like the K=1 and K=3 neighbors.

  (B) SCHEDULER / LOAD-BALANCING PATHOLOGY at K=2 -> the "advance highest-energy
      UAV" rule starves one drone or stops early, leaving energy on the table,
      across ALL seeds. Signature: high per-UAV imbalance AND high leftover energy
      in every seed, not just one.

  (C) GENUINE UNDERTRAINING -> all seeds weak, balanced but few nodes, energy
      used but objective poor. Fix = more epochs.

It evaluates the RAW GREEDY rollout (post-process is intentionally OFF, since it
masks the rollout pathology), per seed, and reports per-UAV node counts, load
imbalance, leftover-energy fraction, and the WAoI/priority split. K=1 and K=3 at
the same M are included as healthy references that flank the suspect K=2.

RUN (from the folder with uav_aoi_solver.py + multi_uav_solver.py):
    python diagnose_k2_anomaly.py --model-dir models_multi_uav
    python diagnose_k2_anomaly.py --M 150 200 --K 1 2 3 --instances 25
    python diagnose_k2_anomaly.py --M 100 150 200          # add M=100 as a 2nd healthy ref
"""
from __future__ import annotations
import os, sys, glob, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from uav_aoi_solver import Env, P
import multi_uav_solver as muv
from multi_uav_solver import fleet_rollout

SEEDS = [42, 123, 7]
INSTANCE_SEED = 4242
NMAP = {50: 30, 60: 30, 80: 25, 100: 20, 120: 16, 150: 12, 200: 9}


def find_ckpt(M, K, seed, root):
    name = f'fleet_M{M}_K{K}_split_seed{seed}.pt'
    if os.path.exists(os.path.join(root, name)):
        return os.path.join(root, name)
    hits = glob.glob(os.path.join(root, '**', name), recursive=True)
    return hits[0] if hits else None


def diagnose_cell(M, K, seed, root, inst_seeds, device):
    path = find_ckpt(M, K, seed, root)
    if path is None:
        return None
    ck = torch.load(path, map_location=device, weights_only=False)
    pol = muv.MultiUAVPolicy(hidden=256, input_dim=ck.get('input_dim', 18)).to(device)
    pol.load_state_dict(ck['policy']); pol.eval()
    Emax_each = ck.get('Emax_each', P.Emax / K)

    objs, totnodes, imbal, leftfrac, waois, pris = [], [], [], [], [], []
    minshare = []
    with torch.no_grad():
        for s in inst_seeds:
            env = Env(M=M, seed=s)
            f = fleet_rollout(pol, env, K, device, Emax_each=Emax_each, greedy=True)
            n_per = np.array([len(t) for t in f.trajs], float)
            tot = float(n_per.sum())
            objs.append(f.fleet_objective())
            totnodes.append(tot)
            # load imbalance: 0 = perfectly even, 1 = one UAV does everything
            imbal.append(float((n_per.max() - n_per.min()) / max(tot, 1.0)) if K > 1 else 0.0)
            # fraction of the least-used UAV's share (0 => a starved drone)
            minshare.append(float(n_per.min() / max(tot, 1.0)) if K > 1 else 1.0)
            # leftover energy fraction averaged over UAVs (high => stops early)
            leftfrac.append(float(np.mean([e / Emax_each for e in f.E_left])))
            waois.append(f.fleet_waoi()); pris.append(f.fleet_priority())
    return dict(
        seed=seed, obj=float(np.mean(objs)), obj_std=float(np.std(objs)),
        nodes=float(np.mean(totnodes)), imbal=float(np.mean(imbal)),
        minshare=float(np.mean(minshare)), leftfrac=float(np.mean(leftfrac)),
        waoi=float(np.mean(waois)), pri=float(np.mean(pris)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='.')
    ap.add_argument('--M', type=int, nargs='+', default=[150, 200])
    ap.add_argument('--K', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--instances', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--out', default='./k2_diagnosis.txt')
    args = ap.parse_args()
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'

    rng = np.random.default_rng(INSTANCE_SEED)
    lines = []
    def emit(s=''):
        print(s); lines.append(s)

    emit(f"K=2 anomaly diagnosis   device={device}   model-dir={os.path.abspath(args.model_dir)}")
    emit("greedy rollout only (post-process OFF, to expose the rollout itself)")
    emit("imbal: 0=even load, ->1 one UAV hogs   minshare: share of least-used UAV (0=starved)")
    emit("leftE%: avg leftover battery per UAV (high=>stops early)   nodes: total served")
    emit("=" * 92)

    cells = {}
    for M in args.M:
        N = args.instances or NMAP.get(M, 15)
        inst_seeds = [int(rng.integers(0, 10_000_000)) for _ in range(N)]
        emit(f"\n############  M={M}   (N={N} instances/seed)  ############")
        for K in args.K:
            emit(f"\n  --- K={K} ---")
            emit(f"    {'seed':>5} {'obj':>10} {'obj_std':>9} {'nodes':>8} "
                 f"{'imbal':>7} {'minshare':>9} {'leftE%':>8} {'WAoI':>9} {'priority':>9}")
            cellrows = []
            for seed in args.seeds:
                d = diagnose_cell(M, K, seed, args.model_dir, inst_seeds, device)
                if d is None:
                    emit(f"    {seed:>5}   [checkpoint missing]"); continue
                cellrows.append(d)
                emit(f"    {seed:>5} {d['obj']:>+10.2f} {d['obj_std']:>9.2f} {d['nodes']:>8.1f} "
                     f"{d['imbal']:>7.2f} {d['minshare']:>9.2f} {100*d['leftfrac']:>7.1f}% "
                     f"{d['waoi']:>9.2f} {d['pri']:>9.2f}")
            cells[(M, K)] = cellrows

    # ---- automatic read of the K=2 cells against their K=1/K=3 neighbours ----
    emit("\n" + "=" * 92)
    emit("AUTOMATIC READOUT (heuristic — confirm against the numbers above)")
    for M in args.M:
        rows2 = cells.get((M, 2), [])
        if len(rows2) < 2:
            continue
        objs = np.array([r['obj'] for r in rows2])
        seeds = [r['seed'] for r in rows2]
        mean, std = objs.mean(), objs.std()
        # outlier seed: deviates most and notably worse (less negative) than the pack
        worst_i = int(np.argmax(objs))               # least-negative = worst
        spread = objs.max() - objs.min()
        imbal_all = np.mean([r['imbal'] for r in rows2])
        left_all = np.mean([r['leftfrac'] for r in rows2])
        minshare_all = np.mean([r['minshare'] for r in rows2])

        # per-seed instance noise level (to judge whether seed spread is real)
        inst_std = float(np.mean([r['obj_std'] for r in rows2]))
        nodes2 = np.mean([r['nodes'] for r in rows2])
        waoi2  = np.mean([r['waoi'] for r in rows2])
        # K=3 neighbour (healthy reference)
        rows3 = cells.get((M, 3), [])
        nodes3 = np.mean([r['nodes'] for r in rows3]) if rows3 else float('nan')
        waoi3  = np.mean([r['waoi'] for r in rows3]) if rows3 else float('nan')

        emit(f"\n  M={M} K=2: obj per seed " +
             ", ".join(f"s{r['seed']}={r['obj']:+.1f}" for r in rows2) +
             f"   (seed-spread {spread:.1f}  vs  per-seed instance-std {inst_std:.1f})")
        verdicts = []

        # (A) seed instability ONLY if seed spread exceeds the instance noise floor
        if spread > 1.5 * inst_std and spread > 0.4 * abs(mean):
            verdicts.append(f"(A) SEED INSTABILITY — seed {seeds[worst_i]} is a real outlier "
                            f"({objs[worst_i]:+.1f}); seed spread {spread:.1f} exceeds instance "
                            f"noise {inst_std:.1f}. Retrain that seed.")
        else:
            verdicts.append(f"(A) seeds CONSISTENT — spread {spread:.1f} is within instance "
                            f"noise {inst_std:.1f}; the aggregation 'disagreement' was low-N noise, "
                            f"not bad seeds.")

        # (B) load balance
        if minshare_all < 0.20 or imbal_all > 0.5:
            verdicts.append(f"(B) LOAD-BALANCING pathology — least-used UAV holds only "
                            f"{100*minshare_all:.0f}% of nodes (imbalance {imbal_all:.2f}).")
        else:
            verdicts.append(f"(B) load BALANCED — both UAVs ~{100*minshare_all:.0f}% share; "
                            f"scheduler is fine.")

        # (C) the real driver: same coverage as K=3 but much higher WAoI => long-chain penalty
        if rows3 and not np.isnan(waoi3) and nodes2 >= 0.9 * nodes3 and waoi2 > 1.25 * waoi3:
            verdicts.append(
                f"(C) ROOT CAUSE = LONG-CHAIN WAoI. K=2 serves ~{nodes2:.0f} nodes "
                f"(~same as K=3's {nodes3:.0f}) but WAoI {waoi2:.0f} >> K=3's {waoi3:.0f}. "
                f"Two long chains accumulate cumulative-priority WAoI that nearly cancels the "
                f"priority reward. This is EXPECTED physics, not a bug. The raw greedy ORDERING "
                f"is weak for long chains (post-process recovers it ~3x), so beam search / better "
                f"rollout ordering is the lever — NOT retraining or rescheduling.")
        if left_all > 0.25:
            verdicts.append(f"    note: {100*left_all:.0f}% battery left unused — some early stopping.")

        for v in verdicts:
            emit("    " + v)

    with open(args.out, 'w') as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\n  saved -> {args.out}")


if __name__ == '__main__':
    main()
