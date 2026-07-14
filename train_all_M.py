"""train_all_M.py  —  Train MOSAC-ATT Transformer on all M = 20..100
==================================================================

Uses the Tiny Transformer (d=64, 4 heads, 1 layer, ~129K params)
matched to MLP scale so REINFORCE gradients are effective.

MOSAC-ATT Fixes applied (both implemented in policy.py + trainer_fixed.py):
  Fix 1 — MaxPool set-representation: Baseline now uses max(node_emb, dim=1)
           instead of mean. The aggregate is M-independent in scale, so a
           single model can be evaluated zero-shot on any M without retraining.
  Fix 2 — Heated-up softmax: temperature tau anneals from tau_start=5.0 down
           to tau_final=1.0 over the first 40% of training epochs, forcing
           broad exploration early and committed selection late. The entropy
           target ceiling H_target also decays over the same window.

Saves to:   models_attn/attn_M{M}.pt     (compatible with all plot scripts)
History:    results_attn/attn_history_M{M}.npy

TIME ESTIMATE (GPU):
  All M=20..100: ~60-90 minutes total (Tiny config)
  Per M: 5-30 minutes depending on M

Usage:
  python train_all_M.py
  python train_all_M.py --M 30 50           # specific M only
  python train_all_M.py --epochs 400        # more epochs
  python train_all_M.py --tau_start 3.0     # gentler exploration schedule
  python train_all_M.py --skip_existing     # skip already-trained models
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

sys.path.insert(0, '.')
from env           import Params, UAVEnv
from policy        import AttentionPolicy
from trainer_fixed import FixedTrainer

os.makedirs('models_attn',  exist_ok=True)
os.makedirs('results_attn', exist_ok=True)

M_VALUES = [20, 30, 40, 50, 60, 70, 80, 90, 100]

# Tiny transformer — matches MLP scale for reliable REINFORCE
TINY = dict(d_model=64, n_heads=4, n_layers=1)

# ─────────────────────────────────────────────────────────────────────────────
# Time estimate table
# ─────────────────────────────────────────────────────────────────────────────
def time_estimate(M_list: list, n_epochs: int):
    base_s = 1.2   # seconds/epoch at M=30 for Tiny transformer on CPU
    print('─' * 55)
    print(f'{"M":>5} {"s/epoch":>9} {"minutes":>9} {"cumul hrs":>11}')
    print('─' * 55)
    total = 0
    for M in M_list:
        scale  = 0.6 + 0.4 * (M / 30) ** 2
        ep_s   = base_s * scale * 0.3   # Tiny is ~3x faster than original
        M_tot  = n_epochs * ep_s
        total += M_tot
        print(f'{M:>5} {ep_s:>9.1f}s {M_tot/60:>9.0f}m {total/3600:>10.1f}h')
    print('─' * 55)
    print(f'{"TOTAL":>5} {"":>9} {total/60:>9.0f}m {total/3600:>10.1f}h')
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Train one M
# ─────────────────────────────────────────────────────────────────────────────
def train_one(M: int, n_epochs: int, device: str, seed: int,
              skip_existing: bool,
              tau_start: float, tau_final: float, T_anneal_frac: float,
              H_frac_start: float, H_frac_final: float):
    """
    Train a fresh FixedTrainer for a single M value.
    Model saved to models_attn/attn_M{M}.pt — the canonical path used by
    all evaluation and plotting scripts (generate_all_tiny_plots.py,
    generate_attn_advanced_plots.py, comparing_the_three.py, plot_nodes_vs_M.py).
    """
    save_path = f'models_attn/attn_M{M}.pt'

    if skip_existing and os.path.exists(save_path):
        print(f'  M={M}: model exists at {save_path} — skipping')
        return None

    params = Params(M=M)

    trainer = FixedTrainer(
        params                 = params,
        # ── architecture ───────────────────────────────────────────────
        d_model                = TINY['d_model'],
        n_heads                = TINY['n_heads'],
        n_layers               = TINY['n_layers'],
        # ── optimiser ──────────────────────────────────────────────────
        lr                     = 3e-4,
        # ── entropy schedule (Fix 2B) ───────────────────────────────────
        entropy_beta           = 0.01,
        entropy_warmup_epochs  = 80,
        H_target_frac_start    = H_frac_start,
        H_target_frac_final    = H_frac_final,
        # ── temperature schedule (Fix 2A) ───────────────────────────────
        tau_start              = tau_start,
        tau_final              = tau_final,
        T_anneal_frac          = T_anneal_frac,
        # ── rollout ─────────────────────────────────────────────────────
        episodes_per_epoch     = 128,
        adv_clip               = 5.0,
        device                 = device,
    )

    t0    = time.time()
    final = trainer.train(
        n_epochs   = n_epochs,
        save_path  = save_path,
        log_every  = 20,
        eval_every = 50,
        seed       = seed,
    )
    elapsed = (time.time() - t0) / 60

    print(f'  M={M} done in {elapsed:.0f}m  '
          f'| obj={final["obj"]:.2f}  nodes={final["nodes"]:.1f}')
    return final

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='Train MOSAC-ATT Tiny Transformer on all M values.')
    ap.add_argument('--M',             type=int,   nargs='+', default=None,
                    help='M values to train (default: all 20..100)')
    ap.add_argument('--epochs',        type=int,   default=300,
                    help='Training epochs per M (default: 300)')
    ap.add_argument('--device',        type=str,   default='',
                    help='Device: cuda / cpu (default: auto)')
    ap.add_argument('--seed',          type=int,   default=42)
    ap.add_argument('--skip_existing', action='store_true',
                    help='Skip M values where models_attn/attn_M{M}.pt exists')

    # Fix 2 schedule arguments
    ap.add_argument('--tau_start',     type=float, default=5.0,
                    help='Initial temperature tau_0 (default: 5.0)')
    ap.add_argument('--tau_final',     type=float, default=1.0,
                    help='Final temperature tau_f (default: 1.0)')
    ap.add_argument('--T_anneal_frac', type=float, default=0.4,
                    help='Fraction of epochs over which tau anneals (default: 0.4)')
    ap.add_argument('--H_frac_start',  type=float, default=0.8,
                    help='Initial H_target fraction of log(M)*steps (default: 0.8)')
    ap.add_argument('--H_frac_final',  type=float, default=0.3,
                    help='Final H_target fraction after annealing (default: 0.3)')
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    M_list = args.M or M_VALUES
    T_anneal_ep = max(1, int(args.epochs * args.T_anneal_frac))

    print('=' * 60)
    print('  MOSAC-ATT Tiny Transformer Training')
    print(f'  Fix 1: MaxPool baseline (M-independent scale)')
    print(f'  Fix 2: tau {args.tau_start:.1f}->{args.tau_final:.1f} '
          f'over {T_anneal_ep} epochs  |  '
          f'H_frac {args.H_frac_start:.2f}->{args.H_frac_final:.2f}')
    print(f'  Architecture: d={TINY["d_model"]}, '
          f'h={TINY["n_heads"]}, L={TINY["n_layers"]}  (~129K params)')
    print(f'  device={device}  epochs={args.epochs}  seed={args.seed}')
    print(f'  M values: {M_list}')
    print('=' * 60)
    print()

    print('Time estimate:')
    time_estimate(M_list, args.epochs)

    results   = {}
    wall_t0   = time.time()

    for i, M in enumerate(M_list):
        print(f'\n{"=" * 60}')
        print(f'  [{i+1}/{len(M_list)}] M={M}')
        print(f'{"=" * 60}')

        final = train_one(
            M              = M,
            n_epochs       = args.epochs,
            device         = device,
            seed           = args.seed,
            skip_existing  = args.skip_existing,
            tau_start      = args.tau_start,
            tau_final      = args.tau_final,
            T_anneal_frac  = args.T_anneal_frac,
            H_frac_start   = args.H_frac_start,
            H_frac_final   = args.H_frac_final,
        )
        if final:
            results[M] = final

    total_h = (time.time() - wall_t0) / 3600

    print(f'\n{"=" * 60}')
    print(f'  Done in {total_h:.1f} hours')
    print(f'{"=" * 60}')

    if results:
        print(f'\n{"M":>5} {"Obj":>8} {"Nodes":>7} {"WAoI":>8}')
        print('─' * 32)
        for M, r in sorted(results.items()):
            print(f'{M:>5} {r["obj"]:>8.2f} {r["nodes"]:>7.1f} {r["waoi"]:>8.1f}')

    print(f'\nModels saved → models_attn/attn_M{{M}}.pt')
    print(f'Histories    → results_attn/attn_history_M{{M}}.npy')
    print(f'Evaluation   → python generate_all_tiny_plots.py')

if __name__ == '__main__':
    main()