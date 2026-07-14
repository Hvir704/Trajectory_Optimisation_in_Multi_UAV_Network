"""
run.py — Train the Attention Policy and generate all comparison figures.

Usage:
    python run.py              # full run (~30-60 min on CPU, ~5-10 min GPU)
    python run.py --quick      # fast demo (~5 min)
    python run.py --eval_only  # skip training, load saved model and plot
    python run.py --train_all  # systematic multi-M training, save best models
                               # (can be combined with --quick)
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from env import Params, UAVEnv
from policy import AttentionPolicy
from trainer import Trainer
from baselines import evaluate_all, BASELINES, greedy_priority, nearest_neighbor, pdr, random_policy
from features import batch_rollout, obs_to_tensors

os.makedirs('results(attn)', exist_ok=True)
os.makedirs('models(attn)', exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────────────
C = {
    'Random':              '#888888',
    'Nearest-Neighbor':    '#E69F00',
    'Greedy-Priority':     '#56B4E9',
    'PDR':                 '#CC79A7',
    'Attention (Ours)':    '#009E73',
}
MARKER = {
    'Random': 's', 'Nearest-Neighbor': '^',
    'Greedy-Priority': 'D', 'PDR': 'v', 'Attention (Ours)': 'o'
}
METHODS = ['Random', 'Nearest-Neighbor', 'Greedy-Priority', 'PDR', 'Attention (Ours)']

MODEL_DIR    = 'models(attn)'
M_LIST_FULL  = [20, 30, 40, 50, 60, 70, 80, 90, 100]
M_LIST_QUICK = [20, 30, 40, 50]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_policy(path, params, device):
    policy = AttentionPolicy(d_model=128, n_heads=8, n_layers=3).to(device)
    ck = torch.load(path, map_location=device)
    policy.load_state_dict(ck['policy'])
    policy.eval()
    return policy


def eval_policy(policy, params, n=200, seed=9999, device='cpu'):
    rng  = np.random.default_rng(seed)
    res  = defaultdict(list)
    with torch.no_grad():
        for _ in range(n):
            s    = int(rng.integers(0, 10_000_000))
            env  = UAVEnv(params, seed=s)
            traj, _, _, _ = batch_rollout(policy, env, device, greedy=True)
            res['obj'].append(env.objective(traj))
            res['nodes'].append(len(traj))
            res['waoi'].append(params.theta1 * env.waoi(traj))
            res['priority'].append(sum(env.wi[j] for j in traj))
    return {k: float(np.mean(v)) for k, v in res.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Training convergence
# ─────────────────────────────────────────────────────────────────────────────
def plot_convergence(history, M):
    ep   = np.arange(1, len(history['reward'])+1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Attention Policy Training Convergence (M={M})', fontsize=13)

    for ax, key, ylabel, color in [
        (axes[0], 'reward',  'Episode Reward',    '#009E73'),
        (axes[1], 'nodes',   'Avg Nodes Visited', '#0072B2'),
        (axes[2], 'entropy', 'Policy Entropy',    '#D55E00'),
    ]:
        raw = np.array(history[key])
        k   = min(50, len(raw))
        sm  = np.convolve(raw, np.ones(k)/k, 'same')
        ax.plot(ep, raw,  color=color, alpha=0.2, lw=0.8)
        ax.plot(ep, sm,   color=color, lw=2.0, label='Smoothed')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.3)
    axes[1].axhline(9.9, color='red', ls='--', lw=1, label='Paper D3QN 9.9')
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    p = f'results(attn)/fig1_convergence_M{M}.png'
    plt.savefig(p, dpi=150); plt.close()
    print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Performance comparison table + bar chart (Table 2 equivalent)
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(all_results, M, n_instances):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Performance Comparison — M={M}, {n_instances} instances', fontsize=13)

    metrics = [
        ('waoi',     '(a) Weighted AoI (θ₁·WAoI)', 'Scaled WAoI'),
        ('priority', '(b) Total Priority',           'Sum of wi'),
        ('obj',      '(c) Composite Objective',      'θ₁·WAoI − θ₂·Σwi'),
    ]
    for ax, (key, title, ylabel) in zip(axes, metrics):
        vals  = [all_results[m][key] for m in METHODS]
        bars  = ax.bar(range(len(METHODS)), vals,
                       color=[C[m] for m in METHODS],
                       edgecolor='black', linewidth=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            yoff = 0.5 if v >= 0 else -2.5
            ax.text(bar.get_x()+bar.get_width()/2,
                    v + yoff, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    p = f'results(attn)/fig2_comparison_M{M}.png'
    plt.savefig(p, dpi=150); plt.close()
    print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Reward distribution box plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_reward_distribution(params, policy, device, n=200, seed=42):
    rng     = np.random.default_rng(seed)
    rewards = defaultdict(list)
    nodes   = defaultdict(list)

    baseline_fns = {
        'Random':           random_policy,
        'Nearest-Neighbor': nearest_neighbor,
        'Greedy-Priority':  greedy_priority,
        'PDR':              pdr,
    }

    for _ in range(n):
        s = int(rng.integers(0, 10_000_000))
        for name, fn in baseline_fns.items():
            env  = UAVEnv(params, seed=s)
            traj = fn(env)
            rewards[name].append(env.return_reward(traj))
            nodes[name].append(len(traj))
        env2 = UAVEnv(params, seed=s)
        with torch.no_grad():
            traj2, _, _, r2 = batch_rollout(policy, env2, device, greedy=True)
        rewards['Attention (Ours)'].append(r2)
        nodes['Attention (Ours)'].append(len(traj2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Reward & Nodes Distribution — M={params.M}, {n} instances', fontsize=13)

    bp = ax1.boxplot([rewards[m] for m in METHODS],
                     patch_artist=True, notch=False,
                     medianprops=dict(color='black', lw=2))
    for patch, m in zip(bp['boxes'], METHODS):
        patch.set_facecolor(C[m]); patch.set_alpha(0.75)
    ax1.set_xticks(range(1, len(METHODS)+1))
    ax1.set_xticklabels(METHODS, rotation=20, ha='right', fontsize=9)
    ax1.set_ylabel('Episode Reward (= −Objective)', fontsize=11)
    ax1.set_title('(a) Reward Distribution', fontsize=11)
    ax1.axhline(0, color='red', ls='--', lw=0.8, alpha=0.6)
    ax1.grid(axis='y', alpha=0.3)

    bp2 = ax2.boxplot([nodes[m] for m in METHODS],
                      patch_artist=True, notch=False,
                      medianprops=dict(color='black', lw=2))
    for patch, m in zip(bp2['boxes'], METHODS):
        patch.set_facecolor(C[m]); patch.set_alpha(0.75)
    ax2.set_xticks(range(1, len(METHODS)+1))
    ax2.set_xticklabels(METHODS, rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('Nodes Visited per Episode', fontsize=11)
    ax2.set_title('(b) Nodes Visited Distribution', fontsize=11)
    ax2.axhline(params.M, color='gray', ls='--', lw=1, label=f'All nodes={params.M}')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p = f'results(attn)/fig3_reward_nodes_dist_M{params.M}.png'
    plt.savefig(p, dpi=150); plt.close()
    print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Scalability M=20→100
# ─────────────────────────────────────────────────────────────────────────────
def plot_scalability(M_list, n_epochs_per_M, n_eval, quick, device):
    results = defaultdict(lambda: defaultdict(list))

    for M in M_list:
        print(f'\n  M = {M}')
        params = Params(M=M)

        # Baselines
        bres = evaluate_all(params, n=n_eval, seed=42)
        for name, vals in bres.items():
            results[name]['obj'].append(vals['obj'])

        # Prefer best_attn model saved by --train_all, then fall back to
        # legacy attn_M*.pt, and only train from scratch if nothing exists.
        best_path = f'{MODEL_DIR}/best_attn_M{M}.pt'
        scal_path = f'{MODEL_DIR}/attn_M{M}.pt'

        if os.path.exists(best_path):
            print(f'    Attention: loading {best_path}')
            policy = load_policy(best_path, params, device)
            ev = eval_policy(policy, params, n=n_eval, device=device)
        elif os.path.exists(scal_path):
            print(f'    Attention: loading {scal_path}')
            policy = load_policy(scal_path, params, device)
            ev = eval_policy(policy, params, n=n_eval, device=device)
        else:
            print(f'    Attention: no saved model, training now …')
            tr = Trainer(params, device=device,
                         episodes_per_epoch=32 if quick else 64)
            tr.train(n_epochs=n_epochs_per_M,
                     save_path=scal_path,
                     log_every=9999, eval_every=9999, seed=42)
            ev = tr.evaluate(n=n_eval)

        results['Attention (Ours)']['obj'].append(ev['obj'])
        print(f'    Attention: obj={ev["obj"]:.2f}')

    fig, ax = plt.subplots(figsize=(9, 5))
    for m in METHODS:
        ax.plot(M_list, results[m]['obj'],
                marker=MARKER[m], color=C[m],
                label=m, lw=1.8, markersize=7)
    ax.set_xlabel('Number of Nodes M', fontsize=12)
    ax.set_ylabel('Composite Objective (lower = better)', fontsize=12)
    ax.set_title('Scalability: Objective vs Network Size', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = 'results(attn)/fig4_scalability.png'
    plt.savefig(p, dpi=150); plt.close()
    print(f'\n  Saved: {p}')
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Battery sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def plot_battery(policy, device, n_eval=100):
    Emax_list = [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000]
    results   = defaultdict(list)

    for Emax in Emax_list:
        params = Params(M=30, Emax=Emax)
        bres   = evaluate_all(params, n=n_eval, seed=42)
        for name, vals in bres.items():
            results[name].append(vals['obj'])
        ev = eval_policy(policy, params, n=n_eval, device=device)
        results['Attention (Ours)'].append(ev['obj'])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = [E/1000 for E in Emax_list]
    for m in METHODS:
        ax.plot(x, results[m], marker=MARKER[m], color=C[m],
                label=m, lw=1.8, markersize=7)
    ax.set_xlabel('Battery Capacity Emax (kJ)', fontsize=12)
    ax.set_ylabel('Composite Objective', fontsize=12)
    ax.set_title('Battery Sensitivity — M=30', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = 'results(attn)/fig5_battery.png'
    plt.savefig(p, dpi=150); plt.close()
    print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Example trajectory
# ─────────────────────────────────────────────────────────────────────────────
def plot_trajectory(policy, params, device, seed=7):
    env  = UAVEnv(params, seed=seed)
    with torch.no_grad():
        traj, _, _, _ = batch_rollout(policy, env, device, greedy=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(-30, params.area+30); ax.set_ylim(-30, params.area+30)

    for i in range(params.M):
        pos = env.pos[i]; wi = env.wi[i]
        sz  = 40 + (wi/params.wi_hi)*140
        clr = C['Attention (Ours)'] if i in traj else '#cccccc'
        ew  = 1.3 if i in traj else 0.3
        ax.scatter(*pos, s=sz, c=clr, edgecolors='black', lw=ew, zorder=4)
        ax.text(pos[0]+12, pos[1]+12, f'w={wi:.1f}', fontsize=6.5, color='#222')

    px = ([params.home[0]]
          + [env.pos[j][0] for j in traj]
          + [params.home[0]])
    py = ([params.home[1]]
          + [env.pos[j][1] for j in traj]
          + [params.home[1]])
    ax.plot(px, py, '-', color=C['Attention (Ours)'], lw=1.8, alpha=0.75, zorder=3)
    for k in range(len(px)-1):
        ax.annotate('', xy=(px[k+1], py[k+1]), xytext=(px[k], py[k]),
                    arrowprops=dict(arrowstyle='->', color=C['Attention (Ours)'],
                                    lw=1.2, mutation_scale=12))
    for order, j in enumerate(traj):
        ax.text(env.pos[j][0]-18, env.pos[j][1]-18, str(order+1),
                fontsize=8, color='white', fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor=C['Attention (Ours)'],
                          edgecolor='none', pad=0.1))

    ax.scatter(*params.home, s=220, marker='*', c='gold',
               edgecolors='black', lw=1.5, zorder=6)
    obj  = env.objective(traj)
    waoi = params.theta1 * env.waoi(traj)
    ax.set_title(f'Attention Policy Trajectory — M={params.M}, visited={len(traj)}\n'
                 f'WAoI={waoi:.1f}, Obj={obj:.2f}', fontsize=12)
    ax.set_xlabel('x (m)', fontsize=11); ax.set_ylabel('y (m)', fontsize=11)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    p = f'results(attn)/fig6_trajectory_M{params.M}.png'
    plt.savefig(p, dpi=150); plt.close()
    print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# Systematic multi-M training  (--train_all)
# ─────────────────────────────────────────────────────────────────────────────
def train_for_M(M: int, *, n_epochs: int, episodes_per_epoch: int,
                device: str, seed: int = 42) -> float:
    """
    Train a fresh Attention policy for a single M value.
    Saves best model (by greedy eval objective) to
        models(attn)/best_attn_M{M}.pt
    Returns the best objective achieved.
    """
    save_path = os.path.join(MODEL_DIR, f'best_attn_M{M}.pt')
    params    = Params(M=M)   # completely fresh — no shared state

    eval_every = max(10, min(50, n_epochs // 10))
    log_every  = max(5,  min(20, n_epochs // 20))

    print(f'\n{"="*65}')
    print(f'  Training  M={M}  |  {n_epochs} epochs × {episodes_per_epoch} episodes')
    print(f'  Save path : {save_path}')
    print(f'  log_every={log_every}  eval_every={eval_every}')
    print(f'{"="*65}')

    trainer = Trainer(
        params,
        d_model=128, n_heads=8, n_layers=3,
        lr=1e-4, entropy_beta=0.01,
        episodes_per_epoch=episodes_per_epoch,
        device=device,
    )
    trainer.train(
        n_epochs=n_epochs,
        save_path=save_path,
        log_every=log_every,
        eval_every=eval_every,
        seed=seed,
    )

    # Final safety-net eval — catches any missed eval_every window
    print(f'\n  [M={M}] Running final evaluation (200 instances) …')
    final_ev = trainer.evaluate(n=200, seed=9999)
    print(f'  [M={M}] Final  obj={final_ev["obj"]:.4f}  '
          f'nodes={final_ev["nodes"]:.1f}  '
          f'waoi={final_ev["waoi"]:.4f}  '
          f'priority={final_ev["priority"]:.4f}')

    if final_ev['obj'] < trainer.best_obj:
        trainer.best_obj = final_ev['obj']
        trainer.save(save_path)
        print(f'  [M={M}] ✓ New best ({final_ev["obj"]:.4f}) — saved to {save_path}')
    else:
        print(f'  [M={M}] Best obj = {trainer.best_obj:.4f} (already saved)')

    return trainer.best_obj


def run_train_all(quick: bool, device: str):
    """Loop through all M values and rigorously save the best model for each."""
    M_list             = M_LIST_QUICK if quick else M_LIST_FULL
    n_epochs           = 100 if quick else 600
    episodes_per_epoch = 32  if quick else 64

    summary = {}
    for M in M_list:
        summary[M] = train_for_M(
            M,
            n_epochs=n_epochs,
            episodes_per_epoch=episodes_per_epoch,
            device=device,
            seed=42,
        )

    print(f'\n{"="*65}')
    print('  Training complete — best objectives per M:')
    print(f'{"="*65}')
    print(f'  {"M":>5}  {"Best Obj":>12}  {"Saved file"}')
    print(f'  {"─"*5}  {"─"*12}  {"─"*35}')
    for M, obj in summary.items():
        fname = f'best_attn_M{M}.pt'
        print(f'  {M:>5}  {obj:>12.4f}  {os.path.join(MODEL_DIR, fname)}')
    print(f'{"="*65}\n')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',     action='store_true',
                        help='Fast demo with fewer epochs/episodes')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training, load saved model and plot')
    parser.add_argument('--train_all', action='store_true',
                        help='Systematic multi-M training, saves best_attn_M*.pt')
    args = parser.parse_args()

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    M          = 30
    n_epochs   = 100 if args.quick else 600
    n_eval     = 50  if args.quick else 200
    n_ep_epoch = 32  if args.quick else 64
    M_list     = M_LIST_QUICK if args.quick else M_LIST_FULL
    scal_epochs= 50  if args.quick else 300

    params = Params(M=M)

    # Prefer the rigorously-saved best model if it exists, fall back to legacy
    MODEL = (f'{MODEL_DIR}/best_attn_M{M}.pt'
             if os.path.exists(f'{MODEL_DIR}/best_attn_M{M}.pt')
             else f'{MODEL_DIR}/attn_M{M}.pt')

    print('='*65)
    print('  UAV AoI — Attention Policy (REINFORCE)')
    print('='*65)

    # ── Step 0: Systematic multi-M training (--train_all) ─────────────────
    if args.train_all:
        run_train_all(quick=args.quick, device=device)
        # Refresh MODEL path in case best_attn_M30.pt was just created
        if os.path.exists(f'{MODEL_DIR}/best_attn_M{M}.pt'):
            MODEL = f'{MODEL_DIR}/best_attn_M{M}.pt'

    # ── Step 1: Train M=30 (skipped if --eval_only or --train_all) ────────
    if not args.eval_only and not args.train_all:
        tr = Trainer(params, d_model=128, n_heads=8, n_layers=3,
                     lr=1e-4, entropy_beta=0.01,
                     episodes_per_epoch=n_ep_epoch,
                     device=device)
        tr.train(n_epochs=n_epochs, save_path=MODEL,
                 log_every=20, eval_every=50, seed=42)
        history = tr.history
        plot_convergence(history, M)
    else:
        hist_path = f'results(attn)/attention_history_M{M}.npy'
        if os.path.exists(hist_path):
            history = np.load(hist_path, allow_pickle=True).item()
            plot_convergence(history, M)

    # ── Step 2: Load best model ───────────────────────────────────────────
    policy = load_policy(MODEL, params, device)

    # ── Step 3: Evaluate all methods ─────────────────────────────────────
    print(f'\n{"="*65}')
    print(f'  Evaluating all methods (M={M}, {n_eval} instances)')
    print(f'{"="*65}')
    all_res  = evaluate_all(params, n=n_eval, seed=42)
    attn_res = eval_policy(policy, params, n=n_eval, device=device)
    all_res['Attention (Ours)'] = attn_res

    # Print table
    paper_d3qn = {'waoi': 38.1, 'priority': 69.1, 'obj': -30.9, 'nodes': 9.9}
    print(f'\n{"Method":<22} {"WAoI":>8} {"Priority":>10} {"Obj":>8} {"Nodes":>7}')
    print('─'*55)
    for m in METHODS:
        r = all_res[m]
        print(f'{m:<22} {r["waoi"]:>8.1f} {r["priority"]:>10.1f} '
              f'{r["obj"]:>8.1f} {r["nodes"]:>7.1f}')
    print(f'\n{"Paper D3QN (ref)":<22} {paper_d3qn["waoi"]:>8.1f} '
          f'{paper_d3qn["priority"]:>10.1f} {paper_d3qn["obj"]:>8.1f} '
          f'{paper_d3qn["nodes"]:>7.1f}')

    waoi_red = (1 - attn_res['waoi'] / all_res['Random']['waoi']) * 100
    print(f'\nWAoI reduction vs Random: {waoi_red:.1f}%')

    # ── Step 4: Figures ───────────────────────────────────────────────────
    plot_comparison(all_res, M, n_eval)
    plot_reward_distribution(params, policy, device, n=n_eval)
    plot_trajectory(policy, params, device)
    plot_battery(policy, device, n_eval=min(n_eval, 100))

    # ── Step 5: Scalability ───────────────────────────────────────────────
    print(f'\n{"="*65}')
    print(f'  Scalability study')
    plot_scalability(M_list, scal_epochs, min(n_eval, 50), args.quick, device)

    print('\n' + '='*65)
    print('  All results saved to results(attn)/')
    print('='*65)