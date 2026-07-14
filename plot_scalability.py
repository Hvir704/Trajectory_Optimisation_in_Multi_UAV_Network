"""
plot_scalability.py  —  Scalability plots for Priority-Aware UAV AoI (v2 MLP)
==============================================================================
Produces two publication-quality figures:

  Figure 1 — Composite Objective vs M
    All four baselines + MLP-greedy + MLP+post-process (beam+2opt+insert)
    with 95% confidence intervals.

  Figure 2 — Average Nodes Visited vs M
    Same six methods. Shows how many nodes each method visits on average.

Usage:
    python plot_scalability.py                  # default: 200 instances/M
    python plot_scalability.py --instances 50   # faster
    python plot_scalability.py --no-pp          # skip post-processing (faster)
    python plot_scalability.py --M 20 30 50 70 100

Requires:
    uav_aoi_solver.py (v2, 16-dim) and models_mlp/policy_M{M}.pt
    in the same directory.
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── import solver ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uav_aoi_solver import (
    Env, Policy, P,
    rollout, rollout_beam, post_process,
    run_baseline, BASELINES,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_M_LIST   = [20, 30, 40, 50, 60, 70, 80, 90, 100]
DEFAULT_N        = 200        # instances per M
BEAM_WIDTH       = 5
SEED             = 9999       # eval seed (different from training seed 42)
MODEL_DIR        = 'models_mlp'
OUT_DIR          = 'results'
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUT_DIR, exist_ok=True)

# ── colour / style palette (colour-blind safe) ────────────────────────────────
STYLE = {
    'Random':            dict(color='#888888', marker='s', ls='--',  lw=1.4, ms=6),
    'Nearest-Neighbor':  dict(color='#E69F00', marker='^', ls='--',  lw=1.4, ms=6),
    'Greedy-Priority':   dict(color='#56B4E9', marker='D', ls='--',  lw=1.4, ms=6),
    'PDR':               dict(color='#CC79A7', marker='v', ls='--',  lw=1.4, ms=6),
    'MLP greedy':        dict(color='#009E73', marker='o', ls='-',   lw=2.2, ms=7),
    'MLP + post-process':dict(color='#005C44', marker='o', ls='-',   lw=2.6, ms=8),
}
BASELINE_KEYS = {
    'Random':           'random',
    'Nearest-Neighbor': 'nearest_neighbor',
    'Greedy-Priority':  'greedy_priority',
    'PDR':              'pdr',
}
# Order for legend
LEGEND_ORDER = [
    'MLP + post-process', 'MLP greedy',
    'Greedy-Priority', 'PDR', 'Nearest-Neighbor', 'Random',
]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_policy(M: int):
    path = os.path.join(MODEL_DIR, f'policy_M{M}.pt')
    if not os.path.exists(path):
        print(f'  [skip] No model found at {path}')
        return None
    ckpt   = torch.load(path, map_location=DEVICE, weights_only=False)
    policy = Policy(hidden=256, input_dim=16).to(DEVICE)
    policy.load_state_dict(ckpt['policy'])
    policy.eval()
    return policy


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_all(M_list, n_instances, use_pp=True):
    """
    Returns a nested dict:
        results[method_name][M] = {'obj_mean', 'obj_ci', 'nodes_mean', 'nodes_ci'}
    where ci = half-width of 95% confidence interval.
    """
    rng = np.random.default_rng(SEED)

    # Initialise storage
    raw = {name: {M: {'obj': [], 'nodes': []} for M in M_list}
           for name in list(BASELINE_KEYS.keys()) + ['MLP greedy', 'MLP + post-process']}

    for M in M_list:
        print(f'\n  M = {M}', end='', flush=True)

        policy = load_policy(M)

        for inst in range(n_instances):
            if inst % 50 == 0:
                print('.', end='', flush=True)

            s   = int(rng.integers(0, 10_000_000))
            env = Env(M=M, seed=s)

            # ── baselines ────────────────────────────────────────────────
            for name, key in BASELINE_KEYS.items():
                traj = run_baseline(env, key)
                raw[name][M]['obj'].append(env.objective(traj))
                raw[name][M]['nodes'].append(len(traj))

            # ── MLP ──────────────────────────────────────────────────────
            if policy is not None:
                with torch.no_grad():
                    # greedy rollout
                    traj_g, *_ = rollout(policy, env, DEVICE, greedy=True)
                    raw['MLP greedy'][M]['obj'].append(env.objective(traj_g))
                    raw['MLP greedy'][M]['nodes'].append(len(traj_g))

                    # post-processed
                    if use_pp:
                        traj_b  = rollout_beam(policy, env, DEVICE, beam_width=BEAM_WIDTH)
                        traj_pp = post_process(env, traj_b)
                    else:
                        traj_pp = traj_g   # same as greedy if pp disabled

                    raw['MLP + post-process'][M]['obj'].append(env.objective(traj_pp))
                    raw['MLP + post-process'][M]['nodes'].append(len(traj_pp))
            else:
                raw['MLP greedy'][M]['obj'].append(np.nan)
                raw['MLP greedy'][M]['nodes'].append(np.nan)
                raw['MLP + post-process'][M]['obj'].append(np.nan)
                raw['MLP + post-process'][M]['nodes'].append(np.nan)

    print()

    # ── aggregate: mean + 95% CI ─────────────────────────────────────────────
    def ci95(arr):
        a = np.array(arr)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return np.nan, np.nan
        return float(np.mean(a)), 1.96 * float(np.std(a)) / np.sqrt(len(a))

    results = {}
    for name in raw:
        results[name] = {}
        for M in M_list:
            obj_mean,   obj_ci   = ci95(raw[name][M]['obj'])
            nodes_mean, nodes_ci = ci95(raw[name][M]['nodes'])
            results[name][M] = {
                'obj_mean':   obj_mean,   'obj_ci':   obj_ci,
                'nodes_mean': nodes_mean, 'nodes_ci': nodes_ci,
            }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Composite Objective vs M
# ══════════════════════════════════════════════════════════════════════════════
def plot_objective(results, M_list, n_instances, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name in LEGEND_ORDER:
        if name not in results:
            continue
        st = STYLE[name]
        Ms    = [M for M in M_list if not np.isnan(results[name][M]['obj_mean'])]
        means = [results[name][M]['obj_mean'] for M in Ms]
        cis   = [results[name][M]['obj_ci']   for M in Ms]

        ax.plot(Ms, means,
                color=st['color'], marker=st['marker'],
                ls=st['ls'], lw=st['lw'], markersize=st['ms'],
                label=name, zorder=4 if 'MLP' in name else 3)

        ax.fill_between(Ms,
                        [m - c for m, c in zip(means, cis)],
                        [m + c for m, c in zip(means, cis)],
                        color=st['color'], alpha=0.12, zorder=2)

    ax.axhline(0, color='black', lw=0.8, ls=':', zorder=1)

    ax.set_xlabel('Number of Sensor Nodes M', fontsize=12)
    ax.set_ylabel('Composite Objective  (lower = better)', fontsize=12)
    ax.set_title(
        'Scalability: Composite Objective vs Network Size\n'
        f'({n_instances} instances per M, 95% CI shaded)',
        fontsize=12
    )
    ax.set_xticks(M_list)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(alpha=0.3)
    ax.grid(which='minor', alpha=0.12)

    # Legend: two columns, MLP entries at top
    handles, labels = ax.get_legend_handles_labels()
    order = {n: i for i, n in enumerate(LEGEND_ORDER)}
    paired = sorted(zip(labels, handles), key=lambda x: order.get(x[0], 99))
    labels_s, handles_s = zip(*paired)
    ax.legend(handles_s, labels_s, fontsize=9, ncol=2,
              loc='lower left', framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(out_dir, 'fig_objective_vs_M.png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {out}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Average Nodes Visited vs M
# ══════════════════════════════════════════════════════════════════════════════
def plot_nodes(results, M_list, n_instances, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Reference line: M itself (all nodes visited)
    ax.plot(M_list, M_list, color='#AAAAAA', lw=1.0, ls=':',
            label='All nodes (upper bound)', zorder=1)

    for name in LEGEND_ORDER:
        if name not in results:
            continue
        st = STYLE[name]
        Ms    = [M for M in M_list if not np.isnan(results[name][M]['nodes_mean'])]
        means = [results[name][M]['nodes_mean'] for M in Ms]
        cis   = [results[name][M]['nodes_ci']   for M in Ms]

        ax.plot(Ms, means,
                color=st['color'], marker=st['marker'],
                ls=st['ls'], lw=st['lw'], markersize=st['ms'],
                label=name, zorder=4 if 'MLP' in name else 3)

        ax.fill_between(Ms,
                        [m - c for m, c in zip(means, cis)],
                        [m + c for m, c in zip(means, cis)],
                        color=st['color'], alpha=0.12, zorder=2)

    ax.set_xlabel('Number of Sensor Nodes M', fontsize=12)
    ax.set_ylabel('Average Nodes Visited per Episode', fontsize=12)
    ax.set_title(
        'Scalability: Average Nodes Visited vs Network Size\n'
        f'({n_instances} instances per M, 95% CI shaded)',
        fontsize=12
    )
    ax.set_xticks(M_list)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(alpha=0.3)
    ax.grid(which='minor', alpha=0.12)

    handles, labels = ax.get_legend_handles_labels()
    # Put "All nodes" at the end
    order = {n: i for i, n in enumerate(LEGEND_ORDER)}
    order['All nodes (upper bound)'] = 99
    paired = sorted(zip(labels, handles), key=lambda x: order.get(x[0], 99))
    labels_s, handles_s = zip(*paired)
    ax.legend(handles_s, labels_s, fontsize=9, ncol=2,
              loc='upper left', framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(out_dir, 'fig_nodes_visited_vs_M.png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {out}')


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED FIGURE — both panels side by side (for paper)
# ══════════════════════════════════════════════════════════════════════════════
def plot_combined(results, M_list, n_instances, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(
        f'Architecture Comparison: MLP v2 vs Baselines  '
        f'({n_instances} instances per M, 95% CI shaded)',
        fontsize=13
    )

    # ── left: objective ───────────────────────────────────────────────────────
    ax = axes[0]
    for name in LEGEND_ORDER:
        if name not in results:
            continue
        st = STYLE[name]
        Ms    = [M for M in M_list if not np.isnan(results[name][M]['obj_mean'])]
        means = [results[name][M]['obj_mean'] for M in Ms]
        cis   = [results[name][M]['obj_ci']   for M in Ms]
        ax.plot(Ms, means, color=st['color'], marker=st['marker'],
                ls=st['ls'], lw=st['lw'], markersize=st['ms'], label=name,
                zorder=4 if 'MLP' in name else 3)
        ax.fill_between(Ms,
                        [m - c for m, c in zip(means, cis)],
                        [m + c for m, c in zip(means, cis)],
                        color=st['color'], alpha=0.12)
    ax.axhline(0, color='black', lw=0.8, ls=':', zorder=1)
    ax.set_xlabel('Number of Sensor Nodes M', fontsize=11)
    ax.set_ylabel('Composite Objective  (lower = better)', fontsize=11)
    ax.set_title('(a) Composite Objective vs M', fontsize=11)
    ax.set_xticks(M_list)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    order = {n: i for i, n in enumerate(LEGEND_ORDER)}
    paired = sorted(zip(labels, handles), key=lambda x: order.get(x[0], 99))
    labels_s, handles_s = zip(*paired)
    ax.legend(handles_s, labels_s, fontsize=8.5, ncol=2,
              loc='lower left', framealpha=0.9)

    # ── right: nodes visited ──────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(M_list, M_list, color='#AAAAAA', lw=1.0, ls=':',
            label='All nodes (upper bound)', zorder=1)
    for name in LEGEND_ORDER:
        if name not in results:
            continue
        st = STYLE[name]
        Ms    = [M for M in M_list if not np.isnan(results[name][M]['nodes_mean'])]
        means = [results[name][M]['nodes_mean'] for M in Ms]
        cis   = [results[name][M]['nodes_ci']   for M in Ms]
        ax.plot(Ms, means, color=st['color'], marker=st['marker'],
                ls=st['ls'], lw=st['lw'], markersize=st['ms'], label=name,
                zorder=4 if 'MLP' in name else 3)
        ax.fill_between(Ms,
                        [m - c for m, c in zip(means, cis)],
                        [m + c for m, c in zip(means, cis)],
                        color=st['color'], alpha=0.12)
    ax.set_xlabel('Number of Sensor Nodes M', fontsize=11)
    ax.set_ylabel('Average Nodes Visited per Episode', fontsize=11)
    ax.set_title('(b) Average Nodes Visited vs M', fontsize=11)
    ax.set_xticks(M_list)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    order['All nodes (upper bound)'] = 99
    paired = sorted(zip(labels, handles), key=lambda x: order.get(x[0], 99))
    labels_s, handles_s = zip(*paired)
    ax.legend(handles_s, labels_s, fontsize=8.5, ncol=2,
              loc='upper left', framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(out_dir, 'fig_scalability_combined.png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {out}')


# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def print_table(results, M_list):
    col_w = 14
    methods = LEGEND_ORDER + [n for n in BASELINE_KEYS if n not in LEGEND_ORDER]

    print('\n' + '=' * 110)
    print('  Scalability Summary Table')
    print('=' * 110)

    # Header — objective
    print('\n  COMPOSITE OBJECTIVE  (lower = better, ± 95% CI)')
    hdr = f'  {"M":>5}' + ''.join(f'{n:>{col_w}}' for n in LEGEND_ORDER if n in results)
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for M in M_list:
        row = f'  {M:>5}'
        for name in LEGEND_ORDER:
            if name not in results:
                continue
            d = results[name][M]
            if np.isnan(d['obj_mean']):
                row += f'{"—":>{col_w}}'
            else:
                row += f'{d["obj_mean"]:>+{col_w-5}.2f} ±{d["obj_ci"]:.2f}'
        print(row)

    # Header — nodes
    print(f'\n  AVERAGE NODES VISITED')
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for M in M_list:
        row = f'  {M:>5}'
        for name in LEGEND_ORDER:
            if name not in results:
                continue
            d = results[name][M]
            if np.isnan(d['nodes_mean']):
                row += f'{"—":>{col_w}}'
            else:
                row += f'{d["nodes_mean"]:>{col_w-5}.1f} ±{d["nodes_ci"]:.1f}'
        print(row)

    print('=' * 110)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Scalability plots — UAV AoI MLP v2')
    ap.add_argument('--instances', type=int, default=DEFAULT_N,
                    help=f'Instances per M (default {DEFAULT_N})')
    ap.add_argument('--M', type=int, nargs='+', default=DEFAULT_M_LIST,
                    help='M values to evaluate')
    ap.add_argument('--no-pp', action='store_true',
                    help='Skip beam+2opt+insert post-processing (much faster)')
    args = ap.parse_args()

    M_list = sorted(args.M)
    use_pp = not args.no_pp

    print('=' * 65)
    print('  Scalability Plot Generator — MLP v2 (16-dim)')
    print(f'  device    = {DEVICE}')
    print(f'  M values  = {M_list}')
    print(f'  instances = {args.instances} per M')
    print(f'  post-proc = {use_pp}  (beam w={BEAM_WIDTH} + 2-opt + node-insert)')
    print(f'  model dir = {MODEL_DIR}/')
    print('=' * 65)

    results = evaluate_all(M_list, args.instances, use_pp=use_pp)

    print_table(results, M_list)

    print('\n  Generating figures...')
    plot_objective(results, M_list, args.instances, OUT_DIR)
    plot_nodes(results, M_list, args.instances, OUT_DIR)
    plot_combined(results, M_list, args.instances, OUT_DIR)

    print('\n  Done. Three figures written to results_mlp_16/:')
    print('    fig_objective_vs_M.png')
    print('    fig_nodes_visited_vs_M.png')
    print('    fig_scalability_combined.png')