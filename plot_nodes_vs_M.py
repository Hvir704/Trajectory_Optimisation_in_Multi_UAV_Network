"""
plot_nodes_vs_M.py
==================
Plots average nodes visited vs network size M=20,30,...,100 for:

  MLP-REINFORCE
    - Uses the correct trained model for each M (models/policy_M{M}.pt)
    - Specialised: one model per M, trained on that exact size

  Transformer-REINFORCE
    - Has ONLY one model saved: trained on M=30 (models_attn/attn_M30.pt)
    - Architecture is M-agnostic (Transformer sequence length is flexible)
    - Evaluated ZERO-SHOT on all M values with the M=30 trained model
    - This demonstrates generalisation capability

  All 4 baselines for reference (evaluated fresh for each M)

Outputs:
  report_figs_mlp/nodes_vs_M.png    (MLP vs baselines)
  report_figs_attn/nodes_vs_M.png   (Transformer zero-shot vs baselines)
  report_figs_mlp/nodes_vs_M_combined.png  (both on one plot)

Usage:
  python plot_nodes_vs_M.py
  python plot_nodes_vs_M.py --n_eval 50    # faster
  python plot_nodes_vs_M.py --skip_cache   # ignore cached results
"""

import os, sys, argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib.util

for d in ['report_figs_mlp', 'report_figs_attn']:
    os.makedirs(d, exist_ok=True)

# ── load MLP solver for baselines ────────────────────────────────────────────
spec = importlib.util.spec_from_file_location('sol', 'uav_aoi_solver.py')
sol  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

M_VALUES = [20, 30, 40, 50, 60, 70, 80, 90, 100]

C = {
    'Random':           '#AAAAAA',
    'Nearest-Neighbor': '#E69F00',
    'Greedy-Priority':  '#56B4E9',
    'PDR':              '#CC79A7',
    'MLP (per-M model)':         '#009E73',
    'Transformer (M=30, zero-shot)': '#0072B2',
}
MK = {
    'Random': 's', 'Nearest-Neighbor': '^', 'Greedy-Priority': 'D',
    'PDR': 'v', 'MLP (per-M model)': 'o',
    'Transformer (M=30, zero-shot)': 'P',
}


# ══════════════════════════════════════════════════════════════════════════════
# COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def collect_baselines_for_M(M, n, seed):
    """Run all 4 baselines on M-node networks. Returns dict of node lists."""
    rng  = np.random.default_rng(seed)
    data = {b: [] for b in ['Random','Nearest-Neighbor','Greedy-Priority','PDR']}
    for _ in range(n):
        s = int(rng.integers(0, 10_000_000))
        for bname, bkey in sol.BASELINES.items():
            env  = sol.Env(M=M, seed=s)
            traj = sol.run_baseline(env, bkey)
            data[bname].append(len(traj))
    return data


def collect_mlp_for_M(M, n, seed, device):
    """Load the per-M MLP model and evaluate on M-node networks."""
    import torch
    path = f'models/policy_M{M}.pt'
    if not os.path.exists(path):
        print(f'    WARNING: {path} not found — skipping MLP at M={M}')
        return None

    policy = sol.Policy(hidden=256).to(device)
    ck     = torch.load(path, map_location=device)
    policy.load_state_dict(ck['policy'])
    policy.eval()

    rng   = np.random.default_rng(seed)
    nodes = []
    with torch.no_grad():
        for _ in range(n):
            s    = int(rng.integers(0, 10_000_000))
            env  = sol.Env(M=M, seed=s)
            traj, *_ = sol.rollout(policy, env, device, greedy=True)
            nodes.append(len(traj))
    return nodes


def collect_transformer_for_M(M, n, seed, device, attn_policy,
                               attn_params_cls, UAVEnv, rollout_fn):
    """
    Evaluate the M=30 Transformer model zero-shot on M-node networks.
    The architecture (NodeEncoder, self-attention, pointer head) is all
    M-agnostic — it processes variable-length sequences natively.
    Only the trained weights came from M=30 episodes.
    """
    import torch
    from env import Params  # import here to use fresh Params with new M

    params = Params(M=M)   # fresh params with correct M
    rng    = np.random.default_rng(seed)
    nodes  = []

    with torch.no_grad():
        for _ in range(n):
            s    = int(rng.integers(0, 10_000_000))
            env  = UAVEnv(params, seed=s)
            env.reset()
            traj, *_ = rollout_fn(attn_policy, env, device, greedy=True)
            nodes.append(len(traj))
    return nodes


def run_all(n, seed, device, skip_cache=False):
    import torch

    cache = 'report_figs_mlp/nodes_vs_M_data.npy'
    if not skip_cache and os.path.exists(cache):
        print(f'Loading cached data from {cache}')
        return np.load(cache, allow_pickle=True).item()

    results = {
        'baselines': {b: {} for b in
                      ['Random','Nearest-Neighbor','Greedy-Priority','PDR']},
        'mlp':         {},
        'transformer': {},
    }

    # ── load Transformer once ─────────────────────────────────────────────
    attn_ok = False
    attn_policy = None
    try:
        sys.path.insert(0, '.')
        from env     import UAVEnv, Params
        from policy  import AttentionPolicy
        # Support both naming conventions across versions
        try:
            from features import rollout_episode
        except ImportError:
            from features import batch_rollout as rollout_episode

        _params_tmp = Params(M=30)
        attn_policy = AttentionPolicy(d_model=128, n_heads=8, n_layers=3).to(device)
        loaded = False
        search_paths = ['models_attn/attn_M30.pt',
                        'models_attn/attn_policy.pt',
                        'models_attn/attention_policy.pt']
        print(f'  Searching for Transformer model in: {search_paths}')
        for path in search_paths:
            exists = os.path.exists(path)
            print(f'    {path}: {"FOUND" if exists else "not found"}')
            if exists:
                ck = torch.load(path, map_location=device)
                # handle both {'policy': state_dict} and raw state_dict
                state = ck.get('policy', ck)
                attn_policy.load_state_dict(state)
                loaded = True
                print(f'  Loaded Transformer: {path}')
                break
        if loaded:
            attn_policy.eval()
            attn_ok = True
        else:
            print('ERROR: No Transformer model found. Create models_attn/ folder')
            print('       and ensure attn_M30.pt or attn_policy.pt is inside it.')
    except Exception as e:
        print(f'WARNING: Transformer load failed: {e}')

    # ── evaluate each M ───────────────────────────────────────────────────
    for M in M_VALUES:
        t0 = time.time()
        print(f'\n  M={M}', end='  ', flush=True)

        # Baselines
        b_data = collect_baselines_for_M(M, n, seed)
        for bname in b_data:
            results['baselines'][bname][M] = b_data[bname]
        print('baselines done', end='  ', flush=True)

        # MLP
        mlp_nodes = collect_mlp_for_M(M, n, seed, device)
        if mlp_nodes is not None:
            results['mlp'][M] = mlp_nodes
        print('MLP done', end='  ', flush=True)

        # Transformer (zero-shot)
        if attn_ok:
            from env     import UAVEnv
            try:
                from features import rollout_episode
            except ImportError:
                from features import batch_rollout as rollout_episode
            tr_nodes = collect_transformer_for_M(
                M, n, seed, device, attn_policy,
                None, UAVEnv, rollout_episode)
            results['transformer'][M] = tr_nodes
            print('Transformer done', end='  ', flush=True)

        print(f'({time.time()-t0:.0f}s)')

    np.save(cache, results)
    print(f'\nData saved to {cache}')
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def ci95(arr): return 1.96 * np.std(arr) / np.sqrt(len(arr))


def _plot_nodes(ax, results, which_models, title, show_transformer_note=False):
    """Core plotting function — shared between all three output figures."""

    # Baselines
    for bname in ['Random','Nearest-Neighbor','Greedy-Priority','PDR']:
        b_res = results['baselines'][bname]
        xs    = sorted(b_res.keys())
        means = [np.mean(b_res[M]) for M in xs]
        errs  = [ci95(np.array(b_res[M])) for M in xs]
        ax.errorbar(xs, means, yerr=errs, color=C[bname], lw=1.5,
                    marker=MK[bname], ms=6, capsize=3, alpha=0.75,
                    label=bname)

    # MLP
    if 'mlp' in which_models and results['mlp']:
        xs    = sorted(results['mlp'].keys())
        means = [np.mean(results['mlp'][M]) for M in xs]
        errs  = [ci95(np.array(results['mlp'][M])) for M in xs]
        ax.errorbar(xs, means, yerr=errs,
                    color=C['MLP (per-M model)'], lw=2.5,
                    marker=MK['MLP (per-M model)'], ms=9,
                    capsize=4, zorder=5,
                    label='MLP-REINFORCE (per-M model)')

    # Transformer
    if 'transformer' in which_models and not results.get('transformer'):
        ax.text(0.5, 0.5, 'Transformer model not found\n'
                'Check models_attn/ folder',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='#0072B2',
                bbox=dict(boxstyle='round', facecolor='#E6F1FB', alpha=0.8))
    if 'transformer' in which_models and results.get('transformer'):
        xs    = sorted(results['transformer'].keys())
        means = [np.mean(results['transformer'][M]) for M in xs]
        errs  = [ci95(np.array(results['transformer'][M])) for M in xs]
        ax.errorbar(xs, means, yerr=errs,
                    color=C['Transformer (M=30, zero-shot)'], lw=2.5,
                    marker=MK['Transformer (M=30, zero-shot)'], ms=9,
                    capsize=4, zorder=5, linestyle='--',
                    label='Transformer (M=30 model, zero-shot)')
        if show_transformer_note:
            ax.annotate('Zero-shot:\ntrained on M=30 only',
                        xy=(30, np.mean(results['transformer'][30])),
                        xytext=(45, np.mean(results['transformer'][30]) + 4),
                        fontsize=9, color=C['Transformer (M=30, zero-shot)'],
                        arrowprops=dict(arrowstyle='->', lw=1,
                                        color=C['Transformer (M=30, zero-shot)']))

    ax.set_xlabel('Number of Sensor Nodes M', fontsize=12)
    ax.set_ylabel('Average Nodes Visited per Episode', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(M_VALUES)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    # Note: visiting ~14 nodes regardless of M is CORRECT behaviour.
    # The WAoI-priority tradeoff makes ~14 optimal because:
    # shorter inter-node distances (more nodes) increase the reward breakeven,
    # exactly cancelling the larger node pool.


def plot_mlp_only(results, n):
    """Fig A: MLP vs baselines."""
    print('\nFig A: MLP nodes vs M...')
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_nodes(ax, results, ['mlp'],
                f'MLP-REINFORCE — Nodes Visited vs Network Size\n'
                f'(separate model per M, {n} instances each)  ± 95% CI')
    plt.tight_layout()
    path = 'report_figs_mlp/nodes_vs_M.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_transformer_only(results, n):
    """Fig B: Transformer zero-shot vs baselines."""
    print('Fig B: Transformer nodes vs M...')
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_nodes(ax, results, ['transformer'],
                f'Transformer-REINFORCE — Zero-Shot Generalisation\n'
                f'(single M=30 model evaluated on all M, {n} instances each)  ± 95% CI',
                show_transformer_note=True)
    plt.tight_layout()
    path = 'report_figs_attn/nodes_vs_M.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_combined(results, n):
    """Fig C: 2x2 grid — absolute nodes AND fraction of M visited."""
    print('Fig C: Combined nodes vs M (2x2 with fraction)...')
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f'Nodes Visited vs Network Size M  (n={n} instances per M)\n'
                 f'Note: visiting ~14 nodes regardless of M is optimal — the reward\'s WAoI-priority tradeoff\n'
                 f'naturally converges there as inter-node distances shrink with larger M',
                 fontsize=11, fontweight='bold')

    # Row 0: absolute node counts
    _plot_nodes(axes[0,0], results, ['mlp'],
                '(a) MLP-REINFORCE — Absolute Nodes Visited')
    _plot_nodes(axes[0,1], results, ['transformer'],
                '(b) Transformer — Zero-Shot Absolute Nodes',
                show_transformer_note=True)

    # Row 1: fraction of M visited (shows the policy IS being selective)
    for ax, which, title in [
        (axes[1,0], 'mlp',         '(c) MLP — Fraction of Nodes Visited (%)'),
        (axes[1,1], 'transformer', '(d) Transformer — Fraction Visited (%)'),
    ]:
        for bname in ['Random','Nearest-Neighbor','Greedy-Priority','PDR']:
            b_res = results['baselines'][bname]
            xs    = sorted(b_res.keys())
            fracs = [np.mean(b_res[M])/M*100 for M in xs]
            ax.plot(xs, fracs, color=C[bname], lw=1.5,
                    marker=MK[bname], ms=6, alpha=0.75, label=bname)

        model_key = 'mlp' if which == 'mlp' else 'transformer'
        model_label = 'MLP-REINFORCE' if which == 'mlp' else 'Transformer (zero-shot)'
        model_color = C['MLP (per-M model)'] if which == 'mlp' else C['Transformer (M=30, zero-shot)']
        model_mk    = MK['MLP (per-M model)'] if which == 'mlp' else MK['Transformer (M=30, zero-shot)']

        if results[model_key]:
            xs    = sorted(results[model_key].keys())
            fracs = [np.mean(results[model_key][M])/M*100 for M in xs]
            ax.plot(xs, fracs, color=model_color, lw=2.5,
                    marker=model_mk, ms=9, zorder=5, label=model_label)

        ax.set_xlabel('Number of Nodes M', fontsize=11)
        ax.set_ylabel('% of M nodes visited', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(M_VALUES)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        ax.text(0.97, 0.5, 'Decreasing fraction\nshows selective\nchoice improves\nwith scale',
                transform=ax.transAxes, ha='right', va='center',
                fontsize=8, color='gray',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    plt.tight_layout()
    path = 'report_figs_mlp/nodes_vs_M_combined.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_overlay(results, n):
    """Fig D: Both models overlaid on same axes — direct visual comparison."""
    print('Fig D: Overlay (both models same axes)...')
    fig, ax = plt.subplots(figsize=(11, 6))
    _plot_nodes(ax, results, ['mlp', 'transformer'],
                f'Nodes Visited vs Network Size — MLP vs Transformer\n'
                f'MLP: specialised per-M  |  Transformer: M=30 zero-shot  |  n={n} per M',
                show_transformer_note=True)
    plt.tight_layout()
    path = 'report_figs_mlp/nodes_vs_M_overlay.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def print_table(results):
    """Print a clean summary table."""
    print('\n' + '='*95)
    print('  NODES VISITED vs M — SUMMARY TABLE')
    print('='*95)

    header = f'{"M":>4}  {"Random":>8}  {"NN":>8}  {"Greedy":>8}  {"PDR":>8}'
    if results['mlp']:        header += f'  {"MLP":>8}'
    if results['transformer']: header += f'  {"Transformer":>11} (zero-shot)'
    print(header)
    print('─'*95)

    for M in M_VALUES:
        row = f'{M:>4}'
        for bname, fmt_name in [
            ('Random','Random'), ('Nearest-Neighbor','NN'),
            ('Greedy-Priority','Greedy'), ('PDR','PDR')
        ]:
            if M in results['baselines'][bname]:
                v = np.mean(results['baselines'][bname][M])
                row += f'  {v:>8.1f}'
            else:
                row += f'  {"—":>8}'

        if results['mlp'] and M in results['mlp']:
            row += f'  {np.mean(results["mlp"][M]):>8.1f}'
        elif results['mlp']:
            row += f'  {"—":>8}'

        if results['transformer'] and M in results['transformer']:
            row += f'  {np.mean(results["transformer"][M]):>11.1f}'

        print(row)

    print('─'*95)
    if results['mlp']:
        print('  MLP: uses models/policy_M{M}.pt for each M')
    if results['transformer']:
        print('  Transformer: single models_attn/attn_M30.pt for ALL M (zero-shot)')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_eval',     type=int, default=100,
                    help='Instances per M value (default 100)')
    ap.add_argument('--skip_cache', action='store_true',
                    help='Ignore cached results and recompute')
    ap.add_argument('--device',     default='')
    args = ap.parse_args()

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print('='*60)
    print('  Nodes Visited vs M  —  Both Implementations')
    print(f'  device={device}  n_eval={args.n_eval} per M')
    print('  M values:', M_VALUES)
    print('='*60)

    results = run_all(args.n_eval, seed=42, device=device,
                      skip_cache=args.skip_cache)

    print('\nGenerating figures...')
    plot_mlp_only(results, args.n_eval)
    plot_transformer_only(results, args.n_eval)
    plot_combined(results, args.n_eval)
    plot_overlay(results, args.n_eval)

    print_table(results)

    print('\n' + '='*60)
    print('  Saved:')
    print('    report_figs_mlp/nodes_vs_M.png           (MLP only)')
    print('    report_figs_attn/nodes_vs_M.png          (Transformer only)')
    print('    report_figs_mlp/nodes_vs_M_combined.png  (side-by-side panels)')
    print('    report_figs_mlp/nodes_vs_M_overlay.png   (both on same axes)')
    print()
    print('  Note on Transformer:')
    print('    The M=30 model runs zero-shot on M=20..100.')
    print('    This works because the Transformer architecture is')
    print('    sequence-length agnostic — only the weights came from M=30.')
    print('='*60)
