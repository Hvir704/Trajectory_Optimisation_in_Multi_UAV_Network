"""
generate_all_tiny_plots.py
=====================================================================
Systematic benchmarking script. Evaluates Attention (Tiny Transformer)
vs. MLP models vs. Deterministic Baselines across all network sizes M.
Generates publication-quality performance curves for the report.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure current directory is in search path
sys.path.insert(0, '.')

# --- Attention & Baseline Imports ---
from env import Params, UAVEnv
from policy import AttentionPolicy
from features import batch_rollout as attn_rollout
from baselines import evaluate_all, BASELINES

# --- MLP Imports (from uav_aoi_solver.py) ---
from uav_aoi_solver import Policy as MLPPolicy
from uav_aoi_solver import Env as MLPEnv
from uav_aoi_solver import rollout as mlp_rollout
from uav_aoi_solver import P as MLP_P

# Configuration
M_VALUES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
N_EVAL_EPISODES = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Consistent plotting style configuration
COLORS = {
    'Random':           '#888888',
    'Nearest-Neighbor': '#E69F00',
    'Greedy-Priority':  '#56B4E9',
    'PDR':              '#CC79A7',
    'MLP (Ours)':       '#D55E00',
    'Attention (Ours)': '#009E73',
}

MARKERS = {
    'Random':           's',
    'Nearest-Neighbor': '^',
    'Greedy-Priority':  'D',
    'PDR':              'v',
    'MLP (Ours)':       'x',
    'Attention (Ours)': 'o',
}

METHODS = ['Random', 'Nearest-Neighbor', 'Greedy-Priority', 'PDR', 'MLP (Ours)', 'Attention (Ours)']

def main():
    os.makedirs('results_attn', exist_ok=True)
    
    # Initialize performance tracking metric buffers
    metrics = ['obj', 'waoi', 'priority', 'nodes']
    data_tracker = {m: {met: [] for met in metrics} for m in METHODS}

    print("="*75)
    print(f"Starting Multi-M Evaluation Pipeline | Instances per M: {N_EVAL_EPISODES} | Device: {DEVICE}")
    print("="*75)

    for M in M_VALUES:
        print(f"\nProcessing Network Size M = {M}...")
        base_params = Params(M=M)
        
        # 1. Evaluate Deterministic Rule-Based Baselines
        base_results = evaluate_all(base_params, n=N_EVAL_EPISODES, seed=42)
        for b_name in BASELINES.keys():
            for met in metrics:
                data_tracker[b_name][met].append(np.mean(base_results[b_name][met]))
        print(f"  -> Baselines Evaluated Successfully.")

        # 2. Evaluate MLP Policy Model
        mlp_policy = MLPPolicy(hidden=256).to(DEVICE)
        mlp_path = f'models_mlp/policy_M{M}.pt'
        
        if os.path.exists(mlp_path):
            checkpoint = torch.load(mlp_path, map_location=DEVICE)
            mlp_policy.load_state_dict(checkpoint['policy'])
            mlp_policy.eval()
            
            m_objs, m_nodes, m_waois, m_prios = [], [], [], []
            rng = np.random.default_rng(42)
            
            with torch.no_grad():
                for _ in range(N_EVAL_EPISODES):
                    s = int(rng.integers(0, 10_000_000))
                    env_mlp = MLPEnv(M=M, seed=s)
                    
                    # mlp_rollout returns 5 values: traj, log_ps, values, entropies, R
                    traj, log_ps, vals, ent, R = mlp_rollout(mlp_policy, env_mlp, DEVICE, greedy=True)
                    
                    m_objs.append(env_mlp.objective(traj))
                    m_nodes.append(len(traj))
                    m_waois.append(MLP_P.theta1 * env_mlp.waoi(traj))
                    m_prios.append(sum(env_mlp.wi[j] for j in traj))
                    
            data_tracker['MLP (Ours)']['obj'].append(np.mean(m_objs))
            data_tracker['MLP (Ours)']['nodes'].append(np.mean(m_nodes))
            data_tracker['MLP (Ours)']['waoi'].append(np.mean(m_waois))
            data_tracker['MLP (Ours)']['priority'].append(np.mean(m_prios))
            print(f"  -> MLP Policy Evaluated. Avg Obj: {np.mean(m_objs):.2f}")
        else:
            print(f"  [!] Warning: Checkpoint missing at {mlp_path}. Filling with NaN.")
            for met in metrics: data_tracker['MLP (Ours)'][met].append(np.nan)

        # 3. Evaluate Attention Policy Model
        attn_policy = AttentionPolicy(d_model=64, n_heads=4, n_layers=1).to(DEVICE)
        attn_path = f'models_attn/attn_M{M}.pt'
        
        if os.path.exists(attn_path):
            checkpoint = torch.load(attn_path, map_location=DEVICE)
            attn_policy.load_state_dict(checkpoint['policy'])
            attn_policy.eval()
            
            a_objs, a_nodes, a_waois, a_prios = [], [], [], []
            rng = np.random.default_rng(42)
            
            for _ in range(N_EVAL_EPISODES):
                s = int(rng.integers(0, 10_000_000))
                env_attn = UAVEnv(base_params, seed=s)
                
                # attn_rollout returns 4 values: traj, log_probs, values, reward
                traj, log_probs, values, R = attn_rollout(attn_policy, env_attn, DEVICE, greedy=True)
                
                a_objs.append(env_attn.objective(traj))
                a_nodes.append(len(traj))
                a_waois.append(base_params.theta1 * env_attn.waoi(traj))
                a_prios.append(sum(env_attn.wi[j] for j in traj))
                
            data_tracker['Attention (Ours)']['obj'].append(np.mean(a_objs))
            data_tracker['Attention (Ours)']['nodes'].append(np.mean(a_nodes))
            data_tracker['Attention (Ours)']['waoi'].append(np.mean(a_waois))
            data_tracker['Attention (Ours)']['priority'].append(np.mean(a_prios))
            print(f"  -> Attention Policy Evaluated. Avg Obj: {np.mean(a_objs):.2f}")
        else:
            print(f"  [!] Warning: Checkpoint missing at {attn_path}. Filling with NaN.")
            for met in metrics: data_tracker['Attention (Ours)'][met].append(np.nan)

    print("\nTraining Complete. Generating Benchmark Visualizations...")

    # --- VISUALIZATION PLOTTING ---
    plot_configs = [
        ('obj',      'Composite Objective (Lower is Better)',      'Objective Value'),
        ('waoi',     'Weighted Age of Information (Lower is Better)', 'Normalised θ₁ · WAoI'),
        ('priority', 'Total Priority Points Harvested (Higher is Better)', 'Σ wᵢ Collected'),
        ('nodes',    'Average Tour Length (Nodes Visited)',          'Nodes Count')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for idx, (m_key, title, ylabel) in enumerate(plot_configs):
        ax = axes[idx]
        
        for method in METHODS:
            if np.isnan(data_tracker[method][m_key]).all():
                continue
                
            ax.plot(M_VALUES, data_tracker[method][m_key], 
                    marker=MARKERS[method], color=COLORS[method], 
                    label=method, linewidth=2, markersize=6, alpha=0.9)
            
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Network Dimension Size (M)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(M_VALUES)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if idx == 0:
            ax.legend(loc='lower left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    output_fig_path = 'results_attn/model_comparison_curves.png'
    plt.savefig(output_fig_path, dpi=300)
    plt.close()
    
    print("="*75)
    print(f"Success! Performance comparison figure exported to: {output_fig_path}")
    print("="*75)

if __name__ == '__main__':
    main()