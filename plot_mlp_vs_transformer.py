import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Environment and Transformer
from env import Params, UAVEnv
from policy import AttentionPolicy
from features import batch_rollout

# Import MLP (Assuming your base file is uav_aoi_solver.py)
import importlib.util
spec = importlib.util.spec_from_file_location('mlp_sol', 'uav_aoi_solver.py')
mlp_sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp_sol)

def ci95(data):
    return 1.96 * np.std(data) / np.sqrt(len(data))

def compare_architectures():
    output_dir = "results_comparison"
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Head-to-Head Evaluation on {device}...")

    m_values = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_instances = 100
    
    results = {
        'MLP-REINFORCE (per-M)': {'obj': [], 'err': []},
        'Tiny Transformer (per-M)': {'obj': [], 'err': []},
        'Random Baseline': {'obj': [], 'err': []} # Added for scale reference
    }

    for m in m_values:
        print(f"  Evaluating Network Size M={m}...")
        params = Params(M=m)
        
        # 1. Load MLP Model
        mlp_policy = mlp_sol.Policy(hidden=256).to(device)
        mlp_path = f"models/policy_M{m}.pt"
        if os.path.exists(mlp_path):
            mlp_ck = torch.load(mlp_path, map_location=device)
            mlp_policy.load_state_dict(mlp_ck['policy'])
        mlp_policy.eval()

        # 2. Load Tiny Transformer Model
        attn_policy = AttentionPolicy(d_model=64, n_heads=4, n_layers=1).to(device)
        attn_path = f"models_attn/attn_M{m}.pt"
        if os.path.exists(attn_path):
            attn_ck = torch.load(attn_path, map_location=device)
            attn_policy.load_state_dict(attn_ck.get('policy', attn_ck))
        attn_policy.eval()

        mlp_objs = []
        attn_objs = []
        rand_objs = []
        
        rng = np.random.default_rng(42 + m)
        
        for _ in range(num_instances):
            seed = int(rng.integers(0, 10_000_000))
            
            # MLP Eval
            env_mlp = mlp_sol.Env(M=m, seed=seed)
            with torch.no_grad():
                traj_mlp, *_ = mlp_sol.rollout(mlp_policy, env_mlp, device, greedy=True)
            mlp_objs.append(env_mlp.objective(traj_mlp))

            # Transformer Eval
            env_attn = UAVEnv(params, seed=seed)
            with torch.no_grad():
                traj_attn, _, _, _ = batch_rollout(attn_policy, env_attn, device, greedy=True)
            attn_objs.append(env_attn.objective(traj_attn))

            # Random Eval (for baseline scale)
            env_rand = mlp_sol.Env(M=m, seed=seed)
            traj_rand = mlp_sol.run_baseline(env_rand, 'random')
            rand_objs.append(env_rand.objective(traj_rand))

        # Store means and 95% Confidence Intervals
        results['MLP-REINFORCE (per-M)']['obj'].append(np.mean(mlp_objs))
        results['MLP-REINFORCE (per-M)']['err'].append(ci95(mlp_objs))
        
        results['Tiny Transformer (per-M)']['obj'].append(np.mean(attn_objs))
        results['Tiny Transformer (per-M)']['err'].append(ci95(attn_objs))

        results['Random Baseline']['obj'].append(np.mean(rand_objs))
        results['Random Baseline']['err'].append(ci95(rand_objs))

    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------
    print("\nGenerating Comparison Graph...")
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    colors = {
        'MLP-REINFORCE (per-M)': '#009E73',      # Green
        'Tiny Transformer (per-M)': '#0072B2',   # Blue
        'Random Baseline': '#AAAAAA'             # Gray
    }
    markers = {'MLP-REINFORCE (per-M)': 'o', 'Tiny Transformer (per-M)': 'P', 'Random Baseline': 's'}

    for method, data in results.items():
        plt.errorbar(
            m_values, data['obj'], yerr=data['err'], 
            marker=markers[method], color=colors[method], 
            label=method, lw=2.5 if 'Baseline' not in method else 1.5, 
            markersize=9 if 'Baseline' not in method else 6, capsize=4
        )

    # Reference line for net-zero objective (where priority exactly cancels WAoI penalty)
    plt.axhline(0, color='black', lw=1, ls='--')

    plt.title('Architecture Showdown: MLP vs Tiny Transformer\nComposite Objective vs Network Size (100 instances/M) ± 95% CI', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Sensor Nodes (M)', fontsize=12)
    plt.ylabel('Composite Objective (lower is better)', fontsize=12)
    plt.xticks(m_values)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "mlp_vs_transformer_objective.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Done! Head-to-head comparison saved to {save_path}")

if __name__ == "__main__":
    compare_architectures()