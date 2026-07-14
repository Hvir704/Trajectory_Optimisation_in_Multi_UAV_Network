import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import from your existing files
from env import Params, UAVEnv
from policy import AttentionPolicy
from features import batch_rollout
from baselines import BASELINES, evaluate_all

def generate_attn_advanced_plots():
    output_dir = "results_attn_advanced"
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Network sizes to evaluate for Boxplot and Inference Time
    m_values = [20, 40, 60, 80, 100]
    num_instances = 100
    
    eval_data = []
    time_data = []
    
    # Custom color palette matching your previous styling
    plot_colors = {
        'Random': '#A9A9A9',             
        'Nearest-Neighbor': '#E69F00',   
        'Greedy-Priority': '#56B4E9',
        'PDR': '#CC79A7',
        'Transformer-REINFORCE': '#009E73' # Green for our model
    }

    print("Starting evaluation for Transformer Robustness and Inference Time plots...")
    
    for m in m_values:
        print(f"  Evaluating Network Size M={m}...")
        params = Params(M=m)
        
        # Initialize the TINY Transformer architecture used in train_all_M.py
        policy = AttentionPolicy(d_model=64, n_heads=4, n_layers=1).to(device)
        
        # Load the specific trained model for this M
        model_path = f"models_attn/attn_M{m}.pt"
        if os.path.exists(model_path):
            ck = torch.load(model_path, map_location=device)
            # Handle standard save format
            state = ck.get('policy', ck)
            policy.load_state_dict(state)
        else:
            print(f"  [Warning] {model_path} not found. Using untrained weights for M={m}.")
            
        policy.eval()

        attn_times = []
        baseline_times = {name: [] for name in BASELINES.keys()}
        
        rng = np.random.default_rng(42 + m)
        
        for _ in range(num_instances):
            seed = int(rng.integers(0, 10_000_000))
            
            # --- Evaluate Transformer ---
            env_attn = UAVEnv(params, seed=seed)
            start_time = time.perf_counter()
            with torch.no_grad():
                traj_attn, _, _, _ = batch_rollout(policy, env_attn, device, greedy=True)
            end_time = time.perf_counter()
            
            attn_times.append(end_time - start_time)
            eval_data.append({
                'Network Size (M)': m, 
                'Objective': env_attn.objective(traj_attn), 
                'Method': 'Transformer-REINFORCE'
            })

            # --- Evaluate Baselines ---
            for name, fn in BASELINES.items():
                env_base = UAVEnv(params, seed=seed)
                start_time = time.perf_counter()
                traj_base = fn(env_base)
                end_time = time.perf_counter()
                
                baseline_times[name].append(end_time - start_time)
                eval_data.append({
                    'Network Size (M)': m, 
                    'Objective': env_base.objective(traj_base), 
                    'Method': name
                })

        # Record mean inference times
        time_record = {'Network Size (M)': m, 'Transformer-REINFORCE': np.mean(attn_times)}
        for name in BASELINES.keys():
            time_record[name] = np.mean(baseline_times[name])
        time_data.append(time_record)

    df_eval = pd.DataFrame(eval_data)
    df_time = pd.DataFrame(time_data)

    # ---------------------------------------------------------
    # PLOT 1: Robustness Grouped Boxplot
    # ---------------------------------------------------------
    print("Generating Robustness Boxplot...")
    plt.figure(figsize=(14, 7), dpi=300)
    sns.set_theme(style="whitegrid")

    sns.boxplot(
        data=df_eval, x='Network Size (M)', y='Objective', hue='Method',
        palette=plot_colors, linewidth=1.2, fliersize=2, showmeans=True, 
        meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black"}
    )

    plt.title('Transformer Robustness at Scale: Objective Distribution vs Network Size', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Sensor Nodes (M)', fontsize=14)
    plt.ylabel('Composite Objective (lower is better)', fontsize=14)
    plt.legend(title='Routing Strategy', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot1_attn_robustness_boxplot.png"), bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Inference Time Scalability
    # ---------------------------------------------------------
    print("Generating Inference Time Plot...")
    plt.figure(figsize=(10, 6), dpi=300)

    methods_to_plot = ['Transformer-REINFORCE', 'Nearest-Neighbor', 'Greedy-Priority', 'PDR']
    for method in methods_to_plot:
        plt.plot(df_time['Network Size (M)'], df_time[method], marker='o', 
                 linewidth=2.5 if method == 'Transformer-REINFORCE' else 1.5, 
                 label=method, color=plot_colors.get(method, '#000000'))

    plt.title('Execution Time Scalability: Transformer vs Baselines', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Sensor Nodes (M)', fontsize=14)
    plt.ylabel('Average Inference Time per Episode (seconds)', fontsize=14)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot2_attn_inference_time.png"))
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: Trajectory Grid (2x2)
    # ---------------------------------------------------------
    print("Generating Trajectory Grid...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=300)
    grid_m_values = [20, 50, 80, 100]

    for idx, ax in enumerate(axes.flatten()):
        m = grid_m_values[idx]
        params = Params(M=m)
        
        # Initialize env and run single greedy rollout
        env = UAVEnv(params, seed=777)
        policy = AttentionPolicy(d_model=64, n_heads=4, n_layers=1).to(device)
        model_path = f"models_attn/attn_M{m}.pt"
        
        if os.path.exists(model_path):
            ck = torch.load(model_path, map_location=device)
            state = ck.get('policy', ck)
            policy.load_state_dict(state)
        policy.eval()
        
        with torch.no_grad():
            traj, _, _, _ = batch_rollout(policy, env, device, greedy=True)
            
        # Draw Trajectory
        for i in range(m):
            pos = env.pos[i]
            wi = env.wi[i]
            sz = 40 + (wi / params.wi_hi) * 140
            clr = plot_colors['Transformer-REINFORCE'] if i in traj else '#cccccc'
            ew = 1.3 if i in traj else 0.3
            ax.scatter(*pos, s=sz, c=clr, edgecolors='black', lw=ew, zorder=4)
            if m <= 50: # Only show priority weights if not too cluttered
                ax.text(pos[0]+12, pos[1]+12, f'w={wi:.1f}', fontsize=6.5, color='#222')

        px = ([params.home[0]] + [env.pos[j][0] for j in traj] + [params.home[0]])
        py = ([params.home[1]] + [env.pos[j][1] for j in traj] + [params.home[1]])
        ax.plot(px, py, '-', color=plot_colors['Transformer-REINFORCE'], lw=1.8, alpha=0.75, zorder=3)
        
        ax.scatter(*params.home, s=220, marker='*', c='gold', edgecolors='black', lw=1.5, zorder=6)
        
        obj = env.objective(traj)
        ax.set_title(f'Learned Transformer Strategy M={m} (Visited: {len(traj)} nodes)\nComposite Obj: {obj:.2f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(alpha=0.2)
        ax.set_xlim(-30, params.area+30)
        ax.set_ylim(-30, params.area+30)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot3_attn_trajectory_grid.png"))
    plt.close()

    print(f"All plots successfully generated and saved to ./{output_dir}/")

if __name__ == "__main__":
    generate_attn_advanced_plots()