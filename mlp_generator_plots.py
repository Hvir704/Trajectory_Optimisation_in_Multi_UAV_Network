import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the environment and model classes directly from your base file
from uav_aoi_solver import Env, Policy, rollout, run_baseline, P, BASELINES, COLORS

def generate_advanced_plots():
    output_dir = "results(mlp)(new)"
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    m_values = [20, 40, 60, 80, 100]
    num_instances = 100
    
    eval_data = []
    time_data = []
    
    # Update colors to include the specific label used in your base code
    plot_colors = dict(COLORS)
    if 'MLP-REINFORCE' not in plot_colors:
        plot_colors['MLP-REINFORCE'] = plot_colors.get('Attention (Ours)', '#009E73')

    print("Starting evaluation for Robustness and Inference Time plots...")
    
    for m in m_values:
        print(f"  Evaluating Network Size M={m}...")
        
        # Load the specific trained model for this M
        model_path = f"models/policy_M{m}.pt"
        policy = Policy(hidden=256).to(device)
        
        if os.path.exists(model_path):
            ck = torch.load(model_path, map_location=device)
            policy.load_state_dict(ck['policy'])
        else:
            print(f"  [Warning] {model_path} not found. Using untrained weights for M={m}.")
            
        policy.eval()

        mlp_times = []
        baseline_times = {name: [] for name in BASELINES.keys()}
        
        rng = np.random.default_rng(42 + m)
        
        for _ in range(num_instances):
            seed = int(rng.integers(0, 10_000_000))
            
            # --- Evaluate MLP ---
            env_mlp = Env(M=m, seed=seed)
            start_time = time.perf_counter()
            with torch.no_grad():
                traj_mlp, *_ = rollout(policy, env_mlp, device, greedy=True)
            end_time = time.perf_counter()
            
            mlp_times.append(end_time - start_time)
            eval_data.append({
                'Network Size (M)': m, 
                'Objective': env_mlp.objective(traj_mlp), 
                'Method': 'MLP-REINFORCE'
            })

            # --- Evaluate Baselines ---
            for name, key in BASELINES.items():
                env_base = Env(M=m, seed=seed)
                start_time = time.perf_counter()
                traj_base = run_baseline(env_base, key)
                end_time = time.perf_counter()
                
                baseline_times[name].append(end_time - start_time)
                eval_data.append({
                    'Network Size (M)': m, 
                    'Objective': env_base.objective(traj_base), 
                    'Method': name
                })

        # Record mean inference times
        time_record = {'Network Size (M)': m, 'MLP-REINFORCE': np.mean(mlp_times)}
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

    plt.title('Robustness at Scale: Objective Distribution vs Network Size', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Sensor Nodes (M)', fontsize=14)
    plt.ylabel('Composite Objective (lower is better)', fontsize=14)
    plt.legend(title='Routing Strategy', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot1_robustness_boxplot.png"), bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Inference Time Scalability
    # ---------------------------------------------------------
    print("Generating Inference Time Plot...")
    plt.figure(figsize=(10, 6), dpi=300)

    methods_to_plot = ['MLP-REINFORCE', 'Nearest-Neighbor', 'Greedy-Priority', 'PDR']
    for method in methods_to_plot:
        plt.plot(df_time['Network Size (M)'], df_time[method], marker='o', 
                 linewidth=2.5 if method == 'MLP-REINFORCE' else 1.5, 
                 label=method, color=plot_colors.get(method, '#000000'))

    plt.title('Execution Time Scalability: RL vs Baselines', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Sensor Nodes (M)', fontsize=14)
    plt.ylabel('Average Inference Time per Episode (seconds)', fontsize=14)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot2_inference_time.png"))
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: Trajectory Grid (2x2)
    # ---------------------------------------------------------
    print("Generating Trajectory Grid...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=300)
    grid_m_values = [20, 50, 80, 100]

    for idx, ax in enumerate(axes.flatten()):
        m = grid_m_values[idx]
        
        # Initialize env and run single greedy rollout
        env = Env(M=m, seed=777)
        model_path = f"models/policy_M{m}.pt"
        policy = Policy(hidden=256).to(device)
        if os.path.exists(model_path):
            ck = torch.load(model_path, map_location=device)
            policy.load_state_dict(ck['policy'])
        policy.eval()
        
        with torch.no_grad():
            traj, *_ = rollout(policy, env, device, greedy=True)
            
        # Draw Trajectory
        for i in range(m):
            sz = 40 + (env.wi[i] / P.wi_hi) * 140
            clr = plot_colors['MLP-REINFORCE'] if i in traj else '#cccccc'
            ew = 1.3 if i in traj else 0.3
            ax.scatter(*env.pos[i], s=sz, c=clr, edgecolors='black', lw=ew, zorder=4)
            if m <= 50: # Only show priority weights if not too cluttered
                ax.text(env.pos[i,0]+12, env.pos[i,1]+12, f'w={env.wi[i]:.1f}', fontsize=6.5)

        px = [P.home[0]] + [env.pos[j,0] for j in traj] + [P.home[0]]
        py = [P.home[1]] + [env.pos[j,1] for j in traj] + [P.home[1]]
        ax.plot(px, py, '-', color=plot_colors['MLP-REINFORCE'], lw=1.8, alpha=0.75, zorder=3)
        
        ax.scatter(*P.home, s=220, marker='*', c='gold', edgecolors='black', lw=1.5, zorder=6)
        
        obj = env.objective(traj)
        ax.set_title(f'Learned Strategy M={m} (Visited: {len(traj)} nodes)\nComposite Obj: {obj:.2f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(alpha=0.2)
        ax.set_xlim(-30, P.area+30)
        ax.set_ylim(-30, P.area+30)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot3_trajectory_grid.png"))
    plt.close()

    print(f"All plots successfully generated and saved to ./{output_dir}/")

if __name__ == "__main__":
    generate_advanced_plots()