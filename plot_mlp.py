import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import directly from your single-file solver
from uav_aoi_solver import Policy, Env, rollout, run_baseline, BASELINES

def evaluate_mlp_nodes_visited(M_list, n_eval=50, seed=42):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating MLP Nodes Visited on {device}...")
    
    methods = ['Random', 'Nearest-Neighbor', 'Greedy-Priority', 'PDR', 'MLP (Ours)']
    results = {m: [] for m in methods}
    valid_M = []

    for M in M_list:
        model_path = f'models/policy_M{M}.pt'
        if not os.path.exists(model_path):
            print(f"  Skipping M={M}: {model_path} not found.")
            continue
        
        print(f"  Evaluating M={M}...")
        valid_M.append(M)
        
        # Load the saved Policy
        policy = Policy(hidden=256).to(device)
        ck = torch.load(model_path, map_location=device)
        policy.load_state_dict(ck['policy'])
        policy.eval()

        rng = np.random.default_rng(seed)
        nodes_tracker = {m: [] for m in methods}

        # Run evaluations
        for _ in range(n_eval):
            s = int(rng.integers(0, 10_000_000))
            
            # Baselines
            for base_name, base_key in BASELINES.items():
                env = Env(M=M, seed=s)
                traj = run_baseline(env, base_key)
                nodes_tracker[base_name].append(len(traj))
            
            # MLP
            env_mlp = Env(M=M, seed=s)
            with torch.no_grad():
                traj_mlp, *_ = rollout(policy, env_mlp, device, greedy=True)
            nodes_tracker['MLP (Ours)'].append(len(traj_mlp))
        
        for m in methods:
            results[m].append(np.mean(nodes_tracker[m]))

    # Plotting
    COLORS = {'Random': '#888888', 'Nearest-Neighbor': '#E69F00', 'Greedy-Priority': '#56B4E9', 'PDR': '#CC79A7', 'MLP (Ours)': '#D55E00'}
    MARKERS = {'Random': 's', 'Nearest-Neighbor': '^', 'Greedy-Priority': 'D', 'PDR': 'v', 'MLP (Ours)': '*'}
    
    plt.figure(figsize=(9, 5))
    
    # Plot a reference line for the total number of nodes available
    plt.plot(valid_M, valid_M, '--', color='gray', label='All Nodes Available (M)', lw=1.5)

    for m in methods:
        plt.plot(valid_M, results[m], marker=MARKERS[m], color=COLORS[m], label=m, lw=1.8, markersize=7)
    
    plt.xlabel('Total Nodes in Network (M)', fontsize=12)
    plt.ylabel('Average Nodes Visited by UAV', fontsize=12)
    plt.title('MLP Policy vs Baselines: Visitation Scaling', fontsize=13)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/mlp_standalone_nodes.png', dpi=150)
    plt.close()
    print("Done! Plot saved to results/mlp_standalone_nodes.png")

if __name__ == '__main__':
    evaluate_mlp_nodes_visited([20, 30, 40, 50, 60, 70, 80, 90, 100], n_eval=50)