import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib.util

# ── 1. Import Transformer Components ─────────────────────────────────────────
from env import Params, UAVEnv
from policy import AttentionPolicy
from features import batch_rollout

# ── 2. Import MLP Components (Dynamically to avoid collision) ────────────────
spec = importlib.util.spec_from_file_location('mlp_sol', 'uav_aoi_solver.py')
mlp_sol = importlib.util.module_from_spec(spec)
sys.modules['mlp_sol'] = mlp_sol
spec.loader.exec_module(mlp_sol)

# ── 3. Configuration & Aesthetics ────────────────────────────────────────────
OUT_DIR = 'results_grand_showdown'
os.makedirs(OUT_DIR, exist_ok=True)

M_LIST_FULL = [20, 30, 40, 50, 60, 70, 80, 90, 100]
M_LIST_SHORT = [20, 30, 40, 50]

METHODS = [
    'Random', 'Nearest-Neighbor', 'Greedy-Priority', 'PDR',
    'MLP-REINFORCE', 'Tiny Transformer', 'PPO Attention'
]

C = {
    'Random':           '#AAAAAA',
    'Nearest-Neighbor': '#E69F00',
    'Greedy-Priority':  '#56B4E9',
    'PDR':              '#CC79A7',
    'MLP-REINFORCE':    '#009E73', # Green
    'Tiny Transformer': '#0072B2', # Blue
    'PPO Attention':    '#D55E00', # Red/Orange
}

MK = {
    'Random':'s', 'Nearest-Neighbor':'^', 'Greedy-Priority':'D', 'PDR':'v',
    'MLP-REINFORCE':'o', 'Tiny Transformer':'P', 'PPO Attention':'X'
}

def ci95(a): return 1.96 * np.std(a) / np.sqrt(max(len(a), 1))

# ── 4. Main Data Collection Loop ─────────────────────────────────────────────
def collect_all_data(device, n_eval=100):
    print(f"Starting Grand Showdown Evaluation ({n_eval} instances per M)...")
    
    res_obj = {m: {} for m in METHODS}
    res_nodes = {m: {} for m in METHODS}
    err_nodes = {m: {} for m in METHODS}

    for M in M_LIST_FULL:
        print(f"  -> Evaluating M={M}...")
        
        # 4a. Baselines (Evaluated using the MLP Env)
        b_raw_obj = {m: [] for m in METHODS[:4]}
        b_raw_nodes = {m: [] for m in METHODS[:4]}
        rng = np.random.default_rng(42)
        for _ in range(n_eval):
            s = int(rng.integers(0, 10_000_000))
            for bname, bkey in mlp_sol.BASELINES.items():
                env = mlp_sol.Env(M=M, seed=s)
                traj = mlp_sol.run_baseline(env, bkey)
                b_raw_obj[bname].append(env.objective(traj))
                b_raw_nodes[bname].append(len(traj))
                
        for m in METHODS[:4]:
            res_obj[m][M] = np.mean(b_raw_obj[m])
            res_nodes[m][M] = np.mean(b_raw_nodes[m])
            err_nodes[m][M] = ci95(b_raw_nodes[m])

        # 4b. MLP-REINFORCE
        mlp_policy = mlp_sol.Policy(hidden=256).to(device)
        mlp_path = f'models/policy_M{M}.pt'
        if os.path.exists(mlp_path):
            ck = torch.load(mlp_path, map_location=device)
            mlp_policy.load_state_dict(ck['policy'])
            mlp_policy.eval()
            
            mlp_objs, mlp_nodes = [], []
            rng = np.random.default_rng(42) # Reset seed for perfect map alignment
            for _ in range(n_eval):
                s = int(rng.integers(0, 10_000_000))
                env = mlp_sol.Env(M=M, seed=s)
                with torch.no_grad():
                    traj, *_ = mlp_sol.rollout(mlp_policy, env, device, greedy=True)
                mlp_objs.append(env.objective(traj))
                mlp_nodes.append(len(traj))
                
            res_obj['MLP-REINFORCE'][M] = np.mean(mlp_objs)
            res_nodes['MLP-REINFORCE'][M] = np.mean(mlp_nodes)
            err_nodes['MLP-REINFORCE'][M] = ci95(mlp_nodes)
        else:
            print(f"     [!] Missing {mlp_path}")

        # 4c. Tiny Transformer
        tiny_policy = AttentionPolicy(d_model=64, n_heads=4, n_layers=1).to(device)
        tiny_path = f'models_attn/attn_M{M}.pt'
        if os.path.exists(tiny_path):
            ck = torch.load(tiny_path, map_location=device)
            tiny_policy.load_state_dict(ck.get('policy', ck))
            tiny_policy.eval()
            
            tiny_objs, tiny_nodes = [], []
            rng = np.random.default_rng(42)
            params = Params(M=M)
            for _ in range(n_eval):
                s = int(rng.integers(0, 10_000_000))
                env = UAVEnv(params, seed=s)
                with torch.no_grad():
                    traj, _, _, _ = batch_rollout(tiny_policy, env, device, greedy=True)
                tiny_objs.append(env.objective(traj))
                tiny_nodes.append(len(traj))
                
            res_obj['Tiny Transformer'][M] = np.mean(tiny_objs)
            res_nodes['Tiny Transformer'][M] = np.mean(tiny_nodes)
            err_nodes['Tiny Transformer'][M] = ci95(tiny_nodes)
        else:
            print(f"     [!] Missing {tiny_path}")

        # 4d. PPO Attention (Only up to M=50)
        if M <= 50:
            ppo_policy = AttentionPolicy(d_model=128, n_heads=8, n_layers=3).to(device)
            ppo_paths = [f'models_ppo/ppo_M{M}.pt', f'models_ppo/ppo_policy_M{M}.pt', f'models_ppo/policy_M{M}.pt', f'models_ppo/attn_M{M}.pt']
            loaded = False
            for p in ppo_paths:
                if os.path.exists(p):
                    ck = torch.load(p, map_location=device)
                    ppo_policy.load_state_dict(ck.get('policy', ck))
                    loaded = True
                    break
                    
            if loaded:
                ppo_policy.eval()
                ppo_objs, ppo_nodes = [], []
                rng = np.random.default_rng(42)
                params = Params(M=M)
                for _ in range(n_eval):
                    s = int(rng.integers(0, 10_000_000))
                    env = UAVEnv(params, seed=s)
                    with torch.no_grad():
                        traj, _, _, _ = batch_rollout(ppo_policy, env, device, greedy=True)
                    ppo_objs.append(env.objective(traj))
                    ppo_nodes.append(len(traj))
                    
                res_obj['PPO Attention'][M] = np.mean(ppo_objs)
                res_nodes['PPO Attention'][M] = np.mean(ppo_nodes)
                err_nodes['PPO Attention'][M] = ci95(ppo_nodes)
            else:
                print(f"     [!] Missing PPO model for M={M} in models_ppo/")

    return res_obj, res_nodes, err_nodes

# ── 5. Plotting Functions ────────────────────────────────────────────────────

def generate_plots(res_obj, res_nodes, err_nodes):
    
    # =========================================================================
    # TASK 1: MLP vs Tiny Transformer (M=20..100)
    # =========================================================================
    print("  -> Plotting Task 1: MLP vs Tiny Transformer (M=20 to 100)...")
    methods_task1 = METHODS[:6] # Excludes PPO
    
    # Plot 1A: Objective vs M
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for m in methods_task1:
        if M_LIST_FULL[0] in res_obj[m]:
            y_vals = [res_obj[m][M] for M in M_LIST_FULL if M in res_obj[m]]
            x_vals = [M for M in M_LIST_FULL if M in res_obj[m]]
            lw = 2.5 if 'REINFORCE' in m or 'Transformer' in m else 1.5
            ax.plot(x_vals, y_vals, marker=MK[m], color=C[m], label=m, lw=lw, markersize=8)
            
    ax.set_xlabel('Number of Nodes M', fontsize=12)
    ax.set_ylabel('Composite Objective (lower = better)', fontsize=12)
    ax.set_title('Architecture Comparison: Objective vs Network Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(M_LIST_FULL)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/Task1_Objective_vs_M_MLP_vs_Tiny.png')
    plt.close()

    # Plot 1B: Nodes vs M
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for m in methods_task1:
        if M_LIST_FULL[0] in res_nodes[m]:
            y_vals = [res_nodes[m][M] for M in M_LIST_FULL if M in res_nodes[m]]
            e_vals = [err_nodes[m][M] for M in M_LIST_FULL if M in err_nodes[m]]
            x_vals = [M for M in M_LIST_FULL if M in res_nodes[m]]
            lw = 2.5 if 'REINFORCE' in m or 'Transformer' in m else 1.5
            ax.errorbar(x_vals, y_vals, yerr=e_vals, marker=MK[m], color=C[m], label=m, lw=lw, markersize=8, capsize=4)
            
    ax.set_xlabel('Number of Sensor Nodes M', fontsize=12)
    ax.set_ylabel('Average Nodes Visited per Episode', fontsize=12)
    ax.set_title('Architecture Comparison: Nodes Visited vs Network Size\n± 95% CI', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xticks(M_LIST_FULL)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/Task1_Nodes_vs_M_MLP_vs_Tiny.png')
    plt.close()

    # =========================================================================
    # TASK 2: All 3 Models (M=20..50)
    # =========================================================================
    print("  -> Plotting Task 2: MLP vs Tiny vs PPO (M=20 to 50)...")
    
    # Plot 2A: Objective vs M
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    for m in METHODS:
        if M_LIST_SHORT[0] in res_obj[m]:
            y_vals = [res_obj[m][M] for M in M_LIST_SHORT if M in res_obj[m]]
            x_vals = [M for M in M_LIST_SHORT if M in res_obj[m]]
            lw = 2.5 if m not in METHODS[:4] else 1.5
            ax.plot(x_vals, y_vals, marker=MK[m], color=C[m], label=m, lw=lw, markersize=9)
            
    ax.set_xlabel('Number of Nodes M', fontsize=12)
    ax.set_ylabel('Composite Objective (lower = better)', fontsize=12)
    ax.set_title('Three-Way Showdown: Objective vs Network Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(M_LIST_SHORT)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/Task2_Objective_vs_M_All_Three.png')
    plt.close()

    # Plot 2B: Nodes vs M
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    for m in METHODS:
        if M_LIST_SHORT[0] in res_nodes[m]:
            y_vals = [res_nodes[m][M] for M in M_LIST_SHORT if M in res_nodes[m]]
            e_vals = [err_nodes[m][M] for M in M_LIST_SHORT if M in err_nodes[m]]
            x_vals = [M for M in M_LIST_SHORT if M in res_nodes[m]]
            lw = 2.5 if m not in METHODS[:4] else 1.5
            ax.errorbar(x_vals, y_vals, yerr=e_vals, marker=MK[m], color=C[m], label=m, lw=lw, markersize=9, capsize=4)
            
    ax.set_xlabel('Number of Sensor Nodes M', fontsize=12)
    ax.set_ylabel('Average Nodes Visited per Episode', fontsize=12)
    ax.set_title('Three-Way Showdown: Nodes Visited vs Network Size\n± 95% CI', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xticks(M_LIST_SHORT)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/Task2_Nodes_vs_M_All_Three.png')
    plt.close()

# ── 6. Execute ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("============================================================")
    print(f"  Executing Grand Comparative Analysis (Device: {device})")
    print("============================================================")
    
    res_obj, res_nodes, err_nodes = collect_all_data(device, n_eval=100)
    generate_plots(res_obj, res_nodes, err_nodes)
    
    print(f"\nDone! All 4 comparative plots saved to ./{OUT_DIR}/")