import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import from your existing files
from env import Params, UAVEnv
from policy import AttentionPolicy
from features import batch_rollout
from baselines import BASELINES

# ── Configuration & Colors ──────────────────────────────────────────────────
OUT_DIR = 'results_ppo_plots'
MODEL_DIR = 'models_ppo'
os.makedirs(OUT_DIR, exist_ok=True)

C = {
    'Random':           '#AAAAAA',
    'Nearest-Neighbor': '#E69F00',
    'Greedy-Priority':  '#56B4E9',
    'PDR':              '#CC79A7',
    'PPO Attention':    '#009E73', # Green
}
MK = {'Random':'s', 'Nearest-Neighbor':'^', 'Greedy-Priority':'D', 'PDR':'v', 'PPO Attention':'o'}
ALL_METHODS = ['Random', 'Nearest-Neighbor', 'Greedy-Priority', 'PDR', 'PPO Attention']

# Restricted to your specified PPO M values
M_LIST = [20, 30, 40, 50]

def ci95(a): return 1.96 * np.std(a) / np.sqrt(max(len(a), 1))

def get_ppo_policy(device, M):
    """Loads the original full-capacity AttentionPolicy used by PPO."""
    policy = AttentionPolicy(d_model=128, n_heads=8, n_layers=3).to(device)
    
    # Check multiple possible naming conventions in models_ppo
    model_paths = [f'{MODEL_DIR}/ppo_M{M}.pt', f'{MODEL_DIR}/ppo_policy_M{M}.pt', 
                   f'{MODEL_DIR}/policy_M{M}.pt', f'{MODEL_DIR}/attn_M{M}.pt']
    
    loaded = False
    for path in model_paths:
        if os.path.exists(path):
            ck = torch.load(path, map_location=device)
            policy.load_state_dict(ck.get('policy', ck))
            print(f"  [Loaded PPO model for M={M} from: {path}]")
            loaded = True
            break
            
    if not loaded:
        print(f"  [!] No PPO model found for M={M} in {MODEL_DIR}/. Using untrained weights.")
        
    policy.eval()
    return policy

# ── 1. Data Collection (M=30 Base) ──────────────────────────────────────────
def collect_m30_data(device, n_eval=100):
    print(f"Collecting data for M=30 ({n_eval} instances)...")
    params = Params(M=30)
    rng = np.random.default_rng(42)
    
    data = {m: {'obj': [], 'nodes': [], 'waoi': [], 'priority': [], 'reward': []} for m in ALL_METHODS}
    
    # 1. Collect Raw Baseline Data
    for _ in range(n_eval):
        s = int(rng.integers(0, 10_000_000))
        for bname, bfunc in BASELINES.items():
            env = UAVEnv(params, seed=s)
            traj = bfunc(env)
            data[bname]['obj'].append(env.objective(traj))
            data[bname]['nodes'].append(len(traj))
            data[bname]['waoi'].append(params.theta1 * env.waoi(traj))
            data[bname]['priority'].append(sum(env.wi[j] for j in traj))
            data[bname]['reward'].append(env.return_reward(traj))
            
    # 2. Collect PPO Data
    policy = get_ppo_policy(device, M=30)
    rng = np.random.default_rng(42)
    
    for _ in range(n_eval):
        s = int(rng.integers(0, 10_000_000))
        env = UAVEnv(params, seed=s)
        with torch.no_grad():
            traj, _, _, r = batch_rollout(policy, env, device, greedy=True)
            
        data['PPO Attention']['obj'].append(env.objective(traj))
        data['PPO Attention']['nodes'].append(len(traj))
        data['PPO Attention']['waoi'].append(params.theta1 * env.waoi(traj))
        data['PPO Attention']['priority'].append(sum(env.wi[j] for j in traj))
        data['PPO Attention']['reward'].append(r)
        
    for m in ALL_METHODS:
        for k in data[m].keys():
            data[m][k] = np.array(data[m][k])
            
    return data, policy

# ── 2. Plotting Functions ───────────────────────────────────────────────────

def plot_performance_bars(data):
    print("  -> Generating Fig 1: Performance Bars...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=150)
    fig.suptitle('Performance Comparison (PPO) — M=30, 100 instances', fontsize=14, fontweight='bold')
    
    metrics = [
        ('waoi', '(a) Weighted AoI (θ₁·WAoI)', 'Scaled WAoI'),
        ('priority', '(b) Total Priority', 'Sum of wi'),
        ('obj', '(c) Composite Objective', 'θ₁·WAoI − θ₂·Σwi')
    ]
    
    for ax, (key, title, ylabel) in zip(axes, metrics):
        vals = [data[m][key].mean() for m in ALL_METHODS]
        bars = ax.bar(range(len(ALL_METHODS)), vals, color=[C[m] for m in ALL_METHODS], edgecolor='black', lw=0.7)
        bars[-1].set_linewidth(2.5)
        
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(ALL_METHODS)))
        ax.set_xticklabels(ALL_METHODS, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, v in zip(bars, vals):
            yoff = 0.5 if v >= 0 else -2.5
            ax.text(bar.get_x() + bar.get_width()/2, v + yoff, f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig1_performance_bars.png')
    plt.close()

def plot_distributions(data):
    print("  -> Generating Fig 2: Reward/Nodes Distributions...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.suptitle('PPO Attention — Result Distribution (Original Style)', fontsize=14, fontweight='bold')
    
    for ax, key, ylabel, title in [(ax1, 'reward', 'Episode Reward (= −Objective)', '(a) Reward Distribution'),
                                   (ax2, 'nodes', 'Nodes Visited', '(b) Nodes Visited per Episode')]:
        bp = ax.boxplot([data[m][key] for m in ALL_METHODS], patch_artist=True, notch=False, whis=[5, 95],
                        medianprops=dict(color='black', lw=2.5), whiskerprops=dict(lw=1.4),
                        capprops=dict(lw=1.4), flierprops=dict(marker='o', ms=3, alpha=0.3))
        for patch, m in zip(bp['boxes'], ALL_METHODS):
            patch.set_facecolor(C[m]); patch.set_alpha(0.8)
            
        ax.set_xticks(range(1, len(ALL_METHODS)+1))
        ax.set_xticklabels(ALL_METHODS, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        if key == 'nodes': ax.axhline(30, color='gray', ls='--', lw=1)
        if key == 'reward': ax.axhline(0, color='red', ls='--', lw=0.8, alpha=0.6)
        ax.grid(axis='y', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig2_reward_nodes_dist.png')
    plt.close()

def plot_battery_sensitivity(policy, device, n_eval=50):
    print("  -> Generating Fig 3: Battery Sensitivity...")
    emax_list = [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000]
    results = {m: [] for m in ALL_METHODS}
    
    for Emax in emax_list:
        params = Params(M=30, Emax=Emax)
        rng = np.random.default_rng(42)
        
        b_raw_obj = {m: [] for m in ALL_METHODS[:-1]}
        for _ in range(n_eval):
            s = int(rng.integers(0, 10_000_000))
            for bname, bfunc in BASELINES.items():
                env = UAVEnv(params, seed=s)
                b_raw_obj[bname].append(env.objective(bfunc(env)))
        for m in ALL_METHODS[:-1]:
            results[m].append(np.mean(b_raw_obj[m]))
            
        t_objs = []
        rng = np.random.default_rng(42)
        for _ in range(n_eval):
            s = int(rng.integers(0, 10_000_000))
            env = UAVEnv(params, seed=s)
            with torch.no_grad():
                traj, _, _, _ = batch_rollout(policy, env, device, greedy=True)
            t_objs.append(env.objective(traj))
        results['PPO Attention'].append(np.mean(t_objs))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    x_vals = [e/1000 for e in emax_list]
    for m in ALL_METHODS:
        ax.plot(x_vals, results[m], marker=MK[m], color=C[m], label=m, lw=2.5 if m == 'PPO Attention' else 1.5, markersize=8)
        
    ax.set_xlabel('Battery Capacity Emax (kJ)', fontsize=12)
    ax.set_ylabel('Composite Objective (lower = better)', fontsize=12)
    ax.set_title('Battery Sensitivity (PPO) — M=30', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig3_battery_sensitivity.png')
    plt.close()

def plot_trajectory(policy, device):
    print("  -> Generating Fig 4: Trajectory Plot...")
    params = Params(M=30)
    env = UAVEnv(params, seed=7)
    with torch.no_grad():
        traj, _, _, _ = batch_rollout(policy, env, device, greedy=True)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    ax.set_xlim(-30, params.area+30); ax.set_ylim(-30, params.area+30)

    for i in range(params.M):
        pos = env.pos[i]; wi = env.wi[i]
        sz = 40 + (wi/params.wi_hi)*140
        clr = C['PPO Attention'] if i in traj else '#cccccc'
        ew = 1.3 if i in traj else 0.3
        ax.scatter(*pos, s=sz, c=clr, edgecolors='black', lw=ew, zorder=4)
        ax.text(pos[0]+15, pos[1]+15, f'w={wi:.1f}', fontsize=7, color='#222')

    px = [params.home[0]] + [env.pos[j][0] for j in traj] + [params.home[0]]
    py = [params.home[1]] + [env.pos[j][1] for j in traj] + [params.home[1]]
    ax.plot(px, py, '-', color=C['PPO Attention'], lw=2, alpha=0.8, zorder=3)
    
    for k in range(len(px)-1):
        ax.annotate('', xy=(px[k+1], py[k+1]), xytext=(px[k], py[k]),
                    arrowprops=dict(arrowstyle='->', color=C['PPO Attention'], lw=1.5, mutation_scale=15))
                    
    for order, j in enumerate(traj):
        ax.text(env.pos[j][0]-18, env.pos[j][1]-18, str(order+1),
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor=C['PPO Attention'], edgecolor='none', pad=0.1))

    ax.scatter(*params.home, s=250, marker='*', c='gold', edgecolors='black', lw=1.5, zorder=6)
    
    obj = env.objective(traj)
    waoi = params.theta1 * env.waoi(traj)
    ax.set_title(f'PPO Trajectory — M=30, visited={len(traj)}\nWAoI={waoi:.1f}, Obj={obj:.2f}', fontsize=13, fontweight='bold')
    ax.set_xlabel('x (m)', fontsize=11); ax.set_ylabel('y (m)', fontsize=11)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_trajectory.png')
    plt.close()

def plot_waoi_reduction(data):
    print("  -> Generating Fig 5: WAoI Reduction Bars...")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ref = data['Random']['waoi'].mean()
    methods = ALL_METHODS[1:]
    reds = [100 * (1 - data[m]['waoi'].mean() / ref) for m in methods]
    
    bars = ax.bar(range(len(methods)), reds, color=[C[m] for m in methods], edgecolor='black', lw=0.7)
    bars[-1].set_linewidth(2.5)
    
    ax.axhline(66, color='red', ls='--', lw=2, label='Paper D3QN claims >66%')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('WAoI Reduction vs Random (%)', fontsize=11)
    ax.set_title('PPO Attention — WAoI Reduction\nvs All Baselines (M=30)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, v in zip(bars, reds):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5 if v > 0 else v - 8, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig5_waoi_reduction.png')
    plt.close()

def plot_grouped_boxplots(data):
    print("  -> Generating Fig 6: Grouped Boxplots (MLP Style)...")
    import seaborn as sns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.suptitle('PPO Attention — Result Distribution (whiskers=5th/95th pct)', fontsize=14, fontweight='bold')
    
    df_obj, df_nodes = [], []
    for m in ALL_METHODS:
        for v in data[m]['obj']: df_obj.append({'Method': m, 'Value': v})
        for v in data[m]['nodes']: df_nodes.append({'Method': m, 'Value': v})
            
    df_obj = pd.DataFrame(df_obj)
    df_nodes = pd.DataFrame(df_nodes)

    sns.boxplot(data=df_obj, x='Method', y='Value', palette=C, ax=ax1, notch=False, whis=[5, 95],
                showmeans=True, meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
    ax1.set_title('(a) Objective  ↓ lower=better', fontsize=11)
    ax1.set_ylabel('Composite Objective', fontsize=11)
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', rotation=20)
    ax1.grid(axis='y', alpha=0.3)

    sns.boxplot(data=df_nodes, x='Method', y='Value', palette=C, ax=ax2, notch=False, whis=[5, 95],
                showmeans=True, meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
    ax2.set_title('(b) Nodes Visited per Episode', fontsize=11)
    ax2.set_ylabel('Nodes Visited', fontsize=11)
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=20)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(30, color='gray', ls='--', lw=1)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig6_grouped_boxplots.png')
    plt.close()

def plot_multi_m_scalability(device, n_eval=100):
    print(f"  -> Generating Fig 7 & 8: Multi-M Scalability for {M_LIST}...")
    
    res_obj = {m: [] for m in ALL_METHODS}
    res_nodes = {m: [] for m in ALL_METHODS}
    err_nodes = {m: [] for m in ALL_METHODS}

    for M in M_LIST:
        params = Params(M=M)
        rng = np.random.default_rng(42)
        
        b_raw_obj = {m: [] for m in ALL_METHODS[:-1]}
        b_raw_nodes = {m: [] for m in ALL_METHODS[:-1]}
        for _ in range(n_eval):
            s = int(rng.integers(0, 10_000_000))
            for bname, bfunc in BASELINES.items():
                env = UAVEnv(params, seed=s)
                traj = bfunc(env)
                b_raw_obj[bname].append(env.objective(traj))
                b_raw_nodes[bname].append(len(traj))
                
        for m in ALL_METHODS[:-1]:
            res_obj[m].append(np.mean(b_raw_obj[m]))
            res_nodes[m].append(np.mean(b_raw_nodes[m]))
            err_nodes[m].append(ci95(b_raw_nodes[m]))
            
        policy = get_ppo_policy(device, M)
        
        rng = np.random.default_rng(42)
        t_objs, t_nodes = [], []
        for _ in range(n_eval):
            s = int(rng.integers(0, 10_000_000))
            env = UAVEnv(params, seed=s)
            with torch.no_grad():
                traj, _, _, _ = batch_rollout(policy, env, device, greedy=True)
            t_objs.append(env.objective(traj))
            t_nodes.append(len(traj))
            
        res_obj['PPO Attention'].append(np.mean(t_objs))
        res_nodes['PPO Attention'].append(np.mean(t_nodes))
        err_nodes['PPO Attention'].append(ci95(t_nodes))

    # Fig 7: Nodes vs M
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for m in ALL_METHODS:
        ax.errorbar(M_LIST, res_nodes[m], yerr=err_nodes[m], marker=MK[m], color=C[m], 
                    label=m + (' (per-M model)' if m == 'PPO Attention' else ''), 
                    lw=2.5 if m == 'PPO Attention' else 1.5, markersize=8, capsize=4)
        
    ax.set_xlabel('Number of Sensor Nodes M', fontsize=12)
    ax.set_ylabel('Average Nodes Visited per Episode', fontsize=12)
    ax.set_title('PPO Attention — Nodes Visited vs Network Size\n(separate model per M, 100 instances each) ± 95% CI', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xticks(M_LIST)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig7_nodes_vs_M.png')
    plt.close()

    # Fig 8: Objective vs M
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for m in ALL_METHODS:
        ax.plot(M_LIST, res_obj[m], marker=MK[m], color=C[m], label=m, lw=2.5 if m == 'PPO Attention' else 1.5, markersize=8)
    ax.set_xlabel('Number of Nodes M', fontsize=12)
    ax.set_ylabel('Composite Objective (lower = better)', fontsize=12)
    ax.set_title('Scalability (PPO): Objective vs Network Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(M_LIST)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig8_objective_vs_M.png')
    plt.close()

# ── Execute ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"============================================================")
    print(f"Generating 8 PPO Attention Plots (Device: {device})")
    print(f"============================================================")
    
    data, m30_policy = collect_m30_data(device, n_eval=100)
    
    plot_performance_bars(data)
    plot_distributions(data)
    plot_battery_sensitivity(m30_policy, device, n_eval=50)
    plot_trajectory(m30_policy, device)
    plot_waoi_reduction(data)
    plot_grouped_boxplots(data)
    plot_multi_m_scalability(device, n_eval=100)
    
    print(f"\nDone! All 8 plots successfully saved to ./{OUT_DIR}/")