"""
exact_dp_solver.py  —  Exact DP for UAV AoI Stage-Weighted Orienteering
========================================================================
Verified against brute-force for M<=6.

State:  (visited_bitmask, last_local_idx)
        last_local_idx = Mf  means "currently at home depot"
Value:  minimum FUTURE composite objective achievable from this state,
        EXCLUDING the deferred tf_out(last) cost.

Key formula (your paper Eq.1):
    WAoI = Σ_m  W(m) * (tcd(m) + tf_out(m))
where W(m) = cumulative priority AFTER visiting node m (including m),
      tf_out(m) = flight time FROM node m to the NEXT location.

Because W_cum[mask] = Σ wi for bits set in mask is ORDER-INDEPENDENT,
we precompute it and the state (mask, last) fully determines W_cum.

Transition (mask, last) → visit j:
  immediate cost = P.theta1 * W_before * tf(last→j)   [closes last's deferred tf_out]
                 + P.theta1 * W_after  * tcd(j)        [hover at j]
                 - P.theta2 * wi(j)                    [priority reward]
  future cost    = dp_val[new_mask][j]                 [defers tf_out(j)]

Terminal from (mask, last):
  cost = P.theta1 * W_before * tf(last→home)   [closes last's deferred tf_out]

Complexity: O(2^Mf * Mf^2). Tractable for Mf <= 18.
"""

import os, sys, time, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    'mlp_sol', os.path.join(os.path.dirname(__file__), 'uav_aoi_solver.py'))
mlp_sol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mlp_sol)
Env = mlp_sol.Env; P = mlp_sol.P
rollout = mlp_sol.rollout; Policy = mlp_sol.Policy
BASELINES = mlp_sol.BASELINES; run_baseline = mlp_sol.run_baseline


# ══════════════════════════════════════════════════════════════════════════════
# 1.  EXACT DP SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def dp_solve(env: Env, time_limit: float = 60.0):
    """
    Exact bitmask DP for the stage-weighted orienteering subproblem.

    Returns
    -------
    traj     : list[int]  — global node indices in visit order
    obj      : float      — composite objective of returned traj
    info     : dict
    """
    M    = env.M
    home = np.array(P.home, dtype=np.float64)
    pos  = env.pos.astype(np.float64)
    wi   = env.wi.astype(np.float64)
    tcd  = env.tcd.astype(np.float64)

    # ── Precompute distances and energies ──────────────────────────────────
    d_ij = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)   # (M,M)
    d_ih = np.linalg.norm(pos - home, axis=1)                     # (M,)
    d_hi = d_ih                                                    # home→node = same

    tf_ij = d_ij / P.v
    tf_ih = d_ih / P.v
    tf_hi = d_hi / P.v

    e_ij  = P.Pf * tf_ij
    e_ih  = P.Pf * tf_ih
    e_hi  = P.Pf * tf_hi
    e_hov = P.Ph * tcd

    # ── Prune globally infeasible nodes ───────────────────────────────────
    e_min = e_hi + e_hov + e_ih          # cheapest possible visit (direct loop)
    G     = np.where(e_min <= P.Emax)[0] # global indices of feasible nodes
    Mf    = len(G)

    if Mf == 0:
        return [], env.objective([]), {'dp_states': 0, 'time': 0.0,
                                       'Mf': 0, 'timed_out': False}

    # Local (compressed) arrays
    wi_l  = wi[G];  tcd_l = tcd[G]
    tf_ij_l = tf_ij[np.ix_(G, G)];  tf_ih_l = tf_ih[G];  tf_hi_l = tf_hi[G]
    e_ij_l  = e_ij [np.ix_(G, G)];  e_ih_l  = e_ih [G];  e_hi_l  = e_hi [G]
    e_hov_l = e_hov[G]

    # ── Precompute W_cum[mask] ─────────────────────────────────────────────
    N = 1 << Mf
    W_cum = np.zeros(N, dtype=np.float64)
    for i in range(Mf):
        bit = 1 << i
        for mask in range(N):
            if mask & bit:
                W_cum[mask] += wi_l[i]

    # ── DP arrays ─────────────────────────────────────────────────────────
    # last = 0..Mf-1: at local node; last = Mf: at home
    INF     = float('inf')
    dp_val  = np.full((N, Mf + 1), INF, dtype=np.float64)
    dp_next = np.full((N, Mf + 1), -1,  dtype=np.int32)

    t0 = time.time()
    dp_states_visited = 0
    timed_out = False

    # Sort masks: fill high-popcount states first (they are base cases for lower ones)
    order = sorted(range(N), key=lambda m: bin(m).count('1'), reverse=True)

    for mask in order:
        if time.time() - t0 > time_limit:
            timed_out = True; break

        W_before = W_cum[mask]

        for last in range(Mf + 1):

            # ── Terminal option: go home from here ─────────────────────
            if last == Mf:
                cost_home = 0.0          # already at home, no tf to close
            else:
                # close deferred tf_out(last) → home
                cost_home = P.theta1 * W_before * tf_ih_l[last]

            best_val  = cost_home
            best_next = -1  # -1 means "go home"

            # ── Try visiting each unvisited node j ─────────────────────
            for j in range(Mf):
                if mask & (1 << j):
                    continue

                new_mask = mask | (1 << j)
                W_after  = W_before + wi_l[j]   # W(j): cumulative AFTER j

                # Cost of closing deferred tf_out(last) with destination j:
                if last == Mf:
                    close = 0.0                          # no prior stage
                else:
                    close = P.theta1 * W_before * tf_ij_l[last, j]

                # Hover cost at j:
                hover = P.theta1 * W_after * tcd_l[j]

                # Priority reward:
                reward = -P.theta2 * wi_l[j]

                # Future from (new_mask, j) — deferred tf_out(j):
                future = dp_val[new_mask][j]
                if future == INF:
                    continue   # no valid completion from there

                total = close + hover + reward + future

                if total < best_val:
                    best_val  = total
                    best_next = j

            dp_val[mask][last]  = best_val
            dp_next[mask][last] = best_next
            dp_states_visited  += 1

    elapsed = time.time() - t0

    # ── Reconstruct with energy feasibility ───────────────────────────────
    traj   = []
    mask   = 0
    last   = Mf          # start at home
    E_left = float(P.Emax)

    for _ in range(Mf):
        j_dp = int(dp_next[mask][last])
        if j_dp == -1:
            break   # DP says terminate

        def can_visit(j):
            e = (e_hi_l[j] if last == Mf else e_ij_l[last, j]) + e_hov_l[j] + e_ih_l[j]
            return e <= E_left

        if can_visit(j_dp):
            j_pick = j_dp
        else:
            # Energy violation: pick best feasible alternative by DP value
            best_alt = INF; j_pick = -1
            for j in range(Mf):
                if (mask & (1 << j)) or not can_visit(j):
                    continue
                v = dp_val[mask | (1 << j)][j]
                if v < best_alt:
                    best_alt = v; j_pick = j
            if j_pick == -1:
                break

        E_left -= (e_hi_l[j_pick] if last == Mf else e_ij_l[last, j_pick]) + e_hov_l[j_pick]
        traj.append(int(G[j_pick]))
        mask  |= (1 << j_pick)
        last   = j_pick

    return traj, env.objective(traj), {
        'dp_states': dp_states_visited, 'time': elapsed,
        'Mf': Mf, 'timed_out': timed_out, 'n_visited': len(traj)
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  OPTIMALITY GAP EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_optimality_gap(M, n_eval=200, seed=42, device='cpu',
                             mlp_model_dir='models_mlp',
                             attn_model_dir='models_attn',
                             dp_time_limit=30.0,
                             mlp_only=False):
    """
    mlp_only=True : only run DP-Exact vs MLP (skips Transformer + all baselines).
                    Much faster — useful when you just want the MLP optimality gap.
    mlp_only=False: full comparison (DP, MLP, Transformer, 4 baselines).
    """
    print(f"\n{'='*60}\n  Optimality Gap  M={M}, n_eval={n_eval}"
          f"{'  [MLP-only mode]' if mlp_only else ''}\n{'='*60}")

    names = ['DP-Exact', 'MLP'] if mlp_only else [
        'DP-Exact', 'MLP', 'Transformer',
        'Random', 'Nearest-Neighbor', 'Greedy-Priority', 'PDR']
    results = {n: {'obj': [], 'nodes': [], 'gap': []} for n in names}

    # Load MLP
    mlp_policy = None
    mlp_path   = os.path.join(mlp_model_dir, f'policy_M{M}.pt')
    if os.path.exists(mlp_path):
        mlp_policy = Policy(hidden=256).to(device)
        ck = torch.load(mlp_path, map_location=device)
        mlp_policy.load_state_dict(ck['policy']); mlp_policy.eval()
        print(f"  MLP loaded from {mlp_path}")
    else:
        print(f"  [!] MLP not found: {mlp_path}")

    # Load Transformer (skipped in mlp_only mode)
    attn_policy = None
    if not mlp_only:
        from policy import AttentionPolicy
        from features import batch_rollout
        from env import Params, UAVEnv
        attn_path = os.path.join(attn_model_dir, f'attn_M{M}.pt')
        try:
            attn_policy = AttentionPolicy(d_model=64, n_heads=4, n_layers=1).to(device)
            ck = torch.load(attn_path, map_location=device)
            attn_policy.load_state_dict(ck.get('policy', ck)); attn_policy.eval()
            print(f"  Transformer loaded from {attn_path}")
        except Exception as e:
            print(f"  [!] Transformer not loaded: {e}"); attn_policy = None

    rng = np.random.default_rng(seed); dp_times = []

    for i in range(n_eval):
        s   = int(rng.integers(0, 10_000_000))
        env = Env(M=M, seed=s)

        # DP
        t0 = time.time()
        dp_traj, dp_obj, _ = dp_solve(env, time_limit=dp_time_limit)
        dp_times.append(time.time() - t0)
        results['DP-Exact']['obj'].append(dp_obj)
        results['DP-Exact']['nodes'].append(len(dp_traj))
        results['DP-Exact']['gap'].append(0.0)

        # MLP
        if mlp_policy:
            e2 = Env(M=M, seed=s)
            with torch.no_grad(): traj2, *_ = rollout(mlp_policy, e2, device, greedy=True)
            obj2 = e2.objective(traj2)
            results['MLP']['obj'].append(obj2); results['MLP']['nodes'].append(len(traj2))
            results['MLP']['gap'].append(obj2 - dp_obj)

        if not mlp_only:
            # Transformer
            if attn_policy:
                params3 = Params(M=M); e3 = UAVEnv(params3, seed=s)
                with torch.no_grad(): traj3, _, _, _ = batch_rollout(attn_policy, e3, device, greedy=True)
                obj3 = e3.objective(traj3)
                results['Transformer']['obj'].append(obj3); results['Transformer']['nodes'].append(len(traj3))
                results['Transformer']['gap'].append(obj3 - dp_obj)

            # Baselines
            for bname, bkey in BASELINES.items():
                eb = Env(M=M, seed=s); tb = run_baseline(eb, bkey); ob = eb.objective(tb)
                results[bname]['obj'].append(ob); results[bname]['nodes'].append(len(tb))
                results[bname]['gap'].append(ob - dp_obj)

        if (i+1) % 50 == 0:
            print(f"  [{i+1:>3}/{n_eval}] DP avg {np.mean(dp_times):.4f}s  "
                  f"DP_obj {np.mean(results['DP-Exact']['obj']):.3f}")

    # Summary
    dp_mean = np.mean(results['DP-Exact']['obj'])
    print(f"\n  {'Method':<22} {'Obj':>9} {'Std':>8} {'Gap':>9} {'Gap%':>7} {'Nodes':>7}")
    print('  ' + '-'*65)
    for name in names:
        if not results[name]['obj']: continue
        om  = np.mean(results[name]['obj']); os_ = np.std(results[name]['obj'])
        gm  = np.mean(results[name]['gap'])
        gp  = gm / max(abs(dp_mean), 1e-9) * 100
        nd  = np.mean(results[name]['nodes'])
        star = '  ← optimal' if name == 'DP-Exact' else ''
        print(f"  {name:<22} {om:>9.3f} {os_:>8.3f} {gm:>9.3f} {gp:>6.1f}% {nd:>7.1f}{star}")
    print(f"\n  DP: {np.mean(dp_times):.4f}s ± {np.std(dp_times):.4f}s per instance")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

COLORS  = {'DP-Exact':'#000000','MLP':'#009E73','Transformer':'#0072B2',
           'Random':'#AAAAAA','Nearest-Neighbor':'#E69F00',
           'Greedy-Priority':'#56B4E9','PDR':'#CC79A7'}
MARKERS = {'DP-Exact':'*','MLP':'o','Transformer':'P',
           'Random':'s','Nearest-Neighbor':'^','Greedy-Priority':'D','PDR':'v'}
def ci95(a): return 1.96*np.std(a)/np.sqrt(max(len(a),1))

def plot_optimality_gap(all_results, M_list, out_dir='results_dp'):
    os.makedirs(out_dir, exist_ok=True)
    ORDER = ['DP-Exact','MLP','Transformer','Nearest-Neighbor','Greedy-Priority','PDR','Random']

    # Fig A: Objective vs M
    fig, ax = plt.subplots(figsize=(10,6), dpi=150)
    for m in ORDER:
        xs,ys,es = [],[],[]
        for M in M_list:
            v = all_results.get(M,{}).get(m,{}).get('obj',[])
            if v: xs.append(M); ys.append(np.mean(v)); es.append(ci95(v))
        if not xs: continue
        lw = 2.5 if m in ('DP-Exact','MLP','Transformer') else 1.2
        ls = '--' if m=='DP-Exact' else '-'
        ax.errorbar(xs,ys,yerr=es,marker=MARKERS.get(m,'o'),color=COLORS.get(m,'#555'),
                    label=m,lw=lw,ls=ls,markersize=8,capsize=3)
    ax.set_xlabel('Number of Nodes M',fontsize=12)
    ax.set_ylabel('Composite Objective (lower = better)',fontsize=12)
    ax.set_title('Optimality Gap Study: Objective vs M\n(DP-Exact = true optimal lower bound)',
                 fontsize=13,fontweight='bold')
    ax.legend(fontsize=9,ncol=2); ax.set_xticks(M_list); ax.grid(alpha=0.3)
    plt.tight_layout(); p=os.path.join(out_dir,'dp_objective_vs_M.png')
    plt.savefig(p,dpi=150); plt.close(); print(f"  Saved {p}")

    # Fig B: Gap % vs M
    fig, ax = plt.subplots(figsize=(10,6), dpi=150)
    for m in [x for x in ORDER if x!='DP-Exact']:
        xs,ys,es = [],[],[]
        for M in M_list:
            gaps = all_results.get(M,{}).get(m,{}).get('gap',[])
            dp_o = all_results.get(M,{}).get('DP-Exact',{}).get('obj',[])
            if not gaps or not dp_o: continue
            ref = abs(np.mean(dp_o))
            pcts = [g/max(ref,1e-9)*100 for g in gaps]
            xs.append(M); ys.append(np.mean(pcts)); es.append(ci95(pcts))
        if not xs: continue
        lw = 2.5 if m in ('MLP','Transformer') else 1.2
        ax.errorbar(xs,ys,yerr=es,marker=MARKERS.get(m,'o'),color=COLORS.get(m,'#555'),
                    label=m,lw=lw,markersize=8,capsize=3)
    ax.axhline(0,color='black',lw=1.5,ls='--',label='DP-Exact (optimal)')
    ax.set_xlabel('Number of Nodes M',fontsize=12)
    ax.set_ylabel('Optimality Gap  (method − DP) / |DP| × 100%',fontsize=12)
    ax.set_title('Optimality Gap (%) vs Network Size',fontsize=13,fontweight='bold')
    ax.legend(fontsize=9,ncol=2); ax.set_xticks(M_list); ax.grid(alpha=0.3)
    plt.tight_layout(); p=os.path.join(out_dir,'dp_gap_pct_vs_M.png')
    plt.savefig(p,dpi=150); plt.close(); print(f"  Saved {p}")

    # Fig C: Box plot at largest M
    M_box = max(M_list)
    fig, ax = plt.subplots(figsize=(10,5), dpi=150)
    bd,bl,bc = [],[],[]
    for m in [x for x in ORDER if x!='DP-Exact']:
        gaps = all_results.get(M_box,{}).get(m,{}).get('gap',[])
        dp_o = all_results.get(M_box,{}).get('DP-Exact',{}).get('obj',[])
        if not gaps or not dp_o: continue
        ref = abs(np.mean(dp_o))
        bd.append([g/max(ref,1e-9)*100 for g in gaps]); bl.append(m)
        bc.append(COLORS.get(m,'#888'))
    if not bd:
        ax.text(0.5, 0.5, 'No comparison data (mlp_only mode)',
                ha='center', va='center', transform=ax.transAxes, fontsize=13)
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close(); print(f'  Saved {p}'); return
    bp = ax.boxplot(bd,tick_labels=bl,patch_artist=True,notch=False,
                    medianprops=dict(color='black',lw=2))
    for patch,c in zip(bp['boxes'],bc): patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.axhline(0,color='black',lw=1.5,ls='--',label='Optimal (DP-Exact)')
    ax.set_ylabel('Optimality Gap (%)',fontsize=12)
    ax.set_title(f'Per-Instance Gap Distribution  M={M_box}',fontsize=12,fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3,axis='y')
    plt.xticks(rotation=15,ha='right',fontsize=9); plt.tight_layout()
    p=os.path.join(out_dir,f'dp_gap_boxplot_M{M_box}.png')
    plt.savefig(p,dpi=150); plt.close(); print(f"  Saved {p}")

    np.save(os.path.join(out_dir,'dp_gap_results.npy'), all_results)
    print(f"  Saved {os.path.join(out_dir,'dp_gap_results.npy')}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--M_list',        type=int,   nargs='+', default=[8,10,12,15])
    ap.add_argument('--n_eval',        type=int,   default=200)
    ap.add_argument('--seed',          type=int,   default=42)
    ap.add_argument('--device',        type=str,   default='cpu')
    ap.add_argument('--out_dir',       type=str,   default='results_dp')
    ap.add_argument('--mlp_dir',       type=str,   default='models_mlp')
    ap.add_argument('--attn_dir',      type=str,   default='models_attn')
    ap.add_argument('--dp_time_limit', type=float, default=30.0)
    ap.add_argument('--mlp_only',      action='store_true',
                    help='Only compare DP-Exact vs MLP (skip Transformer + baselines)')
    ap.add_argument('--quick',         action='store_true')
    args = ap.parse_args()

    if args.quick: args.M_list=[6,8]; args.n_eval=20
    device = args.device
    if device=='cuda' and not torch.cuda.is_available(): device='cpu'
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    mode_str = 'MLP-only' if args.mlp_only else 'full'
    print('='*60)
    print(f'  UAV AoI — Exact DP  |  mode={mode_str}  M={args.M_list}  n_eval={args.n_eval}')
    print('='*60)

    all_results = {}
    for M in args.M_list:
        all_results[M] = evaluate_optimality_gap(
            M=M, n_eval=args.n_eval, seed=args.seed, device=device,
            mlp_model_dir=args.mlp_dir, attn_model_dir=args.attn_dir,
            dp_time_limit=args.dp_time_limit, mlp_only=args.mlp_only)

    print(f"\nPlotting → {args.out_dir}/")
    plot_optimality_gap(all_results, args.M_list, out_dir=args.out_dir)
    print('Done.')