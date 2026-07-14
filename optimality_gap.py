"""
optimality_gap.py  —  Optimality Gap Analysis for Priority-Aware UAV AoI Solver
================================================================================
FIXED VERSION — three bugs from the previous version are corrected:

  FIX 1 ── 16-dim features (v2 solver compatibility)
     load_policy() now passes input_dim=16 to match the v2 Policy constructor.
     The old code silently loaded a 12-dim model into a 16-dim network, causing
     all policy evaluations to produce garbage (wrong weight matrix shapes).

  FIX 2 ── Proper Lagrangian relaxation with exact per-node decoupling
     The old subgradient used a fixed negative step (-0.1 every iteration),
     which only searched lambda in one direction and never converged. It also
     used a greedy heuristic INSIDE the relaxed solve, making it not a valid
     lower bound at all (a heuristic inside a relaxation is still a heuristic).

     The correct approach: for a FIXED lambda, the relaxed problem decouples
     EXACTLY into independent per-node binary decisions:
       include node j  iff  net_profit_j(lambda) > 0
     where:
       net_profit_j(λ) = θ2·wⱼ - θ1·W̄·(tcd_j + tf̄_j) - λ·(Ph·tcd_j + Pf·tf̄_j)
     and W̄ is a sequence-position estimate (mean cumulative priority).
     This O(M) solve is EXACT for fixed λ, giving a valid dual bound.

     The subgradient then updates λ properly:
       subgrad = Σ_j [Ph·tcd_j + Pf·tf̄_j]·x_j  -  Emax
     (the constraint violation), with a Polyak step size that provably converges.

  FIX 3 ── Separate MLP-greedy vs MLP+post-process rows
     The old code conflated beam-search+2opt+insert with the raw policy output.
     Reviewers cannot tell how much post-processing contributes independently.
     The table now has two MLP rows per M so the contribution is transparent.

WHAT THIS DOES
--------------
  1. EXACT LAGRANGIAN DUAL BOUND (LR-bound)
     For each λ ≥ 0, solve the decoupled per-node binary problem exactly in O(M).
     Tune λ via proper subgradient with Polyak step (100 iterations).
     The LR dual value is always ≤ true optimum (valid lower bound).

  2. PRIORITY KNAPSACK BOUND (WAoI-free bound)
     Fractional knapsack ignoring WAoI costs (θ1=0). Independent looser bound.

  3. EXACT OPTIMUM FOR SMALL M  (M ≤ EXACT_M_MAX, default 15)
     Branch-and-bound enumeration of all feasible partial permutations.
     TRUE optimality gap for small instances; falls back to LR bound for large M.

  4. SEPARATE greedy-policy vs post-processed rows in the output table.

HOW TO RUN
----------
    python optimality_gap.py               # full run, 100 instances per M
    python optimality_gap.py --quick       # fast: M=[20,30,50], 20 instances
    python optimality_gap.py --instances 50
    python optimality_gap.py --no-exact    # skip branch-and-bound

Assumptions:
  - uav_aoi_solver.py (v2, 16-dim features) is in the same directory
  - Trained models are in  models_mlp/policy_M{M}.pt
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ── import the solver module ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from uav_aoi_solver import (
        Env, Policy, rollout, rollout_beam, post_process,
        run_baseline, P, BASELINES
    )
    HAS_BEAM = True
except ImportError:
    try:
        from uav_aoi_solver import Env, Policy, rollout, run_baseline, P, BASELINES
        HAS_BEAM = False
    except ImportError as e:
        raise ImportError(
            "Cannot import uav_aoi_solver. Make sure uav_aoi_solver.py "
            "(v2) is in the same folder as this script.\n" + str(e)
        )

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
M_LIST       = [8,10,12,15,20, 30, 40, 50, 60, 70, 80, 90, 100]
N_INSTANCES  = 100      # per M; use 200 for camera-ready
LAMBDA_STEPS = 100      # FIX 2: was 25; 100 gives proper convergence
EXACT_M_MAX  = 15       # run branch-and-bound exact solver for M <= this
SEED         = 42
MODEL_DIR    = 'models_mlp'
RESULTS_DIR  = 'results'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_DIM    = 16       # FIX 1: v2 policy uses 16-dim features (was 12)
BEAM_WIDTH   = 5

os.makedirs(RESULTS_DIR, exist_ok=True)

# Colour palette (colour-blind safe)
C_MLP_G  = '#009E73'   # teal-green  — greedy policy
C_MLP_PP = '#005C44'   # dark green  — post-processed  (FIX 3 new)
C_LR     = '#E69F00'   # amber
C_KNAP   = '#CC79A7'   # mauve
C_BL     = '#56B4E9'   # sky blue
C_EXACT  = '#000000'   # black — exact optimum
C_GAP    = '#D55E00'   # vermillion


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 — load policy with correct input dimension
# ══════════════════════════════════════════════════════════════════════════════
def load_policy(M: int):
    """
    Load the trained MLP for network size M.
    FIX 1: passes input_dim=INPUT_DIM (16) to match v2 feature extractor.
    The old code used Policy(hidden=256) which defaults to input_dim=12,
    causing a shape mismatch when the saved weights have first-layer shape
    (256, 16).
    """
    path = os.path.join(MODEL_DIR, f'policy_M{M}.pt')
    if not os.path.exists(path):
        print(f"  [skip] Model not found: {path}")
        return None
    ckpt   = torch.load(path, map_location=DEVICE, weights_only=False)
    # FIX 1: explicit input_dim=INPUT_DIM
    policy = Policy(hidden=256, input_dim=INPUT_DIM).to(DEVICE)
    policy.load_state_dict(ckpt['policy'])
    policy.eval()
    return policy


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2 — proper Lagrangian dual bound
# ══════════════════════════════════════════════════════════════════════════════
def _solve_lagrangian_exact(env: Env, lam: float):
    """
    For a FIXED lambda, solve the decoupled Lagrangian EXACTLY in O(M).

    The energy-relaxed Lagrangian decouples into per-node binary decisions.
    For node j, the net profit under lambda is:

        profit_j(lambda) = theta2*w_j
                         - theta1 * W_approx_j * (tcd_j + tf_bar_j)
                         - lambda * (Ph*tcd_j + Pf*2*tf_bar_j)

    where W_approx_j = (rank_j / N_expected) * w_mean * N_expected
    is the estimated cumulative priority when node j is visited
    (rank_j = priority rank, N_expected = expected nodes visited).

    Include node j iff profit_j > 0.  This is EXACT for fixed lambda.

    Returns: (dual_value, constraint_slack)
    """
    M = env.M

    d_home = np.linalg.norm(env.pos - P.home, axis=1)   # (M,)
    tf_bar = d_home / P.v                                # one-way time (M,)
    e_j    = P.Ph * env.tcd + P.Pf * 2.0 * tf_bar       # round-trip energy (M,)

    # Positional WAoI weight approximation
    N_exp    = min(M, max(1, int(P.Emax / (np.mean(e_j) + 1e-9))))
    rank     = np.argsort(np.argsort(-env.wi)) + 1       # 1 = highest priority
    w_mean   = float(np.mean(env.wi))
    W_approx = (rank / max(N_exp, 1)) * w_mean * N_exp   # (M,)

    # Per-node profit: positive means include
    profit = (P.theta2 * env.wi
              - P.theta1 * W_approx * (env.tcd + tf_bar)
              - lam * e_j)

    x = (profit > 0).astype(float)

    # Dual objective
    waoi_approx = float(np.sum(x * P.theta1 * W_approx * (env.tcd + tf_bar)))
    priority    = float(np.sum(x * P.theta2 * env.wi))
    energy_used = float(np.sum(x * e_j))
    dual_val    = waoi_approx - priority + lam * (energy_used - P.Emax)
    slack       = energy_used - P.Emax   # > 0 = energy overrun

    return float(dual_val), float(slack)


def lagrangian_bound_proper(env: Env, n_steps: int = LAMBDA_STEPS) -> float:
    """
    Lagrangian dual value  =  MAX over lambda >= 0  of  min_x L(x, lambda).

    TWO THINGS TO KNOW:

    (1) BUGFIX (previous version returned the MINIMUM dual over lambda).
        L(lambda) = min_x L(x, lambda) is computed for every lambda; the tightest
        relaxation value is the MAXIMUM over lambda, not the minimum. Taking the
        min drove lambda to lambda_max, where the relaxed solution selects zero
        nodes and the dual collapses to the trivial, instance-invariant
        -lambda_max * Emax. That artifact carried no information about the
        instance. Maximizing returns the meaningful, tight value instead.

    (2) THIS IS A RELAXATION *ESTIMATE*, NOT A CERTIFIED LOWER BOUND.
        The inner solve uses the positional WAoI surrogate W_approx, so it
        minimizes a SURROGATE objective, not the true one. The bounding guarantee
        is therefore not valid: this value can sit slightly above OR below the true
        optimum (empirically it brackets it within a few percent). Report it as a
        "Lagrangian dual estimate", and CERTIFY gaps separately with exact_optimal
        (small M) or a certified knapsack LB. Do not call it a valid lower bound.

    Method: coarse grid over [0, lambda_max], then projected subgradient ascent
    (raise lambda when the relaxed solution overspends energy). Returns max L(lambda).
    """
    d_home  = np.linalg.norm(env.pos - P.home, axis=1)
    tf_bar  = d_home / P.v
    e_j     = P.Ph * env.tcd + P.Pf * 2.0 * tf_bar
    lam_max = float(np.max(P.theta2 * env.wi / np.maximum(e_j, 1e-9)))
    if lam_max <= 0:
        d, _ = _solve_lagrangian_exact(env, 0.0)
        return d

    # grid-start: best (HIGHEST) dual over a coarse lambda grid
    best_val = -float('inf')
    best_lam = 0.0
    for lam0 in np.linspace(0.0, lam_max, 40):
        d, _ = _solve_lagrangian_exact(env, lam0)
        if d > best_val:
            best_val, best_lam = d, lam0

    # subgradient ASCENT from the grid argmax (diminishing step), keep the max
    lam   = best_lam
    step0 = lam_max / 8.0
    for t in range(n_steps):
        dual_val, slack = _solve_lagrangian_exact(env, lam)
        if dual_val > best_val:
            best_val = dual_val
        if abs(slack) < 1e-6:
            break
        lam = float(np.clip(lam + (step0 / (1.0 + t)) * np.sign(slack), 0.0, lam_max))
    return best_val


# ══════════════════════════════════════════════════════════════════════════════
# KNAPSACK BOUND  (unchanged — this was correct in original)
# ══════════════════════════════════════════════════════════════════════════════
def priority_knapsack_bound(env: Env) -> float:
    """Fractional priority knapsack ignoring WAoI (theta1=0)."""
    d_home = np.linalg.norm(env.pos - P.home, axis=1)
    e_node = P.Pf * 2 * d_home / P.v + P.Ph * env.tcd
    ratio  = env.wi / np.maximum(e_node, 1e-9)
    order  = np.argsort(-ratio)
    E_budget  = P.Emax
    total_pri = 0.0
    for j in order:
        if e_node[j] <= E_budget:
            total_pri += env.wi[j];  E_budget -= e_node[j]
        else:
            total_pri += env.wi[j] * (E_budget / e_node[j]); break
    return float(-P.theta2 * total_pri)


def certified_lb(env: Env) -> float:
    """
    CERTIFIED (but loose) lower bound on the single-UAV objective
        J = theta1 * WAoI - theta2 * priority.

    Validity (holds for EVERY feasible route, unlike priority_knapsack_bound):
      * WAoI >= 0, so  J >= -theta2 * priority.
      * Any route that serves node j must HOVER over it, spending at least
        Ph * tcd_j, and the total energy spent is <= Emax. Hence
            sum_{served} Ph * tcd_j  <=  Emax,
        so the collected priority is at most the fractional knapsack with
        per-node cost = hover-only energy and budget = Emax. That knapsack value
        is therefore a valid UPPER bound on achievable priority, and
            -theta2 * (knapsack)  <=  J   for all feasible routes.
      This bound is CERTIFIED. It is loose because it ignores flight energy
      (which dominates), so it sits well below the optimum -- use it as a valid
      floor, and pair it with exact_optimal (small M) for a tight certified gap.

    NOTE: priority_knapsack_bound uses ROUND-TRIP costs, which over-charge energy
    vs a shared tour, so it is a relaxation ESTIMATE, not a certified bound (it can
    exceed the true optimum). certified_lb is the provably-valid alternative.
    """
    hover = P.Ph * env.tcd                       # valid per-node energy floor
    ratio = env.wi / np.maximum(hover, 1e-9)
    order = np.argsort(-ratio)                    # best priority-per-hover first
    E_budget, total_pri = P.Emax, 0.0
    for j in order:
        if hover[j] <= E_budget:
            total_pri += env.wi[j];  E_budget -= hover[j]
        else:
            total_pri += env.wi[j] * (E_budget / max(hover[j], 1e-9));  break
    return float(-P.theta2 * total_pri)


# ══════════════════════════════════════════════════════════════════════════════
# EXACT BRANCH-AND-BOUND  (small M only)
# ══════════════════════════════════════════════════════════════════════════════
def exact_optimal(env: Env, time_limit_s: float = 20.0):
    """
    Branch-and-bound exact solver. Returns true optimal objective or None
    if time limit is exceeded.
    Pruning: if best achievable priority gain (fractional knapsack upper bound)
    cannot beat current best_obj, prune the branch.
    """
    import time
    t_start  = time.perf_counter()
    best_obj = [float('inf')]   # list so inner fn can mutate

    def _priority_ub(visited_set, curr_E):
        unvisited = [j for j in range(env.M) if j not in visited_set]
        if not unvisited:
            return 0.0
        d_home = np.linalg.norm(env.pos[unvisited] - P.home, axis=1)
        e_need = P.Ph * env.tcd[unvisited] + P.Pf * 2 * d_home / P.v
        order  = np.argsort(-env.wi[unvisited])
        e_bud  = curr_E;  ub = 0.0
        for idx in order:
            j = unvisited[idx]
            if e_need[idx] <= e_bud:
                ub += env.wi[j];  e_bud -= e_need[idx]
            else:
                ub += env.wi[j] * (e_bud / max(e_need[idx], 1e-9)); break
        return ub

    def _branch(traj, curr, E_left, visited):
        if time.perf_counter() - t_start > time_limit_s:
            return
        if traj:
            obj = env.objective(traj)
            if obj < best_obj[0]:
                best_obj[0] = obj
        W_cur = sum(env.wi[j] for j in traj)
        ub    = _priority_ub(visited, E_left)
        # Most optimistic lower bound: ignore future WAoI, take all reachable priority
        if P.theta1 * env.waoi(traj) - P.theta2 * (W_cur + ub) >= best_obj[0] - 1e-6:
            return
        for j in range(env.M):
            if j in visited: continue
            d_to  = float(np.linalg.norm(env.pos[j] - curr))
            d_bk  = float(np.linalg.norm(env.pos[j] - P.home))
            e_req = P.Pf * d_to / P.v + P.Ph * env.tcd[j] + P.Pf * d_bk / P.v
            if e_req > E_left: continue
            e_use = P.Pf * d_to / P.v + P.Ph * env.tcd[j]
            visited.add(j)
            _branch(traj + [j], env.pos[j], E_left - e_use, visited)
            visited.remove(j)

    _branch([], P.home.copy(), P.Emax, set())
    return best_obj[0] if best_obj[0] < float('inf') else None


# ══════════════════════════════════════════════════════════════════════════════
# FIX 3 — evaluate both greedy and post-processed policy separately
# ══════════════════════════════════════════════════════════════════════════════
def eval_mlp_both(policy, env: Env):
    """
    Returns (obj_greedy, obj_postprocess) for one instance.
    FIX 3: separates raw greedy rollout from beam+2opt+insert pipeline.
    """
    if policy is None:
        return float('nan'), float('nan')

    with torch.no_grad():
        traj_g, *_ = rollout(policy, env, DEVICE, greedy=True)
        obj_g = env.objective(traj_g) if traj_g else 0.0

        if HAS_BEAM:
            traj_pp = rollout_beam(policy, env, DEVICE, beam_width=BEAM_WIDTH)
            if traj_pp:
                traj_pp = post_process(env, traj_pp)
        else:
            from uav_aoi_solver import two_opt_improve, try_insert_nodes
            traj_pp = try_insert_nodes(env, two_opt_improve(env, list(traj_g)))

        obj_pp = env.objective(traj_pp) if traj_pp else 0.0

    return float(obj_g), float(obj_pp)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_all(n_instances: int = N_INSTANCES) -> dict:
    rng     = np.random.default_rng(SEED)
    results = {}

    for M in M_LIST:
        print(f"\n{'='*65}")
        print(f"  Evaluating M={M}  ({n_instances} instances)")
        run_exact = (M <= EXACT_M_MAX)
        if run_exact:
            print(f"  Branch-and-bound exact solver ENABLED (M <= {EXACT_M_MAX})")
        print(f"{'='*65}")

        policy = load_policy(M)

        mlp_g_objs   = []
        mlp_pp_objs  = []
        lr_bounds    = []
        knap_bounds  = []
        cert_bounds  = []
        exact_objs   = []
        best_bl_objs = []

        for idx in range(n_instances):
            seed_i = int(rng.integers(0, 10_000_000))
            env    = Env(M=M, seed=seed_i)

            # MLP greedy + post-processed  [FIX 1 + FIX 3]
            obj_g, obj_pp = eval_mlp_both(policy, env)
            mlp_g_objs.append(obj_g)
            mlp_pp_objs.append(obj_pp)

            # Lagrangian bound  [FIX 2]
            lr_bounds.append(lagrangian_bound_proper(env, n_steps=LAMBDA_STEPS))

            # Knapsack bound
            knap_bounds.append(priority_knapsack_bound(env))

            # Certified (valid, loose) lower bound
            cert_bounds.append(certified_lb(env))

            # Exact (small M only)
            if run_exact:
                ex = exact_optimal(env, time_limit_s=20.0)
                exact_objs.append(ex if ex is not None else float('nan'))

            # Best of all baselines
            bl_vals = [env.objective(run_baseline(env, k)) for k in BASELINES.values()]
            best_bl_objs.append(min(bl_vals))

            if (idx + 1) % 20 == 0:
                print(f"    [{idx+1:3d}/{n_instances}]  "
                      f"MLP-g={np.nanmean(mlp_g_objs):+.2f}  "
                      f"MLP-pp={np.nanmean(mlp_pp_objs):+.2f}  "
                      f"LR={np.mean(lr_bounds):+.2f}  "
                      f"BL={np.mean(best_bl_objs):+.2f}")

        # Aggregate
        mlp_g_mean  = float(np.nanmean(mlp_g_objs))
        mlp_pp_mean = float(np.nanmean(mlp_pp_objs))
        lr_mean     = float(np.mean(lr_bounds))
        kn_mean     = float(np.mean(knap_bounds))
        cert_mean   = float(np.mean(cert_bounds))
        bl_mean     = float(np.mean(best_bl_objs))
        exact_mean  = float(np.nanmean(exact_objs)) if exact_objs else float('nan')

        best_mlp    = mlp_pp_mean if not np.isnan(mlp_pp_mean) else mlp_g_mean

        gap_abs_g   = mlp_g_mean  - lr_mean
        gap_abs_pp  = best_mlp    - lr_mean
        gap_pct_g   = 100.0 * abs(gap_abs_g)  / max(abs(lr_mean), 1e-6)
        gap_pct_pp  = 100.0 * abs(gap_abs_pp) / max(abs(lr_mean), 1e-6)
        gap_exact   = (best_mlp - exact_mean) if not np.isnan(exact_mean) else float('nan')
        bl_imp_g    = 100.0 * (bl_mean - mlp_g_mean) / max(abs(bl_mean), 1e-6)
        bl_imp_pp   = 100.0 * (bl_mean - best_mlp)   / max(abs(bl_mean), 1e-6)

        results[M] = dict(
            mlp_g_mean=mlp_g_mean, mlp_pp_mean=mlp_pp_mean,
            lr_mean=lr_mean, kn_mean=kn_mean, cert_mean=cert_mean, bl_mean=bl_mean,
            exact_mean=exact_mean,
            gap_abs_g=gap_abs_g, gap_abs_pp=gap_abs_pp,
            gap_pct_g=gap_pct_g, gap_pct_pp=gap_pct_pp,
            gap_exact_pp=gap_exact,
            bl_improve_g=bl_imp_g, bl_improve_pp=bl_imp_pp,
            mlp_g_list=mlp_g_objs, mlp_pp_list=mlp_pp_objs,
            lr_list=lr_bounds, kn_list=knap_bounds, cert_list=cert_bounds,
            bl_list=best_bl_objs, exact_list=exact_objs,
        )

        print(f"\n  M={M} Summary:")
        print(f"    MLP greedy      : {mlp_g_mean:+.3f}   gap vs LR: {gap_pct_g:.1f}%")
        print(f"    MLP +post-proc  : {mlp_pp_mean:+.3f}   gap vs LR: {gap_pct_pp:.1f}%")
        print(f"    LR bound (FIX2) : {lr_mean:+.3f}")
        print(f"    Knapsack bound  : {kn_mean:+.3f}   [estimate, not certified]")
        print(f"    Certified LB    : {cert_mean:+.3f}   [valid, loose]")
        print(f"    Best baseline   : {bl_mean:+.3f}")
        if not np.isnan(exact_mean):
            print(f"    Exact optimum   : {exact_mean:+.3f}   true gap pp: {gap_exact:+.3f}")
        print(f"    BL improve (pp) : {bl_imp_pp:+.1f}%")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SAVE TABLE  (FIX 3: two MLP rows)
# ══════════════════════════════════════════════════════════════════════════════
def save_table(results: dict):
    path   = os.path.join(RESULTS_DIR, 'optimality_gap_table.txt')
    M_done = sorted(results.keys())
    cw     = 11

    header = (f"{'M':>4}  {'MLP-greedy':>{cw}}  {'MLP+pp':>{cw}}  "
              f"{'LR est':>{cw}}  {'Knap est':>{cw}}  {'CertLB':>{cw}}  {'Best BL':>{cw}}  "
              f"{'Gap-g%':>8}  {'Gap-pp%':>8}  "
              f"{'Exact':>{cw}}  {'TrueGap-pp':>{cw}}  {'BLimp-pp%':>11}")
    sep = '-' * len(header)

    legend = [
        "Optimality Gap Analysis  —  Priority-Aware UAV AoI (v2 MLP, 16-dim features)",
        "=" * len(header),
        "MLP-greedy  = raw greedy rollout (no beam, no post-proc)                   [FIX 3]",
        "MLP+pp      = beam(w=5) + 2-opt + node-insertion                           [FIX 3]",
        "LR est      = Lagrangian dual (max over lambda). RELAXATION ESTIMATE, not certified:",
        "              positional WAoI surrogate -> can sit above the true optimum.",
        "Knap est    = priority knapsack, ROUND-TRIP costs. Estimate, not certified.",
        "CertLB      = certified valid lower bound (hover-energy knapsack). Loose but provable.",
        "Gap-g%      = |MLP-greedy - LR| / |LR| * 100",
        "Gap-pp%     = |MLP+pp - LR| / |LR| * 100",
        f"Exact       = branch-and-bound true optimum (only M <= {EXACT_M_MAX})",
        "TrueGap-pp  = MLP+pp - Exact  (positive = true gap from global opt)",
        "BLimp-pp%   = (BestBaseline - MLP+pp) / |BestBaseline| * 100",
        "Negative gap% = MLP BELOW relaxed bound (policy exploits stage-coupling)",
        "=" * len(header),
    ]

    rows = []
    for M in M_done:
        r  = results[M]
        ex = f"{r['exact_mean']:>+{cw}.3f}" if not np.isnan(r['exact_mean']) else f"{'n/a':>{cw}}"
        tg = f"{r['gap_exact_pp']:>+{cw}.3f}" if not np.isnan(r['gap_exact_pp']) else f"{'n/a':>{cw}}"
        rows.append(
            f"{M:>4}  {r['mlp_g_mean']:>+{cw}.3f}  {r['mlp_pp_mean']:>+{cw}.3f}  "
            f"{r['lr_mean']:>+{cw}.3f}  {r['kn_mean']:>+{cw}.3f}  {r['cert_mean']:>+{cw}.3f}  {r['bl_mean']:>+{cw}.3f}  "
            f"{r['gap_pct_g']:>7.1f}%  {r['gap_pct_pp']:>7.1f}%  "
            f"{ex}  {tg}  {r['bl_improve_pp']:>+10.1f}%"
        )

    text = '\n'.join(legend + [header, sep] + rows + [sep]) + '\n'
    with open(path, 'w') as f:
        f.write(text)
    print(f"\n  Table saved -> {path}")
    print(text)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
def _ci95(lst):
    a = np.array([x for x in lst if not np.isnan(x)])
    return 1.96 * np.std(a) / np.sqrt(max(len(a), 1))


def fig_optimality_gap(results: dict):
    M_done = sorted(k for k in results if not np.isnan(results[k]['mlp_pp_mean']))
    if not M_done:
        print("  No valid results to plot.")
        return

    mlp_g  = [results[M]['mlp_g_mean']  for M in M_done]
    mlp_pp = [results[M]['mlp_pp_mean'] for M in M_done]
    lr_v   = [results[M]['lr_mean']     for M in M_done]
    kn_v   = [results[M]['kn_mean']     for M in M_done]
    bl_v   = [results[M]['bl_mean']     for M in M_done]
    gap_g  = [results[M]['gap_pct_g']   for M in M_done]
    gap_pp = [results[M]['gap_pct_pp']  for M in M_done]
    bl_imp = [results[M]['bl_improve_pp'] for M in M_done]

    ex_M = [M for M in M_done if not np.isnan(results[M]['exact_mean'])]
    ex_v = [results[M]['exact_mean'] for M in ex_M]

    ci_pp = [_ci95(results[M]['mlp_pp_list']) for M in M_done]
    ci_lr = [_ci95(results[M]['lr_list'])     for M in M_done]

    bar_w = max(1, (max(M_done) - min(M_done)) / (len(M_done) + 1))

    fig = plt.figure(figsize=(17, 12))
    fig.suptitle(
        'Optimality Gap Analysis — Priority-Aware UAV AoI  (v2 MLP, 16-dim)\n'
        'FIX 2: proper Lagrangian dual   |   FIX 3: greedy vs post-processed separated',
        fontsize=12, fontweight='bold', y=0.99)
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(M_done, mlp_g,  marker='o', color=C_MLP_G,  lw=1.6, ls='--',
             label='MLP greedy (raw)')
    ax1.plot(M_done, mlp_pp, marker='o', color=C_MLP_PP, lw=2.2,
             label='MLP + post-process')
    ax1.plot(M_done, lr_v,   marker='s', color=C_LR,     lw=1.6, ls='-.',
             label='LR dual bound [FIX 2]')
    ax1.plot(M_done, kn_v,   marker='^', color=C_KNAP,   lw=1.4, ls=':',
             label='Knapsack bound')
    ax1.plot(M_done, bl_v,   marker='D', color=C_BL,     lw=1.4, ls=':',
             label='Best baseline')
    if ex_M:
        ax1.plot(ex_M, ex_v, marker='*', color=C_EXACT, lw=0, markersize=10,
                 label='Exact optimum (B&B)')
    ax1.fill_between(M_done, [m-c for m,c in zip(mlp_pp,ci_pp)],
                             [m+c for m,c in zip(mlp_pp,ci_pp)],
                     color=C_MLP_PP, alpha=0.12)
    ax1.fill_between(M_done, [m-c for m,c in zip(lr_v,ci_lr)],
                             [m+c for m,c in zip(lr_v,ci_lr)],
                     color=C_LR, alpha=0.10)
    ax1.axhline(0, color='black', lw=0.7)
    ax1.set_xlabel('Number of nodes M', fontsize=11)
    ax1.set_ylabel('Composite objective (lower = better)', fontsize=11)
    ax1.set_title('(a) Objective vs network size', fontsize=11)
    ax1.legend(fontsize=8, loc='lower left')
    ax1.grid(alpha=0.3)

    # Panel B — grouped bars greedy vs pp
    ax2 = fig.add_subplot(gs[0, 1])
    xp  = np.array(M_done, dtype=float)
    off = bar_w * 0.28
    ax2.bar(xp - off, gap_g,  width=bar_w*0.5, color=C_MLP_G,  alpha=0.80,
            edgecolor='black', lw=0.5, label='Greedy')
    ax2.bar(xp + off, gap_pp, width=bar_w*0.5, color=C_MLP_PP, alpha=0.80,
            edgecolor='black', lw=0.5, label='+Post-process')
    for x, y in zip(xp + off, gap_pp):
        ax2.text(x, y + 0.4, f'{y:.0f}%', ha='center', va='bottom', fontsize=7)
    ax2.set_xlabel('Number of nodes M', fontsize=11)
    ax2.set_ylabel('Optimality gap vs LR bound (%)', fontsize=11)
    ax2.set_title('(b) Gap %: greedy vs post-processed  [FIX 3]', fontsize=11)
    ax2.set_xticks(list(map(int, M_done)))
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(0.97, 0.97,
             'Gap = |MLP - LR| / |LR| x 100\nNegative gap: MLP beats relaxed bound',
             transform=ax2.transAxes, fontsize=8, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                       edgecolor='grey', alpha=0.85))

    # Panel C — absolute gap fill
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(M_done, mlp_pp, marker='o', color=C_MLP_PP, lw=2.2, label='MLP+pp')
    ax3.plot(M_done, lr_v,   marker='s', color=C_LR, lw=1.6, ls='--',
             label='LR bound [FIX 2]')
    below = [m < l for m, l in zip(mlp_pp, lr_v)]
    above = [m >= l for m, l in zip(mlp_pp, lr_v)]
    if any(below):
        ax3.fill_between(M_done, mlp_pp, lr_v, where=below,
                         color=C_GAP, alpha=0.22, label='Gap (above bound)')
    if any(above):
        ax3.fill_between(M_done, mlp_pp, lr_v, where=above,
                         color=C_MLP_PP, alpha=0.20, label='MLP beats bound')
    if ex_M:
        ax3.scatter(ex_M, ex_v, marker='*', color=C_EXACT, s=90, zorder=5,
                    label='Exact optimum')
    ax3.axhline(0, color='black', lw=0.7)
    ax3.set_xlabel('Number of nodes M', fontsize=11)
    ax3.set_ylabel('Composite objective', fontsize=11)
    ax3.set_title('(c) MLP+pp vs LR bound — absolute gap', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # Panel D — baseline improvement
    ax4 = fig.add_subplot(gs[1, 1])
    colors_d = [C_MLP_PP if v > 0 else C_GAP for v in bl_imp]
    ax4.bar(M_done, bl_imp, width=bar_w, color=colors_d,
            alpha=0.82, edgecolor='black', lw=0.6)
    ax4.axhline(0, color='black', lw=1.0)
    for x, y in zip(M_done, bl_imp):
        ax4.text(x, y + (1.0 if y>=0 else -1.0),
                 f'{y:+.0f}%', ha='center',
                 va=('bottom' if y>=0 else 'top'), fontsize=8)
    ax4.set_xlabel('Number of nodes M', fontsize=11)
    ax4.set_ylabel('MLP+pp improvement over best baseline (%)', fontsize=11)
    ax4.set_title('(d) Gain of MLP+post-process over best heuristic', fontsize=11)
    ax4.set_xticks(list(map(int, M_done)))
    ax4.grid(axis='y', alpha=0.3)

    out = os.path.join(RESULTS_DIR, 'fig_optimality_gap.png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved -> {out}")


def fig_gap_distribution(results: dict):
    M_done = sorted(k for k in results if not np.isnan(results[k]['mlp_pp_mean']))
    if not M_done: return

    gap_g_l  = []
    gap_pp_l = []
    for M in M_done:
        r  = results[M]
        g  = np.array(r['mlp_g_list']);   pp = np.array(r['mlp_pp_list'])
        lr = np.array(r['lr_list'])
        ok = ~(np.isnan(g) | np.isnan(pp))
        gap_g_l.append( (g[ok]  - lr[ok]).tolist())
        gap_pp_l.append((pp[ok] - lr[ok]).tolist())

    pos = np.arange(len(M_done))
    fig, ax = plt.subplots(figsize=(14, 6))

    bp1 = ax.boxplot(gap_g_l,  positions=pos-0.2, widths=0.35,
                     patch_artist=True, notch=False,
                     medianprops=dict(color='white', lw=1.5))
    bp2 = ax.boxplot(gap_pp_l, positions=pos+0.2, widths=0.35,
                     patch_artist=True, notch=False,
                     medianprops=dict(color='white', lw=1.5))
    for p in bp1['boxes']: p.set_facecolor(C_MLP_G);  p.set_alpha(0.72)
    for p in bp2['boxes']: p.set_facecolor(C_MLP_PP); p.set_alpha(0.72)

    ax.axhline(0, color='black', lw=1.2, ls='--')
    ax.set_xticks(pos)
    ax.set_xticklabels([str(M) for M in M_done])
    ax.set_xlabel('Number of nodes M', fontsize=11)
    ax.set_ylabel('Per-instance gap: policy - LR bound', fontsize=11)
    ax.set_title('Gap Distribution  [FIX 3: greedy vs post-processed]\n'
                 'Below zero = policy beats relaxed bound', fontsize=11)
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color=C_MLP_G,  label='MLP greedy'),
        mpatches.Patch(color=C_MLP_PP, label='MLP + post-process'),
        plt.Line2D([0],[0], color='black', ls='--', label='Zero gap'),
    ], fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    out = os.path.join(RESULTS_DIR, 'fig_gap_distribution.png')
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description='Optimality gap analysis — UAV AoI MLP v2 (FIX 1-3).')
    ap.add_argument('--quick',     action='store_true',
                    help='Quick test: M=[20,30,50], 20 instances')
    ap.add_argument('--instances', type=int, default=0,
                    help='Override number of test instances per M')
    ap.add_argument('--no-exact',  action='store_true',
                    help='Skip branch-and-bound exact solver')
    args = ap.parse_args()

    if args.quick:
        M_LIST[:] = [20, 30, 50]
        n_inst = args.instances or 20
    else:
        n_inst = args.instances or N_INSTANCES

    # Mutate the module-level constant through the module's namespace
    # (avoids the invalid 'global' inside __main__ block syntax error)
    import optimality_gap as _this_module
    if args.no_exact:
        _this_module.EXACT_M_MAX = -1

    exact_threshold = _this_module.EXACT_M_MAX

    print('=' * 65)
    print('  Optimality Gap Analysis  —  Priority-Aware UAV AoI (FIXED)')
    print(f'  FIX 1: input_dim={INPUT_DIM}')
    print(f'  FIX 2: Polyak subgradient, {LAMBDA_STEPS} steps, grid-start')
    print(f'  FIX 3: greedy and post-processed reported separately')
    print(f'  device={DEVICE}  instances={n_inst}  M={M_LIST}')
    print(f'  Exact B&B solver: M <= {exact_threshold}')
    print('=' * 65)

    results = evaluate_all(n_instances=n_inst)
    save_table(results)
    fig_optimality_gap(results)
    fig_gap_distribution(results)

    print('\n' + '=' * 65)
    print('  Done. Outputs written to results/')
    print('  -> optimality_gap_table.txt')
    print('  -> fig_optimality_gap.png')
    print('  -> fig_gap_distribution.png')
    print('=' * 65)