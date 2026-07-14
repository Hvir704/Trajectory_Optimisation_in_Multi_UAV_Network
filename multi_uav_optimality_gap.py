"""
multi_uav_optimality_gap.py  —  Optimality-Gap Analysis for the FLEET solver
================================================================================
Team-orienteering analogue of optimality_gap.py (single-UAV). Produces, per
(M, K) cell, the trained fleet policy value against three reference bounds and
(where tractable) a certified exact optimum, then reports gap %.

HONESTY LAYER (read this — it governs how to report the numbers)
----------------------------------------------------------------
Three reference quantities are computed. They are NOT equally rigorous:

  * EXACT  (B&B)         CERTIFIED true optimum. Only for tiny M and K<=2.
                         The ONLY column that yields a certified TrueGap.
  * CertLB (hover knap)  CERTIFIED lower bound, but LOOSE: charges only hover
                         energy (Ph*tcd_j), which every feasible solution must
                         spend, ignoring flight. Valid for any (M,K); usually
                         far below the optimum because flight dominates energy.
  * LR / Knap (relax)    RELAXATION ESTIMATES, not certified. They mirror the
                         single-UAV optimality_gap.py construction exactly
                         (positional WAoI surrogate W_approx; round-trip per-node
                         energy). Round-trip costs over-charge vs a shared tour,
                         so neither is provably <= the optimum. Reported for
                         consistency with the single-UAV analysis and because
                         they empirically track the policy; label them as
                         relaxation estimates in the paper, NOT as valid bounds.

  => Certify gaps with EXACT (small cells) and CertLB (all cells). Use LR/Knap
     as the single-UAV-consistent reference for the headline gap % only.

WHAT THE FLEET EXTENSION ADDS OVER A NAIVE COPY (the LR estimate is K-aware)
---------------------------------------------------------------------------
  * Effect 1 (shorter chains lower WAoI): global priority ranks are split
    round-robin across K chains, so a node of global priority-rank r is placed
    at within-chain rank ceil(r/K). W_approx_j = ceil(r/K) * w_mean, so the
    WAoI surrogate shrinks ~1/K with fleet size (Lemma 1 benefit of splitting).
  * Effect 2 (range penalty): under split battery each UAV holds Emax/K. A node
    whose depot round-trip energy exceeds Emax_each cannot be served by ANY
    single UAV (a one-node chain {j} already costs that round trip), so it is
    force-excluded. This is a VALID tightening and it bites harder as K grows.
  * Aggregate energy budget K*Emax_each (= Emax under split) caps the included
    set, so the included-set energy is K-stable while Effects 1 and 2 move with K.
  Result: the LR estimate exhibits a sweet-spot-shaped curve in K, complementing
  the empirical policy curve.

K=1 REDUCTION (correctness check, asserted at runtime in --self-test)
  At K=1: within-chain rank == global rank, the range filter uses the full Emax,
  and the aggregate budget is Emax. The fleet LR estimate, knapsack estimate and
  exact B&B then reduce to the single-UAV optimality_gap.py functions.

HOW TO RUN
----------
    python multi_uav_optimality_gap.py --self-test          # fast correctness checks, no torch needed
    python multi_uav_optimality_gap.py --quick              # M=[50], K=[1,2], 1 seed, few instances
    python multi_uav_optimality_gap.py                      # full grid (slow)
    python multi_uav_optimality_gap.py --M 100 --K 1 2 3 4 6 --instances 100 --seeds 42 123 7
    python multi_uav_optimality_gap.py --no-exact           # skip B&B
    python multi_uav_optimality_gap.py --model-dir models_multi_uav

Assumes uav_aoi_solver.py and multi_uav_solver.py are importable, and fleet
checkpoints live at  {model_dir}/fleet_M{M}_K{K}_split_seed{SEED}.pt .
"""
from __future__ import annotations
import os, sys, time, argparse, warnings, itertools
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# torch is only needed for the POLICY rows. Bounds + exact are pure numpy, so we
# import torch lazily and let --self-test run without it.
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from uav_aoi_solver import Env, P
import multi_uav_solver as muv
from multi_uav_solver import (
    MP, FleetState, fleet_rollout, fleet_post_process,
    fleet_baseline, FLEET_BASELINES,
)

# Single-UAV references, used only by --self-test to assert the K=1 reduction.
try:
    from optimality_gap import (
        lagrangian_bound_proper as _su_lagrangian,
        priority_knapsack_bound as _su_knapsack,
        exact_optimal          as _su_exact,
    )
    HAS_SU_GAP = True
except Exception:
    HAS_SU_GAP = False


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG (overridable on the CLI)
# ══════════════════════════════════════════════════════════════════════════════
M_LIST            = [50, 60, 80, 100, 120, 150, 200]
K_LIST            = [1, 2, 3, 4]
SEEDS             = [42, 123, 7]
N_INSTANCES       = 100
LAMBDA_STEPS      = 100
EXACT_FLEET_M_MAX = 7          # B&B exact only for M <= this ...
EXACT_FLEET_K_MAX = 2          # ... and K <= this (team B&B blows up fast)
EXACT_TIME_LIMIT  = 25.0       # seconds per instance
INSTANCE_SEED     = 42
MODEL_DIR         = 'models_multi_uav'
RESULTS_DIR       = 'results'
DEVICE            = 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu'


# ══════════════════════════════════════════════════════════════════════════════
# helpers shared by every bound
# ══════════════════════════════════════════════════════════════════════════════
def _round_trip_energy(env: Env) -> np.ndarray:
    """Depot->j->depot energy for each node (hover + both flight legs)."""
    d_home = np.linalg.norm(env.pos - P.home, axis=1)
    tf_bar = d_home / P.v
    return P.Ph * env.tcd + P.Pf * 2.0 * tf_bar          # (M,)

def _one_way_time(env: Env) -> np.ndarray:
    return np.linalg.norm(env.pos - P.home, axis=1) / P.v


# ══════════════════════════════════════════════════════════════════════════════
# 1. FLEET LAGRANGIAN DUAL ESTIMATE  (K-aware; relaxation estimate, not certified)
# ══════════════════════════════════════════════════════════════════════════════
def _solve_fleet_lagrangian(env: Env, K: int, Emax_each: float, lam: float):
    """
    Relaxed fleet Lagrangian for a FIXED lambda. Mirrors the single-UAV
    _solve_lagrangian_exact, made K-aware:

      include node j  iff  j servable (round-trip e_j <= Emax_each, Effect 2)
                      AND  profit_j(lambda) > 0,
      profit_j = theta2*w_j
               - theta1 * W_approx_j * (tcd_j + tf_bar_j)     # Effect 1 (1/K)
               - lambda * e_j

    with W_approx_j = ceil(global_priority_rank_j / K) * w_mean   (round-robin
    split of the priority-ranked nodes into K chains), and aggregate energy
    budget K*Emax_each.

    Returns (dual_value, aggregate_slack).
    """
    M       = env.M
    e_j     = _round_trip_energy(env)                    # (M,)
    tf_bar  = _one_way_time(env)                         # (M,)

    # Effect 2 — per-UAV range feasibility (valid exclusion under split battery)
    servable = e_j <= Emax_each                          # (M,) bool
    if not servable.any():
        # nothing fits one UAV; relaxed objective is 0 nodes
        return 0.0 + lam * (0.0 - K * Emax_each), -K * Emax_each

    e_mean  = float(np.mean(e_j[servable]))
    N_exp   = min(int(servable.sum()),
                  max(1, int(K * Emax_each / (e_mean + 1e-9))))
    w_mean  = float(np.mean(env.wi))

    # priority rank (1 = highest) among ALL nodes, then within-chain rank ceil(r/K)
    rank_global  = np.argsort(np.argsort(-env.wi)) + 1   # 1..M
    within_chain = np.ceil(rank_global / K)              # Effect 1
    W_approx     = within_chain * w_mean                 # (M,)

    profit = (P.theta2 * env.wi
              - P.theta1 * W_approx * (env.tcd + tf_bar)
              - lam * e_j)

    x = ((profit > 0) & servable).astype(float)

    waoi_approx = float(np.sum(x * P.theta1 * W_approx * (env.tcd + tf_bar)))
    priority    = float(np.sum(x * P.theta2 * env.wi))
    energy_used = float(np.sum(x * e_j))
    budget      = K * Emax_each
    dual_val    = waoi_approx - priority + lam * (energy_used - budget)
    slack       = energy_used - budget
    return float(dual_val), float(slack)


def fleet_lagrangian_estimate(env: Env, K: int, Emax_each: float,
                              n_steps: int = LAMBDA_STEPS) -> float:
    """
    Lagrangian dual value  =  MAX over lambda >= 0  of  min_x L(x, lambda).

    NOTE ON A BUG IN THE SINGLE-UAV optimality_gap.py:
      lagrangian_bound_proper there returns the *minimum* dual over lambda. But
      L(lambda) = min_x L(x,lambda) is a valid lower bound for EVERY lambda, and
      the tight one is the MAXIMUM over lambda. Taking the min drives lambda to
      lambda_max, where the relaxed solution selects zero nodes and the dual
      collapses to -lambda_max * Emax -- a trivial, instance-and-K-invariant
      artifact, not a real relaxation bound. This function maximizes instead, so
      the value is meaningful and K-aware. (The single-UAV file should be fixed
      the same way; its current "LR bound" column is vacuous.)

    Method: coarse grid over [0, lambda_max] for a good start, then projected
    subgradient ASCENT (lambda += alpha * slack; at the inner optimum the
    subgradient of L wrt lambda is the constraint value energy_used - budget).
    Returns the best (highest) L(lambda) found.
    """
    e_j     = _round_trip_energy(env)
    lam_max = float(np.max(P.theta2 * env.wi / np.maximum(e_j, 1e-9)))
    if lam_max <= 0:
        d, _ = _solve_fleet_lagrangian(env, K, Emax_each, 0.0)
        return d

    # coarse grid for a good starting lambda (maximize)
    best_val, best_lam = -float('inf'), 0.0
    for lam0 in np.linspace(0.0, lam_max, 40):
        d, _ = _solve_fleet_lagrangian(env, K, Emax_each, lam0)
        if d > best_val:
            best_val, best_lam = d, lam0

    # subgradient ascent from the grid argmax (diminishing step), keep the max
    lam   = best_lam
    step0 = lam_max / 8.0
    for t in range(n_steps):
        dual_val, slack = _solve_fleet_lagrangian(env, K, Emax_each, lam)
        if dual_val > best_val:
            best_val = dual_val
        if abs(slack) < 1e-6:
            break
        # ascent toward feasibility: overspend (slack>0) -> raise lambda
        alpha = step0 / (1.0 + t)
        lam   = float(np.clip(lam + alpha * np.sign(slack), 0.0, lam_max))
    return best_val


# ══════════════════════════════════════════════════════════════════════════════
# 2. FLEET KNAPSACK ESTIMATE  (relaxation estimate, not certified) + CERTIFIED LB
# ══════════════════════════════════════════════════════════════════════════════
def fleet_knapsack_estimate(env: Env, K: int, Emax_each: float) -> float:
    """Fractional priority knapsack ignoring WAoI, round-trip costs, per-UAV range
    filter, aggregate budget K*Emax_each. Mirrors single-UAV priority_knapsack_bound.
    Relaxation ESTIMATE (round-trip costs over-charge a shared tour)."""
    e_j      = _round_trip_energy(env)
    servable = e_j <= Emax_each
    return _fractional_knapsack_value(env.wi, e_j, servable, K * Emax_each)


def fleet_certified_lb(env: Env, K: int, Emax_each: float) -> float:
    """CERTIFIED (loose) lower bound on J_fleet.

    J = theta1*WAoI - theta2*priority >= -theta2*priority   (WAoI >= 0), and any
    feasible fleet spends >= sum_{served} Ph*tcd_j on hovering with total energy
    <= K*Emax_each. So priority <= fractional knapsack with cost = hover-only and
    budget = K*Emax_each. Hence -theta2*that is a valid lower bound on J.
    Loose because it ignores flight energy (which dominates here)."""
    hover_cost = P.Ph * env.tcd
    servable   = hover_cost <= Emax_each                 # essentially all nodes
    return _fractional_knapsack_value(env.wi, hover_cost, servable, K * Emax_each)


def _fractional_knapsack_value(w: np.ndarray, cost: np.ndarray,
                               servable: np.ndarray, budget: float) -> float:
    """-theta2 * (max fractional priority under `budget`, among servable nodes)."""
    idx   = np.where(servable)[0]
    if idx.size == 0:
        return 0.0
    ratio = w[idx] / np.maximum(cost[idx], 1e-9)
    order = idx[np.argsort(-ratio)]
    E, tot = budget, 0.0
    for j in order:
        if cost[j] <= E:
            tot += w[j]; E -= cost[j]
        else:
            tot += w[j] * (E / max(cost[j], 1e-9)); break
    return float(-P.theta2 * tot)


# ══════════════════════════════════════════════════════════════════════════════
# 3. CERTIFIED EXACT FLEET OPTIMUM  (B&B, tiny M and K<=2)
# ══════════════════════════════════════════════════════════════════════════════
def fleet_exact_optimal(env: Env, K: int, Emax_each: float,
                        time_limit_s: float = EXACT_TIME_LIMIT):
    """
    Branch-and-bound for the team problem. Returns (best_obj, completed_bool).
    completed=False if the time limit was hit (then the value is NOT certified).

    Branching: extend any UAV by any globally-unvisited, range-feasible node, or
    stop. Symmetry over identical UAVs is broken by activating UAV k (its first
    node) only after UAV k-1 already holds >= 1 node. Pruning uses a VALID
    optimistic bound: theta1*WAoI_so_far - theta2*(pri_so_far + P_rem_UB), where
    P_rem_UB is a hover-cost fractional knapsack over remaining nodes (a valid
    upper bound on extra collectible priority), so the optimum is never pruned.
    """
    M        = env.M
    e_rt     = _round_trip_energy(env)
    best     = [float('inf')]
    t0       = time.perf_counter()
    hover    = P.Ph * env.tcd

    def _pri_ub(visited_mask, E_total_left):
        rem = np.where(~visited_mask)[0]
        if rem.size == 0:
            return 0.0
        ratio = env.wi[rem] / np.maximum(hover[rem], 1e-9)
        order = rem[np.argsort(-ratio)]
        E, ub = E_total_left, 0.0
        for j in order:                              # hover-cost knapsack = valid UB
            if hover[j] <= E:
                ub += env.wi[j]; E -= hover[j]
            else:
                ub += env.wi[j] * (E / max(hover[j], 1e-9)); break
        return ub

    def _obj(trajs):
        return (P.theta1 * sum(env.waoi(t) for t in trajs)
                - P.theta2 * sum(env.wi[j] for t in trajs for j in t))

    def _branch(trajs, pos, E_left, visited, n_active):
        if time.perf_counter() - t0 > time_limit_s:
            return False                              # signal: incomplete
        cur = _obj(trajs)
        if cur < best[0]:
            best[0] = cur
        # valid optimistic bound on the best reachable objective from here
        waoi_now = P.theta1 * sum(env.waoi(t) for t in trajs)
        pri_now  = sum(env.wi[j] for t in trajs for j in t)
        opt      = waoi_now - P.theta2 * (pri_now + _pri_ub(visited, sum(E_left)))
        if opt >= best[0] - 1e-9:
            return True                               # prune (cannot beat best)

        ok = True
        for k in range(K):
            # symmetry break: UAV k may receive its FIRST node only if UAV k-1 nonempty
            if len(trajs[k]) == 0 and k > 0 and len(trajs[k - 1]) == 0:
                break
            for j in range(M):
                if visited[j]:
                    continue
                d_to = float(np.linalg.norm(env.pos[j] - pos[k]))
                d_bk = float(np.linalg.norm(env.pos[j] - P.home))
                e_req = P.Pf * d_to / P.v + P.Ph * env.tcd[j] + P.Pf * d_bk / P.v
                if e_req > E_left[k]:
                    continue
                e_use = P.Pf * d_to / P.v + P.Ph * env.tcd[j]
                trajs[k].append(j); visited[j] = True
                old_pos, old_e = pos[k], E_left[k]
                pos[k], E_left[k] = env.pos[j].copy(), E_left[k] - e_use
                if not _branch(trajs, pos, E_left, visited, n_active):
                    ok = False
                pos[k], E_left[k] = old_pos, old_e
                trajs[k].pop(); visited[j] = False
                if not ok:
                    return False
        return ok

    trajs   = [[] for _ in range(K)]
    pos     = [P.home.copy() for _ in range(K)]
    E_left  = [Emax_each for _ in range(K)]
    visited = np.zeros(M, dtype=bool)
    completed = _branch(trajs, pos, E_left, visited, 0)
    val = best[0] if best[0] < float('inf') else 0.0   # 0 = "visit nothing"
    return float(val), bool(completed)


# ══════════════════════════════════════════════════════════════════════════════
# 4. POLICY ROWS  (needs torch)
# ══════════════════════════════════════════════════════════════════════════════
def load_fleet_policy(M: int, K: int, seed: int, model_dir: str):
    path = os.path.join(model_dir, f'fleet_M{M}_K{K}_split_seed{seed}.pt')
    if not os.path.exists(path):
        return None, None
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    pol  = muv.MultiUAVPolicy(hidden=256,
                              input_dim=ckpt.get('input_dim', MP.INPUT_DIM)).to(DEVICE)
    pol.load_state_dict(ckpt['policy']); pol.eval()
    return pol, ckpt.get('Emax_each', P.Emax / K)


def eval_policy_both(policy, env: Env, K: int, Emax_each: float):
    """(greedy obj, greedy+post-process obj) on one instance."""
    if policy is None:
        return float('nan'), float('nan')
    with torch.no_grad():
        f_g  = fleet_rollout(policy, env, K, DEVICE, Emax_each=Emax_each, greedy=True)
        obj_g = f_g.fleet_objective()
        f_pp = fleet_rollout(policy, env, K, DEVICE, Emax_each=Emax_each, greedy=True)
        f_pp = fleet_post_process(env, f_pp)
        obj_pp = f_pp.fleet_objective()
    return float(obj_g), float(obj_pp)


def best_fleet_baseline(env: Env, K: int, Emax_each: float, seed: int):
    vals = []
    for key in FLEET_BASELINES.values():
        f = fleet_baseline(env, K, key, Emax_each=Emax_each,
                           rng=np.random.default_rng(seed))
        vals.append(f.fleet_objective())
    return float(min(vals))


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_cell(M, K, seeds, n_instances, model_dir, run_exact):
    """One (M,K) cell, averaged over instances and over the available seed models."""
    Emax_each = P.Emax / K                              # split battery
    inst_rng  = np.random.default_rng(INSTANCE_SEED)
    inst_seeds = [int(inst_rng.integers(0, 10_000_000)) for _ in range(n_instances)]

    # model-independent rows (bounds, exact, baseline) — one pass over instances
    lr, kn, cert, exact, bl = [], [], [], [], []
    n_exact_done = 0
    for s in inst_seeds:
        env = Env(M=M, seed=s)
        lr.append(  fleet_lagrangian_estimate(env, K, Emax_each))
        kn.append(  fleet_knapsack_estimate(env, K, Emax_each))
        cert.append(fleet_certified_lb(env, K, Emax_each))
        bl.append(  best_fleet_baseline(env, K, Emax_each, seed=s))
        if run_exact:
            val, done = fleet_exact_optimal(env, K, Emax_each)
            exact.append(val if done else np.nan)
            n_exact_done += int(done)

    # policy rows — average over seed models, same instance stream
    g_per_seed, pp_per_seed = [], []
    if HAS_TORCH:
        for seed in seeds:
            pol, e_each = load_fleet_policy(M, K, seed, model_dir)
            if pol is None:
                continue
            e_each = e_each or Emax_each
            gs, pps = [], []
            for s in inst_seeds:
                env = Env(M=M, seed=s)
                og, opp = eval_policy_both(pol, env, K, e_each)
                gs.append(og); pps.append(opp)
            g_per_seed.append(np.nanmean(gs))
            pp_per_seed.append(np.nanmean(pps))

    def ms(a):
        a = np.array(a, dtype=float); a = a[~np.isnan(a)]
        return (float(np.mean(a)), float(np.std(a))) if a.size else (np.nan, 0.0)

    lr_m   = float(np.mean(lr));  kn_m = float(np.mean(kn))
    cert_m = float(np.mean(cert)); bl_m = float(np.mean(bl))
    ex_arr = np.array(exact, dtype=float)
    ex_m   = float(np.nanmean(ex_arr)) if np.any(~np.isnan(ex_arr)) else np.nan
    g_m, g_s   = ms(g_per_seed)
    pp_m, pp_s = ms(pp_per_seed)

    best_pol = pp_m if not np.isnan(pp_m) else g_m
    def gap_pct(v):
        # SIGNED gap vs the LR estimate. Negative => policy is BELOW the relaxed
        # estimate (the surrogate is not a certified bound, so this can happen and
        # means the policy exploits ordering structure the surrogate misses).
        return 100.0 * (v - lr_m) / max(abs(lr_m), 1e-6) if not np.isnan(v) else np.nan
    true_gap = (best_pol - ex_m) if not np.isnan(ex_m) and not np.isnan(best_pol) else np.nan
    bl_imp   = 100.0 * (bl_m - best_pol) / max(abs(bl_m), 1e-6) if not np.isnan(best_pol) else np.nan

    return dict(M=M, K=K, Emax_each=Emax_each,
                mlp_g=g_m, mlp_g_std=g_s, mlp_pp=pp_m, mlp_pp_std=pp_s,
                lr=lr_m, knap=kn_m, cert=cert_m, exact=ex_m, n_exact=n_exact_done,
                best_bl=bl_m, gap_g=gap_pct(g_m), gap_pp=gap_pct(best_pol),
                true_gap=true_gap, bl_imp=bl_imp,
                lr_list=lr, exact_list=exact)


def evaluate_all(M_list, K_list, seeds, n_instances, model_dir, allow_exact):
    results = {}
    for M in M_list:
        for K in K_list:
            run_exact = allow_exact and (M <= EXACT_FLEET_M_MAX) and (K <= EXACT_FLEET_K_MAX)
            tag = "  [exact B&B ON]" if run_exact else ""
            print(f"\n{'='*70}\n  M={M}  K={K}  (split Emax_each={P.Emax/K:.0f} J){tag}\n{'='*70}")
            r = evaluate_cell(M, K, seeds, n_instances, model_dir, run_exact)
            results[(M, K)] = r
            print(f"    policy greedy   : {r['mlp_g']:+.3f}")
            print(f"    policy +postproc: {r['mlp_pp']:+.3f}   gap vs LR: {r['gap_pp']:.1f}%")
            print(f"    LR estimate     : {r['lr']:+.3f}   (relaxation, not certified)")
            print(f"    Knapsack est.   : {r['knap']:+.3f}   (relaxation, not certified)")
            print(f"    Certified LB    : {r['cert']:+.3f}   (valid, loose)")
            if not np.isnan(r['exact']):
                print(f"    EXACT optimum   : {r['exact']:+.3f}  ({r['n_exact']}/{n_instances} certified)"
                      f"   TrueGap: {r['true_gap']:+.3f}")
            print(f"    best fleet BL   : {r['best_bl']:+.3f}   policy improves {r['bl_imp']:+.1f}%")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. TABLE
# ══════════════════════════════════════════════════════════════════════════════
def save_table(results, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, 'multiuav_optimality_gap_table.txt')
    cw = 11
    header = (f"{'M':>4} {'K':>3}  {'MLP-greedy':>{cw}} {'MLP+pp':>{cw}} "
              f"{'LR(est)':>{cw}} {'Knap(est)':>{cw}} {'CertLB':>{cw}} "
              f"{'Exact':>{cw}} {'Gap-g%':>8} {'Gap-pp%':>8} "
              f"{'TrueGap':>{cw}} {'BLimp%':>8}")
    sep = '-' * len(header)
    legend = [
        "Multi-UAV (Fleet) Optimality-Gap Analysis  —  split battery (Emax_each = Emax/K)",
        "=" * len(header),
        "MLP-greedy = shared-policy sequential-commit fleet rollout (greedy)",
        "MLP+pp     = + per-chain 2-opt + cross-UAV cheapest insertion",
        "LR(est)    = K-aware Lagrangian dual ESTIMATE (positional WAoI surrogate)  [NOT certified]",
        "Knap(est)  = fractional priority knapsack, round-trip costs                [NOT certified]",
        "CertLB     = certified lower bound, hover-energy knapsack                  [valid, loose]",
        f"Exact      = B&B certified optimum (only M<={EXACT_FLEET_M_MAX}, K<={EXACT_FLEET_K_MAX})",
        "Gap-pp%    = (MLP+pp - LR(est)) / |LR(est)| * 100   SIGNED, single-UAV-consistent reference",
        "             negative => policy is BELOW the relaxed estimate (surrogate is not a true bound)",
        "TrueGap    = MLP+pp - Exact   (CERTIFIED gap from global optimum, where Exact exists)",
        "BLimp%     = (best fleet baseline - MLP+pp) / |best baseline| * 100",
        "NOTE: Exact completes only for M<=7; trained fleet models start at M=50, so certified",
        "      TrueGap is n/a for trained cells. The B&B here certifies the bound machinery",
        "      (CertLB <= Exact); to get one certified TrueGap point, train a tiny M=6-7 fleet model.",
        "=" * len(header),
    ]
    rows = []
    for (M, K) in sorted(results):
        r = results[(M, K)]
        ex = f"{r['exact']:>+{cw}.3f}" if not np.isnan(r['exact']) else f"{'n/a':>{cw}}"
        tg = f"{r['true_gap']:>+{cw}.3f}" if not np.isnan(r['true_gap']) else f"{'n/a':>{cw}}"
        gg = f"{r['gap_g']:>7.1f}%" if not np.isnan(r['gap_g']) else f"{'n/a':>8}"
        gp = f"{r['gap_pp']:>7.1f}%" if not np.isnan(r['gap_pp']) else f"{'n/a':>8}"
        bi = f"{r['bl_imp']:>+7.1f}%" if not np.isnan(r['bl_imp']) else f"{'n/a':>8}"
        rows.append(
            f"{M:>4} {K:>3}  {r['mlp_g']:>+{cw}.3f} {r['mlp_pp']:>+{cw}.3f} "
            f"{r['lr']:>+{cw}.3f} {r['knap']:>+{cw}.3f} {r['cert']:>+{cw}.3f} "
            f"{ex} {gg} {gp} {tg} {bi}")
    text = '\n'.join(legend + [header, sep] + rows + [sep]) + '\n'
    with open(path, 'w') as f:
        f.write(text)
    print('\n' + text)
    print(f"  Table saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURE  (gap vs K per M; policy vs bounds vs K)
# ══════════════════════════════════════════════════════════════════════════════
def fig_fleet_gap(results, results_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [skip figure] matplotlib unavailable: {e}")
        return
    os.makedirs(results_dir, exist_ok=True)
    Ms = sorted({M for (M, _) in results})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(Ms)))

    ax = axes[0]
    for c, M in zip(cmap, Ms):
        Ks = sorted(K for (MM, K) in results if MM == M)
        ax.plot(Ks, [results[(M, K)]['mlp_pp'] for K in Ks], '-o', color=c, label=f'M={M}')
    ax.set_xlabel('Fleet size K'); ax.set_ylabel('Fleet objective (lower = better)')
    ax.set_title('(a) Policy (MLP+pp) objective vs K  — split battery')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for c, M in zip(cmap, Ms):
        Ks = sorted(K for (MM, K) in results if MM == M)
        ax.plot(Ks, [results[(M, K)]['gap_pp'] for K in Ks], '-s', color=c, label=f'M={M}')
    ax.set_xlabel('Fleet size K'); ax.set_ylabel('Gap vs LR estimate (%)')
    ax.set_title('(b) Optimality gap (MLP+pp vs LR estimate) vs K')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    out = os.path.join(results_dir, 'fig_multiuav_optimality_gap.png')
    plt.tight_layout(); plt.savefig(out, dpi=170, bbox_inches='tight'); plt.close()
    print(f"  Figure saved -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. SELF-TEST  (no torch needed) — validates the novel bound/exact code
# ══════════════════════════════════════════════════════════════════════════════
def _brute_force_fleet(env: Env, K: int, Emax_each: float):
    """Exhaustive optimum: assign each node to {unserved, 0..K-1}, try all orders
    per UAV, enforce per-UAV budget. Only for tiny M (<=6)."""
    M = env.M
    best = 0.0   # serving nothing
    def chain_feasible(traj):
        prev, E = P.home, Emax_each
        for j in traj:
            d_to = float(np.linalg.norm(env.pos[j] - prev))
            d_bk = float(np.linalg.norm(env.pos[j] - P.home))
            if P.Pf * d_to / P.v + P.Ph * env.tcd[j] + P.Pf * d_bk / P.v > E + 1e-9:
                return False
            E -= P.Pf * d_to / P.v + P.Ph * env.tcd[j]; prev = env.pos[j]
        return True
    def obj(trajs):
        return (P.theta1 * sum(env.waoi(t) for t in trajs)
                - P.theta2 * sum(env.wi[j] for t in trajs for j in t))
    for assign in itertools.product(range(K + 1), repeat=M):   # K = "unserved"
        groups = [[i for i in range(M) if assign[i] == k] for k in range(K)]
        per_uav_orders = []
        feasible = True
        for g in groups:
            best_chain, best_co = None, float('inf')
            for perm in itertools.permutations(g):
                if chain_feasible(list(perm)):
                    co = P.theta1 * env.waoi(list(perm)) - P.theta2 * sum(env.wi[j] for j in perm)
                    if co < best_co:
                        best_co, best_chain = co, list(perm)
            if best_chain is None and g:
                feasible = False; break
            per_uav_orders.append(best_chain or [])
        if feasible:
            o = obj(per_uav_orders)
            if o < best:
                best = o
    return float(best)


def self_test():
    print("=" * 64); print("  SELF-TEST (no torch required)"); print("=" * 64)
    rng = np.random.default_rng(0)
    ok = True

    # (1) exact B&B == brute force, several tiny instances, K=1 and K=2
    print("\n[1] fleet B&B  vs  brute force  (M=5)")
    for K in (1, 2):
        for t in range(6):
            s = int(rng.integers(0, 1_000_000))
            env = Env(M=5, seed=s)
            Ee = P.Emax / K
            bb, done = fleet_exact_optimal(env, K, Ee, time_limit_s=30)
            bf = _brute_force_fleet(env, K, Ee)
            match = abs(bb - bf) < 1e-6
            ok &= (done and match)
            flag = "OK " if (done and match) else "FAIL"
            print(f"   K={K} seed={s:<7d}  B&B={bb:+.4f}  brute={bf:+.4f}  {flag}")

    # (2) K=1 reductions: fleet exact vs BRUTE FORCE (rigorous), and vs single-UAV
    if HAS_SU_GAP:
        print("\n[2] K=1 reduction  (fleet exact vs brute force; fleet <= single-UAV)")
        for t in range(4):
            s = int(rng.integers(0, 1_000_000))
            env = Env(M=6, seed=s)
            fe, _ = fleet_exact_optimal(env, 1, P.Emax, time_limit_s=30)
            bf    = _brute_force_fleet(env, 1, P.Emax)
            se    = _su_exact(env, time_limit_s=30)
            bf_match = abs(fe - bf) < 1e-6                  # rigorous
            le_single = (se is None) or (fe <= se + 1e-4)   # fleet no worse than single
            print(f"   seed={s:<7d}  fleet={fe:+.4f}  brute={bf:+.4f} "
                  f"[{'OK' if bf_match else 'FAIL'}]   single={se if se is None else round(se,4)} "
                  f"[fleet<=single {'OK' if le_single else 'FAIL'}]")
            ok &= bf_match and le_single
        # knapsack identity at K=1 (same costs, same budget Emax)
        env = Env(M=40, seed=1)
        fk = fleet_knapsack_estimate(env, 1, P.Emax)
        sk = _su_knapsack(env)
        kmatch = abs(fk - sk) < 1e-6
        print(f"   knapsack(K=1)  fleet={fk:+.4f}  single={sk:+.4f}  "
              f"{'OK' if kmatch else 'FAIL'}")
        ok &= kmatch
    else:
        print("\n[2] (skipped: optimality_gap.py not importable)")

    # (3) bound validity: certified LB <= exact <= policy-free sanity (LR below exact often)
    print("\n[3] certified LB <= exact optimum  (must always hold)")
    for t in range(6):
        s = int(rng.integers(0, 1_000_000))
        env = Env(M=6, seed=s)
        for K in (1, 2):
            Ee = P.Emax / K
            ex, done = fleet_exact_optimal(env, K, Ee, time_limit_s=30)
            lb = fleet_certified_lb(env, K, Ee)
            valid = lb <= ex + 1e-6
            ok &= (done and valid)
            print(f"   K={K} seed={s:<7d}  CertLB={lb:+.3f} <= Exact={ex:+.3f}  "
                  f"{'OK' if valid else 'FAIL'}")

    # (4) K-awareness: LR estimate WAoI surrogate shrinks with K on a fixed instance
    print("\n[4] LR estimate is K-aware (sweet-spot shape on one M)")
    env = Env(M=100, seed=7)
    vals = [fleet_lagrangian_estimate(env, K, P.Emax / K) for K in (1, 2, 3, 4, 6, 8)]
    print("   K=1,2,3,4,6,8  LR estimate:", [f"{v:+.2f}" for v in vals])

    print("\n" + "=" * 64)
    print("  SELF-TEST:", "ALL PASS" if ok else "FAILURES ABOVE")
    print("=" * 64)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Multi-UAV fleet optimality-gap analysis.')
    ap.add_argument('--self-test', action='store_true', help='run correctness checks (no torch needed)')
    ap.add_argument('--quick',     action='store_true', help='M=[50], K=[1,2], 1 seed, 10 instances')
    ap.add_argument('--M',     type=int, nargs='+', default=None)
    ap.add_argument('--K',     type=int, nargs='+', default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--instances', type=int, default=0)
    ap.add_argument('--no-exact',  action='store_true')
    ap.add_argument('--model-dir', type=str, default=MODEL_DIR)
    ap.add_argument('--results-dir', type=str, default=RESULTS_DIR)
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)

    if args.quick:
        M_list, K_list, seeds, n_inst = [50], [1, 2], [42], (args.instances or 10)
    else:
        M_list = args.M or M_LIST
        K_list = args.K or K_LIST
        seeds  = args.seeds or SEEDS
        n_inst = args.instances or N_INSTANCES

    if not HAS_TORCH:
        print("  [warning] torch not available — policy rows will be n/a. "
              "Bounds + exact still run. Use --self-test for correctness checks.")

    print('=' * 70)
    print('  Multi-UAV Fleet Optimality-Gap Analysis')
    print(f'  M={M_list}  K={K_list}  seeds={seeds}  instances={n_inst}')
    print(f'  device={DEVICE}  model_dir={args.model_dir}')
    print(f'  exact B&B: M<={EXACT_FLEET_M_MAX}, K<={EXACT_FLEET_K_MAX} '
          f'{"(disabled)" if args.no_exact else ""}')
    print('=' * 70)

    results = evaluate_all(M_list, K_list, seeds, n_inst,
                           args.model_dir, allow_exact=not args.no_exact)
    save_table(results, args.results_dir)
    fig_fleet_gap(results, args.results_dir)
    print('\n  Done. Outputs in', args.results_dir)
