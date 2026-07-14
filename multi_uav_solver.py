"""
multi_uav_solver.py  —  Multi-UAV Priority-Aware AoI Trajectory Optimisation
=============================================================================
Extension of the single-UAV solver (uav_aoi_solver.py) to a FLEET of K UAVs.

This file documents EVERY change from the single-UAV formulation. Read the
THEORY block below before the code — it is the documentation reference.

═══════════════════════════════════════════════════════════════════════════
                          THEORETICAL FOUNDATION
═══════════════════════════════════════════════════════════════════════════

── 1. PROBLEM RESTATEMENT ───────────────────────────────────────────────────
Single-UAV (your paper):
  One rotary-wing UAV departs the depot S0, visits an ordered subset of nodes,
  and returns to S0 before its battery E_max is exhausted, minimising the
  composite objective  J = theta1 * WAoI - theta2 * (collected priority).

Multi-UAV (this file):
  K UAVs, all starting and ending at the SAME depot S0, each with its OWN
  battery budget E_max^k. The fleet collectively partitions the node set:
  every node is visited by AT MOST ONE UAV. The objective sums the per-UAV
  WAoI over all K chains and rewards the TOTAL priority collected by the fleet.

── 2. WHAT CHANGES IN THE FORMULATION ───────────────────────────────────────

  (a) DECISION VARIABLES
      Single: one partial permutation  Q : V(0)->...->V(N)->S0
      Multi:  K partial permutations  Q^1, ..., Q^K, mutually node-disjoint.
              We introduce a binary assignment  x_i^k in {0,1}
              ( = 1 iff node i is served by UAV k ) and a per-UAV order.

  (b) PARTITION CONSTRAINT  [NEW — did not exist in single-UAV]
          sum_{k=1..K}  x_i^k  <=  1     for every node i = 1..M
      i.e. no node is collected twice. A node may be left unvisited (the LHS
      can be 0) — this is the orienteering aspect: not all nodes must be served.

  (c) ENERGY CONSTRAINT  [now K separate constraints]
      Single:   sum_n ( Ph*tcd_n + Pf*tf_n )  <=  E_max
      Multi:    for each UAV k:
                  sum_{n in Q^k} ( Ph*tcd_n + Pf*tf_n^k )  <=  E_max^k
      The budgets are independent. tcd*_i = D_i / R is UNCHANGED (closed form),
      because the channel model (fixed altitude H, uniform rate R) is identical.

  (d) WAoI OBJECTIVE  [Lemma 1 applied PER CHAIN, then summed]
      Single:  WAoI = sum_{m=1..N} W(m) * ( tcd_(m) + tf_(m) )
               with W(m) = sum_{k<=m} w_(k)   (cumulative priority of one chain)

      Multi:   WAoI = sum_{k=1..K} sum_{m=1..N_k} W^k(m) * ( tcd_(m) + tf^k_(m) )
               where W^k(m) = sum_{i<=m} w_(i) is the cumulative priority along
               UAV k's OWN trajectory. Each UAV maintains an INDEPENDENT W(m)
               accumulator — the stage-coupling of Lemma 1 holds within each
               chain but NOT across chains. This is the key structural fact:
               the AoI of a node is measured against the return time of the UAV
               that served it, so chains are coupled only through the shared
               node-disjointness constraint, not through the AoI sum.

      Composite objective (P1-multi):
          min  theta1 * sum_k WAoI^k  -  theta2 * sum_k sum_{i in Q^k} w_i
          s.t. per-UAV energy budgets, data causality, partition constraint.

  (e) PROBLEM CLASS
      Single: stage-weighted Orienteering Problem (NP-hard).
      Multi:  stage-weighted TEAM Orienteering Problem (TOP). The classical
              TOP (Chao et al., 1996) is NP-hard, and contains single-vehicle
              orienteering as the special case K=1; therefore the multi-UAV
              problem is also NP-hard. (Proof: restrict K=1 -> recovers your
              Proposition 1.)

── 3. WHAT STAYS EXACTLY THE SAME ───────────────────────────────────────────
  * Closed-form hover time tcd*_i = D_i / R                      (unchanged)
  * Lemma 1 stage-weighted structure within each chain          (unchanged)
  * Channel model: LoS, |h|^2 = kappa0/H^2, uniform rate R      (unchanged)
  * theta1 / theta2 trade-off weights                           (unchanged)
  * Safe action masking principle (now applied per-UAV)         (adapted)
  * Energy model Ph (hover), Pf (cruise)                        (unchanged)

── 4. MDP / RL REFORMULATION ────────────────────────────────────────────────
  Single-UAV MDP:
      State  s_n = (L_n, E_n, W_n, visited)
      Action a_n = next node OR return-home

  Multi-UAV — we use a SEQUENTIAL DECENTRALISED construction (CTDE-lite):
      The fleet is built by a SINGLE SHARED policy that is queried UAV-by-UAV.
      Concretely we use a *round-robin / sequential-commit* scheme:
        - All K UAVs start at S0 with full batteries.
        - At each macro-step we advance the UAV with the MOST remaining energy
          (a load-balancing heuristic), letting the shared policy pick its next
          node from the GLOBALLY unvisited & per-UAV-feasible set.
        - A UAV that can no longer feasibly extend its route (energy mask empty)
          is retired to the depot.
      This keeps a single policy network (parameter sharing — the standard CTDE
      actor) while the global "visited" array enforces the partition constraint
      automatically: once any UAV claims node i, it is masked for all UAVs.

  Why sequential-commit rather than fully simultaneous MARL?
      * It guarantees the partition constraint by construction (no two UAVs can
        commit to the same node — the global mask removes it instantly).
      * It reuses your exact 16-dim feature pipeline and PPO trainer with only
        TWO added context features (UAV index fraction, fleet energy spread),
        so the architecture is a minimal, defensible delta from the single-UAV
        model — easy to ablate and explain to reviewers.
      * It scales to K UAVs with O(K*M) per macro-step, same complexity class.

── 5. FEATURE-SPACE CHANGES ─────────────────────────────────────────────────
  Single-UAV: 16-dim ( 10 node feats + 6 context feats ).
  Multi-UAV : 18-dim ( 10 node feats + 8 context feats ).
      Two NEW context features encode fleet state so the shared policy can
      condition on WHICH UAV it is currently acting for:
        ctx[6] = k / K              (which UAV is deciding — index fraction)
        ctx[7] = E_left^k / mean_j(E_left^j)   (this UAV's energy vs fleet mean)
      Everything else (node features 0..9, context 0..5) is byte-for-byte the
      single-UAV feature vector, computed from the ACTIVE UAV's state.
      => The single-UAV model is the K=1 special case with ctx[6]=0, ctx[7]=1.

── 6. BASELINES (multi-UAV versions) ────────────────────────────────────────
  Each single-UAV heuristic is lifted to K UAVs by the SAME sequential-commit
  scheme: at each macro-step the chosen UAV applies the heuristic rule over the
  globally-unvisited feasible set. This yields fair multi-UAV baselines:
  Multi-Random, Multi-NearestNeighbor, Multi-GreedyPriority, Multi-PDR.

── 7. EVALUATION METRICS ────────────────────────────────────────────────────
  Same composite objective, now fleet-summed. We additionally report:
    * total nodes served by the fleet
    * per-UAV load (nodes / priority) to show balance
    * fleet WAoI and fleet priority separately
  Headline comparison: objective vs K (K=1,2,3,4) at fixed M, and vs M for
  fixed K, against the multi-UAV baselines.
═══════════════════════════════════════════════════════════════════════════
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse all single-UAV components — same physics, same closed-form tcd*, etc.
from uav_aoi_solver import P, Env

os.makedirs('results',          exist_ok=True)
os.makedirs('models_multi_uav', exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MULTI-UAV PARAMETERS  (extends single-UAV P)
# ══════════════════════════════════════════════════════════════════════════════
class MP:
    """Multi-UAV parameters. Physics inherited from P (uav_aoi_solver)."""
    K          = 3          # number of UAVs in the fleet (default)
    # Per-UAV battery. Default: each UAV gets the SAME E_max as the single case,
    # so a K-UAV fleet has K times the total energy — this is the realistic
    # "buy more drones" scenario. Set to P.Emax / K to model "split one battery".
    Emax_each  = P.Emax     # J, per UAV

    # Feature dims: 10 node + 8 context = 18  (was 16)
    INPUT_DIM  = 18


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FLEET STATE  — tracks K UAVs sharing one node set
# ══════════════════════════════════════════════════════════════════════════════
class FleetState:
    """
    Holds the live state of all K UAVs over a shared node set (one Env).
    The GLOBAL visited array enforces the partition constraint:
    once any UAV visits node i, it is removed from every UAV's feasible set.
    """
    def __init__(self, env: Env, K: int, Emax_each: float):
        self.env     = env
        self.K       = K
        self.M       = env.M
        self.Emax_each = Emax_each                              # per-UAV budget (split or full)
        self.pos     = [P.home.copy()      for _ in range(K)]   # current pos per UAV
        self.E_left  = [Emax_each          for _ in range(K)]   # battery per UAV
        self.W_cum   = [0.0                for _ in range(K)]   # cumul. priority per UAV
        self.trajs   = [[]                 for _ in range(K)]   # node order per UAV
        self.active  = [True               for _ in range(K)]   # still flying?
        self.visited = np.zeros(self.M, dtype=bool)             # GLOBAL partition mask

    # ── per-UAV feasibility over the GLOBAL unvisited set ────────────────────
    def feasible_mask_k(self, k: int):
        """Nodes UAV k can still reach AND return home, excluding globally-visited."""
        curr   = self.pos[k]
        d_to   = np.linalg.norm(self.env.pos - curr,   axis=1)
        d_home = np.linalg.norm(self.env.pos - P.home, axis=1)
        e_need = P.Pf * (d_to + d_home) / P.v + P.Ph * self.env.tcd
        return (~self.visited) & (e_need <= self.E_left[k])

    def commit(self, k: int, j: int):
        """UAV k visits node j: update its state and the GLOBAL partition mask."""
        tf_to = float(np.linalg.norm(self.env.pos[j] - self.pos[k])) / P.v
        self.E_left[k] -= P.Pf * tf_to + P.Ph * self.env.tcd[j]
        self.pos[k]     = self.env.pos[j].copy()
        self.W_cum[k]  += self.env.wi[j]
        self.trajs[k].append(j)
        self.visited[j] = True          # <-- enforces partition for ALL UAVs

    def retire(self, k: int):
        self.active[k] = False

    def any_active(self):
        return any(self.active)

    # ── fleet-level objective (sum over chains) ──────────────────────────────
    def fleet_waoi(self):
        return float(sum(self.env.waoi(t) for t in self.trajs))

    def fleet_priority(self):
        return float(sum(sum(self.env.wi[j] for j in t) for t in self.trajs))

    def fleet_objective(self):
        # J = theta1 * sum_k WAoI^k  -  theta2 * sum_k priority^k
        return P.theta1 * self.fleet_waoi() - P.theta2 * self.fleet_priority()

    def fleet_nodes(self):
        return int(sum(len(t) for t in self.trajs))


print("multi_uav_solver: core classes loaded OK")


# ══════════════════════════════════════════════════════════════════════════════
# 2b. DECONFLICTION LAYER  (post-hoc launch scheduling — does NOT touch rollout,
#     fleet_objective, or training; the trajectories are frozen)
# ══════════════════════════════════════════════════════════════════════════════
#
# All K UAVs launch from the depot. Flown simultaneously (all at t=0) they pass
# through shared corridors at the same instant and violate the safety distance
# `delta`. We resolve this purely in TIME: give each UAV a launch offset o_k so
# that no two are within delta while both are airborne. This keeps the spatial
# trajectories — and therefore per-chain separability, the exact B&B, the
# certified LB and the LR estimate — completely intact.
#
# COST / MODELING NOTE:
#   env.waoi(traj) is built from intra-mission durations only, so it is SHIFT-
#   INVARIANT: a launch delay does not change it. For a delay to have a cost, AoI
#   must be referenced to a COMMON t=0 (start of the monitoring window). Under
#   that standard model, delaying chain k by o_k adds  o_k * W_total(chain_k)  to
#   the weighted age, i.e. an objective penalty  theta1 * o_k * W_cum[k]
#   (W_cum[k] is exactly the chain's total priority). fleet_objective_deconflicted
#   adds this term; plain fleet_objective is left unchanged so existing results,
#   bounds and training are unaffected. The makespan / launch-delay outputs are
#   model-free and valid regardless of whether you adopt the t=0 AoI term.
#
import itertools as _itertools


def _uav_airborne_samples(chain, env, dt):
    """Positions sampled every dt over a UAV's own mission clock [0, t_end]
    (fly -> hover tcd -> ... -> fly home). Empty chain -> single depot sample."""
    if not chain:
        return P.home.astype(float)[None, :]
    segs = []; t = 0.0; cur = P.home.astype(float)
    for j in chain:
        d = float(np.linalg.norm(env.pos[j] - cur)); dur = d / P.v
        segs.append((t, t + dur, cur.copy(), env.pos[j].astype(float))); t += dur
        segs.append((t, t + float(env.tcd[j]), env.pos[j].astype(float),
                     env.pos[j].astype(float))); t += float(env.tcd[j])
        cur = env.pos[j].astype(float)
    d = float(np.linalg.norm(cur - P.home)); dur = d / P.v
    segs.append((t, t + dur, cur.copy(), P.home.astype(float))); t_end = t + dur
    grid = np.arange(0.0, t_end + dt, dt)
    pos = np.tile(P.home.astype(float), (len(grid), 1))
    for (ts, te, p0, p1) in segs:
        m = (grid >= ts) & (grid < te)
        if m.any():
            frac = (grid[m] - ts) / max(te - ts, 1e-9)
            pos[m] = p0[None, :] + frac[:, None] * (p1 - p0)[None, :]
    return pos


def _forbidden_offset_intervals(Pa, Pb, delta, dt, inflate):
    """Forbidden relative offsets tau = o_b - o_a where the pair conflicts (both
    airborne within delta). `inflate` (= 2*v*dt) widens the threshold so the set
    is a CONSERVATIVE superset accounting for motion between samples: any offset
    strictly outside the result is genuinely conflict-free. Returns (lo,hi) list."""
    thr = delta + inflate
    diff = Pa[:, None, :] - Pb[None, :, :]
    D = np.sqrt((diff * diff).sum(axis=2))
    ii, jj = np.where(D < thr)
    if ii.size == 0:
        return []
    taus = np.sort(np.unique(ii - jj).astype(float) * dt)
    forb = []; lo = prev = taus[0]
    for x in taus[1:]:
        if x - prev <= dt + 1e-9:
            prev = x
        else:
            forb.append((lo - dt, prev + dt)); lo = prev = x
    forb.append((lo - dt, prev + dt))
    return forb


def _earliest_feasible(blocked):
    """Smallest o >= 0 not inside any (lo,hi) in `blocked`."""
    o = 0.0; blocked = sorted(blocked); changed = True
    while changed:
        changed = False
        for (lo, hi) in blocked:
            if lo - 1e-9 <= o <= hi + 1e-9:
                o = hi + 1e-3; changed = True      # escape step >> membership tol
    return o


def _schedule_for_order(order, F):
    """Earliest-feasible launch placement following `order`. F[(a,b)] = forbidden
    tau = o_b - o_a. Returns dict uav -> launch offset (s)."""
    o = {}
    for k in order:
        blocked = []
        for j in o:
            for (lo, hi) in F[(j, k)]:
                blocked.append((o[j] + lo, o[j] + hi))   # o_k not in o_j + F[(j,k)]
        o[k] = _earliest_feasible(blocked)
    return o


def deconfliction_schedule(fleet: 'FleetState', delta: float = 25.0, dt: float = 0.25,
                           optimize_order: bool = True, order_samples: int = 2000,
                           verify: bool = True):
    """
    Compute conflict-free launch offsets for a completed FleetState.

    Returns a dict:
      offsets         : {uav_k: launch_offset_seconds}
      makespan        : max offset (extra wall-clock to launch the fleet safely)
      total_delay     : sum of offsets (s)            [model-free]
      weighted_delay  : sum_k offset_k * W_cum[k]     [priority-weighted delay]
      aoi_penalty     : theta1 * weighted_delay       [objective units; needs t=0 AoI]
      conflicts_left  : residual conflicts after scheduling (0 if fully deconflicted)
      order           : the launch order used
      method          : 'greedy' or 'best-of-orderings'

    Strategy: order UAVs by chain priority (high first), give each the earliest
    feasible launch time vs already-placed UAVs. If optimize_order, also search
    over launch orderings (all K! for K<=6; sampled otherwise) and keep the
    min-weighted-delay schedule. The forbidden sets are conservative, so the
    result is provably conflict-free at the continuous level (verify re-checks on
    a fine grid)."""
    env = fleet.env
    K = fleet.K
    chains = [list(t) for t in fleet.trajs]
    Wtot = {k: float(fleet.W_cum[k]) for k in range(K)}
    samp = [_uav_airborne_samples(chains[k], env, dt) for k in range(K)]
    inflate = 2.0 * P.v * dt

    F = {}
    for a in range(K):
        for b in range(a + 1, K):
            fab = _forbidden_offset_intervals(samp[a], samp[b], delta, dt, inflate)
            F[(a, b)] = fab
            F[(b, a)] = [(-hi, -lo) for (lo, hi) in fab]

    greedy_order = sorted(range(K), key=lambda k: -Wtot[k])
    o_best = _schedule_for_order(greedy_order, F)
    method = 'greedy'; order_best = greedy_order

    def wdelay(o):
        return float(sum(o[k] * Wtot[k] for k in o))

    if optimize_order and K >= 2:
        if K <= 6:
            orders = [list(p) for p in _itertools.permutations(range(K))]
        else:
            rng = np.random.default_rng(0)
            orders = [list(rng.permutation(K)) for _ in range(order_samples)]
            orders.append(greedy_order)
        best_c = wdelay(o_best)
        for od in orders:
            oo = _schedule_for_order(od, F)
            c = wdelay(oo)
            if c < best_c - 1e-9:
                best_c, o_best, order_best = c, oo, od
        method = 'best-of-orderings'

    conflicts_left = 0
    if verify:
        conflicts_left = int(_residual_conflict(chains, env, o_best, delta, 0.1))

    wd = wdelay(o_best)
    return dict(offsets=o_best,
                makespan=float(max(o_best.values()) if o_best else 0.0),
                total_delay=float(sum(o_best.values())),
                weighted_delay=wd,
                aoi_penalty=float(P.theta1 * wd),
                conflicts_left=conflicts_left,
                order=list(order_best), method=method)


def _residual_conflict(chains, env, offsets, delta, dt):
    """True if any pair still conflicts under launch `offsets` (fine-grid check)."""
    K = len(chains)
    samp = [_uav_airborne_samples(chains[k], env, dt) for k in range(K)]
    for a in range(K):
        for b in range(a + 1, K):
            ga = offsets[a] + np.arange(len(samp[a])) * dt
            gb = offsets[b] + np.arange(len(samp[b])) * dt
            lo = max(ga[0], gb[0]); hi = min(ga[-1], gb[-1])
            if hi < lo:
                continue
            ts = np.arange(lo, hi + dt, dt)
            ia = np.clip(((ts - offsets[a]) / dt).round().astype(int), 0, len(samp[a]) - 1)
            ib = np.clip(((ts - offsets[b]) / dt).round().astype(int), 0, len(samp[b]) - 1)
            d = np.linalg.norm(samp[a][ia] - samp[b][ib], axis=1)
            if (d < delta).any():
                return True
    return False


def deconfliction_penalty(fleet: 'FleetState', offsets: dict) -> float:
    """AoI objective penalty for launch offsets under a common-t=0 AoI model:
    theta1 * sum_k offset_k * W_cum[k]  (W_cum[k] = chain k's total priority)."""
    return float(P.theta1 * sum(offsets[k] * fleet.W_cum[k] for k in offsets))


def fleet_objective_deconflicted(fleet: 'FleetState', offsets: dict) -> float:
    """Fleet objective INCLUDING the launch-delay AoI penalty (common-t=0 model).
    Equals fleet_objective() when all offsets are 0. Use this ONLY if you adopt
    the common-t=0 AoI term; otherwise report fleet_objective() + the seconds of
    makespan/delay separately."""
    return fleet.fleet_objective() + deconfliction_penalty(fleet, offsets)


print("multi_uav_solver: deconfliction layer loaded OK")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MULTI-UAV POLICY  (18-dim input — single-UAV 16-dim + 2 fleet features)
# ══════════════════════════════════════════════════════════════════════════════
class MultiUAVPolicy(nn.Module):
    """
    Identical architecture to the single-UAV Policy, but input_dim=18.
    A SINGLE shared network serves all K UAVs (parameter sharing = the
    centralised-training/decentralised-execution actor). The two extra context
    features tell the network which UAV it is acting for and how that UAV's
    energy compares to the fleet, so one set of weights handles the whole fleet.
    """
    def __init__(self, hidden=256, input_dim=MP.INPUT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.score_head = nn.Linear(hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask):
        h      = self.encoder(x)
        scores = self.score_head(h).squeeze(-1)
        scores = scores.masked_fill(mask, -1e9)
        log_p  = F.log_softmax(scores, dim=0)
        valid_h = h[~mask]
        value   = self.value_head(valid_h.mean(0)).squeeze() if valid_h.shape[0] > 0 \
                  else torch.zeros(1, device=x.device).squeeze()
        return log_p, value


# ══════════════════════════════════════════════════════════════════════════════
# 4.  18-DIM FEATURE EXTRACTION  (single-UAV 16-dim + 2 fleet-context dims)
# ══════════════════════════════════════════════════════════════════════════════
def make_features_multi(fleet: FleetState, k: int, device: str):
    """
    Build the (M, 18) feature tensor for the ACTIVE UAV k.

    Node features 0..9 and context 0..5 are computed EXACTLY as in the
    single-UAV make_features(), but from UAV k's own (pos, E_left, W_cum) and
    against the GLOBAL visited mask (so claimed nodes are infeasible for all).

    Two NEW fleet-context features (ctx 6, 7) condition the shared policy:
      ctx[6] = k / K                          (which UAV is deciding)
      ctx[7] = E_left[k] / mean(E_left active) (energy relative to the fleet)
    """
    env   = fleet.env
    M     = env.M
    curr  = fleet.pos[k]
    E_left= fleet.E_left[k]
    W_cum = fleet.W_cum[k]
    n_vis = len(fleet.trajs[k])

    # ── distances (vectorised), exactly as single-UAV ───────────────────────
    diff_curr = env.pos - curr
    diff_home = env.pos - P.home
    d_curr_m  = np.linalg.norm(diff_curr, axis=1)
    d_home_m  = np.linalg.norm(diff_home, axis=1)
    d_curr    = d_curr_m / P.area
    d_home    = d_home_m / P.area

    # ── feasibility against GLOBAL visited mask (partition enforcement) ──────
    e_need   = P.Pf * (d_curr_m + d_home_m) / P.v + P.Ph * env.tcd
    feasible = (~fleet.visited) & (e_need <= E_left)

    # ── marginal WAoI (Lemma 1), priority/WAoI ratio, energy slack ───────────
    tf_curr_j    = d_curr_m / P.v
    marginal_w   = W_cum * tf_curr_j + (W_cum + env.wi) * env.tcd
    tcd_max      = float(P.Di_hi / P.R)
    norm_mw      = float(M * P.wi_hi * tcd_max)
    marginal_w_n = marginal_w / (norm_mw + 1e-9)

    ratio   = P.theta2 * env.wi / (P.theta1 * marginal_w + 1e-6)
    ratio_n = np.clip(ratio / 20.0, 0.0, 1.0)

    slack = np.clip((E_left - e_need) / P.Emax, -1.0, 1.0)

    # ── node features (M, 10) — IDENTICAL layout to single-UAV ───────────────
    node_f = np.stack([
        env.pos[:, 0] / P.area,
        env.pos[:, 1] / P.area,
        env.wi / P.wi_hi,
        env.tcd / tcd_max,
        d_curr,
        d_home,
        feasible.astype(np.float32),
        marginal_w_n.astype(np.float32),
        ratio_n.astype(np.float32),
        slack.astype(np.float32),
    ], axis=1)

    # ── context (8 dims) = single-UAV 6 + 2 fleet features ───────────────────
    waoi_rate = P.theta1 * W_cum / (M * P.wi_hi + 1e-9)
    active_E  = [fleet.E_left[j] for j in range(fleet.K) if fleet.active[j]]
    mean_E    = float(np.mean(active_E)) if active_E else 1.0
    ctx = np.array([
        curr[0] / P.area,                  # 0  (single-UAV)
        curr[1] / P.area,                  # 1
        E_left / P.Emax,                   # 2
        W_cum / (M * P.wi_hi),             # 3
        n_vis / M,                         # 4
        float(waoi_rate),                  # 5
        k / max(fleet.K, 1),               # 6  [NEW: which UAV]
        E_left / (mean_E + 1e-9),          # 7  [NEW: energy vs fleet mean]
    ], dtype=np.float32)
    ctx_rep = np.broadcast_to(ctx, (M, 8))

    x    = np.concatenate([node_f, ctx_rep], axis=1).astype(np.float32)  # (M, 18)
    mask = ~feasible

    return (torch.from_numpy(x).to(device),
            torch.from_numpy(mask).to(device),
            feasible)


print("multi_uav_solver: policy + features loaded OK")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SEQUENTIAL-COMMIT FLEET ROLLOUT
# ══════════════════════════════════════════════════════════════════════════════
def fleet_rollout(policy: MultiUAVPolicy, env: Env, K: int, device: str,
                  Emax_each: float = MP.Emax_each, greedy: bool = False,
                  collect: bool = False):
    """
    Build the K-UAV fleet trajectory with a single shared policy.

    SCHEME (sequential-commit, load-balanced):
      Repeat until no UAV can move:
        1. Among ACTIVE UAVs, pick the one with the MOST remaining energy
           (load balancing — keeps routes comparable length).
        2. Build that UAV's 18-dim features over the GLOBAL unvisited set.
        3. If it has no feasible node, retire it to the depot.
        4. Else sample/argmax an action and COMMIT it (updates global mask).

    The global mask guarantees the partition constraint with zero extra logic.

    Returns:
      fleet (FleetState)                          always
      if collect: also (transitions, R_fleet) for PPO training, where each
        transition = (x_np, mask_np, action, log_prob, value) and R_fleet is
        the terminal fleet reward (= -fleet objective), shared by all steps.
    """
    fleet = FleetState(env, K, Emax_each)
    transitions = []

    policy.eval() if greedy else policy.train()
    grad_ctx = torch.no_grad() if (greedy and not collect) else torch.enable_grad()

    with (torch.no_grad() if not collect else torch.enable_grad()):
        # hard cap on macro-steps: at most M commits total + K retirements
        for _ in range(env.M + K + 1):
            if not fleet.any_active():
                break

            # 1. choose active UAV with most remaining energy
            active_ids = [j for j in range(K) if fleet.active[j]]
            if not active_ids:
                break
            k = max(active_ids, key=lambda j: fleet.E_left[j])

            # 2. features for UAV k over global unvisited set
            x, mask, feasible = make_features_multi(fleet, k, device)

            # 3. retire if stuck
            if not feasible.any():
                fleet.retire(k)
                continue

            # 4. act
            log_p, value = policy(x, mask)
            if greedy:
                action = int(log_p.argmax().item())
            else:
                dist   = torch.distributions.Categorical(logits=log_p)
                action = int(dist.sample().item())

            if collect:
                transitions.append((
                    x.detach().cpu().numpy(),
                    mask.detach().cpu().numpy(),
                    action,
                    float(log_p[action].item()),
                    float(value.item()),
                ))

            fleet.commit(k, action)

    if collect:
        R_fleet = -fleet.fleet_objective()       # terminal reward (higher better)
        return fleet, transitions, R_fleet
    return fleet


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MULTI-UAV BASELINES  (sequential-commit lift of single-UAV heuristics)
# ══════════════════════════════════════════════════════════════════════════════
def fleet_baseline(env: Env, K: int, method: str,
                   Emax_each: float = MP.Emax_each, rng=None):
    """
    Lift a single-UAV heuristic to K UAVs via the SAME sequential-commit scheme.
    At each macro-step the most-energy UAV applies the heuristic rule over the
    globally-unvisited feasible set, then commits (updating the global mask).
    """
    if rng is None:
        rng = np.random
    fleet = FleetState(env, K, Emax_each)

    for _ in range(env.M + K + 1):
        if not fleet.any_active():
            break
        active_ids = [j for j in range(K) if fleet.active[j]]
        if not active_ids:
            break
        k  = max(active_ids, key=lambda j: fleet.E_left[j])
        fm = fleet.feasible_mask_k(k)
        if not fm.any():
            fleet.retire(k)
            continue

        curr = fleet.pos[k]
        if method == 'random':
            j = int(rng.choice(np.where(fm)[0]))
        elif method == 'greedy_priority':
            j = int(np.where(fm)[0][np.argmax(env.wi[fm])])
        elif method == 'nearest_neighbor':
            d = np.linalg.norm(env.pos - curr, axis=1); d[~fm] = np.inf
            j = int(np.argmin(d))
        elif method == 'pdr':
            d = np.maximum(np.linalg.norm(env.pos - curr, axis=1), 1e-9)
            sc = env.wi / d; sc[~fm] = -np.inf
            j = int(np.argmax(sc))
        else:
            raise ValueError(method)
        fleet.commit(k, j)

    return fleet


FLEET_BASELINES = {
    'Multi-Random':           'random',
    'Multi-NearestNeighbor':  'nearest_neighbor',
    'Multi-GreedyPriority':   'greedy_priority',
    'Multi-PDR':              'pdr',
}


print("multi_uav_solver: rollout + baselines loaded OK")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FLEET POST-PROCESSING  (per-chain 2-opt + cross-UAV node insertion)
# ══════════════════════════════════════════════════════════════════════════════
def _chain_energy(env: Env, traj: list) -> float:
    """Propulsion energy of one UAV's chain, including return to depot."""
    E = 0.0; prev = P.home
    for j in traj:
        E += P.Pf * float(np.linalg.norm(env.pos[j] - prev)) / P.v + P.Ph * env.tcd[j]
        prev = env.pos[j]
    E += P.Pf * float(np.linalg.norm(prev - P.home)) / P.v
    return E


"""
fleet_post_process_fast.py  --  drop-in, OUTPUT-IDENTICAL replacements for
_two_opt_chain and fleet_post_process in multi_uav_solver.py.

Paste these two functions over the existing definitions (same names, same
signatures, same module-level helpers: _chain_energy, FleetState, Env.objective).
Results are bit-identical to the originals -- same scan order, same
first-improvement break, same strict-'>' tie-breaking -- so no eval number
changes. Only redundant work is removed:

  _two_opt_chain : objective computed ONCE per candidate (was twice on accepts);
                   best_obj carried across passes instead of recomputed.
  fleet_post_process : base_obj hoisted out of the j-loop; per-(j,k) insertion
                   gains CACHED and only the changed chain's column recomputed
                   after each insertion, instead of a full rescan every pass.

Expected speedup ~4-8x depending on the cell (biggest at K=1 / large M, where
the original blow-up was worst). This does NOT change the asymptotics of
env.objective itself -- for the order-of-magnitude win you need a delta-evaluated
objective (see note at bottom).

Verify identity after pasting (a few instances):
    from uav_aoi_solver import Env
    from multi_uav_solver import fleet_rollout, fleet_post_process
    # keep a copy of the OLD fleet_post_process as fleet_post_process_ref, then:
    for s in range(20):
        env = Env(M=100, seed=s)
        a = fleet_post_process(env, fleet_rollout(pol, env, K, dev, Emax_each=Ee, greedy=True))
        env = Env(M=100, seed=s)
        b = fleet_post_process_ref(env, fleet_rollout(pol, env, K, dev, Emax_each=Ee, greedy=True))
        assert abs(a.fleet_objective() - b.fleet_objective()) < 1e-9
"""


def _two_opt_chain(env, traj, budget):
    """2-opt within one chain; objective-improving + energy-feasible. Identical
    output to the original, but env.objective(cand) is computed once per candidate
    and best_obj is carried across passes."""
    if len(traj) < 3:
        return traj
    best = list(traj)
    best_obj = env.objective(best)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                cand = best[:i + 1] + best[i + 1:j + 1][::-1] + best[j + 1:]
                cand_obj = env.objective(cand)                 # ONCE (was 2x)
                if cand_obj < best_obj - 1e-6 and _chain_energy(env, cand) <= budget:
                    best = cand
                    best_obj = cand_obj
                    improved = True
                    break
            if improved:
                break
    return best


def fleet_post_process(env, fleet):
    from fleet_post_process_delta import fleet_post_process_delta
    return fleet_post_process_delta(env, fleet)

# ─────────────────────────────────────────────────────────────────────────────
# Order-of-magnitude next step (needs uav_aoi_solver.Env.objective / waoi):
# The priority term -TH2*sum(w) shifts by exactly -TH2*w[j] on inserting j,
# independent of position; only the WAoI term is position-dependent. With a
# delta form of chain WAoI, each insertion/2-opt candidate becomes O(1..L_suffix)
# instead of a full O(L) objective recompute -- typically another 10-50x. Send
# the body of Env.objective / waoi and I'll write the delta evaluator.
# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# 8.  PPO TRAINING  (fleet-level terminal reward, shared policy)
# ══════════════════════════════════════════════════════════════════════════════
def _collect_fleet_buffer(policy, M, K, n_episodes, rng, device, Emax_each):
    """Collect n_episodes of fleet rollouts into a flat PPO transition buffer."""
    bx, bm, ba, blo, bvo, bret, bend = [], [], [], [], [], [], []
    for _ in range(n_episodes):
        s   = int(rng.integers(0, 10_000_000))
        env = Env(M=M, seed=s)
        _, trans, R = fleet_rollout(policy, env, K, device,
                                    Emax_each=Emax_each, greedy=False, collect=True)
        if not trans:
            continue
        for idx, (xn, mn, an, lpn, vn) in enumerate(trans):
            is_last = (idx == len(trans) - 1)
            bx.append(xn); bm.append(mn); ba.append(an)
            blo.append(lpn); bvo.append(vn)
            bret.append(R if is_last else 0.0)
            bend.append(is_last)
    return (np.array(bx, dtype=np.float32), np.array(bm, dtype=bool),
            np.array(ba, dtype=np.int64),   np.array(blo, dtype=np.float32),
            np.array(bvo, dtype=np.float32), np.array(bret, dtype=np.float32),
            np.array(bend, dtype=bool))


def _gae(bret, bval, bend, gamma=1.0, lam=0.95):
    T = len(bret); adv = np.zeros(T, dtype=np.float32); g = 0.0
    for t in reversed(range(T)):
        nxt   = 0.0 if bend[t] else bval[min(t+1, T-1)]
        delta = bret[t] + gamma*nxt - bval[t]
        g     = delta + gamma*lam*(0.0 if bend[t] else g)
        adv[t]= g
    return adv, adv + bval


def train_fleet(M=40, K=3, n_epochs=300, eps_per_epoch=64, lr=3e-4,
                entropy_beta=0.02, device='cpu', seed=42,
                save_path='models_multi_uav/fleet_M40_K3.pt',
                Emax_each=MP.Emax_each, ppo_epochs=3, ppo_clip=0.15,
                value_coef=0.5, gae_gamma=1.0, gae_lambda=0.95,
                minibatch_size=64, log_every=10):
    """
    PPO training of the SHARED fleet policy. Mirrors the single-UAV trainer
    exactly, but the rollout is a K-UAV sequential-commit episode and the
    terminal reward is the FLEET objective. Parameter sharing across UAVs is
    automatic — every UAV queries the same network.
    """
    rng    = np.random.default_rng(seed)
    policy = MultiUAVPolicy(hidden=256, input_dim=MP.INPUT_DIM).to(device)
    opt    = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr/20)

    n_p = sum(x.numel() for x in policy.parameters())
    print(f'\nFleetPolicy: {n_p:,} params | M={M} K={K} | device={device}')
    print(f'{"Ep":>5} {"R_fleet":>9} {"Nodes":>7} {"Obj":>9} {"PG":>8} {"s/ep":>6}')
    print('─' * 50)

    history  = {'reward': [], 'nodes': [], 'obj': []}
    best_obj = float('inf')
    t_win    = deque(maxlen=10)

    for ep in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        bx, bm, ba, blo, bvo, bret, bend = _collect_fleet_buffer(
            policy, M, K, eps_per_epoch, rng, device, Emax_each)
        T = len(ba)
        if T == 0:
            continue

        adv_np, ret_np = _gae(bret, bvo, bend, gae_gamma, gae_lambda)
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
        adv_t = torch.from_numpy(adv_np).to(device)
        ret_t = torch.from_numpy(ret_np).to(device)
        lpo_t = torch.from_numpy(blo).to(device)
        act_t = torch.from_numpy(ba).to(device)

        pg_log = []
        indices = np.arange(T)
        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, minibatch_size):
                idx = indices[start:start+minibatch_size]
                if len(idx) == 0:
                    continue
                lp_b, val_b, ent_b = [], [], []
                for i in idx:
                    xi = torch.from_numpy(bx[i]).to(device)
                    mi = torch.from_numpy(bm[i]).to(device)
                    lp_i, val_i = policy(xi, mi)
                    lp_b.append(lp_i[act_t[i]])
                    val_b.append(val_i)
                    ent_b.append(-(lp_i.exp() * lp_i).sum())
                lp_new  = torch.stack(lp_b)
                val_new = torch.stack(val_b)
                ent_new = torch.stack(ent_b).mean()

                ratio = torch.exp(lp_new - lpo_t[idx])
                ab    = adv_t[idx]
                surr1 = ratio * ab
                surr2 = torch.clamp(ratio, 1-ppo_clip, 1+ppo_clip) * ab
                pg    = -torch.min(surr1, surr2).mean()
                vl    = F.mse_loss(val_new, ret_t[idx])
                loss  = pg + value_coef*vl - entropy_beta*ent_new

                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                pg_log.append(pg.item())

        sched.step()
        t_win.append(time.perf_counter() - t0)

        ep_R   = float(np.mean(bret[bend])) if bend.any() else 0.0
        ep_obj = -ep_R
        history['reward'].append(ep_R); history['obj'].append(ep_obj)

        if ep_obj < best_obj:
            best_obj = ep_obj
            torch.save({'policy': policy.state_dict(),
                        'M': M, 'K': K, 'input_dim': MP.INPUT_DIM,
                        'Emax_each': Emax_each}, save_path)

        if ep % log_every == 0 or ep == 1:
            n_steps = int(bend.sum())
            avg_nodes = T / max(n_steps, 1)
            print(f'{ep:>5} {ep_R:>9.2f} {avg_nodes:>7.1f} {ep_obj:>9.3f} '
                  f'{np.mean(pg_log):>8.4f} {np.mean(t_win):>6.2f}')

    print(f'\nBest fleet objective: {best_obj:.3f}  ->  saved to {save_path}')
    return policy, history


print("multi_uav_solver: post-process + PPO trainer loaded OK")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  EVALUATION  (fleet policy vs fleet baselines)
# ══════════════════════════════════════════════════════════════════════════════
def eval_fleet(policy, M, K, n=200, seed=9999, device='cpu',
               Emax_each=MP.Emax_each, use_postprocess=True):
    """Average fleet metrics over n random instances."""
    rng = np.random.default_rng(seed)
    obj, nodes, waoi, pri, bal = [], [], [], [], []
    policy.eval()
    with torch.no_grad():
        for _ in range(n):
            s   = int(rng.integers(0, 10_000_000))
            env = Env(M=M, seed=s)
            f   = fleet_rollout(policy, env, K, device, Emax_each=Emax_each, greedy=True)
            if use_postprocess:
                f = fleet_post_process(env, f)
            obj.append(f.fleet_objective())
            nodes.append(f.fleet_nodes())
            waoi.append(P.theta1 * f.fleet_waoi())
            pri.append(f.fleet_priority())
            lens = [len(t) for t in f.trajs]
            bal.append(float(np.std(lens)))     # load-balance: lower = more even
    return dict(obj=float(np.mean(obj)), nodes=float(np.mean(nodes)),
                waoi=float(np.mean(waoi)), priority=float(np.mean(pri)),
                load_std=float(np.mean(bal)),
                obj_list=obj, nodes_list=nodes)


def eval_fleet_baselines(M, K, n=200, seed=42, Emax_each=MP.Emax_each):
    rng = np.random.default_rng(seed)
    out = {name: {'obj': [], 'nodes': []} for name in FLEET_BASELINES}
    for _ in range(n):
        s = int(rng.integers(0, 10_000_000))
        for name, key in FLEET_BASELINES.items():
            env = Env(M=M, seed=s)
            f   = fleet_baseline(env, K, key, Emax_each=Emax_each, rng=np.random.default_rng(s))
            out[name]['obj'].append(f.fleet_objective())
            out[name]['nodes'].append(f.fleet_nodes())
    return {k: {'obj': float(np.mean(v['obj'])), 'nodes': float(np.mean(v['nodes'])),
                'obj_list': v['obj']} for k, v in out.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 10.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════
_C_FLEET = '#009E73'
_C_BASE  = {'Multi-Random': '#888888', 'Multi-NearestNeighbor': '#E69F00',
            'Multi-GreedyPriority': '#56B4E9', 'Multi-PDR': '#CC79A7'}


def fig_objective_vs_K(policy_by_K, M, n, device, budget_fn=None):
    """
    Objective vs fleet size K (the headline multi-UAV figure).
    budget_fn(K) -> Emax_each for that K. Under split battery this is
    lambda K: P.Emax / K; under full battery lambda K: MP.Emax_each.
    Each K is evaluated at the SAME budget its policy was trained on.
    """
    if budget_fn is None:
        budget_fn = lambda K: MP.Emax_each
    Ks = sorted(policy_by_K.keys())
    fleet_obj, fleet_nodes = [], []
    base_obj = {name: [] for name in FLEET_BASELINES}
    for K in Ks:
        e_each = budget_fn(K)
        pr = eval_fleet(policy_by_K[K], M, K, n=n, device=device, Emax_each=e_each)
        fleet_obj.append(pr['obj']); fleet_nodes.append(pr['nodes'])
        bl = eval_fleet_baselines(M, K, n=n, Emax_each=e_each)
        for name in FLEET_BASELINES:
            base_obj[name].append(bl[name]['obj'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(Ks, fleet_obj, 'o-', color=_C_FLEET, lw=2.2, ms=8, label='Fleet MLP (ours)')
    for name in FLEET_BASELINES:
        ax1.plot(Ks, base_obj[name], 's--', color=_C_BASE[name], lw=1.4, ms=5, label=name)
    ax1.axhline(0, color='black', lw=0.7)
    ax1.set_xlabel('Number of UAVs K'); ax1.set_ylabel('Fleet composite objective (lower=better)')
    ax1.set_title(f'(a) Objective vs fleet size  (M={M})'); ax1.set_xticks(Ks)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(Ks, fleet_nodes, 'o-', color=_C_FLEET, lw=2.2, ms=8, label='Fleet MLP (ours)')
    ax2.axhline(M, color='red', lw=1.0, ls=':', label=f'All M={M} nodes')
    ax2.set_xlabel('Number of UAVs K'); ax2.set_ylabel('Total nodes served by fleet')
    ax2.set_title(f'(b) Fleet coverage vs K  (M={M})'); ax2.set_xticks(Ks)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/fig_multiuav_objective_vs_K_M{M}.png', dpi=160)
    plt.close()
    print(f'  Saved results/fig_multiuav_objective_vs_K_M{M}.png')


def fig_fleet_trajectory(policy, M, K, device, seed=7, Emax_each=MP.Emax_each):
    """Plot the K-UAV fleet trajectories with per-UAV colours."""
    env = Env(M=M, seed=seed)
    f   = fleet_rollout(policy, env, K, device, Emax_each=Emax_each, greedy=True)
    f   = fleet_post_process(env, f)

    uav_colors = ['#009E73', '#D55E00', '#0072B2', '#CC79A7', '#E69F00']
    served = set(j for t in f.trajs for j in t)
    fig, ax = plt.subplots(figsize=(8, 7.5))
    for i in range(M):
        sz  = 40 + (env.wi[i]/P.wi_hi)*140
        clr = '#cccccc' if i not in served else 'white'
        ax.scatter(*env.pos[i], s=sz, c=clr, edgecolors='black', lw=0.5, zorder=3)
        ax.text(env.pos[i,0]+10, env.pos[i,1]+10, f'{env.wi[i]:.0f}', fontsize=6)
    for k in range(K):
        t = f.trajs[k]
        if not t:
            continue
        px = [P.home[0]] + [env.pos[j,0] for j in t] + [P.home[0]]
        py = [P.home[1]] + [env.pos[j,1] for j in t] + [P.home[1]]
        ax.plot(px, py, '-o', color=uav_colors[k % len(uav_colors)], lw=1.8,
                ms=5, alpha=0.85, zorder=4, label=f'UAV {k+1} ({len(t)} nodes)')
    ax.scatter(*P.home, s=260, marker='*', c='gold', edgecolors='black', lw=1.5, zorder=6)
    obj  = f.fleet_objective(); waoi = P.theta1 * f.fleet_waoi()
    ax.set_title(f'Fleet trajectory  M={M}, K={K}, served={f.fleet_nodes()}\n'
                 f'Fleet WAoI={waoi:.1f}  Obj={obj:.2f}')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.legend(fontsize=8)
    ax.grid(alpha=0.2); ax.set_xlim(-30, P.area+30); ax.set_ylim(-30, P.area+30)
    plt.tight_layout()
    plt.savefig(f'results/fig_multiuav_trajectory_M{M}_K{K}.png', dpi=160)
    plt.close()
    print(f'  Saved results/fig_multiuav_trajectory_M{M}_K{K}.png')


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Multi-UAV priority-aware AoI solver.')
    ap.add_argument('--quick',   action='store_true', help='Fast demo')
    ap.add_argument('--M',       type=int, default=100, help='Number of sensor nodes')
    ap.add_argument('--K',       type=int, nargs='+', default=[1, 2, 3, 4],
                    help='Fleet sizes to train/compare')
    ap.add_argument('--epochs',  type=int, default=0)
    ap.add_argument('--seed',    type=int, default=42, help='Training seed')
    ap.add_argument('--instances', type=int, default=0, help='Override eval instances')
    ap.add_argument('--full-battery', action='store_true',
                    help='Give each UAV a full E_max (default is SPLIT: E_max/K each)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    M      = args.M
    K_list = args.K
    split  = not args.full_battery          # split battery is the DEFAULT

    if args.quick:
        n_epochs = args.epochs or 40
        eps_pe   = 32
        n_eval   = args.instances or 50
    else:
        n_epochs = args.epochs or 300
        eps_pe   = 64
        n_eval   = args.instances or 200

    # Per-K budget rule
    budget_fn = (lambda K: P.Emax / K) if split else (lambda K: MP.Emax_each)

    print('=' * 60)
    print('  Multi-UAV Priority-Aware AoI — Team Orienteering Solver')
    print(f'  device={device}  M={M}  K={K_list}  epochs={n_epochs}  seed={args.seed}')
    print(f'  battery mode: {"SPLIT E_max/K (default)" if split else "full E_max each"}')
    print('=' * 60)

    policy_by_K = {}
    for K in K_list:
        Emax_each = budget_fn(K)
        bat = 'split' if split else 'full'
        print(f'\n{"="*60}\n  Training fleet  M={M}  K={K}  '
              f'(Emax_each={Emax_each:.0f} J, {bat})\n{"="*60}')
        save = f'models_multi_uav/fleet_M{M}_K{K}_{bat}_seed{args.seed}.pt'
        pol, hist = train_fleet(M=M, K=K, n_epochs=n_epochs, eps_per_epoch=eps_pe,
                                device=device, save_path=save, Emax_each=Emax_each,
                                seed=args.seed)
        ckpt = torch.load(save, map_location=device)
        pol.load_state_dict(ckpt['policy'])
        policy_by_K[K] = pol

        pr = eval_fleet(pol, M, K, n=n_eval, device=device, Emax_each=Emax_each)
        print(f'  -> [M={M} K={K}] Obj={pr["obj"]:.3f}  Nodes={pr["nodes"]:.1f}  '
              f'WAoI={pr["waoi"]:.2f}  Priority={pr["priority"]:.2f}  '
              f'LoadStd={pr["load_std"]:.2f}')
        fig_fleet_trajectory(pol, M, K, device, Emax_each=Emax_each)

    if len(policy_by_K) >= 2:
        fig_objective_vs_K(policy_by_K, M, n_eval, device, budget_fn=budget_fn)

    print('\n' + '=' * 60)
    print('  Done. Models in models_multi_uav/, figures in results/')
    print('=' * 60)