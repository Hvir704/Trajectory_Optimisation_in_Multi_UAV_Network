"""
deconfliction_scheduler.py
================================================================================
Solve for per-UAV LAUNCH OFFSETS that keep every pair of UAVs >= delta apart for
the whole mission, at minimum AoI cost, on the existing trained models. No
retraining; trajectories are frozen, we only choose WHEN each UAV launches.

IMPORTANT MODELING NOTE (read this):
  Your waoi(traj) is built from intra-mission durations only, so it is SHIFT-
  INVARIANT: a launch delay does not change it. For a launch delay to have a
  cost, AoI must be referenced to a COMMON t=0 (start of the monitoring window).
  Under that standard model, delaying chain k by o_k adds  o_k * W_total(chain_k)
  to the weighted age, i.e. an objective penalty  theta1 * o_k * W_total(chain_k).
  We report cost BOTH ways:
    * launch delay (seconds): makespan max o_k and total/weighted delay   [model-free]
    * objective penalty theta1 * sum_k o_k * W_total_k                     [needs t=0 AoI]
  To actually USE the penalty, add  theta1 * o_k * W_total_k  to each UAV's
  objective in your evaluation. The seconds-of-delay numbers are valid regardless.

CONFLICT MODEL: two UAVs conflict if both are AIRBORNE at the same instant and
within delta. A UAV waiting on the ground (pre-launch) or landed (post-return) is
inactive and cannot collide (idle costs no battery — your stated assumption).

SCHEDULERS:
  GREEDY  : order UAVs by chain priority (high first so they launch earliest),
            give each the earliest launch time feasible vs already-placed UAVs.
  BEST    : same earliest-feasible placement, but searched over launch ORDERINGS
            (all K! for K<=6; sampled for K>=7) and keep the min-cost schedule.
            This is the near-optimal baseline; the greedy-vs-best gap shows the
            value of optimizing the order.

RUN:
  python deconfliction_scheduler.py --model-dir models_multi_uav --M 100 200 --K 4 6 8 --instances 30
  python deconfliction_scheduler.py --delta 25 --instances 20
"""
from __future__ import annotations
import os, sys, glob, argparse, warnings, itertools, random
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from uav_aoi_solver import Env, P
import multi_uav_solver as muv
from multi_uav_solver import fleet_rollout

INSTANCE_SEED = 7777


def find_ckpt(M, K, seed, root):
    name = f'fleet_M{M}_K{K}_split_seed{seed}.pt'
    if os.path.exists(os.path.join(root, name)):
        return os.path.join(root, name)
    hits = glob.glob(os.path.join(root, '**', name), recursive=True)
    return hits[0] if hits else None


def airborne_samples(chain, env, dt):
    """Positions sampled every dt over a UAV's own mission clock [0, t_end].
    Returns array (n,2). Empty chain -> single depot sample."""
    if not chain:
        return P.home.astype(float)[None, :]
    pts = []; t = 0.0; cur = P.home.astype(float)
    segs = []
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


def forbidden_intervals(Pa, Pb, delta, dt, inflate=0.0):
    """Forbidden set of relative offsets tau = o_b - o_a (both launched, airborne
    overlap within delta). Returns sorted list of (lo,hi) closed intervals in sec.
    `inflate` widens the distance threshold so the set is a CONSERVATIVE superset
    that accounts for motion between samples (use inflate = 2*v*dt): any offset
    strictly outside the returned set is then genuinely conflict-free."""
    thr = delta + inflate
    diff = Pa[:, None, :] - Pb[None, :, :]
    D = np.sqrt((diff * diff).sum(axis=2))
    ii, jj = np.where(D < thr)
    if ii.size == 0:
        return []
    shifts = np.unique(ii - jj)                 # forbidden integer shifts s
    taus = np.sort(shifts.astype(float) * dt)
    forb = []
    lo = prev = taus[0]
    for x in taus[1:]:
        if x - prev <= dt + 1e-9:
            prev = x
        else:
            forb.append((lo - dt, prev + dt)); lo = prev = x   # pad by dt (non-grid offsets)
    forb.append((lo - dt, prev + dt))
    return forb


def earliest_feasible(o_k_lowerbound, blocked):
    """Smallest o >= o_k_lowerbound not inside any (lo,hi) in `blocked`."""
    o = max(0.0, o_k_lowerbound)
    blocked = sorted(blocked)
    changed = True
    while changed:
        changed = False
        for (lo, hi) in blocked:
            if lo - 1e-9 <= o <= hi + 1e-9:
                o = hi + 1e-3      # escape step (>> 1e-9 membership tol; ms granularity)
                changed = True
    return o


def schedule_for_order(order, F, offsets_cache=None):
    """Earliest-feasible placement following `order`. F[(a,b)] = forbidden tau=o_b-o_a.
    Returns dict uav->offset."""
    o = {}
    for k in order:
        blocked = []
        for j in o:
            # constraint: o_k - o_j not in F[(j,k)]  ->  o_k not in o_j + F[(j,k)]
            for (lo, hi) in F[(j, k)]:
                blocked.append((o[j] + lo, o[j] + hi))
        o[k] = earliest_feasible(0.0, blocked)
    return o


def cost_weighted_delay(o, Wtot):
    """sum_k o_k * W_total_k  (priority-weighted launch delay)."""
    return float(sum(o[k] * Wtot[k] for k in o))


def residual_conflict(chains, env, o, delta, dt):
    """True if any pair still conflicts under offsets o (validation)."""
    K = len(chains)
    samp = [airborne_samples(chains[k], env, dt) for k in range(K)]
    ends = [o[k] + (len(samp[k]) - 1) * dt for k in range(K)]
    for a in range(K):
        for b in range(a + 1, K):
            # global grids
            ta0, tb0 = o[a], o[b]
            # align on global dt grid
            ga = ta0 + np.arange(len(samp[a])) * dt
            gb = tb0 + np.arange(len(samp[b])) * dt
            # overlap region
            lo = max(ga[0], gb[0]); hi = min(ga[-1], gb[-1])
            if hi < lo:
                continue
            ts = np.arange(lo, hi + dt, dt)
            ia = np.clip(((ts - ta0) / dt).round().astype(int), 0, len(samp[a]) - 1)
            ib = np.clip(((ts - tb0) / dt).round().astype(int), 0, len(samp[b]) - 1)
            d = np.linalg.norm(samp[a][ia] - samp[b][ib], axis=1)
            if (d < delta).any():
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='.')
    ap.add_argument('--M', type=int, nargs='+', default=[100, 200])
    ap.add_argument('--K', type=int, nargs='+', default=[4, 6, 8])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--instances', type=int, default=30)
    ap.add_argument('--delta', type=float, default=25.0)
    ap.add_argument('--dt', type=float, default=0.25, help='time resolution (s)')
    ap.add_argument('--order-samples', type=int, default=2000,
                    help='random orderings sampled when K!>this for BEST')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--out', default='./deconfliction_report.txt')
    args = ap.parse_args()
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'

    rng = np.random.default_rng(INSTANCE_SEED)
    inst_seeds = [int(rng.integers(0, 10_000_000)) for _ in range(args.instances)]
    lines = []
    def emit(s=''):
        print(s); lines.append(s)

    emit(f"Deconfliction launch-scheduler   device={device}   delta={args.delta}m   dt={args.dt}s")
    emit(f"model-dir={os.path.abspath(args.model_dir)}   instances={args.instances}")
    emit("cost = launch delay (model-free) AND theta1*sum(o_k*W_total_k) objective penalty")
    emit(f"theta1={P.theta1}, theta2={P.theta2}")
    emit("=" * 96)

    for M in args.M:
        for K in args.K:
            if K < 2:
                continue
            # gather rollouts
            cell = []
            for seed in args.seeds:
                path = find_ckpt(M, K, seed, args.model_dir)
                if path is None:
                    continue
                ck = torch.load(path, map_location=device, weights_only=False)
                pol = muv.MultiUAVPolicy(hidden=256, input_dim=ck.get('input_dim', 18)).to(device)
                pol.load_state_dict(ck['policy']); pol.eval()
                Ee = ck.get('Emax_each', P.Emax / K)
                with torch.no_grad():
                    for s in inst_seeds:
                        env = Env(M=M, seed=s)
                        f = fleet_rollout(pol, env, K, device, Emax_each=Ee, greedy=True)
                        cell.append(([list(t) for t in f.trajs], env, f.fleet_objective()))
            if not cell:
                emit(f"\n  M={M} K={K}: [no checkpoint]"); continue

            g_make, g_wdel, g_pen, g_penpct = [], [], [], []
            b_make, b_wdel, b_pen, b_penpct = [], [], [], []
            g_conf = b_conf = 0
            for chains, env, base_obj in cell:
                Kk = len(chains)
                Wtot = {k: float(sum(env.wi[j] for j in chains[k])) for k in range(Kk)}
                samp = [airborne_samples(chains[k], env, args.dt) for k in range(Kk)]
                inflate = 2.0 * P.v * args.dt          # conservative motion bound
                # forbidden sets per ordered pair
                F = {}
                for a in range(Kk):
                    for b in range(Kk):
                        if a == b:
                            continue
                        if (a, b) in F:
                            continue
                        fab = forbidden_intervals(samp[a], samp[b], args.delta, args.dt, inflate)
                        F[(a, b)] = fab
                        # tau' = o_a - o_b = -(o_b - o_a): mirror intervals
                        F[(b, a)] = [(-hi, -lo) for (lo, hi) in fab]

                # GREEDY: order by priority desc
                g_order = sorted(range(Kk), key=lambda k: -Wtot[k])
                og = schedule_for_order(g_order, F)
                g_conf += int(residual_conflict(chains, env, og, args.delta, 0.1))

                # BEST: search orderings
                if Kk <= 6:
                    orders = list(itertools.permutations(range(Kk)))
                else:
                    orders = [tuple(np.random.permutation(Kk)) for _ in range(args.order_samples)]
                    orders.append(tuple(g_order))   # include greedy order
                best_o, best_c = og, cost_weighted_delay(og, Wtot)
                for od in orders:
                    oo = schedule_for_order(list(od), F)
                    c = cost_weighted_delay(oo, Wtot)
                    if c < best_c:
                        best_c, best_o = c, oo
                b_conf += int(residual_conflict(chains, env, best_o, args.delta, 0.1))

                # metrics
                base_absobj = abs(base_obj) if abs(base_obj) > 1e-9 else 1.0
                for (o, mk, wd, pn, pp) in [
                    (og, g_make, g_wdel, g_pen, g_penpct),
                    (best_o, b_make, b_wdel, b_pen, b_penpct)]:
                    mk.append(max(o.values()))
                    wd.append(cost_weighted_delay(o, Wtot))
                    pen = P.theta1 * cost_weighted_delay(o, Wtot)
                    pn.append(pen); pp.append(100 * pen / base_absobj)

            emit(f"\n  ===== M={M} K={K}  ({len(cell)} instances) =====")
            emit(f"    conflicts after scheduling:   greedy {g_conf}/{len(cell)}   "
                 f"best {b_conf}/{len(cell)}   (should be 0)")
            emit(f"    {'':14}{'makespan(s)':>13}{'wDelay(s*pri)':>15}"
                 f"{'objPenalty':>12}{'penalty %obj':>14}")
            emit(f"    {'GREEDY':14}{np.mean(g_make):>13.1f}{np.mean(g_wdel):>15.1f}"
                 f"{np.mean(g_pen):>12.2f}{np.mean(g_penpct):>13.1f}%")
            emit(f"    {'BEST(order)':14}{np.mean(b_make):>13.1f}{np.mean(b_wdel):>15.1f}"
                 f"{np.mean(b_pen):>12.2f}{np.mean(b_penpct):>13.1f}%")
            impr = 100 * (np.mean(g_wdel) - np.mean(b_wdel)) / max(np.mean(g_wdel), 1e-9)
            emit(f"    optimizing launch order cuts weighted delay by {impr:.0f}% vs greedy")

    emit("\n" + "=" * 96)
    emit("READING IT:")
    emit("  * conflicts after scheduling should be 0/0 -> staggering fully deconflicts (no spatial")
    emit("    coupling needed; per-chain separability and the optimality-gap machinery all survive).")
    emit("  * 'penalty %obj' is the AoI price of deconfliction as a fraction of the fleet objective,")
    emit("    UNDER a common-t=0 AoI model. Watch how it grows with K -> that is your AoI-cost-vs-")
    emit("    fleet-size tradeoff (and a mechanism-backed version of 'optimal fleet size').")
    emit("  * makespan(s) is model-free: the extra wall-clock to launch the whole fleet safely.")
    emit("  * greedy-vs-best gap = value of optimizing the launch order (your algorithmic content).")
    with open(args.out, 'w') as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\n  saved -> {args.out}")


if __name__ == '__main__':
    main()
