"""
conflict_probe.py  (v2 — adds timing-mitigation analysis)
================================================================================
Measure how often the trained fleet's UAVs conflict in spacetime, AND whether
those conflicts are TIMING-RESOLVABLE (fixable by staggering launches or small
speed differences, which preserve per-chain separability) or GENUINE spatial
contention (which forces the coupled spatiotemporal model and breaks the
optimality-gap machinery). No retraining — runs on existing checkpoints.

PHYSICS: each UAV launches from the depot (optionally after a launch offset o_k),
flies at speed v*speed_scale_k between assigned nodes, hovers tcd_j at each, and
returns home. Position(t) is piecewise-linear; we sample all UAVs on one common
time grid and check pairwise distance at each instant.

PHASE 1  baseline characterization: conflict rate vs safety distance delta
         (RAW includes depot launch-stack; MID excludes a depot radius).
PHASE 2  mitigations at a reference delta:
   (a) LAUNCH STAGGER sweep  o_k = k*Delta  for several Delta  [realizable]
   (b) SPEED JITTER sweep     v_k = v*(1 +/- j), random per UAV [realizable]
   (c) RESOLVABILITY CEILING  per conflicting pair, is there ANY relative launch
       offset in a practical range that removes the conflict? Fraction resolvable
       = optimistic upper bound on what pure timing can fix.

READING IT:
   * stagger/jitter collapse MID% -> conflicts are timing artifacts; a temporal
     scheduling layer fixes them and your separability/optimality story survives.
   * MID% stays high after stagger AND ceiling is low -> genuine same-place
     contention; the coupled spatiotemporal model is justified (and necessary).

RUN:
   python conflict_probe.py --model-dir models_multi_uav --M 100 200 --K 4 6 8 --instances 30
   python conflict_probe.py --ref-delta 25 --stagger 0 5 10 20 30 --jitter 0 0.05 0.1
"""
from __future__ import annotations
import os, sys, glob, argparse, warnings
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


def uav_endtime(chain, env, speed_scale=1.0):
    t = 0.0; cur = P.home
    for j in chain:
        t += float(np.linalg.norm(env.pos[j] - cur)) / (P.v * speed_scale)
        t += float(env.tcd[j]); cur = env.pos[j]
    t += float(np.linalg.norm(cur - P.home)) / (P.v * speed_scale)
    return t


def sample_uav_positions(chain, env, grid, offset=0.0, speed_scale=1.0):
    """Position at each grid time, with launch `offset` and `speed_scale`.
    active = True only while airborne (after launch, before return)."""
    n = len(grid)
    pos = np.tile(P.home.astype(float), (n, 1))
    active = np.zeros(n, dtype=bool)
    if not chain:
        return pos, active
    segs = []; t = offset; cur = P.home.astype(float)
    for j in chain:
        d = float(np.linalg.norm(env.pos[j] - cur)); dur = d / (P.v * speed_scale)
        segs.append((t, t + dur, cur.copy(), env.pos[j].astype(float)))
        t += dur
        segs.append((t, t + float(env.tcd[j]), env.pos[j].astype(float),
                     env.pos[j].astype(float)))
        t += float(env.tcd[j]); cur = env.pos[j].astype(float)
    d = float(np.linalg.norm(cur - P.home)); dur = d / (P.v * speed_scale)
    segs.append((t, t + dur, cur.copy(), P.home.astype(float)))
    t_end = t + dur
    for (ts, te, p0, p1) in segs:
        m = (grid >= ts) & (grid < te)
        if not m.any():
            continue
        frac = (grid[m] - ts) / max(te - ts, 1e-9)
        pos[m] = p0[None, :] + frac[:, None] * (p1 - p0)[None, :]
    active = (grid >= offset) & (grid <= t_end)
    return pos, active


def fleet_positions(chains, env, dt, offsets=None, speeds=None):
    K = len(chains)
    offsets = offsets if offsets is not None else [0.0] * K
    speeds  = speeds  if speeds  is not None else [1.0] * K
    T = max((offsets[k] + uav_endtime(chains[k], env, speeds[k]) for k in range(K)), default=0.0)
    if T <= 0:
        return None, None, None
    grid = np.arange(0.0, T + dt, dt)
    POS = np.zeros((K, len(grid), 2)); ACT = np.zeros((K, len(grid)), bool)
    for k in range(K):
        POS[k], ACT[k] = sample_uav_positions(chains[k], env, grid, offsets[k], speeds[k])
    dist_home = np.linalg.norm(POS - P.home[None, None, :], axis=2)
    return POS, ACT, dist_home


def mid_conflict(POS, ACT, dist_home, delta, depot_radius):
    """Return (any_mid, n_pairs, conf_steps, min_sep) for mid-mission conflicts."""
    if POS is None:
        return False, 0, 0, np.inf
    K = POS.shape[0]; any_mid = False; pairs = set(); steps = 0; min_sep = np.inf
    for a in range(K):
        for b in range(a + 1, K):
            both = ACT[a] & ACT[b]
            if not both.any():
                continue
            d_ab = np.linalg.norm(POS[a] - POS[b], axis=1)
            away = (dist_home[a] > depot_radius) & (dist_home[b] > depot_radius)
            mid = both & away & (d_ab < delta)
            if mid.any():
                any_mid = True; pairs.add((a, b)); steps += int(mid.sum())
            sep = d_ab[both & away]
            if sep.size:
                min_sep = min(min_sep, float(sep.min()))
    return any_mid, len(pairs), steps, min_sep


def raw_conflict(POS, ACT, delta):
    if POS is None:
        return False
    K = POS.shape[0]
    for a in range(K):
        for b in range(a + 1, K):
            both = ACT[a] & ACT[b]
            if both.any() and (np.linalg.norm(POS[a] - POS[b], axis=1)[both] < delta).any():
                return True
    return False


def pair_resolvable(chains, env, a, b, delta, depot_radius, dt, taus):
    """Can a relative launch offset tau (UAV b delayed by tau) remove the a-b
    mid-mission conflict, for some tau in `taus`? Returns True if any tau works."""
    for tau in taus:
        offs = [0.0] * len(chains)
        offs[b] = max(0.0, tau); offs[a] = max(0.0, -tau)
        POS, ACT, dh = fleet_positions([chains[a], chains[b]], env, dt,
                                       offsets=[offs[a], offs[b]])
        mid, *_ = mid_conflict(POS, ACT, dh, delta, depot_radius)
        if not mid:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='.')
    ap.add_argument('--M', type=int, nargs='+', default=[100, 200])
    ap.add_argument('--K', type=int, nargs='+', default=[4, 6, 8])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--instances', type=int, default=30)
    ap.add_argument('--deltas', type=float, nargs='+', default=[10, 25, 50, 100])
    ap.add_argument('--ref-delta', type=float, default=25.0,
                    help='safety distance used for the mitigation analysis')
    ap.add_argument('--stagger', type=float, nargs='+', default=[0, 5, 10, 20, 30],
                    help='launch-stagger step Delta (s); o_k = k*Delta')
    ap.add_argument('--jitter', type=float, nargs='+', default=[0, 0.05, 0.10],
                    help='speed jitter fractions; v_k = v*(1 +/- j)')
    ap.add_argument('--jitter-draws', type=int, default=3)
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--depot-radius', type=float, default=50.0)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--out', default='./conflict_probe_report.txt')
    args = ap.parse_args()
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'

    rng = np.random.default_rng(INSTANCE_SEED)
    inst_seeds = [int(rng.integers(0, 10_000_000)) for _ in range(args.instances)]
    jit_rng = np.random.default_rng(INSTANCE_SEED + 1)
    lines = []
    def emit(s=''):
        print(s); lines.append(s)

    emit(f"Spacetime conflict probe v2   device={device}   model-dir={os.path.abspath(args.model_dir)}")
    emit(f"instances={args.instances}  dt={args.dt}s  depot-R={args.depot_radius}m  "
         f"ref-delta={args.ref_delta}m")
    emit("RAW=incl depot stack (fixed by staggering)   MID=mid-mission (needs deconfliction)")
    emit("=" * 96)

    rollouts = {}    # (M,K) -> list of (chains, env)
    for M in args.M:
        for K in args.K:
            if K < 2:
                continue
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
                        cell.append(([list(t) for t in f.trajs], env))
            if cell:
                rollouts[(M, K)] = cell

    # ---------- PHASE 1: baseline delta sweep ----------
    emit("\nPHASE 1 — baseline conflict rate vs safety distance (synchronized launch)\n")
    for (M, K), cell in rollouts.items():
        emit(f"  ===== M={M} K={K} ({len(cell)} rollouts) =====")
        emit(f"    {'delta':>6} {'RAW%':>7} {'MID%':>7} {'median min-sep':>15}")
        for d in args.deltas:
            raw = mid = 0; seps = []
            for chains, env in cell:
                POS, ACT, dh = fleet_positions(chains, env, args.dt)
                raw += int(raw_conflict(POS, ACT, d))
                m, _, _, ms = mid_conflict(POS, ACT, dh, d, args.depot_radius)
                mid += int(m)
                if np.isfinite(ms):
                    seps.append(ms)
            emit(f"    {d:>6.0f} {100*raw/len(cell):>6.0f}% {100*mid/len(cell):>6.0f}% "
                 f"{(np.median(seps) if seps else float('nan')):>14.1f}m")

    # ---------- PHASE 2: mitigations at ref-delta ----------
    rd = args.ref_delta
    emit(f"\nPHASE 2 — mitigations at delta={rd:.0f}m  (does timing fix it?)\n")
    mit_summary = []
    for (M, K), cell in rollouts.items():
        # baseline MID%
        base_mid = 0
        for chains, env in cell:
            POS, ACT, dh = fleet_positions(chains, env, args.dt)
            m, *_ = mid_conflict(POS, ACT, dh, rd, args.depot_radius)
            base_mid += int(m)
        base_pct = 100 * base_mid / len(cell)

        # (a) launch stagger sweep
        stag_pcts = {}
        for Delta in args.stagger:
            mid = 0
            for chains, env in cell:
                offs = [k * Delta for k in range(len(chains))]
                POS, ACT, dh = fleet_positions(chains, env, args.dt, offsets=offs)
                m, *_ = mid_conflict(POS, ACT, dh, rd, args.depot_radius)
                mid += int(m)
            stag_pcts[Delta] = 100 * mid / len(cell)

        # (b) speed jitter sweep (avg over draws)
        jit_pcts = {}
        for j in args.jitter:
            if j == 0:
                jit_pcts[j] = base_pct; continue
            tot = 0; cnt = 0
            for _ in range(args.jitter_draws):
                mid = 0
                for chains, env in cell:
                    speeds = list(1.0 + jit_rng.uniform(-j, j, size=len(chains)))
                    POS, ACT, dh = fleet_positions(chains, env, args.dt, speeds=speeds)
                    m, *_ = mid_conflict(POS, ACT, dh, rd, args.depot_radius)
                    mid += int(m)
                tot += 100 * mid / len(cell); cnt += 1
            jit_pcts[j] = tot / cnt

        # (c) per-pair resolvability ceiling (practical relative offsets)
        taus = [-30, -20, -10, -5, 5, 10, 20, 30]
        n_conf_pairs = 0; n_resolvable = 0
        for chains, env in cell:
            POS, ACT, dh = fleet_positions(chains, env, args.dt)
            Kk = len(chains)
            for a in range(Kk):
                for b in range(a + 1, Kk):
                    both = ACT[a] & ACT[b]
                    if not both.any():
                        continue
                    away = (dh[a] > args.depot_radius) & (dh[b] > args.depot_radius)
                    d_ab = np.linalg.norm(POS[a] - POS[b], axis=1)
                    if (both & away & (d_ab < rd)).any():
                        n_conf_pairs += 1
                        if pair_resolvable(chains, env, a, b, rd, args.depot_radius, args.dt, taus):
                            n_resolvable += 1
        resolv = 100 * n_resolvable / max(n_conf_pairs, 1)

        emit(f"  ===== M={M} K={K} =====   baseline MID%={base_pct:.0f}%")
        emit("    launch-stagger Delta(s): " +
             "  ".join(f"{d:g}->{stag_pcts[d]:.0f}%" for d in args.stagger))
        emit("    speed-jitter j:          " +
             "  ".join(f"{j:g}->{jit_pcts[j]:.0f}%" for j in args.jitter))
        emit(f"    pairwise resolvable by a practical relative offset: "
             f"{n_resolvable}/{n_conf_pairs} = {resolv:.0f}%")
        mit_summary.append((M, K, base_pct, stag_pcts, jit_pcts, resolv))

    # ---------- verdict ----------
    emit("\n" + "=" * 96)
    emit(f"VERDICT (mitigations at delta={rd:.0f}m)")
    best_stag = args.stagger[-1]
    worst_after_stag = max((s[3][best_stag] for s in mit_summary), default=0.0)
    med_resolv = np.median([s[5] for s in mit_summary]) if mit_summary else 0.0
    emit(f"  largest stagger tried Delta={best_stag:g}s: worst residual MID% = {worst_after_stag:.0f}%")
    emit(f"  median pairwise resolvability by a practical offset: {med_resolv:.0f}%")
    emit("")
    if worst_after_stag < 15 and med_resolv > 70:
        emit("  => TIMING-RESOLVABLE. Staggered launches (and/or speed offsets) collapse the")
        emit("     mid-mission conflicts. A temporal-scheduling layer fixes contention WITHOUT")
        emit("     coupling the spatial trajectories — your per-chain separability, the exact B&B,")
        emit("     certified LB and LR estimate all survive. Cheap, rigor-preserving contribution.")
        emit("     (Cost: staggering delays UAV k by k*Delta s, which adds to AoI — evaluate that")
        emit("      objective penalty separately; it is the price of deconfliction.)")
    elif worst_after_stag < 40 or med_resolv > 40:
        emit("  => PARTIALLY TIMING-RESOLVABLE. Timing removes much but not all contention.")
        emit("     A hybrid (stagger + light spatial penalty) is likely the right scope; the fully")
        emit("     coupled model may be more machinery than needed. Inspect the residual cells.")
    else:
        emit("  => GENUINE SPATIAL CONTENTION. Conflicts survive practical staggering and speed")
        emit("     offsets, so they are same-place-same-corridor, not bad timing. The coupled")
        emit("     spatiotemporal model is justified — and it WILL break per-chain separability,")
        emit("     so the optimality-gap machinery must be reformulated, not reused.")
    emit("  (All delta-sensitive; rerun --ref-delta at your true UAV safety radius.)")

    with open(args.out, 'w') as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\n  saved -> {args.out}")


if __name__ == '__main__':
    main()