"""
fleet_beam.py  —  Beam-search rollout for the K-UAV fleet
================================================================================
Direct port of the single-UAV `rollout_beam` (uav_aoi_solver.py) to the fleet.
It is a TEST-TIME search on the FROZEN policy: it never touches fleet_rollout,
fleet_objective, the reward, or training. Separability and every bound / gap
therefore stay exactly valid — this only changes the inference-time trajectory
the frozen policy produces, the same category of change as beam vs greedy in the
single-UAV paper.

WHY THIS IS A CLEAN PORT
------------------------
The greedy fleet rollout is sequential-commit + load-balanced: at each macro-step
it (1) picks the ACTIVE UAV with the most remaining energy, (2) argmaxes one node
for that UAV, (3) commits it. The UAV choice is a deterministic function of the
state; only the NODE choice is a policy decision. So beam search branches over the
node choice for that deterministically-selected UAV — structurally identical to the
single-UAV beam, but each beam entry is a whole FleetState instead of one trajectory.

We reuse FleetState.commit / feasible_mask_k / make_features_multi verbatim, so the
per-step scoring, feasibility, split-battery energy check, and partition mask are
byte-for-byte the same as greedy. Consequences:

  * beam_width == 1  reproduces fleet_rollout(greedy=True) EXACTLY   (regression test below)
  * ranking is by cumulative log-prob (exactly like rollout_beam); the final pick
    among surviving + completed beams is by best fleet_objective (also like rollout_beam)

ONE DELIBERATE, DOCUMENTED DEVIATION FROM rollout_beam
------------------------------------------------------
Log-prob-guided beam is NOT guaranteed to contain the greedy trajectory (global
top-B pruning can drop the greedy path if B other partials out-score it on
cumulative log-prob). So a pure port can, on some instances, return a WORSE
objective than greedy — true of the single-UAV rollout_beam too. To make the
"beam improves the raw-policy number" story monotone and reviewer-safe, we add the
greedy rollout as a guaranteed final candidate (include_greedy=True, default). Then

        fleet_objective(beam)  <=  fleet_objective(greedy)   always.

Set include_greedy=False for a pure-port comparison identical in spirit to the
single-UAV version.

API
---
    fleet_rollout_beam(policy, env, K, device,
                       Emax_each=MP.Emax_each, beam_width=5,
                       include_greedy=True)  ->  FleetState

Drop-in wherever fleet_rollout(...) is used at test time. Follow with
fleet_post_process(env, f) exactly as with greedy.
"""
from __future__ import annotations
import numpy as np
import torch

from uav_aoi_solver import P, Env
from multi_uav_solver import (MP, FleetState, MultiUAVPolicy,
                              make_features_multi, fleet_rollout)


# ── FleetState cloning (shares the read-only Env; copies all mutable state) ────
def _clone_fleet(f: FleetState) -> FleetState:
    g = FleetState.__new__(FleetState)
    g.env       = f.env                       # read-only during rollout -> share
    g.K         = f.K
    g.M         = f.M
    g.Emax_each = f.Emax_each
    g.pos       = [p.copy() for p in f.pos]
    g.E_left    = list(f.E_left)
    g.W_cum     = list(f.W_cum)
    g.trajs     = [t[:] for t in f.trajs]
    g.active    = list(f.active)
    g.visited   = f.visited.copy()
    return g


# ── advance a state to its next actionable UAV, retiring stuck ones in place ───
def _next_actionable_uav(f: FleetState):
    """Mirror the greedy loop's UAV selection: repeatedly take the most-energy
    active UAV; if it cannot reach any unvisited node and return home, retire it
    and try the next. Returns the UAV index to act, or None if the state is
    terminal (all UAVs retired). Mutates f.active for retired UAVs — this is the
    same state change greedy makes, and here f is unique to one beam entry."""
    while True:
        active_ids = [j for j in range(f.K) if f.active[j]]
        if not active_ids:
            return None
        k = max(active_ids, key=lambda j: f.E_left[j])   # most-energy, tie -> lowest idx
        if f.feasible_mask_k(k).any():
            return k
        f.retire(k)


def fleet_rollout_beam(policy: MultiUAVPolicy, env: Env, K: int, device: str,
                       Emax_each: float = MP.Emax_each, beam_width: int = 5,
                       include_greedy: bool = True) -> FleetState:
    """Beam-search fleet rollout. See module docstring."""
    policy.eval()
    beams = [(0.0, FleetState(env, K, Emax_each))]   # (cum_logprob, FleetState)
    completed = []                                    # terminal FleetStates

    with torch.no_grad():
        # macro-step cap: at most M commits + K retirements
        for _ in range(env.M + K + 1):
            if not beams:
                break
            candidates = []
            all_terminal = True

            for lp_sum, f in beams:
                k = _next_actionable_uav(f)          # retires stuck UAVs in-place
                if k is None:
                    completed.append(f)              # state fully done
                    continue
                all_terminal = False

                x, mask, feasible = make_features_multi(f, k, device)
                log_p, _ = policy(x, mask)
                lp_np = log_p.cpu().numpy()

                feasible_idx = np.where(feasible)[0]
                lp_feasible  = lp_np[feasible_idx]
                # top beam_width feasible nodes for THIS UAV (same as rollout_beam)
                top_local = np.argsort(lp_feasible)[-beam_width:]

                for ki in top_local:
                    j = int(feasible_idx[ki])
                    g = _clone_fleet(f)
                    g.commit(k, j)                   # reuse the exact greedy commit
                    candidates.append((lp_sum + float(lp_np[j]), g))

            if all_terminal:
                break

            candidates.sort(key=lambda b: -b[0])     # keep global top-B by cum logprob
            beams = candidates[:beam_width]

    final_states = [f for _, f in beams] + completed
    if include_greedy:
        final_states.append(fleet_rollout(policy, env, K, device,
                                           Emax_each=Emax_each, greedy=True))
    if not final_states:
        return FleetState(env, K, Emax_each)         # nothing feasible: empty fleet

    return min(final_states, key=lambda f: f.fleet_objective())


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST  (no trained models needed; a random-init policy is deterministic)
#     python fleet_beam.py
# ══════════════════════════════════════════════════════════════════════════════
def _self_test():
    torch.manual_seed(0)
    device = 'cpu'
    pol = MultiUAVPolicy(hidden=256, input_dim=18).to(device)
    pol.eval()

    print("== regression: beam_width=1 must reproduce greedy exactly ==")
    ok = True
    for (M, K, sd) in [(30, 2, 1), (30, 3, 2), (50, 4, 3), (40, 5, 7), (60, 3, 11)]:
        env = Env(M=M, seed=sd)
        Ee  = P.Emax / K                                  # split battery, as in the sweep
        g  = fleet_rollout(pol, env, K, device, Emax_each=Ee, greedy=True)
        b1 = fleet_rollout_beam(pol, env, K, device, Emax_each=Ee,
                                beam_width=1, include_greedy=False)
        same = g.trajs == b1.trajs
        ok &= same
        print(f"   M={M:>3} K={K}  greedy.trajs == beam1.trajs : {same}"
              f"   (obj g={g.fleet_objective():+.3f}  b1={b1.fleet_objective():+.3f})")
    assert ok, "beam_width=1 diverged from greedy — port is not faithful!"
    print("   PASS: width-1 beam == greedy on every case\n")

    print("== invariants + monotonicity (include_greedy guarantees beam <= greedy) ==")
    for (M, K, sd) in [(50, 3, 5), (80, 4, 9), (60, 5, 13)]:
        env = Env(M=M, seed=sd)
        Ee  = P.Emax / K
        g = fleet_rollout(pol, env, K, device, Emax_each=Ee, greedy=True)
        for B in (3, 5, 8):
            b = fleet_rollout_beam(pol, env, K, device, Emax_each=Ee,
                                   beam_width=B, include_greedy=True)
            # partition: no node served twice, across all chains
            served = [j for t in b.trajs for j in t]
            assert len(served) == len(set(served)), "partition violated (dup node)!"
            # energy: each chain must respect its own split budget (round-trip)
            from multi_uav_solver import _chain_energy
            for t in b.trajs:
                assert _chain_energy(env, t) <= Ee + 1e-6, "energy budget violated!"
            # monotonicity
            assert b.fleet_objective() <= g.fleet_objective() + 1e-9, \
                "include_greedy broken: beam worse than greedy!"
        print(f"   M={M:>3} K={K}  partition OK, energy OK, beam<=greedy OK"
              f"   (greedy {g.fleet_objective():+.2f} -> beam8 "
              f"{fleet_rollout_beam(pol, env, K, device, Emax_each=Ee, beam_width=8).fleet_objective():+.2f})")
    print("   PASS: all structural invariants hold\n")
    print("SELF-TEST PASSED — note the objective numbers above are from a RANDOM")
    print("policy and are meaningless; only the equalities/invariants matter here.")


if __name__ == '__main__':
    _self_test()
