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


def _two_opt_chain(env: Env, traj: list, budget: float) -> list:
    """
    2-opt within a single UAV's chain (objective uses that chain's WAoI).
    A reordering is accepted only if it improves the objective AND keeps the
    chain within `budget` joules — 2-opt optimises objective, not energy, so an
    unguarded reversal can push a chain over its (split) budget.
    """
    if len(traj) < 3:
        return traj
    best = list(traj); improved = True
    while improved:
        improved = False
        best_obj = env.objective(best)          # single-chain objective
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                cand = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                if env.objective(cand) < best_obj - 1e-6 and \
                   _chain_energy(env, cand) <= budget:
                    best = cand; best_obj = env.objective(cand); improved = True; break
            if improved:
                break
    return best


def fleet_post_process(env: Env, fleet: FleetState) -> FleetState:
    """
    Test-time fleet improvement (no retraining):
      1. 2-opt each UAV's chain independently (fixes crossings per chain).
      2. Cross-UAV cheapest insertion: greedily insert any GLOBALLY unvisited
         node into whichever (UAV, position) yields the best feasible gain.
    The partition constraint is preserved because a node is inserted into
    exactly one chain and immediately marked globally visited.
    """
    # 1. per-chain 2-opt (energy-guarded against this UAV's budget)
    for k in range(fleet.K):
        fleet.trajs[k] = _two_opt_chain(env, fleet.trajs[k], fleet.Emax_each)

    # 2. cross-UAV insertion of leftover nodes
    served = set(j for t in fleet.trajs for j in t)
    changed = True
    while changed:
        changed = False
        best_gain = 1e-6
        best_move = None    # (k, new_chain, j)
        for j in range(env.M):
            if j in served:
                continue
            for k in range(fleet.K):
                base_obj = env.objective(fleet.trajs[k])
                for pos in range(len(fleet.trajs[k]) + 1):
                    cand = fleet.trajs[k][:pos] + [j] + fleet.trajs[k][pos:]
                    # Feasibility: inserted chain must fit THIS UAV's budget
                    # (split or full — read from the FleetState, never hardcoded).
                    if _chain_energy(env, cand) > fleet.Emax_each:
                        continue
                    gain = base_obj - env.objective(cand)
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (k, cand, j)
        if best_move is not None:
            k, cand, j = best_move
            fleet.trajs[k] = cand
            served.add(j)
            fleet.visited[j] = True
            changed = True

    return fleet
