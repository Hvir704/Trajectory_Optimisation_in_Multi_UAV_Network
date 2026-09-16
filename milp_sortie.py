"""milp_sortie.py -- CP-SAT solver for the per-decision set-orienteering problem (14).

    max  sum_{j in S} v_j          v_j = w~_j * a^_j  (same value as SA / Held-Karp)
    s.t. e_fly(start -> S in order sigma -> home) + sum_{j in S} e_hover(dwell_est_j) <= E_rem

Formulation: one Circuit constraint over {depot} + candidates, optional nodes via self-loop
literals (self-loop true <=> node skipped). Energy is a single linear knapsack over arc and
node literals. Launch-time decisions only (start == home), which is what the paper certifies.

INTEGER SCALING (CP-SAT is integer-only) -- the one place this can silently disagree with
Held-Karp. Arc and hover costs are CEILED to 1 J and the budget FLOORED, so every route
CP-SAT accepts is feasible in real arithmetic (conservative by < n+1 J out of ~1e5 J).
Values are scaled to integers with VALUE_RES relative resolution; the reported objective is
recomputed in floating point from the chosen set. cross_check() compares against
certify_sortie.exact_sortie at small n and must be run before any number is quoted.
"""
from __future__ import annotations
import math, time
from typing import List, Optional, Tuple
import numpy as np
from ortools.sat.python import cp_model

VALUE_RES = 1e6     # integer value units per max(v): relative resolution 1e-6


def candidates(req, value) -> List[int]:
    """Sensors the decision may use: not excluded, positive value, feasible alone."""
    p = req.p
    start = req.home if req.start is None else req.start
    d0 = np.linalg.norm(req.pos - start, axis=1)
    dh = np.linalg.norm(req.pos - req.home, axis=1)
    alone = p.e_fly(d0 + dh) + p.e_hover(req.dwell_est)
    ok = (value > 0) & (alone <= req.E_usable + 1e-6)
    if req.excluded is not None:
        ok &= ~np.asarray(req.excluded, bool)
    return [int(j) for j in np.where(ok)[0]]


def solve_sortie(req, value, time_limit: float = 60.0, workers: int = 8,
                 hint: Optional[List[int]] = None, log: bool = False, params: Optional[dict] = None) -> dict:
    """Returns dict(route, value, bound, status, secs, n_cand). bound is CP-SAT's upper bound
    (float units); status OPTIMAL means value == bound up to integer scaling."""
    if req.start is not None:
        raise ValueError("launch-time decisions only")
    p = req.p
    cand = candidates(req, value)
    n = len(cand)
    t0 = time.time()
    if n == 0:
        return dict(route=[], value=0.0, bound=0.0, status="EMPTY", secs=0.0, n_cand=0)

    P = np.vstack([req.home[None, :], req.pos[cand]])          # node 0 = depot
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    arc_cost = np.ceil(p.Pf * D / p.v).astype(np.int64)        # e_fly, ceiled
    hov = np.ceil(p.e_hover(req.dwell_est[cand])).astype(np.int64)
    E_int = int(math.floor(req.E_usable))
    v = np.asarray(value, float)[cand]
    scale = VALUE_RES / max(v.max(), 1e-12)
    v_int = np.maximum(1, np.round(v * scale)).astype(np.int64)

    m = cp_model.CpModel()
    y = [None] + [m.NewBoolVar(f"y{i}") for i in range(1, n + 1)]
    arcs, x = [], {}
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j:
                continue
            # prune arcs that cannot lie on any feasible route: 0->i->j->0 already too long
            if i and j and arc_cost[0, i] + arc_cost[i, j] + arc_cost[j, 0] + hov[i - 1] + hov[j - 1] > E_int:
                continue
            lit = m.NewBoolVar(f"x{i}_{j}")
            x[i, j] = lit; arcs.append((i, j, lit))
            if i: m.AddImplication(lit, y[i])
            if j: m.AddImplication(lit, y[j])
    for i in range(1, n + 1):
        arcs.append((i, i, y[i].Not()))                         # self-loop <=> skipped
    m.AddCircuit(arcs)
    m.Add(sum(int(arc_cost[i, j]) * lit for (i, j), lit in x.items())
          + sum(int(hov[i - 1]) * y[i] for i in range(1, n + 1)) <= E_int)
    m.Maximize(sum(int(v_int[i - 1]) * y[i] for i in range(1, n + 1)))

    if hint:
        idx = {c: k + 1 for k, c in enumerate(cand)}
        seq = [0] + [idx[j] for j in hint if j in idx] + [0]
        on = set(zip(seq[:-1], seq[1:]))
        for i in range(1, n + 1):
            m.AddHint(y[i], i in set(seq[1:-1]))
        for (i, j), lit in x.items():
            m.AddHint(lit, (i, j) in on)

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_search_workers = workers
    s.parameters.log_search_progress = log
    for k, val in (params or {}).items():
        setattr(s.parameters, k, val)
    st = s.Solve(m)
    secs = time.time() - t0
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(route=None, value=float("nan"), bound=float("nan"),
                    status=s.StatusName(st), secs=secs, n_cand=n)

    # recover the order by walking successor arcs from the depot
    succ = {i: j for (i, j), lit in x.items() if s.Value(lit)}
    route, cur = [], succ.get(0)
    while cur not in (None, 0):
        route.append(cand[cur - 1]); cur = succ.get(cur)
    val = float(sum(value[j] for j in route))
    bound = float(s.BestObjectiveBound()) / scale
    return dict(route=route, value=val, bound=bound, status=s.StatusName(st), secs=secs, n_cand=n)


def route_energy(req, route) -> float:
    p = req.p; cur = req.home; E = 0.0
    for j in route:
        E += p.e_fly(float(np.linalg.norm(req.pos[j] - cur))) + p.e_hover(req.dwell_est[j]); cur = req.pos[j]
    return E + p.e_fly(float(np.linalg.norm(cur - req.home)))


CPSAT_WORKERS = 4   # part of the algorithm's definition under interleave_search: keep fixed


def build_milp_planner(dtime: float = 5.0, workers: int = CPSAT_WORKERS, stats=None):
    """Independent-optimiser baseline: CP-SAT, COLD start (no SA hint), deterministic.

    Reproducibility: interleave_search + max_deterministic_time make the result a function of
    the input only, not of wall-clock or machine load -- a wall-clock limit would let parallel
    grid runs on a busy machine silently get less search than a quiet one.
    If CP-SAT returns no solution the greedy ratio rule is used and the fallback is COUNTED in
    stats (planner.stats) -- never silently an empty sortie."""
    from dyn_env import greedy_ratio_planner
    st = stats if stats is not None else {"n": 0, "optimal": 0, "fallback": 0, "gap_sum": 0.0}
    prm = {"interleave_search": True, "max_deterministic_time": float(dtime)}

    def planner(req):
        value = req.weight_est * req.age
        if req.p_live is not None:
            value = value * (1 + (req.p.event_gain - 1) * req.p_live)
        st["n"] += 1
        if value.max() <= 0:                  # t = 0 launch: every age is 0, nothing has value
            st["optimal"] += 1                # (any route is optimal); match SA/greedy, which still fly
            return greedy_ratio_planner(req)
        r = solve_sortie(req, value, time_limit=3600.0, workers=workers, params=prm)
        if r["route"] is None:
            st["fallback"] += 1
            return greedy_ratio_planner(req)
        if r["status"] in ("OPTIMAL", "EMPTY"):
            st["optimal"] += 1
        elif r["bound"] > 0:
            st["gap_sum"] += (r["bound"] - r["value"]) / r["bound"]
        return r["route"]
    planner.stats = st
    return planner
