"""rr_planner.py -- age-blind patrol baselines for the external-baseline comparison.

"What an operator would deploy without this paper": a fixed cyclic order over all sensors,
cut into sorties by the energy budget. Classical route-first / cluster-second.

  rr_tour  : cyclic order = one closed tour through depot + all sensors
             (nearest-neighbour then 2-opt), depot removed. Beasley (1983) style.
  rr_sweep : cyclic order = polar angle about the depot. Gillett-Miller (1974) sweep.

Selection ignores age AND priority entirely. The only use of the age vector is through
req.dwell_est, i.e. hover-energy BUDGETING (feasibility), which every planner must do.

Fleet semantics. One shared pointer per simulation (the closure is built fresh per DynSim,
exactly like build_sa_planner). A launching drone takes the next contiguous run of sensors
from the pointer while (fly-from-current + hover + fly-home) still fits, and stops at the
first sensor that does not fit; that sensor starts the next sortie. Two kinds of sensor are
SKIPPED rather than stopping the sortie:
  * excluded  -- committed to another airborne drone (coordination honoured; only bites when
                 the pointer laps pending commitments, i.e. small M / large K)
  * unreachable alone -- start->i->home + hover > budget. These are the stranded sensors;
                 without the skip the pointer would stall on them forever at K > K_reach.
At most one lap per decision, so every sensor is considered at most once.

Launch-time only: a route is a segment of a fixed cycle, so per-leg replanning is undefined.
"""
from __future__ import annotations
from typing import List
import numpy as np
from dyn_env import SortieRequest


def _two_opt(order: np.ndarray, P: np.ndarray, max_pass: int = 50) -> np.ndarray:
    """Closed-tour 2-opt, first-improvement per i, vectorised over j. order[0] is kept fixed."""
    n = len(order)
    if n < 4:
        return order
    order = order.copy()
    for _ in range(max_pass):
        improved = False
        for i in range(0, n - 2):
            a, b = P[order[i]], P[order[i + 1]]
            js = np.arange(i + 2, n if i > 0 else n - 1)
            if len(js) == 0:
                continue
            c = P[order[js]]; d = P[order[(js + 1) % n]]
            delta = (np.linalg.norm(a - c, axis=1) + np.linalg.norm(b - d, axis=1)
                     - np.linalg.norm(a - b) - np.linalg.norm(c - d, axis=1))
            k = int(np.argmin(delta))
            if delta[k] < -1e-9:
                j = int(js[k])
                order[i + 1:j + 1] = order[i + 1:j + 1][::-1]
                improved = True
        if not improved:
            break
    return order


def tour_order(pos: np.ndarray, home: np.ndarray) -> np.ndarray:
    """Cyclic sensor order from a closed depot tour (NN + 2-opt). Deterministic in pos."""
    P = np.vstack([home[None, :], pos])            # node 0 = depot
    n = len(P)
    left = np.ones(n, dtype=bool); left[0] = False
    order = [0]; cur = 0
    for _ in range(n - 1):
        d = np.linalg.norm(P - P[cur], axis=1); d[~left] = np.inf
        cur = int(np.argmin(d)); order.append(cur); left[cur] = False
    order = _two_opt(np.array(order), P)
    order = np.roll(order, -int(np.where(order == 0)[0][0]))   # depot first
    return order[1:] - 1                                         # drop depot, back to sensor ids


def sweep_order(pos: np.ndarray, home: np.ndarray) -> np.ndarray:
    th = np.arctan2(pos[:, 1] - home[1], pos[:, 0] - home[0])
    r = np.linalg.norm(pos - home, axis=1)
    return np.lexsort((r, th))                                   # angle, then radius


def build_rr_planner(kind: str = "tour"):
    """Factory -> SortiePlanner. Build fresh per DynSim (the pointer is simulation state)."""
    st = {"order": None, "ptr": 0, "calls": 0}

    def planner(req: SortieRequest) -> List[int]:
        if req.start is not None:
            raise ValueError("round-robin baseline is launch-time only (use --replan launch)")
        p = req.p
        if st["order"] is None:
            st["order"] = tour_order(req.pos, req.home) if kind == "tour" else sweep_order(req.pos, req.home)
        order = st["order"]; M = len(order)
        hov = p.e_hover(req.dwell_est)
        d_home = np.linalg.norm(req.pos - req.home, axis=1)
        alone = p.e_fly(2.0 * d_home) + hov                      # start == home at launch
        excl = np.zeros(M, dtype=bool) if req.excluded is None else np.asarray(req.excluded, bool)

        route: List[int] = []
        cur = req.home; E = req.E_usable
        ptr = st["ptr"]; stop_at = None
        for step in range(M):
            j = int(order[(ptr + step) % M])
            if excl[j] or alone[j] > req.E_usable:          # skip, do not stop
                continue
            e_leg = p.e_fly(float(np.linalg.norm(req.pos[j] - cur))) + hov[j]
            if e_leg + p.e_fly(d_home[j]) <= E + 1e-6:
                route.append(j); E -= e_leg; cur = req.pos[j]
            else:
                stop_at = (ptr + step) % M                       # first misfit opens next sortie
                break
        if stop_at is None:                                      # lapped: resume after last taken
            if route:
                stop_at = (int(np.where(order == route[-1])[0][0]) + 1) % M
            else:
                stop_at = ptr
        st["ptr"] = stop_at; st["calls"] += 1
        return route

    planner.state = st
    return planner
