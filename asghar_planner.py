"""asghar_planner.py -- external baseline: Asghar, Sundaram, Smith, "Multirobot Persistent Monitoring:
Minimizing Latency and Number of Robots With Recharging Constraints", IEEE Trans. Robotics 41:236-252,
2025, doi 10.1109/TRO.2024.3502497. Implemented from the PUBLISHED version (Xplore), not arXiv v1.

Problem 13 of the paper: R robots, vertex weights phi(v), recharging depot mu, discharge limit D;
minimise the maximum weighted latency  max_v phi(v) * L(W, v), where L is the longest time between
consecutive visits. With phi = our priority w_i this is the PEAK form of our priority-weighted
collection-side age (time since last visit). Algorithm 4 (LatencyWalks) solves it for fixed R:
  * vertices are partitioned into weight classes V_i: 1/2^i < phi <= 1/2^(i-1), i = 1..ceil(log2 rho)
    (phi normalised to max 1; rho = max/min, +1 if an exact power of 2);
  * R <  log rho : robot j gets classes ceil((j-1)/R log rho)+1 .. ceil(j/R log rho) and follows the
                   single-robot walk of Algorithm 3 on them (heavier classes visited more often:
                   class i's cycle cover is split into 2^(i-1) sub-walks interleaved over t sub-walks);
  * R >= log rho : floor(R / ceil(log rho)) robots equally spaced on each class's RMCCP solution;
                   remaining robots added one at a time to the class with the highest cost.
RMCCP (rooted minimum cycle cover, cycles of length <= D) is solved by the paper's own method: greedy
orienteering (repeatedly take the depot-rooted cycle covering the most uncovered vertices within D).

ADAPTATION (disclosed; GAPS dict below is paper-ready):
  * D (a time) -> our per-sortie energy E_usable = (1-rho_res) U / K, with flight power Pf and hover
    power Ph. Latency, cost and walk lengths are in TIME (flight + dwell), as in the paper.
  * Vertex inspection time = hover dwell, folded into the plan exactly as the paper's
    l'(v,u) = l(v,u) + I(v)/2 + I(u)/2 does (we add it per visited vertex, which is equivalent).
  * One cycle = one sortie. A robot follows its walk (a cyclic list of depot-rooted cycles) one cycle
    per sortie. "Equally spacing" m robots on a class walk is emulated by those robots taking that
    walk's cycles round-robin from a shared pointer (same visit frequency; launches are staggered by
    the simulator, exact lags are not enforced).
  * Offline plan, built once per simulation from planner-side information only: positions, depot,
    priorities, nominal generation rate, buffer, link rate, E_usable. No ages, no true rates.
  * At launch the cycle is checked against the live dwell estimate; sensors excluded (pending in
    another airborne drone) or unreachable alone are skipped, and the cycle is truncated if energy
    runs short (counted in state['trunc_nodes']).
  * Sensors that cannot be reached alone within E_usable are dropped from the plan (the paper's
    Assumption 3, D >= 2 max l(mu, v), fails for them; every other planner also strands them).
Launch-time only.
"""
from __future__ import annotations
import math
import time
from typing import List
import numpy as np
from dyn_env import SortieRequest

GAPS = {
    "class_indexing": "classes numbered from 1 as in Alg. 4 line 4 (Alg. 3's box loops from i=0): class i split "
                      "into 2^(i-1) sub-walks; t = 2^(c-1) (box: 2^ceil(log rho), prose: 2^(ceil(log rho)+1)), "
                      "the smallest t that uses every sub-walk",
    "subgraph_weights": "Alg. 3 on robot j's subgraph G_j uses the global classes of G_j relabelled from 1 "
                        "(equivalent to normalising by the class upper bound)",
    "split_into_2^i": "contiguous groups of whole cycles, balanced by duration (not specified in the paper)",
    "rmccp": "greedy orienteering (unit scores) as in the paper; each orienteering instance solved by CP-SAT "
             "(OR-Tools) instead of Gurobi, single worker, deterministic time budget; greedy-insertion "
             "solution as hint and fallback",
    "leftover_cost": "cost of class i = max phi in class x walk duration / robots on the walk",
    "empty_classes": "robots that Alg. 4 line 10 would place on an empty class are treated as leftover robots "
                     "(literal reading leaves them idle)",
    "D": "energy budget E_usable with Pf/Ph, not a time limit; recharge time 0 (immediate relaunch, as in the paper)",
    "equal_spacing": "shared round-robin pointer over the class walk's cycles",
    "dwell": "nominal dwell for planning: 'full' = B/R or 'rev' = min(lam x planned revisit, B)/R (one fixed-point "
             "pass); chosen on pilot seeds",
}


# ---------------------------------------------------------------------------- geometry / energy helpers
def _cycle_energy(cyc, X, home, dwell, p):
    if not cyc: return 0.0
    P = np.vstack([home, X[cyc], home])
    return p.e_fly(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) + p.e_hover(dwell[cyc]).sum()


def _cycle_time(cyc, X, home, dwell, v):
    if not cyc: return 0.0
    P = np.vstack([home, X[cyc], home])
    return np.linalg.norm(np.diff(P, axis=0), axis=1).sum() / v + dwell[cyc].sum()


def _two_opt_open(seq, X, home):
    """2-opt on a depot-rooted cycle (depot fixed at both ends). Shortens flight energy only."""
    seq = list(seq)
    if len(seq) < 3: return seq
    P = lambda s: np.vstack([home, X[s], home])
    improved = True
    while improved:
        improved = False
        Q = P(seq)
        n = len(Q)
        for i in range(0, n - 3):
            for j in range(i + 2, n - 1):
                a, b, c, d = Q[i], Q[i + 1], Q[j], Q[j + 1]
                if np.linalg.norm(a - c) + np.linalg.norm(b - d) < np.linalg.norm(a - b) + np.linalg.norm(c - d) - 1e-9:
                    seq[i:j] = seq[i:j][::-1]
                    improved = True
                    break
            if improved: break
    return seq


def _greedy_insertion(cand, X, home, dwell, p, E):
    """Cheapest-insertion orienteering with unit scores: hint and fallback for CP-SAT."""
    cyc: List[int] = []
    left = set(cand)
    while left:
        best = None
        P = np.vstack([home, X[cyc], home]) if cyc else np.vstack([home, home])
        base = _cycle_energy(cyc, X, home, dwell, p)
        for j in left:
            for at in range(len(P) - 1):
                a, b = P[at], P[at + 1]
                de = p.e_fly(np.linalg.norm(a - X[j]) + np.linalg.norm(X[j] - b) - np.linalg.norm(a - b)) + p.e_hover(dwell[j])
                if base + de <= E + 1e-6 and (best is None or de < best[0]):
                    best = (de, j, at)
        if best is None: break
        _, j, at = best
        cyc.insert(at, j); left.discard(j)
    return _two_opt_open(cyc, X, home)


def orienteering_cpsat(cand, X, home, dwell, p, E, time_s=5.0, seed=0):
    """Depot-rooted cycle within energy E maximising the number of vertices visited (unit scores),
    ties broken by lower energy. Returns (cycle, status_name)."""
    from ortools.sat.python import cp_model
    cand = list(cand)
    hint = _greedy_insertion(cand, X, home, dwell, p, E)
    n = len(cand)
    if n == 0: return [], "EMPTY"
    if n == 1: return hint, "TRIVIAL"
    pts = np.vstack([home, X[cand]])
    D = np.linalg.norm(pts[:, None, :] - pts[None], axis=-1)
    efly = np.rint(p.e_fly(D)).astype(np.int64)
    ehov = np.rint(p.e_hover(dwell[cand])).astype(np.int64)
    m = cp_model.CpModel()
    arcs, lit = [], {}
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                x = m.NewBoolVar(""); lit[i, j] = x; arcs.append((i, j, x))
    skip = [None] + [m.NewBoolVar("") for _ in range(n)]
    for j in range(1, n + 1):
        arcs.append((j, j, skip[j]))
    m.AddCircuit(arcs)
    energy = sum(int(efly[i, j]) * x for (i, j), x in lit.items()) + \
             sum(int(ehov[j - 1]) * (1 - skip[j]) for j in range(1, n + 1))
    m.Add(energy <= int(math.floor(E)))
    BIG = int(efly.max() * (n + 2) + ehov.sum() + 1)
    m.Maximize(BIG * sum(1 - skip[j] for j in range(1, n + 1)) - energy)
    # hint: the greedy cycle
    on = {cand.index(j) + 1 for j in hint}
    order = [0] + [cand.index(j) + 1 for j in hint] + [0]
    used = {(order[k], order[k + 1]) for k in range(len(order) - 1)} if hint else set()
    for (i, j), x in lit.items(): m.AddHint(x, 1 if (i, j) in used else 0)
    for j in range(1, n + 1): m.AddHint(skip[j], 0 if j in on else 1)
    s = cp_model.CpSolver()
    s.parameters.num_workers = 1
    s.parameters.max_deterministic_time = float(time_s)
    s.parameters.random_seed = int(seed) % (2**31 - 1)
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return hint, "FALLBACK"
    succ = {i: j for (i, j), x in lit.items() if s.Value(x)}
    cyc, cur = [], succ.get(0)
    while cur is not None and cur != 0:
        cyc.append(cand[cur - 1]); cur = succ.get(cur)
    if len(cyc) < len(hint):          # never worse than the hint in the objective's first term
        return hint, "HINT_BETTER"
    return _two_opt_open(cyc, X, home), s.StatusName(st)


def rmccp(cand, X, home, dwell, p, E, time_s=5.0, seed=0, stats=None):
    """Greedy orienteering cover: cycles rooted at home, each within energy E, covering cand."""
    left = list(cand); cycles = []
    while left:
        cyc, st = orienteering_cpsat(left, X, home, dwell, p, E, time_s=time_s, seed=seed + len(cycles))
        if stats is not None: stats[st] = stats.get(st, 0) + 1
        if not cyc:                       # cannot happen for cand reachable alone; guard anyway
            break
        cycles.append(cyc)
        cs = set(cyc); left = [j for j in left if j not in cs]
    return cycles


# ---------------------------------------------------------------------------- Algorithms 3 and 4
def weight_classes(phi):
    """Alg. 4 lines 1-4: returns (list of index arrays V_1..V_c, log2 rho, c)."""
    phi = np.asarray(phi, float) / np.max(phi)
    rho = phi.max() / phi.min()
    if abs(math.log2(rho) - round(math.log2(rho))) < 1e-12: rho += 1.0      # 'is a power of 2'
    lr = math.log2(rho); c = max(1, math.ceil(lr))
    cls = [np.where((phi > 2.0 ** -i) & (phi <= 2.0 ** -(i - 1)))[0] for i in range(1, c + 1)]
    return cls, lr, c


def _split_balanced(cycles, parts, X, home, dwell, v):
    """Partition the cycle list into `parts` contiguous groups of whole cycles, balanced by duration."""
    if parts <= 1: return [list(cycles)]
    dur = [_cycle_time(c, X, home, dwell, v) for c in cycles]
    tot = sum(dur); groups = [[] for _ in range(parts)]; g = 0; acc = 0.0
    for c, d in zip(cycles, dur):
        if g < parts - 1 and acc >= tot * (g + 1) / parts:
            g += 1
        groups[g].append(c); acc += d
    return groups


def min_max_latency_one_robot(class_cycles, X, home, dwell, v):
    """Alg. 3 on pre-computed per-class RMCCP cycles (class_cycles[0] = heaviest class of G_j).
    Returns the walk S as a list of cycles."""
    c = len(class_cycles)
    t = 2 ** (c - 1)
    sub = [_split_balanced(cc, 2 ** i, X, home, dwell, v) for i, cc in enumerate(class_cycles)]
    S = []
    for k in range(1, t + 1):
        for i in range(c):
            S.extend(cyc for cyc in sub[i][(k - 1) % (2 ** i)] if cyc)
    return S


def latency_walks(phi, X, home, dwell, p, R, E, time_s=5.0, seed=0, stats=None):
    """Alg. 4. Returns (streams, robot_stream): streams = list of walks (each a list of cycles);
    robot_stream[r] = index of the walk robot r follows (robots sharing a stream are 'equally spaced')."""
    cls, lr, c = weight_classes(phi)
    cyc_of = [rmccp(list(Vi), X, home, dwell, p, E, time_s, seed + 1000 * i, stats) for i, Vi in enumerate(cls)]
    if R < lr:                                        # line 5
        streams, robot_stream = [], []
        for j in range(1, R + 1):
            lo = math.ceil((j - 1) / R * lr) + 1; hi = math.ceil(j / R * lr)
            mine = [cyc_of[i - 1] for i in range(lo, hi + 1) if 1 <= i <= c]   # empty classes kept (no cycles)
            streams.append(min_max_latency_one_robot(mine, X, home, dwell, p.v) if any(mine) else [])
            robot_stream.append(j - 1)
        return streams, robot_stream, dict(branch="R<log rho", classes=[len(v_) for v_ in cls], log2rho=lr)
    # line 9: R >= log rho
    per = R // c
    streams = [list(cc) for cc in cyc_of]
    count = [per if cyc_of[i] else 0 for i in range(c)]
    leftover = R - sum(count)
    period = [sum(_cycle_time(cy, X, home, dwell, p.v) for cy in cc) for cc in cyc_of]
    wmax = [float(np.max(np.asarray(phi)[cls[i]] / np.max(phi))) if len(cls[i]) else 0.0 for i in range(c)]
    for _ in range(leftover):                         # lines 11-13
        cost = [wmax[i] * period[i] / count[i] if count[i] else (wmax[i] * period[i] if cyc_of[i] else -1.0)
                for i in range(c)]
        i = int(np.argmax(cost)); count[i] += 1
    robot_stream = [i for i in range(c) for _ in range(count[i])]
    return streams, robot_stream, dict(branch="R>=log rho", classes=[len(v_) for v_ in cls], log2rho=lr, robots=count)


def _revisit_times(streams, robot_stream, X, home, dwell, v, M):
    """Planned per-node revisit interval: stream period / (occurrences x robots on the stream)."""
    T = np.full(M, np.nan)
    m = np.bincount(robot_stream, minlength=len(streams))
    for s, walk in enumerate(streams):
        if not walk or m[s] == 0: continue
        per = sum(_cycle_time(cy, X, home, dwell, v) for cy in walk)
        occ = {}
        for cy in walk:
            for j in cy: occ[j] = occ.get(j, 0) + 1
        for j, o in occ.items():
            T[j] = per / (o * m[s])
    return T


def build_plan(pos, home, phi, p, K, seed=0, dwell_mode="rev", time_s=5.0):
    t0 = time.time(); stats = {}
    E = p.E_usable
    full = np.full(len(pos), p.B_bits / p.R)
    d_home = np.linalg.norm(pos - home, axis=1)
    reach = (p.e_fly(2 * d_home) + p.e_hover(full)) <= E + 1e-6
    idx = np.where(reach)[0]                          # dropped: unreachable alone even with a full buffer
    if len(idx) == 0:
        return [[] for _ in range(K)], list(range(K)), dict(build_s=0.0, dropped=len(pos))
    X = pos[idx]; ph = np.asarray(phi)[idx]
    dwell = full[idx].copy()
    passes = 1 if dwell_mode == "full" else 2
    for ps in range(passes):
        streams, rs, info = latency_walks(ph, X, home, dwell, p, K, E, time_s, seed * 7919 + ps, stats)
        if dwell_mode == "rev" and ps == 0:
            T = _revisit_times(streams, rs, X, home, dwell, p.v, len(idx))
            T = np.where(np.isnan(T), np.nanmax(T) if np.any(~np.isnan(T)) else 0.0, T)
            dwell = np.minimum(p.lam_bits * T, p.B_bits) / p.R
    streams = [[[int(idx[j]) for j in cy] for cy in walk] for walk in streams]
    info.update(build_s=time.time() - t0, dropped=int((~reach).sum()), cpsat=stats,
                n_cycles=[len(w) for w in streams])
    return streams, rs, info


# ---------------------------------------------------------------------------- planner
def build_asghar_planner(dwell_mode="rev", seed=0, time_s=2.0):
    """Factory -> SortiePlanner. Build fresh per DynSim; run_grid calls planner.bind_sim(sim)."""
    st = {"streams": None, "robot_stream": None, "ptr": None, "k": 0, "K": None, "info": None,
          "calls": 0, "trunc_sorties": 0, "trunc_nodes": 0, "skipped": 0, "dwell_mode": dwell_mode}

    def bind_sim(sim):
        orig = sim._plan
        def _plan(k, *a, **kw):
            st["k"] = k
            return orig(k, *a, **kw)
        sim._plan = _plan
        st["K"] = sim.p.K

    def planner(req: SortieRequest) -> List[int]:
        if req.start is not None:
            raise ValueError("asghar baseline is launch-time only (use --replan launch)")
        p = req.p
        if st["K"] is None:
            raise RuntimeError("asghar planner needs bind_sim(sim) before the first launch")
        if st["streams"] is None:
            st["streams"], st["robot_stream"], st["info"] = build_plan(
                req.pos, req.home, req.weight_est, p, st["K"], seed, dwell_mode, time_s)
            st["ptr"] = [0] * len(st["streams"])
        st["calls"] += 1
        s = st["robot_stream"][st["k"] % st["K"]]
        walk = st["streams"][s]
        if not walk:
            return []
        cyc = walk[st["ptr"][s]]; st["ptr"][s] = (st["ptr"][s] + 1) % len(walk)
        hov = p.e_hover(req.dwell_est)
        d_home = np.linalg.norm(req.pos - req.home, axis=1)
        alone = p.e_fly(2.0 * d_home) + hov
        excl = np.zeros(len(req.pos), bool) if req.excluded is None else np.asarray(req.excluded, bool)
        route, cur, E = [], req.home, req.E_usable
        for i, j in enumerate(cyc):
            if excl[j] or alone[j] > req.E_usable:
                st["skipped"] += 1; continue
            e_leg = p.e_fly(float(np.linalg.norm(req.pos[j] - cur))) + hov[j]
            if e_leg + p.e_fly(d_home[j]) <= E + 1e-6:
                route.append(j); E -= e_leg; cur = req.pos[j]
            else:
                st["trunc_sorties"] += 1; st["trunc_nodes"] += len(cyc) - i; break
        return route

    planner.bind_sim = bind_sim
    planner.state = st
    return planner
