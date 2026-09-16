"""mip_gsec.py -- exact set-orienteering by MIP + iterative subtour-elimination (GSEC) cuts.
Same problem and same integer scaling as milp_sortie.solve_sortie; a second, independent exact
route whose LP bound is much tighter than CP-SAT's circuit relaxation on these instances."""
import math, time
import numpy as np
from ortools.linear_solver import pywraplp
from milp_sortie import candidates


def _cycles(succ):
    seen, cyc = set(), []
    for s0 in succ:
        if s0 in seen: continue
        c, cur = [], s0
        while cur not in seen:
            seen.add(cur); c.append(cur); cur = succ[cur]
        cyc.append(c)
    return cyc


def solve_gsec(req, value, time_limit=120.0, backend="SCIP", max_rounds=200, threads=8):
    p = req.p
    cand = candidates(req, value); n = len(cand); t0 = time.time()
    if n == 0: return dict(route=[], value=0.0, status="EMPTY", secs=0.0, n_cand=0, rounds=0)
    P = np.vstack([req.home[None, :], req.pos[cand]])
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    c = np.ceil(p.Pf * D / p.v); h = np.ceil(p.e_hover(req.dwell_est[cand])); E = math.floor(req.E_usable)
    v = np.asarray(value, float)[cand]
    S = pywraplp.Solver.CreateSolver(backend)
    S.SetNumThreads(threads)
    y = [S.BoolVar(f"y{i}") for i in range(n + 1)]
    S.Add(y[0] == 1)
    x = {}
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j and not (i and j and c[0, i] + c[i, j] + c[j, 0] + h[i-1] + h[j-1] > E):
                x[i, j] = S.BoolVar(f"x{i}_{j}")
    for i in range(n + 1):
        S.Add(sum(x[i, j] for j in range(n + 1) if (i, j) in x) == y[i])
        S.Add(sum(x[j, i] for j in range(n + 1) if (j, i) in x) == y[i])
    for (i, j) in x:
        if i < j and (j, i) in x and i:
            S.Add(x[i, j] + x[j, i] <= y[i])                 # 2-cycles among sensors
    S.Add(sum(c[i, j] * x[i, j] for (i, j) in x) + sum(h[i-1] * y[i] for i in range(1, n + 1)) <= E)
    S.Maximize(sum(v[i-1] * y[i] for i in range(1, n + 1)))
    rounds = 0
    while True:
        left = time_limit - (time.time() - t0)
        if left <= 0: return dict(route=None, value=float("nan"), status="TIMEOUT", secs=time.time()-t0, n_cand=n, rounds=rounds)
        S.SetTimeLimit(int(left * 1000))
        st = S.Solve(); rounds += 1
        if st != pywraplp.Solver.OPTIMAL:
            return dict(route=None, value=float("nan"), status=f"STATUS{st}", secs=time.time()-t0, n_cand=n, rounds=rounds)
        succ = {i: j for (i, j), var in x.items() if var.solution_value() > 0.5}
        cyc = _cycles(succ)
        sub = [cy for cy in cyc if 0 not in cy]
        if not sub:
            route, cur = [], succ[0]
            while cur != 0: route.append(cand[cur-1]); cur = succ[cur]
            return dict(route=route, value=float(sum(value[j] for j in route)), status="OPTIMAL",
                        secs=time.time()-t0, n_cand=n, rounds=rounds)
        for cy in sub:                                          # GSEC: arcs inside Q <= |Q|-1 when Q used
            Q = set(cy)
            for k in cy:
                S.Add(sum(x[i, j] for i in Q for j in Q if (i, j) in x) <= sum(y[i] for i in Q) - y[k])
        if rounds >= max_rounds:
            return dict(route=None, value=float("nan"), status="MAXROUNDS", secs=time.time()-t0, n_cand=n, rounds=rounds)
