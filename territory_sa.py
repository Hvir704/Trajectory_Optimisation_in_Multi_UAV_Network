"""territory_sa.py -- SA inside fixed territories: the published clustering idea + this paper's planner.

Rahimi & Shafieinejad (2024) give each UAV a fixed k-means territory and patrol it blindly; the SA
planner shares the whole field and chooses by age and priority. This hybrid keeps both: UAV k may only
serve sensors in territory k, and inside it the SA planner runs unchanged. The territory is enforced
through SA's existing exclusion input (the same mechanism that stops two UAVs chasing one sensor), so
SA's search, repair and hover budgeting are untouched.

Variants (fixed before any evaluation):
  km   plain k-means, exactly as the published method clusters
  bal  size-balanced: k-means centroids, then capacity-constrained assignment (ceil(M/K) per territory),
       re-centred and re-assigned for a few rounds -- k-means balances area, not workload
The planner needs the drone index; run_grid calls planner.bind_sim(sim), which wraps sim._plan to record
k (the simulator is unchanged). Territories are computed once per simulation (K is fixed per run).
"""
import dataclasses
import numpy as np
from sa_sortie import build_sa_planner
from rr_planner import _kmeans


def _balanced(X, k, seed=0, rounds=10, s=None):
    """Capacity-constrained territories. s: per-sensor load (default 1 = balance counts)."""
    n = len(X); s = np.ones(n) if s is None else np.asarray(s, float); cap = s.sum() / k * (1 + 1e-9)
    lab = _kmeans(X, k, seed=seed)
    C = np.array([X[lab == j].mean(axis=0) if np.any(lab == j) else X[j % n] for j in range(k)])
    for _ in range(rounds):
        D = np.linalg.norm(X[:, None, :] - C[None], axis=-1)
        order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
        new = -np.ones(n, int); load = np.zeros(k)
        for i, j in order:
            if new[i] < 0 and load[j] + s[i] <= cap:
                new[i] = j; load[j] += s[i]
        for i in np.where(new < 0)[0]:                      # leftovers (load granularity): least-loaded territory
            j = int(np.argmin(load)); new[i] = j; load[j] += s[i]
        if np.array_equal(new, lab): break
        lab = new
        C = np.array([X[lab == j].mean(axis=0) if np.any(lab == j) else C[j] for j in range(k)])
    return lab


def build_territory_sa_planner(variant="km", iters=1200, seed_base=0):
    inner = build_sa_planner(iters=iters, seed_base=seed_base)
    st = {"mask": None, "k": 0, "K": None}

    def bind_sim(sim):
        orig = sim._plan
        def _plan(k, *a, **kw):
            st["k"] = k
            return orig(k, *a, **kw)
        sim._plan = _plan
        st["K"] = sim.p.K

    def planner(req):
        if st["mask"] is None:
            K = st["K"]
            lab = _kmeans(req.pos, K, seed=0) if variant == "km" else _balanced(req.pos, K, seed=0)
            st["mask"] = [lab != j for j in range(K)]          # True = outside territory j
            st["sizes"] = [int(np.sum(lab == j)) for j in range(K)]
        outside = st["mask"][st["k"] % st["K"]]
        ex = outside if req.excluded is None else (np.asarray(req.excluded, bool) | outside)
        return inner(dataclasses.replace(req, excluded=ex))

    planner.bind_sim = bind_sim
    planner.state = st
    return planner


def build_territory_due_planner():
    """Periodic territorial patrol with sqrt-weighted visit frequencies ('terr_due').

    Built from the decomposition of why the cluster patrol beats SA on compact fields: fixed short loops
    give perfectly periodic revisits and short hops, but equal visit frequencies waste the allocation
    advantage that priorities offer. Here each UAV owns a size-balanced territory and walks that
    territory's tour order (short hops); at each launch it selects its most DUE sensors, ranked by
    age_i * sqrt(w_i) -- the Prop. 4 allocation f_i ~ sqrt(w_i/c_i) with c_i ~ constant inside a compact
    territory -- adding them in rank order while the tour-ordered route through the selection still fits
    the sortie energy. No annealing, no tuned parameter. Launch-time only."""
    from rr_planner import tour_order
    st = {"lab": None, "k": 0, "K": None}

    def bind_sim(sim):
        orig = sim._plan
        def _plan(k, *a, **kw):
            st["k"] = k
            return orig(k, *a, **kw)
        sim._plan = _plan
        st["K"] = sim.p.K

    def route_energy(p, req, seq, hov):
        cur, E = req.home, 0.0
        for j in seq:
            E += p.e_fly(float(np.linalg.norm(req.pos[j] - cur))) + hov[j]; cur = req.pos[j]
        return E + p.e_fly(float(np.linalg.norm(req.home - cur)))

    def planner(req):
        if req.start is not None:
            raise ValueError("terr_due is launch-time only")
        p = req.p
        if st["lab"] is None:
            K = st["K"]
            # Prop. 4: equal fleet effort per unit of sqrt-weight => balance territories by sum sqrt(w_i)
            lab = _balanced(req.pos, K, seed=0, s=np.sqrt(req.weight_est))
            st["lab"] = lab
            st["order"] = []; st["rank"] = []
            for j in range(K):
                idx = np.where(lab == j)[0]
                o = idx[tour_order(req.pos[idx], req.home)] if len(idx) else idx
                st["order"].append(o); st["rank"].append({int(s): r for r, s in enumerate(o)})
        k = st["k"] % st["K"]; order = st["order"][k]; rank = st["rank"][k]
        if len(order) == 0: return []
        hov = p.e_hover(req.dwell_est)
        excl = np.zeros(len(req.pos), bool) if req.excluded is None else np.asarray(req.excluded, bool)
        cand = [int(j) for j in order if not excl[j]]
        due = sorted(cand, key=lambda j: -req.age[j] * np.sqrt(req.weight_est[j]))
        chosen = []
        for j in due:
            trial = sorted(chosen + [j], key=lambda s: rank[s])      # visit in the territory's tour order
            if route_energy(p, req, trial, hov) <= req.E_usable + 1e-6:
                chosen = trial
        return chosen

    planner.bind_sim = bind_sim
    planner.state = st
    return planner
