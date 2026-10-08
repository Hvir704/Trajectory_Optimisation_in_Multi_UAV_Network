# usage: python divert_check.py   (repo root). Before the patch: diverted UAVs still serve the OLD target.
# Does mid-leg diversion actually change where the UAV flies?
import sys; sys.path.insert(0, ".")
import numpy as np
from dyn_env import DynParams, DynSim
from sa_sortie import build_sa_planner
p = DynParams(M=100, K=4, Emax=1.5e6, layout="paper", replan_mode="per_leg", divert_mode=True,
              T_horizon=5 * 3600, T_burnin=1 * 3600)
sim = DynSim(p, planner=build_sa_planner(iters=300, seed_base=1), seed=1)
log_div, log_vis = [], []
orig_try = sim._try_divert
def try_div(k, d, t):
    j_old = d.route[d.ri] if d.ri < len(d.route) else None
    rest_before = list(d.route[d.ri:])
    ok = orig_try(k, d, t)
    if ok: log_div.append((t, k, j_old, d.route[d.ri], len(rest_before)))
    return ok
sim._try_divert = try_div
orig_visit = sim.field.visit
cur_k = {"k": None}
import heapq
orig_pop = heapq.heappop
def pop(h):
    item = orig_pop(h); cur_k["k"] = item[3]; return item
heapq.heappop = pop
def visit(t, j):
    log_vis.append((t, cur_k["k"], j)); return orig_visit(t, j)
sim.field.visit = visit
m = sim.run()
heapq.heappop = orig_pop
went_old = went_new = 0; dropped = []
for (t, k, j_old, j_new, nrest) in log_div:
    nxt = next(((tv, j) for tv, kk, j in log_vis if kk == k and tv >= t), None)
    if nxt is None: continue
    went_old += nxt[1] == j_old; went_new += nxt[1] == j_new; dropped.append(nrest - 1)
print(f"diversions: {len(log_div)}  next node actually served = OLD target {went_old}, NEW target {went_new}")
print(f"planned stops discarded per diversion (beyond the current leg): mean {np.mean(dropped):.1f}")
claimed_never = sum(1 for (t, k, jo, jn, _) in log_div if not any(kk == k and j == jn and tv >= t for tv, kk, j in log_vis))
print(f"diverted-to targets this UAV never served afterwards: {claimed_never} of {len(log_div)}")
