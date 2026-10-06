"""slot_gate.py -- does the slotted port (dyn_slot_env.SlotSim) reproduce DynSim?

A launch-time planner's routes are flown in SlotSim by a scripted pilot (straight flight to each route
node, drain only that node, fly home when the route is done), and J is compared with DynSim run on the
same seed. The residual is the cost of slotting itself (one move per slot, idle hover to the slot end,
return margin), NOT of the DRL method. Run on pilot seeds only.

usage: python slot_gate.py [--slots 20 10 5] [--planners rr_tour greedy] [--cells paper:50:1.5e6:3 ...]
"""
import argparse, time
import numpy as np
from dyn_env import DynParams, DynSim, SortieRequest, greedy_ratio_planner
from dyn_slot_env import SlotSim


def make_planner(name, seed=0):
    """Any launch-time planner from run_grid's registry (sa planners need iters: default 1200)."""
    if name == "greedy":
        return greedy_ratio_planner
    if name in ("rr_tour", "rr_sweep"):
        from rr_planner import build_rr_planner
        return build_rr_planner("tour" if name == "rr_tour" else "sweep")
    if name == "cluster_patrol":
        from rr_planner import build_cluster_planner
        return build_cluster_planner()
    if name in ("gzz_full", "gzz_rev"):
        from gzz_planner import build_gzz_planner
        return build_gzz_planner(dwell_mode=name.split("_")[1], seed=seed)
    if name in ("asghar_full", "asghar_rev"):
        from asghar_planner import build_asghar_planner
        return build_asghar_planner(dwell_mode=name.split("_")[1], seed=seed)
    if name in ("cyc",):
        from cyclic_sched import build_cyclic_planner
        return build_cyclic_planner()
    if name == "sa":
        from sa_sortie import build_sa_planner
        return build_sa_planner(iters=1200, seed_base=seed)
    raise ValueError(name)


class _Shim:
    """Stands in for DynSim for planners that need bind_sim(sim): they wrap sim._plan to learn the drone index."""
    def __init__(self, p): self.p = p
    def _plan(self, k, *a, **kw): return None


def fly_scripted(p, seed, planner, slot, rc, drain="target"):
    """Fly a launch-time planner in the slotted world. Returns SlotSim metrics."""
    S = SlotSim(p, seed, slot=slot, collect_radius=rc, motion="direct", drain=drain)
    F = S.field; K = p.K
    shim = _Shim(p)
    if hasattr(planner, "bind_sim"): planner.bind_sim(shim)
    route = [[] for _ in range(K)]; ri = np.zeros(K, int); seen_launch = np.full(K, -1.0)
    Ts = None
    while not S.done:
        targets = [None] * K
        for k in range(K):
            if S.state[k] == "air" and S.t_launch[k] != seen_launch[k] and S.busy_until[k] <= S.clock:
                # (re)launch: plan like DynSim._plan (launch dwell estimate at elapsed_frac 0.5, exclusion)
                seen_launch[k] = S.t_launch[k]
                age = F.age(S.clock)
                lam = F.lam_est if p.learn_lambda else np.full(p.M, p.lam_bits)
                Ts_ = Ts if Ts is not None else p.t_c
                dwell = np.minimum((age + 0.5 * max(Ts_, p.t_c)) * lam, p.B_bits) / p.R
                excl = np.zeros(p.M, bool)
                for kk in range(K):
                    if kk != k: excl[route[kk][ri[kk]:]] = True
                req = SortieRequest(pos=F.pos, home=p.home, age=age, weight_est=F.wi_base.copy(),
                                    dwell_est=dwell, E_usable=S.E[k], p=p, E_usable_full=p.E_usable,
                                    lam_est=lam, excluded=excl)
                shim._plan(k)                                  # sets the drone index for bind_sim planners
                route[k] = list(planner(req)); ri[k] = 0
            if S.state[k] == "air" and ri[k] < len(route[k]):
                targets[k] = F.pos[route[k][ri[k]]]
            elif S.state[k] == "air" and S.busy_until[k] <= S.clock + slot:
                S.state[k] = "return"
        before = F.t_last_visit.copy()
        # [fix 2026-10-05] abort a route only if the UAV was already on its target when the slot began (so it
        # had the slot to drain) and still did not drain. Previously the test ran after the step, so a UAV
        # arriving exactly at the slot end (no time left to drain) was wrongly sent home and the rest of its
        # route abandoned -- on ring layouts this stranded whole territories (gate_pilot ring seed 15).
        on_tgt = [targets[k] is not None and np.linalg.norm(S.pos[k] - targets[k]) <= rc and
                  S.busy_until[k] <= S.clock for k in range(K)]
        n_rec = len(S.records)
        S.step(None, targets)
        for k in range(K):
            if targets[k] is not None:
                j = route[k][ri[k]]
                if F.t_last_visit[j] > before[j]:
                    ri[k] += 1
                elif on_tgt[k] and S.busy_until[k] <= S.clock:
                    S.state[k] = "return"; ri[k] = len(route[k])      # could not afford the drain
        for r in S.records[n_rec:]:
            dur = r[1] - r[0]; Ts = dur if Ts is None else 0.9 * Ts + 0.1 * dur
    return S.metrics()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", nargs="+", type=float, default=[20.0, 10.0])
    ap.add_argument("--planners", nargs="+", default=["rr_tour", "greedy"])
    ap.add_argument("--cells", nargs="+", default=["paper:50:1.5e6:3", "paper:50:1.5e6:5", "core:100:3e6:6"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[13])
    ap.add_argument("--rc", type=float, default=1.0)
    ap.add_argument("--drain", default="target", choices=["target", "radius"],
                    help="radius: also drain every sensor within --rc on the way (same physics as the DRL env)")
    ap.add_argument("--Th", type=float, default=12 * 3600.0)
    ap.add_argument("--csv", default=None, help="append one row per (cell, planner, seed, slot) to this CSV")
    a = ap.parse_args()
    print("cell, planner, seed, slot: J DynSim | J slotted | ratio-1 (J_age ratio-1)", flush=True)
    for cell in a.cells:
        lay, M, E, K = cell.split(":"); M, E, K = int(M), float(E), int(K)
        for name in a.planners:
            for s in a.seeds:
                p = DynParams(M=M, K=K, Emax=E, layout=lay, T_horizon=a.Th, T_burnin=3 * 3600.0)
                pl0 = make_planner(name, s); sim = DynSim(p, pl0, seed=s)
                if hasattr(pl0, "bind_sim"): pl0.bind_sim(sim)
                m0 = sim.run()
                for slot in a.slots:
                    t = time.time()
                    m1 = fly_scripted(p, s, make_planner(name, s), slot, a.rc, a.drain)
                    if a.csv:
                        import csv, os
                        new = not os.path.exists(a.csv)
                        with open(a.csv, "a", newline="") as fh:
                            w = csv.writer(fh)
                            if new: w.writerow(["layout","M","Emax","K","seed","planner","slot","rc","drain","J_dynsim","J_slot","Jage_dynsim","Jage_slot","n_never_slot"])
                            w.writerow([lay, M, E, K, s, name, slot, a.rc, a.drain, m0["J_timeavg"], m1["J_timeavg"], m0["J_age"], m1["J_age"], m1["n_never"]])
                    print(f"{cell:18s} {name:8s} s={s} slot={slot:4.0f}s: {m0['J_timeavg']:.4e} | {m1['J_timeavg']:.4e} | "
                          f"{m1['J_timeavg']/m0['J_timeavg']-1:+.3f} ({m1['J_age']/m0['J_age']-1:+.3f})  "
                          f"n/sortie {m0['mean_n_visited']:.2f}|{m1['mean_n']:.2f}  sorties {m0['n_sorties']}|{m1['n_sorties']}  "
                          f"never {m0['n_never_visited']}|{m1['n_never']}  trunc {m1['trunc']}  {time.time()-t:.0f}s", flush=True)
