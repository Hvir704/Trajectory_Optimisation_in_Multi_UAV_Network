"""sat_check.py -- is Proposition 3's no-overflow assumption met where it is used?

Prop. 3 uses conservation, H = sum_i lambda_i / R_u, which holds only if the fleet collects
everything generated. With drop-head buffers that fails when a sensor waits longer than B/lambda_i
between visits. Per run, post-burn-in only:
  p_sat     fraction of served visits that find a full buffer
  drop_frac bits dropped / bits generated over the measurement window  (the quantity that
            actually enters H: the realised hover load is (1 - drop_frac) * sum lambda / R_u)
usage: python sat_check.py out.csv --layout ring --M 100 --Emax 3e6 --K 4 5 6 --seeds 1-2 [--planner sa|rr_tour]
Appends and skips rows already present.
"""
import argparse, csv, os, time
import numpy as np
from dyn_env import DynParams, DynSim
from sa_sortie import build_sa_planner

FIELDS = ["planner", "layout", "M", "Emax", "K", "seed", "J", "visits", "p_sat", "drop_frac", "n_never", "secs"]

def run(planner, layout, M, Emax, K, seed):
    p = DynParams(M=M, K=K, Emax=Emax, layout=layout, T_horizon=43200.0, T_burnin=3 * 3600.0)
    if planner == "sa":
        pl = build_sa_planner(iters=1200, seed_base=seed)
    else:
        from rr_planner import build_rr_planner
        pl = build_rr_planner("tour")
    sim = DynSim(p, pl, seed=seed); F = sim.field
    st = {"n": 0, "sat": 0, "drop0": None}
    orig_visit, orig_sync = F.visit, F.sync if hasattr(F, "sync") else None
    def visit(t, i):
        if t >= p.T_burnin:
            if st["drop0"] is None: st["drop0"] = F.bits_dropped
            st["n"] += 1; st["sat"] += F.backlog[i] >= p.B_bits - 1e-6
        return orig_visit(t, i)
    F.visit = visit
    t0 = time.time(); m = sim.run()
    window = p.T_horizon - p.T_burnin
    gen = float(F.lam_bits.sum()) * window
    dropped = F.bits_dropped - (st["drop0"] or 0.0)
    return dict(planner=planner, layout=layout, M=M, Emax=Emax, K=K, seed=seed, J=m["J_timeavg"],
                visits=st["n"], p_sat=st["sat"] / max(st["n"], 1), drop_frac=dropped / gen,
                n_never=m.get("n_never_visited", float("nan")), secs=time.time() - t0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--layout", default="ring")
    ap.add_argument("--M", type=int, default=100); ap.add_argument("--Emax", type=float, default=3e6)
    ap.add_argument("--K", nargs="+", type=int, required=True); ap.add_argument("--seeds", default="1-2")
    ap.add_argument("--planner", default="sa")
    a = ap.parse_args()
    lo, hi = (a.seeds.split("-") + [a.seeds])[:2]
    done = set()
    if os.path.exists(a.out) and os.path.getsize(a.out):
        done = {(r["planner"], r["layout"], int(r["M"]), float(r["Emax"]), int(r["K"]), int(r["seed"])) for r in csv.DictReader(open(a.out))}
    new = not done
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        for K in a.K:
            for s in range(int(lo), int(hi) + 1):
                if (a.planner, a.layout, a.M, a.Emax, K, s) in done: continue
                r = run(a.planner, a.layout, a.M, a.Emax, K, s); w.writerow(r); f.flush()
                print(f"{a.planner} {a.layout} M={a.M} E={a.Emax:.1e} K={K} s={s}: p_sat={r['p_sat']:.4f} "
                      f"drop_frac={r['drop_frac']:.4f} never={r['n_never']} ({r['secs']:.0f}s)", flush=True)
