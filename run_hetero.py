"""run_hetero.py -- does equal sharing of the energy inventory drive the core-family results?

Equal allocation gives every airborne UAV U/K. The alternative tested here reserves, for UAV 0, just
enough energy to serve the farthest sensor from any buffer state (out-and-back flight plus a
full-buffer dwell, +2% margin) and splits the rest equally among the other K-1:
    E_0 = 1.02 * (2 Pf r_max / v + Ph B / R_u),     E_k = (U - E_0) / (K - 1),  k >= 1.
The total is still U, so the fleet-level inventory is unchanged; only its split differs.

Energy per sortie enters the simulator in exactly two functional places, both in DynSim._launch: the
drone's starting energy and the planning budget. Both are patched to the per-UAV value. The
`energy_used` record (and hence the reported P_bar) still assumes equal shares, so P_bar in these rows
is wrong for UAV 0; this experiment uses only J and the argmin.

Protocol matches run_grid (12 h, burn-in 3 h, SA 1200 iterations, coordinated, launch-time), with
the anti-censoring extension of run_depot.py. K starts at 2 (K=1 leaves nothing to split).
usage: python run_hetero.py out.csv --layouts core --M 100 200 --Emax 1.5e6 --seeds 1-12 --procs 12
"""
import os, sys, csv, argparse, itertools, time
import numpy as np
import dyn_env
from dyn_env import DynParams, SensorField


def apply_hetero():
    def _launch(self, k, t):
        p = self.p
        if not hasattr(self, "_E_alloc"):
            r = np.linalg.norm(self.field.pos - p.home, axis=1)
            U = p.E_usable * p.K
            e0 = 1.02 * (p.e_fly(2 * r.max()) + p.Ph * p.B_bits / p.R)
            e0 = min(e0, U)                                  # cannot reserve more than exists
            rest = (U - e0) / max(p.K - 1, 1)
            self._E_alloc = [e0] + [rest] * (p.K - 1)
        e = self._E_alloc[k]
        d = self._Drone()
        d.pos = p.home.copy(); d.E = e; d.t_launch = t
        d.commute = d.travel = d.dwell = d.tour_len = 0.0
        d.n_visited = 0; d.phase = "flying"; d.route, d.ri = [], 0
        d.leg_t0 = t; d.leg_len = 0.0
        self._live[k] = d
        d.route = self._plan(k, t, None, e, 0.5)
        return d
    dyn_env.DynSim._launch = _launch


if os.environ.get("HETERO_ALLOC") == "1":        # spawn workers re-import and inherit the env
    apply_hetero()

from run_grid import one, k_range, FIELDS, parse_seeds
OUT_FIELDS = FIELDS + ["alloc"]


def job(args):
    row = one(args); row["alloc"] = "reserve_far"; return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--layouts", nargs="+", default=["core"])
    ap.add_argument("--M", nargs="+", type=int, default=[100]); ap.add_argument("--Emax", nargs="+", type=float, default=[1.5e6])
    ap.add_argument("--seeds", default="1-12"); ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--budget", type=float, default=0)
    a = ap.parse_args()
    os.environ["HETERO_ALLOC"] = "1"; apply_hetero()
    t0 = time.time(); done = {}
    if os.path.exists(a.out) and os.path.getsize(a.out):
        for r in csv.DictReader(open(a.out)):
            done[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]), int(r["K"]))] = float(r["J"])
    new = not done
    f = open(a.out, "a", newline=""); w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
    if new: w.writeheader(); f.flush()

    def record(row):
        w.writerow(row); f.flush()
        done[(row["layout"], int(row["M"]), float(row["Emax"]), int(row["seed"]), int(row["K"]))] = float(row["J"])
        print(f"{row['layout']} M={row['M']} E={float(row['Emax']):.1e} K={row['K']} s={row['seed']} "
              f"J={float(row['J']):.3e} never={row['n_never']} ({float(row['secs']):.0f}s)", flush=True)

    def run_many(keys):
        todo = [k for k in keys if k not in done]
        args = [(lay, M, E, K, s, "exclude", "launch", 0.0, 0.0, "prior", 0.0, "sa", 12600.0, 43200.0, 1200)
                for (lay, M, E, s, K) in todo]
        if a.procs > 1 and args:
            from multiprocessing import Pool
            with Pool(a.procs) as pool:
                for row in pool.imap_unordered(job, args): record(row)
        else:
            for x in args:
                if a.budget and time.time() - t0 > a.budget: f.close(); sys.exit(0)
                record(job(x))

    insts = list(itertools.product(a.layouts, a.M, a.Emax, parse_seeds(a.seeds)))
    run_many([(lay, M, E, s, K) for lay, M, E, s in insts for K in k_range(lay, M, E, s) if K >= 2])
    for _ in range(8):
        ext = []
        for lay, M, E, s in insts:
            Ks = sorted(K for (l, m, e, ss, K) in done if (l, m, e, ss) == (lay, M, E, s))
            if Ks and min(Ks, key=lambda K: done[(lay, M, E, s, K)]) == Ks[-1]:
                ext += [(lay, M, E, s, K) for K in range(Ks[-1] + 1, Ks[-1] + 4)]
        if not ext: break
        print(f"extending {len(set(k[:4] for k in ext))} censored instance(s)", flush=True)
        run_many(ext)
    f.close()


if __name__ == "__main__":
    main()
