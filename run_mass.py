"""run_mass.py -- does the fleet-size optimum move when rotor power depends on battery mass?

The shared inventory U is divided among K airframes, so each carries E_u = U/K of battery. Take-off
mass m(K) = m_dry + (U/K)/e_b, constant within a run. Flight and hover power follow
    Pf(K) = Pf0 (m(K)/m_ref)^gamma_f,    Ph(K) = Ph0 (m(K)/m_ref)^1.5   (momentum theory, hover),
with Pf0 = 300 W, Ph0 = 400 W referenced to m_ref = m_dry + m_bref. Defaults: m_dry 2.0 kg,
180 Wh/kg, m_bref 0.5 kg, gamma_f 1.0. Everything else is run_grid's protocol (12 h, SA 1200 iters,
coordinated, launch-time), with the anti-censoring AND descending-tail extension of fill_tail.py.

The rule for this model is the fixed point K = v U / (2 Pf(K) r_max + v Ph(K) B / R_u) for the reach
cap, and the numerical argmax of Gamma(K) with Pf(K), Pbar(K) for the commute cap (the closed form
assumes constant powers).

usage: python run_mass.py mass.csv --layouts paper ring core --M 100 --Emax 1.5e6 3e6 --seeds 1-12 --procs 22
       [--gamma 1.0] [--mdry 2.0] [--whkg 180] [--mbref 0.5]
"""
import os, sys, csv, argparse, itertools, math
import dyn_env


def apply_mass(gamma, mdry, whkg, mbref):
    # DynParams is a dataclass: whether its generated __init__ calls __post_init__ is fixed when the
    # class is created, so a __post_init__ attached later is silently ignored. Wrap __init__ instead.
    e_b = whkg * 3600.0; m_ref = mdry + mbref
    cls = dyn_env.DynParams
    if getattr(cls, "_mass_wrapped", False): return
    orig_init = cls.__init__
    def init(self, *args, **kw):
        orig_init(self, *args, **kw)
        U = (1 - self.rho) * self.Emax
        m = mdry + (U / self.K) / e_b
        self.Pf = 300.0 * (m / m_ref) ** gamma
        self.Ph = 400.0 * (m / m_ref) ** 1.5
    cls.__init__ = init; cls._mass_wrapped = True


if os.environ.get("MASS_MODEL"):                       # spawn workers re-import and inherit the env
    apply_mass(*(float(v) for v in os.environ["MASS_MODEL"].split(",")))

from run_grid import one, k_range, FIELDS, parse_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--layouts", nargs="+", default=["paper"])
    ap.add_argument("--M", nargs="+", type=int, default=[100]); ap.add_argument("--Emax", nargs="+", type=float, default=[1.5e6])
    ap.add_argument("--seeds", default="1-12"); ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=1.0); ap.add_argument("--mdry", type=float, default=2.0)
    ap.add_argument("--whkg", type=float, default=180.0); ap.add_argument("--mbref", type=float, default=0.5)
    a = ap.parse_args()
    os.environ["MASS_MODEL"] = f"{a.gamma},{a.mdry},{a.whkg},{a.mbref}"; apply_mass(a.gamma, a.mdry, a.whkg, a.mbref)
    done = {}
    if os.path.exists(a.out) and os.path.getsize(a.out):
        for r in csv.DictReader(open(a.out)): done[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]), int(r["K"]))] = float(r["J"])
    new = not done
    f = open(a.out, "a", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS)
    if new: w.writeheader(); f.flush()
    def run_many(keys):
        args = [(l, M, E, K, s, "exclude", "launch", 0.0, 0.0, "prior", 0.0, "sa", 12600.0, 43200.0, 1200)
                for (l, M, E, s, K) in keys if (l, M, E, s, K) not in done]
        if not args: return
        from multiprocessing import Pool
        with Pool(max(1, min(a.procs, len(args)))) as pool:
            for r in pool.imap_unordered(one, args):
                w.writerow(r); f.flush(); done[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]), int(r["K"]))] = float(r["J"])
                print(f"{r['layout']} M={r['M']} E={float(r['Emax']):.1e} s={r['seed']} K={r['K']} J={float(r['J']):.3e}", flush=True)
    insts = list(itertools.product(a.layouts, a.M, a.Emax, parse_seeds(a.seeds)))
    run_many([(l, M, E, s, K) for l, M, E, s in insts for K in k_range(l, M, E, s)])
    for _ in range(30):                                  # top-edge AND descending-tail extension
        ext = []
        for l, M, E, s in insts:
            b = {K: J for (ll, mm, ee, ss, K), J in done.items() if (ll, mm, ee, ss) == (l, M, E, s)}
            Ks = sorted(b)
            if len(Ks) < 2: continue
            if min(b, key=b.get) == Ks[-1] or b[Ks[-1]] < b[Ks[-2]]:
                ext += [(l, M, E, s, K) for K in range(Ks[-1] + 1, Ks[-1] + 4) if K <= 60]
        if not ext: break
        run_many(ext)
    f.close()


if __name__ == "__main__":
    main()
