"""heldout_v3.py -- registered held-out test of the v3 fleet-sizing theory (theory_v3.py) on NEW seeds.

Protocol (fixed before any held-out run):
  instances   the 20 primary cells (paper M=50,100,200,400 at 1.5 MJ; paper 50,100,200 at 3 MJ; paper 100 at 6 MJ;
              ring and core M=50,100,200 at 1.5 and 3 MJ), seeds 13-20 (never used before), L = 12.6 km.
  planners    sa (annealer), rr_tour (age-blind tour patrol), cyc (C4).
  predictions computed by `predict` from the site survey only (ex-ante commute inputs at the smallest swept K):
              old  = min(floor Kr, round Kc)              (the paper's rule)
              new  = argmin_K of the served-set surrogate  (theory_v3.surrogate_rule, a priori)
              br   = [round Kc, floor Kr] on commute-bound instances
              written to heldout_v3_predictions.csv; its sha256 is printed and must be recorded before `run`.
  pilot       evaluated after the runs, defined now: scale = measured J_age / surrogate served term at the planner's
              smallest stranding-free swept K; pilot = argmin of the rescaled surrogate.
  sweep       contiguous K from min(k_range) to max(k_range, old+2, new+2, floor Kflight(S_1)+2), extended by 2 at
              the top until the argmin is interior and J at the three largest K exceeds 1.5 x the minimum
              (the paper's rule), per planner; never below K = 1.
  hypotheses  H1 (core, sa): new has lower median AND lower 90th-percentile regret than old.
              H2 (all, sa): new within-one agreement >= old within-one agreement.
              H3 (ring, sa): the annealer's optimum lies in the bracket in >= 90% of ring instances.
              Reported for every planner and family, whatever the outcome: exact, within-one, regret median/p90/max.

usage:
  python heldout_v3.py predict                      -> heldout_v3_predictions.csv + sha256
  python heldout_v3.py run results_v2/heldout_v3.csv --sha <sha256> [--planners sa rr_tour cyc] [--procs 22]
  python heldout_v3.py evaluate results_v2/heldout_v3.csv
"""
import sys, os, csv, math, hashlib, argparse, itertools, collections, time
import numpy as np
from multiprocessing import Pool
from theory_v3 import field, old_rule, surrogate_rule, bracket, pilot_scale, caps
from exante_rc import exante
from run_grid import one, k_range, FIELDS, SIM_VERSION

CELLS = ([("paper", M, 1.5e6) for M in (50, 100, 200, 400)] + [("paper", M, 3e6) for M in (50, 100, 200)]
         + [("paper", 100, 6e6)] + [(lay, M, E) for lay in ("ring", "core") for E in (1.5e6, 3e6) for M in (50, 100, 200)])
SEEDS = list(range(13, 21)); L = 12600.0; PRED = "heldout_v3_predictions.csv"


def pred_sha():
    """sha256 of the predictions file with line endings normalised to LF (git on Windows may check it out as CRLF)."""
    return hashlib.sha256(open(PRED, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def predict():
    rows = []
    for (lay, M, E), s in itertools.product(CELLS, SEEDS):
        p, F = field(lay, M, E, L, s); r = np.linalg.norm(F.pos - p.home, axis=1)
        kr = k_range(lay, M, E, s, L); K0 = min(kr)
        rc, Pb, _ = exante(p, r, np.full(M, p.lam_bits), F.pos, K0)
        old = old_rule(p, r, rc, Pb); new, j = surrogate_rule(p, F, rc, Pb)
        br = bracket(p, r, rc, Pb); Kc, Kr = caps(p, r, rc, Pb)
        U = (1 - p.rho) * E; r1 = np.sort(r)[-2]                      # second-farthest sensor
        kfl1 = math.floor(p.v * U / (2 * p.Pf * r1))
        hi = max(max(kr), old + 2, new + 2, kfl1 + 2)
        rows.append(dict(layout=lay, M=M, Emax=E, L=L, seed=s, K0=K0, rc_exante=round(rc, 3), Pb_exante=round(Pb, 3),
                         Kc=round(Kc, 4), Kr=round(Kr, 4), old=old, new=new, new_stranded=j,
                         br_lo=(br[0] if br else ""), br_hi=(br[1] if br else ""), K_lo=K0, K_hi=hi))
        print(f"{lay} M={M} E={E/1e6} s={s}: old {old} new {new} (strands {j}) bracket {br} sweep {K0}-{hi}", flush=True)
    with open(PRED, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("sha256", pred_sha())


def load_pred():
    return {(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"])): r for r in csv.DictReader(open(PRED))}


def read_out(path):
    J = collections.defaultdict(dict)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        for r in csv.DictReader(open(path)):
            if r.get("sim_version") != SIM_VERSION: sys.exit(f"{path}: row with sim_version {r.get('sim_version')}")
            J[(r["planner"], r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]))][int(r["K"])] = r
    return J


def needs_more(byK):
    Ks = sorted(byK); Jv = {K: float(byK[K]["J"]) for K in Ks}; am = min(Jv, key=Jv.get)
    top3 = Ks[-3:]
    return am >= Ks[-1] - 1 or not all(Jv[K] > 1.5 * Jv[am] for K in top3)


def run(a):
    sha = pred_sha()
    if sha != a.sha: sys.exit(f"{PRED} sha256 {sha} != registered {a.sha}: predictions changed after registration")
    P = load_pred(); new = (not os.path.exists(a.out)) or os.path.getsize(a.out) == 0
    with open(a.out, "a", newline="") as f, Pool(a.procs) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader(); f.flush()
        for rnd in itertools.count():
            done = read_out(a.out); jobs = []
            for (lay, M, E, s), pr in P.items():
                for pl in a.planners:
                    have = done.get((pl, lay, M, E, s), {})
                    if rnd == 0 or not have: Ks = range(int(pr["K_lo"]), int(pr["K_hi"]) + 1)
                    elif needs_more(have): Ks = range(max(have) + 1, max(have) + 3)
                    else: Ks = []
                    jobs += [(lay, M, E, K, s, "exclude", "launch", 0, 0.0, "prior", 0.0, pl, L, 43200.0, 1200)
                             for K in Ks if K not in have]
            if not jobs:
                print("ALL DONE: every sweep interior and extended", flush=True); return
            jobs.sort(key=lambda j: (-j[1], -j[3]))
            print(f"round {rnd}: {len(jobs)} runs", flush=True); t0 = time.time()
            for i, row in enumerate(pool.imap_unordered(one, jobs)):
                w.writerow(row); f.flush()
                if i % 20 == 0:
                    print(f"[{i+1}/{len(jobs)}] {row['planner']} {row['layout']} M={row['M']} E={row['Emax']:.1e} "
                          f"K={row['K']} s={row['seed']} ({time.time()-t0:.0f}s)", flush=True)


def evaluate(a):
    P = load_pred(); D = read_out(a.out)
    print(f"predictions sha256 {pred_sha()}")
    for pl in sorted({k[0] for k in D}):
        fam = collections.defaultdict(lambda: collections.defaultdict(list)); cens = 0
        for (lay, M, E, s), pr in P.items():
            byK = D.get((pl, lay, M, E, s))
            if not byK: continue
            if needs_more(byK): cens += 1
            Jv = {K: float(x["J"]) for K, x in byK.items()}; Ks = sorted(Jv); am = min(Jv, key=Jv.get)
            rg = lambda K: Jv[min(Ks, key=lambda x: abs(x - K))] / Jv[am] - 1
            preds = dict(old=int(pr["old"]), new=int(pr["new"]))
            sf = [K for K in Ks if int(float(byK[K]["n_never"])) == 0]
            if sf:
                p, F = field(lay, M, E, L, s); rc, Pb = float(pr["rc_exante"]), float(pr["Pb_exante"])
                sc = pilot_scale(p, F, rc, Pb, sf[0], float(byK[sf[0]]["J_age"]))
                preds["pilot"] = surrogate_rule(p, F, rc, Pb, scale=sc)[0]
            for fm in (lay, "all"):
                for nm, K in preds.items():
                    fam[fm][nm + "_ex"].append(am == K); fam[fm][nm + "_w1"].append(abs(am - K) <= 1); fam[fm][nm + "_rg"].append(rg(K))
                if pr["br_lo"] != "": fam[fm]["br"].append(int(pr["br_lo"]) <= am <= int(pr["br_hi"]))
        print(f"\n== {pl}  (sweeps still needing extension: {cens})")
        for fm in ("all", "core", "paper", "ring"):
            R = fam[fm]
            if not R["old_ex"]: continue
            line = f"  {fm:5s} n={len(R['old_ex']):3d}"
            for nm in ("old", "new", "pilot"):
                if R[nm + "_ex"]:
                    g = np.array(R[nm + "_rg"])
                    line += (f" | {nm}: exact {np.mean(R[nm+'_ex']):.2f} w1 {np.mean(R[nm+'_w1']):.2f} "
                             f"regret med {np.median(g):.1%} p90 {np.quantile(g, .9):.1%} max {g.max():.1%}")
            if R["br"]: line += f" | bracket covers {np.mean(R['br']):.2f} of {len(R['br'])}"
            print(line)
        if pl == "sa":
            c, al, ri = fam["core"], fam["all"], fam["ring"]
            if c["old_rg"]:
                h1 = np.median(c["new_rg"]) < np.median(c["old_rg"]) and np.quantile(c["new_rg"], .9) < np.quantile(c["old_rg"], .9)
                print(f"  H1 (core: new lower median and p90 regret): {'HOLDS' if h1 else 'FAILS'}")
            if al["old_w1"]:
                print(f"  H2 (all: new within-one >= old): {'HOLDS' if np.mean(al['new_w1']) >= np.mean(al['old_w1']) else 'FAILS'}")
            if ri["br"]:
                print(f"  H3 (ring: bracket covers >= 90%): {'HOLDS' if np.mean(ri['br']) >= 0.9 else 'FAILS'} ({np.mean(ri['br']):.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["predict", "run", "evaluate"])
    ap.add_argument("out", nargs="?"); ap.add_argument("--sha"); ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--planners", nargs="+", default=["sa", "rr_tour", "cyc"])
    a = ap.parse_args()
    {"predict": lambda: predict(), "run": lambda: run(a), "evaluate": lambda: evaluate(a)}[a.mode]()
