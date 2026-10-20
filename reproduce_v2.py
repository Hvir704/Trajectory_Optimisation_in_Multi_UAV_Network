"""reproduce_v2.py -- check the simulator-v2 run files, assemble them, and regenerate every table, statistic and
figure of the paper from them. No simulation is run.

usage:  python reproduce_v2.py results_v2 [--out results_v2/report] [--only STEP ...] [--allow-v1]

1. CHECK   every CSV in results_v2 carries sim_version = run_grid.SIM_VERSION (--allow-v1 skips this, for testing the
           pipeline on old files), and no assembled file contains a duplicate run key.
2. ASSEMBLE the v2 files under the names the analysis scripts were written for (report/assembled/):
           sa_rerun.csv (SA, 18 primary cells), grid.csv (SA, paper M=400, 6 MJ, field-size cells), rr_tour_win.csv,
           rr_sweep_win.csv, cluster_win.csv, eval_all_v2.csv (C4, C4 sectors-only, cluster patrol), gzz_eval.csv,
           asghar_eval.csv, liu_all.csv, m40_*.csv, depot_*.csv, hetero.csv, mass.csv; then grid_final.csv.
3. DERIVE  the design-time verdicts on the v2 commute inputs (verdicts.csv for the held-out split, apriori_regret.csv
           for Table VII, tab:regret), the Fig. 4 deployment selection (selected.json, by the registered rule), and a censoring
           report: every sweep must meet the registered 1.5x rule on the union K grid.
4. RUN     every analysis step below; each writes report/<step>.txt. A failing step is reported and the run continues,
           so one missing file does not hide the rest; the summary at the end lists failures.
"""
import argparse, csv, glob, json, os, subprocess, sys, time
from kref import kref
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_grid import FIELDS, SIM_VERSION          # noqa: E402

KEYF = ["layout", "M", "Emax", "K", "seed", "coord", "replan", "divert", "q", "belief", "tau", "planner", "L", "Th"]
PRIMARY = lambda r: (int(float(r["M"])) <= 200 and float(r["Emax"]) in (1.5e6, 3e6) and float(r.get("L", 12600)) == 12600)


def read(path):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r and r.get("M") not in (None, "", "M")]


def key(r, extra=()):
    return tuple(str(float(r[k])) if k in ("M", "Emax", "K", "seed", "divert", "q", "tau", "L", "Th") else r[k]
                 for k in KEYF) + tuple(r.get(e, "") for e in extra)


def write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res"); ap.add_argument("--out", default=None); ap.add_argument("--only", nargs="*")
    ap.add_argument("--allow-v1", action="store_true")
    a = ap.parse_args()
    RES = os.path.abspath(a.res); OUT = os.path.abspath(a.out or os.path.join(RES, "report"))
    ASM = os.path.join(OUT, "assembled"); os.makedirs(ASM, exist_ok=True)
    R = lambda n: os.path.join(RES, n); A = lambda n: os.path.join(ASM, n)
    have = lambda n: os.path.exists(R(n)) and os.path.getsize(R(n)) > 0
    problems = []

    # ---------------------------------------------------------------- 1. check
    print("== 1. check")
    for p in sorted(glob.glob(os.path.join(RES, "*.csv"))):
        rows = read(p); v = Counter(r.get("sim_version") for r in rows)
        ok_versions = {SIM_VERSION, "legacy"} if os.path.basename(p).startswith("fixcheck") else {SIM_VERSION}
        bad = {k: n for k, n in v.items() if k not in ok_versions}
        if bad and not a.allow_v1:
            problems.append(f"{os.path.basename(p)}: rows with sim_version {bad}")
        print(f"   {os.path.basename(p):32s} {len(rows):6d} rows  versions {dict(v)}")
    if problems and not a.allow_v1:
        print("\nSTOP: files from another simulator version:\n   " + "\n   ".join(problems)); sys.exit(1)

    # ---------------------------------------------------------------- 2. assemble
    print("\n== 2. assemble")
    def cat(names, planner=None, filt=None, extra=()):
        out, seen, dup, rows = [], set(), 0, []
        for n in names:
            if not have(n): continue
            for r in read(R(n)):
                if planner and r.get("planner") != planner: continue
                if filt and not filt(r): continue
                r["_fill"] = "_fill" in n; rows.append(r)
        # fill files written by run_fill.py carry no depot columns. Each depot_* base file holds one depot position, so
        # its fills inherit it (verified: their r_max and K_reach_i equal the base rows'); without this, fill rows formed
        # separate instances with their own K_ref (review 3, Low-8)
        if extra:
            vals = {tuple(r.get(e, "") for e in extra) for r in rows if not r["_fill"]}
            if len(vals) == 1 and all(v not in ("", None) for v in next(iter(vals))):
                for r in rows:
                    if r["_fill"] and all(r.get(e, "") in ("", None) for e in extra):
                        for e, v in zip(extra, next(iter(vals))): r[e] = v
        for r in rows:
            k = key(r, extra)
            if k in seen: dup += 1; continue
            seen.add(k); out.append(r)
        # K_ref: the first K of each instance's ORIGINAL sweep (base file, not a fill); the rule's measured commute
        # inputs are read at this K (kref.py), so they do not move when a fill extends a sweep in either direction
        def num(x):
            try: return str(float(x))
            except (TypeError, ValueError): return x if x is not None else ""
        # numeric fields are normalised ('43200' and '43200.0' are the same horizon), as in key()
        ikey = lambda r: tuple(num(r.get(k, "")) for k in ("layout", "M", "Emax", "L", "seed")) + (r.get("planner", "sa"),
                          r.get("coord", ""), r.get("replan", "")) + tuple(num(r.get(k, "")) for k in ("divert", "q")) + (
                          r.get("belief", ""),) + tuple(num(r.get(k, "")) for k in ("tau", "Th")) + (
                          r.get("alloc", ""), r.get("mass_model", "")) + tuple(num(r.get(e, "")) for e in extra)
        base, allk = {}, {}
        for r in out:
            K = int(float(r["K"])); i = ikey(r)
            allk[i] = min(allk.get(i, K), K)
            if not r["_fill"]: base[i] = min(base.get(i, K), K)
        for r in out:
            i = ikey(r); r["K_ref"] = base.get(i, allk[i]); del r["_fill"]
        return out, dup
    def emit(name, names, planner=None, filt=None, fields=FIELDS, extra=()):
        fields = list(fields) + ["K_ref"]
        rows, dup = cat(names, planner, filt, extra)
        if dup: problems.append(f"{name}: {dup} duplicate run key(s) across {names} (first kept)")
        if rows: write(A(name), rows, fields)
        print(f"   {name:24s} {len(rows):6d} rows from {[n for n in names if have(n)]}" + (f"  DUPLICATES {dup}" if dup else ""))
        return rows
    std = lambda r: (r["coord"] == "exclude" and r["replan"] == "launch" and float(r["Th"]) == 43200
                     and float(r.get("q", 0)) == 0 and r.get("belief", "prior") == "prior" and float(r.get("tau", 0)) == 0
                     and str(r.get("divert", "0")) in ("0", "0.0"))
    sa_files = ["sa_grid.csv", "sa_fill.csv"]
    emit("sa_rerun.csv", sa_files, "sa", lambda r: std(r) and PRIMARY(r))
    emit("grid.csv", ["sa_m400.csv", "sa_6mj.csv"] + [f"sa_L{L}.csv" for L in (8000, 10000, 16000, 20000)] + ["sa_fill.csv"],
         "sa", lambda r: std(r) and not PRIMARY(r))
    emit("rr_tour_win.csv", ["rr_tour.csv", "rr_tour_fill.csv"], "rr_tour", std)
    emit("rr_sweep_win.csv", ["rr_sweep.csv", "rr_sweep_fill.csv"], "rr_sweep", std)
    emit("cluster_win.csv", ["cluster_patrol.csv", "cluster_patrol_fill.csv"], "cluster_patrol", std)
    emit("eval_all_v2.csv", ["cyc.csv", "cyc_fill.csv", "cyc_sector0.csv", "cyc_sector0_fill.csv",
                             "cluster_patrol.csv", "cluster_patrol_fill.csv"], None, std)
    emit("gzz_eval.csv", ["gzz_rev.csv", "gzz_rev_fill.csv"], "gzz_rev", std)
    emit("asghar_eval.csv", ["asghar_rev.csv", "asghar_rev_fill.csv"], "asghar_rev", std)
    emit("liu_all.csv", ["liu_mpga.csv", "liu_mpga_fill.csv"], "liu_mpga", std)
    emit("m40_sa.csv", ["m40_sa.csv"]); emit("m40_rr.csv", ["m40_rr_tour.csv"]); emit("m40_cluster.csv", ["m40_cluster_patrol.csv", "m40_cluster_patrol_fill.csv"])
    # Table XIII (tab:baseline) bottom: CP-SAT at its default budget on paper, at the converged threefold budget on ring
    cp = [r for r in (read(R("m40_cpsat_d5.csv")) if have("m40_cpsat_d5.csv") else []) if r["layout"] == "paper"]
    cp += read(R("m40_cpsat_d15.csv")) if have("m40_cpsat_d15.csv") else []
    for r in cp: r["planner"] = "cpsat"
    k0 = {}
    for r in cp: i = (r["layout"], r["M"], r["Emax"], r["seed"]); k0[i] = min(k0.get(i, int(float(r["K"]))), int(float(r["K"])))
    for r in cp: r["K_ref"] = k0[(r["layout"], r["M"], r["Emax"], r["seed"])]
    if cp: write(A("m40_cp_final.csv"), cp, FIELDS + ["K_ref"])
    print(f"   {'m40_cp_final.csv':24s} {len(cp):6d} rows")
    DF = FIELDS + ["depot_x", "depot_y"]
    for d in ("q", "e"):
        emit(f"depot_{d}.csv", [f"depot_{d}.csv", f"depot_{d}_fill.csv"], "sa", None, DF, ("depot_x", "depot_y"))
        for pl in ("cyc", "cluster_patrol"):
            emit(f"depot_{d}_{pl}.csv", [f"depot_{d}_{pl}.csv", f"depot_{d}_{pl}_fill.csv"], pl, None, DF, ("depot_x", "depot_y"))
    emit("hetero.csv", ["hetero.csv"], None, None, FIELDS + ["alloc"], ("alloc",))
    emit("mass.csv", ["mass.csv"], None, None, FIELDS + ["mass_model"])
    # ablation: every variant in one file for ablation.py
    emit("ablation.csv", sorted(os.path.basename(p) for p in glob.glob(R("abl_*.csv"))))
    gf = A("grid_final.csv")
    if os.path.exists(A("sa_rerun.csv")) and os.path.exists(A("grid.csv")):
        r = subprocess.run([sys.executable, "build_grid_final.py", A("sa_rerun.csv"), A("grid.csv"), gf], cwd=HERE,
                           capture_output=True, text=True)
        print("   grid_final.csv           " + (r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "FAILED: " + r.stdout + r.stderr))
        if r.returncode: problems.append("build_grid_final failed")

    # ---------------------------------------------------------------- 3. derive
    print("\n== 3. derive")
    if os.path.exists(gf):
        dump = os.path.join(OUT, "verdicts_global.json")
        subprocess.run([sys.executable, "criterion_variants.py", gf, "-", f"--dump={dump}"], cwd=HERE,
                       stdout=subprocess.DEVNULL, check=True)
        VERD = json.load(open(dump))
        import math
        inst = defaultdict(dict)
        for r in read(gf):
            if float(r["L"]) == 12600: inst[(r["layout"], int(float(r["M"])), float(r["Emax"]), int(float(r["seed"])))][int(float(r["K"]))] = r
        apr, ver = [], []
        for k, byK in sorted(inst.items()):
            r0 = byK[kref(byK)]; kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"])
            rb = min(kr, kc) == kr; v = VERD.get(str(k))
            J = {K: float(x["J"]) for K, x in byK.items()}; l = kr if rb else int(round(kc))
            apr.append(dict(lay=k[0], M=k[1], E=k[2], seed=k[3], passes=("NA" if not rb else str(bool(v))),
                            reg=(J[l] / min(J.values()) - 1) if l in J else "", reach_bound=str(rb), ratio=""))
            if k[1] <= 200 and k[2] in (1.5e6, 3e6):
                ver.append(dict(layout=k[0], M=k[1], Emax=k[2], seed=k[3],
                                verdict_current=("not_applicable" if not rb else ("pass" if v else "flag")), verdict_superseded=""))
        write(A("apriori_regret.csv"), apr, ["lay", "M", "E", "seed", "passes", "reg", "reach_bound", "ratio"])
        write(A("verdicts.csv"), ver, ["layout", "M", "Emax", "seed", "verdict_current", "verdict_superseded"])
        c = Counter((x["layout"], x["verdict_current"]) for x in ver)
        print(f"   verdicts (held-out 216): {dict(sorted(c.items()))}")
    # Fig. 4 selection rule (registered): per family (paper, ring), the M=100, 1.5 MJ deployment whose C4-vs-cluster
    # own-optimum ratio is closest to the median of that ratio over the WHOLE family (all M and budgets)
    if os.path.exists(A("eval_all_v2.csv")):
        import numpy as np
        d = defaultdict(lambda: defaultdict(dict))
        for r in read(A("eval_all_v2.csv")):
            if r["layout"] in ("paper", "ring"):
                d[(r["layout"], int(float(r["M"])), float(r["Emax"]), int(float(r["seed"])))][r["planner"]][int(float(r["K"]))] = float(r["J"])
        sel = {}
        for fam in ("paper", "ring"):
            rat = {k: min(b["cyc"].values()) / min(b["cluster_patrol"].values()) - 1 for k, b in d.items()
                   if k[0] == fam and "cyc" in b and "cluster_patrol" in b}
            cand = {k: v for k, v in rat.items() if k[1] == 100 and k[2] == 1.5e6}
            if cand:
                med = float(np.median(list(rat.values())))
                sel[fam] = min(cand, key=lambda k: (abs(cand[k] - med), k[3]))[3]
        json.dump(sel, open(A("selected.json"), "w")); print(f"   Fig. 4 deployments (registered rule): {sel}  (v1: paper 10, ring 12)")
    # censoring report: every planner's sweeps on the union grid must meet the registered rule
    import fill_tail
    os.environ["ALLOW_V1"] = "1" if a.allow_v1 else os.environ.get("ALLOW_V1", "")
    def merged(pairs):
        """Union K grid exactly as the fills build it: the listed (file, planner) pairs and nothing else."""
        u = {}
        for n, pl in pairs:
            if os.path.exists(A(n)):
                for k, b in fill_tail.load([A(n)], planner=pl).items(): u.setdefault(k, {}).update(b)
        return u
    # = $U in RERUN_V2.md / run_all_ryzen.ps1: SA, C4, cluster patrol, Gao, Asghar, Liu, tour patrol.
    # (cyc_sector0 and rr_sweep are filled ON this grid but do not add to it.)
    U_PAIRS = [("sa_rerun.csv", None), ("eval_all_v2.csv", "cyc"), ("cluster_win.csv", "cluster_patrol"),
               ("gzz_eval.csv", "gzz_rev"), ("asghar_eval.csv", "asghar_rev"), ("liu_all.csv", "liu_mpga"),
               ("rr_tour_win.csv", "rr_tour")]
    union = merged(U_PAIRS)
    unions = {"primary": union, "none": None,
              "q": merged([(f"depot_q{x}.csv", None) for x in ("", "_cyc", "_cluster_patrol")]),
              "e": merged([(f"depot_e{x}.csv", None) for x in ("", "_cyc", "_cluster_patrol")])}
    print("   censoring (top and, since v2.2.1, bottom edge): runs fill_tail.py would still add (must all be 0):")
    for n, pl, un in (("sa_rerun.csv", None, "primary"), ("grid.csv", None, "none"), ("rr_tour_win.csv", "rr_tour", "primary"),
                      ("rr_sweep_win.csv", "rr_sweep", "primary"), ("cluster_win.csv", "cluster_patrol", "primary"),
                      ("eval_all_v2.csv", "cyc", "primary"), ("eval_all_v2.csv", "cyc_sector0", "primary"),
                      ("gzz_eval.csv", "gzz_rev", "primary"), ("asghar_eval.csv", "asghar_rev", "primary"),
                      ("liu_all.csv", "liu_mpga", "primary"),
                      ("depot_q.csv", None, "q"), ("depot_q_cyc.csv", None, "q"), ("depot_q_cluster_patrol.csv", None, "q"),
                      ("depot_e.csv", None, "e"), ("depot_e_cyc.csv", None, "e"), ("depot_e_cluster_patrol.csv", None, "e")):
        if not os.path.exists(A(n)): continue
        dd = fill_tail.load([A(n)], planner=pl)
        jobs = fill_tail.needed(dd, unions[un], 60, True)
        over = len(fill_tail.needed(dd, unions[un], 10**6, True)) - len(jobs)     # wanted K above kmax = 60
        print(f"      {n:26s} {pl or '-':15s} {len(jobs):4d}" + (f"   + {over} wanted above K=60 (cannot be run)" if over else ""))
        if over: problems.append(f"{n}: {over} sweeps would need K > 60; the extension rule cannot be met there")
        if jobs: problems.append(f"{n}: {len(jobs)} runs still needed to meet the extension rule (rerun RERUN_V2 section C)")

    # ---------------------------------------------------------------- 4. run
    o = lambda n: os.path.join(OUT, n)
    STEPS = [
        ("table4", "Table IV (tab:law), Sec. VII-B statistics, warning indicator, Table X (tab:L, field size)", ["make_table4.py", gf]),
        ("estimators", "Appendix B: choice of estimator", ["make_estimators.py", gf]),
        ("strand_rad", "Table V (tab:strand), Prop. 8 at the radial scale", ["strand_final.py", gf]),
        ("strand_consistent", "Prop. 6 at the consistent scale (Lemma: cost scale)", ["strand_final.py", gf], {"P6_SCALE": "consistent"}),
        ("strand_old_algebra", "Sec. VII-C: the earlier form of Prop. 6 with Q held at Q_K (exact / within-one)", ["strand_final.py", gf], {"P6_ALGEBRA": "old"}),
        ("strand_planners", "Sec. VII-C: other planners' optima vs l and Prop. 6 on the flagged held-out deployments",
         ["strand_planners.py", A("sa_rerun.csv"), A("verdicts.csv"), A("eval_all_v2.csv"), A("gzz_eval.csv"), A("asghar_eval.csv")]
         + ([A("liu_all.csv")] if os.path.exists(A("liu_all.csv")) else [])),
        ("scale_lemma", "Lemma (cost scale): verdict pairs, alpha, regret of changed verdicts", ["scale_lemma.py", gf]),
        ("criterion", "criterion (5): pass/flag regret by family (global form)", ["criterion_variants.py", gf, "-", "--tex"]),
        ("table8", "Table VII (tab:regret)", ["make_table8.py", gf, A("apriori_regret.csv")]),
        ("exante", "Sec. VII-J ex-ante commute inputs (incl. the fixed-point existence count)", ["exante_rc.py", gf, "--out", A("exante.csv")]),
        ("criterion_exante_fp", "Sec. VII-J criterion with ex-ante inputs at the commute fixed point (not used in the paper)",
         ["criterion_variants.py", gf, "-", A("exante.csv"), "--fixedpoint", f"--dump={o('verdicts_exante_fp.json')}"]),
        ("criterion_exante_min", "Sec. VII-J criterion with ex-ante inputs at the smallest candidate K (the paper's estimator)",
         ["criterion_variants.py", gf, "-", A("exante.csv"), f"--dump={o('verdicts_exante_min.json')}"]),
        ("verdict_changes", "Sec. VII-J: verdicts changed by ex-ante inputs (and in which direction)", None),
        ("reach_bracket", "Sec. V-A reach bracket, K_flight comparison", ["reach_bracket.py", gf]),
        ("perturb_inputs", "Sec. VII-J sensitivity to known inputs", ["perturb_inputs.py", gf]),
        ("theory", "Sec. V and VII-I: r_c(K), P_bar spread, tau_hop(K), Prop. 3 with measured tau_hop", ["theory_checks.py", gf, "--tour", A("rr_tour_win.csv")]),
        ("bootstrap", "95% intervals (Sec. VII-B, VII-K), per-cell CSV", ["bootstrap_ci.py", gf, A("sa_rerun.csv"), A("rr_tour_win.csv"),
                                                                        A("rr_sweep_win.csv"), A("m40_sa.csv"), A("m40_rr.csv"), A("m40_cp_final.csv"), f"--out={o('per_cell_ci.csv')}"]),
        ("baseline_patrols", "Table XIII (tab:baseline): tour and sweep patrol rows", ["baseline_table.py", A("sa_rerun.csv"), A("rr_tour_win.csv"), A("rr_sweep_win.csv"), "--by-family", "--rule-from", "sa"]),
        ("baseline_cluster", "Table XIII (tab:baseline): cluster patrol row", ["baseline_table.py", A("sa_rerun.csv"), A("cluster_win.csv"), "--rule-from", "sa"]),
        ("baseline_m40", "Table XIII (tab:baseline) bottom (M=40)", ["baseline_table.py", A("m40_sa.csv"), A("m40_rr.csv"), A("m40_cp_final.csv"), "--by-family", "--rule-from", "sa"]),
        ("baseline_cluster_m40", "Table XIII (tab:baseline) bottom: cluster patrol row", ["baseline_table.py", A("m40_sa.csv"), A("m40_cluster.csv"), "--rule-from", "sa"]),
        ("paired_diff", "Sec. VII-K paired agreement differences", ["paired_diff.py", A("sa_rerun.csv"), A("rr_tour_win.csv")]),
        ("cluster", "Sec. VII-K cluster patrol vs SA", ["cluster_compare.py", A("sa_rerun.csv"), A("cluster_win.csv")]),
        ("planners", "Tables XIII (tab:baseline, dagger rows), XIV (tab:split), XV (tab:published); C4 vs SA; cost of sizing",
         ["planners_v2.py", "--sa", A("sa_rerun.csv"), "--files"] + [A(n) for n in ("eval_all_v2.csv", "gzz_eval.csv", "asghar_eval.csv", "liu_all.csv")
                                                                     if os.path.exists(A(n))] + ["--verdicts", A("verdicts.csv")]),
        ("c4_sectors", "C4 paragraph: equal sectors without search vs the published method",
         ["own_opt_compare.py", A("eval_all_v2.csv"), A("eval_all_v2.csv"), "--pa", "cluster_patrol", "--pb", "cyc_sector0"]),
        ("liu", "Liu et al. paragraph (registered analysis)", ["analyze_liu.py", "--liu", A("liu_all.csv"), "--sa", A("sa_rerun.csv"),
                                                              "--others", A("eval_all_v2.csv"), A("gzz_eval.csv"), A("asghar_eval.csv"), "--verdicts", A("verdicts.csv")]),
        ("depot", "Table XI (tab:depot)", ["depot_table.py", A("depot_q.csv"), A("depot_e.csv")]),
        ("depot_c4_q", "depot: C4 vs cluster patrol, quarter point", ["own_opt_compare.py", A("depot_q_cluster_patrol.csv"), A("depot_q_cyc.csv")]),
        ("depot_c4_e", "depot: C4 vs cluster patrol, edge", ["own_opt_compare.py", A("depot_e_cluster_patrol.csv"), A("depot_e_cyc.csv")]),
        ("hetero", "Sec. VII-C reserved share vs equal shares", ["analyze_hetero.py", A("sa_rerun.csv"), A("hetero.csv")]),
        ("mass", "Remark 1 battery mass", ["analyze_mass.py", A("mass.csv"), A("sa_rerun.csv")]),
        ("mass_static", "Remark 1 battery share and static fixed point (pack = E_max/K)", ["mass_static.py", gf]),
        ("ablation", "Table III (tab:ablation, K=4); per-leg + diversion against per-leg alone", ["ablation.py", A("ablation.csv"), "4"]),
        ("ablation_K2", "Sec. VI hover budgeting at K=2", ["ablation.py", A("ablation.csv"), "2"]),
        ("singlecell", "Tables VI (tab:horizon), XII (tab:coord), XVI (tab:info); per-leg paragraph (Sec. VII-H)", ["singlecell_v2.py", RES]),
        ("outage", "Sec. VII-B grounding test", None),
        ("regularity", "Sec. V-B CV of inter-visit intervals; Sec. VII-K rate/allocation/regularity decomposition", None),
        ("reserve", "Sec. VII-A: reserve use, departure skips and skipped stops over the SA grid", None),
        ("sweep_stats", "Sec. VII-A: run count, largest K per cell, sweeps extended by the fills, optima moved by them", None),
        ("edges", "optima at the top of their sweep in the files the censoring check does not cover (M=40, hetero, mass)", None),
        ("cpsat_proofs", "Sec. VII-K: CP-SAT share of decisions proven optimal", None),
        ("alpha", "Table XVII (tab:alpha, fitted commute exponent) and App. B per-sortie energy regressions (pooled, between-run, within-run)",
         ["alpha_fit.py", gf, "--sorties", R("sortie_ring.csv")]),
        ("relcost_drift", "Sec. V-B drift of the relative-cost profile (paper M=100 1.5 MJ, seeds 1-6)",
         ["relcost_drift.py", os.path.join(RES, "nodes", "nodes_sa_paper_M100_E1500000_K*_s*_exclude_launch.npz")]),
        ("eta_mc", "Sec. IV-D eta(M): formula and full-model Monte Carlo; hotspot sensors beyond the inscribed circle (minutes)",
         ["eta_mc.py", "2000"]),
        ("commute_exact", "Sec. V-B Corollary (exact productive power): hover conservation and K_H against K_commute",
         ["commute_exact.py", gf]),
        ("paper_numbers", "remaining numbers of the text (each labelled with its section in the output)", ["paper_numbers.py", RES, ASM]),
        ("c4_classes", "Sec. VII-K: C4 frequency classes at its own optimum (rebuilds the 216 designs; minutes)", ["c4_classes.py", A("eval_all_v2.csv")]),
        ("gao_sizing", "Sec. VII-K: Gao et al.'s own fleet-size rule (Alg. 2) on the primary grid (minutes)",
         ["gao_sizing.py", gf, A("gzz_eval.csv")]),
        ("fig_JvsK_regime", "Figs. 2-3", ["make_figures.py", gf]),
        ("fig_planners", "Fig. 5 (planners, M=40)", ["fig_planners.py", A("m40_sa.csv"), A("m40_rr.csv"), A("m40_cp_final.csv"), o("fig_planners.pdf")]),
        ("fig_five", "Fig. 4 (five planners)", ["fig_five.py", o("fig_five.pdf"), "--sa", A("sa_rerun.csv"), "--eval", A("eval_all_v2.csv"),
                                                "--extra", A("gzz_eval.csv"), A("asghar_eval.csv"), "--selected", A("selected.json")]),
    ]
    failed = []
    print("\n== 4. analyses")
    for st in STEPS:
        name, feeds, cmd = st[0], st[1], st[2]; env = dict(os.environ, **(st[3] if len(st) > 3 else {}))
        if a.only and name not in a.only: continue
        log = o(f"{name}.txt"); t0 = time.time()
        if cmd is None:
            with open(log, "w") as f:
                try:
                    {"outage": outage_summary, "cpsat_proofs": cpsat_summary, "verdict_changes": verdict_changes,
                     "regularity": regularity_summary, "reserve": reserve_summary, "sweep_stats": sweep_stats,
                     "edges": edges_summary}[name](
                        OUT if name == "verdict_changes" else (ASM if name in ("reserve", "edges") else RES), f)
                    ok = True
                except Exception as e:
                    f.write(f"FAILED: {e!r}\n"); ok = False
        else:
            outs = {cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i] == "--out"}
            missing = [c for c in cmd if c.endswith(".csv") and os.path.isabs(c) and not os.path.exists(c) and c not in outs]
            if missing:
                open(log, "w").write("missing inputs: " + ", ".join(missing) + "\n"); ok = False
            else:
                with open(log, "w") as f:
                    ok = subprocess.run([sys.executable] + cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env).returncode == 0
        print(f"   [{'ok' if ok else 'FAILED'}] {name:22s} {time.time()-t0:6.1f}s  {feeds}")
        if not ok: failed.append(name)
    if os.path.isdir(os.path.join(HERE, "figs")):
        print("   (make_figures.py writes figs/fig_JvsK.pdf and figs/fig_regime.pdf in the repository folder)")
    print("\n== summary")
    for p in problems: print("   PROBLEM:", p)
    for n in failed: print(f"   FAILED step: {n} (see {o(n + '.txt')})")
    if not problems and not failed: print("   all checks passed, all steps ran")
    print(f"   outputs in {OUT}")


def verdict_changes(OUT, f):
    base = json.load(open(os.path.join(OUT, "verdicts_global.json")))
    for tag in ("exante_fp", "exante_min"):
        v = json.load(open(os.path.join(OUT, f"verdicts_{tag}.json")))
        common = [k for k in base if k in v]
        to_pass = sum(1 for k in common if not base[k] and v[k]); to_flag = sum(1 for k in common if base[k] and not v[k])
        f.write(f"{tag}: {to_pass + to_flag} of {len(common)} reach-bound verdicts change ({to_pass} toward passing, {to_flag} toward flagging)\n")


def reserve_summary(ASM, f):
    """Per family over the SA primary grid: sorties whose next stop was skipped because the usable energy could not
    cover it plus the return (breach_frac; mid-route ones are trunc_frac), landings that dipped into the reserve,
    and landings that exhausted it."""
    import numpy as np
    rows = read(os.path.join(ASM, "sa_rerun.csv"))
    for fam in ("paper", "core", "ring", "all"):
        x = [r for r in rows if fam == "all" or r["layout"] == fam]
        if not x: continue
        b = np.array([float(r["breach_frac"]) for r in x]); t = np.array([float(r["trunc_frac"]) for r in x])
        dips = sum(int(float(r["reserve_dips"])) for r in x); viol = sum(int(float(r["reserve_violations"])) for r in x)
        ds = np.array([float(r.get("depart_skip_frac") or 0) for r in x]); um = max(float(r.get("reserve_used_max_J") or 0) for r in x)
        f.write(f"{fam:5s} runs {len(x):5d}  arrival skips (true dwell too long) median {np.median(b):.3f} p90 {np.quantile(b, .9):.3f}  "
                f"departure skips median {np.median(ds):.3f} p90 {np.quantile(ds, .9):.3f}  "
                f"mid-route truncation median {np.median(t):.3f} p90 {np.quantile(t, .9):.3f}  "
                f"landings into reserve {dips} (largest {um / 1e3:.2f} kJ)  reserve exhausted {viol}\n")


def sweep_stats(RES, f):
    """Counts for Sec. VI-A: runs in the primary analysis file, largest K per cell, how many sweeps the fills extended
    and whether extending moved any per-instance optimum (argmin with vs without the fill rows)."""
    def load(names):
        d = defaultdict(dict)
        for n in names:
            p = os.path.join(RES, n)
            if not os.path.exists(p): continue
            for r in read(p):
                if r.get("planner") != "sa" or r["coord"] != "exclude" or r["replan"] != "launch" or float(r["Th"]) != 43200: continue
                if str(r.get("divert", "0")) not in ("0", "0.0") or float(r.get("q", 0)) != 0 or r.get("belief", "prior") != "prior": continue
                d[(r["layout"], int(float(r["M"])), float(r["Emax"]), float(r["L"]), int(float(r["seed"])))][int(float(r["K"]))] = float(r["J"])
        return d
    base_names = ["sa_grid.csv", "sa_m400.csv", "sa_6mj.csv"] + [f"sa_L{L}.csv" for L in (8000, 10000, 16000, 20000)]
    base, full = load(base_names), load(base_names + ["sa_fill.csv"])
    f.write(f"runs (standard SA rows, grid + fills): {sum(len(b) for b in full.values())} over {len(full)} instances\n")
    ext = [k for k in full if len(full[k]) > len(base.get(k, {}))]
    moved = [k for k in ext if k in base and min(base[k], key=base[k].get) != min(full[k], key=full[k].get)]
    f.write(f"sweeps extended by the fills: {len(ext)}; per-instance optimum moved by the extension: {len(moved)}\n")
    dn = [k for k in ext if k in base and min(full[k]) < min(base[k])]; up = [k for k in ext if k in base and max(full[k]) > max(base[k])]
    f.write(f"   of these extended upward {len(up)}, downward {len(dn)} (both {len(set(up) & set(dn))}); optima moved downward "
            f"{sum(1 for k in moved if min(full[k], key=full[k].get) < min(base[k], key=base[k].get))}\n")
    for k in moved: f.write(f"   moved: {k}  {min(base[k], key=base[k].get)} -> {min(full[k], key=full[k].get)}\n")
    cells = defaultdict(list)
    for k, b in full.items(): cells[k[:4]].append(max(b))
    for c in sorted(cells): f.write(f"   {c}: largest K swept {max(cells[c])}, extended instances {sum(1 for k in ext if k[:4] == c)}\n")


def edges_summary(ASM, f):
    for n in ("m40_sa.csv", "m40_rr.csv", "m40_cluster.csv", "m40_cp_final.csv", "hetero.csv", "mass.csv"):
        p = os.path.join(ASM, n)
        if not os.path.exists(p): f.write(f"{n}: missing\n"); continue
        d = defaultdict(dict)
        for r in read(p): d[(r["layout"], r["M"], r["Emax"], r["seed"], r.get("alloc", ""))][int(float(r["K"]))] = float(r["J"])
        top = [k for k, b in d.items() if min(b, key=b.get) == max(b)]
        bot = [k for k, b in d.items() if min(b, key=b.get) == min(b) and min(b) > 1]
        r15 = [k for k, b in d.items() if not all(b[K] > 1.5 * min(b.values()) for K in sorted(b)[-3:])]
        f.write(f"{n}: {len(top)} of {len(d)} instances have their optimum at the largest K swept{(': ' + str(top[:6])) if top else ''}; "
                f"{len(bot)} at the smallest K swept (> 1){(': ' + str(bot[:6])) if bot else ''}; "
                f"{len(r15)} do not meet the 1.5x rule (three largest K > 1.5 x min)\n")
    # v2.2.1: own optima moved by fleet sizes BELOW each instance's original sweep (K < K_ref), per planner
    f.write("\nown optima moved by the fleet sizes below each instance's original sweep (K < K_ref; v2.2.1 downward fill):\n")
    for n, pls in (("sa_rerun.csv", None), ("eval_all_v2.csv", ("cyc", "cluster_patrol", "cyc_sector0")), ("gzz_eval.csv", None),
                   ("asghar_eval.csv", None), ("liu_all.csv", None), ("rr_tour_win.csv", None),
                   ("depot_q.csv", None), ("depot_q_cyc.csv", None), ("depot_q_cluster_patrol.csv", None),
                   ("depot_e.csv", None), ("depot_e_cyc.csv", None), ("depot_e_cluster_patrol.csv", None)):
        p = os.path.join(ASM, n)
        if not os.path.exists(p): continue
        d, kr = defaultdict(dict), {}
        for r in read(p):
            if float(r.get("Th", 43200)) != 43200: continue
            # each depot_* file holds one depot position; its fill rows carry no depot columns, so the depot is not part
            # of the key (keying on it split every filled instance in two; review 3, Low-8)
            k = (r.get("planner", ""), r["layout"], r["M"], r["Emax"], r.get("L", ""), r["seed"])
            d[k][int(float(r["K"]))] = float(r["J"])
            if r.get("K_ref") not in (None, ""): kr[k] = int(float(r["K_ref"]))
        for pl in (pls or (None,)):
            keys = [k for k in d if pl is None or k[0] == pl]
            mv = []
            for k in keys:
                b = d[k]; lo = {K: J for K, J in b.items() if K >= kr.get(k, min(b))}
                if lo and min(b, key=b.get) != min(lo, key=lo.get):
                    mv.append((k, min(lo, key=lo.get), min(b, key=b.get), min(b.values()) / min(lo.values()) - 1))
            f.write(f"  {n}{('/' + pl) if pl else ''}: {len(mv)} of {len(keys)}"
                    + (": J " + ", ".join(f"{100 * x[3]:+.1f}%" for x in mv[:12]) + "; layouts " + str(sorted({x[0][1] for x in mv})) if mv else "") + "\n")


def regularity_summary(RES, f):
    import numpy as np
    files = sorted(glob.glob(os.path.join(RES, "regularity*.csv")))
    if not files: raise FileNotFoundError("no regularity*.csv")
    g = defaultdict(list)
    for p in files:
        for r in read(p): g[(os.path.basename(p), r["planner"], r["layout"], r["M"], r["Emax"], int(r["K"]))].append(r)
    cols = ("CV", "cv2_wf", "share_sparse", "tau_hop", "tau_hop_per_hop", "Gamma", "E_alloc", "E_reg", "r_c", "J")
    f.write("file planner layout M Emax K n | " + " ".join(cols) + "  (means over seeds; Gamma in visits/h)\n")
    for k, rs in sorted(g.items()):
        vals = []
        for c in cols:
            x = [float(r[c]) for r in rs if r.get(c) not in (None, "", "nan")]
            v = np.mean(x) * (3600 if c == "Gamma" else 1) if x else float("nan"); vals.append(f"{v:.4g}")
        f.write(" ".join(str(x) for x in k) + f" {len(rs)} | " + " ".join(vals) + "\n")


def outage_summary(RES, f):
    rows = read(os.path.join(RES, "outage24.csv"))
    by = defaultdict(dict)
    for r in rows: by[(r["layout"], r["M"], r["Emax"], r["seed"])][(r["which"], float(r["G_h"]))] = r
    for k, b in sorted(by.items()):
        try:
            j = lambda w, g: float(b[(w, g)]["J"])
            f.write(f"{k}: no grounding  K_flight vs K_reach {j('K_flight', 0) / j('K_reach', 0) - 1:+.1%};  "
                    f"3 h grounding {j('K_flight', 3) / j('K_reach', 3) - 1:+.1%};  gray lost (flight) "
                    f"{b[('K_flight', 3.0)]['gray_lost']}/{b[('K_flight', 3.0)]['n_gray']}, (reach) "
                    f"{b[('K_reach', 3.0)]['gray_lost']}/{b[('K_reach', 3.0)]['n_gray']}\n")
        except KeyError as e:
            f.write(f"{k}: incomplete ({e})\n")


def cpsat_summary(RES, f):
    for n in ("m40_cpsat_d5.csv", "m40_cpsat_d15.csv", "m40_cpsat_d45.csv"):
        p = os.path.join(RES, n)
        if not os.path.exists(p): f.write(f"{n}: missing\n"); continue
        rows = [r for r in read(p) if r.get("cp_decisions") not in (None, "")]
        for lay in ("paper", "ring"):
            x = [r for r in rows if r["layout"] == lay]
            if not x: continue
            nd = sum(int(float(r["cp_decisions"])) for r in x); no = sum(int(float(r["cp_optimal"])) for r in x)
            f.write(f"{n} {lay}: {no}/{nd} decisions proven optimal ({no / max(nd, 1):.1%})\n")
            for K in sorted({int(r["K"]) for r in x}):
                y = [r for r in x if int(r["K"]) == K]
                fr = [int(float(r["cp_optimal"])) / max(int(float(r["cp_decisions"])), 1) for r in y]
                f.write(f"     K={K}: proven share per run {min(fr):.2f}-{max(fr):.2f}\n")
    # the runs Table XIII (tab:baseline) uses: default budget on paper, threefold budget on ring (review 4, M8b)
    tot = {}
    for n, lay in (("m40_cpsat_d5.csv", "paper"), ("m40_cpsat_d15.csv", "ring")):
        p = os.path.join(RES, n)
        if not os.path.exists(p): continue
        x = [r for r in read(p) if r.get("cp_decisions") not in (None, "") and r["layout"] == lay]
        tot[lay] = (sum(int(float(r["cp_optimal"])) for r in x), sum(int(float(r["cp_decisions"])) for r in x))
    if len(tot) == 2:
        no = sum(v[0] for v in tot.values()); nd = sum(v[1] for v in tot.values())
        f.write(f"runs used in Table XIII (paper at the default budget, ring at three times it): {no}/{nd} proven optimal ({no / nd:.1%})\n")


if __name__ == "__main__":
    main()
