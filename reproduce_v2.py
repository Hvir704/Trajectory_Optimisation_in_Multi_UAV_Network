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
           for Table VIII), the Fig. 5 deployment selection (selected.json, by the registered rule), and a censoring
           report: every sweep must meet the registered 1.5x rule on the union K grid.
4. RUN     every analysis step below; each writes report/<step>.txt. A failing step is reported and the run continues,
           so one missing file does not hide the rest; the summary at the end lists failures.
"""
import argparse, csv, glob, json, os, subprocess, sys, time
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
        if os.path.basename(p).startswith(("fixcheck", "regularity")): continue   # legacy-vs-v2 pairs; own-sim diagnostics
        rows = read(p); v = Counter(r.get("sim_version") for r in rows)
        bad = {k: n for k, n in v.items() if k != SIM_VERSION}
        if bad and not a.allow_v1:
            problems.append(f"{os.path.basename(p)}: rows with sim_version {bad}")
        print(f"   {os.path.basename(p):32s} {len(rows):6d} rows  versions {dict(v)}")
    if problems and not a.allow_v1:
        print("\nSTOP: files from another simulator version:\n   " + "\n   ".join(problems)); sys.exit(1)

    # ---------------------------------------------------------------- 2. assemble
    print("\n== 2. assemble")
    def cat(names, planner=None, filt=None, extra=()):
        out, seen, dup = [], set(), 0
        for n in names:
            if not have(n): continue
            for r in read(R(n)):
                if planner and r.get("planner") != planner: continue
                if filt and not filt(r): continue
                k = key(r, extra)
                if k in seen: dup += 1; continue
                seen.add(k); out.append(r)
        return out, dup
    def emit(name, names, planner=None, filt=None, fields=FIELDS, extra=()):
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
    emit("m40_sa.csv", ["m40_sa.csv"]); emit("m40_rr.csv", ["m40_rr_tour.csv"]); emit("m40_cluster.csv", ["m40_cluster_patrol.csv"])
    # Table IX bottom: CP-SAT at its default budget on paper, at the converged threefold budget on ring
    cp = [r for r in (read(R("m40_cpsat_d5.csv")) if have("m40_cpsat_d5.csv") else []) if r["layout"] == "paper"]
    cp += read(R("m40_cpsat_d15.csv")) if have("m40_cpsat_d15.csv") else []
    for r in cp: r["planner"] = "cpsat"
    if cp: write(A("m40_cp_final.csv"), cp, FIELDS)
    print(f"   {'m40_cp_final.csv':24s} {len(cp):6d} rows")
    DF = FIELDS + ["depot_x", "depot_y"]
    for d in ("q", "e"):
        emit(f"depot_{d}.csv", [f"depot_{d}.csv", f"depot_{d}_fill.csv"], "sa", None, DF, ("depot_x", "depot_y"))
        for pl in ("cyc", "cluster_patrol"):
            emit(f"depot_{d}_{pl}.csv", [f"depot_{d}_{pl}.csv", f"depot_{d}_{pl}_fill.csv"], pl, None, DF, ("depot_x", "depot_y"))
    emit("hetero.csv", ["hetero.csv"], None, None, FIELDS + ["alloc"], ("alloc",))
    emit("mass.csv", ["mass.csv"])
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
            r0 = byK[min(byK)]; kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"])
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
    # Fig. 5 selection rule (registered): per family (paper, ring) at M=100, 1.5 MJ, the deployment whose
    # C4-vs-cluster own-optimum ratio is closest to that family's median
    if os.path.exists(A("eval_all_v2.csv")):
        import numpy as np
        d = defaultdict(lambda: defaultdict(dict))
        for r in read(A("eval_all_v2.csv")):
            if int(float(r["M"])) == 100 and float(r["Emax"]) == 1.5e6 and r["layout"] in ("paper", "ring"):
                d[(r["layout"], int(float(r["seed"])))][r["planner"]][int(float(r["K"]))] = float(r["J"])
        sel = {}
        for fam in ("paper", "ring"):
            rat = {s: min(b["cyc"].values()) / min(b["cluster_patrol"].values()) - 1 for (f_, s), b in d.items()
                   if f_ == fam and "cyc" in b and "cluster_patrol" in b}
            if rat:
                med = float(np.median(list(rat.values()))); sel[fam] = min(rat, key=lambda s: (abs(rat[s] - med), s))
        json.dump(sel, open(A("selected.json"), "w")); print(f"   Fig. 5 deployments (rule re-applied on v2): {sel}  (v1: paper 10, ring 12)")
    # censoring report: every planner's sweeps on the union grid must meet the registered rule
    import fill_tail
    os.environ["ALLOW_V1"] = "1" if a.allow_v1 else os.environ.get("ALLOW_V1", "")
    union_files = [A(n) for n in ("sa_rerun.csv", "rr_tour_win.csv", "cluster_win.csv", "eval_all_v2.csv", "gzz_eval.csv",
                                  "asghar_eval.csv", "liu_all.csv") if os.path.exists(A(n))]
    union = fill_tail.load(union_files)
    print("   censoring: runs fill_tail.py would still add (must all be 0):")
    for n, pl in (("sa_rerun.csv", None), ("grid.csv", None), ("rr_tour_win.csv", "rr_tour"), ("rr_sweep_win.csv", "rr_sweep"),
                  ("cluster_win.csv", "cluster_patrol"), ("eval_all_v2.csv", "cyc"), ("eval_all_v2.csv", "cyc_sector0"), ("gzz_eval.csv", "gzz_rev"),
                  ("asghar_eval.csv", "asghar_rev"), ("liu_all.csv", "liu_mpga")):
        if not os.path.exists(A(n)): continue
        dd = fill_tail.load([A(n)], planner=pl)
        jobs = fill_tail.needed(dd, union if n != "grid.csv" else None, 60, True)
        print(f"      {n:20s} {pl or 'sa':15s} {len(jobs):4d}")
        if jobs: problems.append(f"{n}: {len(jobs)} runs still needed to meet the extension rule (rerun RERUN_V2 section C)")

    # ---------------------------------------------------------------- 4. run
    o = lambda n: os.path.join(OUT, n)
    STEPS = [
        ("table4", "Table IV (tab:law), Sec. VI-B statistics, warning indicator, Table (field size)", ["make_table4.py", gf]),
        ("estimators", "Appendix B: choice of estimator", ["make_estimators.py", gf]),
        ("strand_rad", "Table V (tab:strand), Prop. 6 at the radial scale", ["strand_final.py", gf]),
        ("strand_consistent", "Prop. 6 at the consistent scale (Lemma: cost scale)", ["strand_final.py", gf], {"P6_SCALE": "consistent"}),
        ("scale_lemma", "Lemma (cost scale): verdict pairs, alpha, regret of changed verdicts", ["scale_lemma.py", gf]),
        ("criterion", "criterion (5): pass/flag regret by family (global form)", ["criterion_variants.py", gf, "-", "--tex"]),
        ("table8", "Table VIII (tab:regret)", ["make_table8.py", gf, A("apriori_regret.csv")]),
        ("exante", "Sec. VI-H ex-ante commute inputs", ["exante_rc.py", gf, "--out", A("exante.csv")]),
        ("criterion_exante_fp", "Sec. VI-H criterion with ex-ante inputs at the commute fixed point",
         ["criterion_variants.py", gf, "-", A("exante.csv"), "--fixedpoint", f"--dump={o('verdicts_exante_fp.json')}"]),
        ("criterion_exante_min", "Sec. VI-H criterion with ex-ante inputs at the smallest candidate K",
         ["criterion_variants.py", gf, "-", A("exante.csv"), f"--dump={o('verdicts_exante_min.json')}"]),
        ("verdict_changes", "Sec. VI-H: verdicts changed by ex-ante inputs (and in which direction)", None),
        ("reach_bracket", "Sec. IV-A reach bracket, K_flight comparison", ["reach_bracket.py", gf]),
        ("perturb_inputs", "Sec. VI-H sensitivity to known inputs", ["perturb_inputs.py", gf]),
        ("theory", "Sec. IV: r_c(K), P_bar spread, tau_hop(K), Prop. 3 with measured tau_hop", ["theory_checks.py", gf, "--tour", A("rr_tour_win.csv")]),
        ("bootstrap", "95% intervals (Sec. VI-B, VI-J), per-cell CSV", ["bootstrap_ci.py", gf, A("sa_rerun.csv"), A("rr_tour_win.csv"),
                                                                        A("rr_sweep_win.csv"), A("m40_sa.csv"), A("m40_rr.csv"), A("m40_cp_final.csv"), f"--out={o('per_cell_ci.csv')}"]),
        ("baseline_patrols", "Table IX: tour and sweep patrol rows", ["baseline_table.py", A("sa_rerun.csv"), A("rr_tour_win.csv"), A("rr_sweep_win.csv"), "--by-family"]),
        ("baseline_cluster", "Table IX: cluster patrol row", ["baseline_table.py", A("sa_rerun.csv"), A("cluster_win.csv")]),
        ("baseline_m40", "Table IX bottom (M=40)", ["baseline_table.py", A("m40_sa.csv"), A("m40_rr.csv"), A("m40_cp_final.csv"), "--by-family"]),
        ("baseline_cluster_m40", "Table IX bottom: cluster patrol row", ["baseline_table.py", A("m40_sa.csv"), A("m40_cluster.csv")]),
        ("paired_diff", "Sec. VI-J paired agreement differences", ["paired_diff.py", A("sa_rerun.csv"), A("rr_tour_win.csv")]),
        ("cluster", "Sec. VI-J cluster patrol vs SA", ["cluster_compare.py", A("sa_rerun.csv"), A("cluster_win.csv")]),
        ("planners", "Tables IX (dagger rows), XI (split), XII (published); C4 vs SA; cost of sizing",
         ["planners_v2.py", "--sa", A("sa_rerun.csv"), "--files"] + [A(n) for n in ("eval_all_v2.csv", "gzz_eval.csv", "asghar_eval.csv", "liu_all.csv")
                                                                     if os.path.exists(A(n))] + ["--verdicts", A("verdicts.csv")]),
        ("c4_sectors", "C4 paragraph: equal sectors without search vs the published method",
         ["own_opt_compare.py", A("eval_all_v2.csv"), A("eval_all_v2.csv"), "--pa", "cluster_patrol", "--pb", "cyc_sector0"]),
        ("liu", "Liu et al. paragraph (registered analysis)", ["analyze_liu.py", "--liu", A("liu_all.csv"), "--sa", A("sa_rerun.csv"),
                                                              "--others", A("eval_all_v2.csv"), A("gzz_eval.csv"), A("asghar_eval.csv"), "--verdicts", A("verdicts.csv")]),
        ("depot", "Table (depot)", ["depot_table.py", A("depot_q.csv"), A("depot_e.csv")]),
        ("depot_c4_q", "depot: C4 vs cluster patrol, quarter point", ["own_opt_compare.py", A("depot_q_cluster_patrol.csv"), A("depot_q_cyc.csv")]),
        ("depot_c4_e", "depot: C4 vs cluster patrol, edge", ["own_opt_compare.py", A("depot_e_cluster_patrol.csv"), A("depot_e_cyc.csv")]),
        ("hetero", "Sec. VI-C reserved sortie vs equal shares", ["analyze_hetero.py", A("sa_rerun.csv"), A("hetero.csv")]),
        ("mass", "Remark 1 battery mass", ["analyze_mass.py", A("mass.csv"), A("sa_rerun.csv")]),
        ("ablation", "Table III (ablation, K=4)", ["ablation.py", A("ablation.csv"), "4"]),
        ("ablation_K2", "Sec. V hover budgeting at K=2", ["ablation.py", A("ablation.csv"), "2"]),
        ("singlecell", "Tables VI (horizon), VII (coord), XIV (info); per-leg paragraph", ["singlecell_v2.py", RES]),
        ("outage", "Sec. VI-B grounding test", None),
        ("regularity", "Sec. IV-B CV of inter-visit intervals; Sec. VI-J rate/allocation/regularity decomposition", None),
        ("reserve", "Sec. VI-A: reserve use and skipped final stops over the SA grid", None),
        ("cpsat_proofs", "Sec. VI-J: CP-SAT share of decisions proven optimal", None),
        ("fig_JvsK_regime", "Figs. 2-3", ["make_figures.py", gf]),
        ("fig_planners", "Fig. (planners, M=40)", ["fig_planners.py", A("m40_sa.csv"), A("m40_rr.csv"), A("m40_cp_final.csv"), o("fig_planners.pdf")]),
        ("fig_five", "Fig. 5 (five planners)", ["fig_five.py", o("fig_five.pdf"), "--sa", A("sa_rerun.csv"), "--eval", A("eval_all_v2.csv"),
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
                     "regularity": regularity_summary, "reserve": reserve_summary}[name](
                        OUT if name == "verdict_changes" else (ASM if name == "reserve" else RES), f)
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
        f.write(f"{fam:5s} runs {len(x):5d}  skipped-stop share median {np.median(b):.3f} p90 {np.quantile(b, .9):.3f}  "
                f"mid-route truncation median {np.median(t):.3f} p90 {np.quantile(t, .9):.3f}  "
                f"landings into reserve {dips}  reserve exhausted {viol}\n")


def regularity_summary(RES, f):
    import numpy as np
    files = sorted(glob.glob(os.path.join(RES, "regularity*.csv")))
    if not files: raise FileNotFoundError("no regularity*.csv")
    g = defaultdict(list)
    for p in files:
        for r in read(p): g[(os.path.basename(p), r["planner"], r["layout"], r["M"], r["Emax"], int(r["K"]))].append(r)
    cols = ("CV", "CV_w", "share_sparse", "tau_hop", "Gamma", "E_alloc", "E_reg", "r_c", "J")
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


if __name__ == "__main__":
    main()
