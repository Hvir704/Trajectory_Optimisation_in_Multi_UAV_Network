"""reproduce.py -- regenerate every table, statistic and figure in the paper from the raw run files.

usage:  python reproduce.py [--only STEP ...] [--out results/]

Inputs (all in this folder): sa_rerun.csv, grid.csv, rr_tour_win.csv, rr_sweep_win.csv, m40_sa.csv,
m40_rr.csv, m40_cp.csv, m40_cp15.csv, apriori_regret.csv. No simulation is run; everything below is
analysis of recorded runs, deterministic apart from the bootstrap (fixed seed).

Each step writes its output to --out and the paper element it feeds is named. A step that fails
stops the run; nothing is silently skipped.
"""
import argparse, csv, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def m40_cp_final(out):
    """paper cells at the default CP-SAT budget, ring cells at the converged threefold budget."""
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "m40_cp.csv"))) if r["layout"] == "paper"]
    rows += list(csv.DictReader(open(os.path.join(HERE, "m40_cp15.csv"))))
    for r in rows: r["planner"] = "cpsat"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


STEPS = [
    # name, feeds, command (list); {o} = output dir
    ("grid_final", "single analysis file for everything below",
     ["build_grid_final.py", "sa_rerun.csv", "grid.csv", "{o}/grid_final.csv"]),
    ("table4", "Table IV and the Sec. VII-B statistics",
     ["make_table4.py", "{o}/grid_final.csv"]),
    ("estimators", "Appendix: choice of estimator",
     ["make_estimators.py", "{o}/grid_final.csv"]),
    ("strand", "Table V, Proposition 6 evaluation",
     ["strand_final.py", "{o}/grid_final.csv"]),
    ("criterion", "Table VIII (global surrogate test, measured inputs)",
     ["criterion_variants.py", "{o}/grid_final.csv", "apriori_regret.csv", "--tex"]),
    ("exante", "Sec. VII-I ex-ante commute inputs",
     ["exante_rc.py", "{o}/grid_final.csv", "--out", "{o}/exante.csv"]),
    ("criterion_exante", "Sec. VII-I: criterion with ex-ante inputs at the commute fixed point",
     ["criterion_variants.py", "{o}/grid_final.csv", "apriori_regret.csv", "{o}/exante.csv", "--fixedpoint"]),
    ("reach_bracket", "Sec. V-A / VII-B reach bracket and the K_flight comparison",
     ["reach_bracket.py", "{o}/grid_final.csv"]),
    ("baseline", "Table IX (planner robustness, primary grid)",
     ["baseline_table.py", "sa_rerun.csv", "rr_tour_win.csv", "rr_sweep_win.csv", "--by-family"]),
    ("m40_cp_final", "M=40 CP-SAT file (paper default budget, ring converged budget)", None),
    ("baseline_m40", "Table IX (M=40 three-planner rows)",
     ["baseline_table.py", "m40_sa.csv", "m40_rr.csv", "{o}/m40_cp_final.csv", "--by-family"]),
    ("bootstrap", "all 95% intervals in Sec. VII-B, VII-J; per-cell intervals",
     ["bootstrap_ci.py", "{o}/grid_final.csv", "sa_rerun.csv", "rr_tour_win.csv", "rr_sweep_win.csv",
      "m40_sa.csv", "m40_rr.csv", "{o}/m40_cp_final.csv", "--out={o}/per_cell_ci.csv"]),
    ("figure_planners", "Fig. (planner robustness)",
     ["fig_planners.py", "m40_sa.csv", "m40_rr.csv", "{o}/m40_cp_final.csv", "{o}/fig_planners.pdf"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results"); ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    for name, feeds, cmd in STEPS:
        if a.only and name not in a.only: continue
        t0 = time.time()
        log = os.path.join(out, f"{name}.txt")
        if cmd is None:
            m40_cp_final(os.path.join(out, "m40_cp_final.csv"))
            print(f"[ok] {name:18s} {time.time()-t0:5.1f}s  -> {feeds}"); continue
        args = [c.replace("{o}", out) for c in cmd]
        with open(log, "w") as f:
            r = subprocess.run([PY] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[FAILED] {name}: see {log}"); sys.exit(1)
        print(f"[ok] {name:18s} {time.time()-t0:5.1f}s  -> {feeds}   ({os.path.relpath(log)})")
    print(f"\nall outputs in {out}")


if __name__ == "__main__":
    main()
