"""
dyn_c4_grid.py -- Dynamic C4 grid runner: M x K x seeds, SA or stub planner.
=============================================================================
Combines dyn_env.py (simulator, CONTEXT_67-fixed) and sa_sortie.py (SA sortie
planner) into a grid sweep matching the CLI conventions of sa_repair2.py /
compare_baseline.py: argparse, printed table, CSV output, verdict block.

WHY THIS IS SEPARATE FROM sa_repair2.py / compare_baseline.py
  Those solve the STATIC K-way partition at the OLD scale (Emax=50000,
  Pf=150/Ph=200, AREA=1000). This solves the DYNAMIC continuous-operation
  problem (CONTEXT_60 rev2) at the NEW scale (Emax>=1.5M, Pf=300/Ph=400,
  L~12.6km), one drone per sortie, live ages, estimated dwell. Different
  problem, different code path. Do not merge the two.

RUNTIME. This was designed and smoke-tested in a sandboxed environment with a
~5 minute wall-clock ceiling per command, which limited testing to iters<=800,
3 seeds, single M/K cells. On a real machine (esp. the 5090/64GB box), raise
--iters and --instances substantially -- 500 SA iters is thin; CONTEXT_16's
static work used 2000-6000. Start with the defaults below to confirm the script
runs end-to-end in a few minutes, THEN scale up.

WHAT TO READ FIRST IN THE OUTPUT
  1. `agree` -- how many seeds' individual argmin matches the mean argmin. If
     this is close to half of --instances, the cell is NOT resolved and the
     printed K* is not trustworthy (CONTEXT_60 §5.1b resolution-floor logic,
     ported to the dynamic setting).
  2. `sep%` -- gap between best and second-best K. Under ~5% is a tie, same
     convention as the static repair work.
  3. `Ts/tc` vs `pred` -- CONTEXT_64 §5 criterion 6. Should sit in [1.87,2.0]
     roughly at the argmin. Large deviation is a planner or scale problem, not
     noise -- see CONTEXT_66.
  4. `thru` -- throughput margin (CONTEXT_63 §4). Must be < 1 in every cell or
     that cell is infeasible by construction and K* there means nothing.
  5. `e*/Eu` -- energy actually spent per sortie as a fraction of E_usable.
     NEW. The law divides (1-rho)*Emax by e*, which presumes drones spend their
     budget. Truncated sorties come home with fuel, so e*_measured understates
     it, and truncation varies with M and K -- an M-dependent bias sitting in
     the exact quantity whose M-drift is the headline result. Regress this
     against M before attributing any measured K* drift to the P_bar mechanism.

PROVENANCE (added after the C3 session).
  Every CSV row now carries `commit`, `dirty`, and SHA-256 prefixes of
  dyn_env.py / sa_sortie.py. Reason: a C3 probe run produced rows that were
  byte-identical to its control because dyn_env.py in that tree lacked the
  planner observation hook -- the output looked plausible and meant nothing.
  CONTEXT_16's banked "SA" values were similarly mislabelled greedy_init
  outputs. A row that cannot name the source that produced it is not evidence.
  `dirty=True` means the working tree had uncommitted changes: the commit hash
  alone does NOT identify that run.

SEEDING. `--instances N` draws N seeds from a single meta-seed (--seed,
default 2025, matching the project convention). Each (M,K,seed) cell is
independent; the SA planner's internal per-sortie seeding is handled inside
sa_sortie.build_sa_planner and does NOT reuse the CONTEXT_60 §5.4 restart-index
pattern -- confirmed absent, see sa_sortie.py docstring.

Run (defaults are the smoke-test scale -- raise before trusting the output):
    python dyn_c4_grid.py --M 100 --K 3 4 5 6 --Emax 1.5e6 --instances 5 --iters 1500
    python dyn_c4_grid.py --M 50 100 200 400 --K 3 4 5 6 8 --instances 8 --iters 2000 --out-dir c4_grid
"""
import argparse
import csv
import hashlib
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from dyn_env import DynParams
from dyn_env import DynSim
from dyn_env import greedy_ratio_planner
from sa_sortie import build_sa_planner

DEFAULT_SEED = 2025  # matches project convention (meta-seed draws instance seeds)

# files whose contents change what the numbers MEAN, not just how fast they run
_SOURCE_FILES = ("dyn_env.py", "sa_sortie.py", "dyn_c4_grid.py")


def _sha8(path):
    """First 8 hex chars of a file's SHA-256, or a marker if unreadable.
    Catches the case the git hash cannot: an edited-but-uncommitted source."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:8]
    except OSError:
        return "missing"


def _git(*args):
    try:
        r = subprocess.run(("git",) + args, capture_output=True, text=True,
                           timeout=10, cwd=os.path.dirname(os.path.abspath(__file__)))
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def provenance():
    """Identify the source that produced these numbers. Stamped on every row."""
    here = os.path.dirname(os.path.abspath(__file__))
    commit = _git("rev-parse", "--short", "HEAD") or "nogit"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "nogit"
    dirty = bool(_git("status", "--porcelain"))
    p = {"commit": commit, "branch": branch, "dirty": dirty}
    for f in _SOURCE_FILES:
        p["sha_" + f.replace(".py", "")] = _sha8(os.path.join(here, f))
    return p


def run_cell(args):
    """Top-level (picklable) worker: runs ONE (M,K,Emax,seed) cell.
    Kept as a plain function taking a single tuple so ProcessPoolExecutor can
    pickle it -- a bound method or closure cannot cross process boundaries."""
    M, K, Emax, L, planner_name, iters, T_horizon, T_burnin, seed = args
    p = DynParams(M=M, K=K, Emax=Emax, L=L, T_horizon=T_horizon, T_burnin=T_burnin)
    planner = (build_sa_planner(iters=iters) if planner_name == "sa"
               else greedy_ratio_planner)
    m = DynSim(p, planner=planner, seed=seed).run()
    # E_usable is a property of p, not returned by metrics(); attach it here so
    # the parent can compute the utilisation ratio without rebuilding DynParams.
    m = dict(m)
    m["E_usable"] = p.E_usable
    return (M, K, Emax, seed, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[100])
    ap.add_argument("--K", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--Emax", type=float, nargs="+", default=[1.5e6],
                     help="one or more energy budgets, e.g. 1.5e6 3e6 6e6")
    ap.add_argument("--L", type=float, default=12_600.0,
                     help="field side, m. CONTEXT_64 anchor for K*=4 at Emax=1.5M. "
                          "NOTE (CONTEXT_71): L was SOLVED FOR by assuming K*=4 at "
                          "1.5M, so K*=4 at 1.5M is an anchor, not a prediction. "
                          "Only the 3M and 6M cells test the law.")
    ap.add_argument("--planner", choices=["sa", "stub"], default="sa")
    ap.add_argument("--iters", type=int, default=1500,
                     help="SA iterations per sortie. Ignored for --planner stub. "
                          "This is a FIXED budget applied at every K -- but "
                          "expected nodes-per-sortie shrinks with K (more energy "
                          "per drone at low K => bigger combinatorial problem), "
                          "so low-K cells are relatively under-converged at a "
                          "fixed iters. Prefer --iters-per-node over raising this "
                          "alone if K spans a wide range.")
    ap.add_argument("--iters-per-node", type=int, default=0,
                     help="if >0, SA iters for a cell = --iters + this * "
                          "(Emax/K / E_usable_at_K4_reference) roughly scaling "
                          "with expected sortie size, so low-K (big-sortie) "
                          "cells get proportionally more search. Start at "
                          "iters=1000 --iters-per-node=150 as a first try.")
    ap.add_argument("--instances", type=int, default=5,
                     help="seeds per (M,K,Emax) cell")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                     help="meta-seed drawing instance seeds")
    ap.add_argument("--T-horizon", type=float, default=6 * 3600.0)
    ap.add_argument("--T-burnin", type=float, default=1.5 * 3600.0)
    ap.add_argument("--out-dir", default="dyn_c4_grid")
    ap.add_argument("--tag", default="",
                     help="free-text label written to every row, e.g. the "
                          "workstream or question this sweep was run to answer. "
                          "Costs nothing and saves an archaeology session later.")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                     help="parallel processes. Every (M,K,Emax,seed) cell is "
                          "independent, so this scales close to linearly. "
                          "Default: all logical cores.")
    a = ap.parse_args()

    prov = provenance()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    seeds = [int(rng.integers(0, 10_000_000)) for _ in range(a.instances)]

    print("=" * 112)
    print(f"  DYNAMIC C4 GRID | planner={a.planner} L={a.L:.0f}m "
          f"iters={a.iters if a.planner=='sa' else '-'} instances={a.instances} "
          f"(meta-seed {a.seed})")
    print(f"  M={a.M}  K={a.K}  Emax={a.Emax}")
    print(f"  T_horizon={a.T_horizon/3600:.1f}h  T_burnin={a.T_burnin/3600:.1f}h")
    print(f"  provenance: branch={prov['branch']} commit={prov['commit']}"
          f"{'  *** DIRTY WORKING TREE ***' if prov['dirty'] else ''}")
    print(f"              dyn_env={prov['sha_dyn_env']} "
          f"sa_sortie={prov['sha_sa_sortie']} grid={prov['sha_dyn_c4_grid']}")
    if a.tag:
        print(f"  tag: {a.tag}")
    if prov["dirty"]:
        print("  NOTE: the commit hash does NOT identify this run. The source "
              "hashes above do -- keep them with any banked number.")
    print("=" * 112)

    all_rows = []
    all_tasks = []
    for Emax in a.Emax:
        for M in a.M:
            for K in a.K:
                # scale iters with expected sortie size: fewer drones -> more
                # energy each -> bigger per-sortie combinatorial problem, and a
                # FIXED iters budget under-converges it relative to high-K cells.
                # CONTEXT_70: reference was max(K) IN THIS SWEEP, so the boost's
                # magnitude depended on how wide a --K range was requested, not
                # on the actual problem size -- a narrow high-K sweep (e.g.
                # 6..10) gave low-K cells a much smaller boost than an
                # equal-sized-problem cell got in a wider sweep (e.g. 3..6),
                # silently under-converging it. Fixed reference point (K=1)
                # makes the boost depend only on K itself, not on sibling
                # --K values, so it is comparable across different sweeps.
                iters_K = a.iters
                if a.iters_per_node > 0:
                    iters_K = a.iters + int(a.iters_per_node * (1.0 / K) * 6)
                    # the *6 keeps rough parity with the old default's typical
                    # magnitude at K~3-6 so existing --iters-per-node values
                    # from before this fix don't need re-tuning from scratch.
                for s in seeds:
                    all_tasks.append((M, K, Emax, a.L, a.planner, iters_K,
                                       a.T_horizon, a.T_burnin, s))

    print(f"  {len(all_tasks)} cells total, {a.workers} workers")
    t_start = time.time()
    results = {}  # (Emax,M,K) -> list of (seed, metrics)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_cell, t): t for t in all_tasks}
        for fut in as_completed(futs):
            M, K, Emax, seed, m = fut.result()
            results.setdefault((Emax, M, K), []).append((seed, m))
            done += 1
            if done % max(1, len(all_tasks) // 20) == 0 or done == len(all_tasks):
                el = time.time() - t_start
                eta = el / done * (len(all_tasks) - done)
                print(f"  {done}/{len(all_tasks)}  elapsed={el:.0f}s  eta={eta:.0f}s")

    for Emax in a.Emax:
        for M in a.M:
            J = {K: [] for K in a.K}
            diag = {K: [] for K in a.K}
            for K in a.K:
                cell = sorted(results[(Emax, M, K)], key=lambda x: x[0])  # order by seed
                for seed, m in cell:
                    J[K].append(m["J_timeavg"])
                    diag[K].append(m)

            arg = [min(a.K, key=lambda K: J[K][i]) for i in range(a.instances)]
            means = {K: float(np.mean(J[K])) for K in a.K}
            stds = {K: float(np.std(J[K])) for K in a.K}
            srt = sorted(a.K, key=lambda K: means[K])
            best, second = srt[0], srt[1] if len(srt) > 1 else srt[0]
            sep = 100 * (means[second] / means[best] - 1) if means[best] != 0 else 0.0
            agree = arg.count(best)

            print(f"\n--- Emax={Emax:.2e}  M={M} ---")
            print(f"{'K':>3} {'J_mean':>13} {'J_std':>11} {'P_bar':>6} "
                  f"{'Ts/tc':>6} {'pred':>6} {'n_vis':>6} {'thru':>6} "
                  f"{'Trev/tau_e':>10} {'catch':>6} {'empty':>6} {'trunc%':>7} "
                  f"{'e*/Eu':>6} {'K*pred':>7}")
            for K in a.K:
                d = diag[K]
                pb = np.mean([x["P_bar"] for x in d])
                ts = np.mean([x["T_s_over_t_c"] for x in d])
                tp = np.mean([x["T_s_over_t_c_pred"] for x in d])
                nv = np.mean([x["mean_n_visited"] for x in d])
                thru = np.mean([x["throughput_margin"] for x in d])
                trev = np.mean([x["T_rev_over_tau_e"] for x in d])
                catch = np.mean([x["catch_rate"] for x in d])
                empty = int(np.sum([x["empty_sorties"] for x in d]))
                trunc = int(np.sum([x.get("truncated_sorties", 0) for x in d]))
                n_tot = int(np.sum([x.get("n_sorties_total", 1) for x in d]))
                trunc_pct = 100 * trunc / max(n_tot, 1)
                # NEW: energy actually spent vs energy the law assumes is spent.
                e_meas = np.mean([x["e_star_measured"] for x in d])
                e_pred = np.mean([x["e_star_pred"] for x in d])
                util = np.mean([x["e_star_measured"] / x["E_usable"] for x in d])
                kpred = np.mean([x["kstar_pred"] for x in d])
                flag = " <=ARGMIN" if K == best else ""
                # T_s/t_c ~ 2 is derived to hold AT THE OPTIMUM only; T_s/t_c
                # decreases monotonically with K away from it by construction
                # (less energy per drone -> shorter sorties), so deviation at a
                # non-argmin K is EXPECTED and not flagged. Only the argmin row
                # deviating is worth a second look.
                anomaly = (" !!ARGMIN DEVIATES FROM PREDICTION -- check convergence "
                           "(--iters-per-node) before trusting this K"
                           if (K == best and not (1.87 <= ts <= 2.02)) else "")
                print(f"{K:>3} {means[K]:>13.0f} {stds[K]:>11.0f} {pb:>6.0f} "
                      f"{ts:>6.2f} {tp:>6.2f} {nv:>6.1f} {thru:>6.2f} "
                      f"{trev:>10.2f} {catch:>6.2f} {empty:>6} {trunc_pct:>6.1f}% "
                      f"{util:>6.3f} {kpred:>7.2f}"
                      f"{flag}{anomaly}")
                all_rows.append(dict(
                    Emax=Emax, M=M, K=K, planner=a.planner,
                    J_mean=means[K], J_std=stds[K], P_bar=pb,
                    Ts_over_tc=ts, Ts_over_tc_pred=tp, mean_n_visited=nv,
                    throughput_margin=thru,
                    Trev_over_tau_e=trev, catch_rate=catch, empty_sorties=empty,
                    truncated_pct=trunc_pct,
                    e_star_measured=e_meas, e_star_pred=e_pred,
                    e_star_over_usable=util, kstar_pred=kpred,
                    instances=a.instances, iters=a.iters if a.planner == "sa" else None,
                    L=a.L, T_horizon=a.T_horizon, T_burnin=a.T_burnin,
                    meta_seed=a.seed, tag=a.tag,
                    **prov,
                ))

            resolved = agree >= (0.8 * a.instances)
            print(f"\n  => K*={best} (2nd={second}, sep={sep:.1f}%)  "
                  f"per-seed agreement {agree}/{a.instances}"
                  f"  [{'RESOLVED' if resolved else 'NOT RESOLVED -- treat as a tie'}]")
            if not resolved:
                print("     CONTEXT_60 §5.1b: report this cell as a band, not a point.")

            # --- J-argmin trustworthiness (SESSION_SUMMARY §4) -------------
            # Near the optimum J is flat -- that is what an optimum IS -- so the
            # argmin carries little signal. Two independent tells:
            #   (a) best vs second-best within 2 SE of the difference;
            #   (b) a non-monotonic cost curve, which a convex curve cannot have.
            # Either one means the printed K* is noise, whatever `agree` says.
            n = a.instances
            se_b = stds[best] / max(np.sqrt(n), 1e-9)
            se_2 = stds[second] / max(np.sqrt(n), 1e-9)
            se_diff = float(np.sqrt(se_b ** 2 + se_2 ** 2))
            gap = means[second] - means[best]
            if gap < 2 * se_diff:
                print(f"     ARGMIN NOT SIGNIFICANT: gap={gap:.0f} < 2*SE={2*se_diff:.0f}. "
                      f"K*={best} is not distinguishable from K={second}.")
            curve = [means[K] for K in sorted(a.K)]
            turns = sum(1 for i in range(1, len(curve) - 1)
                        if (curve[i] - curve[i - 1]) * (curve[i + 1] - curve[i]) < 0)
            if turns > 1:
                print(f"     NON-MONOTONIC COST CURVE ({turns} turning points) -- "
                      f"a convex curve has one. Treat this cell as unresolved.")

            if any(np.mean([x["throughput_margin"] for x in diag[K]]) >= 1.0 for K in a.K):
                print("     WARNING: throughput_margin >= 1 in at least one K -- "
                      "that cell is infeasible by construction (CONTEXT_63 §4).")

            # --- energy-utilisation bias (this session's finding 2) ---------
            # The law divides (1-rho)*Emax by e*, presuming the budget is spent.
            # Truncated sorties return with fuel, so e*_measured understates it.
            util_best = np.mean([x["e_star_measured"] / x["E_usable"] for x in diag[best]])
            if util_best < 0.95:
                print(f"     ARGMIN K={best}: energy utilisation {util_best:.3f} -- "
                      f"{100*(1-util_best):.1f}% of the budget goes unspent. If this "
                      f"varies with M, any measured K* drift is partly a utilisation "
                      f"artefact, not the P_bar mechanism. Regress e_star_over_usable "
                      f"on M across cells before attributing drift.")

            sensors_per_hotspot = M / 15.0  # DynParams.n_hotspots default
            nv_best = np.mean([x["mean_n_visited"] for x in diag[best]])
            ts_best = np.mean([x["T_s_over_t_c"] for x in diag[best]])
            if nv_best > sensors_per_hotspot and not (1.87 <= ts_best <= 2.02):
                print(f"     ARGMIN K={best}: mean_n_visited={nv_best:.1f} exceeds "
                      f"~{sensors_per_hotspot:.1f} sensors/hotspot AND Ts/tc deviates "
                      f"from prediction -- consistent with sorties jumping between "
                      f"hotspots (CONTEXT_63 §3), OR under-converged SA at this K's "
                      f"larger problem size. Re-run with --iters-per-node before "
                      f"concluding either way.")

    # filename carries the source hash: two runs from different working trees
    # can no longer silently overwrite each other, and pooling by glob cannot
    # merge them by accident (Part C item 7 -- pooling c4_emax with c4_emax_v2
    # double-counted a cell because the files looked interchangeable).
    out = os.path.join(a.out_dir,
                       f"dyn_c4_{a.planner}_{prov['commit']}"
                       f"{'-dirty' if prov['dirty'] else ''}"
                       f"_{prov['sha_dyn_env']}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {out}")
    print("=" * 112)


if __name__ == "__main__":
    main()