# GZZ baseline — registration (draft, fix before any seed 1-12 run)

Baseline: Gao, Zhu, Zhai, "AoI-Sensitive Data Collection in Multi-UAV-Assisted Wireless Sensor
Networks", IEEE TWC 22(8):5185-5197, Aug. 2023, doi 10.1109/TWC.2022.3232366.
Code: gzz_planner.py (planners `gzz_full`, `gzz_rev`), registered in run_grid.py.
Deviations and filled gaps: `GZZ_DEVIATIONS` in gzz_planner.py (paste into the paper's method text).

## 1. Pilot (seeds 13-15 only)
Grid: paper, ring, core x M in {50,100,200} x Emax in {1.5, 3} MJ, 12 h horizon, 3 h burn-in, auto K range.

    python run_grid.py gzz_pilot.csv --layouts paper ring core --M 50 100 200 --Emax 1.5e6 3e6 --seeds 13-15 --planner gzz_full --procs 8
    python run_grid.py gzz_pilot.csv --layouts paper ring core --M 50 100 200 --Emax 1.5e6 3e6 --seeds 13-15 --planner gzz_rev  --procs 8

Implementation checks (all must pass): no crash; plan uses no age/priority/true-rate input; `[gzz]` lines show
trunc_sorties small (report the fraction); `dropped` equals the sensors unreachable alone (same n_never as
cluster_patrol at the same cell).

**Dwell-mode decision rule (the only pilot choice):** deploy the mode with the lower median own-optimum J over the
54 pilot deployments (paired by deployment). Ties (|median diff| < 1%) -> `gzz_rev` (closer to the paper's fixed
per-SN upload time at the operating revisit). Tuning favours the baseline, which is the fair direction.
Record the choice here, then hash: `sha256sum gzz_planner.py run_grid.py dyn_env.py rr_planner.py`.

Chosen mode: ______   hashes: ______   date: ______

## 2. Held-out evaluation (seeds 1-12), primary grid M <= 200, 216 deployments
Own optima on the union K grid of {gzz, cluster_patrol, C4} (SA added after the i9 tail fill), extended until each
planner's three extreme K values exceed 1.5x its minimum (no censored optimum), same rule as the C4 evaluation.

**Primary criterion:** per family, median of J*_C4 / J*_gzz - 1 with 95% percentile bootstrap over deployments
and win counts. Claim if the median is below 0 in each family.
**Reported without a directional claim:** J*_gzz / J*_cluster - 1 per family (two published methods side by side);
J*_SA / J*_gzz - 1 once SA own-optima are final.
**Secondary:** same-K comparison at the rule's l; rule agreement (exact / within one) for gzz's own K*, as in
Table IX; truncation fraction; the paper's own K rule (`gzz_k_choice`) per deployment.

**Expectation, written before the run:** gzz is partition-then-tour like [18]; a result near cluster_patrol is the
expected outcome, not a reason to revise the implementation. Pilot seed 13 at M = 200 already shows the paper's
own K rule finding no feasible K under the shared inventory (best load ratio 1.2-3.4); report as found on 1-12.

If anything breaks mid-evaluation: fix on pilot data, rerun the evaluation from scratch, disclose.
