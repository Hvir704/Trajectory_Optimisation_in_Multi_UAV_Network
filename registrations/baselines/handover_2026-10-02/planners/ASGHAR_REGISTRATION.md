# Asghar et al. baseline — registration (draft, fix before any seed 1-12 run)

Baseline: A. B. Asghar, S. Sundaram, S. L. Smith, "Multirobot Persistent Monitoring: Minimizing Latency and Number of
Robots With Recharging Constraints", IEEE Trans. Robotics 41:236-252, 2025, doi 10.1109/TRO.2024.3502497.
Implemented from the published (Xplore) version: Algorithm 4 (LatencyWalks) with Algorithm 3
(MinMaxLatencyOneRobot); RMCCP by greedy orienteering as in the paper.
Code: asghar_planner.py (planners `asghar_full`, `asghar_rev`), registered in run_grid.py.
Filled gaps and deviations: `GAPS` dict in asghar_planner.py (paper-ready wording).

## What the method optimises, and why it is still the right comparator
It minimises the maximum priority-weighted latency (time between visits) for a fixed fleet, with recharging at a
depot: the peak form of our collection-side weighted age. We evaluate it on our time-average objective. It is a
persistent, multi-robot, depot-recharging, priority-aware, non-learning method, and its visit allocation (weight
classes, interleaved cycle covers) differs in structure from every other planner in the paper, which makes it an
independent test of the fleet-size rule.

Expected and NOT a reason to tune: Algorithm 4 assigns robots per weight class, not per workload. With priorities in
[1,10] there are four classes (about 55% / 28% / 14% / 3% of sensors); at K = 2-3 one robot carries most sensors while
others circle a handful, and at K >= 4 each class gets floor(K/4) robots. On pilot seed 13 (paper, M = 50, 6 h smoke)
J was about 5x cluster_patrol at K = 3 and about 2x at K = 5.

## 1. Pilot (seeds 13-15 only)
    python run_grid.py asghar_pilot.csv --layouts paper ring core --M 50 100 200 --Emax 1.5e6 3e6 --seeds 13-15 --planner asghar_full --procs 8
    python run_grid.py asghar_pilot.csv --layouts paper ring core --M 50 100 200 --Emax 1.5e6 3e6 --seeds 13-15 --planner asghar_rev  --procs 8
Checks: no crash; `[asghar]` lines show every reachable sensor in some cycle (`dropped` = sensors unreachable alone,
same as cluster_patrol's n_never at that cell); CP-SAT status counts (FALLBACK = greedy-insertion cycle used; report
the fraction); truncation fraction.
Runtime: plan build 15-50 s per pass at M = 200 (rev = 2 passes) on one core; about 5-6 h for the held-out grid on 8 procs.

**Dwell-mode rule (the only pilot choice):** deploy the mode with the lower median own-optimum J over the 54 pilot
deployments (paired). Ties (|median diff| < 1%) -> `asghar_rev`. Record and hash:
`sha256sum asghar_planner.py run_grid.py dyn_env.py`. Chosen mode: ______ hashes: ______ date: ______

## 2. Held-out evaluation (seeds 1-12), 216 deployments
Own optima on the union K grid of {asghar, gzz, cluster_patrol, C4} (+ SA after the i9 fill), extended until each
planner's three extreme K values exceed 1.5x its minimum. Asghar's optimum may sit at larger K than the others
(class-based allocation); the extension rule handles it — do not cap the grid.

**Primary criterion:** per family, median J*_C4 / J*_asghar - 1 with 95% percentile bootstrap and win counts; claim if
below 0 in each family. **Reported without a directional claim:** J*_asghar vs J*_cluster and vs J*_gzz.
**Secondary:** rule agreement (exact / within one) for Asghar's own K* (Table IX format) — the key evidence this
baseline contributes; same-K comparison at the rule's l; CP-SAT fallback fraction.
If anything breaks mid-evaluation: fix on pilot data, rerun from scratch, disclose.
