# Liu et al. (MPGA, adapted) on the full held-out grid -- registration
Written 7 Oct 2026, BEFORE any run of the extension. Only the 36 deployments at M=50, 1.5 MJ (plus two
K=1 edge runs) have been seen, from the report of the same date; that is disclosed with the results.

## Scope and measurement (identical to the Gao / Asghar / C4 evaluations)
- 216 held-out deployments: paper, ring, core x M in {50,100,200} x Emax in {1.5, 3} MJ x seeds 1-12.
- Liu run at the UNION of every K at which SA, C4, the cluster patrol, Gao et al. or Asghar et al. was run
  for that deployment (build_union_grid.py -> liu_jobs.csv), then extended with fill_tail.py --rule15 until
  its three largest K values all exceed 1.5x its minimum J. Own optimum = argmin over its own sweep.
- Planner exactly as in the report (baselines/liu2022, commit dce4df2), GA seed 0, no retuning.
- ALL Liu runs come from one machine (the Ryzen): the 404 M=50 rows from the i5 machine and the 2 K=1 edge
  runs from the paper chat are re-run there (liu_jobs_all.csv, 3,613 runs). The earlier rows are used only to
  report cross-machine determinism (one spot check already matched to the last digit).

## Primary comparison (registered)
C4 against Liu: per family, median of J*_C4 / J*_Liu - 1 at each planner's own optimum, 95% percentile
bootstrap over deployments (10^4 resamples), and deployments where C4 is lower. Criterion: C4's median
below Liu's in EACH family. Reported whatever the outcome; no directional claim beyond this criterion.

## Secondary (descriptive, registered)
1. Same as primary at K = l (the rule's prediction).
2. SA, cluster patrol, Gao et al., Asghar et al. each against Liu, same statistics.
3. Rule agreement for Liu (own K* vs l, exact / within one), all and by family -> Table IX row.
4. That agreement split by the design-time test (criterion5_verdicts.csv, verdict_current) -> tab:split row.
5. Cost of sizing with l, J(l)/J* - 1, median and 90th percentile (exploratory, as for the other planners).
6. The 36 already-seen deployments reported separately as well, so the effect of seeing them is visible.

## Citation check still required
VERIFIED against the publisher PDF (7 Oct 2026): C. Liu, Y. Guo, N. Li, X. Song, "AoI-Minimal Task Assignment and
Trajectory Optimization in Multi-UAV-Assisted IoT Networks," IEEE IoT-J 9(21):21777-21791, 1 Nov 2022,
doi 10.1109/JIOT.2022.3182160. MPGA settings in the code (Np=8, N0=60, Niter=100, pcr in [0.5,0.95], pmu in
[0.05,0.3], roulette + best-individual parents, sequential crossover, swap/inversion/slide, breakpoint update)
match Sec. III-B and IV.

## Amendment (7 Oct 2026, before any extension statistic was computed)
The first Ryzen run used code that differs from the reference (see LIU_RYZEN_RUN_INVALID.md) and is discarded.
The grid is rerun from a clean clone of the reference repo after check_liu_code.py passes.
