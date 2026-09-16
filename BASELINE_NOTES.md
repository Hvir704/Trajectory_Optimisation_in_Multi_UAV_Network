# External baseline -- working notes

## Code
- `rr_planner.py` -- age-blind patrols. `rr_tour`: one closed NN+2-opt tour through depot + all sensors,
  cut into sorties (route-first/cluster-second). `rr_sweep`: polar-angle order. Shared cyclic pointer per
  simulation; contiguous runs; skips excluded and alone-unreachable sensors; launch-time only.
- `milp_sortie.py` -- CP-SAT set-orienteering (Circuit + optional nodes + energy knapsack). Costs ceiled,
  budget floored (accepted routes are truly feasible). `build_milp_planner(dtime)`: COLD start,
  `interleave_search` + `max_deterministic_time` => results independent of machine load (verified: two
  repeats identical). Greedy fallback is counted, never a silent empty sortie. t=0 all-zero-age launch
  falls back to greedy.
- `run_grid.py` -- planners `rr_tour`, `rr_sweep`, `cpsat_d2|d5|d15`. No schema change.
- `compare_baseline.py` -- per-instance, K-set intersection, Th=43200 only, duplicate-row guard,
  edge-censoring for both planners.
- `milp_probe.py` -- `check`: CP-SAT vs Held-Karp; `time`: post-burn-in decisions, SA vs SA-10x vs CP-SAT.
- `mip_gsec.py` -- MIP + iterative GSEC cuts (SCIP/CBC). NEGATIVE: did not close the hard M=40 decision
  in 120 s (28/40 full re-solves). Kept for the record; not used.

## Verified
- CP-SAT == Held-Karp on 16 real M=12 decisions (max rel diff 2e-16), all OPTIMAL, all feasible.

## Results so far (Linux sandbox, 1 core -- provisional)
- rr_tour vs banked SA, 272 instances: argmin within one of SA's in 97%; exact 74%; bias -0.25.
  J at own optimum: paper +23% (SA better), ring -10%, core -6% (RR better).
- Paired same-env check, ring M=100 K=3: RR/SA-1 = -21%, +16%, -11%.
- CP-SAT at M=40 (1.5 MJ, K=4), 7 decisions: n_cand <= 25 solve to optimality in 1-11 s;
  n_cand = 31-34 do not close in 30-60 s (bound stays ~1.6x SA; linearization_level=2 no help).
  SA/opt on closed decisions: 0.76, 1.00, 0.87, 0.97, 1.00 (+0.78 vs incumbent, burn-in).
  The 0.87 case is NOT search budget: SA with 10x iterations and 3 seeds is also 0.871.
- Census of full M=40 runs (paper, seed 1): 46-545 launch decisions per run over K=2..7,
  38-50% with n_cand > 25.

## Flags
1. Banked SA grid does not bit-reproduce on Linux for ring/core (paper does). 0.3-3%.
2. RR beats SA on ring/core. Reviewer risk for "SA is a strong planner" in Sec. VI.
3. SA per-decision optimality degrades with candidate count; Table II (n <= 15) should be scoped.
4. CENSORING: core M=200 SA argmin at top of swept K in 11/12 (1.5 MJ) and 6/12 (3 MJ).
   Table IV reports 8.25 / 13.75 for these cells; Sec. VII-C and Table V report 13.67 / 23.50.
   The paper contradicts itself; Table IV's values and its agreement stats for these cells are censored.
5. Paper vs code: code excludes every node pending in another drone's route; Sec. III-D and
   Algorithm 2 line 4 say a node is kept if this drone would arrive first.
