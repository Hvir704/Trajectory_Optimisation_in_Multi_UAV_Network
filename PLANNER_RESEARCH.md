# Planner research: options A (regime-aware selection) and C (new planner) -- protocol and stored findings

Written BEFORE any pilot data exists. Evaluation seeds 1-12 (the 216 reported deployments) are never used
for any choice; every choice is made on pilot seeds 13-15.

## Stored findings (not yet in the paper; include once A or C gives them a conclusion to support)
- Four hybrids failed on pilot seeds 13-14 (BASELINE_NOTES 3l): SA in k-means / balanced territories;
  periodic sqrt(w)-due territorial patrol (1/2 core seeds, starves costly-to-add sensors); territories
  balanced by sum sqrt(w); Whittle-type index w*a^2 (more regular, slower hops, net worse).
- IN THE PAPER already: the mechanism (core: cluster patrol wins by regularity + short hops; paper: SA wins
  by commute efficiency, +45% visit rate; allocation worth <= 7.1%).

## Option A -- regime-aware planner selection (pre-registered)
Components: field-wide SA ('sa') vs a territorial planner ('cluster_patrol', and 'terr_due' as the
option-C territorial candidate).
Selector statistic, computable before deployment at the rule's fleet size K = l (ex-ante inputs):
    rho = mean_k  min_{i in territory k} r_i   /   r_c_hat
territories = k-means into K (as the published method); r_c_hat = order-statistic estimate (Sec. VII-I).
Rule: territorial planner iff rho <= theta. theta: the single fitted quantity, chosen on the pilot to
maximise the number of pilot deployments where the selected planner has the lower own-optimum J
(ties -> the smaller theta). Evaluated ONCE on the 216 deployments.
Success (fixed now): overall median own-optimum J below cluster patrol's AND no family whose median is
worse than cluster patrol's by more than 1%. Reported whatever it shows.
Known limitation: with cluster_patrol as the territorial component the selector can at best TIE the
published method where it picks territories; a strict per-family win needs a territorial component that
beats cluster patrol (option C).

## Revision to A, made BEFORE any pilot objective values existed (geometry-only check)
The registered rho (nearest-territory-sensor / r_c_hat) points the wrong way: pilot geometry gives paper
1.04-5.15, ring 1.00, core 4.6-6.6 (r_c_hat is tiny on the core layout). Candidate statistics are now
pre-declared as a set, and the pilot chooses (statistic, rule, thresholds) jointly, still on seeds 13-15:
  rho1 = mean_k nearest-sensor distance of territory k / r_c_hat
  rho2 = mean_k centroid distance of territory k / r_c_hat
  rho3 = mean_k centroid distance / mean sensor radius
  rho4 = mean sensor radius / r_max
  rules: territories iff stat <= t;  iff stat >= t;  iff stat <= t1 or stat >= t2 (band)
On pilot geometry the paper family lies BETWEEN ring and core on every stable statistic, so only a band
rule can separate them -- and with three synthetic families any separating statistic risks being a
family classifier. Therefore an OUT-OF-DISTRIBUTION test is added to the success criterion: the selector
fitted on centred-depot pilot seeds must also meet the criterion on the off-centre depot deployments
(depot_q, depot_e; SA rows exist; cluster_patrol rows to be run). Failing that, A is reported as failed.

## Option C -- candidate directions (after A)
- Starvation-free periodic territorial patrol: keep cluster patrol's rotating pointer (no sensor skipped
  twice), let sortie composition follow sqrt(w)-dueness within the rotation.
- SA with an explicit regularity term: penalise selections that raise the variance of revisit intervals
  (the 18-19% loss on compact fields), keeping field-wide commute efficiency.
- Territories only for the far part of the field (shared near the depot): keeps commute efficiency near
  the depot and periodic service far away.
Each goes through the same protocol: design on pilot seeds, one evaluation on seeds 1-12.

## RESULT -- Option A: FAILED both pre-registered criteria (run 2026-09-29)

Pilot (seeds 13-15, 54 instances x {sa, cluster_patrol, terr_due}, no censoring) chose:
  rho4 = mean sensor radius / farthest radius, band rule: territories iff rho4 <= 0.332 or >= 0.845
  (45/54 correct on the pilot). Pilot rho4: core 0.18-0.25, paper 0.41-0.55, ring 0.84-0.88 -- the band
  separates the three FAMILIES; the upper threshold sits at the bottom edge of the ring range.

Single evaluation, 216 deployments (seeds 1-12), selected vs cluster_patrol at own optima:
  all +0.0% median (better 59, tied 142, worse 15); paper -24.2% (59/72 better); ring +0.0% (70 tied,
  2 worse); core +0.0% (72 tied). Criterion "overall median < 0" NOT MET: the selector picks cluster
  patrol itself on ring/core, so it can only tie there (predicted in the protocol).
Out-of-distribution (fitted rule unchanged, depot override verified applied):
  depot (0.25L,0.25L): all -2.4%, paper -21.9%, ring +11.7% (58/72 worse), core -2.6%  -> NOT MET
  depot (0.10L,0.50L): all -9.1%, paper -21.1%, ring +19.1% (17/24 worse)              -> NOT MET
  Moving the depot puts every layout's rho4 inside the band (0.35-0.68), so the selector picks SA
  everywhere -- right for paper, wrong for the off-centre ring. The statistic had learned which
  centred-depot family it was looking at, not when territories pay. Exactly the risk the OOD test was
  added for.

Pilot result for option C's starting point (terr_due, own-optimum J, seeds 13-15):
  vs cluster patrol: paper +7.7% (5/18 better), ring +36.8% (0/18), core +19.4% (2/18) -> worse everywhere
  vs SA:             paper +36.5%, ring +29.6%, core -18.2% (15/18 better)
  The earlier single-seed win over cluster patrol (core s13) was noise. Do NOT build C on terr_due; start
  from the starvation-free periodic variant or SA-with-regularity (section "Option C").

Conclusion for the paper (until C succeeds): no planner here beats the published method overall. A
regime-aware selector weakly dominates it on centred-depot fields only by choosing it where it is strong,
and does not generalise to off-centre depots.
## Option C4 -- optimised cyclic schedule seeded by cluster patrol (registered 27 Sep 2026)

Definition (cyclic_sched.py). Each UAV owns a disjoint sensor set and flies a cyclic visit sequence
over it, cut into sorties at launch by EXACTLY cluster patrol's rule (shared function cut_sortie).
Sequence = frame of 2^max(c) sub-cycles; sensor i in every 2^c_i-th sub-cycle at phase phi_i, in the
order of one master depot tour of the set. Sets, classes and tours are chosen by first-improvement
local search on a per-UAV surrogate: each UAV simulated alone with DynSim's launch stagger, dwell
estimate, cutting rule, reserve check and exact weighted-age integral, using only planner-side
information (positions, priorities, system parameters, NOMINAL generation rate; no events, no true
rates). Surrogate window 3 h burn-in / 12 h, fixed constants (not read from DynParams).
Starts: k-means (seed 0, = cluster patrol) and equal-count angular sectors; best surrogate J deployed.
Sanity: class-0 k-means schedule reproduces cluster_patrol bit-for-bit (selftest, 3 instances).

Knobs, fixed BEFORE any pilot objective value was seen: CMAX=3, NRELOC=2, MAX_PASSES=6,
MAX_EVALS=20000, REL_TOL=1e-4, INITS=(kmeans, sector), LS_SEED=0.

Variants: cyc (full), cyc_noopt (= cluster patrol, sanity), cyc_sector0 (sector partition, no
search: ablation separating partition from search).

First pilot observations (sandbox, seeds 13-14, M=100, 1.5 MJ only -- design data, not evidence):
- surrogate fidelity on the cluster schedule: core 0.998-0.999, ring 0.95-1.04, paper 0.88-1.03;
  worst beyond K_reach (gray zone depends on true rates, unknown to the planner). Events: ~37% of J
  on paper, 14-18% ring, 3-4% core; not in the surrogate.
- own-optimum J, cyc vs cluster: core -11.0%, -7.8%; paper -17.6%, -18.3%; ring -1.8%, -2.5%.
  Core M=200 s13 K=6: -25%.
- Mechanism: the gain is the PARTITION (depot-aware sectors and relocation) plus search; frequency
  classes are almost never accepted (<= 15/200 sensors off class 0). Negative sub-finding: on these
  fields a skipped visit saves only a short detour (hover is schedule-invariant, Prop. 3), so the
  allocation lever stays weak even with periodic, starvation-free frequency classes.
- Sector start alone is NOT uniformly better than k-means (core M=50 s13 K=4: +8% surrogate).

Success (fixed now, unchanged from the handoff): median own-optimum J below cluster patrol's in
EACH family on seeds 1-12, with 95% bootstrap intervals and win counts; cyc's own K* agreement
with the rule reported; cyc_sector0 ablation reported. Evaluated once.
Kill rule: if on the full pilot (seeds 13-15, M 50/100/200, 1.5/3 MJ) cyc's median own-optimum J
is not below cluster patrol's in every family, C4 is not taken to seeds 1-12 in this form.
Known risk: ring margin ~2% on the first pilot deployments.

## C4 secondary analyses (registered 29 Sep 2026: after option A's result, before the full C4 pilot)
Reported alongside the primary criterion; none of them changes the pass/fail rule above.
1. Common fleet size: C4 vs cluster patrol at K = the rule's point prediction l per deployment
   (median, 95% bootstrap interval, wins) -- separates better scheduling from a different K*.
2. C4 vs SA at own optimum (pilot and evaluation; SA rows from the same machine only).
3. Out-of-distribution depots: C4 vs cluster patrol on depot_q (0.25L,0.25L) and depot_e (0.10L,0.50L),
   seeds 1-12, cluster rows already exist. Motivation: option A passed centred depots by classifying
   families and failed exactly here. A C4 loss here is reported in the paper next to the main result.
4. Ablation cyc_sector0 (sector partition, no search) at own optimum.
5. Information disclosure: C4 uses priorities w_i; the published method is priority-blind. SA also
   uses w_i. Stated in the write-up.
6. Reproducibility check before the pilot: cluster_patrol J in pilot.csv (Ryzen) must equal the sandbox
   value for core M=100 1.5 MJ s13 K=5: 1.444978e+05 (both planners are deterministic).

## RESULT -- C4 full pilot (seeds 13-15, 54 deployments, Ryzen, 29 Sep 2026)

Own optimum, cyc vs cluster_patrol: core -10.2% [-16.0, -6.1] 17/18; paper -23.4% [-29.0, -18.2] 18/18;
ring -5.3% [-8.3, -2.2] 16/18; all -11.0% 51/54. KILL RULE PASSED (median below cluster in every family).
Reproducibility: cluster core M=100 1.5 MJ s13 K=5 = 1.445e5 on the Ryzen, same as the sandbox.
Censoring: cluster's optimum sits at the LOWER edge of the swept range in 3 ring deployments
(M=50 1.5 MJ s13, s15; M=200 3 MJ s14), cyc's in 1 (M=50 1.5 MJ s15). The anti-censoring extension
only extended upward. Worst case (all three flip to cluster): ring median still -3.8%.
Ablation cyc_sector0 (sector partition, no search) vs cluster: core +11.5% (4/18), paper -6.7%
(14/18), ring +4.7% (5/18), all +2.5%. CORRECTION to the sandbox reading: the partition alone does
NOT carry the gain; on core and ring it loses. The gain is the surrogate-guided search.
cyc vs SA: core -41.4% (17/18), ring -8.4% (17/18), paper -13.4% [-19.7, +12.2] 10/18 -- SA still
wins the tight-energy paper cells (1.5 MJ, M=100/200: +2% to +21%).
Only core loss: M=200 3 MJ s13, cyc K*=9 (+2.8%) where cluster K*=24 and SA K*=31 -- to inspect.

## Measurement rule for own optima (registered 29 Sep 2026, BEFORE the C4 evaluation)
Own optimum = argmin over the UNION of all compared planners' K values per deployment, extended by
fill_k.py (lower edge down to K=1, upper edge up) until no planner's optimum sits on an edge. Applied
identically to every planner. Reason: the anti-censoring extension fires only at the top edge, so a
planner with an interior local minimum is never extended past it (non-convex J(K) on core M=200).

Pilot after fill_k (69 extra runs, 2 rounds): cyc vs cluster core -10.2% [-16.0, -7.2] 18/18; paper
-23.4% 18/18; ring -5.3% [-8.3, -2.2] 16/18; all -11.0% 52/54; no censoring left. cyc vs SA: core
-41.4% 18/18, ring -8.4% 17/18, paper -13.4% [-19.7, +12.2] 10/18.
The fill moved CLUSTER PATROL's own optimum in 3/54 pilot deployments, all core M=200: 1.5 MJ s13
K* 10->12 (J -0.8%), 1.5 MJ s15 K* 8->12 (J -8.6%), 3 MJ s13 K* 24->29 (J -0.4%). cyc's earlier
core loss (3 MJ s13, K*=9) was this artefact: on the full grid cyc K*=30, -8.4%.
IMPLICATION FOR THE PAPER: the already-reported cluster (and SA) own optima on seeds 1-12 were taken
on unequal, top-extended grids; the same artefact can move them in the core M=200 cells.

## Trailing-margin rule (registered 29 Sep 2026: evaluation runs in progress, NO evaluation comparison
## computed or viewed yet)
An interior optimum is not proof: J(K) is non-convex on core M=200. Own optima are taken after
fill_k.py --tail 3 1.5: for every compared planner the common grid is extended upward until that
planner's 3 largest K all have J > 1.5 x its current minimum, and downward likewise or to K=1 (cap 80).
Symmetric in the planners; applied to cluster_patrol, cyc, cyc_sector0 on seeds 1-12 before the
reported comparison. Check (pilot seed 13, core M=200 3 MJ, cluster only): starting from the default
range K=4..15 the rule walks to K=34 and finds K*=29, the basin the earlier top-edge extension found only
by chance; J/J* at K=32-34 = 1.85, 4.69, 8.0 (reach cliff).
The comparison tables printed by the running command chain predate this pass and are superseded by it.
SA: the cyc-vs-SA secondary on seeds 1-12 uses sa_rerun.csv (i9, top-edge extension only) unless SA is
rerun on the Ryzen with the same rule; the write-up states which.

## INCIDENT -- evaluation v1 aborted; C4 v2 (29-30 Sep 2026)
What happened: the evaluation run of cyc (v1, as committed / hashed) hung after 1896/1901 runs; five
workers busy for over an hour on M=50 jobs that normally build in ~3 s.
Root cause (surrogate bug, not a tuning issue): on a reserve breach at arrival, the v1 surrogate broke
out WITHOUT advancing time and gave no empty-sortie turnaround. DynSim instead flies the leg, skips the
node, flies home from the previous position, and adds t_c after a sortie that served nothing. When a
UAV's whole sequence is one sensor whose planned hover estimate passes but whose true hover at arrival
breaches (a band of ~75 J for a far core sensor), the v1 surrogate clock never advanced: infinite loop.
Reproduced on PILOT seed 13 (core M=50, far sensor, energy set just above its fly-alone cost): v1 hangs
for E_usable - fly_alone in {250, 275, 300} J; v2 finishes everywhere. No evaluation data used.
Fix (v2): surrogate follows DynSim exactly on breach (leg time, home from previous position, t_c after
an empty-served sortie) + a hard error if the surrogate clock ever fails to advance. No new knob.
Effect: surrogate value of the cluster schedule unchanged on the fidelity instances (breaches are rare
there), selftest still bit-identical, but search trajectories change: pilot core M=100 1.5 MJ s13 K=5
cyc J 1.2867e5 (v1) -> 1.2513e5 (v2). So the planner changed and everything is redone:
  - pilot cyc rerun with v2, fill_k --tail 3 1.5, kill rule re-checked BEFORE any evaluation;
  - evaluation redone from scratch with v2 (cyc and cyc_sector0; cyc_sector0 does no search, so v1/v2
    give identical schedules for it).
What was seen from seeds 1-12 before the abort: console lines with per-run cyc J values; NO cluster
comparison, no table, no own-optimum computed. v1 partial output archived unopened
(eval_cyc_v1_aborted.csv); after the final evaluation it may be compared with v2 only to report how
much the bug fix changed results.
Record of the aborted v1 run (30 Sep 2026): v1 committed AFTER the abort as 57b2c74 ("code as run");
SHA-256 of the files as run: cyclic_sched.py F992C521...6416 (byte-identical to the v1 tested in the
sandbox), run_grid.py 66F1149E...BE45, fill_k.py 85262673...020C, own_opt_compare.py BD7B32B7...432F.
eval_cyc.csv at abort: 1896 cyc rows, 0 cyc_sector0 rows; no fill, no eval_all, no kstar file, so no
comparison could have been computed. Archived unopened in eval_v1_aborted/.
v2 cyclic_sched.py SHA-256 (LF): A85C0B7C49D212FAF5E43DDF7613F8BC22D51321E844705F18515E0957EC3E74.

## RESULT -- C4 v2 pilot (seeds 13-15, commit 882a473, 30 Sep 2026)
Reproducibility: cyc core M=100 1.5 MJ s13 K=5 = 125126.6 on the Ryzen = sandbox v2 value (v1: 128666.7).
Own optimum after fill_k --tail 3 1.5, cyc vs cluster_patrol: core -10.4% [-16.0, -6.9] 18/18;
paper -23.2% [-28.8, -18.5] 18/18; ring -4.8% [-8.4, -1.7] 16/18; all -11.5% 52/54; no censoring.
KILL RULE PASSED for v2. Ablation cyc_sector0 unchanged (no search): core +10.5%, paper -6.7%, ring +4.7%.
Tool bug found in this pass (fill_k tail rule, measurement only): a planner's missing row at the newest
top K counted as a tail failure, so with several planners the grid ratcheted up one K per round to the
cap (K=80 in 5 paper deployments; 1599 fill runs). This only ADDS K values beyond what the registered
rule requires; it cannot move an optimum except by finding a lower basin, and none was found (J/J* at
K=80 is 9-12). Pilot numbers stand. Fixed: a missing row defers the tail decision to the next round.
Check on pilot paper M=50 1.5 MJ s13 (cluster + cyc_sector0): old walks to K=80 (160 rows), fixed stops
at K=8 (16 rows), identical K* and J* for both planners. fill_k.py SHA-256 (LF) E5E3D24B...A11A7F.

## RESULT -- C4 v2 SINGLE EVALUATION (seeds 1-12, 216 deployments, Ryzen, 30 Sep - 1 Oct 2026)
Code: cyclic_sched.py v2 (882a473, SHA-256 A85C0B7C...EC3E74); fill_k.py with ratchet fix (E5E3D24B...A11A7F),
--tail 3 1.5, 13 fill rounds, 2258 fill rows; no grid reached the K=80 cap (max K = 35); no duplicate rows.
PRIMARY (registered criterion: median own-optimum J below cluster_patrol in EACH family) -- MET:
  core  -10.6% [-12.1, -9.5]  72/72 | paper -21.0% [-26.3, -18.2] 72/72 | ring -6.6% [-8.1, -4.4] 60/72
  all   -11.4% [-12.7, -10.2] 204/216; no censoring for either planner.
  Per cell: every core and paper cell 12/12 wins. Ring is the thin family: M=100 1.5 MJ median +0.1% (5/12
  wins, i.e. a tie), M=50 1.5 MJ -3.8% (9/12), M=200 1.5 MJ -5.4% (11/12, worst +9.4%); the three 3 MJ ring
  cells -6.6% to -12.6%. Largest single loss +9.4% (ring M=200 1.5 MJ s8).
SECONDARY (reported, not pass/fail):
  common K = rule's l: core -10.6% 72/72, paper -26.9% 72/72, ring -3.4% 58/72, all -12.4% 202/216.
  K* vs rule (exact / within one): cyc 0.54 / 0.88 vs cluster 0.49 / 0.84 overall; paper 0.71 / 0.99 vs
  0.47 / 0.85 (within-one diff +0.139 [+0.056, +0.222]); ring 0.46 / 0.86 vs 0.56 / 0.89; core 0.44 / 0.78
  vs 0.43 / 0.78.
  Ablation cyc_sector0 vs cluster: core +10.2% (16/72), paper -6.6% (51/72), ring +2.4% (27/72), all +2.2%:
  the partition alone does not beat the published method; the surrogate-guided search does.
  Surrogate fidelity over all 2803 cyc runs (J_sur / simulated J_age): core median 0.999 [p10 0.991];
  paper 0.997 [0.954]; ring 0.996 [0.922]. Start chosen: sector 1909, k-means 894. Sensors off class 0:
  median 4%, max 28%. Build time median 5.6 s, max 40.6 s.
  Grid check on the published-method numbers: the union + tail fill changed cluster_patrol's own optimum in
  0 of 216 evaluation deployments (it had in 3/54 pilot ones), so cluster_win.csv's own optima stand
  under the registered rule for the cluster / cyc / cyc_sector0 grid union.
NOT YET DONE: cyc vs SA on seeds 1-12 (SA rows are i9 data, top-extended only; needs the paper chat's SA
fill on the i9 or a Ryzen rerun); off-centre depot secondary (run_depot.py --planner cyc).
