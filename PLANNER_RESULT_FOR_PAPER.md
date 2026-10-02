# Planner result for the paper chat (C4, single held-out evaluation, 1 Oct 2026)

## One-line claim
On the held-out 216-deployment evaluation, a depot-aware cyclic scheduler (C4), whose no-search
initialisation reproduces our energy/depot adaptation of the published clustering method of [18]
exactly, lowers own-optimum AoI relative to that method in every layout family.

## Primary result (registered criterion: median own-optimum J below cluster patrol in EACH family)
Median of J*_C4 / J*_cluster - 1, 95% percentile bootstrap over deployments, wins = deployments where C4 is lower.

| family | median | 95% CI | wins |
|---|---|---|---|
| paper (clustered) | -21.0% | [-26.3, -18.2] | 72/72 |
| core | -10.6% | [-12.1, -9.5] | 72/72 |
| ring | -6.6% | [-8.1, -4.4] | 60/72 |
| all | -11.4% | [-12.7, -10.2] | 204/216 |

No optimum is censored for either planner. Ring is the thin family: one ring cell (M=100, 1.5 MJ) is a tie
(median +0.1%, 5/12 wins); largest single loss +9.4% (ring M=200 1.5 MJ). Every core and paper cell is 12/12.

## Secondary results
- Same fleet size K = rule's l: paper -26.9% (72/72), core -10.6% (72/72), ring -3.4% (58/72), all -12.4%.
  So the win is not an artefact of choosing a different K.
- C4's own K* vs the rule (exact / within one): 0.54 / 0.88 overall, against 0.49 / 0.84 for cluster patrol;
  paper 0.71 / 0.99 (within-one +13.9 pts over cluster, [+5.6, +22.2]); ring 0.46 / 0.86; core 0.44 / 0.78.
  The rule predicts the stronger planner's optimum at least as well as the published method's.
- Ablation (sector partition, no search) vs cluster: core +10.2%, ring +2.4%, paper -6.6%. The partition
  alone does not beat the published method; the surrogate-guided search does.
- Surrogate fidelity (planner's internal model vs simulated age term, all 2803 runs): median 0.996-0.999 per
  family, 10th percentile 0.92 (ring) to 0.99 (core).

## Method, in the terms the paper can use
Each UAV owns a disjoint sensor set and flies a fixed cyclic visit sequence, cut into energy-feasible sorties at
launch by the same rule as the cluster patrol. Sets and tours are chosen before deployment by local search
(relocation between UAVs, tour re-optimisation, frequency classes) on a per-UAV simulation surrogate that uses
only planner-side information: positions, priorities, system parameters, nominal generation rate. Two starts:
the published k-means partition and equal angular sectors; the lower surrogate value is deployed.
Disclose: C4 uses priorities, which the published method ignores (SA uses them too). Frequency classes are
nearly inert (median 4% of sensors leave class 0): sell the partition search, not frequency allocation.

## Protocol statement (accurate wording)
Designed on pilot seeds 13-15 only; knobs fixed before any pilot value. Own optima on the union of all compared
planners' K values, extended until each planner's three extreme K values exceed 1.5x its minimum (rule
registered before the evaluation comparison). A surrogate clock bug surfaced as a hang during the first
evaluation attempt; it was reproduced and fixed on pilot data, the pilot criterion re-checked, and the
evaluation rerun from scratch with the corrected, hashed code. No held-out comparative statistic, own
optimum or baseline comparison was computed before the rerun. Full record: PLANNER_RESEARCH.md.

## Consequences for existing paper text
- The abstract/VII-K sentence "published policy ... 32% better on the core layout" stays true for SA vs cluster;
  C4 adds a third planner. Suggested framing: SA = reference planner of the analysis; C4 = improved territorial
  planner developed from the published-baseline diagnosis; the rule predicts both.
- The cluster patrol's own optima in cluster_win.csv did not move under the union + tail fill (0 of 216), so
  Table XI's cluster rows stand under the registered rule (for the cluster / C4 grid union; SA grid pending).
- Negative results worth one sentence: regime-aware selection (option A) separated the synthetic families
  rather than regimes and failed on off-centre depots; the earlier hybrids (SA in territories, sqrt-due patrol,
  balanced territories, quadratic index) were worse than one of the two parents.

## Off-centre depot (secondary, same measurement rule, no censoring)
| depot | family | median | 95% CI | wins |
|---|---|---|---|---|
| (0.25L, 0.25L) | core | -13.5% | [-18.7, -10.8] | 72/72 |
| | paper | -20.0% | [-25.4, -16.8] | 72/72 |
| | ring | -8.2% | [-11.6, -5.8] | 65/72 |
| (0.10L, 0.50L) | paper | -22.8% | [-31.7, -14.6] | 24/24 |
| | ring | -1.9% | [-3.7, -0.7] | 20/24 |

C4's advantage transfers to off-centre depots, including the off-centre ring where the regime-aware selector
(option A) failed; the edge-depot ring margin is small (-1.9%) and should be stated as such.

## C4 vs SA (descriptive, seeds 1-12)
| family | median J*_C4 / J*_SA - 1 | 95% CI | C4 wins |
|---|---|---|---|
| core | -39.6% | [-45.8, -31.0] | 72/72 |
| ring | -9.6% | [-11.7, -7.7] | 64/72 |
| paper | -5.3% | [-10.3, +2.1] | 40/72 |
| all | -12.5% | [-14.7, -11.2] | 176/216 |

SA still wins the tight-energy clustered cells (paper 1.5 MJ: M=100 +11.8%, M=200 +26.4%, C4 0/12 in each).
Do not write that C4 supersedes SA. Suggested: "C4 improves on the published method everywhere and on our
reference planner on compact and radial fields; field-wide age-driven scheduling remains preferable on
clustered fields under tight energy." SA rows: sa_rerun (1).csv as uploaded; its grid satisfies the upward
tail rule everywhere and has no edge optima (downward side: no deployment within 10% of its optimum at the
smallest K swept).
