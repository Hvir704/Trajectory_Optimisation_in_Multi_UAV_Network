# Fleet-sizing theory v3: served-set rule and commute bracket

Status 2026-10-14 (third review taken; fourth test run, section 13.1): v3.2, plus section 10 (v3.3 and the second registered test) and section 11 (v3.4, the crowding
hop, and the third registered test: **all 8 verdicts hold**, 11.2). Since v3.1 this revision adds:
- the H5 result;
- the second independent review (`claude/THEORY_V3_VERIFY.md`). Every number in its reproduction table was
  reproduced there, and its findings are taken here. **Its main finding corrects an error of ours:** seeds 13-15 of
  the held-out set are not new (section 6).
- The first review (`claude/THEORY_V3_REVIEW.md`) is taken except for one point, H4 for C4, which could not be
  registered because the held-out run had already been done. C4's held-out rows are therefore descriptive (section 6).

Code (branch `theory-v3`, not yet in the repository):

| file | what it does |
|---|---|
| `theory_v3.py` | the rule |
| `retro_v3.py` | the 272 existing instances |
| `heldout_v3.py` + `heldout_v3_predictions.csv` | the registered test; unchanged since registration. Its docstring's "seeds 13-20 (never used before)" is wrong, see section 6 |
| `followup_v3.py` | post-registration analyses and the H5 run and evaluation; unchanged since H5 was registered |
| `predict_horizon_v3.py` | regenerates the H5 predictions file byte for byte (the original generator was removed with the reverted H4 amendment) |
| `seeds_split_v3.py` | post-registration: the held-out evaluation split into seeds 13-15 and 16-20 |
| `horizon_v3.py` | the rule as a function of mission length |
| `heldout2_v3.py` + `heldout2_predictions.csv` | the second registered test (section 10); fixed by its manifest hash |
| `theory_v34.py` | v3.4, the crowding-corrected hop (section 11); `theory_v3.py` stays as registered |
| `kappa_diag.py` | the level-error decomposition (section 11) |
| `heldout3_v3.py` + `heldout3_predictions.csv` | the third registered test (11.1, 11.2); fixed by its manifest hash |

Everything uses only the site survey (positions, priorities), nominal generation rates, the airframe, the energy
inventory and the mission length, unless marked *pilot*.

**Notation.** T is the scored window, as in the paper; T_b is the burn-in. The *mission length* is
T_mis = T_b + T (12 h as run). The code's `horizon` argument is T_mis. H5's predictions are the same whether "24 h"
is read as T_mis or as the window.

## 0. Result in one paragraph

**Current state (2026-10-13, after the third review; details in sections 10-12, which supersede this section, 8 and
9 where they differ).** Three registered held-out tests. The rule is v3.3; v3.4 (crowding-corrected hop) is a
registered, tested extension for the annealer and the tour patrol.
- Test 2 (seeds 21-40, hash committed publicly, author-dated before the run): H1-H4 and H6 hold, **T2-H5 (C4
  within-one) and T2-H7 (mission-length switch) fail.**
- Test 3 (seeds 41, 43-49, same registration): all 8 verdicts hold, the gain concentrated in the ring 3 MJ cells.
  SA within-one 0.93 and p90 regret 10.6% (paper rule, pre-specified without a verdict: 0.79 and 29.7%); tour patrol
  0.90 and 9.1%.
- Test 4 (seeds 50-57, hash pushed before the run, 2026-10-14): the **two-candidate local check** (about 4 simulated
  fleet sizes per field) holds all four criteria for SA, the tour patrol and C4 (worst regret 8.7%, 6.7%, 6.2%; mean
  0.2-0.3%) and three of four for the cluster patrol (mean 1.006% against < 1%; worst 23.3%). v3.4 replicates
  (R1-R3). v3.3 alone does not improve the worst case for C4 (37.8% against 27.6%) or the cluster patrol (tie).
- Weak points: under-stranding at core 3 MJ (served term undervalued), reach-side misses by one UAV (up to 30%),
  the hop trend with K, no point-prediction gain for the territorial planners (C4, cluster patrol), Gao worse under
  v3.3, and v3.4's implementation not being the fixed point its derivation describes (section 11).

The paragraph below is the original test-1 summary, kept as the record.

On 160 held-out instances, predicted and hashed before any run, the three registered hypotheses hold for the
annealer. 54 of these instances (seeds 13-15) are fields that had already been swept, under the older simulator, in
the planner comparison's pilot runs; the other 100 (seeds 16-20) had never been run.
- **On the core layout, all 160:** v3 cuts the regret of using the rule instead of a sweep from median 8.6% / 90th
  percentile 108% / max 146% to 0.5% / 18.1% / 31.5%.
- **On the core layout, seeds 16-20 only:** p90 109% -> 3.9% and max 146% -> 31.5%. The median is 0.0% under both
  rules, so H1's "lower median" is not met on that subset.
- **Within one UAV:** 0.65 -> 0.88 on the core and 0.81 -> 0.88 overall (seeds 16-20: 0.63 -> 0.90 and
  0.81 -> 0.89).
- **Ring:** the commute bracket contains the annealer's optimum in 48 of 48.

The gain comes entirely from instances where abandoning one far outlier pays (post hoc):
- the annealer is better in 15 of 15 (10 of 10 on seeds 16-20);
- the tour patrol in 14 of 15;
- C4 in 11 of 15 (two-sided sign test p = .12), positive on average but not established. On seeds 16-20 alone C4
  is 6 better, 4 worse, and C4 was designed on seeds 13-15.

On clustered fields, abandoning a few far sensors does **not** pay:
- the annealer is even (3 better, 4 worse);
- the tour patrol is worse in 7 of 7;
- C4 is worse in 5 of 7.

The rule is for a mission of length T_mis. Its stranding branch switches off as T_mis grows; at 7 days it equals the
plain rule. A second registered test at T_mis = 24 h (H5) meets its criterion: where v3's prediction changes with the
mission length, the measured optimum is closer to the new prediction in 4 of 4 (p = .06; the registered criterion
was at least 3 of 4).

## 1. What was wrong with the v2.2 theory

**Core layout.**
- The reach cap is set by the single farthest sensor, but there the optimum abandons that sensor and flies many
  more UAVs. J(K) has two minima.
- Prop. 6 (stranding condition) tests one UAV at a time and stops at the near minimum.
- It is capped at the full-set commute cap.
- It prices visits at about 5x the energy actually spent per stop (kappa median 0.18, Lemma (cost scale)).

**Ring at 3 MJ.**
- The closed form predicts 6 for every M; the measured optima are 8.5, 7.9 and 6.7.
- The paper explains this only with the *measured* fall of hop time with K.

## 2. The served-set surrogate (a model for a mission of length T_mis)

Notation as in the paper:
- U = (1 - rho) E_max.
- r_reach(K) = v (U/K - P_h B/R) / (2 P_f).
- Phi(K) = K (U - K P_f t_c) / (U + K t_c (P_bar - P_f)).

For a served set S:

- H_S = sum_{i in S} lambda_i / R. This is the fleet's hover load (nominal rates).
- d_S = (closed nearest-neighbour + 2-opt tour length through S) / |S|. This is the a-priori hop.
- Gamma_S(K) = v (Phi(K) - H_S) / d_S. This is the visit rate (Prop. 3 with a uniform hop).
- pi_i = r_i / mean_S r, A_S = sum sqrt(w pi), B_S = sum sqrt(w / pi). This is the radial cost profile of Prop. 4.

With R(K) = {i : r_i <= r_reach(K)}, the surrogate is

    J~(K; T) = A_R B_R / (2 Gamma_R(K))  +  (T_b + T/2) * sum_{i not in R(K)} w_i,      defined where Phi(K) > H_R(K)   (S1)

- **First term:** Prop. 4's form J* = A B / (2 R*), with R* *approximated* by the uniform-hop visit rate Gamma_S.
  This is not Prop. 4 itself. If the uniform hop were literal, every visit would cost the same, and the optimum
  would be (sum sqrt w)^2 / (2 Gamma), which is <= A B / (2 Gamma). The radial profile pi carries the cost
  differences the uniform hop leaves out.
- **Second term:** the exact age integral, over the scored window [T_b, T_b + T], of a sensor never served, with
  every age zero at t = 0.

**What it leaves out**, stated plainly:
- the regularity factor (1 + CV^2) of irregular visits;
- the planner's allocation inefficiency;
- the event term;
- route-dependent marginal costs, since it uses the radial profile only.

Its absolute level is therefore an approximation. The served term is low in every family: the measured
kappa = J_age / (served term) for the annealer has these medians (retrospective):

| family | 1.5 MJ | 3 MJ |
|---|---|---|
| core | 1.35 | 2.38 |
| clustered | 1.26 | 1.48 |
| ring | 1.32 | 1.84 |

The overall range is 0.92-3.13, so the served term is roughly 1.2-3x short. On the core this biases the rule against
stranding, most at 3 MJ. The *pilot* variant rescales the served term by kappa, measured at the smallest
stranding-free fleet size of the original sweep, from one run of the planner that will be flown.

**The rule (v3):** K_hat(T_mis) = argmin over feasible integers 1 <= K <= 200 of J~(K; T).
- Feasible means |R(K)| >= 2 and Phi(K) > H_R(K).
- `theory_v3.surrogate_rule` evaluates every such integer by brute force, stopping once K is infeasible past the
  peak of Phi, so it does not depend on Theorem 1.
- In 7 of 432 instances (clustered M=100, 6 MJ) feasible K > 200 exist. An uncapped search (to K = 400) gives the
  same predictions everywhere (second review).
- The mission length is an explicit input (`horizon` = T_mis).

## 3. Theorem 1 (nested candidates)

Sort the distances r_(1) >= r_(2) >= ... Let S_j be all sensors except the j farthest, and
Kr(S_j) = v U / (2 P_f max_{S_j} r + v P_h B / R).

Then:
- Kr(S_j) is non-decreasing in j.
- R(K) = S_j exactly on I_j = (Kr(S_{j-1}), Kr(S_j)], with Kr(S_{-1}) := 0.

*Unimodality.* With a = P_f t_c and b = t_c (P_bar - P_f) >= 0, the numerator of Phi'(K) is U^2 - 2aUK - abK^2.
It is positive at K = 0 and decreasing for K > 0, so Phi has one maximiser,
K_c = U / (t_c (P_f + sqrt(P_f P_bar))), which is U / (2a) when b = 0. Phi rises, then falls. Let K_c* be its
**integer** argmax.

**Theorem 1.** For each j, let C_j be the integer argmax of Phi over the integers K of I_j at which Phi(K) > H_{S_j}
and |S_j| >= 2.
- C_j is empty when I_j contains no such integer.
- When C_j is non-empty, it is the integer of I_j nearest to K_c*. That is min(floor Kr(S_j), K_c*) if
  K_c* > Kr(S_{j-1}), and the smallest integer of I_j otherwise.

Every minimiser of (S1) over the feasible integers lies in the union of the C_j.

*Proof.*
1. On I_j the served set is the constant S_j, so A, B, H, d and the stranded sum are constant.
2. Where Phi > H_{S_j}, J~ is strictly decreasing in Phi(K) - H_{S_j}, so its minimiser on the integers of I_j is
   Phi's integer argmax there.
3. That argmax is the integer nearest to K_c*, because Phi is unimodal.
4. The global minimiser is the best of these. QED

Consequences:
- **j = 0** gives the paper's rule. round(K_c) = K_c* in all 272 + 160 instances here; the theorem does not rely
  on that.
- **"Stranding pays"** is exactly "the best candidate has j > 0".
- **Prop. 6** is a local search on the same sequence.
- **Brute force:** the argmin of (S1) equals the candidate rule in 272/272 and 160/160 (both reviews).

**With the exact productive power (paper v2.2.1, Corollary 1).** Under hover conservation, P_bar is not a free
input. Per served set,
Phi_S(K) = K (U - K P_f t_c - (P_h - P_f) H_S t_c) / U,
a concave quadratic in K. Theorem 1 holds unchanged in structure, with a served-set-dependent peak
K_H(S_j) = (U/t_c - (P_h - P_f) H_{S_j}) / (2 P_f). With this Phi:
- no prediction changes (0 of 272 at 12 h, 24 h and 7 days; 0 of 160; 0 of 16), and the same holds with the
  full-field H;
- the candidate rule equals its own brute force in 432 of 432 (second review).

v3 keeps the fixed-P_bar Phi, the same approximation the paper's closed form makes.

*Served-set optimality inside R(K), to first order.*
- Holding the profile normalisation, H and d fixed, dropping any served sensor never lowers (S1) at the chosen K,
  in all 272 instances. The same holds with H updated.
- So serving the whole reach set is surrogate-optimal to first order only.

## 4. Proposition 2 (commute bracket, conditional)

Assume:
- (a) weighted age is minimised where the visit rate is maximised. This is the Prop. 4 equivalence; it ignores the
  regularity factor, so it is an approximation for every planner.
- (b) tau_hop(K) >= tau_hop(K_c*) for every K < K_c*.
- Phi(K_c*) > H.

Then Gamma(K) = (Phi(K) - H) / tau_hop(K) has **a** maximiser at K >= K_c*.

*Proof.*
1. For K < K_c*, Phi(K) <= Phi(K_c*) by unimodality.
2. tau_hop(K) >= tau_hop(K_c*) by (b).
3. So Gamma(K) <= Gamma(K_c*), and no K below K_c* does better. QED

On a commute-bound field without stranding, an optimum therefore lies in [K_c*, floor Kr]. The upper end comes from
the no-stranding assumption, not from the proposition.

This is a lower bound **under (a)-(b)**, not unconditionally.
- **(b)** holds as a trend for every planner: tau_hop falls with K, with median Spearman correlation -0.8 to -1.0.
  tau_hop is estimated from per-run means as (T_s - 2 r_c / v)(1 - h) / n, h = (P_bar - P_f)/(P_h - P_f).
- **(b) in the form the proof needs** holds less often (second review, retrospective ring, 72 instances):

  | planner | instances where (b) holds as needed |
  |---|---|
  | SA | 72 |
  | tour patrol | 69 |
  | Gao | 66 |
  | Liu | 62 |
  | C4 | 54 |
  | cluster patrol | 49 |
  | Asghar | 44 of 60 |

- **(a)** is supported by App. B of the paper: the measured-visit-rate argmax is within one UAV of the AoI optimum in
  every ring instance for SA and the tour patrol, in 97% for C4, and 92% for the cluster patrol and Liu.

## 5. Retrospective (272 existing instances; v3 was designed on these)

Ex-ante commute inputs.

| annealer | exact | within one | regret median | p90 | max |
|---|---|---|---|---|---|
| old rule, all 272 | 0.56 | 0.80 | 0.0% | 25.2% | 167.8% |
| v3, all 272 | 0.60 | 0.85 | 0.0% | 14.8% | 35.1% |
| v3 + pilot | 0.61 | 0.85 | 0.0% | 13.1% | 27.4% |
| old, core 72 | 0.35 | 0.56 | 11.7% | 88.2% | 167.8% |
| v3, core 72 | 0.44 | 0.75 | 0.5% | 21.5% | 35.1% |
| old / v3, clustered 128 | 0.71 / 0.75 | 1.00 / 1.00 | 0 / 0 | 6.4% / 6.3% | 39.2% / 18.3% |

- p90 here is linear interpolation (`np.quantile`). The paper's Table VII uses nearest rank; one convention must be
  chosen before this goes into the paper. Example: held-out SA core v3 is 18.1% linear and 21.4% nearest rank.
- **v1 replication (first review):** the same fields under the older simulator give the same direction and size:
  all 272 within-one 0.81 -> 0.87, max 176% -> 35%.

**Other planners, v2.2** (own optima, 216 deployments). Within one UAV, old -> v3 (pilot):

| planner | core | clustered | all | core p90 regret old -> v3 | core max | ring bracket coverage |
|---|---|---|---|---|---|---|
| C4 | 0.79 -> 0.83 (0.83) | 0.96 -> 0.94 | 0.88 -> 0.88 | 33.8 -> 15.2% | 84 -> 32.5% | 59/72 |
| cluster patrol | 0.78 -> 0.81 (0.76) | 0.85 -> 0.85 | 0.84 -> 0.85 | 43.6 -> 15.1% | 109 -> 37% | 58/72 |
| Liu et al. | 0.75 -> 0.75 (0.72) | 0.96 -> 0.96 | 0.86 -> 0.86 | 40.3 -> 12.1% | 89 -> 23% | 61/72 |
| tour patrol | 0.58 -> 0.71 (0.75) | 0.99 -> 0.99 | 0.79 -> 0.83 | 87.2 -> 19.8% | 159 -> 41% | 69/72 |
| Asghar et al. | 0.49 -> 0.51 (0.47) | 0.47 -> 0.50 | 0.60 -> 0.62 | 107 -> 36% | 409 -> 167% | 41/72 |
| Gao et al. | 0.86 -> **0.71** (0.67) | 0.68 -> 0.65 | 0.76 -> **0.70** | 27 -> **63%** | 71 -> **191%** | **32/72** |

(The annealer's ring bracket coverage is 72/72.)

- The first review found within-one *falling* for every non-annealer planner on **v1** data (C4 0.85 -> 0.71). On
  v2.2 it falls only for Gao.
- Under v1 the other planners stranded far less than the annealer. Under v2.2, at core M=200 3 MJ:
  - SA, the tour patrol and Asghar strand in 11 of 12 deployments;
  - Liu in 6;
  - C4 and the cluster patrol in 4;
  - Gao in 0.
- The planner dependence is real either way, so state both.
- **The pilot** does not help agreement uniformly.
  - Retrospectively it helps the annealer (core within-one 0.75 -> 0.79) and the tour patrol, and is neutral for C4.
  - On the core it lowers within-one slightly for the cluster patrol, Liu, Gao and Asghar.
  - On clustered fields it raises Asghar's within-one (0.50 -> 0.57) and cuts its max regret (124.5% -> 84.5%).
  - **On the held-out set it lowers the annealer's within-one** (core 0.88 -> 0.81, all 0.88 -> 0.85), while
    improving its p90 and max (core p90 13.1%, max 26.9%).

**Reference K (v2.2.1 data).** After the downward fill, union fills gave the annealer rows below its original sweep.
`retro_v3.py` and `horizon_v3.py` now take the ex-ante inputs, and the pilot run, at the instance's reference K: the
first K of the original sweep (column `K_ref`), or the smallest swept K on older files.
- On the pre-fill data this reproduces every table above exactly.
- On the filled data, K_ref differs from the smallest swept K in 56 instances, and no annealer prediction changes.
  The annealer, tour patrol, cluster patrol, C4 and Liu rows are identical.
- Only Gao et al. and Asghar et al. change, because their own optima moved. Asghar's all-216 within-one is now
  0.60 -> 0.62, and its ring within-one is 0.83 under both rules. Gao's ring regret median is 2.1%.
- `horizon_v3.py` gives the same table as before.

## 6. Registered held-out test

- **Instances:** 160 (20 primary cells x seeds 13-20).
  - **Seeds 13-15 are not new.** They are the pilot seeds of the planner comparison. The same fields had been swept
    for SA, C4, C4-sector0, the cluster patrol, terr_due, Gao and Asghar under the pre-v2.2 simulator:
    - `pilot*.csv` on `sim-v2`, committed 29-30 Sep;
    - `gzz_pilot.csv` and `asghar_pilot.csv`, 7 Oct.

    That makes 54 instances, with full K-sweeps including the core stranding region (core optimum strands in 5 of 18).
    Their old-simulator SA optima equal the held-out v2.2 optima in 81% of them and are within one UAV in 96%. C4
    and both published methods were designed or tuned on these seeds.
  - **v3 was not designed on them:** its design used seeds 1-12 only. But the earlier statement that seeds 13-14
    were "used only in a 4-run regularity decomposition, not in any sweep" was wrong. So are the registered script's
    "never used before" and the first review's suggestion to use seeds 15-22, since seed 15 is equally used.
  - **Seeds 16-20 (100 instances) appear in no file before the held-out run.**
- **Predictions:** generated by code fixed before the run. The sha256 (4c77e53e...) was recorded in the project log
  and conversation before the run. There is no third-party timestamp: the local git dates are author-controlled, and
  this branch's history was re-created once. The claude.ai conversation record shows the hash set on 2026-10-11
  around 11:56 UTC and the results first uploaded on 2026-10-12 at 05:22 UTC, but it is not public. The predictions
  regenerate byte for byte from the code, so a reader can verify that they follow from the rule, but not when the
  rule was fixed. For any future test, push the hash alone to a public place before running.
- **Planners:** SA, tour patrol and C4. The run took about 4.6 h on 22 processes.
- **Sweeps:** every sweep interior by the registered rule. The registered rule checked only the top edge; see below.

| planner, family | old: w1, regret med / p90 / max | v3: w1, regret med / p90 / max |
|---|---|---|
| SA, all 160 | 0.81, 0.9 / 26.9 / 146.0% | **0.88**, 0.6 / 16.7 / 31.5% |
| SA, core 48 | 0.65, 8.6 / 108.2 / 146.0% | **0.88**, 0.5 / 18.1 / 31.5% |
| SA, clustered 64 | 1.00, 0.2 / 12.1 / 18.9% | 1.00, 0.5 / 12.1 / 18.9% |
| SA, ring 48 | 0.71 (identical) | bracket covers **48/48** |
| tour, core 48 | 0.65, 8.0 / 89.5 / 123.4% | 0.75, 2.4 / 12.3 / 30.1% |
| tour, clustered 64 | 0.98, 0.0 / 10.2 / 23.8% | 0.98, 1.1 / 15.4 / 28.1% |
| C4, core 48 | 0.81, 1.0 / 38.7 / 98.8% | 0.81, 1.5 / 21.7 / 29.3% |
| C4, clustered 64 | 0.92, 0.0 / 5.8 / 18.3% | 0.92, 0.0 / 11.9 / 26.8% |

**Registered hypotheses (SA, all 160):**
- **H1** (core: lower median and p90 regret): **holds**.
- **H2** (within-one not lower): **holds** (0.81 -> 0.88).
- **H3** (bracket >= 90% on the ring): **holds** (1.00).

**No hypothesis was registered for C4 or the tour patrol; their rows are descriptive.** H4 for C4, suggested by the
first review, was drafted after the run had already been done and was withdrawn.

**Seeds 16-20 only** (post-registration sensitivity check, `seeds_split_v3.py`; seeds never run before):

| SA | old: exact, w1, regret med / p90 / max | v3: exact, w1, regret med / p90 / max | v3 != old: better / worse |
|---|---|---|---|
| core 30 | 0.57, 0.63, 0.0 / 109.1 / 146.0% | 0.60, **0.90**, 0.0 / **3.9** / 31.5% | 10 / 0 (p = .002) |
| clustered 40 | 0.47, 1.00, 0.2 / 11.5 / 18.9% | 0.47, 1.00, 0.4 / 11.5 / 18.9% | 2 / 2 |
| ring 30 | 0.43, 0.73, 1.3 / 19.6 / 30.7% | identical | bracket **30/30** |
| all 100 | 0.49, 0.81, 0.1 / 27.4 / 146.0% | 0.50, **0.89**, 0.1 / 15.8 / 31.5% | 12 / 2 (p = .013) |

- **H1 as worded is not met on this subset.** The core median is 0.0% under both rules: 17 and 18 of the 30
  instances have zero regret. The p90 part holds strongly.
- H2 and H3 hold.
- On seeds 13-15 the annealer's core result has the same direction (5 better, 0 worse).
- **C4 on seeds 16-20, core:** 6 better, 4 worse (p = .75); p90 38.2% -> 22.7%; within-one 0.80 under both rules.
  C4's overall 11 / 4 includes 5 / 0 on its own design seeds 13-15.

**Pilot (evaluated as registered):**
- SA core: w1 0.81, p90 13.1%, max 26.9%. This is lower within-one than v3 a priori (0.88), with a lighter tail.
- C4 core: w1 0.88, p90 17.4%, max 24.1%. Here the pilot beats both a-priori rules.
- Tour core: w1 0.77, p90 11.6%.

**Post-registration analyses** (`followup_v3.py posthoc`; decided after the run, labelled as such).

*Paired comparison on the 22 instances where v3 != old* (15 core, 7 clustered, 0 ring). Regret change = v3 - old:

| planner | core: better / worse, mean change | clustered: better / worse, mean change | all: sign test |
|---|---|---|---|
| SA | 15 / 0, -78.5% [-98.6, -56.7] | 3 / 4, +1.8% [-0.2, +4.0] | 18 / 4, p = .004 |
| tour patrol | 14 / 1, -65.1% | 0 / 7, +15.5% [+10.2, +21.0] | 14 / 8, p = .29 |
| C4 | 11 / 4 (p = .12), -20.0% [-37.1, -2.5] | 2 / 5, +8.7% [+1.8, +16.2] | 13 / 9, p = .52 |

So:
- Abandoning the core outlier is right for the annealer and the tour patrol, and positive on average for C4.
- Abandoning 1-6 far sensors on clustered fields (paper M=200 and 400 at 1.5 MJ) is not: it is even for SA and
  hurts the others.
- A rule that strands only when the stranded set is a small fraction of total priority, or that requires a
  margin, may fix this. That would be a new rule needing its own test, not a change to this one.

*Sweep edges.* The registered sweep checked only the top edge. Five optima sit at the smallest swept K (K = 2,
bracket [3, 4]):
- tour patrol, ring M=50 1.5 MJ s=15 and 17;
- C4, s=15, 17 and 18.

These are already counted as "below the bracket", and a true optimum at K = 1 would still be below, so the counts
are exact. What may be optimistic is those planners' ring agreement and regret on these 5. SA has none, so H3 is
unaffected.

*Bracket.*
- Width: 2 UAVs in 23 instances, 4 in 23, and 3 and 5 in one each.
- SA's optimum is at K_c in 20 and above it in 28.
- For the tour patrol: 26 at K_c, 19 above, 3 below.
- For C4: 24 at K_c, 15 above, 9 below. Of the 9:
  - 3 are the lower-edge cases above;
  - 5 of the other 6 (ring M=200, 3 MJ) violate assumption (b) in its needed form.

  So C4's misses come from (b), not (a). C4's held-out ring satisfies (b)'s needed form in 36 of 47 (second
  review's computation).

## 7. Horizon dependence

All of v3's gain comes from abandoning sensors, which pays only because a never-served sensor costs
(T_b + T/2) w_i, which is finite. Surrogate only (`horizon_v3.py`; v2.2 inputs, identical to the first review's v1
table):

| mission T_mis | core: changed vs 12 h | core: strand | clustered: changed | clustered: strand | ring |
|---|---|---|---|---|---|
| 12 h (as run) | - | 21 of 72 | - | 17 of 128 | never strands |
| 24 h | 7 | 14 | 11 | 6 | |
| 7 days | 21 | 0 | 17 | 0 | |

At 7 days v3 equals the plain rule in 272/272. Present v3 as the rule for a mission of length T_mis, and the plain
rule as its long-mission limit.

**Registered follow-up (H5): run 2026-10-12. The registered criterion is met (4 of 4).**
- **Instances:** core M=100 and 200 at 1.5 MJ, seeds 13-20, SA, T_mis = 24 h. 259 + 18 runs on the Ryzen (a 259-run
  initial sweep, 4 of it from an interrupted first start, plus 18 extensions); every sweep interior.
- **Predictions:** `heldout_v3_horizon_predictions.csv`, sha256 (LF-normalised)
  eb955eb077a865301b5678cbd4697420a7f8ab01855d65cc52a3e724cb26eda7, fixed before the run.
  `predict_horizon_v3.py` regenerates it byte for byte.
- **H5** (the measured 24 h optimum is closer to v3(24 h) than to v3(12 h) in at least 3 of the 4 instances where
  they differ): **met, 4 of 4.** One-sided sign-test p = 1/16 = .06, so this is the registered criterion met, not a
  significant test on its own.
  - M=100 s=17, M=200 s=18 and s=20: the 24 h optimum is the plain rule's K. v3(24 h) has regret 0.0%; v3(12 h)
    (K=14) has 30.2%, 36.6% and 8.9%.
  - M=200 s=14: optimum 5, v3(24 h) 4, v3(12 h) 14. Closer in K, but the regrets are nearly equal (25.9% vs 26.8%).
    This is the marginal case.
  - The test was not stacked: in all 4, the 12 h optimum (held-out run) stranded (K = 13-15), and at 24 h it moved
    to the plain rule (K = 4-5).
- **All 16 instances, regret at the 24 h optimum** (registered to be reported on all 16):

  | rule | median | max |
  |---|---|---|
  | old | 22.8% | 110.1% |
  | v3(12 h) | 0.3% | 36.6% |
  | v3(24 h) | 0.0% | 25.9% |

- **Switch table** (post hoc): does v3 predict that the optimum switches between stranding and not when the mission
  goes from 12 h to 24 h, and does it?

  | | optimum switched | did not switch |
  |---|---|---|
  | v3 predicted a switch | 4 | 0 |
  | v3 predicted no switch | 1 | 11 |

  The one miss is M=100 s=13: its optimum moved from 15 (12 h) to 5 (24 h), but v3 predicted 5 at both horizons. A
  strand / no-strand count over all 16 at 24 h (16/16) is not independent evidence: v3(12 h) already classifies 12
  of them correctly, and the step to 16 is the 4 discriminating instances.
- **Remaining misses are on the reach side:**
  - M=100 s=14 and s=15: optimum 6; old, v3(12 h) and v3(24 h) all predict 5; regret 20.5% and 25.1%.
  - M=200 s=14: optimum 5, v3(24 h) and old 4, regret 25.9%.
- **Independent checks:** recomputed from the raw `heldout_v3_h24.csv` (277 rows, all v2.2, T_mis = 24 h, no
  duplicates) by two separate pieces of code. Every line matches. No optimum sits at a sweep edge (sweeps K = 2-17
  to 2-19).

## 8. Known limits (state them)

1. **Horizon.** v3 is the rule for a mission of length T_mis. At long missions it reverts to the plain rule.
2. **Clustered stranding does not pay** (section 6). The annealer is even; the others are worse.
3. **Planner dependence.** Whether and where stranding pays depends on the planner.
   - Gao's fixed territories never exploit it, so v3 is worse for Gao on the core, and the ring bracket covers
     only 32 of Gao's 72.
   - C4's benefit is positive on average but not established on the clean seeds.
   - Under v1 the other planners also stranded much less (first review).
4. **A-priori level.** The surrogate omits regularity, allocation inefficiency and events. Its served term is about
   1.2-3x short in every family (most at 3 MJ), so on the core it under-strands at 3 MJ. The pilot fixes the level
   for the planner flown, but does not uniformly improve agreement, and held out it lowers the annealer's
   within-one.
5. **Core M=200, 3 MJ** (retrospective, 12 instances).
   - v3 strands in 6 with regret 0.6-4.2% (median 2.0%). Where stranding pays, the optima are 29-31, inside the
     remaining set's gray bracket against floor Kr(S_1) = 28.
   - In the other 6 v3 does not strand. One is exact (s=5, optimum 10). The other 5 have regret 12.8-35.1%, and this
     cell holds v3's retrospective maximum.
6. **Ring.** No point prediction better than K_c, only the conditional bracket, and (b) in its needed form fails
   more often for the territorial planners.
7. **Held-out novelty.** Only seeds 16-20 are new fields (section 6).

## 9. What changes in the paper

**Theorem 1** replaces:
- Prop. 6 (stranding condition) and its sequential stopping point;
- Table V;
- Lemma (cost scale), except its kappa sentence ("about 5x", kappa median 0.18), which section 1 uses as motivation
  and which stays;
- criterion (5) as the reach-branch test.

**Criterion (5) is wired into more of the paper than a flag.** Removing it touches:
- **Result 1** (`eq:law`): its second branch is defined by the criterion. Restate Result 1 with Theorem 1 as the
  reach branch.
- **The operator algorithm** (the decision step that runs the criterion).
- **Table VII** (`tab:regret`) and **Table XII** (`tab:split`), whose split variable is the criterion. Either
  regenerate them with v3's stranding verdict, or drop them.
- **Sec. VII-I** ("keeps its properties") and the warning-indicator comparison (indicator vs criterion).
- Text that cites Prop. 6 for the stranding mechanism (two places) should cite Theorem 1.

**Two things go with these changes, and must be said:**
- **The old criterion's safety property is lost.** "No passing instance loses more than 27%" no longer exists.
  v3's own tail is 31.5% on the held-out annealer runs (35.1% retrospectively). An operator who cannot accept that
  should sweep.
- **Stranding is now recommended deliberately.** Operationally, the named sensors are not served for the mission.
  That is the price of serving the rest more often, and it is acceptable only if the mission length is known.

**Proposition 2** (conditional) replaces:
- the fitted-exponent part of App. B (Table XV and the per-sortie regressions);
- the measured-tau_hop substitution as the *account* of the ring miss. That substitution remains as the evidence
  for (b).

**App. B also holds two paragraphs that stay:**
- the estimator-choice paragraph;
- the measured-visit-rate evidence for (a), which Prop. 2 now needs.

The ring keeps K_c as its point prediction.

**Held-out section.** Report:
- H1-H3 on all 160, stating that 54 instances are on previously swept pilot fields, with the seeds 16-20 table as a
  declared post-registration check and the H1 median tie stated;
- the paired comparison, the clustered result and the planner table, with C4 and the tour patrol marked
  descriptive;
- H5 with its switch table, as the evidence that the stranding branch tracks the mission length (**retired**: T2-H7
  failed; see section 12, limit 3);
- the registration as "predictions generated by code fixed before the runs, hashes recorded in the project log
  before each run; no third-party timestamp".

Pick one p90 convention for the whole paper.

## 10. v3.3 (gray-bracket exclusion) and the second registered test

### 10.1 The refinement

**Diagnosis**, on seeds 1-20 (all seen data). Where v3 strands, two very different things happen.
- **Core.** v3 strands one outlier and jumps from about 5 to 14 UAVs (or 9-11 to 28). The outlier is then beyond
  *flight-only* reach (`2 P_f r_i / v > U/K`): no planner can spend energy on it.
- **Clustered.** v3 strands 1-9 background sensors to gain exactly one UAV (4 -> 5). At K = floor Kr + 1 those
  sensors sit in the gray bracket: beyond full-buffer reach, but still within flight reach. Planners keep trying to
  serve them, so the surrogate both charges them as abandoned and treats their energy as freed. It double-counts the
  gain. Measured: the annealer is even, and the tour patrol and C4 are worse.

**Rule v3.3.** A fleet size that strands sensors (j > 0) is a candidate only if no sensor lies in its gray bracket
`r_reach(K) < r_i <= v U / (2 P_f K)`. Fleet sizes with j = 0 are unchanged. Implemented as
`theory_v3.surrogate_rule(..., gray="exclude")`; the default (`"charge"`) is v3, so every registered v3 prediction is
unchanged (both hashes verified after the change).

**On seen data** (designed here, so this is not evidence; nearest-rank p90):
- v3.3 differs from v3 in 16 instances, all clustered and none on the core, so every core gain is kept.
- Clustered p90 regret:

  | planner | retrospective, v3 -> v3.3 | held-out 1, v3 -> v3.3 |
  |---|---|---|
  | tour patrol | 12.3 -> 9.3% | 15.9 -> 10.5% |
  | C4 | 9.4 -> 8.5% | 12.3 -> 10.9% |
  | cluster patrol | 34.1 -> 30.7% | |
  | Liu | 18.0 -> 14.6% | |

- v3.3 is no worse than v3 in every planner and family summary except Asghar et al.: on clustered fields (mean
  regret 30.5 -> 31.7%, within-one 0.50 -> 0.49) and therefore also in Asghar's all-family row (within-one
  0.62 -> 0.61, mean 23.2 -> 23.6%; the same instances).

### 10.2 Second registered held-out test (`heldout2_v3.py`), registered 2026-10-12, run 2026-10-12/13 (results 10.3)

It fixes every weakness the two reviews found in the first test.

| | test 1 | test 2 |
|---|---|---|
| fields | seeds 13-20, of which 13-15 previously swept | **seeds 21-40, in no file before** (seeds 42 and 123 avoided) |
| planners | SA, tour, C4 | SA, tour, C4, **cluster patrol** |
| hypothesis for C4 | none | **H5** |
| censoring check | top edge only | **both edges** |
| H1 | median (tied at 0 on clean seeds) | **p90 (nearest rank) and mean** |
| horizon | 4 discriminating instances | **19 discriminating + 26 controls** |
| registration | predictions hash | **manifest hash of predictions + evaluation code + rule, published before the run** |

**Design.**
- **Main arm:** 20 primary cells x seeds 21-28 (160 instances), T_mis = 12 h. v3.3 differs from the old rule in 19
  instances (17 core, 2 clustered).
- **Horizon arm:** SA at T_mis = 24 h on every core or clustered instance with seeds 21-40 where v3.3's 12 h and
  24 h predictions differ (19), plus 26 control instances where they do not (core M=100/200, seeds 21-28). The 12
  discriminating instances with seeds 29-40 also get a 12 h sweep (corrected from 13 after the third review).
- **Estimated size:** about 9,700 runs, roughly 8 h on the Ryzen.

**Registered hypotheses** (primary rule v3.3):

| | planner and set | criterion |
|---|---|---|
| H1 | SA, core | p90 and mean regret below the old rule's |
| H2 | SA, all | within-one not below the old rule's |
| H3 | SA, ring | bracket coverage >= 90% |
| H4 | SA, instances where v3.3 != old | better more often than worse, one-sided sign test p < .05 |
| H5 | C4, core | p90 regret below the old rule's, and within-one not below |
| H6 | tour patrol and C4, clustered, each | mean regret no more than 1 point above the old rule's |
| H7 | horizon, discriminating set | the 24 h optimum is closer to v3.3(24 h) than to v3.3(12 h) in >= 75% and binomial p < .05 |

**Reported without a verdict:** all rules (old, v3, v3.3) for all planners and families; the 2x2 switch table with
controls; medians; linear p90.

**Registration.**
- Manifest sha256 (LF-normalised): `9497b1f41e223b0a39c8573881110245c2b60532c5256b41405af9630c8c0b8e`.
- Component hashes:

  | file | sha256 |
  |---|---|
  | `heldout2_predictions.csv` | `3b916c30eac802ff10c0bb9c7814d8b564791243862efa1c679053d829bbbd08` |
  | `heldout2_v3.py` | `053c6d78bdcdebe491c497a1a2590991c7360ccdd2a92b633c7cb17266ae9b53` |
  | `theory_v3.py` | `4d566cf640297d6eb1a4990ecfb6a8ebc20a42e00005c0bb99e5108de99bb907` |

- The predictions regenerate byte for byte from a clean copy, with no tour cache.
- The machinery was tested on seed 1, which is not a test seed: 35 rows, two-sided extension rounds, 0 censored.
  The evaluation code was tested on synthetic data.
- The hash is to be published before the run (commit of a hash-only file to `sim-v2`). That gives the third-party
  timestamp the second review asked for.

### 10.3 Result of the second test

The Ryzen run produced 10,155 runs: all v2.2, no duplicates, no optimum at a sweep edge. Every number below was
recomputed independently from the raw CSV.

| | hypothesis | measured | verdict |
|---|---|---|---|
| H1 | SA core, p90 and mean regret | p90 102.5 -> 19.6%, mean 31.2 -> 4.6%, max 145.8 -> 31.9% | **holds** |
| H2 | SA within-one | 0.744 -> 0.812 (core 23/48 -> 34/48) | **holds** |
| H3 | SA ring bracket | 48/48 | **holds** |
| H4 | SA paired | v3.3 better in 18 of 19, worse in 1 | **holds** |
| H5 | C4 core | p90 39.2 -> 16.4%, but within-one 36/48 -> 35/48 | **fails** |
| H6 | clustered: tour +0.43, C4 +0.58 points (limit 1) | unmodified v3 would give tour +2.1 | **holds** |
| H7 | horizon | closer to v3.3(24 h) in 10 of 19 | **fails** |

- **H5.** v3.3 changes C4's within-one verdict in 9 core instances. It is right in 4 (M=200, where C4 strands) and
  wrong in 5 (M=100 1.5 MJ and M=200 3 MJ, where C4 does not strand). v3.3 cuts C4's tail regret (max 93 -> 24%) but
  does not predict its optimum better. This is the planner dependence of section 8, now confirmed on fresh data.
- **H7.** The direction holds and the switch point does not.
  - By set (corrected after the third review; the earlier breakdown "7 of 10 ... 0 of 6" did not add up to 10):

    | set | discriminating | 24 h optimum closer to v3.3(24 h) |
    |---|---|---|
    | core 1.5 MJ | 10 | 7 (12 h optima 13-15 dropped to 4-6) |
    | core 3 MJ | 7 | 1 (M=200 s=33: 30 -> 11); in the other 6 the 24 h optimum still strands at 30-31 |
    | clustered 1.5 MJ | 2 | 2 |
    | total | 19 | 10 |

  - The 3 MJ controls show the same bias at 12 h (optimum 30-31 where v3.3 predicted no stranding).
  - This matches the known level bias: the served term is about 2.4x short at 3 MJ against 1.35x at 1.5 MJ. That
    undervalues stranding, so the surrogate places the switch too early, most at 3 MJ.
  - Use the horizon dependence qualitatively only. Any pilot-rescaled analysis would be post hoc.
- 2x2 table (switch = optimum moves by >= 2): TP 8, FN 6, FP 11, TN 20.

## 11. v3.4: crowding-corrected hop (designed on seeds 1-40; third registered test passed, 11.2)

**Diagnosis** (`kappa_diag.py`, design seeds 1-3; `rate_split.py` on the full design grid; the design-time scripts are in the verification bundle's `design/` folder). The
surrogate's level error kappa = J_age / served term is written as a product of rate, allocation, regularity and a
residual. The product is a telescoping identity, exact by construction; its content is that the residual is about
1.00. It was computed on 4 core 3 MJ stranding runs only (M=100 seeds 1-3 at K = 28, 29, 28; M=200 seed 1 at K = 28).
Per-factor medians over those 4: 2.9 = 2.51 (rate) x 0.98 (allocation) x 1.17 (regularity) x 1.00 (residual); the
single run M=100 s=1, K=28 gives 2.93 = 2.53 x 0.96 x 1.21 x 1.00 (corrected after the third review: the earlier text
called the medians the M=100 K=28 value). k_alloc < 1 does not measure allocation inefficiency alone: A_R B_R/(2 Gamma)
is not the minimum of sum w/(2f) at fixed sum f (that is (sum sqrt w)^2/(2 Gamma)), so k_alloc mixes allocation with
the radial-profile approximation. `kappa_diag.py`'s docstring says the residual contains stranded age; it does not
(J_age there is served-only). The log-based age equals the simulator's J_age exactly. **The error is the a-priori
visit rate**, and within it the hop:
measured hop / d_S has median 1.05-1.22 in most cells and 1.66 at core 3 MJ. The productive-time part contributes
1.03-1.12. Evaluating the commute radius at each K changes almost nothing.

**v3.4** (`theory_v34.py`; `theory_v3.py` stays as registered).
- For planners whose UAVs share the field, the hop is d_S / sqrt(1 - x), with x = min(0.9, (K - 1) n / (2 |S|))
  the share of served sensors held by other airborne UAVs (each holding half its route on average, all K - 1
  assumed airborne). n is the a-priori stop count at that hop, n = max(1, (U/K - P_f t_c) / (P_bar tau_s)) with
  tau_s = (hop / v) / (1 - h), and h = min(0.95, H/Phi(K)) the hover share, computed once per K outside the loop.
- **As implemented and registered**, the hop is 20 plain iterations d <- f(d) from d = d_S. The map f is
  non-increasing, so a unique fixed point exists, but plain iteration falls into a 2-cycle where |f'| > 1
  (third review, H-2; confirmed here with an independent bisection):
  - at the registered pick, the hop is 0.68-0.76 of the true fixed point in 33 of 160 test-3 instances (all 24
    core 3 MJ, all 8 clustered M=100 6 MJ, 1 core M=50 1.5 MJ); 39 differ by more than 0.1%. The cycle alternates
    between the cap's branch d_S / sqrt(0.1) = 3.16 d_S and about 1.53 d_S, and iteration 20 lands on the low branch;
  - ring cells converge, so the 24 ring changes that decide hypotheses A and B are unaffected;
  - sensitivity of the 160 test-3 predictions: 19 or 21 iterations change 4 (all core 3 MJ, including both misses of
    11.2); 50 iterations 0; the true fixed point (cap 0.9) 1 (core M=100 s=47: 10 -> 28, one of the two misses);
    cap 0.8 changes 1, cap 0.95 2, no cap 4. Within-one moves by at most 0.006 and p90 by at most 2 points under
    every variant.
  - So "n(K) from a fixed point" was false in those cells, and the 0.9 cap is not an inactive safety bound: it sets
    the high branch of the cycle. `theory_v34.py` is registered and stays unchanged; a repaired solver would be a new
    rule needing its own test.
- **"No fitted constant"** means no constant was estimated from data. The 1/2, the all-airborne (K - 1), the 0.9 and
  0.95 caps, the n >= 1 floor and the iteration count are modelling choices made on seeds 1-40; two of them move
  predictions. The thinning law 1/sqrt(1 - x) is a mean-field assumption (claims are contiguous routes, not random
  thinning).
- For fixed-territory planners (C4, cluster patrol, Liu, Gao, Asghar), x = 0 and v3.4 = v3.3. The planner's
  coordination model is a declared input.

**Hop model check** (SA, design grid; `hop_check.py`). Model / measured hop has median 1.04-1.11 in five of the six
cells and 0.85 at core 3 MJ, where the hop is still underestimated (uniform hop: 0.60-0.95; corrected 2026-10-13,
the earlier text said 1.04-1.11 in every cell; the docstring of `theory_v34.py` still says so and is left unchanged
because the file is registered).
The trend with K is right only on the ring at 3 MJ: measured hop falls with K everywhere, while the model's falls
only there and rises on the core and clustered layouts, and on the ring at 1.5 MJ (+0.40). So v3.4 fixes the level, and the ring trend, but not the
general fall of hop with K.

**On seen data** (within one UAV, all instances; p90 nearest rank):

| | old | v3.3 | v3.4 | p90 v3.3 -> v3.4 | ring within-one v3.3 -> v3.4 |
|---|---|---|---|---|---|
| SA retrospective | 0.76 | 0.82 | **0.91** | 18.1 -> 11.1% | 0.68 -> 0.92 |
| SA held-out 1 | 0.81 | 0.88 | **0.94** | 16.7 -> 11.6% | 0.71 -> 0.94 |
| SA held-out 2 | 0.70 | 0.80 | **0.89** | 16.1 -> 9.5% | 0.67 -> 0.96 |
| tour retrospective | 0.79 | 0.83 | **0.90** | 19.4 -> 12.3% | 0.81 -> 0.97 |
| tour held-out 1 | 0.83 | 0.86 | **0.91** | 15.9 -> 11.3% | 0.81 -> 0.98 |
| tour held-out 2 | 0.79 | 0.84 | **0.92** | 14.7 -> 10.0% | 0.81 -> 1.00 |

- The retrospective rows exclude the paper M=100 6 MJ cell (12 instances per planner; script `v34_full.py`), so SA's
  retrospective n is 228, not 240. On that cell old, v3.3 and v3.4 agree and are within one of the optimum in 12 of
  12, so including it would raise each SA retrospective within-one by about 0.01 (old 0.77, v3.3 0.83, v3.4 0.92).
  The held-out 2 SA rows have n = 172: the 160 main instances plus the 12 h sweeps of 12 horizon-arm instances.
- Ring 3 MJ predictions move from 6 to 7, against measured optima of 8.5, 7.9 and 6.7.
- The territorial planners are unchanged by construction.
- Test 2's horizon arm (SA): within one of the 24 h optimum in 34 -> 36 of 45 (slight).
- Applying the crowding hop to C4 (wrongly, as a check) lowers its core within-one from 0.83 to 0.75, consistent with
  the mechanism (claims never thin a fixed territory).

**Status.** Every number above is on data used to design v3.4 (seeds 1-40); it is evidence only after the third test.

### 11.1 Third registered held-out test (`heldout3_v3.py`), registered 2026-10-13, run 2026-10-13 (results 11.2)

- **Seeds:** 41 and 43-49, fresh (42 avoided). 20 primary cells x 8 seeds = 160 instances, T_mis = 12 h.
- **Planners:** SA and the tour patrol, the shared-field planners; v3.4 = v3.3 for territorial planners by
  construction.
- **Sweeps:** two-sided.
- **Rules:** old, v3.3, v3.4. v3.4 differs from v3.3 in 28 instances (24 ring 3 MJ, 4 core).
- **Size:** about 4,300 runs.
- **Hypotheses**, each for SA and for the tour patrol, 8 verdicts:

  | | criterion |
  |---|---|
  | A | all 160: v3.4 within-one > v3.3's |
  | B | ring: v3.4 within-one >= 0.85 and > v3.3's |
  | C | all: v3.4 p90 (nearest rank) < v3.3's |
  | D | where v3.4 != v3.3, better more often than worse, one-sided sign test p < .05 |

- **Manifest sha256** `275e503a021376785f8b138ca6471e8b9e6857dca060789e7c50d1e812ae871e`, over these four files
  (`theory_v3.py` is unchanged since test 2):

  | file | sha256 |
  |---|---|
  | `heldout3_predictions.csv` | `e7599fe7...` |
  | `heldout3_v3.py` | `f9905453...` |
  | `theory_v34.py` | `f24705f6...` |
  | `theory_v3.py` | `4d566cf6...` |

- **Checks:** predictions regenerate byte for byte from a clean copy; the machinery was smoke-tested on seed 1.
- **Hash committed** to the public repository: commit 93a3f6e on `sim-v2` (`registrations/heldout3_manifest.sha256`,
  the only file in that commit), author-dated 2026-10-13 05:37 UTC; the run started immediately after. The push
  time is to be confirmed from GitHub's push activity (third review, M-6). The hashed files are not yet public.

### 11.2 Result of the third test: all 8 verdicts hold

The Ryzen run produced 4,471 runs (4,266 in round 0, 205 in extension rounds). All are v2.2, seeds 41 and 43-49
only, with no duplicates and no non-finite J. All 320 sweeps are contiguous, cover [K_lo, K_hi] and end with an
interior optimum; every predicted K was swept. The verdicts, the family rows and the comparison with the paper rule
were recomputed independently from the raw CSV (`verify_heldout3.py`, pandas only, no code from `heldout3_v3.py`)
and by the third review. The post hoc analysis of the remaining errors was not recomputed independently at first;
its strand / serve-all labels were reversed until the third review caught it (corrected below).

| | | SA | tour patrol |
|---|---|---|---|
| A | within-one, all 160: v3.3 -> v3.4 | 0.856 -> **0.925** | 0.831 -> **0.900** |
| B | within-one, ring 48: v3.3 -> v3.4 | 0.729 -> **0.938** | 0.792 -> **1.000** |
| C | p90 regret, all: v3.3 -> v3.4 | 19.3 -> **10.6%** | 16.7 -> **9.1%** |
| D | where v3.4 != v3.3 (28): better / worse, p | **28 / 0**, p < 1e-8 | **22 / 6**, p = .0019 |

All 8 hold.

**D per cell** (third review, E1). 24 of the 28 changed instances are three ring 3 MJ cells, each moved from 6 to 7;
the design data had already put those cells' optima at 8.5, 7.9 and 6.7.
- SA: 8/0, 8/0, 8/0 on the ring cells, 4/0 on core (fields). The 28 changes lie in **6 cells** (ring 50/100/200 at 3 MJ;
  core M=100 3 MJ, core M=200 1.5 MJ, core M=200 3 MJ). With cells as units (sign of the mean regret change),
  **6 of 6 better, p = .016** (corrected 18 Oct after review 5, M3: the earlier "7 of 7, p = .008" counted the three
  ring cells plus the four core *fields*). Without the ring 3 MJ cells, 4/0 fields, p = .06.
- Tour: 8/0, 8/0, **3/5** (M=200), 3/1 on core (fields). By cell, **4 of 6 better, p = .34** (worse in ring M=200 3 MJ
  and core M=200 1.5 MJ, 0/1); the earlier "5 of 7, p = .23" mixed cells and fields. Without the ring 3 MJ cells,
  3/1 fields, p = .31.
- So D is a replication on new fields of the same conditions, mainly ring 3 MJ, not evidence of a broad improvement.

**Against the v2.2 paper rule (old)** (pre-specified, reported without a registered verdict), on all 160: SA
within-one 0.79 -> 0.93, p90 29.7 -> 10.6%, max 141 -> 30.1%; tour 0.79 -> 0.90, p90 28.5 -> 9.1%, max 109 -> 25.4%.

By family (within-one, v3.4; p90 regret v3.3 -> v3.4):
- **Paper:** SA 1.00, tour 0.95; v3.4 = v3.3 here (no crowding change).
- **Ring:** SA 0.94, tour 1.00; p90 21.5 -> 11.5% (SA), 24.3 -> 12.1% (tour). The 24 ring 3 MJ instances move from
  6 to 7: SA better in 24 of 24; tour better in 19, worse in 5 (where the tour's optimum is 6, regret 2-14%).
- **Core:** SA 0.81, tour 0.73; p90 25.9 -> 12.5% (SA), 18.8 -> 13.7% (tour). This is where the remaining error is.

**What is still wrong (post hoc, on this test's data; to be reported, not fixed on it).**
1. **Core 3 MJ (24 instances per planner).** Here the far outlier caps the fleet near floor Kr, so K = 9-12 serves
   every sensor and K = 28-31 strands the outlier (corrected after the third review, H-1: the first version of this
   item had the two labels swapped, and with them the explanation). The optimum serves all at K = 9-12 in 15
   instances (n_never = 0) and strands at K = 30-31 (tour 29-31) in 9 (n_never = 1). v3.4 makes the right
   strand-or-serve choice in 22 of 24 for both planners; every one of the 15 serve-all instances is within one.
   - In the 2 wrong ones (core M=50 s=45, M=100 s=47), v3.4 serves all (K = 11, 10) where the optimum strands
     (SA 31, 31; tour 30, 31): SA regret 15.5% and 30.1%, tour 5.4% and 18.7%. This is under-stranding, the same
     direction as the level bias of 10.3 (H7): the undervalued served term makes stranding look less attractive than
     it is. The second of these is also the instance the true fixed point would change to 28 (section 11).
   - Where v3.4 strands (7 instances), it predicts 28 against SA optima of 30-31: 0 of 7 within one, regret
     0.7-2.7% (flat optimum). Tour: 5 of 7 within one, regret up to 7.0%.
   - SA's core within-one (0.81) is therefore 1.00 at 1.5 MJ and 0.62 at 3 MJ.
2. **Tour patrol, core M=200 1.5 MJ.** v3.4 predicts 14 against optima of 10-12 in 8 of 8 seeds (regret 1-9%), so
   the tour's core within-one at 1.5 MJ is 0.62. SA on the same fields is within one. v3.3 already predicted 14 in 7
   of 8: this is a v3.3 miss that v3.4 does not fix. Attributing it to the hop trend is a conjecture.
3. **Reach-side misses by one UAV** (third review, M-1). SA's five largest v3.4 regrets are 30.1, 29.7, 19.5, 18.4 and
   15.5%. The 29.7% case (also the tour's worst, 25.4%) is core M=50 1.5 MJ s=43: old, v3.3 and v3.4 all predict 4,
   the optimum is 5. A miss by one on the steep left flank of J(K) counts as within one but can cost up to 30%. The
   same mode appeared in test 1's H5 arm (section 7).

**Cross-machine spot check.** 11 rows were rerun on Linux (9 SA, 2 tour). Ten match the Ryzen exactly; one SA row
(ring M=50 3 MJ seed 46 K=7) differs by 1.4%, and Linux reproduces its own value on a second run. This is the known
platform effect (HANDOFF pitfalls: SA's np.exp differs in the last bit across C runtimes, so a search can take another
path). The test ran entirely on the Ryzen, so no verdict is affected.

**Status.** v3.4, as implemented, has passed one registered test for SA and the tour patrol, with its gain
concentrated in the ring 3 MJ cells. The sweep patrol and the greedy planner are in `SHARED` in the code but have no
data. For territorial planners it is v3.3 by construction and has no new test.

## 12. What the paper should say now (supersedes sections 8 and 9 where they differ; revised after the third review)

**Naming.** "H5" names two hypotheses. In the paper and below: **T1-H5** is test 1's 24 h horizon test (section 7);
**T2-H5** is test 2's C4 hypothesis (10.3). Likewise T2-H7 and T3-A to T3-D.

**After test 4 (13.1), the headline deliverable is the rule plus the local check**: about 4 simulated fleet sizes
from the rule's two Theorem-1 candidates, registered and holding for SA, the tour patrol and C4 (worst 6-9%), with
the cluster patrol's mean just over its threshold. For the territorial planners, v3.3 alone is not recommended over
the paper rule (T2-H5, T4-R4); it serves them as the source of the two candidates.

**Rule.** The paper's rule is **v3.3**: Theorem 1's candidate rule with gray-bracket exclusion, hop d_S.
**v3.4** is presented as a registered, tested extension for the annealer and the tour patrol only: the crowding hop,
defined **as implemented** (20 iterations from d_S, section 11), with the gap to the true fixed point disclosed (1 of
160 test-3 predictions differs) and its sensitivity table. Its gain is concentrated in the commute-bound ring at 3 MJ
(24 of its 28 changed predictions), the cells where App. B already explains the miss by the falling hop. The sweep
patrol and the greedy planner are untested and get no v3.4 claim.

Why not v3.4 as "the rule for shared-field planners": it has one test; its derivation as a fixed point does not
match its code; and the shared / territorial split was settled on seen data (the C4 check in section 11).

**Evidence, in the order a reader should weigh it.**
1. Test 3 (registered, seeds never used): T3-A to T3-D hold for SA and the tour patrol. For v3.4 this is the **only**
   test of the rule as stated; tests 1 and 2 tested v3 and v3.3, and their v3.4 rows (section 11) are design
   evidence. T3-D is reported per cell beside the registered verdict (11.2): with cells as units, SA 6/6 (p = .016),
   tour 4/6 (p = .34) (corrected; see 11.2).
2. Test 2 (registered, seeds never used): v3.3 holds H1-H4 and H6 for the annealer and the clustered limit; it fails
   T2-H5 (C4) and T2-H7 (mission-length switch) (10.3).
3. Test 1 (registered, hash in the project log only; 54 of 160 instances on previously swept pilot fields) and T1-H5
   (4 of 4, p = .06) (sections 6-7).
4. Retrospective and seen-data tables (sections 5, 10.1, 11): design evidence only.
- The comparisons with the paper rule in tests 2 and 3 beyond the hypotheses are "pre-specified, no registered
  verdict" and are labelled so.

**Limits to state.**
1. **Territorial planners: no positive point-prediction evidence.** In test 2, v3.3 against the paper rule, within one
   UAV: C4 0.881 -> 0.875 (core 36 -> 35, T2-H5 fails); cluster patrol 0.844 -> 0.831 (core 34 -> 33, clustered
   57 -> 56). Both tails fall (C4 max 93 -> 24%).
2. **Gao et al.** v3.3 is worse than the paper rule for Gao (retrospective core within-one 0.86 -> 0.71, p90
   27.8 -> 63.2%, max 71 -> 191%; all 216: 0.76 -> 0.71). Gao's fixed territories never exploit stranding. The paper
   either excludes Gao from the v3.3 recommendation or states this.
3. **Mission length.** The stranding branch switches off as T_mis grows, in the right direction (T1-H5), but the
   switch point is not predicted reliably (T2-H7: 10 of 19; core 3 MJ 1 of 7). Qualitative only. Section 9's wording
   ("H5 as the evidence that the stranding branch tracks the mission length") is retired.
4. **Core 3 MJ under-stranding.** The served term is low by about 2.4x there, which makes stranding look less
   attractive than it is. v3.4 makes the right strand-or-serve choice in 22 of 24 test-3 instances per planner; the 2
   misses serve all where the optimum strands (SA 15.5%, 30.1%). Where it strands, the prediction (K = 28) sits 2-3
   UAVs below the optimum (SA regret <= 2.7%).
5. **Reach-side misses by one UAV** can cost up to 30% (test 3: SA 29.7%, tour 25.4%; old, v3.3 and v3.4 alike), and
   within-one does not register them.
6. **Hop trend.** The crowding hop fixes the hop's level in five of six cells (0.85 at core 3 MJ), and its trend only
   on the ring at 3 MJ.
7. **Clustered fields.** Stranding there does not pay; v3.3's gray-bracket exclusion removes most of those
   recommendations (10.1), and the clustered limit (H6) holds.
8. **Tail, per planner the rule is recommended for** (worst regret on held-out data). The old criterion's "no passing
   instance loses more than 27%" is gone.

   | rule | planner | worst regret |
   |---|---|---|
   | v3.3 | SA | 38.7% (test 3), 31.9% (test 2) |
   | v3.3 | tour patrol | 31.1% (test 3), 33.5% (test 2) |
   | v3.3 | C4 | 24.3% (test 2) |
   | v3.3 | cluster patrol | 46.5% (test 2; core M=100 1.5 MJ s=25, predicts 14, optimum 6) |
   | v3.4 | SA | 30.1% (test 3) |
   | v3.4 | tour patrol | 25.4% (test 3) |

   Operators who cannot accept that should sweep.
9. **SA reproducibility.** SA results are reproducible per machine, not across machines (11.2); every test ran on one
   machine. Optima in flat regions (ring 3 MJ, core 28-31) can shift by one on another machine.

**Registration wording.**
- Tests 2 and 3, if GitHub's push activity confirms that abac195 and 93a3f6e each reached GitHub before their run
  started: "Predictions, evaluation code and rule were fixed by a SHA-256 manifest committed alone to the public
  repository before each run (commits abac195, 93a3f6e); the hashed files are released with the paper."
- If it does not: "... committed to the public repository (author-dated before each run)".
- Either way, the hashed files (`theory_v3.py`, `theory_v34.py`, `heldout2_v3.py`, `heldout3_v3.py` and the two
  predictions files) must be public when the paper is, or no reader can check the hashes.
- Test 1 keeps "hashes recorded in the project log; no third-party timestamp".

**Paper changes (section 9 applies, with these differences).**
- The ring keeps K_c as the rule's point prediction; the v3.4 extension gives the shared-field point prediction there
  (test-3 ring within-one 0.94 SA, 1.00 tour). Prop. 2's bracket stays for every planner.
- The held-out section reports tests 2 and 3 as the primary evidence, and test 1 with T1-H5 as earlier evidence with
  its disclosures.
- p90 is nearest rank throughout (tests 2 and 3 use it; paper Table VII's `method='higher'` equals nearest rank at
  all its n, none of which is a multiple of 10).
- The v3.4 paragraph needs: the n(K) equation (section 11), the thinning law stated as a mean-field assumption, the
  caps and the iteration count stated as modelling choices, and the fixed-point disclosure.

## 13. The two-candidate local check and the fourth registered test (registered 2026-10-13, run 2026-10-14; results 13.1)

**Why.** The rule alone has a typical regret of a few percent but a worst case near 30% (12, limit 8). A full sweep
is always at least as good, but needs a faithful simulator of the field (about 14 runs per field in test 3). The
local check sits in between and needs about 4 runs.

**Procedure** (`localcheck.py`; uses the registered rules unchanged).
1. c1 = the rule's fleet size: v3.4 for the annealer and the tour patrol, v3.3 for C4 and the cluster patrol.
2. c2 = the surrogate's best fleet size at an adjacent stranding level (one sensor more or fewer stranded than at
   c1): Theorem 1's neighbouring served set. None if no adjacent level is feasible.
3. Simulate or fly {c - 1, c, c + 1} for c in {c1, c2} (K >= 1); keep the best measured size.

It was found post hoc on test 3, then evaluated on all seen data (seeds 1-49) before test 4 was registered. The
first version took the best size at *any* other stranding level; on clustered fields that is sometimes K ~ 60-120,
stranding dozens of sensors, which no operator would fly. Restricting to the adjacent level changed no regret on any
seen set and cut the checked sizes from 4.4 to 4.0 on average.

**On seen data** (design evidence only; regret of the check, the rule alone in brackets):

| set | planner | sizes | exact | mean | worst | > 10% |
|---|---|---|---|---|---|---|
| retro (240/216) | SA | 4.0 | 0.91 (0.62) | 0.15% (2.6%) | 6.6% (26.4%) | 0 |
| | tour | 4.1 | 0.92 (0.54) | 0.30% (3.8%) | 8.9% (39.3%) | 0 |
| | C4 | 4.1 | 0.92 (0.52) | 0.44% (4.2%) | 13.6% (42.6%) | 1 |
| | cluster patrol | 4.1 | 0.88 (0.50) | 0.77% (7.1%) | 18.4% (79.1%) | 6 |
| test 1 (160) | SA | 4.0 | 0.95 | 0.07% | 3.2% | 0 |
| | tour | 4.0 | 0.92 | 0.15% | 4.5% | 0 |
| | C4 | 4.0 | 0.90 | 0.53% | 14.5% | 1 |
| test 2 (172/160) | SA | 4.2 | 0.89 | 0.15% | 9.7% | 0 |
| | tour | 4.1 | 0.93 | 0.26% | 8.8% | 0 |
| | C4 | 4.1 | 0.91 | 0.45% | 16.7% | 1 |
| | cluster patrol | 4.1 | 0.86 | 0.75% | 14.1% | 3 |
| test 3 (160) | SA | 4.0 | 0.93 (0.64) | 0.15% (2.6%) | 7.3% (30.1%) | 0 |
| | tour | 4.0 | 0.90 (0.54) | 0.29% (3.3%) | 5.9% (25.4%) | 0 |

Checking only the rule's size +-1 (3 sizes) leaves test 3's worst case at 30.1% (SA): the second candidate is what
catches the strand-or-serve misses.

**Test 4** (`heldout4_v3.py`).
- **Seeds 50-57**, in no CSV of the repository (sim-v2 at 230c5ff) or of the results archive. 20 primary cells x 8
  seeds = 160 instances, T_mis = 12 h, planners SA, tour patrol, C4, cluster patrol. Two-sided sweeps that cover
  every checked size. About 8,500 runs, roughly 5 h on the Ryzen.
- **Registered hypotheses** (24 verdicts):

  | | planners | criterion |
  |---|---|---|
  | L1 | each of the 4 | mean check regret < 1.0% |
  | L2 | each | check finds the swept optimum exactly in >= 85% (SA, tour), >= 80% (C4, cluster patrol) |
  | L3 | each | check regret > 10% in at most 3 of 160 (SA, tour), at most 8 (C4, cluster patrol) |
  | L4 | each | worst check regret <= 15% (SA, tour), <= 25% (C4, cluster patrol) |
  | R1 | SA, tour | v3.4 within-one, all 160 >= 0.85 (replication of test 3) |
  | R2 | SA, tour | v3.4 within-one, ring 48 >= 0.85 |
  | R3 | SA, tour | v3.4 p90 regret < the paper rule's |
  | R4 | C4, cluster patrol | v3.3 worst regret < the paper rule's |

  Thresholds were set from the seen-data table above; every seen set meets them.
- **Reported without a verdict:** each rule alone and with a one-candidate +-1 check, by family; sizes per instance.
- **Manifest sha256** `4c09326fa0517ed3438cf7cc457f0864eb81ab9cb6158336d2c293775f30b0aa` over:

  | file | sha256 |
  |---|---|
  | `heldout4_predictions.csv` | `0758e051...` |
  | `heldout4_v3.py` | `34676903...` |
  | `localcheck.py` | `83f946f1...` |
  | `theory_v34.py` | `f24705f6...` (unchanged since test 3) |
  | `theory_v3.py` | `4d566cf6...` (unchanged since test 2) |

- **Checks:** the evaluation code, pointed at test 3's data with test 3's seeds, reproduces the seen-data row above
  exactly; the machinery was smoke-tested on seed 1 (not a test seed), including a wrong-hash refusal.
- **To publish before the run:** a hash-only commit `registrations/heldout4_manifest.sha256` on `sim-v2`, and this
  time a note of the push time and the run start (third review, M-6).

### 13.1 Result of the fourth test: 21 of 24 verdicts hold

**Registration.** Hash-only commit 48c72ce pushed (console output) by 2026-10-14 12:54:02 IST; run started
12:54:09 IST (`Get-Date` before and after, user's console). The CSV has 8,848 runs (8,480 in round 0): all v2.2,
seeds 50-57 only, default settings, no duplicates, no non-finite J; all 640 sweeps contiguous, covering
[K_lo, K_hi], with an interior optimum; every checked size swept. Every number below was recomputed independently
from the raw CSV (`verify_heldout4.py`, pandas only, no code from `heldout4_v3.py` or `localcheck.py`) and agrees
with the registered evaluation to the printed digit.

| | | SA | tour patrol | C4 | cluster patrol |
|---|---|---|---|---|---|
| L1 | mean check regret < 1% | 0.19% **holds** | 0.32% **holds** | 0.20% **holds** | 1.006% **fails** |
| L2 | exact (>= 0.85 / 0.80) | 0.919 **holds** | 0.912 **holds** | 0.944 **holds** | 0.875 **holds** |
| L3 | > 10% (<= 3 / 8 of 160) | 0 **holds** | 0 **holds** | 0 **holds** | 7 **holds** |
| L4 | worst (<= 15% / 25%) | 8.7% **holds** | 6.7% **holds** | 6.2% **holds** | 23.3% **holds** |
| R1 | v3.4 within-one >= 0.85 | 0.881 **holds** | 0.869 **holds** | | |
| R2 | v3.4 ring within-one >= 0.85 | 0.938 **holds** | 1.000 **holds** | | |
| R3 | v3.4 p90 < paper rule's | 10.2 vs 27.4% **holds** | 12.0 vs 26.5% **holds** | | |
| R4 | v3.3 worst < paper rule's | | | 37.8 vs 27.6% **fails** | 90.3 vs 90.3% **fails** (tie) |

Checked sizes per field: 4.05 on average (3.0 ring, 3.4 clustered, 6.0 core).

**Rule alone against the check** (all 160; rule = v3.4 for SA and tour, v3.3 for C4 and cluster patrol):

| planner | rule alone: mean / worst | local check: mean / worst | check exact |
|---|---|---|---|
| SA | 2.9% / 49.0% | 0.19% / 8.7% | 0.92 |
| tour patrol | 3.9% / 38.4% | 0.32% / 6.7% | 0.91 |
| C4 | 4.9% / 37.8% | 0.20% / 6.2% | 0.94 |
| cluster patrol | 7.5% / 90.3% | 1.01% / 23.3% | 0.88 |

**What the failures say.**
- **L1, cluster patrol (1.006% against < 1%).** All 7 cases above 10% are clustered or ring fields where the cluster
  patrol's optimum lies 2-3 UAVs below the rule's size (e.g. optimum 7 against rule 9-10), out of reach of c +- 1,
  and in 5 of them there is no second candidate. The rule over-sizes for this planner on clustered fields; the check
  cannot correct a bias of more than one UAV. Without its single worst field the mean would be 0.87%.
- **R4, C4 (v3.3 worst 37.8% against the paper rule's 27.6%).** On the core at 1.5 MJ v3.3 strands one outlier and
  jumps to 14 UAVs; C4's optimum stays at 5-6 in most of those fields. Where v3.3 differs from the paper rule, C4 is
  better in 4 and worse in 11 (SA: 14 better, 1 worse; tour 14/1; cluster patrol 8/7). With T2-H5 this is now
  consistent on two fresh sets: **for C4, v3.3 alone is a worse point predictor than the paper rule on the core.**
  The local check repairs it completely (C4 check worst 6.2%), because its second candidate is the paper rule's size.
- **R4, cluster patrol.** A tie: the worst field (clustered M=50 1.5 MJ s=54: optimum 4, both rules 5, 90.3%) has the
  same prediction under both rules; the check finds the optimum there. Registered as strict, so it fails.

**Replication.** v3.4's test-3 result replicates on fresh fields (R1-R3). Where v3.4 differs from v3.3: SA ring 3 MJ
24/0, tour 17/7; core 4/2 (SA), 4/2 (tour). SA's v3.4 worst case is 49.0% (core M=50 3 MJ s=52: predicts 9, optimum
32, under-stranding as in 11.2); the check brings it to 3.9%.

**Status.** The local check is the strongest result of the theory: registered on fresh fields, it holds every
criterion for SA, the tour patrol and C4 (worst 6-9%, mean 0.2-0.3%, about 4 simulated fleet sizes per field) and
three of four for the cluster patrol. For the territorial planners v3.3 *alone* should not be recommended over the
paper rule; v3.3 is useful to them only as the source of the check's two candidates.


## 14. Reviews 1-5 (18 Oct): corrections to this file

**Test 4 and the crowding-hop solver (review 2, H-2; verified with my own fixed-point code).** The registered v3.4
uses 20 plain iterations, which do not reach the fixed point in core 3 MJ, core M=50 1.5 MJ and clustered M=100 6 MJ
(smaller shortfalls on clustered M=50 3 MJ and core M=100 1.5 MJ). The true fixed point changes 1 of 160 test-3
predictions and **7 of 160 test-4 predictions** (core M=50 1.5 MJ s54: 14 -> 6; six core 3 MJ fields: 9-12 -> 28);
19 or 21 iterations change 4 and 15. At the fixed point the rule-alone figures of test 4 would be: SA max 49.0 ->
23.1%, p90 10.2 -> 7.1%; tour max 38.4 -> 26.5%, p90 12.0 -> 10.1%. The two worst fields (core M=50 3 MJ s52) are
predicted within 6.2% (SA) and 5.3% (tour). So the 49% worst case in section 13 is a solver artefact as much as an
undervalued served term. The local check is unchanged at the fixed point and at 19 iterations; at 21 iterations
one field (core M=200 3 MJ s50: candidates 28 and 7 instead of 11 and 28, optimum 12) gives worst check regret 21.5%
(SA) and 18.6% (tour), which would fail L4. Registered results stand as run; the paper now says all of this.

**Theorem 1 and v3.4 (review 2, H-1).** Theorem 1 needs a hop that depends on the served set only. With the
crowding hop the minimiser need not be a candidate: ring 3 MJ picks 7 against C_0 = l = 6 (24 of 160 fields in
tests 3 and 4). Result 2 is now defined as a search over admissible integers K <= 200; for v3/v3.3 that equals the
best Theorem-1 candidate, and C_0 = l, in 924/924 instances.

**Corollary 1 (review 2, M-1).** With the exact Phi, one v3.4 prediction changes (test 4, paper M=200 1.5 MJ s51,
4 -> 5, a 0.06% near-tie whose local-check set is unchanged), not none.

**Registered verdicts overall (review 5, H3).** 38 of 43 held: T2-H5 (C4 within-one), T2-H7 (horizon), T4-R4 (C4
and cluster patrol), T4-L1 (cluster patrol) failed. On test 1's unseen seeds 16-20 alone, T1-H1 ties (median 0 vs 0).
