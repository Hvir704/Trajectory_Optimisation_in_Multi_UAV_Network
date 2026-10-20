# Baseline freeze — 5 Oct 2026 (registered before any held-out baseline run)

## 1. Pilot decisions (rules registered in GZZ_REGISTRATION.md / ASGHAR_REGISTRATION.md)
| baseline | deployed mode | pilot evidence (54 deployments, seeds 13-15, paired own optima) |
|---|---|---|
| Gao, Zhu, Zhai, TWC 22(8) 2023 | `gzz_rev` | median J*_rev/J*_full - 1 = -19.8% (core -27.7, paper -17.7, ring -19.3); rev lower in 52/54 |
| Asghar, Sundaram, Smith, T-RO 41 2025 | `asghar_rev` | -9.6% (core -10.7, paper -9.4, ring -11.2); rev lower in 44/54 |
Pilot code = delivered code (sha256 below). No truncation problem (max trunc_frac 0 / 0.024).
Pilot optima at the BOTTOM of the auto K range: gzz_rev 4, asghar_rev 8 deployments -> held-out runs use
`fill_tail.py --union` with the cluster/C4/SA grids (which start at K = 1-2) so no optimum is censored below.
Descriptive, pilot only (not a claim): vs cluster patrol, own optima: gzz_rev +28.5%, asghar_rev +124% (0/54 wins),
C4 -11.5%. Expected mechanisms: both plan offline with a nominal dwell (gzz: k-means + ACO territories; Asghar:
robots allocated per weight class, not per load).

## 2. Defects found after the handover, fixed (both in the slotted world only)
1. `slot_gate.py` scripted pilot: a UAV arriving exactly at a slot boundary was judged unable to drain and sent
   home, abandoning its route. Effect: ring M=100 3 MJ K=6 seed 15, cluster patrol, +418% J, 10 sensors never
   visited. Fix: abort only if the UAV was on its target at the START of the slot. Affects the gate and the
   slotted comparators, never the DRL environment.
2. `dyn_slot_env.py`: a straight flight to a sensor lying exactly on the field edge could overshoot by ~1e-12 m
   and be rejected as out of field. Fix: clip overshoots < 1 mm. Affects the DRL environment too (vendored
   copy updated by update_2026-10-05.patch). No completed run depends on it (throughput pilots only).
Gate rerun (pilot seeds 13-15, slot 5 s, fixed code): never-visited counts now equal DynSim's everywhere; slotting
cost median +1..+8% J (cluster patrol +0.8..+7.9%, greedy +3.2..+7.2%, rr_tour +1.3..+3.7% except ring +14.9%),
max +16.7%. Supersedes gate_pilot.csv.

## 3. Beijing reproduction (pipeline check) — verdict: partial
Metric mapping, verified from `env_ucs.summary_info`: kappa (threshold AoI) = f_weighted_bar_aoi, omega =
e_weighted_aoi, chi = b_emergency_violation_ratio, zeta = a_poi_collect_ratio, xi = h_energy_consuming_ratio (all AoI
in timeslots). Consistency check: kappa - omega ~ 75 chi in the paper's Table IV row, ~79 chi in ours.
| | paper (Table IV, U=3) | ours (1e7 steps) |
|---|---|---|
| kappa | 26.84 | 52.48 |
| omega | 25.95 | 42.81 |
| chi | 0.012 | 0.122 |
| zeta | 0.969 | 0.784 |
| xi | 0.781 | 0.791 |
Budget: 1e7 env steps vs the released 1e8; the paper's Fig. 14 reports convergence at ~41.6k episodes x 240 =
~1e7 steps, so the budget does not obviously explain the 2x gap. Evaluation: the evaluation actor is greedy and
the Beijing world is fixed, so the 30 "episodes" are one deterministic rollout (n = 1 effective); the paper
averaged 50 runs. The training-log REWARD includes the intrinsic reward and is not an extrinsic learning curve.
Also disclose: released code fails under its own strict FP setting with float16 UAV positions (fixed to float64,
same equations); the 'BJ' map-scale branch never fires for dataset "beijing".
Optional cheap check (section 6 of COMMANDS): kappa at every saved 0.5M checkpoint.

## 4. DRL scope (frozen)
Deployments x K windows (comparator own optima cluster/C4/SA and the rule's ell, -2/+2, K >= 1), in this order:
| order | deployment (M=50, 1.5 MJ) | regime | K window | comparator K* (clu/C4/SA), ell |
|---|---|---|---|---|
| 1 | paper s1 | reach | 2-7 | 5/4/5, 5 |
| 2 | ring s1 | commute | 1-5 | 2/2/3, 3 |
| 3 | paper s2 | reach | 1-6 | 3/4/4, 4 |
| 4 | ring s2 | commute | 1-6 | 3/3/4, 3 |
| 5 | paper s3 | reach | 2-7 | 4/5/5, 5 |
| 6 | ring s3 | commute | 1-5 | 3/3/3, 3 |
34 runs; 1e7 env steps each; 11 workers x 8 envs, learner 0.5 GPU, workers 0.04 GPU; the authors' released
hyperparameters; slot 5 s; collection radius 50 m (registered default; not piloted); eps = 0; training world =
positions + priorities + nominal rates + fresh events per episode; evaluation = final checkpoint, true world, one
deterministic episode (greedy actor, deterministic world), refused unless the run reports z_eval_world = 1.
Extension: after a window, add K+1 while DRL's argmin is the last K or J(K_last) < J(K_last-1), at most +4.
Cut-off: no deployment starts after 2026-10-16 08:00; an unfinished deployment is excluded whole.
Estimated 3.5-6 h per run (718 steps/s measured at K=3), about 6 days.
K = 1 checked: single-agent training runs in their framework.

Primary criterion: median over completed deployments of J*_C4 / J*_DRL - 1, both in the slotted world (own optima
on common K), percentile bootstrap, wins. Secondary: DRL vs cluster patrol, gzz_rev, asghar_rev, rr_tour in the
same world; DRL K* vs ell (exact / within one; ell is the DynSim rule). Report as found, no retraining.

## 5. Held-out classical baselines
gzz_rev and asghar_rev on seeds 1-12, primary grid (paper/ring/core x M 50/100/200 x 1.5/3 MJ = 216 deployments),
auto K range + `fill_tail.py --union` (cluster, C4, SA grids). Primary: per family, J*_C4 / J*_X - 1 (bootstrap,
wins). No directional claim vs cluster/SA. Secondary: rule agreement (Table IX format), same-K at ell.
Statistics: `baseline_stats.py` (validated: reproduces C4 vs cluster -21.0/-10.6/-6.6/-11.4%, 204/216 wins,
agreement 0.54/0.88 vs 0.49/0.84, same-K -12.4% exactly).

## 6. Code (sha256)
```
64db235894cc68631d9ee46c95f2d432b09ee4adf8f282e4ba62989af9149f9a  gzz_planner.py      (= pilot)
fa534d615eebcf1061a1df6f8012518cf57324b820f5329a00626cea249395a6  asghar_planner.py   (= pilot)
87097122dba4a6ecdb6bb4e699705698fb1959f2af1be972b98d7c95d351fba5  baseline_stats.py
223c0aa448962edbf80c6b85e6a5e3a6f4a4efe5e07c07863676bf4ca2028b50  dyn_slot_env.py
061ce60d07176cffc27092c376dc50a83fee401a2ceb4dce207f86acdae759d0  slot_gate.py
1cce07a58fb76ef895db9fda6b8a3aa157dc6297ab8ec0c48ba4e3953f5472f1  drl_scope.py
1fe5d9c58e049d490f927e7f8539cada06aa5a2f334cbe0b069798c1efa5223c  drl_stats.py
973bcf63b26a91a0706049d2183944b30c78302ab215d58b9b21ed1148c0b6a2  make_drl_configs.py
71aa1e4e047d8905d34e7f5c7ea6a936f0727f7f9b1c94c1b40556b4602ed9ed  update_2026-10-05.patch
```
