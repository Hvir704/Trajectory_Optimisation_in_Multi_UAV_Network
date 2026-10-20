# DRL-UCS(AoI_th) attempt: results and logs (Sec. VII-L, "A learned baseline, not reproduced")

Ran in WSL (Ubuntu 24.04) on the authors' port of the released code, in the time-slotted world (`dyn_slot_env.py`,
slot 5 s), before simulator v2.2. Model checkpoints and TensorBoard event files (2.2 GB) are not included.

| run (logs/EnvDyn-v0/...) | radius | evaluation | sensors served of 50 | J |
|---|---|---|---|---|
| paper M=50 1.5 MJ s1 K=2 (10^7 steps) | 50 m (registered) | deterministic | 0 | 8.82e6 |
| s1 K=3 | 50 m | deterministic | 0 | 8.83e6 |
| s1 K=4 | 50 m | deterministic / sampled | 1 / 5 | 8.65e6 |
| s1 K=5 | 50 m | training not finished | - | - |
| s13 K=4 (pilot seed, 3e6 steps) | 200 m (after the 50 m failure) | deterministic / sampled | 2 / 14 | 8.88e6 / 7.11e6 |
| s13 K=4 | 500 m | training only | - | - |

`drl_results.csv` holds the three completed 50 m evaluations (`n_never` = sensors never served). On the same
field (seed 13, K = 3 and 5) the cluster patrol's J in the slotted world is 3.47e5 and 5.18e5
(`registrations/baselines/freeze_2026-10-05/evidence/gate_pilot_v2.csv`), so the 200 m policy's J is 14-26 times
the patrol's. The Beijing reproduction (EnvUCS-v0) reached a weighted threshold AoI of 52.5 at 10^7 steps
(`results/bj_curve.txt`) against the reported 26.8.

The registered scope (6 deployments, K windows of 5-6 sizes each) was not completed: the first deployment stopped
after K = 2-4, so by the registered rule (an unfinished deployment is excluded whole) the C4 comparison has no data.
