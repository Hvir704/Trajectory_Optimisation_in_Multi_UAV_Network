# DRL-UCS(AoI_th) baseline — registration (draft; scope fixed after the throughput pilot, before any held-out run)

Baseline: H. Wang, C. H. Liu, H. Yang, G. Wang, K. K. Leung, "Ensuring Threshold AoI for UAV-Assisted Mobile
Crowdsensing by Multi-Agent Deep Reinforcement Learning With Transformer", IEEE/ACM Trans. Netw. 32(1):566-581, 2024,
doi 10.1109/TNET.2023.3289172. Authors' code: github.com/BIT-MCS/DRL-UCS-AoI-Threshold (algorithm, network, IMPALA
learner, GTrXL, RND/KNN intrinsic reward used unmodified; hyperparameters = their released bj.json).

## Released code vs paper (disclose)
1. Actions are 13 discrete moves (stay, 8 directions of 1.5 units, 4 axis moves of one slot of flight), not the
   paper's continuous (angle, distance). We use the released action set.
2. The observation includes the elapsed fraction of the episode (a horizon signal); it omits UAV energy, which the
   paper's text lists. We replace the former with own remaining energy fraction.
3. The intrinsic-reward map scale is chosen by `'BJ' in dataset`, which is false for "beijing", so both datasets
   used San Francisco's scale. Our env passes its own scale.
4. Their training process never exits after the last checkpoint (queue/thread deadlock); fixed.

## Port of our operating model (dyn_slot_env.py + adept/env/env_dyn)
Same SensorField and accounting as DynSim; K UAVs always airborne; E_u = (1-rho)U/K per sortie; hover-drain visits;
return autopilot when E - e_fly(home) < 2 Pf tau, immediate replacement. Their method acts once per slot.
* **Slot length: 5 s** (theirs 20 s). Equivalence gate (slot_gate.py, scripted fixed routes, pilot seeds): slotting
  itself costs +19..+160% at 20 s, +5..+30% at 10 s, mostly +3..+10% at 5 s (rr_tour on ring +16..+29%), +2..+5% at
  1-2 s. Action table scaled so 1 unit = v tau / 4 (identical to theirs at tau = 20 s).
* **Collection radius: 50 m** (theirs 500 m on a 7.8 km map; ours 12.6 km with hover-at-sensor collection). Pilot may
  compare {25, 50, 100} m on seeds 13-15 with the scripted gate; fix before training.
* **Observation radius:** global (their sensing range is local; our planners see all ages). Disclose.
* **Information parity:** training world = deployment's positions and priorities with NOMINAL rates and fresh event
  draws per episode; evaluation world = true deployment (= DynSim world). The policy never sees true rates, true
  backlog (an age-based estimate is shown), events, or the clock.
* **Reward:** w_i x age-at-drain / 3600 s per drain (their code: weight x packet gap), epsilon = 0 (our objective).
  Secondary run, if budget allows: epsilon > 0 with AoI_th = 1 h.

## Protocol
1. **Pipeline check:** reproduce one Beijing number from the paper (their EnvUCS, bj.json, U = 3) on the modernised
   stack. Record the gap; a large gap is a finding to report, not to fix by tuning.
2. **Throughput pilot:** configs/throughput_pilot/*.json for 20 min each; read env steps/s. Training hours per run
   = nb_step / rate.
3. **Budget per run:** 1e7 env steps (their convergence scale, Fig. 14), same for every run, no early stopping on
   evaluation data. Training is per deployment (their method trains on a fixed map).
4. **Scope (fill in after step 2):** deployments = ______ (default proposal: paper family, M = 50, Emax in {1.5, 3}
   MJ, seeds 1-3 -> 6 deployments) x the union K grid of those deployments. If K grid x deployments x hours exceeds the
   GPU window, cut deployments, never the K grid (the K* comparison needs full sweeps).
5. **Evaluation:** last checkpoint, `evaluate.py --nb-episode 1` (deterministic world; their eval actor), J from
   `e_weighted_aoi` (= J_timeavg, true world, 12 h horizon, 3 h burn-in).
6. **Comparators in the same slotted world:** `slot_gate.py --csv` on the same deployments, slot 5 s, for
   cluster_patrol, gzz, asghar, C4 (and SA after the i9 fill).

**Primary criterion:** median over the chosen deployments of J*_C4,slot / J*_DRL - 1 (own optima over K, both in the
slotted world), with a percentile bootstrap and win counts. **Secondary:** DRL's K* vs the rule (exact / within one);
DRL vs each published baseline in the slotted world; DynSim-referenced numbers with the slotting cost stated.
Kill/report rule: if a run's evaluation J is worse than rr_tour's in the same world, report it as found (with its
learning curve), do not retrain with new settings.
