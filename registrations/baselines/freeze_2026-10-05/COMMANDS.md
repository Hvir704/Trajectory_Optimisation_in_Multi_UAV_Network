# Remaining runs — command list (WSL2, same machine for everything)

Order: 0 setup -> 1 Beijing checkpoint curve (minutes, GPU) -> 2 classical held-out + 3 slotted comparators
(together, CPU, a few hours) -> 4 DRL scope (about 6 days, GPU + all cores) -> 5 statistics.
Run 2/3 BEFORE 4: the DRL workers saturate every core, so CPU jobs alongside would slow both.

```bash
# ===== 0. Setup ===========================================================================
source ~/drlucs.env                                 # venv + PYTHONPATH (from the first handover)
export PROJ=/mnt/c/harshvir-aoi/Trajectory_Optimisation_in_Multi_UAV_Network
export DRL=~/DRL-UCS-AoI-Threshold
export DRLS=$DRL/DRL_UCS_AoI/adept/scripts
export PKG=~/baseline_freeze
cd ~ && unzip -o /mnt/c/path/to/baseline_freeze_2026-10-05.zip      # edit the path
cp $PKG/planners/{dyn_slot_env.py,slot_gate.py,baseline_stats.py} $PROJ/
cd $DRL && patch -p1 < $PKG/drl/update_2026-10-05.patch            # env edge fix, eval-world marker, eval keys
cd $PROJ && sha256sum gzz_planner.py asghar_planner.py dyn_slot_env.py slot_gate.py baseline_stats.py
#   compare with BASELINE_FREEZE_2026-10-05.md section 6 (dyn_slot_env/slot_gate/baseline_stats must match;
#   if git turned LF into CRLF the hashes differ -- check with: tr -d '\r' < f | sha256sum)

# ===== 1. Beijing: kappa at every saved checkpoint (deterministic, 1 episode each) =========
RUN=~/drl_logs/EnvUCS-v0/bj_repro_ActorLearner_ImpalaHostActor_Linear_2026-10-04_17-24-55
cd $DRLS
for ck in $(ls -d $RUN/[0-9]* | xargs -n1 basename | sort -n); do
  echo -n "$ck "
  python -u evaluate.py --logdir $RUN --epoch $ck --nb-episode 1 --gpu-id 0 --custom-network RealNetwork 2>&1 \
    | grep -E " (b_emergency_violation_ratio|e_weighted_aoi|f_weighted_bar_aoi) :" | awk '{printf "%s=%s ", $1, $3}'
  echo
done | tee ~/bj_curve.txt

# ===== 2. Held-out gzz_rev / asghar_rev (216 deployments) + union fill =====================
cd $PROJ
P=$(( $(nproc) - 3 ))
python run_grid.py results/gzz_eval.csv    --layouts paper ring core --M 50 100 200 --Emax 1.5e6 3e6 --seeds 1-12 --planner gzz_rev    --procs $P
python run_grid.py results/asghar_eval.csv --layouts paper ring core --M 50 100 200 --Emax 1.5e6 3e6 --seeds 1-12 --planner asghar_rev --procs $P
U="results/eval_all_v2.csv results/eval_fill_v2.csv results/eval_cyc_v2.csv results/cluster_win.csv results/fill_cluster.csv results/grid_final.csv results/fill_sa.csv"
python fill_tail.py results/fill_gzz.csv    --inputs results/gzz_eval.csv    --planner gzz_rev    --union $U --procs $P
python fill_tail.py results/fill_asghar.csv --inputs results/asghar_eval.csv --planner asghar_rev --union $U --procs $P
#   (if your final SA files are not grid_final.csv + fill_sa.csv, put the final ones in U)

# ===== 3. Slotted-world comparators for the DRL deployments (start in a second terminal with step 2)
cd $PROJ
CELLS="$(for K in $(seq 1 11); do printf 'paper:50:1.5e6:%s ' $K; done)$(for K in $(seq 1 10); do printf 'ring:50:1.5e6:%s ' $K; done)"
nohup python slot_gate.py --cells $CELLS --slots 5 --planners cyc cluster_patrol gzz_rev asghar_rev rr_tour \
      --seeds 1 2 3 --csv results/drl_comparators.csv > ~/slot_comparators.log 2>&1 &

# ===== 4. DRL scope (34 runs + extensions; resumable; cut-off 16 Oct 08:00) ================
# start after 2 and 3 have finished
cd $PKG/drl
nohup python -u drl_scope.py --cfgdir ~/drl_scope_cfg --logdir ~/drl_logs --results ~/drl_results.csv \
      --cutoff "2026-10-16 08:00" > ~/drl_scope.log 2>&1 &
tail -f ~/drl_scope.log            # first line per run: "train+eval (...) K=..", then "J=... n_never=... h"
# after the FIRST run finishes: check ~/drl_results.csv has eval_world=1 and a plausible J; if the driver stops
# with an error, fix the cause and simply rerun the same command (finished runs are skipped).

# ===== 5. Statistics ========================================================================
cd $PROJ
C4="results/eval_all_v2.csv results/eval_fill_v2.csv results/eval_cyc_v2.csv"
python baseline_stats.py --a cyc --b gzz_rev    --csv $C4 results/gzz_eval.csv results/fill_gzz.csv       --ell results/ell_eval.csv --out results/stats_c4_vs_gzz.csv    | tee results/stats_c4_vs_gzz.txt
python baseline_stats.py --a cyc --b asghar_rev --csv $C4 results/asghar_eval.csv results/fill_asghar.csv --ell results/ell_eval.csv --out results/stats_c4_vs_asghar.csv | tee results/stats_c4_vs_asghar.txt
CL="results/cluster_win.csv results/fill_cluster.csv"
python baseline_stats.py --a gzz_rev    --b cluster_patrol --csv $CL results/gzz_eval.csv results/fill_gzz.csv       --ell results/ell_eval.csv | tee results/stats_gzz_vs_cluster.txt
python baseline_stats.py --a asghar_rev --b cluster_patrol --csv $CL results/asghar_eval.csv results/fill_asghar.csv --ell results/ell_eval.csv | tee results/stats_asghar_vs_cluster.txt
python $PKG/drl/drl_stats.py --drl ~/drl_results.csv --slot results/drl_comparators.csv --ell results/ell_eval.csv | tee results/drl_stats.txt

# ===== 6. Send back ============================================================================
cd ~ && tar czf results_final.tgz bj_curve.txt drl_results.csv drl_scope.log slot_comparators.log \
   -C $PROJ/results gzz_eval.csv fill_gzz.csv asghar_eval.csv fill_asghar.csv drl_comparators.csv \
   stats_c4_vs_gzz.txt stats_c4_vs_asghar.txt stats_gzz_vs_cluster.txt stats_asghar_vs_cluster.txt drl_stats.txt
```
Steps 2-3 can be sent back as soon as they finish (with step 1) -- the classical-baseline result and the paper text
do not need to wait for the DRL runs.
