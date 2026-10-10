# run_all_ryzen.ps1 -- the whole simulator v2.2 rerun on ONE machine (the Ryzen), in priority order.
#
# Run from the repository folder, in the activated venv:
#     powershell -ExecutionPolicy Bypass -File .\run_all_ryzen.ps1
# Safe to stop and restart at any time: every command skips runs already in its output file.
# Stops at the first command that fails (e.g. a version refusal) and says which one.
# Full console log (including the C4 class counts and CP-SAT proof lines): results_v2\logs\run_all.log
#
# Errata update (Oct 2026): before restarting after the errata fixes, delete results_v2\mass.csv and
# results_v2\regularity_*.csv if they exist (their scripts changed; both refuse the old files and stop here).
# Steps 1-6 and the other step-7 files are unaffected and are skipped where already done.

$P = 20                                           # worker processes; the Ryzen has 24 threads
New-Item -ItemType Directory -Path results_v2\logs -Force | Out-Null
Start-Transcript -Path results_v2\logs\run_all.log -Append | Out-Null

# keep Windows from sleeping while this script runs (released automatically when the window closes; no admin needed)
try {
    Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
    [Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null
    Write-Host "Sleep is blocked while this script runs." -ForegroundColor DarkGray
} catch { Write-Host "Could not block sleep; set Power & sleep to Never by hand." -ForegroundColor Yellow }

function Run([string]$cmd) {
    Write-Host "`n>>> $cmd  ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED (exit $LASTEXITCODE): $cmd" -ForegroundColor Red
        Stop-Transcript | Out-Null; exit 1
    }
}

function RunOptional([string]$cmd, [string]$out) {   # a failure is logged and its output removed; the run goes on
    Write-Host "`n>>> (optional) $cmd  ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "optional step failed (exit $LASTEXITCODE), continuing: $cmd" -ForegroundColor Yellow
        if ($out) { Remove-Item $out -ErrorAction SilentlyContinue }
    }
}

if (-not (Test-Path run_grid.py)) { Write-Host "Run this from the repository folder." -ForegroundColor Red; exit 1 }
$G = "--layouts paper ring core --M 50 100 200 --Emax 1.5e6 3e6 --seeds 1-12 --procs $P"

# ---------------------------------------------------------------- 1. SA primary grid (the core of the paper, ~10 h)
Run "python run_grid.py results_v2\sa_grid.csv $G"
# full K = 2..8 coverage for the single-cell tables (coordination, horizon) -- adds only missing K
Run "python run_grid.py results_v2\sa_grid.csv --layouts paper --M 100 --Emax 1.5e6 --seeds 1-12 --K 2 3 4 5 6 7 8 --procs $P"
Run "python run_grid.py results_v2\sa_grid.csv --layouts ring --M 100 --Emax 1.5e6 --seeds 1-6 --K 2 3 4 5 6 --procs $P"

# ---------------------------------------------------------------- 2. every other planner on the primary grid (~8 h, Liu ~4 h)
foreach ($pl in "cyc","cyc_sector0","cluster_patrol","gzz_rev","asghar_rev","liu_mpga","rr_tour") {
    Run "python run_grid.py results_v2\$pl.csv $G --planner $pl" }
Run "python run_grid.py results_v2\rr_sweep.csv --layouts paper ring --M 50 100 200 --Emax 1.5e6 3e6 --seeds 1-12 --planner rr_sweep --procs $P"

# ---------------------------------------------------------------- 3. SA extra cells, depot runs
Run "python run_grid.py results_v2\sa_m400.csv --layouts paper --M 400 --Emax 1.5e6 --seeds 1-12 --procs $P"
Run "python run_grid.py results_v2\sa_6mj.csv --layouts paper --M 100 --Emax 6e6 --seeds 1-12 --procs $P"
foreach ($L in 8000,10000,16000,20000) {
    Run "python run_grid.py results_v2\sa_L$L.csv --layouts paper --M 100 --Emax 1.5e6 --seeds 1-8 --L $L --procs $P" }
foreach ($pl in "sa","cyc","cluster_patrol") {
    $q = if ($pl -eq "sa") { "depot_q" } else { "depot_q_$pl" }
    $e = if ($pl -eq "sa") { "depot_e" } else { "depot_e_$pl" }
    Run "python run_depot.py results_v2\$q.csv --layouts paper ring core --M 50 100 200 --Emax 1.5e6 3e6 --seeds 1-12 --depot 0.25 0.25 --planner $pl --procs $P"
    Run "python run_depot.py results_v2\$e.csv --layouts paper ring --M 100 --Emax 1.5e6 3e6 --seeds 1-12 --depot 0.10 0.50 --planner $pl --procs $P" }

# ---------------------------------------------------------------- 4. fills on the union K grid, iterated
$U  = "results_v2\sa_grid.csv results_v2\sa_fill.csv results_v2\cyc.csv results_v2\cyc_fill.csv results_v2\cluster_patrol.csv results_v2\cluster_patrol_fill.csv results_v2\gzz_rev.csv results_v2\gzz_rev_fill.csv results_v2\asghar_rev.csv results_v2\asghar_rev_fill.csv results_v2\liu_mpga.csv results_v2\liu_mpga_fill.csv results_v2\rr_tour.csv results_v2\rr_tour_fill.csv"
$DQ = "results_v2\depot_q.csv results_v2\depot_q_fill.csv results_v2\depot_q_cyc.csv results_v2\depot_q_cyc_fill.csv results_v2\depot_q_cluster_patrol.csv results_v2\depot_q_cluster_patrol_fill.csv"
$DE = "results_v2\depot_e.csv results_v2\depot_e_fill.csv results_v2\depot_e_cyc.csv results_v2\depot_e_cyc_fill.csv results_v2\depot_e_cluster_patrol.csv results_v2\depot_e_cluster_patrol_fill.csv"
for ($round = 1; $round -le 3; $round++) {
    Write-Host "`n===== fill round $round =====" -ForegroundColor Yellow
    Run "python fill_tail.py results_v2\sa_fill.csv --inputs results_v2\sa_grid.csv results_v2\sa_m400.csv results_v2\sa_6mj.csv results_v2\sa_L8000.csv results_v2\sa_L10000.csv results_v2\sa_L16000.csv results_v2\sa_L20000.csv --planner sa --union $U --rule15 --procs $P"
    foreach ($pl in "cyc","cyc_sector0","cluster_patrol","gzz_rev","asghar_rev","liu_mpga","rr_tour","rr_sweep") {
        Run "python fill_tail.py results_v2\${pl}_fill.csv --inputs results_v2\$pl.csv --planner $pl --union $U --rule15 --procs $P" }
    Run "python fill_tail.py results_v2\depot_q_fill.csv --inputs results_v2\depot_q.csv --planner sa --depot 0.25 0.25 --union $DQ --rule15 --procs $P"
    Run "python fill_tail.py results_v2\depot_e_fill.csv --inputs results_v2\depot_e.csv --planner sa --depot 0.10 0.50 --union $DE --rule15 --procs $P"
    foreach ($pl in "cyc","cluster_patrol") {
        Run "python fill_tail.py results_v2\depot_q_${pl}_fill.csv --inputs results_v2\depot_q_$pl.csv --planner $pl --depot 0.25 0.25 --union $DQ --rule15 --procs $P"
        Run "python fill_tail.py results_v2\depot_e_${pl}_fill.csv --inputs results_v2\depot_e_$pl.csv --planner $pl --depot 0.10 0.50 --union $DE --rule15 --procs $P" }
}
Run "python reproduce_v2.py results_v2 --only none"      # the 'censoring:' block must be all 0; if not, rerun this script

# ---------------------------------------------------------------- 5. single-cell studies (paper, M=100, 1.5 MJ)
$A = "--layouts paper --M 100 --Emax 1.5e6 --seeds 1-12 --procs $P"
foreach ($pl in "sa","greedy","sa_norepair","sa_nolambda","sa_launchdwell","sa_sqrt","sa_whittle") { Run "python run_grid.py results_v2\abl_$pl.csv $A --K 4 --planner $pl" }
foreach ($pl in "sa","sa_launchdwell") { Run "python run_grid.py results_v2\abl_$pl.csv $A --K 2 --planner $pl" }
Run "python run_grid.py results_v2\abl_coord_none.csv $A --K 4 --coord none"
Run "python run_grid.py results_v2\abl_coord_project.csv $A --K 4 --coord project"
Run "python run_grid.py results_v2\abl_perleg.csv $A --K 4 --replan per_leg_sa"
Run "python run_grid.py results_v2\abl_divert.csv $A --K 4 --replan per_leg_sa --divert 1"
Run "python run_grid.py results_v2\coord_none.csv $A --K 2 3 4 5 6 7 8 --coord none"
Run "python run_grid.py results_v2\coord_perleg.csv $A --K 2 3 4 5 6 7 8 --replan per_leg_sa"
Run "python run_grid.py results_v2\horizon24.csv --layouts paper --M 100 --Emax 1.5e6 --seeds 1-6 --K 2 3 4 5 6 7 8 --Th 86400 --procs $P"
Run "python run_grid.py results_v2\horizon24.csv --layouts ring --M 100 --Emax 1.5e6 --seeds 1-6 --K 2 3 4 5 6 --Th 86400 --procs $P"
$I = "--layouts paper --M 100 --Emax 1.5e6 --seeds 1-12 --K 4 --replan per_leg_sa --procs $P"
foreach ($b in "prior","prior_rates","kernel") {
    foreach ($q in "0","0.5","1") { Run "python run_grid.py results_v2\info_${b}_q$q.csv $I --belief $b --q $q" }
    foreach ($t in "15","5")      { Run "python run_grid.py results_v2\info_${b}_tau$t.csv $I --belief $b --q 1 --tau $t" } }
foreach ($b in "kernel","prior") {
    Run "python run_grid.py results_v2\info_${b}_q1_K.csv $A --K 2 3 4 5 6 7 8 --replan per_leg_sa --belief $b --q 1" }

# ---------------------------------------------------------------- 6. M = 40 controlled comparison and the CP-SAT budget study (~4-6 h)
foreach ($pl in "sa","rr_tour","cluster_patrol","cpsat_d5") {
    Run "python run_grid.py results_v2\m40_$pl.csv --layouts paper ring --M 40 --Emax 1.5e6 --seeds 1-12 --planner $pl --procs $P" }
Run "python run_grid.py results_v2\m40_cpsat_d15.csv --layouts ring --M 40 --Emax 1.5e6 --seeds 1-12 --planner cpsat_d15 --procs $P"
Run "python run_grid.py results_v2\m40_cpsat_d45.csv --layouts ring --M 40 --Emax 1.5e6 --seeds `"1,3,4`" --K 3 4 --planner cpsat_d45 --procs $P"

# ---------------------------------------------------------------- 7. remaining SA studies and diagnostics
Run "python run_hetero.py results_v2\hetero.csv --layouts core --M 100 200 --Emax 1.5e6 --seeds 1-12 --procs $P"
Run "python run_mass.py results_v2\mass.csv --layouts paper ring core --M 100 --Emax 1.5e6 3e6 --seeds 1-12 --procs $P"
Run "python outage_test.py results_v2\outage24.csv 1e9"
if (-not (Test-Path results_v2\diag_sec5.txt)) { Run "python diag_sec5.py > results_v2\diag_sec5.txt" }
Run "python measure_regularity.py results_v2\regularity_paper.csv --layout paper --M 100 --Emax 1.5e6 --K 2 3 4 5 6 7 8 --seeds 1-6 --planners sa --export-nodes results_v2\nodes"
Run "python measure_regularity.py results_v2\regularity_ring.csv --layout ring --M 100 --Emax 3e6 --K 4 6 8 --seeds 1-2 --planners sa rr_tour"
Run "python measure_regularity.py results_v2\regularity_core_cl.csv --layout core --M 100 --Emax 1.5e6 --K 6 --seeds 13-14 --planners sa cluster_patrol"
Run "python measure_regularity.py results_v2\regularity_paper_cl.csv --layout paper --M 100 --Emax 1.5e6 --K 4 --seeds 13-14 --planners sa cluster_patrol"
Run "python sortie_regress.py results_v2\sortie_ring.csv --M 50 100 200 --Emax 3e6 --seeds 1-4 --procs $P"
if (-not (Test-Path results_v2\cert_M14.txt)) { Run "python certify_sortie.py 14 1-6 3e5 2 39 > results_v2\cert_M14.txt" }
if (-not (Test-Path results_v2\cert_M15.txt)) { Run "python certify_sortie.py 15 1-6 3e5 2 39 > results_v2\cert_M15.txt" }
if (-not (Test-Path results_v2\fig1_reach.pdf)) { Run "python plot_snapshot_pair.py 100 1.5e6 4 5 1 6 results_v2\fig1_reach.pdf > results_v2\fig1_reach.txt" }
if (-not (Test-Path results_v2\fixcheck.csv)) { Run "python paired_fix_check.py results_v2\fixcheck.csv --layouts ring paper core --M 100 --Emax 1.5e6 3e6 --seeds 1-12 --procs $P" }

# per-decision CP-SAT probes for Sec. VI-A (CP-SAT = Held-Karp at M=12; SA vs proven optima at M=40, K=4, 1.5 MJ)
if (-not (Test-Path results_v2\milp_check_m12.txt)) { RunOptional "python milp_probe.py check 12 1-1 3e5 2 16 > results_v2\milp_check_m12.txt" "results_v2\milp_check_m12.txt" }
if (-not (Test-Path results_v2\milp_time_m40.txt))  { RunOptional "python milp_probe.py time 40 1-6 1.5e6 4 3 120 8 > results_v2\milp_time_m40.txt" "results_v2\milp_time_m40.txt" }

# ---------------------------------------------------------------- 8. regenerate everything
Run "python reproduce_v2.py results_v2"
Write-Host "`nALL DONE. Zip results_v2\ (including logs\ and report\) and send it." -ForegroundColor Green
Stop-Transcript | Out-Null
