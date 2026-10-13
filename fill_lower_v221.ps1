# fill_lower_v221.ps1 -- extend the sweeps whose optimum sits at the SMALLEST fleet swept (v2.2.1 audit fix).
#
# The fill rule now also extends downward: when a planner's optimum is the smallest K swept and that K > 1, the two
# next smaller K are run, iterated until the optimum is interior or at K = 1. With the union grid, every compared
# planner gets the same new K for that deployment. Expected: about 110 runs first, then a few hundred short ring runs
# from the union, plus 9 depot runs and the M=40 cluster-patrol fill. Under an hour on the Ryzen.
#
# Run from the repository folder, in the activated venv:
#     powershell -ExecutionPolicy Bypass -File .\fill_lower_v221.ps1
# Safe to stop and restart: every command skips runs already in its output file.

$P = 20
New-Item -ItemType Directory -Path results_v2\logs -Force | Out-Null
Start-Transcript -Path results_v2\logs\fill_lower_v221.log -Append | Out-Null
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
if (-not (Test-Path run_grid.py)) { Write-Host "Run this from the repository folder." -ForegroundColor Red; exit 1 }

$U  = "results_v2\sa_grid.csv results_v2\sa_fill.csv results_v2\cyc.csv results_v2\cyc_fill.csv results_v2\cluster_patrol.csv results_v2\cluster_patrol_fill.csv results_v2\gzz_rev.csv results_v2\gzz_rev_fill.csv results_v2\asghar_rev.csv results_v2\asghar_rev_fill.csv results_v2\liu_mpga.csv results_v2\liu_mpga_fill.csv results_v2\rr_tour.csv results_v2\rr_tour_fill.csv"
$DQ = "results_v2\depot_q.csv results_v2\depot_q_fill.csv results_v2\depot_q_cyc.csv results_v2\depot_q_cyc_fill.csv results_v2\depot_q_cluster_patrol.csv results_v2\depot_q_cluster_patrol_fill.csv"
$DE = "results_v2\depot_e.csv results_v2\depot_e_fill.csv results_v2\depot_e_cyc.csv results_v2\depot_e_cyc_fill.csv results_v2\depot_e_cluster_patrol.csv results_v2\depot_e_cluster_patrol_fill.csv"

for ($round = 1; $round -le 4; $round++) {
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
# M=40: the cluster patrol's own optimum (its sweeps were never filled); no union, as in the M=40 study
Run "python fill_tail.py results_v2\m40_cluster_patrol_fill.csv --inputs results_v2\m40_cluster_patrol.csv --planner cluster_patrol --procs $P"

Run "python reproduce_v2.py results_v2 --only none"      # the 'censoring' block must now be all 0, bottom edge included
Run "python reproduce_v2.py results_v2"
Write-Host "`nALL DONE. Zip results_v2\report and send it." -ForegroundColor Green
Stop-Transcript | Out-Null
