"""check_sim.py -- is this folder running the corrected simulator (v2), built on the code behind the paper?

1) LEGACY mode (LEGACY=1: every old behaviour) must reproduce five published values EXACTLY (core, M=50, 1.5 MJ,
   seed 3, K=2) and the v1 diversion run; LEGACY_V21=1 must reproduce the v2.1 reference value. These planners are deterministic across machines, so this proves the
   code base is the one that produced the paper's numbers.
2) The corrected simulator must pass the reserve-contract and event-model unit tests.
3) An unknown planner name must raise (no silent fallback to SA).
4) The corrected simulator must differ from legacy on the reference job (i.e. the fix is actually active).
usage (repo root):  python check_sim.py
"""
import os, subprocess, sys
REF = {"cluster_patrol": 137176.70340212446, "cyc": 117185.01812055834, "liu_mpga": 137183.12975865908,
       "gzz_rev": 155847.0955335653, "asghar_rev": 369485.91885155096}
JOB = lambda pl: ("core", 50, 1.5e6, 2, 3, "exclude", "launch", 0.0, 0.0, "prior", 0.0, pl, 12600.0, 43200.0, 1200)
os.environ["LEGACY"] = "1"
from run_grid import one
ok = True
for pl, ref in REF.items():
    J = float(one(JOB(pl))["J"]); same = J == ref; ok &= same
    print(f"  {'EXACT' if same else 'MISMATCH':8s} legacy {pl:15s} {J!r}  (published {ref!r})", flush=True)
# diversion, 6 h. GREEDY launch and greedy per-leg replanning: SA is not bit-reproducible across platforms (np.exp in
# the Metropolis test differs in the last bit between C runtimes), so an SA reference measured on one OS can fail on
# another with correct code. Greedy has no random draws or exp. v1 value from unpatched b1656d3 (284 diversions).
DIV = ("paper", 100, 1.5e6, 4, 2, "exclude", "per_leg_greedy", 1, 0.0, "prior", 0.0, "greedy", 12600.0, 21600.0, 1200)
rdiv = one(DIV); J = float(rdiv["J"]); same = J == 975174.924837339 and int(rdiv["n_diverts"]) == 284; ok &= same
print(f"  {'EXACT' if same else 'MISMATCH':8s} legacy diversion      {J!r}  (v1 975174.924837339, 284 diverts; got {rdiv['n_diverts']})", flush=True)
del os.environ["LEGACY"]
os.environ["LEGACY_V21"] = "1"                   # v2.1 = v2.2 without the departure check and the event streams
J = float(one(JOB("cluster_patrol"))["J"]); same = J == 137417.93836971052; ok &= same
print(f"  {'EXACT' if same else 'MISMATCH':8s} v2.1 mode cluster_patrol {J!r}  (v2.1 137417.93836971052)", flush=True)
del os.environ["LEGACY_V21"]
from run_grid import SIM_VERSION
row = one(JOB("cluster_patrol")); Jv2 = float(row["J"]); active = Jv2 != REF["cluster_patrol"] and row["sim_version"] == SIM_VERSION
ok &= active
print(f"  {'PASS' if active else 'FAIL':8s} corrected simulator ({row['sim_version']}) differs from legacy on the reference job ({Jv2!r})")
t = subprocess.run([sys.executable, "-m", "unittest", "-q", "test_reserve_contract.py", "test_events.py", "test_v22.py"], capture_output=True, text=True)
ok &= t.returncode == 0
print(f"  {'PASS' if t.returncode == 0 else 'FAIL':8s} reserve-contract, event-model and v2.2 unit tests")
try:
    one(JOB("not_a_planner")); print("  FAIL     unknown planner did not raise"); ok = False
except ValueError:
    print("  PASS     unknown planner raises")
print("\nRESULT:", "PASS -- corrected simulator on the paper's code base" if ok else "FAIL -- do not run experiments here")
sys.exit(0 if ok else 1)
