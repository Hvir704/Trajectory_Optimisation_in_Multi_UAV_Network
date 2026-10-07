"""check_liu_code.py -- does this machine run the SAME Liu / simulator code as the reference (BMP repo, dce4df2)?
Run from the folder you will run the Liu grid from:   python check_liu_code.py
1) fingerprints the five files the Liu planner depends on (line endings ignored) against the reference;
2) re-runs two reference jobs (core, M=50, 1.5 MJ, seed 3, K=2) and requires EXACT equality:
     cluster_patrol J = 137176.70340212446   (the canonical value behind the paper's tables)
     liu_mpga       J = 137183.12975865908   (reproduced on two machines: the i5 and the paper-chat sandbox)
PASS on both = this machine may run the Liu grid. Anything else: do not run it; report the output."""
import hashlib, os, sys, time
REF = {"liu_mpga.py": "9cb6265ff95c253a", "dyn_env.py": "e8ce0034e3bafdff", "run_grid.py": "f7e03880e976347d",
       "rr_planner.py": "4ab4272c7216a5a0", "cyclic_sched.py": "a85c0b7c49d212fa"}
found = {}
for root, dirs, files in os.walk("."):
    if ".git" in root or "venv" in root or "__pycache__" in root: continue
    for f in files:
        if f in REF: found.setdefault(f, []).append(os.path.join(root, f))
ok_files = True
for name, ref in REF.items():
    paths = found.get(name, [])
    if not paths: print(f"  MISSING  {name}"); ok_files = False; continue
    for p in paths:
        h = hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()[:16]
        tag = "same" if h == ref else "DIFFERENT"
        if h != ref: ok_files = False
        print(f"  {tag:9s} {name:16s} {h}  ({p})")
    if len(paths) > 1: print(f"  NOTE: {len(paths)} copies of {name}; Python imports whichever comes first on sys.path")
for d in (".", "sim", "experiments", os.path.join("baselines", "liu2022")):
    if os.path.isdir(d) and os.path.abspath(d) not in sys.path: sys.path.append(os.path.abspath(d))
from run_grid import one
ok_runs = True
for pl, ref in (("cluster_patrol", 137176.70340212446), ("liu_mpga", 137183.12975865908)):
    t = time.time()
    J = float(one(("core", 50, 1.5e6, 2, 3, "exclude", "launch", 0.0, 0.0, "prior", 0.0, pl, 12600.0, 43200.0, 1200))["J"])
    same = J == ref; ok_runs &= same
    print(f"  {'EXACT' if same else 'MISMATCH':8s} {pl:15s} J = {J!r}  (reference {ref!r}, {time.time()-t:.0f}s)")
print("\nRESULT:", "PASS -- this machine runs the reference code" if (ok_files and ok_runs)
      else "FAIL -- do not run the Liu grid here; send this output" if not ok_runs
      else "RUNS MATCH but some files differ -- send this output before running")
