"""depot_table.py -- Table tab:depot from the off-centre sweeps.
usage: python depot_table.py depot_q.csv depot_e.csv
Runs analyze_depot.py on each file (it applies that file's depot override) and aggregates by family."""
import sys, re, subprocess
import numpy as np
from collections import defaultdict
for path in sys.argv[1:]:
    out = subprocess.run([sys.executable, "analyze_depot.py", path], capture_output=True, text=True, check=True).stdout
    fam = defaultdict(list)
    for line in out.splitlines():
        m = re.match(r"(\w+) M=\d+ [\d.]+MJ s\d+: argmin (\d+)\s+rule (\d+).*regret ([\d.]+)%", line)
        if m: fam[m.group(1)].append((int(m.group(2)), int(m.group(3)), float(m.group(4))))
    print(f"== {path}: {out.strip().splitlines()[-1]}")
    for f in ("paper", "core", "ring"):
        if f not in fam: continue
        v = np.array(fam[f])
        print(f"  {f:5s} n={len(v):3d} exact {np.mean(v[:,0]==v[:,1]):.2f}  within-one {np.mean(abs(v[:,0]-v[:,1])<=1):.2f}  "
              f"bias {np.mean(v[:,1]-v[:,0]):+.2f}  regret median {np.median(v[:,2]):.1f}% p90 {np.quantile(v[:,2],.9):.1f}%")
