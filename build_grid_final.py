"""build_grid_final.py -- the single analysis file every table and figure is generated from.
Takes the same-machine rerun (sa_rerun.csv: 18 primary cells with all anti-censoring extensions) and
adds from the banked grid.csv only the cells the rerun does not cover (paper M=400, paper M=100 at
6 MJ, and the four L-sweep cells). Standard settings only; a duplicate key aborts the build.
usage: python build_grid_final.py sa_rerun.csv grid.csv grid_final.csv"""
import sys, csv
def std(r):
    return (r.get("planner", "sa") == "sa" and float(r["Th"]) == 43200 and r["coord"] == "exclude"
            and r["replan"] == "launch" and str(r.get("divert", "0")) in ("0", "0.0")
            and float(r.get("q", 0)) == 0 and r.get("belief", "prior") == "prior" and float(r.get("tau", 0)) == 0)
rer = [r for r in csv.DictReader(open(sys.argv[1])) if std(r)]
cells = {(r["layout"], r["M"], r["Emax"], r["L"]) for r in rer}
g = list(csv.DictReader(open(sys.argv[2])))
if list(rer[0].keys()) != list(g[0].keys()): sys.exit("schema mismatch between the two inputs")
out = rer + [r for r in g if std(r) and (r["layout"], r["M"], r["Emax"], r["L"]) not in cells]
seen = set()
for r in out:
    k = (r["layout"], r["M"], r["Emax"], r["L"], r["seed"], r["K"])
    if k in seen: sys.exit(f"duplicate key {k}")
    seen.add(k)
with open(sys.argv[3], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(g[0].keys())); w.writeheader(); w.writerows(out)
print(f"{len(out)} rows, {len({k[:4] for k in seen})} cells -> {sys.argv[3]}")
