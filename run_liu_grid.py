"""run_liu_grid.py -- run the adapted Liu MPGA at every (deployment, K) in liu_jobs.csv. Resumable.
Place in baselines/liu2022/ of the BMP repo and run from the repo root:
    python baselines/liu2022/run_liu_grid.py liu_jobs.csv liu_grid.csv --procs 22
Output schema = experiments/run_grid.py FIELDS (planner 'liu_mpga'), so fill_tail.py and the analysis read it."""
import os, sys, csv, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for d in (HERE, os.path.join(ROOT, "sim"), os.path.join(ROOT, "experiments")):
    if d not in sys.path: sys.path.insert(0, d)
os.environ["PYTHONPATH"] = os.pathsep.join([HERE, os.path.join(ROOT, "sim"), os.path.join(ROOT, "experiments"),
                                             os.environ.get("PYTHONPATH", "")])   # spawn workers inherit this
from run_grid import one, FIELDS

def job(t):
    lay, M, E, s, K = t
    return one((lay, M, E, K, s, "exclude", "launch", 0.0, 0.0, "prior", 0.0, "liu_mpga", 12600.0, 43200.0, 1200))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("jobs"); ap.add_argument("out"); ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    todo = [(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]), int(r["K"])) for r in csv.DictReader(open(a.jobs))]
    done = set()
    if os.path.exists(a.out) and os.path.getsize(a.out):
        done = {(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]), int(r["K"])) for r in csv.DictReader(open(a.out))}
    todo = [t for t in todo if t not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    # longest first, so the M=200 runs do not straggle at the end
    todo.sort(key=lambda t: (-t[1], -t[4]))
    new = not done
    from multiprocessing import Pool
    with open(a.out, "a", newline="") as f, Pool(a.procs) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader(); f.flush()
        for i, r in enumerate(pool.imap_unordered(job, todo), 1):
            w.writerow(r); f.flush()
            if i % 25 == 0 or i == len(todo): print(f"  {i}/{len(todo)}  last: {r['layout']} M={r['M']} K={r['K']} s={r['seed']} J={float(r['J']):.3e}", flush=True)
