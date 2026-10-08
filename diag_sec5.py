"""diag_sec5.py -- the buffer and gray-set numbers quoted in Sec. IV (Prop. 3 discussion) and Sec. IV-A (reach bracket).

  A) ring, M=100, 3 MJ, K = 4, 6, 8, seeds 1-2: served visits that find a full buffer; fraction of generated data
     dropped in the evaluation window.
  B) ring, M=100, 1.5 MJ, K = 2, 3, 4, 6, seeds 1-12: fraction dropped, sensors never served.
  C) paper, M=50, 3 MJ, seed 3, K = 11 (inside the reach bracket): visits after burn-in to each sensor beyond the
     full-buffer radius, and the largest buffer fill they are found at.
Runs the simulator exactly as run_grid.one() does (same parameters, planner and version), serially.
usage: python diag_sec5.py [--procs N]   -> prints the three blocks; about 10-20 min on one core
"""
import sys, numpy as np
import run_grid
from run_grid import one
from dyn_env import DynParams, SensorField

LOG = {}

def hook(sim):
    F = sim.field; p = sim.p; orig = F.visit; rec = LOG.setdefault("visits", [])
    def visit(t, i):
        rec.append((t, int(i), float(F.backlog[i]) / p.B_bits))
        return orig(t, i)
    F.visit = visit

def job(layout, M, E, K, seed):
    LOG.clear(); run_grid._SIM_HOOK = hook
    row = one((layout, M, E, K, seed, "exclude", "launch", 0.0, 0.0, "prior", 0.0, "sa", 12600.0, 43200.0, 1200))
    run_grid._SIM_HOOK = None
    v = [x for x in LOG.get("visits", []) if x[0] >= 3 * 3600.0]
    return row, v

print("A) ring M=100 3 MJ: full-buffer visits and drop fraction")
for K in (4, 6, 8):
    for s in (1, 2):
        row, v = job("ring", 100, 3e6, K, s)
        print(f"   K={K} s={s}: visits {len(v)}, finding a full buffer {sum(1 for x in v if x[2] >= 0.999)}, "
              f"max fill {max(x[2] for x in v):.2f}, drop_frac {float(row['drop_frac']):.4%}, never served {row['n_never']}")

print("B) ring M=100 1.5 MJ: drop fraction and never-served sensors (mean over seeds 1-12)")
for K in (2, 3, 4, 6):
    d, nn = [], []
    for s in range(1, 13):
        row, _ = job("ring", 100, 1.5e6, K, s); d.append(float(row["drop_frac"])); nn.append(int(row["n_never"]))
    print(f"   K={K}: drop_frac mean {np.mean(d):.3%} max {max(d):.3%}; never served mean {np.mean(nn):.1f}")

print("C) paper M=50 3 MJ seed 3 K=11: sensors beyond the full-buffer radius")
p = DynParams(M=50, Emax=3e6, layout="paper"); U = (1 - p.rho) * 3e6
F = SensorField(DynParams(M=50, Emax=3e6, layout="paper"), np.random.default_rng(3))
r = np.linalg.norm(F.pos - p.home, axis=1)
Kr = p.v * U / (2 * p.Pf * r.max() + p.v * p.Ph * p.B_bits / p.R); Kf = p.v * U / (2 * p.Pf * r.max())
r_reach = p.v * (U / 11 - p.Ph * p.B_bits / p.R) / (2 * p.Pf)
gray = [int(i) for i in np.where(r > r_reach)[0]]
row, v = job("paper", 50, 3e6, 11, 3)
print(f"   K_reach={Kr:.2f} K_flight={Kf:.2f}; gray sensors {gray}; never served {row['n_never']}")
for i in gray:
    vi = [x for x in v if x[1] == i]
    print(f"   sensor {i}: {len(vi)} visits after burn-in, max fill at visit {max((x[2] for x in vi), default=float('nan')):.0%}")
