"""
c3_arm2_probe.py -- does per-sensor dwell memory close the A3 truncation gap?

Matched-seed A/B. Same planner, same instances; only dwell_est differs.
Reports the three C3 metrics: trunc%, T_s/t_c vs prediction, J_timeavg.
"""
import argparse
import numpy as np

from dyn_env import DynParams, DynSim, greedy_ratio_planner
from dwell_memory import build_dwell_memory_planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[100])
    ap.add_argument("--K", type=int, nargs="+", default=[4])
    ap.add_argument("--Emax", type=float, default=1.5e6)
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--T-horizon", type=float, default=6 * 3600.0)
    ap.add_argument("--T-burnin", type=float, default=1.5 * 3600.0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    seeds = [int(rng.integers(0, 10_000_000)) for _ in range(a.instances)]

    print(f"{'M':>4} {'K':>3} {'arm':>10} {'J_mean':>12} {'trunc%':>7} "
          f"{'Ts/tc':>6} {'pred':>6} {'n_vis':>6} {'rateMAE':>8}")
    for M in a.M:
        for K in a.K:
            for arm in ("nominal", "memory"):
                J, tr, ts, tp, nv, mae = [], [], [], [], [], []
                for s in seeds:
                    p = DynParams(M=M, K=K, Emax=a.Emax,
                                  T_horizon=a.T_horizon, T_burnin=a.T_burnin)
                    if arm == "nominal":
                        pl = greedy_ratio_planner
                    else:
                        pl = build_dwell_memory_planner(greedy_ratio_planner, p)
                    sim = DynSim(p, planner=pl, seed=s)
                    m = sim.run()
                    J.append(m["J_timeavg"])
                    tr.append(100 * m["truncated_sorties"] / max(m["n_sorties_total"], 1))
                    ts.append(m["T_s_over_t_c"]); tp.append(m["T_s_over_t_c_pred"])
                    nv.append(m["mean_n_visited"])
                    if arm == "memory":
                        mae.append(pl.rate_mae(sim.field.lam_bits))
                mstr = f"{np.mean(mae):8.3f}" if mae else f"{'-':>8}"
                print(f"{M:>4} {K:>3} {arm:>10} {np.mean(J):>12.0f} "
                      f"{np.mean(tr):>7.1f} {np.mean(ts):>6.2f} {np.mean(tp):>6.2f} "
                      f"{np.mean(nv):>6.1f} {mstr}")


if __name__ == "__main__":
    main()
