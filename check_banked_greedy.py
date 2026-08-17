import numpy as np
from compare_baseline import gen, greedy_init, fleet_obj

rng = np.random.default_rng(2025)
seeds = [int(rng.integers(0, 10_000_000)) for _ in range(12)]
for M, K, Emax, banked in [(200, 4, 50000, -348.39),
                           (100, 4, 50000, -235.45),
                           (100, 1, 50000, -89.56)]:
    Ee = Emax / K
    J = []
    for s in seeds:
        pos, wi, tcd = gen(M, s)
        tr, _ = greedy_init(pos, wi, tcd, K, Ee, M)
        J.append(fleet_obj(tr, pos, wi, tcd))
    m12, m8 = float(np.mean(J)), float(np.mean(J[:8]))
    print(f'M={M:>3} K={K}  n=12 {m12:>9.3f}  n=8 {m8:>9.3f}  '
          f'banked {banked:>9.2f}  delta12 {m12-banked:>+7.3f}')