"""analyze_depot.py -- rule vs measured optimum for an off-centre-depot sweep, with measured and
ex-ante commute inputs.  usage: python analyze_depot.py depot.csv"""
import sys, csv, math, os
from collections import defaultdict
import numpy as np
rows = list(csv.DictReader(open(sys.argv[1])))
os.environ["DEPOT_FRAC"] = f"{rows[0]['depot_x']},{rows[0]['depot_y']}"
import run_depot                                   # applies the same depot override
from dyn_env import DynParams, SensorField
import exante_rc as X
inst = defaultdict(dict)
for r in rows: inst[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]))][int(r["K"])] = r
out = []
for k, byK in sorted(inst.items()):
    lay, M, E, s = k; Ks = sorted(byK); J = {K: float(byK[K]["J"]) for K in Ks}; r0 = byK[Ks[0]]
    am = min(J, key=J.get); kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"])
    law = lambda c: (int(min(kr, c)) if min(kr, c) == kr else int(round(min(kr, c))))
    p = DynParams(M=M, Emax=E, layout=lay); F = SensorField(DynParams(M=M, Emax=E, layout=lay), np.random.default_rng(s))
    r = np.linalg.norm(F.pos - p.home, axis=1)
    rc_e, Pb_e, _ = X.exante(p, r, F.lam_bits, F.pos, Ks[0])
    l_m, l_e = law(kc), law(X.Kc(p, rc_e, Pb_e))
    reg = lambda l: J[min(max(l, Ks[0]), Ks[-1])] / J[am] - 1
    out.append((lay, M, E, s, am, l_m, l_e, float(r0["rmax_over_rc"]), float(r0["regime_bnd"]), reg(l_m), reg(l_e), Ks[-1] == am))
    print(f"{lay} M={M} {E/1e6:g}MJ s{s}: argmin {am}  rule {l_m} (ex-ante {l_e})  K_reach {float(r0['K_reach_i']):.2f}  "
          f"K_commute {kc:.2f}  rmax/rc {float(r0['rmax_over_rc']):.1f} vs bnd {float(r0['regime_bnd']):.2f}  regret {reg(l_m):.1%}"
          f"{'  CENSORED' if Ks[-1]==am else ''}")
a = np.array([o[4] for o in out]); lm = np.array([o[5] for o in out]); le = np.array([o[6] for o in out])
print(f"\nn={len(out)}  measured inputs: exact {np.mean(a==lm):.2f} within-one {np.mean(abs(a-lm)<=1):.2f} | "
      f"ex-ante: exact {np.mean(a==le):.2f} within-one {np.mean(abs(a-le)<=1):.2f} | "
      f"regret median {np.median([o[9] for o in out]):.1%} max {max(o[9] for o in out):.1%} | censored {sum(o[11] for o in out)}")
