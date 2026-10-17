"""predict_horizon_v3.py -- regenerates heldout_v3_horizon_predictions.csv (H5, T = 24 h; registered 2026-10-12).

The original generator was a `predict-horizon` mode of followup_v3.py that was removed together with the reverted H4
amendment; followup_v3.py is kept unchanged since H5 was registered, so the generator lives here. Same columns, same
rules: ex-ante commute inputs at the first K of the registered sweep (min k_range), v3 at the run window (12 h) and at
24 h, and the sweep range the run used (K_lo = min k_range; K_hi = largest of max k_range, old + 2, new12 + 2,
new24 + 2, floor K_flight(S_1) + 2, where S_1 drops the farthest sensor).

usage: python predict_horizon_v3.py [out.csv]       (default: print to stdout; never overwrites the registered file)
"""
import sys, csv, math, hashlib, io
import numpy as np
from theory_v3 import field, old_rule, surrogate_rule
from exante_rc import exante
from run_grid import k_range

L = 12600.0; E = 1.5e6; LAY = "core"; TH = 86400.0
rows = []
for M in (100, 200):
    for s in range(13, 21):
        p, F = field(LAY, M, E, L, s); r = np.linalg.norm(F.pos - p.home, axis=1)
        kr = k_range(LAY, M, E, s, L); K0 = min(kr)
        rc, Pb, _ = exante(p, r, np.full(M, p.lam_bits), F.pos, K0)
        old = old_rule(p, r, rc, Pb)
        n12, _ = surrogate_rule(p, F, rc, Pb)
        n24, j24 = surrogate_rule(p, F, rc, Pb, horizon=TH)
        U = (1 - p.rho) * E; kfl1 = math.floor(p.v * U / (2 * p.Pf * np.sort(r)[-2]))
        rows.append(dict(layout=LAY, M=M, Emax=E, L=L, seed=s, Th=TH, old=old, new12=n12, new24=n24, new24_stranded=j24,
                         K_lo=K0, K_hi=max(max(kr), old + 2, n12 + 2, n24 + 2, kfl1 + 2)))
buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
text = buf.getvalue()
print("sha256 (LF-normalised):", hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest(), file=sys.stderr)
if len(sys.argv) > 1:
    if sys.argv[1].endswith("heldout_v3_horizon_predictions.csv"): sys.exit("refusing to overwrite the registered file")
    open(sys.argv[1], "w", newline="").write(text)
else:
    sys.stdout.write(text)
