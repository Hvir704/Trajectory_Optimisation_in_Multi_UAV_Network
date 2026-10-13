"""eta_mc.py -- Prop. 'geometric constant' (Sec. IV-D): eta(M) = E[r_max] / (L / sqrt 2) from the formula (eq. eta, uniform
fraction u = 0.3) and by Monte Carlo on the full clustered placement model, plus how often a HOTSPOT sensor lies beyond the
inscribed circle and how often one sets r_max (the formula assumes neither happens).

usage: python eta_mc.py [N fields per M, default 2000]
"""
import sys, math, numpy as np
from dyn_env import DynParams, SensorField

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
_trap = lambda y, x: float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2)   # numpy-version independent
def formula(M, L, u=0.3, n=4000):
    x = np.linspace(L / 2, L / math.sqrt(2), n)
    Aout = L**2 - math.pi * x**2 + 4 * (x**2 * np.arccos(np.minimum(1, L / (2 * x))) - L / 2 * np.sqrt(np.maximum(0, x**2 - L**2 / 4)))
    return (L / math.sqrt(2) - _trap((1 - Aout / L**2) ** (u * M), x)) / (L / math.sqrt(2))
print(f"eta(M): formula (u = 0.3) | Monte Carlo, full model, {N} fields | hotspot sensor beyond L/2 | hotspot sensor sets r_max")
for M in (50, 100, 200, 400):
    p = DynParams(M=M, Emax=1.5e6, layout="paper"); rm = []; beyond = sets = 0
    for s in range(N):
        F = SensorField(p, np.random.default_rng(10**6 + s)); r = np.linalg.norm(F.pos - p.home, axis=1)
        rm.append(r.max()); h = F.in_hotspot; beyond += bool((r[h] > p.L / 2).any()); sets += bool(h[np.argmax(r)])
    print(f"  M={M:3d}  {formula(M, p.L):.3f} | {np.mean(rm) / (p.L / math.sqrt(2)):.3f} | {100*beyond/N:4.0f}% | {100*sets/N:4.1f}%")
