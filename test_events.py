"""test_events.py -- the v2 event model and its integral.

1) The event-age integral equals an independent analytic computation (events merged into episodes that keep the
   oldest fire time), for several step sizes -- i.e. the result does not depend on how the clock is stepped.
2) The old (legacy) handling does depend on the step size and undercounts, which is the bug being fixed.
3) The stationary P(event live) matches 1 - exp(-nu * E[tau]) (Monte Carlo), not nu*tau/(1+nu*tau).
usage: python -m unittest -v test_events.py
"""
import math
import unittest
import numpy as np
from dyn_env import DynParams, SensorField

FIRINGS = {0: [100.0, 150.0, 400.0, 1500.0], 1: [50.0], 2: [900.0, 905.0, 2000.0, 2001.0, 2002.0]}
TAU = 300.0        # fixed lifetime so the reference is exact
T0, T1 = 0.0, 3000.0
BURN, HOR = 200.0, 2800.0


def make_field(legacy):
    p = DynParams(M=3, layout="paper", T_burnin=BURN, T_horizon=HOR)
    p.tau_e_lo = p.tau_e_hi = TAU
    p.legacy_events = legacy
    F = SensorField(p, np.random.default_rng(0))
    q = {i: list(v) for i, v in FIRINGS.items()}
    F.next_event_at[:] = [q[i].pop(0) for i in range(3)]
    F._draw_next_event = lambda t, i: q[i].pop(0) if q[i] else math.inf
    return p, F


def reference():
    """Analytic: merge overlapping events per sensor into episodes (oldest fire time kept), integrate
    g*w*(t - fire) over each episode clipped to [BURN, HOR]."""
    p, F = make_field(False)
    tot = 0.0
    for i, fs in FIRINGS.items():
        eps = []
        for s in sorted(fs):
            if eps and eps[-1][1] > s:
                eps[-1][1] = max(eps[-1][1], s + TAU)
            else:
                eps.append([s, s + TAU])
        for f, u in eps:
            a, b = max(f, BURN), min(u, HOR)
            if b > a:
                tot += p.event_gain * F.wi_base[i] * 0.5 * ((b - f) ** 2 - (a - f) ** 2)
    return tot


def simulate(legacy, dt):
    p, F = make_field(legacy)
    tot, t = 0.0, T0
    while t < T1:
        t1 = min(T1, t + dt)
        F.advance(t, t1)
        a, b = max(t, BURN), min(t1, HOR)
        if legacy:
            tot += F.event_age_integral(a, b).sum() if b > a else 0.0
        else:
            tot += F._ev_last.sum()
        t = t1
    return tot


class EventModel(unittest.TestCase):
    def test_v2_matches_reference_for_every_step_size(self):
        ref = reference()
        for dt in (1.0, 7.0, 60.0, 333.0, 3000.0):
            self.assertAlmostEqual(simulate(False, dt) / ref, 1.0, places=9, msg=f"dt={dt}")

    def test_legacy_was_step_dependent_and_low(self):
        ref = reference()
        coarse, fine = simulate(True, 333.0), simulate(True, 1.0)
        self.assertLess(fine, ref * 0.999)              # overwrite restarts the age: undercounts even when fine
        self.assertNotAlmostEqual(coarse, fine, places=3)

    def test_prior_matches_monte_carlo(self):
        p = DynParams(M=400, layout="paper", T_burnin=0.0, T_horizon=1e9)
        F = SensorField(p, np.random.default_rng(7))
        hot = F.in_hotspot.copy()
        live_time = np.zeros(p.M); t, dt = 0.0, 60.0
        while t < 2.0e6:
            F.advance(t, t + dt); t += dt
            live_time += (F.event_until > t) * dt
        frac = live_time / t
        tau = 0.5 * (p.tau_e_lo + p.tau_e_hi)
        for mask, nu in ((hot, p.event_rate_hot), (~hot, p.event_rate_cold)):
            if mask.sum() < 20:
                continue
            target = 1.0 - math.exp(-nu * tau)
            self.assertAlmostEqual(frac[mask].mean(), target, delta=0.02)
            self.assertAlmostEqual(F.p_live_prior()[mask].mean(), target, places=12)


if __name__ == "__main__":
    unittest.main()
