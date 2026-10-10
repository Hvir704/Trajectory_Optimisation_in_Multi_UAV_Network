"""test_v22.py -- the two v2.2 changes.

1) Common random numbers: with per-sensor event streams, a field's event history does not depend on how the clock
   is stepped (and therefore not on the planner or K, which only change the step boundaries). Under the legacy
   single stream it does.
2) Departure check: on an instance where v2.1 landed into the reserve, v2.2 never does, and every sortie's energy
   stays within the usable budget; with legacy_depart the dips come back.
usage: python -m unittest -v test_v22.py
"""
import unittest
import numpy as np
from dyn_env import DynParams, DynSim, SensorField
from sa_sortie import build_sa_planner


def history(legacy, dt, T=60000.0):
    p = DynParams(M=60, layout="paper", T_burnin=0.0, T_horizon=1e9)
    p.legacy_rng = legacy
    F = SensorField(p, np.random.default_rng(11))
    t, fires = 0.0, []
    while t < T:
        t1 = min(T, t + dt); F.advance(t, t1); t = t1
        fires.append(F.event_fire.copy())
    live = np.unique(np.concatenate([x[np.isfinite(x)] for x in fires]))
    return F.next_event_at.copy(), F.event_until.copy(), live


class CommonRandomNumbers(unittest.TestCase):
    def test_history_independent_of_step_partition(self):
        a, b = history(False, 7.0), history(False, 333.0)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_legacy_stream_depends_on_partition(self):
        a, b = history(True, 7.0), history(True, 333.0)
        self.assertFalse(np.array_equal(a[0], b[0]))

    def test_field_layout_unchanged(self):
        p1 = DynParams(M=50, layout="core"); p2 = DynParams(M=50, layout="core"); p2.legacy_rng = True
        F1 = SensorField(p1, np.random.default_rng(3)); F2 = SensorField(p2, np.random.default_rng(3))
        np.testing.assert_array_equal(F1.pos, F2.pos); np.testing.assert_array_equal(F1.wi_base, F2.wi_base)
        np.testing.assert_array_equal(F1.lam_bits, F2.lam_bits)


def run(legacy_depart):
    p = DynParams(M=50, K=3, Emax=1.5e6, layout="core")
    p.legacy_depart = legacy_depart
    sim = DynSim(p, build_sa_planner(iters=1200, seed_base=3), seed=3)
    m = sim.run()
    return p, sim, m


class DepartureCheck(unittest.TestCase):
    def test_no_reserve_use(self):
        p, sim, m = run(False)
        self.assertEqual(m["reserve_dips"], 0)
        self.assertLessEqual(max(r.energy_used for r in sim.records), p.E_usable + 1e-6)
        for r in sim.records:      # energy bookkeeping still exact
            self.assertAlmostEqual(r.energy_used, p.Pf * (r.commute_time + r.travel_time) + p.Ph * r.dwell_time, places=4)

    def test_legacy_dips(self):
        p, sim, m = run(True)
        self.assertGreater(m["reserve_dips"], 0)


if __name__ == "__main__":
    unittest.main()
