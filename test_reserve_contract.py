"""Focused regression checks for the reserve-breach patch.

Written as part of a STATIC audit. This file has NOT been executed.
Run later from the patched project directory with:
    python -m unittest -v test_reserve_contract.py
"""
import unittest
import numpy as np

from dyn_env import DynParams, DynSim


class ReserveBreachContract(unittest.TestCase):
    def make_case(self, emax):
        p = DynParams(M=1, K=1, Emax=emax,
                      T_burnin=0.0, T_horizon=25.0)
        # Deliberately unconditional route: true dwell is unavailable
        # until arrival. The empty buffer at launch does not predict
        # the 5-second hover required after the 10-second flight.
        sim = DynSim(p, planner=lambda req: [0], seed=42)
        sim.field.pos[0] = p.home + np.array([200.0, 0.0])
        sim.field.lam_bits[0] = 1e6
        sim.field.backlog[0] = 0.0
        return sim

    def test_failed_service_pays_both_flights_and_does_not_visit(self):
        # Spendable energy 6100 J; round-trip flight uses 6000 J,
        # but true hover requires another 2000 J. Return remains feasible.
        sim = self.make_case(emax=7625.0)
        metrics = sim.run()
        first = sim.records[0]
        self.assertEqual(first.n_visited, 0)
        self.assertEqual(first.visited, ())
        self.assertAlmostEqual(first.t_land - first.t_launch, 20.0)
        self.assertAlmostEqual(first.energy_used, 6000.0)
        self.assertAlmostEqual(first.commute_time, 20.0)
        self.assertEqual(metrics['reserve_breaches'], 1)
        self.assertEqual(metrics['truncated_sorties'], 0)
        self.assertAlmostEqual(metrics['reserve_breach_frac'], 1.0)

    def test_successful_service_has_no_breach(self):
        # Spendable energy 12000 J; 6000 J flight + 2000 J hover.
        sim = self.make_case(emax=15000.0)
        metrics = sim.run()
        first = sim.records[0]
        self.assertEqual(first.n_visited, 1)
        self.assertEqual(first.visited, (0,))
        self.assertAlmostEqual(first.t_land - first.t_launch, 25.0)
        self.assertAlmostEqual(first.energy_used, 8000.0)
        self.assertAlmostEqual(first.dwell_time, 5.0)
        self.assertEqual(metrics['reserve_breaches'], 0)


if __name__ == '__main__':
    unittest.main()
