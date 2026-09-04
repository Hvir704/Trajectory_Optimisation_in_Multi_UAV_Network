"""
dwell_memory.py -- ARM 2 for C3: per-sensor dwell memory, no learning.

WHY THIS EXISTS
  HANDOFF_C3 A3 states the sharp hypothesis: SA and the stub re-estimate dwell
  from a NOMINAL rate every sortie with no memory, true rates are heterogeneous
  (0.5-1.5x), this costs measurable truncation, and a policy that LEARNS
  per-sensor dwell patterns should fix it.

  The second half of that does not follow. In dyn_env.py:

      SensorField.__init__ :  self.lam_bits = p.lam_bits * rng.uniform(.5,1.5,M)
      SensorField.advance  :  backlog += lam_bits * dt   (clamped at B_bits)
      SensorField.dwell_time: return backlog / R

  lam_bits[i] is drawn ONCE and never changes. Backlog growth is DETERMINISTIC.
  So on any visit at age a with measured dwell td:

      backlog = lam_bits[i] * a          (whenever not clamped at B)
      td      = backlog / R
      => lam_bits[i] = td * R / a        EXACT, from a SINGLE visit.

  This is one-shot system identification, not a learning problem. There is no
  noise to average over and no pattern to generalise. If this arm closes the
  truncation gap, the A3 mechanism is NOT evidence for RL -- it is evidence for
  four lines of bookkeeping, and a referee will say so.

INFORMATION SET -- legitimate.
  observe() receives only (node, time, measured dwell, age at visit): what a
  drone physically measures while servicing a node. It never sees lam_bits,
  backlog at unvisited nodes, event state, or the horizon (CONTEXT_60 §7).

CLAMP HANDLING.
  If backlog hit B_bits the sample is censored and gives a LOWER bound on the
  rate, so it is discarded rather than fitted. Detected by td >= B/R - eps.
"""

from __future__ import annotations

import numpy as np

from dyn_env import DynParams, SortieRequest


class DwellMemoryPlanner:
    """Wraps any SortiePlanner, substituting a memory-based dwell estimate.

    The wrapped planner is unchanged; it simply receives a better dwell_est.
    """

    def __init__(self, inner, p: DynParams, prior_weight: float = 1.0):
        self.inner = inner
        self.p = p
        self.lam_hat = np.full(p.M, p.lam_bits, dtype=float)  # nominal prior
        self.n_obs = np.zeros(p.M, dtype=int)
        self.prior_weight = prior_weight
        self.n_censored = 0
        self.n_shortage = 0
        # a sensor cannot plausibly be revisited faster than one buffer-drain
        # time; below that the sample is contaminated (see observe()).
        self.min_age = 120.0

    # --- observation side -----------------------------------------------
    def observe(self, i: int, t: float, td_measured: float, age: float) -> None:
        if age <= 1e-9:
            return
        td_cap = self.p.B_bits / self.p.R
        if td_measured >= td_cap - 1e-6:
            self.n_censored += 1        # censored: lower bound only, discard
            return
        # SHORT-AGE REJECTION. dyn_env zeroes the buffer at arrival but then
        # advances the world across the dwell, so the sensor accrues bits during
        # its OWN service window. A revisit at small age therefore reads
        # backlog = lam*(age + td_prev), inflating the implied rate (measured up
        # to 33x). The visiting drone cannot know td_prev, so such samples are
        # dropped rather than corrected.
        if age < self.min_age:
            self.n_shortage += 1
            return
        lam_obs = td_measured * self.p.R / age
        n = self.n_obs[i]
        # running mean against the nominal prior; converges in one step when
        # prior_weight -> 0, kept at 1 so a single odd sample cannot dominate.
        self.lam_hat[i] = ((self.prior_weight * self.p.lam_bits + n * self.lam_hat[i]
                            + lam_obs) / (self.prior_weight + n + 1))
        self.n_obs[i] = n + 1

    # --- planning side ---------------------------------------------------
    def __call__(self, req: SortieRequest):
        p = req.p
        dwell_est = np.minimum(self.lam_hat * req.age, p.B_bits) / p.R
        req2 = SortieRequest(pos=req.pos, home=req.home, age=req.age,
                             weight_est=req.weight_est, dwell_est=dwell_est,
                             E_usable=req.E_usable, p=p)
        return self.inner(req2)

    # --- diagnostics ------------------------------------------------------
    def rate_mae(self, true_lam: np.ndarray) -> float:
        """Mean absolute relative error of the rate estimate. Diagnostic only --
        called AFTER a run, never inside planning."""
        return float(np.mean(np.abs(self.lam_hat - true_lam) / true_lam))


def build_dwell_memory_planner(inner, p: DynParams, prior_weight: float = 1.0):
    return DwellMemoryPlanner(inner, p, prior_weight)
