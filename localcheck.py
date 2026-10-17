"""localcheck.py -- the two-candidate local check: a four-run (typically) refinement of the v3.3 / v3.4 rule.

Procedure (needs no theory beyond Theorem 1 and the registered rules):
  1. c1 = the rule's fleet size: theory_v34.surrogate_rule(..., shared=<planner shares the field>), i.e. v3.4 for
     shared-field planners and v3.3 (gray-bracket exclusion, hop d_S) for fixed-territory planners.
  2. c2 = the surrogate's best fleet size at an ADJACENT stranding level: j1 - 1 or j1 + 1 stranded sensors, where
     j1 is c1's (Theorem 1's neighbouring served set; if c1 serves all, c2 is the best size stranding one sensor, and
     vice versa). Of the two adjacent levels the one with the lower surrogate J is used. None if neither level has a
     feasible candidate. (Adjacent, not any other level: a far level, e.g. stranding dozens of sensors at K ~ 120 on a
     clustered field, is never worth flying, and on the seen data it never mattered.)
  3. Simulate (or fly) K in {c-1, c, c+1} for c in {c1, c2} (K >= 1), and keep the K with the lowest measured J.
The union usually has 6 sizes when c1 and c2 are far apart, 3-5 when they are close, 3 when c2 is None.

This module only computes the candidates; it does not modify the registered rules (theory_v3.py, theory_v34.py).
"""
import numpy as np
import theory_v34

SHARED_TESTED = {"sa", "rr_tour"}          # planners whose UAVs share the field AND for which v3.4 has been tested


def candidates(p, F, rc, Pb, key=None, shared=True, horizon=None):
    """(c1, c2): the rule's pick and the best fleet size at another stranding level (c2 may be None)."""
    K, j, curve = theory_v34.surrogate_rule(p, F, rc, Pb, key=key, shared=shared, return_curve=True, horizon=horizon)
    other = [(v[0], k) for k, v in curve.items() if abs(v[1] - j) == 1]
    return int(K), (int(min(other)[1]) if other else None)


def check_set(c1, c2):
    """the fleet sizes the local check evaluates"""
    S = set()
    for c in (c1, c2):
        if c is not None: S |= {k for k in (c - 1, c, c + 1) if k >= 1}
    return sorted(S)
