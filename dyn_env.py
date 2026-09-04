"""
dyn_env.py -- Continuous-operation dynamic UAV AoI environment.

Implements the settled problem statement (CONTEXT_60 rev2) and instruments the
C4 criteria (CONTEXT_64 §5).

WHAT THIS IS NOT
----------------
This is NOT a modification of env.py / uav_aoi_solver.py. Those model ONE sortie
with a fixed per-sensor data volume and no wall clock. This models continuous
operation: drones fly repeated sorties, buffers refill between visits, events
fire and decay, and the objective is a time-average over a horizon.

It deliberately exposes the SAME attribute names as Env (pos, wi, tcd, M) so an
existing sortie solver can be dropped into plan_sortie() unchanged -- but the
internals differ, because tcd is no longer a constant per sensor. It is
backlog/R and changes at every visit.

SETTLED DECISIONS IMPLEMENTED (CONTEXT_60 rev2)
----------------------------------------------
  §1.1  continuous operation, no rounds; T is a measurement window
  §1.3  age accounting: A~_i = time since last visit (see §2 of CONTEXT_60)
  §1.4  drop-head eviction -- LOAD-BEARING, see note in SensorField.advance()
  §1.5  buffer size out of objective, in the energy constraint (dwell cap)
  §1.6  abandonment permitted -- no coverage constraint anywhere
  §1.9  fixed 20% reserve
  §1.10 exactly K airborne at all times
  §1.11 dwell cost unknown to the planner before arrival
  §3    events: hotspots, unknown firing, own decay, cleared early by a visit

Energy model follows uav_aoi_solver.Env.e_segment: Pf for flight, Ph for hover.
CONTEXT_64: hover is more expensive than flight, and this drives the law.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

# ==============================================================================
# 1. PARAMETERS
# ==============================================================================


@dataclass
class DynParams:
    """Dynamic-extension parameters. See CONTEXT_63 §1 and CONTEXT_64 §4."""

    # --- field / fleet ---
    M: int = 100                  # sensors
    K: int = 4                    # drones AIRBORNE (held fixed, CONTEXT_60 §1.10)
    L: float = 12_600.0           # field side, m (CONTEXT_64 §4)
    Emax: float = 1.5e6           # total airborne energy, J (>= 1.5M)
    rho: float = 0.20             # reserve fraction (CONTEXT_60 §1.9)

    # --- flight physics (CONTEXT_64 §4: ratio from code, absolutes rescaled) ---
    Pf: float = 300.0             # W, flight power
    Ph: float = 400.0             # W, hover power   (Ph/Pf = 4/3, preserved)
    v: float = 20.0               # m/s

    # --- data (CONTEXT_63 §1) ---
    R: float = 2e6                # bits/s, air-to-ground link
    lam_bits: float = 5e3         # bits/s generated per sensor
    B_bits: float = 5e7           # buffer capacity, bits (~167 min of generation)
    # CONTEXT_65 §4: was 6e6, which saturated in 20 min against 17-67 min revisit
    # intervals. Dwell pinned at B/R, P_bar inert, two-power correction dead.

    # --- events (CONTEXT_60 rev2 §3) ---
    n_hotspots: int = 15          # many small pockets
    hotspot_radius: float = 0.06  # as fraction of L
    tau_e_lo: float = 45 * 60.0   # s, event lifetime lower  (45 min)
    tau_e_hi: float = 90 * 60.0   # s, event lifetime upper  (90 min)
    event_rate_hot: float = 1 / (3600.0)    # per-sensor firing rate inside hotspot
    event_rate_cold: float = 1 / (6 * 3600.0)
    event_gain: float = 5.0       # multiplier on wi while an event is live

    # --- priority baseline ---
    wi_lo: float = 1.0
    wi_hi: float = 10.0

    # --- measurement protocol (CONTEXT_60 §7) ---
    T_horizon: float = 12 * 3600.0   # s, evaluation window
    T_burnin: float = 3 * 3600.0     # s, discarded before measuring

    # ---- derived ----
    @property
    def home(self) -> np.ndarray:
        return np.array([self.L / 2.0, self.L / 2.0])

    @property
    def E_each(self) -> float:
        """Per-drone energy per sortie."""
        return self.Emax / self.K

    @property
    def E_usable(self) -> float:
        """Per-drone energy actually spendable before the reserve bites."""
        return (1.0 - self.rho) * self.E_each

    @property
    def t_c(self) -> float:
        """Expected out-and-back commute time. 0.7652 = 2 x mean centre->uniform
        distance in a square (CONTEXT_62 §2). Survives static->dynamic."""
        return 0.7652 * self.L / self.v

    def e_fly(self, d: float) -> float:
        return self.Pf * d / self.v

    def e_hover(self, t: float) -> float:
        return self.Ph * t

    # ---- a priori predictions, for comparison against measurement ----
    def kstar_predicted(self, P_bar: float) -> float:
        """CONTEXT_64 §2. P_bar must be MEASURED, not assumed."""
        return (1.0 - self.rho) * self.Emax / (self.t_c * (self.Pf + math.sqrt(self.Pf * P_bar)))

    def kstar_band(self) -> tuple:
        """(lo, hi) over the full range P_bar in [Pf, Ph]. CONTEXT_64 §3.
        The width of this band IS the predicted M-drift."""
        return (self.kstar_predicted(self.Ph), self.kstar_predicted(self.Pf))


# ==============================================================================
# 2. SENSOR FIELD
# ==============================================================================


class SensorField:
    """
    Sensors, buffers, and events. Owns the age accounting.

    AGE ACCOUNTING (CONTEXT_60 §2) -- the non-obvious part:

      Node cost is A~_i = time since last visit, ALWAYS, saturated or not.
      Under drop-head, observed age pins at tau_i = B/lambda while dropped
      packets accumulate at exactly lambda, so the two terms hand off cleanly:
          tau_i + (t - t_last - tau_i) = t - t_last
      Buffer size and arrival rate both cancel from the objective.

      Therefore this class does NOT need to track dropped packets to compute
      the objective -- t - t_last is sufficient and exact. It tracks backlog
      only because backlog drives DWELL TIME, which is an energy cost.

      This is provable, not assumed. Do not "fix" it by adding a drop penalty:
      that double-charges (CONTEXT_60 §4).
    """

    def __init__(self, p: DynParams, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        M = p.M

        # --- hotspot centres, then sensors clustered around them ---
        self.hotspots = rng.uniform(0.15, 0.85, (p.n_hotspots, 2)) * p.L
        self.in_hotspot = np.zeros(M, dtype=bool)
        pos = np.empty((M, 2))
        # 70% of sensors sit in hotspots, 30% scattered -- keeps the field
        # covered while giving the policy structure to learn.
        n_hot = int(0.70 * M)
        for i in range(M):
            if i < n_hot:
                c = self.hotspots[rng.integers(0, p.n_hotspots)]
                off = rng.normal(0.0, p.hotspot_radius * p.L, 2)
                pos[i] = np.clip(c + off, 0.0, p.L)
                self.in_hotspot[i] = True
            else:
                pos[i] = rng.uniform(0.0, p.L, 2)
        self.pos = pos.astype(np.float64)

        # --- baseline priority ---
        self.wi_base = rng.uniform(p.wi_lo, p.wi_hi, M)

        # --- per-sensor generation rate (heterogeneous, UNKNOWN to planner) ---
        self.lam_bits = p.lam_bits * rng.uniform(0.5, 1.5, M)

        # --- dynamic state ---
        self.t_last_visit = np.zeros(M)      # wall-clock of last visit
        self.backlog = np.zeros(M)           # bits held (capped at B)
        self.event_until = np.full(M, -np.inf)   # event live while t < this
        self.next_event_at = np.empty(M)
        for i in range(M):
            self.next_event_at[i] = self._draw_next_event(0.0, i)

        # --- diagnostics ---
        self.events_fired = 0
        self.events_caught = 0
        self.events_expired = 0
        self.bits_dropped = 0.0

    # -- events ----------------------------------------------------------
    def _rate(self, i: int) -> float:
        return self.p.event_rate_hot if self.in_hotspot[i] else self.p.event_rate_cold

    def _draw_next_event(self, t: float, i: int) -> float:
        return t + self.rng.exponential(1.0 / self._rate(i))

    def advance(self, t_from: float, t_to: float) -> None:
        """Advance world state. Buffers fill; events fire and expire."""
        dt = t_to - t_from
        if dt <= 0:
            return

        # buffers fill, drop-head clamps at capacity.
        # We record dropped bits for DIAGNOSTICS ONLY -- see class docstring.
        grown = self.backlog + self.lam_bits * dt
        over = np.maximum(0.0, grown - self.p.B_bits)
        self.bits_dropped += float(over.sum())
        self.backlog = np.minimum(grown, self.p.B_bits)

        # events fire / expire
        for i in np.where(self.next_event_at <= t_to)[0]:
            fire_t = self.next_event_at[i]
            if fire_t < t_from:
                fire_t = t_from
            tau = self.rng.uniform(self.p.tau_e_lo, self.p.tau_e_hi)
            self.event_until[i] = fire_t + tau
            self.events_fired += 1
            self.next_event_at[i] = self._draw_next_event(fire_t, i)

    def expire_check(self, t: float) -> None:
        """Count events that died unvisited. Called once per sortie boundary."""
        dead = (self.event_until > -np.inf) & (self.event_until <= t)
        n = int(dead.sum())
        if n:
            self.events_expired += n
            self.event_until[dead] = -np.inf

    # -- observables -----------------------------------------------------
    def age(self, t: float) -> np.ndarray:
        """A~_i = time since last visit. Exact under overflow (see docstring)."""
        return t - self.t_last_visit

    def weights(self, t: float) -> np.ndarray:
        """Effective priority: baseline, boosted while an event is live."""
        w = self.wi_base.copy()
        live = self.event_until > t
        w[live] *= self.p.event_gain
        return w

    def dwell_time(self, i: int) -> float:
        """Hover time to drain sensor i. UNKNOWN to the planner before arrival
        (CONTEXT_60 §1.11) -- planners must use an estimate, not this."""
        return self.backlog[i] / self.p.R

    def visit(self, t: float, i: int) -> float:
        """Service sensor i at time t. Returns actual dwell time."""
        td = self.dwell_time(i)
        self.backlog[i] = 0.0
        self.t_last_visit[i] = t
        if self.event_until[i] > t:      # event cleared early (CONTEXT_60 §3)
            self.events_caught += 1
            self.event_until[i] = -np.inf
        return td

    # -- interface compatibility with Env (for drop-in sortie solvers) ----
    @property
    def M(self) -> int:
        return self.p.M

    @property
    def wi(self) -> np.ndarray:
        return self.wi_base


# ==============================================================================
# 3. SORTIE PLANNING INTERFACE
# ==============================================================================


@dataclass
class SortieRequest:
    """Everything a planner may legitimately see at launch.

    Note what is ABSENT: true backlog, true generation rates, event state at
    unvisited nodes, and remaining horizon (CONTEXT_60 §7 -- a planner that sees
    the clock learns to slack off near T).
    """
    pos: np.ndarray            # (M,2) sensor positions
    home: np.ndarray           # depot
    age: np.ndarray            # (M,) time since last visit -- fully observable
    weight_est: np.ndarray     # (M,) baseline priority (event state NOT visible)
    dwell_est: np.ndarray      # (M,) ESTIMATED dwell, from age (not truth)
    E_usable: float            # spendable energy this sortie
    p: DynParams


SortiePlanner = Callable[[SortieRequest], List[int]]


def greedy_ratio_planner(req: SortieRequest) -> List[int]:
    """
    STUB PLANNER -- weighted-age-per-unit-energy greedy with reserve check.

    Rationale for this specific rule:
      * It is the natural dynamic lift of the static `pdr` baseline already in
        multi_uav_solver.fleet_baseline, so it is comparable to banked work.
      * It scores w_i * age_i / (marginal energy), i.e. value per joule -- the
        right currency when energy is the binding constraint.
      * It is myopic and event-blind, which makes it the correct C3 floor:
        RL must beat it on anticipation, not on arithmetic.

    Replace with SA (replan-each-sortie) for C4. Interface is stable.
    """
    p = req.p
    chosen: List[int] = []
    cur = req.home.copy()
    E = req.E_usable
    used = np.zeros(len(req.age), dtype=bool)

    while True:
        d_to = np.linalg.norm(req.pos - cur, axis=1)
        d_home = np.linalg.norm(req.pos - req.home, axis=1)
        e_need = p.e_fly(d_to + d_home) + p.e_hover(req.dwell_est)
        feasible = (~used) & (e_need <= E)
        if not feasible.any():
            break

        e_marginal = np.maximum(p.e_fly(d_to) + p.e_hover(req.dwell_est), 1.0)
        score = np.where(feasible, req.weight_est * req.age / e_marginal, -np.inf)
        j = int(np.argmax(score))

        E -= p.e_fly(d_to[j]) + p.e_hover(req.dwell_est[j])
        cur = req.pos[j].copy()
        used[j] = True
        chosen.append(j)

    return chosen


# ==============================================================================
# 4. CONTINUOUS-OPERATION SIMULATOR
# ==============================================================================


@dataclass
class SortieRecord:
    t_launch: float
    t_land: float
    n_visited: int
    commute_time: float
    travel_time: float      # in-field, excluding commute
    dwell_time: float
    energy_used: float
    tour_len: float         # in-field path length (for the exponent test)


class DynSim:
    """
    Exactly K drones airborne at all times (CONTEXT_60 §1.10). A drone that
    lands is replaced immediately by a charged one, so ground time and the
    airframe pool are outside the model -- deferred extension 9.2.

    Implementation -- REWRITTEN (CONTEXT_67): a single GLOBAL discrete-event
    loop. Every drone's sortie is decomposed into individual leg-completion
    events (fly-to-node, visit, fly-home, land) pushed onto ONE shared heap
    keyed on absolute time. The shared field is advanced and queried only at
    the true global minimum time on each pop.

    WHY THE REWRITE. The previous version executed each sortie ATOMICALLY using
    a local time variable that ran ahead of the shared clock. If drone A's
    sortie took long enough to run past drone B's already-scheduled launch,
    A's visits were stamped with timestamps LATER than B's subsequent launch
    time -- so B, planning "now", could see itself at t=150 while a node's
    last-visit was stamped t=160 by A, giving negative age. Confirmed
    numerically (age = -10 in the minimal repro) and confirmed to affect every
    prior dynamic result in this session, stub and SA alike -- the stub never
    crashed on it only because argmax over a signed score doesn't validate
    sign. This version cannot exhibit that failure mode: the field is only
    ever touched at the monotonically increasing sequence of popped event
    times, so no drone can act on, or write, a timestamp another drone hasn't
    reached yet.
    """

    def __init__(self, p: DynParams, planner: SortiePlanner = greedy_ratio_planner,
                 seed: int = 0):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.field = SensorField(p, self.rng)
        self.planner = planner
        self.records: List[SortieRecord] = []
        self._age_integral = 0.0
        self._measure_time = 0.0
        self._empty_sorties = 0
        self._truncated_sorties = 0   # CONTEXT_69: reserve breach mid-route
        self._clock = 0.0          # global simulation clock -- monotonic
        self._seq = 0               # heap tiebreaker

    # -- age integration -------------------------------------------------
    def _accumulate(self, t0: float, t1: float) -> None:
        """Integrate weighted age over [t0, t1], counting only post-burn-in time."""
        t0 = max(t0, self.p.T_burnin)
        if t1 <= t0:
            return
        w = self.field.weights(t0)
        a0 = self.field.age(t0)
        a1 = self.field.age(t1)
        self._age_integral += float((w * 0.5 * (a0 + a1)).sum()) * (t1 - t0)
        self._measure_time += (t1 - t0)

    def _advance_global(self, t_to: float) -> None:
        """Advance the shared world and the objective from the current global
        clock up to t_to, then move the clock. The ONLY place either happens."""
        if t_to <= self._clock:
            return
        self.field.advance(self._clock, t_to)
        self._accumulate(self._clock, t_to)
        self._clock = t_to

    # -- per-drone state --------------------------------------------------
    class _Drone:
        __slots__ = ("route", "ri", "pos", "E", "t_launch", "commute", "travel",
                     "dwell", "tour_len", "n_visited", "phase")

    def _launch(self, k: int, t: float) -> "_Drone":
        p, F = self.p, self.field
        dwell_est = np.minimum(F.age(t) * p.lam_bits, p.B_bits) / p.R
        req = SortieRequest(pos=F.pos, home=p.home, age=F.age(t),
                             weight_est=F.wi_base.copy(), dwell_est=dwell_est,
                             E_usable=p.E_usable, p=p)
        route = self.planner(req)
        d = self._Drone()
        d.route, d.ri = route, 0
        d.pos = p.home.copy()
        d.E = p.E_usable
        d.t_launch = t
        d.commute = d.travel = d.dwell = d.tour_len = 0.0
        d.n_visited = 0
        d.phase = "flying"
        return d

    def _next_event_time(self, d: "_Drone") -> tuple:
        """Time of this drone's next action, and what that action is."""
        p, F = self.p, self.field
        if d.ri < len(d.route):
            j = d.route[d.ri]
            dist = float(np.linalg.norm(F.pos[j] - d.pos))
            return self._clock + dist / p.v, ("arrive_node", j, dist)
        else:
            dist = float(np.linalg.norm(p.home - d.pos))
            return self._clock + dist / p.v, ("arrive_home", None, dist)

    # -- main loop ---------------------------------------------------------
    def run(self) -> dict:
        p = self.p
        drones: dict = {}
        heap: list = []

        for k in range(p.K):
            t0 = k * p.t_c / max(p.K, 1)
            heapq.heappush(heap, (t0, self._seq, "launch", k))
            self._seq += 1

        while heap:
            t_event, _, kind, k = heapq.heappop(heap)
            if kind == "launch":
                if t_event >= p.T_horizon:
                    continue
                self._advance_global(t_event)
                d = self._launch(k, t_event)
                drones[k] = d
                t_next, action = self._next_event_time(d)
                heapq.heappush(heap, (t_next, self._seq, action, k))
                self._seq += 1
                continue

            # action tuple was stashed as `kind` above for non-launch events
            action = kind
            d = drones[k]
            self._advance_global(t_event)  # world catches up to THIS event, globally

            if action[0] == "arrive_node":
                j = action[1]
                td_true = self.field.dwell_time(j)
                d_home = float(np.linalg.norm(self.field.pos[j] - p.home))
                dist_leg = action[2]
                e_step = p.e_fly(dist_leg) + p.e_hover(td_true)
                if e_step + p.e_fly(d_home) > d.E:
                    # reserve breached -- do not serve this node, head home instead
                    if d.ri < len(d.route) - 1:
                        # nodes remained in the PLANNED route beyond this one:
                        # a genuine mid-route truncation (CONTEXT_69), not just
                        # the route's last stop happening to be tight.
                        self._truncated_sorties += 1
                    d.ri = len(d.route)  # force "arrive_home" branch next
                    t_next, nxt = self._next_event_time(d)
                    heapq.heappush(heap, (t_next, self._seq, nxt, k))
                    self._seq += 1
                    continue

                if d.n_visited == 0:
                    d.commute += dist_leg / p.v
                else:
                    d.travel += dist_leg / p.v
                    d.tour_len += dist_leg

                td = self.field.visit(t_event, j)
                self._advance_global(t_event + td)   # dwell consumes time too
                d.dwell += td
                d.E -= e_step
                d.pos = self.field.pos[j].copy()
                d.n_visited += 1
                d.ri += 1

                t_next, nxt = self._next_event_time(d)
                heapq.heappush(heap, (t_next, self._seq, nxt, k))
                self._seq += 1

            else:  # arrive_home -> land, record, relaunch
                dist_leg = action[2]
                d.commute += dist_leg / p.v
                d.E -= p.e_fly(dist_leg)
                self.field.expire_check(t_event)

                rec = SortieRecord(
                    t_launch=d.t_launch, t_land=t_event, n_visited=d.n_visited,
                    commute_time=d.commute, travel_time=d.travel,
                    dwell_time=d.dwell, energy_used=p.E_usable - d.E,
                    tour_len=d.tour_len,
                )
                self.records.append(rec)

                t_next_launch = t_event
                if d.n_visited == 0:
                    # empty sortie landed instantly -- force a turnaround so we
                    # do not spin (CONTEXT_65's original guard, same rationale)
                    self._empty_sorties += 1
                    t_next_launch = t_event + p.t_c
                heapq.heappush(heap, (t_next_launch, self._seq, "launch", k))
                self._seq += 1

        return self.metrics()

    # -- C4 instrumentation ---------------------------------------------
    def metrics(self) -> dict:
        p, F = self.p, self.field
        recs = [r for r in self.records if r.t_launch >= p.T_burnin]
        if not recs:
            return {"error": "no post-burn-in sorties"}

        n_vis = np.array([r.n_visited for r in recs], dtype=float)
        commute = np.array([r.commute_time for r in recs])
        travel = np.array([r.travel_time for r in recs])
        dwell = np.array([r.dwell_time for r in recs])
        T_s = np.array([r.t_land - r.t_launch for r in recs])

        # P_bar: time-weighted mean power during PRODUCTIVE work (CONTEXT_64 §2)
        prod_t = travel + dwell
        P_bar = float((p.Pf * travel.sum() + p.Ph * dwell.sum()) / max(prod_t.sum(), 1e-9))

        total_visits = n_vis.sum()
        span = max(recs[-1].t_land - recs[0].t_launch, 1e-9)
        T_rev = p.M * span / max(total_visits, 1e-9)

        tau_e_mid = 0.5 * (p.tau_e_lo + p.tau_e_hi)
        gen = p.M * p.lam_bits
        cap = p.K * p.R * (1.0 - commute.mean() / max(T_s.mean(), 1e-9))

        lo, hi = p.kstar_band()
        return {
            # --- objective ---
            "J_timeavg": self._age_integral / max(self._measure_time, 1e-9),
            "measured_s": self._measure_time,
            # --- crit 6: T_s/t_c, predicted 1 + sqrt(P_bar/Pf) in [2, 2.16] ---
            "T_s_over_t_c": float(T_s.mean() / p.t_c),
            "T_s_over_t_c_pred": 1.0 + math.sqrt(p.Pf / P_bar),
            "commute_measured": float(commute.mean()),
            "commute_over_tc": float(commute.mean() / p.t_c),
            # --- crit 3: P_bar, must lie in [Pf, Ph] ---
            "P_bar": P_bar,
            "P_bar_frac": (P_bar - p.Pf) / (p.Ph - p.Pf),
            # --- crit 1: e* ---
            "e_star_measured": float(np.mean([r.energy_used for r in recs])),
            "e_star_pred": p.t_c * (p.Pf + math.sqrt(p.Pf * P_bar)),
            "kstar_pred": p.kstar_predicted(P_bar),
            "kstar_band": (lo, hi),
            # --- crit 8: throughput (CONTEXT_63 §4) ---
            "throughput_margin": gen / max(cap, 1e-9),
            "throughput_ok": bool(gen < cap),
            # --- crit 9: event regime ---
            "T_rev": T_rev,
            "T_rev_over_tau_e": T_rev / tau_e_mid,
            "events_fired": F.events_fired,
            "events_caught": F.events_caught,
            "catch_rate": F.events_caught / max(F.events_fired, 1),
            # --- crit 7: tour exponent (regress offline over cells) ---
            "mean_n_visited": float(n_vis.mean()),
            "mean_tour_len": float(np.mean([r.tour_len for r in recs])),
            # --- sanity ---
            "n_sorties": len(recs),
            "commute_frac": float(commute.sum() / max(T_s.sum(), 1e-9)),
            "bits_dropped": F.bits_dropped,
            "empty_sorties": self._empty_sorties,
            "truncated_sorties": self._truncated_sorties,
            "n_sorties_total": len(self.records),
        }


# ==============================================================================
# 5. SMOKE TEST
# ==============================================================================

if __name__ == "__main__":
    p = DynParams()
    print(f"t_c = {p.t_c:.1f} s   E_usable/drone = {p.E_usable:,.0f} J")
    print(f"K* band over P_bar in [Pf,Ph]: {p.kstar_band()[0]:.2f} .. {p.kstar_band()[1]:.2f}")
    print(f"predicted drift = {100*(p.kstar_band()[1]/p.kstar_band()[0]-1):.1f}%  (CONTEXT_64 §3: 7.7%)")
    print()
    for K in (2, 3, 4, 5, 6, 8):
        p_k = DynParams(K=K)
        m = DynSim(p_k, seed=1).run()
        print(f"K={K}  J={m['J_timeavg']:11.1f}  T_s/t_c={m['T_s_over_t_c']:.2f} "
              f"(pred {m['T_s_over_t_c_pred']:.2f})  P_bar={m['P_bar']:.0f}  "
              f"n={m['mean_n_visited']:.1f}  T_rev={m['T_rev']/60:.0f}m  "
              f"catch={m['catch_rate']:.2f}  thru={m['throughput_margin']:.2f}")