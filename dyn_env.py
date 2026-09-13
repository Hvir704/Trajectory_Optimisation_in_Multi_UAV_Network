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
    event_gain: float = 5.0       # "boost": multiplier on wi while live; "separate": coefficient g on event age
    event_model: str = "separate" # "separate": J = sum w*A~ + sum g*w*(t - t_fire) over live uncaught events (event AoI)
                                  # "boost":    legacy -- w -> g*w while live, charged against buffer age

    # --- planner information (W1 additions) ---
    coord_mode: str = "exclude"   # "none" | "exclude" | "project" (ETA-aware projected age)
    planner_knows_rates: bool = False   # if True, p_live from known hotspot map/rates
    learn_lambda: bool = True           # per-node lambda estimate from visits
    layout: str = "paper"         # "paper" | "ring" | "core"
    replan_mode: str = "launch"   # "launch": route fixed at launch | "per_leg": re-optimise remaining
                                  # route after every stop from current position/energy; the updated
                                  # commitment is visible to all other drones immediately
                                  # (continuous inter-drone communication).
    replan_iters: int = 300       # SA iterations per mid-sortie replan (launch uses the planner's own)
    replan_planner: str = "same"  # "same": the sortie planner with replan_iters | "greedy": fast ratio rule

    # --- v3: latent correlated hotspot events (PROBLEM_STATEMENT_v3 §2-3) ---
    event_corr_q: float = 0.0     # 0: independent per-sensor events (v2 model). >0: each hotspot fires
                                  # as a unit at rate event_rate_hot/q; every member goes live w.p. q,
                                  # so the per-sensor event rate is unchanged and only correlation varies.
    belief_mode: str = "prior"    # planner-side P(event live): "none" | "prior" (known rates, M2)
                                  # | "kernel" (spatio-temporal kernel over observations, M1)
    belief_sigma_frac: float = 0.06   # kernel width as a fraction of L (= hotspot_radius)

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
        self.hot_id = np.full(M, -1)
        pos = np.empty((M, 2))
        # 70% of sensors sit in hotspots, 30% scattered -- keeps the field
        # covered while giving the policy structure to learn.
        n_hot = int(0.70 * M)
        for i in range(M):
            if i < n_hot:
                h = int(rng.integers(0, p.n_hotspots))
                c = self.hotspots[h]
                off = rng.normal(0.0, p.hotspot_radius * p.L, 2)
                pos[i] = np.clip(c + off, 0.0, p.L)
                self.in_hotspot[i] = True; self.hot_id[i] = h
            else:
                pos[i] = rng.uniform(0.0, p.L, 2)
        if p.layout == "ring":
            r = rng.uniform(0.80, 1.0, M) * (0.45 * p.L * np.sqrt(2))
            th = rng.uniform(0, 2 * np.pi, M)
            pos = np.c_[p.L/2 + r*np.cos(th), p.L/2 + r*np.sin(th)]
            self.in_hotspot[:] = False; self.hot_id[:] = -1
        elif p.layout == "core":
            r = rng.uniform(0.02, 0.20, M) * p.L
            r[0] = 0.45 * p.L * np.sqrt(2)          # one peripheral outlier sets r_max
            th = rng.uniform(0, 2 * np.pi, M)
            pos = np.c_[p.L/2 + r*np.cos(th), p.L/2 + r*np.sin(th)]
            self.in_hotspot[:] = False; self.hot_id[:] = -1
        self.pos = np.clip(pos, 0.0, p.L).astype(np.float64)
        # per-node lambda estimate (planner-side); nominal until first visit
        self.lam_est = np.full(M, p.lam_bits)
        self.n_obs = np.zeros(M, dtype=int)

        # --- baseline priority ---
        self.wi_base = rng.uniform(p.wi_lo, p.wi_hi, M)

        # --- per-sensor generation rate (heterogeneous, UNKNOWN to planner) ---
        self.lam_bits = p.lam_bits * rng.uniform(0.5, 1.5, M)

        # --- dynamic state ---
        self.t_last_visit = np.zeros(M)      # wall-clock of last visit
        self.backlog = np.zeros(M)           # bits held (capped at B)
        self.event_until = np.full(M, -np.inf)   # event live while t < this
        self.event_fire = np.full(M, np.inf)     # fire time of the current live event
        self.next_event_at = np.empty(M)
        for i in range(M):
            self.next_event_at[i] = self._draw_next_event(0.0, i)
        # v3: hotspot-level firing (members fire together w.p. q). Hotspot members' own
        # per-sensor clocks are disabled when q > 0.
        if p.event_corr_q > 0:
            self.next_event_at[self.hot_id >= 0] = np.inf
            self.next_hot_fire = np.array([self.rng.exponential(p.event_corr_q / p.event_rate_hot)
                                           for _ in range(p.n_hotspots)])
        # observations: what drones have SEEN (event state on arrival). Planner-side only.
        self.obs_t = np.full(M, -np.inf)     # time of last observation of node i
        self.obs_live = np.zeros(M, dtype=bool)

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
            self.event_fire[i] = fire_t
            self.events_fired += 1
            self.next_event_at[i] = self._draw_next_event(fire_t, i)
        if self.p.event_corr_q > 0:
            for h in np.where(self.next_hot_fire <= t_to)[0]:
                fire_t = max(self.next_hot_fire[h], t_from)
                members = np.where(self.hot_id == h)[0]
                hit = members[self.rng.random(len(members)) < self.p.event_corr_q]
                for i in hit:
                    self.event_until[i] = fire_t + self.rng.uniform(self.p.tau_e_lo, self.p.tau_e_hi)
                    self.event_fire[i] = fire_t
                    self.events_fired += 1
                self.next_hot_fire[h] = fire_t + self.rng.exponential(self.p.event_corr_q / self.p.event_rate_hot)

    def expire_check(self, t: float) -> None:
        """Count events that died unvisited. Called once per sortie boundary."""
        dead = (self.event_until > -np.inf) & (self.event_until <= t)
        n = int(dead.sum())
        if n:
            self.events_expired += n
            self.event_until[dead] = -np.inf
            self.event_fire[dead] = np.inf

    # -- observables -----------------------------------------------------
    def age(self, t: float) -> np.ndarray:
        """A~_i = time since last visit. Exact under overflow (see docstring)."""
        return t - self.t_last_visit

    def weights(self, t: float) -> np.ndarray:
        """Effective priority: baseline, boosted while an event is live."""
        w = self.wi_base.copy()
        if self.p.event_model == "boost":
            live = self.event_until > t
            w[live] *= self.p.event_gain
        return w

    def event_age_integral(self, t0: float, t1: float) -> np.ndarray:
        """Per-node integral over [t0,t1] of g*w*(t - t_fire) for the currently live
        event, exact: integrates over [max(t0,fire), min(t1,until)]."""
        a = np.maximum(t0, self.event_fire); b = np.minimum(t1, self.event_until)
        live = b > a
        out = np.zeros(self.p.M)
        f = self.event_fire[live]
        out[live] = self.p.event_gain * self.wi_base[live] * 0.5 * ((b[live]-f)**2 - (a[live]-f)**2)
        return out

    def dwell_time(self, i: int) -> float:
        """Hover time to drain sensor i. UNKNOWN to the planner before arrival
        (CONTEXT_60 §1.11) -- planners must use an estimate, not this."""
        return self.backlog[i] / self.p.R

    def p_live_prior(self) -> np.ndarray:
        """Stationary P(event live) from known rates: nu*tau/(1+nu*tau)."""
        tau = 0.5 * (self.p.tau_e_lo + self.p.tau_e_hi)
        nu = np.where(self.in_hotspot, self.p.event_rate_hot, self.p.event_rate_cold)
        x = nu * tau
        return x / (1.0 + x)

    def belief(self, t: float) -> np.ndarray:
        """Planner-side P(event live at i | observations). Kernel model:
        a positive observation at j at time t_o raises the belief of every i within the
        hotspot scale, decayed over the mean event lifetime; the observed node itself is
        0 right after a visit (event caught/cleared) and relaxes to the prior."""
        p = self.p
        prior = self.p_live_prior()
        if p.belief_mode == "none": return None
        if p.belief_mode == "prior": return prior
        tau = 0.5 * (p.tau_e_lo + p.tau_e_hi); sig = p.belief_sigma_frac * p.L
        b = prior.copy()
        recent = np.where((t - self.obs_t) < tau)[0]
        pos_obs = recent[self.obs_live[recent]]
        if len(pos_obs):
            d2 = ((self.pos[:, None, :] - self.pos[None, pos_obs, :]) ** 2).sum(-1)
            k = np.exp(-d2 / (2 * sig ** 2)) * np.exp(-(t - self.obs_t[pos_obs])[None, :] / tau)
            b = np.maximum(b, max(p.event_corr_q, 1e-3) * k.max(1))
        # a node observed recently is known: caught events are cleared, so belief -> 0, relaxing to prior
        b[recent] = np.minimum(b[recent], prior[recent] * (t - self.obs_t[recent]) / tau)
        return np.clip(b, 0.0, 1.0)

    def visit(self, t: float, i: int) -> float:
        """Service sensor i at time t. Returns actual dwell time."""
        td = self.dwell_time(i)
        self.obs_t[i] = t; self.obs_live[i] = bool(self.event_until[i] > t)
        age = t - self.t_last_visit[i]
        if age > 0 and self.backlog[i] < self.p.B_bits - 1e-6:
            # backlog/age reveals lambda exactly when the buffer has not capped
            self.lam_est[i] = self.backlog[i] / age; self.n_obs[i] += 1
        self.backlog[i] = 0.0
        self.t_last_visit[i] = t
        if self.event_until[i] > t:      # event cleared early (CONTEXT_60 §3)
            self.events_caught += 1
            self.event_until[i] = -np.inf
            self.event_fire[i] = np.inf
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
    start: Optional[np.ndarray] = None      # planning start position (None -> home); route always ends at home
    E_usable_full: float = 0.0              # full sortie budget (E_usable is REMAINING when replanning)
    lam_est: Optional[np.ndarray] = None    # (M,) per-node generation-rate estimate (from visits)
    p_live: Optional[np.ndarray] = None     # (M,) prior P(event live) from known rates, or None
    eta_other: Optional[np.ndarray] = None  # (M,) ETA of another drone at node (inf if none)
    excluded: Optional[np.ndarray] = None   # (M,) bool: nodes pending in ANOTHER
    # airborne drone's route. Legitimate information (the operator knows its own
    # fleet's plans). Without it, 57-69% of visits land on a node another drone
    # reset after this one launched (measured, M=100, K=3-6, seed 1).


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
    cur = (req.home if req.start is None else req.start).copy()
    E = req.E_usable
    used = np.zeros(len(req.age), dtype=bool)
    if req.excluded is not None:
        used |= req.excluded

    while True:
        d_to = np.linalg.norm(req.pos - cur, axis=1)
        d_home = np.linalg.norm(req.pos - req.home, axis=1)
        e_need = p.e_fly(d_to + d_home) + p.e_hover(req.dwell_est)
        feasible = (~used) & (e_need <= E)
        if not feasible.any():
            break

        e_marginal = np.maximum(p.e_fly(d_to) + p.e_hover(req.dwell_est), 1.0)
        w_eff = req.weight_est if req.p_live is None else req.weight_est * (1 + (p.event_gain - 1) * req.p_live)
        score = np.where(feasible, w_eff * req.age / e_marginal, -np.inf)
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
                 seed: int = 0, coordinate: bool = True):
        self.p = p
        self.coordinate = coordinate and p.coord_mode != "none"
        self._live: dict = {}          # k -> _Drone currently airborne
        self._n_vis_post = 0           # post-burn-in visits
        self._n_dup_post = 0           # ...of which node was reset by another drone after launch
        self._age_at_visit: list = []  # post-burn-in age at arrival (diagnostic)
        self._node_int = np.zeros(p.M) # per-node integral of w*age (+event term) post-burn-in
        self._event_integral = 0.0
        self._n_replans = 0
        self._Ts_mean = None            # running mean sortie duration (for dwell-at-arrival)
        self._visits = np.zeros(p.M, dtype=int)
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
        t1 = min(t1, self.p.T_horizon)   # FIX: was unclamped; the tail after
        # T_horizon (drones stop relaunching one by one) was being measured.
        if t1 <= t0:
            return
        w = self.field.weights(t0)
        a0 = self.field.age(t0)
        a1 = self.field.age(t1)
        contrib = w * 0.5 * (a0 + a1) * (t1 - t0)
        if self.p.event_model == "separate":
            ev = self.field.event_age_integral(t0, t1)
            self._event_integral += float(ev.sum())
            contrib = contrib + ev
        self._node_int += contrib
        self._age_integral += float(contrib.sum())
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

    def _plan(self, k: int, t: float, start, E_rem: float, elapsed_frac: float, iters=None, own_done=()):
        """Build a SortieRequest from the CURRENT fleet state and return a route.
        start/E_rem: where the drone is and what it can still spend (return to home included)."""
        p, F = self.p, self.field
        lam = F.lam_est if p.learn_lambda else np.full(p.M, p.lam_bits)
        age = F.age(t)
        # dwell at EXPECTED ARRIVAL, not at decision time: the buffer fills during the flight.
        # Expected elapsed time = elapsed_frac x the fleet's running mean sortie duration
        # (t_c until one sortie has landed). Using t_c alone under-budgeted long sorties:
        # at K=2 (T_s ~ 4 t_c) 84% of sorties truncated, biasing J against low K.
        Ts = self._Ts_mean if self._Ts_mean is not None else p.t_c
        dwell_est = np.minimum((age + elapsed_frac * max(Ts, p.t_c)) * lam, p.B_bits) / p.R
        excluded = None; eta = None
        if self.coordinate:
            eta = np.full(p.M, np.inf)
            for kk, other in self._live.items():
                if kk == k: continue
                tt, cur = self._clock, other.pos
                for j in other.route[other.ri:]:
                    tt += float(np.linalg.norm(F.pos[j] - cur)) / p.v; cur = F.pos[j]
                    tt += np.minimum(age[j] * lam[j], p.B_bits) / p.R
                    eta[j] = min(eta[j], tt)
            if p.coord_mode == "exclude":
                excluded = np.isfinite(eta)
            else:
                proj = np.where(np.isfinite(eta), np.maximum(0.0, (t + elapsed_frac * p.t_c) - eta), age)
                excluded = np.isfinite(eta) & (proj <= 0.0)
                age = np.where(np.isfinite(eta), proj, age)
        if own_done:
            # never re-serve a node within the same sortie: its age is ~0 and its marginal
            # cost is ~0, so a ratio rule can otherwise loop on it indefinitely
            if excluded is None: excluded = np.zeros(p.M, dtype=bool)
            excluded = excluded.copy(); excluded[list(own_done)] = True
        p_live = F.belief(t) if (p.planner_knows_rates or p.belief_mode != "prior") else None
        req = SortieRequest(pos=F.pos, home=p.home, age=age,
                             weight_est=F.wi_base.copy(), dwell_est=dwell_est,
                             E_usable=E_rem, p=p, start=start, E_usable_full=p.E_usable,
                             lam_est=lam, p_live=p_live, eta_other=eta, excluded=excluded)
        if iters is not None:
            if p.replan_planner == "greedy":
                return greedy_ratio_planner(req)
            if hasattr(self.planner, "with_iters"):
                return self.planner.with_iters(iters)(req)
        return self.planner(req)

    def state_features(self, k: int, t: float, start, E_rem: float):
        """Per-node feature matrix for a learned policy (M3). Same information as _plan.
        cols: age, w, belief, lam_est, dist_from_drone, dist_to_home, excluded, eta_other-t"""
        p, F = self.p, self.field
        start = p.home if start is None else start
        age = F.age(t); b = F.belief(t); b = np.zeros(p.M) if b is None else b
        excluded = np.zeros(p.M, dtype=bool); eta = np.full(p.M, np.inf)
        for kk, o in self._live.items():
            if kk == k: continue
            excluded[o.route[o.ri:]] = True
        X = np.c_[age, F.wi_base, b, F.lam_est, np.linalg.norm(F.pos - start, axis=1),
                  np.linalg.norm(F.pos - p.home, axis=1), excluded.astype(float)]
        return X, np.array([E_rem / p.E_usable, *(start / p.L)])

    def _launch(self, k: int, t: float) -> "_Drone":
        p = self.p
        d = self._Drone()
        d.pos = p.home.copy(); d.E = p.E_usable; d.t_launch = t
        d.commute = d.travel = d.dwell = d.tour_len = 0.0
        d.n_visited = 0; d.phase = "flying"; d.route, d.ri = [], 0
        self._live[k] = d
        d.route = self._plan(k, t, None, p.E_usable, 0.5)
        return d

    def _replan(self, k: int, d: "_Drone", t: float):
        """Per-leg replanning: re-optimise the remaining route from the current node with the
        remaining energy, against the other drones' CURRENT commitments. The new route replaces
        the old commitment in self._live immediately (continuous communication)."""
        p = self.p
        done = d.route[:d.ri]
        d.route = done  # own pending commitment cleared while planning
        rest = self._plan(k, t, d.pos.copy(), d.E, 0.15, iters=p.replan_iters, own_done=done)
        d.route = done + list(rest)
        self._n_replans += 1

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

                if t_event >= p.T_burnin:
                    a_arr = t_event - self.field.t_last_visit[j]
                    self._n_vis_post += 1
                    self._age_at_visit.append(a_arr)
                    self._visits[j] += 1
                    if a_arr < t_event - d.t_launch:
                        self._n_dup_post += 1
                td = self.field.visit(t_event, j)
                self._advance_global(t_event + td)   # dwell consumes time too
                d.dwell += td
                d.E -= e_step
                d.pos = self.field.pos[j].copy()
                d.n_visited += 1
                d.ri += 1
                if p.replan_mode == "per_leg":
                    self._replan(k, d, t_event + td)

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
                dur = t_event - d.t_launch
                self._Ts_mean = dur if self._Ts_mean is None else 0.9 * self._Ts_mean + 0.1 * dur

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
        P_bar = max(P_bar, 1e-9)   # empty-fleet guard: no productive time -> no division by zero

        total_visits = n_vis.sum()
        span = max(recs[-1].t_land - recs[0].t_launch, 1e-9)
        T_rev = p.M * span / max(total_visits, 1e-9)

        tau_e_mid = 0.5 * (p.tau_e_lo + p.tau_e_hi)
        gen = p.M * p.lam_bits
        cap = p.K * p.R * (1.0 - commute.mean() / max(T_s.mean(), 1e-9))

        lo, hi = p.kstar_band()
        r = np.linalg.norm(F.pos - p.home, axis=1)
        never = self._visits == 0
        r_max_inst = float(r.max())
        E_reach = p.E_usable - p.Ph * p.B_bits / p.R          # D3: dwell at cap
        reach = p.v * E_reach / (2.0 * p.Pf)
        K_cov_inst = p.v * (1 - p.rho) * p.Emax / (2 * p.Pf * r_max_inst + p.v * p.Ph * p.B_bits / p.R)
        aav = np.array(self._age_at_visit) if self._age_at_visit else np.array([np.nan])
        return {
            # --- coordination / abandonment / reach (added in review) ---
            "dup_frac": self._n_dup_post / max(self._n_vis_post, 1),
            "frac_visits_age_lt_300s": float(np.mean(aav < 300.0)),
            "median_age_at_visit": float(np.median(aav)),
            "n_never_visited": int(never.sum()),
            "share_J_never_visited": float(self._node_int[never].sum() / max(self._node_int.sum(), 1e-9)),
            "n_reach_infeasible": int((r > reach).sum()),
            "n_reach_infeasible_never": int((never & (r > reach)).sum()),
            "r_max_instance": r_max_inst,
            "r_reach": reach,
            "K_cov_instance": K_cov_inst,
            "r_c_measured": float(commute.mean() * p.v / 2.0),
            "K_commute_instance": float((1 - p.rho) * p.Emax / max(commute.mean() * (p.Pf + math.sqrt(p.Pf * P_bar)), 1e-9)),
            "rmax_over_rc": float(r_max_inst / max(commute.mean() * p.v / 2.0, 1e-9)),
            "regime_boundary": 1.0 + math.sqrt(P_bar / p.Pf),
            "empty_frac": self._empty_sorties / max(len(self.records), 1),
            "trunc_frac": self._truncated_sorties / max(len(self.records), 1),
            # --- objective ---
            "J_timeavg": self._age_integral / max(self._measure_time, 1e-9),
            "J_event": self._event_integral / max(self._measure_time, 1e-9),
            "J_age": (self._age_integral - self._event_integral) / max(self._measure_time, 1e-9),
            "event_model": p.event_model,
            "replan_mode": p.replan_mode,
            "event_corr_q": p.event_corr_q, "belief_mode": p.belief_mode,
            "n_replans": self._n_replans,
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