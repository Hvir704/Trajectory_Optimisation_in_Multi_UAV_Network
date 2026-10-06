"""dyn_slot_env.py -- our operating model in the time-slotted form used by DRL-UCS(AoI_th)
(Wang, Liu, Yang, Wang, Leung, IEEE/ACM ToN 32(1):566-581, 2024).

The DRL baseline acts once per slot (their tau = 20 s): each airborne UAV moves by one action, then
spends the rest of the slot collecting from sensors in range. This module reproduces OUR world in that
form so their algorithm can be trained and evaluated on our objective:

  * same SensorField (positions, priorities, heterogeneous rates, buffers, events) as DynSim for the
    same seed; same objective accounting (DynSim._accumulate, burn-in and horizon clamps) -> J_timeavg,
    J_age, J_event are directly comparable with run_grid output;
  * K UAVs airborne at all times, each sortie with E_usable = (1-rho) U / K, flight power Pf, hover power
    Ph, immediate relaunch on landing, initial launches staggered by t_c / K (as DynSim);
  * a visit = hover-drain of a sensor's whole backlog at R (true dwell = backlog / R, age reset at the
    start of the drain, energy Ph x dwell) -- exactly DynSim's visit; a drain may run past the slot end,
    the UAV is then busy into the next slot;
  * idle time left in a slot is spent hovering (their energy model: hover power for the collection time);
  * RETURN AUTOPILOT (our adaptation; their UAVs have one battery and no depot): at the start of a slot,
    if E - e_fly(home) < 2 Pf tau the UAV flies straight home (its action is ignored and masked), lands,
    and is replaced by a charged UAV at the depot. Drains are only started if the UAV can still pay the
    drain, the slot's idle hover, and the flight home.

Motion: 'ucs' = their released action table (13 discrete actions: stay, 8 directions of 1.5 units,
4 axis moves of v*tau), scaled so 1 unit = UNIT m (theirs: 100 m); 'direct' = straight flight towards
a target point (used only by the scripted controller in the equivalence gate).
Drain rule: 'radius' = all sensors within collect_radius, nearest first, at most update_num per slot
(their rule); 'target' = only the scripted target (gate only, mirrors DynSim's one-node visit).
"""
from __future__ import annotations
import heapq
import math
import numpy as np
from dyn_env import DynParams, SensorField

UNIT = None           # metres per action-table unit: v*tau/4, so the max move is one slot of flight (theirs: 100 m at tau=20 s)


def ucs_action_table(v, tau):
    """Their 13-action table (env_ucs._get_vector_by_action) with 1 unit = v*tau/4: at their tau = 20 s and
    v = 20 m/s this is exactly their 100 m unit (moves of 150 m diagonal/axis and 400 m axis)."""
    unit = v * tau / 4.0
    single = 1.5; base = single / math.sqrt(2); mx = 4.0
    t = [[0, 0], [-base, base], [0, single], [base, base], [-single, 0], [single, 0],
         [-base, -base], [0, -single], [base, -base], [0, mx], [0, -mx], [mx, 0], [-mx, 0]]
    return np.asarray(t, float) * unit


class SlotSim:
    def __init__(self, p: DynParams, seed: int, slot: float = 20.0, collect_radius: float = 200.0,
                 update_num: int = 10, motion: str = "ucs", drain: str = "radius",
                 obs_radius: float | None = None, reward_ref: float = 3600.0):
        self.p, self.seed, self.tau = p, seed, float(slot)
        self.rc, self.update_num, self.motion, self.drain = collect_radius, update_num, motion, drain
        self.obs_radius = obs_radius
        self.reward_ref = reward_ref
        self.A = ucs_action_table(p.v, self.tau)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        p = self.p
        self.rng = np.random.default_rng(self.seed)
        self.field = SensorField(p, self.rng)          # identical construction to DynSim(seed)
        self.clock = 0.0
        K = p.K
        self.state = np.array(["wait"] * K, dtype=object)
        self.launch_at = np.array([k * p.t_c / max(K, 1) for k in range(K)])
        self.pos = np.tile(p.home, (K, 1)).astype(float)
        self.E = np.full(K, p.E_usable)
        self.busy_until = np.zeros(K)
        self.t_launch = np.zeros(K)
        self.n_vis = np.zeros(K, int)
        self.records = []                               # (t_launch, t_land, n_visited, k)
        self._age_integral = self._event_integral = self._measure_time = 0.0
        self._visits = np.zeros(p.M, int)
        self.trunc = 0
        self.last_reward = np.zeros(K)
        return self

    # ---------------------------------------------------------- accounting (verbatim from DynSim)
    def _accumulate(self, t0, t1):
        p = self.p
        t0 = max(t0, p.T_burnin); t1 = min(t1, p.T_horizon)
        if t1 <= t0: return
        w = self.field.weights(t0); a0 = self.field.age(t0); a1 = self.field.age(t1)
        contrib = w * 0.5 * (a0 + a1) * (t1 - t0)
        if p.event_model == "separate":
            ev = self.field.event_age_integral(t0, t1)
            self._event_integral += float(ev.sum()); contrib = contrib + ev
        self._age_integral += float(contrib.sum()); self._measure_time += (t1 - t0)

    def _advance(self, t):
        if t <= self.clock: return
        self.field.advance(self.clock, t); self._accumulate(self.clock, t); self.clock = t

    # ---------------------------------------------------------------- helpers
    def d_home(self, k):
        return float(np.linalg.norm(self.pos[k] - self.p.home))

    def must_return(self, k):
        p = self.p
        return self.E[k] - p.e_fly(self.d_home(k)) < 2.0 * p.Pf * self.tau

    def avail_actions(self, k):
        """Their mask semantics: 1 = allowed. Out-of-map moves masked; only 'stay' while not steerable."""
        m = np.zeros(len(self.A), int)
        if self.state[k] != "air" or self.must_return(k) or self.busy_until[k] >= self.clock + self.tau:
            m[0] = 1; return m
        nxt = self.pos[k] + self.A
        m[:] = ((nxt >= 0) & (nxt <= self.p.L)).all(1)
        return m

    def _land_and_relaunch(self, k, t):
        self.records.append((self.t_launch[k], t, int(self.n_vis[k]), k))
        self.field.expire_check(t)
        self.pos[k] = self.p.home; self.E[k] = self.p.E_usable
        self.t_launch[k] = t; self.n_vis[k] = 0; self.busy_until[k] = t

    # ------------------------------------------------------------------ step
    def step(self, actions, targets=None):
        """actions: K ints (motion 'ucs'); targets: K points or None (motion 'direct').
        Returns reward vector (per UAV: sum of w * age-at-drain / reward_ref; epsilon = 0)."""
        p, tau = self.p, self.tau
        t0 = self.clock; t1 = t0 + tau
        reward = np.zeros(p.K); ready = []
        for k in range(p.K):
            if self.state[k] == "wait":
                if self.launch_at[k] <= t0:
                    self.state[k] = "air"; self.t_launch[k] = self.launch_at[k]
                    self.busy_until[k] = t0
                else:
                    continue
            free = max(t0, self.busy_until[k])
            if free >= t1:
                continue                                   # still draining through this whole slot
            avail_t = t1 - free
            if self.must_return(k) or self.state[k] == "return":
                self.state[k] = "return"
                dist = self.d_home(k); fly = min(dist, p.v * avail_t)
                if dist > 0:
                    self.pos[k] += (p.home - self.pos[k]) * (fly / dist)
                self.E[k] -= p.e_fly(fly)
                if fly >= dist - 1e-9:
                    self._land_and_relaunch(k, free + dist / p.v); self.state[k] = "air"
                    self.busy_until[k] = t1                # the replacement acts from the next slot
                continue
            if self.motion == "direct":
                tgt = np.asarray(targets[k], float) if targets is not None and targets[k] is not None else self.pos[k]
                vec = tgt - self.pos[k]; dist = float(np.linalg.norm(vec))
                if dist <= self.rc:
                    mv = np.zeros(2)          # [fix 2026-10-06] already within collection range: hold and drain now
                else:                         # (a full-slot move would end at the slot boundary, leaving no time)
                    mv = vec if dist <= p.v * avail_t else vec * (p.v * avail_t / dist)
            else:
                mv = self.A[int(actions[k])].copy()
                n = float(np.linalg.norm(mv))
                if n > p.v * avail_t: mv *= p.v * avail_t / n
            new = self.pos[k] + mv
            # [fix 2026-10-05] sensors can sit exactly on the field edge (layouts clip to [0, L]); a straight
            # flight to them overshoots by float rounding (~1e-12 m) and the move used to be rejected, so the
            # UAV never arrived. Clip overshoots below 1 mm; still reject genuinely out-of-field moves.
            if ((new >= -1e-3) & (new <= p.L + 1e-3)).all():
                new = np.clip(new, 0.0, p.L); mv = new - self.pos[k]
            else:
                mv[:] = 0.0; new = self.pos[k]
            m = float(np.linalg.norm(mv))
            self.pos[k] = new; self.E[k] -= p.e_fly(m)
            ready.append((free + m / p.v, k))
        # --- drains, processed in global time order across UAVs (exact accounting) ---
        heapq.heapify(ready); done = {k: set() for k in range(p.K)}; busy_end = {}
        while ready:
            t, k = heapq.heappop(ready)
            busy_end[k] = t
            if t >= t1 or len(done[k]) >= self.update_num: continue
            self._advance(t)
            if self.drain == "target":
                cand = [] if targets is None or targets[k] is None else \
                       [int(j) for j in np.where(np.all(np.isclose(self.field.pos, targets[k]), axis=1))[0]]
                cand = [j for j in cand if np.linalg.norm(self.field.pos[j] - self.pos[k]) <= self.rc]
            else:
                d = np.linalg.norm(self.field.pos - self.pos[k], axis=1)
                cand = [int(j) for j in np.argsort(d) if d[j] <= self.rc]
            cand = [j for j in cand if j not in done[k]]
            if not cand: continue
            j = cand[0]
            td = self.field.dwell_time(j)
            idle_after = max(0.0, t1 - (t + td))
            if self.E[k] - p.e_hover(td) - p.e_hover(idle_after) - p.e_fly(self.d_home(k)) < 0:
                self.trunc += 1; continue                 # cannot pay this drain and still get home
            a = t - self.field.t_last_visit[j]
            if t >= p.T_burnin: self._visits[j] += 1
            self.field.visit(t, j)
            reward[k] += self.field.wi_base[j] * a / self.reward_ref
            self.E[k] -= p.e_hover(td); self.n_vis[k] += 1; done[k].add(j)
            self.busy_until[k] = t + td
            heapq.heappush(ready, (t + td, k))
        self._advance(t1)
        for k, te in busy_end.items():                    # idle hover for the rest of the slot
            if self.state[k] == "air":
                self.E[k] -= p.e_hover(max(0.0, t1 - max(te, self.busy_until[k])))
        self.last_reward = reward
        return reward

    @property
    def done(self):
        return self.clock >= self.p.T_horizon

    def metrics(self):
        mt = max(self._measure_time, 1e-9)
        recs = [r for r in self.records if r[0] >= self.p.T_burnin]
        return dict(J_timeavg=self._age_integral / mt, J_age=(self._age_integral - self._event_integral) / mt,
                    J_event=self._event_integral / mt, n_sorties=len(recs),
                    mean_n=float(np.mean([r[2] for r in recs])) if recs else float("nan"),
                    n_never=int((self._visits == 0).sum()), trunc=self.trunc, measured_s=self._measure_time)

    # ------------------------------------------------------------ observation (their layout, our info)
    def obs_agent(self, k):
        """Per-UAV observation in DRL-UCS's layout: [UAV positions (2K)] + per sensor (x, y, est. backlog,
        age, priority) for sensors within obs_radius (zeros otherwise) + [own energy fraction].
        Departures from their code (disclosed): true queue length -> planner-side backlog ESTIMATE
        min(age*lam_est, B)/B; their generation-time list -> the sensor's priority w/w_hi; their
        elapsed-episode feature (step/T, a horizon leak our rules forbid) -> own remaining energy fraction,
        which the paper's text lists in the observation."""
        p, F = self.p, self.field
        r = np.inf if self.obs_radius is None else self.obs_radius
        out = []
        for i in range(p.K):
            if i == k or np.linalg.norm(self.pos[i] - self.pos[k]) < r:
                out += [self.pos[i, 0] / p.L, self.pos[i, 1] / p.L]
            else:
                out += [0.0, 0.0]
        age = F.age(self.clock)
        est = np.minimum(age * F.lam_est, p.B_bits) / p.B_bits
        vis = np.linalg.norm(F.pos - self.pos[k], axis=1) < r
        feat = np.c_[F.pos / p.L, est, np.minimum(age / (12 * 3600.0), 1.0), F.wi_base / p.wi_hi]
        feat[~vis] = 0.0
        out += feat.ravel().tolist()
        out.append(self.E[k] / p.E_usable)
        return np.asarray(out, np.float32)

    def obs_size(self):
        return 2 * self.p.K + 5 * self.p.M + 1
