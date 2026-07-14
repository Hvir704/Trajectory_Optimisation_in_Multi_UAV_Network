"""
env.py — UAV Environment reconstructed directly from trainer/policy interactions
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Params:
    M: int = 30
    area: float = 1000.0
    Emax: float = 50000.0
    Ph: float = 200.0          # Hover power
    Pf: float = 150.0          # Flight power
    v: float = 20.0            # UAV Velocity
    Di_lo: float = 0.5e6       # Data lower bound
    Di_hi: float = 5.0e6       # Data upper bound
    wi_lo: float = 1.0         # Priority lower bound
    wi_hi: float = 10.0        # Priority upper bound
    theta1: float = 0.01       # Objective WAoI coefficient
    R: float = 1e6             # Transmission rate
    home: Tuple[float, float] = (500.0, 500.0)

    def tcd(self, Di: float) -> float:
        """Calculate data collection time based on data volume."""
        return Di / self.R

    def e_fly(self, pos1, pos2) -> float:
        """Calculate energy consumed during flight between two positions."""
        dist = float(np.linalg.norm(np.array(pos1) - np.array(pos2)))
        return (dist / self.v) * self.Pf


class UAVEnv:
    def __init__(self, p: Params, seed=None):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self._sample()

    def _sample(self):
        """Generates random locations, data demands, and priorities for the nodes."""
        p = self.p
        self.pos = self.rng.uniform(0, p.area, (p.M, 2)).astype(np.float32)
        self.Di = self.rng.uniform(p.Di_lo, p.Di_hi, p.M).astype(np.float32)
        self.wi = self.rng.uniform(p.wi_lo, p.wi_hi, p.M).astype(np.float32)
        self.tcd_ = np.array([p.tcd(d) for d in self.Di], dtype=np.float32)
        self.home = np.array(p.home, dtype=np.float32)

    def reset(self) -> dict:
        """Resets the environment for a new rollout and returns the initial observation."""
        self.visited = np.zeros(self.p.M, dtype=bool)
        self.traj = []
        self.curr_pos = self.home.copy()
        self.E_left = self.p.Emax
        return self._obs()

    def _obs(self) -> dict:
        """Packages the state exactly as required by features.py `obs_to_tensors`."""
        return {
            'curr_pos': self.curr_pos.copy(),
            'E_left': self.E_left,
            'node_pos': self.pos.copy(),
            'wi': self.wi.copy(),
            'tcd': self.tcd_.copy(),
            'visited': self.visited.copy()
        }

    def step(self, action: int) -> Tuple[dict, float, bool]:
        """
        Executes an action.
        If action is -1, the UAV returns home and the episode is terminal.
        """
        if action == -1:
            # Return home sequence
            e_req = self.p.e_fly(self.curr_pos, self.home)
            self.E_left -= e_req
            done = True
            
            # REINFORCE attempts to maximize reward, so reward is negative objective
            reward = -self.objective(self.traj)
            return self._obs(), reward, done

        # Visit a specific node sequence
        e_fly = self.p.e_fly(self.curr_pos, self.pos[action])
        e_hover = self.p.Ph * self.tcd_[action]
        
        # Deduct energy, update position, and mark visited
        self.E_left -= (e_fly + e_hover)
        self.curr_pos = self.pos[action].copy()
        self.visited[action] = True
        self.traj.append(action)

        # Intermediate steps do not return a reward (0.0)
        return self._obs(), 0.0, False

    def waoi(self, traj: List[int]) -> float:
        """Calculates Weighted Age of Information proxy for a trajectory."""
        if not traj:
            return 0.0
        
        val = 0.0
        curr = self.home
        for j in traj:
            dist = float(np.linalg.norm(curr - self.pos[j]))
            t_fly = dist / self.p.v
            val += self.wi[j] * (t_fly + self.tcd_[j])
            curr = self.pos[j]
            
        return val

    def objective(self, traj: List[int]) -> float:
        """
        Calculates the composite objective.
        Called directly in trainer_fixed.py to monitor convergence and evaluation.
        """
        if not traj:
            return 0.0
            
        prios = sum(self.wi[j] for j in traj)
        return self.p.theta1 * self.waoi(traj) - prios