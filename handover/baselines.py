from __future__ import annotations

import numpy as np

from .env import EDGE, Observation


class StayPolicy:
    def reset(self, num_ues: int) -> None:
        pass

    def act(self, observation: Observation) -> np.ndarray:
        return np.zeros(observation["ue"].shape[0], dtype=np.int64)


class StrongestRsrpPolicy:
    def reset(self, num_ues: int) -> None:
        pass

    def act(self, observation: Observation) -> np.ndarray:
        scores = observation["edge"][:, :, EDGE.RSRP].copy()
        scores[~observation["mask"]] = -np.inf
        return scores.argmax(axis=1).astype(np.int64)


class A3TttPolicy:
    """Event-A3-style baseline operating on the same candidate list."""

    def __init__(self, dt_s: float, hysteresis_db: float = 3.0, ttt_s: float = 0.4):
        self.dt_s = float(dt_s)
        self.hysteresis_db = float(hysteresis_db)
        self.ttt_s = float(ttt_s)
        self.timer: np.ndarray | None = None
        self.tracked_candidate: np.ndarray | None = None

    def reset(self, num_ues: int) -> None:
        self.timer = np.zeros(num_ues, dtype=np.float32)
        self.tracked_candidate = np.full(num_ues, -1, dtype=np.int64)

    def act(self, observation: Observation) -> np.ndarray:
        num_ues = observation["ue"].shape[0]
        if self.timer is None or self.timer.shape[0] != num_ues:
            self.reset(num_ues)
        assert self.timer is not None and self.tracked_candidate is not None

        scores = observation["edge"][:, :, EDGE.RSRP].copy()
        scores[~observation["mask"]] = -np.inf
        best = scores.argmax(axis=1)
        best_cell = observation["candidate_bs"][np.arange(num_ues), best]
        gain_db = observation["edge"][np.arange(num_ues), best, EDGE.DELTA_RSRP] * 20.0
        condition = (best != 0) & (gain_db >= self.hysteresis_db)
        same_candidate = best_cell == self.tracked_candidate
        self.timer = np.where(condition & same_candidate, self.timer + self.dt_s, 0.0)
        self.tracked_candidate = np.where(condition, best_cell, -1)
        action = np.where(condition & (self.timer >= self.ttt_s), best, 0).astype(np.int64)
        self.timer[action != 0] = 0.0
        self.tracked_candidate[action != 0] = -1
        return action


class LoadAwarePolicy:
    def __init__(self, load_penalty: float = 0.35, stay_bonus: float = 0.02):
        self.load_penalty = float(load_penalty)
        self.stay_bonus = float(stay_bonus)

    def reset(self, num_ues: int) -> None:
        pass

    def act(self, observation: Observation) -> np.ndarray:
        edge = observation["edge"]
        score = edge[:, :, EDGE.RSRP] - self.load_penalty * edge[:, :, EDGE.TARGET_LOAD]
        score[:, 0] += self.stay_bonus
        score[~observation["mask"]] = -np.inf
        return score.argmax(axis=1).astype(np.int64)
