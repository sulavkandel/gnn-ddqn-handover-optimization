from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


Observation = Dict[str, np.ndarray]


def compress_observation(observation: Observation) -> Observation:
    return {
        "ue": observation["ue"].astype(np.float16, copy=True),
        "bs": observation["bs"].astype(np.float16, copy=True),
        "edge": observation["edge"].astype(np.float16, copy=True),
        "candidate_bs": observation["candidate_bs"].astype(np.int16, copy=True),
        "mask": observation["mask"].astype(bool, copy=True),
    }


def stack_observations(observations: List[Observation]) -> Observation:
    return {
        "ue": np.stack([item["ue"] for item in observations]).astype(np.float32),
        "bs": np.stack([item["bs"] for item in observations]).astype(np.float32),
        "edge": np.stack([item["edge"] for item in observations]).astype(np.float32),
        "candidate_bs": np.stack([item["candidate_bs"] for item in observations]).astype(np.int64),
        "mask": np.stack([item["mask"] for item in observations]).astype(bool),
    }


@dataclass
class Transition:
    observation: Observation
    action: np.ndarray
    reward: np.ndarray
    next_observation: Observation
    done: bool


class PrioritizedReplayBuffer:
    """Simple proportional PER at graph-transition level.

    Priority sampling is O(capacity), which is acceptable for the supplied
    12k-transition prototype and keeps the implementation auditable.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, seed: int = 0):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.rng = np.random.default_rng(seed)
        self.storage: List[Transition | None] = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.position = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: Observation,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: Observation,
        done: bool,
    ) -> None:
        transition = Transition(
            compress_observation(observation),
            np.asarray(action, dtype=np.int16).copy(),
            np.asarray(reward, dtype=np.float16).copy(),
            compress_observation(next_observation),
            bool(done),
        )
        maximum = float(self.priorities[: self.size].max()) if self.size else 1.0
        self.storage[self.position] = transition
        self.priorities[self.position] = max(maximum, 1e-3)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> Dict:
        if self.size < batch_size:
            raise ValueError("Not enough transitions in replay buffer")
        scaled = np.power(self.priorities[: self.size].clip(min=1e-6), self.alpha)
        probabilities = scaled / scaled.sum()
        indices = self.rng.choice(self.size, size=batch_size, replace=False, p=probabilities)
        transitions = [self.storage[int(index)] for index in indices]
        assert all(item is not None for item in transitions)
        typed = [item for item in transitions if item is not None]

        weights = np.power(self.size * probabilities[indices], -float(beta))
        weights /= weights.max()
        return {
            "observation": stack_observations([item.observation for item in typed]),
            "action": np.stack([item.action for item in typed]).astype(np.int64),
            "reward": np.stack([item.reward for item in typed]).astype(np.float32),
            "next_observation": stack_observations([item.next_observation for item in typed]),
            "done": np.asarray([item.done for item in typed], dtype=np.float32),
            "weights": weights.astype(np.float32),
            "indices": indices.astype(np.int64),
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.asarray(priorities, dtype=np.float32)
        self.priorities[np.asarray(indices, dtype=np.int64)] = np.maximum(priorities, 1e-4)
