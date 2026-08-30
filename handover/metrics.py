from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np


class EpisodeAccumulator:
    def __init__(self, num_ues: int, dt_s: float):
        self.num_ues = int(num_ues)
        self.dt_s = float(dt_s)
        self.steps = 0
        self.sums: Dict[str, float] = defaultdict(float)

    def update(self, info: Dict[str, float]) -> None:
        self.steps += 1
        for key, value in info.items():
            self.sums[key] += float(value)

    def finalize(self) -> Dict[str, float]:
        if self.steps == 0:
            raise ValueError("Cannot finalize an empty episode")
        duration_minutes = self.steps * self.dt_s / 60.0
        handovers = self.sums["handovers"]
        return {
            "steps": float(self.steps),
            "duration_s": float(self.steps * self.dt_s),
            "mean_ue_throughput_mbps": self.sums["mean_ue_throughput_mbps"] / self.steps,
            "p05_ue_throughput_mbps": self.sums["p05_ue_throughput_mbps"] / self.steps,
            "jain_demand_satisfaction": self.sums["jain_demand_satisfaction"] / self.steps,
            "mean_cell_load": self.sums["mean_cell_load"] / self.steps,
            "overloaded_cell_fraction": self.sums["overloaded_cell_fraction"] / self.steps,
            "handover_rate_per_ue_min": handovers / max(self.num_ues * duration_minutes, 1e-9),
            "pingpong_fraction": self.sums["pingpongs"] / max(handovers, 1.0),
            "rlf_rate_per_ue_min": self.sums["rlfs"] / max(self.num_ues * duration_minutes, 1e-9),
            "blocked_action_rate": self.sums["blocked_actions"] / max(self.num_ues * self.steps, 1),
            "mean_reward": self.sums["mean_reward"] / self.steps,
            "total_handovers": handovers,
            "total_pingpongs": self.sums["pingpongs"],
            "total_rlfs": self.sums["rlfs"],
        }


def summarize_metric(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    standard_deviation = float(values.std(ddof=1)) if values.size > 1 else 0.0
    ci95 = 1.96 * standard_deviation / np.sqrt(max(values.size, 1))
    return {
        "mean": float(values.mean()),
        "std": standard_deviation,
        "ci95_half_width": float(ci95),
        "n": float(values.size),
    }
