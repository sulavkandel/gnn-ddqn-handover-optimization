from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


Observation = Dict[str, np.ndarray]


@dataclass(frozen=True)
class FeatureIndex:
    # Edge feature locations used by baselines and the neural network.
    RSRP: int = 0
    SINR: int = 1
    DISTANCE: int = 2
    IS_SERVING: int = 3
    TARGET_LOAD: int = 4
    DELTA_RSRP: int = 5


EDGE = FeatureIndex()


def jain_fairness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    numerator = float(values.sum() ** 2)
    denominator = float(values.size * np.square(values).sum())
    return numerator / denominator if denominator > 1e-12 else 0.0


class DenseUrbanHandoverEnv:
    """Fast, trace-free dense-urban handover prototype.

    The environment intentionally uses a transparent radio abstraction instead
    of claiming full 3GPP compliance. It is suitable for algorithm development.
    Final paper claims should be revalidated in ns-3/5G-LENA or with field logs.

    An action is a candidate-list position for each UE. Candidate position zero
    is always "stay with the current serving cell". The remaining positions are
    the strongest safe neighboring cells, so the joint action space never has
    to be enumerated.
    """

    UE_FEATURE_DIM = 7
    BS_FEATURE_DIM = 5
    EDGE_FEATURE_DIM = 6

    def __init__(self, config: Dict):
        self.cfg = dict(config)
        self.num_bs = int(config["num_bs"])
        self.num_ues = int(config["num_ues"])
        self.area = float(config["area_size_m"])
        self.steps_per_episode = int(config["episode_steps"])
        self.dt = float(config["dt_s"])
        self.k = int(config["candidate_k"])
        self.rng = np.random.default_rng(int(config.get("seed", 0)))

        self.step_index = 0
        self.bs_xy = np.empty((self.num_bs, 2), dtype=np.float32)
        self.ue_xy = np.empty((self.num_ues, 2), dtype=np.float32)
        self.ue_velocity = np.empty((self.num_ues, 2), dtype=np.float32)
        self.demand_mbps = np.empty(self.num_ues, dtype=np.float32)
        self.shadow_db = np.empty((self.num_ues, self.num_bs), dtype=np.float32)
        self.bs_activity = np.ones(self.num_bs, dtype=np.float32)
        self.serving = np.zeros(self.num_ues, dtype=np.int64)
        self.previous_serving = np.zeros(self.num_ues, dtype=np.int64)
        self.time_since_handover = np.zeros(self.num_ues, dtype=np.float32)
        self.cell_load = np.zeros(self.num_bs, dtype=np.float32)
        self.cell_user_count = np.zeros(self.num_bs, dtype=np.float32)
        self.rsrp_dbm = np.empty((self.num_ues, self.num_bs), dtype=np.float32)
        self.sinr_db = np.empty((self.num_ues, self.num_bs), dtype=np.float32)
        self.distance_m = np.empty((self.num_ues, self.num_bs), dtype=np.float32)
        self._last_candidates = np.empty((self.num_ues, self.k), dtype=np.int64)
        self._last_mask = np.empty((self.num_ues, self.k), dtype=bool)

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_index = 0
        self.bs_xy = self._build_bs_layout()
        self.ue_xy = self.rng.uniform(0.0, self.area, size=(self.num_ues, 2)).astype(np.float32)
        self.ue_velocity = self._sample_velocities()

        lo = np.log(float(self.cfg["demand_min_mbps"]))
        hi = np.log(float(self.cfg["demand_max_mbps"]))
        self.demand_mbps = np.exp(self.rng.uniform(lo, hi, size=self.num_ues)).astype(np.float32)

        shadow_std = float(self.cfg["shadow_std_db"])
        self.shadow_db = self.rng.normal(0.0, shadow_std, size=(self.num_ues, self.num_bs)).astype(np.float32)
        self.bs_activity.fill(0.35)
        self._compute_radio()

        self.serving = self.rsrp_dbm.argmax(axis=1).astype(np.int64)
        self.previous_serving = self.serving.copy()
        self.time_since_handover.fill(float(self.cfg["pingpong_window_s"]) * 2.0)
        _, self.cell_load, self.cell_user_count = self._throughput_and_load(self.serving)
        self._refresh_activity()
        self._compute_radio()
        return self._build_observation()

    def step(self, action_positions: np.ndarray) -> Tuple[Observation, np.ndarray, bool, Dict[str, float]]:
        action_positions = np.asarray(action_positions, dtype=np.int64)
        if action_positions.shape != (self.num_ues,):
            raise ValueError(f"Expected actions with shape ({self.num_ues},), got {action_positions.shape}")

        clipped = np.clip(action_positions, 0, self.k - 1)
        valid = self._last_mask[np.arange(self.num_ues), clipped]
        invalid_action = (clipped != action_positions) | (~valid)
        clipped = np.where(valid, clipped, 0)
        targets = self._last_candidates[np.arange(self.num_ues), clipped]

        old_serving = self.serving.copy()
        handover = targets != old_serving
        pingpong_window = float(self.cfg["pingpong_window_s"])
        pingpong = handover & (targets == self.previous_serving) & (self.time_since_handover <= pingpong_window)

        self.previous_serving[handover] = old_serving[handover]
        self.serving[handover] = targets[handover]

        throughput, self.cell_load, self.cell_user_count = self._throughput_and_load(self.serving)
        interruption_fraction = min(1.0, float(self.cfg["handover_interruption_s"]) / self.dt)
        throughput = throughput * (1.0 - interruption_fraction * handover.astype(np.float32))

        ue_rows = np.arange(self.num_ues)
        serving_sinr = self.sinr_db[ue_rows, self.serving]
        serving_rsrp = self.rsrp_dbm[ue_rows, self.serving]
        rlf = (serving_sinr < float(self.cfg["rlf_sinr_db"])) | (
            serving_rsrp < float(self.cfg["min_rsrp_dbm"])
        )
        demand_ratio = np.clip(throughput / np.maximum(self.demand_mbps, 1e-3), 0.0, 1.0)
        fairness = jain_fairness(demand_ratio)
        overload = np.clip(self.cell_load[self.serving] - 1.0, 0.0, 1.0)

        # Per-UE reward improves credit assignment; the global fairness term
        # still coordinates the parameter-shared agents.
        reward = (
            0.60 * demand_ratio
            + 0.15 * fairness
            - 0.05 * handover.astype(np.float32)
            - 0.25 * pingpong.astype(np.float32)
            - 0.60 * rlf.astype(np.float32)
            - 0.10 * overload.astype(np.float32)
        ).astype(np.float32)

        info = {
            "mean_ue_throughput_mbps": float(np.mean(throughput)),
            "p05_ue_throughput_mbps": float(np.quantile(throughput, 0.05)),
            "jain_demand_satisfaction": float(fairness),
            "mean_cell_load": float(np.mean(self.cell_load)),
            "overloaded_cell_fraction": float(np.mean(self.cell_load > 1.0)),
            "handovers": float(np.sum(handover)),
            "pingpongs": float(np.sum(pingpong)),
            "rlfs": float(np.sum(rlf)),
            "blocked_actions": float(np.sum(invalid_action)),
            "mean_reward": float(np.mean(reward)),
        }

        self.time_since_handover += self.dt
        self.time_since_handover[handover] = 0.0
        self._refresh_activity()
        self._move_ues()
        self._update_shadowing()
        self._compute_radio()
        self.step_index += 1

        done = self.step_index >= self.steps_per_episode
        next_observation = self._build_observation()
        return next_observation, reward, done, info

    def _build_bs_layout(self) -> np.ndarray:
        side = int(np.ceil(np.sqrt(self.num_bs)))
        spacing = self.area / side
        coordinates = []
        for row in range(side):
            for col in range(side):
                coordinates.append(((col + 0.5) * spacing, (row + 0.5) * spacing))
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if len(coordinates) > self.num_bs:
            selected = self.rng.choice(len(coordinates), size=self.num_bs, replace=False)
            coordinates = coordinates[selected]
        jitter = float(self.cfg["grid_jitter_fraction"]) * spacing
        coordinates += self.rng.normal(0.0, jitter, size=coordinates.shape).astype(np.float32)
        return np.clip(coordinates, 0.0, self.area).astype(np.float32)

    def _sample_velocities(self) -> np.ndarray:
        # Dense-urban mix: 60% pedestrian, 30% urban vehicle, 10% faster vehicle.
        mode = self.rng.choice(3, size=self.num_ues, p=[0.60, 0.30, 0.10])
        speed = np.empty(self.num_ues, dtype=np.float32)
        speed[mode == 0] = self.rng.uniform(0.4, 1.8, size=np.sum(mode == 0))
        speed[mode == 1] = self.rng.uniform(3.0, 11.0, size=np.sum(mode == 1))
        speed[mode == 2] = self.rng.uniform(11.0, 18.0, size=np.sum(mode == 2))
        angle = self.rng.uniform(0.0, 2.0 * np.pi, size=self.num_ues)
        return np.column_stack((speed * np.cos(angle), speed * np.sin(angle))).astype(np.float32)

    def _move_ues(self) -> None:
        turn = self.rng.random(self.num_ues) < float(self.cfg["turn_probability"])
        if np.any(turn):
            speed = np.linalg.norm(self.ue_velocity[turn], axis=1)
            angle = self.rng.uniform(0.0, 2.0 * np.pi, size=np.sum(turn))
            self.ue_velocity[turn, 0] = speed * np.cos(angle)
            self.ue_velocity[turn, 1] = speed * np.sin(angle)

        self.ue_xy += self.ue_velocity * self.dt
        for axis in (0, 1):
            low = self.ue_xy[:, axis] < 0.0
            high = self.ue_xy[:, axis] > self.area
            bounced = low | high
            self.ue_velocity[bounced, axis] *= -1.0
            self.ue_xy[:, axis] = np.clip(self.ue_xy[:, axis], 0.0, self.area)

    def _update_shadowing(self) -> None:
        rho = float(self.cfg["shadow_correlation"])
        sigma = float(self.cfg["shadow_std_db"])
        innovation = self.rng.normal(0.0, sigma, size=self.shadow_db.shape).astype(np.float32)
        self.shadow_db = rho * self.shadow_db + np.sqrt(max(0.0, 1.0 - rho * rho)) * innovation

    def _compute_radio(self) -> None:
        delta = self.ue_xy[:, None, :] - self.bs_xy[None, :, :]
        distance_2d = np.maximum(np.linalg.norm(delta, axis=-1), 10.0)
        height_delta = float(self.cfg["bs_height_m"]) - float(self.cfg["ue_height_m"])
        self.distance_m = np.sqrt(distance_2d**2 + height_delta**2).astype(np.float32)

        # Transparent dense-urban log-distance model. Do not label this as a
        # complete TR 38.901 implementation in a paper.
        frequency_ghz = float(self.cfg["carrier_ghz"])
        pathloss_db = 32.4 + 20.0 * np.log10(frequency_ghz) + 30.0 * np.log10(self.distance_m)
        self.rsrp_dbm = (
            float(self.cfg["bs_tx_power_dbm"]) - pathloss_db + self.shadow_db
        ).astype(np.float32)

        received_mw = np.power(10.0, self.rsrp_dbm / 10.0, dtype=np.float64)
        weighted_received = received_mw * self.bs_activity[None, :]
        bandwidth_hz = float(self.cfg["bandwidth_mhz"]) * 1e6
        noise_dbm = -174.0 + 10.0 * np.log10(bandwidth_hz) + float(self.cfg["noise_figure_db"])
        noise_mw = 10.0 ** (noise_dbm / 10.0)
        interference = np.maximum(weighted_received.sum(axis=1, keepdims=True) - weighted_received, 0.0)
        sinr_linear = received_mw / (interference + noise_mw)
        self.sinr_db = (10.0 * np.log10(np.maximum(sinr_linear, 1e-12))).astype(np.float32)

    def _throughput_and_load(self, serving: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.arange(self.num_ues)
        sinr_linear = np.power(10.0, self.sinr_db[rows, serving] / 10.0)
        spectral_efficiency = np.clip(np.log2(1.0 + sinr_linear), 0.0, 7.4)
        link_capacity_mbps = float(self.cfg["bandwidth_mhz"]) * spectral_efficiency
        user_count = np.bincount(serving, minlength=self.num_bs).astype(np.float32)
        equal_share = np.maximum(user_count[serving], 1.0)
        throughput = np.minimum(self.demand_mbps, link_capacity_mbps / equal_share).astype(np.float32)

        requested_resource = self.demand_mbps / np.maximum(link_capacity_mbps, 0.05)
        load = np.bincount(serving, weights=requested_resource, minlength=self.num_bs).astype(np.float32)
        return throughput, load, user_count

    def _refresh_activity(self) -> None:
        floor = float(self.cfg["bs_activity_floor"])
        self.bs_activity = (floor + (1.0 - floor) * np.clip(self.cell_load, 0.0, 1.0)).astype(np.float32)

    def _candidate_table(self) -> Tuple[np.ndarray, np.ndarray]:
        candidates = np.repeat(self.serving[:, None], self.k, axis=1).astype(np.int64)
        mask = np.zeros((self.num_ues, self.k), dtype=bool)
        mask[:, 0] = True
        min_rsrp = float(self.cfg["min_rsrp_dbm"])
        max_loss = float(self.cfg["max_handover_loss_db"])

        for ue in range(self.num_ues):
            order = np.argsort(self.rsrp_dbm[ue])[::-1]
            position = 1
            serving_rsrp = self.rsrp_dbm[ue, self.serving[ue]]
            for bs in order:
                if bs == self.serving[ue]:
                    continue
                if position >= self.k:
                    break
                candidates[ue, position] = int(bs)
                target_rsrp = self.rsrp_dbm[ue, bs]
                mask[ue, position] = (target_rsrp >= min_rsrp) and (target_rsrp >= serving_rsrp - max_loss)
                position += 1
        return candidates, mask

    def _build_observation(self) -> Observation:
        candidates, mask = self._candidate_table()
        self._last_candidates = candidates
        self._last_mask = mask

        rows = np.arange(self.num_ues)[:, None]
        candidate_rsrp = self.rsrp_dbm[rows, candidates]
        candidate_sinr = self.sinr_db[rows, candidates]
        candidate_distance = self.distance_m[rows, candidates]
        serving_rsrp = self.rsrp_dbm[np.arange(self.num_ues), self.serving][:, None]
        is_serving = candidates == self.serving[:, None]
        target_load = self.cell_load[candidates]

        diagonal = np.sqrt(2.0) * self.area
        edge_features = np.stack(
            [
                np.clip((candidate_rsrp + 140.0) / 80.0, 0.0, 1.0),
                np.clip((candidate_sinr + 20.0) / 50.0, 0.0, 1.0),
                np.clip(candidate_distance / diagonal, 0.0, 1.0),
                is_serving.astype(np.float32),
                np.clip(target_load / 2.0, 0.0, 1.0),
                np.clip((candidate_rsrp - serving_rsrp) / 20.0, -1.0, 1.0),
            ],
            axis=-1,
        ).astype(np.float32)

        max_speed = 18.0
        current_sinr = self.sinr_db[np.arange(self.num_ues), self.serving]
        ue_features = np.column_stack(
            [
                self.ue_xy[:, 0] / self.area,
                self.ue_xy[:, 1] / self.area,
                self.ue_velocity[:, 0] / max_speed,
                self.ue_velocity[:, 1] / max_speed,
                np.log1p(self.demand_mbps) / np.log1p(float(self.cfg["demand_max_mbps"])),
                np.clip(self.time_since_handover / float(self.cfg["pingpong_window_s"]), 0.0, 2.0) / 2.0,
                np.clip((current_sinr + 20.0) / 50.0, 0.0, 1.0),
            ]
        ).astype(np.float32)

        bs_features = np.column_stack(
            [
                self.bs_xy[:, 0] / self.area,
                self.bs_xy[:, 1] / self.area,
                np.clip(self.cell_load / 2.0, 0.0, 1.0),
                self.bs_activity,
                np.clip(self.cell_user_count / max(1.0, self.num_ues / self.num_bs * 4.0), 0.0, 1.0),
            ]
        ).astype(np.float32)

        return {
            "ue": ue_features,
            "bs": bs_features,
            "edge": edge_features,
            "candidate_bs": candidates,
            "mask": mask,
        }
