from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "env": {
        "num_bs": 100,
        "num_ues": 100,
        "area_size_m": 2000.0,
        "episode_steps": 500,
        "dt_s": 0.2,
        "candidate_k": 8,
        "carrier_ghz": 3.5,
        "bandwidth_mhz": 20.0,
        "bs_tx_power_dbm": 30.0,
        "noise_figure_db": 7.0,
        "bs_height_m": 15.0,
        "ue_height_m": 1.5,
        "shadow_std_db": 7.0,
        "shadow_correlation": 0.97,
        "bs_activity_floor": 0.10,
        "min_rsrp_dbm": -120.0,
        "max_handover_loss_db": 6.0,
        "rlf_sinr_db": -6.0,
        "pingpong_window_s": 5.0,
        "handover_interruption_s": 0.08,
        "turn_probability": 0.02,
        "demand_min_mbps": 1.0,
        "demand_max_mbps": 12.0,
        "grid_jitter_fraction": 0.12,
        "seed": 7,
    },
    "model": {
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.10,
    },
    "train": {
        "seed": 17,
        "total_steps": 300000,
        "warmup_steps": 5000,
        "buffer_size": 12000,
        "batch_size": 32,
        "gamma": 0.99,
        "learning_rate": 0.0003,
        "weight_decay": 0.00001,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_end": 1.0,
        "target_tau": 0.005,
        "gradient_clip": 10.0,
        "train_every": 1,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 180000,
        "eval_interval": 20000,
        "eval_episodes": 5,
        "checkpoint_interval": 20000,
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> Dict[str, Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    config = _deep_update(DEFAULT_CONFIG, user_config)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Dict[str, Any]]) -> None:
    env = config["env"]
    model = config["model"]
    train = config["train"]

    if env["num_bs"] < 2:
        raise ValueError("num_bs must be at least 2")
    if env["num_ues"] < 1:
        raise ValueError("num_ues must be positive")
    if not 2 <= env["candidate_k"] <= env["num_bs"]:
        raise ValueError("candidate_k must be between 2 and num_bs")
    if env["dt_s"] <= 0 or env["episode_steps"] <= 0:
        raise ValueError("dt_s and episode_steps must be positive")
    if model["hidden_dim"] % model["num_heads"] != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if train["batch_size"] > train["buffer_size"]:
        raise ValueError("batch_size cannot exceed buffer_size")
