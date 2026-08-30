from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .model import BipartiteDuelingQNetwork


Observation = Dict[str, np.ndarray]


def observation_to_torch(observation: Observation, device: torch.device, add_batch: bool) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for key, value in observation.items():
        array = np.asarray(value)
        if add_batch:
            array = array[None]
        if key == "candidate_bs":
            result[key] = torch.as_tensor(array, dtype=torch.long, device=device)
        elif key == "mask":
            result[key] = torch.as_tensor(array, dtype=torch.bool, device=device)
        else:
            result[key] = torch.as_tensor(array, dtype=torch.float32, device=device)
    return result


class DDQNAgent:
    def __init__(
        self,
        ue_dim: int,
        bs_dim: int,
        edge_dim: int,
        model_config: Dict,
        train_config: Dict,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.gamma = float(train_config["gamma"])
        self.tau = float(train_config["target_tau"])
        self.gradient_clip = float(train_config["gradient_clip"])
        self.rng = np.random.default_rng(int(train_config["seed"]))

        kwargs = {
            "ue_dim": ue_dim,
            "bs_dim": bs_dim,
            "edge_dim": edge_dim,
            "hidden_dim": int(model_config["hidden_dim"]),
            "num_layers": int(model_config["num_layers"]),
            "num_heads": int(model_config["num_heads"]),
            "dropout": float(model_config["dropout"]),
        }
        self.online = BipartiteDuelingQNetwork(**kwargs).to(self.device)
        self.target = BipartiteDuelingQNetwork(**kwargs).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(),
            lr=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
        )
        self.gradient_steps = 0

    def select_action(self, observation: Observation, epsilon: float = 0.0) -> np.ndarray:
        tensor_observation = observation_to_torch(observation, self.device, add_batch=True)
        self.online.eval()
        with torch.no_grad():
            q_values = self.online(tensor_observation)[0].cpu().numpy()
        self.online.train()
        actions = q_values.argmax(axis=-1).astype(np.int64)

        mask = observation["mask"]
        explore = self.rng.random(actions.shape[0]) < float(epsilon)
        for ue in np.flatnonzero(explore):
            valid = np.flatnonzero(mask[ue])
            actions[ue] = int(self.rng.choice(valid))
        return actions

    def update(self, batch: Dict) -> Dict[str, np.ndarray | float]:
        observation = observation_to_torch(batch["observation"], self.device, add_batch=False)
        next_observation = observation_to_torch(batch["next_observation"], self.device, add_batch=False)
        action = torch.as_tensor(batch["action"], dtype=torch.long, device=self.device)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)

        q_values = self.online(observation)
        selected_q = q_values.gather(-1, action[..., None]).squeeze(-1)

        with torch.no_grad():
            online_next = self.online(next_observation)
            next_action = online_next.argmax(dim=-1)
            target_next = self.target(next_observation)
            next_q = target_next.gather(-1, next_action[..., None]).squeeze(-1)
            td_target = reward + self.gamma * (1.0 - done[:, None]) * next_q

        td_error = td_target - selected_q
        per_user_loss = F.smooth_l1_loss(selected_q, td_target, reduction="none")
        loss = (per_user_loss.mean(dim=1) * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip)
        self.optimizer.step()
        self._soft_update()
        self.gradient_steps += 1

        priorities = td_error.detach().abs().mean(dim=1).cpu().numpy() + 1e-4
        return {
            "loss": float(loss.detach().cpu()),
            "mean_abs_td_error": float(td_error.detach().abs().mean().cpu()),
            "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
            "priorities": priorities,
        }

    def _soft_update(self) -> None:
        with torch.no_grad():
            for target_parameter, online_parameter in zip(self.target.parameters(), self.online.parameters()):
                target_parameter.mul_(1.0 - self.tau).add_(online_parameter, alpha=self.tau)

    def save(self, path: str | Path, extra: Dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "gradient_steps": self.gradient_steps,
                "extra": extra or {},
            },
            path,
        )

    def load(self, path: str | Path, load_optimizer: bool = True) -> Dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.gradient_steps = int(checkpoint.get("gradient_steps", 0))
        return checkpoint.get("extra", {})
