from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from handover.agent import DDQNAgent
from handover.config import load_config
from handover.env import DenseUrbanHandoverEnv
from handover.metrics import EpisodeAccumulator
from handover.replay import PrioritizedReplayBuffer


def linear_schedule(start: float, end: float, step: int, duration: int) -> float:
    fraction = min(max(step / max(duration, 1), 0.0), 1.0)
    return start + fraction * (end - start)


def append_csv(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate(agent: DDQNAgent, env_config: Dict, episodes: int, seed_start: int) -> Dict[str, float]:
    rows = []
    for episode in range(episodes):
        env = DenseUrbanHandoverEnv(env_config)
        observation = env.reset(seed=seed_start + episode)
        accumulator = EpisodeAccumulator(env.num_ues, env.dt)
        done = False
        while not done:
            action = agent.select_action(observation, epsilon=0.0)
            observation, _, done, info = env.step(action)
            accumulator.update(info)
        rows.append(accumulator.finalize())
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sparse bipartite GNN + Dueling DDQN")
    parser.add_argument("--config", default="configs/dense_urban_100x100.yaml")
    parser.add_argument("--out", default="runs/dense_urban_100x100")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    args = parser.parse_args()

    config = load_config(args.config)
    env_config, model_config, train_config = config["env"], config["model"], config["train"]
    total_steps = int(args.steps or train_config["total_steps"])
    seed = int(train_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    env = DenseUrbanHandoverEnv(env_config)
    agent = DDQNAgent(
        env.UE_FEATURE_DIM,
        env.BS_FEATURE_DIM,
        env.EDGE_FEATURE_DIM,
        model_config,
        train_config,
        device=args.device,
    )
    if args.resume:
        agent.load(args.resume, load_optimizer=True)

    replay = PrioritizedReplayBuffer(
        int(train_config["buffer_size"]),
        alpha=float(train_config["per_alpha"]),
        seed=seed + 1,
    )
    observation = env.reset(seed=seed * 1000)
    episode_index = 0
    episode_accumulator = EpisodeAccumulator(env.num_ues, env.dt)
    latest_loss = float("nan")
    best_eval_reward = -np.inf

    print(f"device={agent.device} parameters={agent.online.parameter_count():,}")
    print(f"training graph: {env.num_ues} UEs, {env.num_bs} cells, K={env.k}")

    for step in range(1, total_steps + 1):
        epsilon = linear_schedule(
            float(train_config["epsilon_start"]),
            float(train_config["epsilon_end"]),
            step,
            int(train_config["epsilon_decay_steps"]),
        )
        action = agent.select_action(observation, epsilon)
        next_observation, reward, done, info = env.step(action)
        replay.add(observation, action, reward, next_observation, done)
        episode_accumulator.update(info)
        observation = next_observation

        if (
            step >= int(train_config["warmup_steps"])
            and len(replay) >= int(train_config["batch_size"])
            and step % int(train_config["train_every"]) == 0
        ):
            beta = linear_schedule(
                float(train_config["per_beta_start"]),
                float(train_config["per_beta_end"]),
                step,
                total_steps,
            )
            batch = replay.sample(int(train_config["batch_size"]), beta)
            update = agent.update(batch)
            replay.update_priorities(batch["indices"], update["priorities"])
            latest_loss = float(update["loss"])

        if done:
            row = episode_accumulator.finalize()
            row.update({"episode": episode_index, "global_step": step, "epsilon": epsilon, "loss": latest_loss})
            append_csv(output / "train_episodes.csv", row)
            episode_index += 1
            observation = env.reset(seed=seed * 1000 + episode_index)
            episode_accumulator = EpisodeAccumulator(env.num_ues, env.dt)

        if step % 1000 == 0:
            print(
                f"step={step}/{total_steps} eps={epsilon:.3f} replay={len(replay)} "
                f"reward={info['mean_reward']:.3f} loss={latest_loss:.5f}"
            )

        if step % int(train_config["eval_interval"]) == 0:
            metrics = evaluate(agent, env_config, int(train_config["eval_episodes"]), seed_start=900000 + step)
            metrics.update({"global_step": step})
            append_csv(output / "validation.csv", metrics)
            if metrics["mean_reward"] > best_eval_reward:
                best_eval_reward = metrics["mean_reward"]
                agent.save(output / "checkpoint_best.pt", {"step": step, "validation": metrics, "config": config})

        if step % int(train_config["checkpoint_interval"]) == 0:
            agent.save(output / "checkpoint_last.pt", {"step": step, "config": config})

    agent.save(output / "checkpoint_last.pt", {"step": total_steps, "config": config})
    print(f"finished; outputs are in {output}")


if __name__ == "__main__":
    main()
