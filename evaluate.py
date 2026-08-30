from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from handover.agent import DDQNAgent
from handover.baselines import A3TttPolicy, LoadAwarePolicy, StayPolicy, StrongestRsrpPolicy
from handover.config import load_config
from handover.env import DenseUrbanHandoverEnv
from handover.metrics import EpisodeAccumulator, summarize_metric


class LearnedPolicy:
    def __init__(self, agent: DDQNAgent):
        self.agent = agent

    def reset(self, num_ues: int) -> None:
        pass

    def act(self, observation):
        return self.agent.select_action(observation, epsilon=0.0)


def write_rows(path: Path, rows: list[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_episode(env_config: Dict, seed: int, policy) -> Dict[str, float]:
    env = DenseUrbanHandoverEnv(env_config)
    observation = env.reset(seed=seed)
    policy.reset(env.num_ues)
    accumulator = EpisodeAccumulator(env.num_ues, env.dt)
    done = False
    while not done:
        action = policy.act(observation)
        observation, _, done, info = env.step(action)
        accumulator.update(info)
    return accumulator.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired-seed evaluation of GNN-DDQN and baselines")
    parser.add_argument("--config", default="configs/dense_urban_100x100.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default="evaluation/dense_urban_100x100")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    config = load_config(args.config)
    env_config = config["env"]
    dt = float(env_config["dt_s"])
    factories: Dict[str, Callable[[], object]] = {
        "stay": lambda: StayPolicy(),
        "strongest_rsrp": lambda: StrongestRsrpPolicy(),
        "a3_ttt": lambda: A3TttPolicy(dt_s=dt, hysteresis_db=3.0, ttt_s=0.4),
        "load_aware": lambda: LoadAwarePolicy(),
    }

    if args.checkpoint:
        probe = DenseUrbanHandoverEnv(env_config)
        agent = DDQNAgent(
            probe.UE_FEATURE_DIM,
            probe.BS_FEATURE_DIM,
            probe.EDGE_FEATURE_DIM,
            config["model"],
            config["train"],
            device=args.device,
        )
        agent.load(args.checkpoint, load_optimizer=False)
        factories["gnn_ddqn"] = lambda: LearnedPolicy(agent)

    per_seed: list[Dict] = []
    for offset in range(args.seeds):
        seed = args.seed_start + offset
        for policy_name, factory in factories.items():
            metrics = run_episode(env_config, seed, factory())
            metrics.update({"policy": policy_name, "seed": seed})
            per_seed.append(metrics)
            print(
                f"seed={seed} policy={policy_name} throughput={metrics['mean_ue_throughput_mbps']:.3f} "
                f"pingpong={metrics['pingpong_fraction']:.4f} reward={metrics['mean_reward']:.3f}"
            )

    metric_names = [key for key in per_seed[0] if key not in {"policy", "seed"}]
    summary: list[Dict] = []
    for policy_name in factories:
        policy_rows = [row for row in per_seed if row["policy"] == policy_name]
        for metric in metric_names:
            stats = summarize_metric(np.asarray([row[metric] for row in policy_rows], dtype=float))
            summary.append({"policy": policy_name, "metric": metric, **stats})

    output = Path(args.out)
    write_rows(output / "per_seed.csv", per_seed)
    write_rows(output / "summary.csv", summary)
    print(f"wrote {output / 'per_seed.csv'} and {output / 'summary.csv'}")


if __name__ == "__main__":
    main()
