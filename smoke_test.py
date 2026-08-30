from __future__ import annotations

import numpy as np

from handover.baselines import A3TttPolicy
from handover.config import load_config
from handover.env import DenseUrbanHandoverEnv


def main() -> None:
    config = load_config("configs/smoke.yaml")
    env = DenseUrbanHandoverEnv(config["env"])
    observation = env.reset(seed=123)
    assert observation["ue"].shape == (env.num_ues, env.UE_FEATURE_DIM)
    assert observation["bs"].shape == (env.num_bs, env.BS_FEATURE_DIM)
    assert observation["edge"].shape == (env.num_ues, env.k, env.EDGE_FEATURE_DIM)
    assert np.all(observation["candidate_bs"][:, 0] == env.serving)
    assert np.all(observation["mask"][:, 0])

    policy = A3TttPolicy(env.dt)
    policy.reset(env.num_ues)
    done = False
    steps = 0
    while not done:
        action = policy.act(observation)
        observation, reward, done, info = env.step(action)
        assert action.shape == (env.num_ues,)
        assert reward.shape == (env.num_ues,)
        assert np.isfinite(reward).all()
        assert np.isfinite(list(info.values())).all()
        steps += 1
    assert steps == env.steps_per_episode
    print("environment smoke test passed", info)


if __name__ == "__main__":
    main()
