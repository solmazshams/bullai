import ray
import argparse
import numpy as np
from json import load, dump
from typing import Dict, Optional, Union
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
from ray.tune.logger import pretty_print
import gym
from envs.trade_env import TradeEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

with open("./experiments/default/config.json", "r") as f:
    config = load(f)


class trade_callbacks(DefaultCallbacks):
    def on_episode_end(self,
                       *,
                       worker: RolloutWorker,
                       base_env: BaseEnv,
                       policies: Dict[str, Policy],
                       episode: Episode,
                       env_index: int,
                       **kwargs
                       ):

        episode.custom_metrics["sharpe_ratio"] = episode._last_infos["agent0"]["sharpe_ratio"]
        episode.custom_metrics["portfolio_value"] = episode._last_infos["agent0"]["portfolio_value"]
        for i,s in enumerate(config["symbols"]):
            episode.custom_metrics[s] = episode._last_infos["agent0"]["portions"][i]

        episode.custom_metrics["balance"] = episode._last_infos["agent0"]["portions"][-1]
        episode.custom_metrics["all_portions"] = episode._last_infos["agent0"]["all_portions"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iter", type=int, default=1000, help="number of training iterations")
    parser.add_argument("--render", type=bool, default=False, help="enable rendering")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--alg", type=str, default="PPO", help="DRL algorithm to be used")
    parser.add_argument("--env", type=str, default="NN", help="env name")
    parser.add_argument("--exp", type=str, default="default", help="experiment name")
    parser.add_argument("--policy", type=str, default="NN", help="experiment name")

    opt = parser.parse_args()

    input_args = ["render", "num_workers", "alg", "env", "exp", "policy"]
    with open("./experiments/" + opt.exp + "/config.json", "r") as f:
        config = load(f)
    for arg in input_args:
        config[arg] = vars(opt)[arg]
    
    ray.init(num_gpus=0)

   
    algo = (
        PPOConfig()
        .training(
            gamma = config["gamma"],
            lr =  config["lr"],
            train_batch_size = config["batch_size"],
            sgd_minibatch_size = config["minibatch_size"],
            # model = {
            #     "use_lstm": True,
            # },
        )
        .callbacks(trade_callbacks)
        .rollouts(num_rollout_workers=config["num_workers"])
        .resources(num_gpus=0)
        .environment(env=TradeEnv, env_config = config)
        .build()
    )
    
    for i in range(config["num_iterations"]):
        result = algo.train()
        print(pretty_print(result))

        if i % 50 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")    

    ray.shutdown()