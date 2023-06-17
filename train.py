""" Train script """

import argparse
from json import load

import ray
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.tune.logger import pretty_print
import numpy as np

import wandb

from envs.trade_env import TradeEnv
from envs.trade_callbacks import TradeCallbacks
from evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, default="default", help="experiment name")

if __name__ == "__main__":
    opt = parser.parse_args()

    input_args = ["exp"]
    with open("./experiments/" + opt.exp + "/config.json", "r", encoding='utf-8') as f:
        config = load(f)
    with open("./experiments/evaluation/eval_config.json", "r", encoding='utf-8') as f:
        eval_config = load(f)
    for arg in input_args:
        config[arg] = vars(opt)[arg]

    wandb.init(
        project="trade_env",
        config=config
    )

    ray.init(num_gpus=0)
    eval_env = TradeEnv(config = eval_config)
    trainer_config = (
        PPOConfig()
        .training(
            gamma = config["gamma"],
            lr =  config["lr"],
            train_batch_size = config["batch_size"],
            sgd_minibatch_size = config["minibatch_size"],
            model = {
                "fcnet_hiddens": [4, 32, 32],
                "fcnet_activation": "tanh",
            },
        )
        .callbacks(TradeCallbacks)
        .rollouts(num_rollout_workers=config["num_workers"])
        .resources(num_gpus=0)
        .environment(env=TradeEnv, env_config = config)
    )

    algo = trainer_config.build()
    for i in range(config["num_iterations"]):
        result = algo.train()
        print("iteration")
        print(pretty_print(result))
        wandb.log(result["custom_metrics"], step = i)
        wandb.log({"episode_reward_mean" : result["episode_reward_mean"]}, step = i)

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
            eval_results= evaluate(eval_config=eval_config,
                                    checkpoint_dir=checkpoint_dir,
                                    render = True,
                                    iteration = i)
            wandb.log(eval_results, step = i)

    ray.shutdown()
    wandb.finish()