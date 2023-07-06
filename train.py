""" Train script """

import argparse
from json import load

import ray
from ray.rllib.algorithms.ppo.ppo import PPOConfig

import wandb

from envs.trade_env import TradeEnv
from envs.trade_callbacks import TradeCallbacks
from evaluate import evaluate
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

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
    trainer_config = (
        PPOConfig()
        .training(
            gamma = config["gamma"],
            lr =  config["lr"],
            train_batch_size = config["batch_size"],
            sgd_minibatch_size = config["minibatch_size"],
            num_sgd_iter = 20,
            model = {
                "fcnet_hiddens": [32, 32],
                # "fcnet_activation": "relu",
            },
        )
        .callbacks(TradeCallbacks)
        .rollouts(num_rollout_workers=config["num_workers"])
        .resources(num_gpus=0)
        .environment(env=TradeEnv, env_config = config)
    )

    algo = trainer_config.build()
    eval_env = TradeEnv(config = eval_config)

    for i in range(config["num_iterations"]):
        results = algo.train()
        print(f"\033[4miteration = {i}\033[0m")
        learner_stats = results["info"]["learner"]["default_policy"]["learner_stats"]
        _results = {"episode_reward_mean" : results["episode_reward_mean"],
                   "policy_loss" : learner_stats["policy_loss"],
                   "vf_loss" : learner_stats["vf_loss"],
                   "total_loss" : learner_stats["total_loss"]
                   }
        for k,f in _results.items():
            print(f"{k} : {f:.2f}")
        wandb.log(results["custom_metrics"], step = i)
        wandb.log(_results, step = i)

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
            eval_results= evaluate(eval_env=eval_env,
                                    checkpoint_dir=checkpoint_dir,
                                    render = True,
                                    iteration = i)
            wandb.log(eval_results, step = i)

    ray.shutdown()
    wandb.finish()
