""" Train script """

import argparse
from json import load

import ray
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm

from envs.trade_env import TradeEnv
from envs.trade_callbacks import TradeCallbacks
from evaluate import evaluate

import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="trade_env",
)

parser = argparse.ArgumentParser()
parser.add_argument("--num_iter", type=int, default=1000, help="number of training iterations")
parser.add_argument("--render", type=bool, default=False, help="enable rendering")
parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
parser.add_argument("--alg", type=str, default="PPO", help="DRL algorithm to be used")
parser.add_argument("--env", type=str, default="NN", help="env name")
parser.add_argument("--exp", type=str, default="default", help="experiment name")
parser.add_argument("--policy", type=str, default="NN", help="experiment name")

if __name__ == "__main__":
    opt = parser.parse_args()

    input_args = ["render", "num_workers", "alg", "env", "exp", "policy"]
    with open("./experiments/" + opt.exp + "/config.json", "r", encoding='utf-8') as f:
        config = load(f)
    with open("./experiments/evaluation/eval_config.json", "r", encoding='utf-8') as f:
        eval_config = load(f)
    for arg in input_args:
        config[arg] = vars(opt)[arg]

    ray.init(num_gpus=0)

    trainer_config = (
        PPOConfig()
        .training(
            gamma = config["gamma"],
            lr =  config["lr"],
            train_batch_size = config["batch_size"],
            sgd_minibatch_size = config["minibatch_size"],
            model = {
                "fcnet_hiddens": [32, 32],
                "fcnet_activation" : "relu"
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
        print(pretty_print(result))
        wandb.log(result["custom_metrics"], step = i)

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
            eval_results = evaluate(eval_config=eval_config,
                                    checkpoint_dir=checkpoint_dir)
            wandb.log({"eval_portfolio_value" : eval_results}, step = i)

    print("Training completed. Restoring new Trainer for action inference.")
    # Get the last checkpoint from the above training run.
    # checkpoint = result.get_best_result().checkpoint
    # Create new Algorithm and restore its state from the last checkpoint.

    eval_portfolio_value = evaluate(eval_config=eval_config,
                                    checkpoint_dir=checkpoint_dir)

    ray.shutdown()
    wandb.finish()