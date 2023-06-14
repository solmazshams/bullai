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
COLOR_BLUE = "\033[34m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_CYAN = "\033[36m"
COLOR_RESET = "\033[0m"
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
    eval_env = TradeEnv(config = eval_config)
    trainer_config = (
        PPOConfig()
        .training(
            gamma = config["gamma"],
            lr =  config["lr"],
            train_batch_size = config["batch_size"],
            sgd_minibatch_size = config["minibatch_size"],
            model = {
                "fcnet_hiddens": [64, 64],
                # "fcnet_activation" : "relu"
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
            
            obs, info = eval_env.reset()

            episode = 0
            num_episodes_during_inference = 1
            episode_reward = 0.0
            all_portfolio_value = []
            t = 0
            while episode < num_episodes_during_inference:
                a = algo.compute_single_action(
                    observation=obs,
                    explore=False,
                    policy_id="default_policy",
                )
                obs, reward, done, truncated, info = eval_env.step(a)
                adj_stock_price = eval_env.stocks[eval_env.symbols[0]].data["Close"]/eval_env.stocks[eval_env.symbols[0]].data["Close"][0]
                all_portfolio_value.append(info["portfolio_value"]/eval_env.init_balance)
                wandb.log
                episode_reward += reward
                if done:
                    print(f"Episode done: Total reward = {episode_reward}")
                    print("Portfolio:")
                    print(f"Trading:          {COLOR_BLUE}{eval_config['symbols']}{COLOR_RESET}")
                    print(f"Start Date:       {COLOR_GREEN}{eval_config['start_date']}{COLOR_RESET}")
                    print(f"End Date:         {COLOR_GREEN}{eval_config['end_date']}{COLOR_RESET}")
                    print(f"Portfolio Value:  {COLOR_YELLOW}{info['portfolio_value']}{COLOR_RESET}")
                    obs, _ = eval_env.reset()
                    episode += 1
                    episode_reward = 0.0

            # wandb.log({
            #     eval_env.symbols[0] + "_" + str(i//5) : all_portfolio_value,
            #     eval_env.symbols[0]: adj_stock_price
            #     })                
            wandb.log({"eval_portfolio_value" : info["portfolio_value"]}, step = i)
            
    print("Training completed. Restoring new Trainer for action inference.")
    # Get the last checkpoint from the above training run.
    # checkpoint = result.get_best_result().checkpoint
    # Create new Algorithm and restore its state from the last checkpoint.

    eval_portfolio_value = evaluate(eval_config=eval_config,
                                    checkpoint_dir=checkpoint_dir)

    ray.shutdown()
    wandb.finish()