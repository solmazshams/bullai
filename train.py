""" Train script """

import argparse
from json import load

import ray
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm

from envs.trade_env import TradeEnv
from envs.trade_callbacks import TradeCallbacks

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
            # model = {
            #     "use_lstm": True,
            # },
        )
        .callbacks(TradeCallbacks)
        .rollouts(num_rollout_workers=config["num_workers"])
        .resources(num_gpus=0)
        .environment(env=TradeEnv, env_config = config)
        .evaluation(evaluation_config=eval_config, evaluation_interval=2)
    )

    algo = trainer_config.build()
    for i in range(config["num_iterations"]):
        result = algo.train()
        print(pretty_print(result))
        wandb.log(result["custom_metrics"])

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
            eval_results = algo.evaluate()
            wandb.log(eval_results["custom_metrics"])
    print("Training completed. Restoring new Trainer for action inference.")
    # Get the last checkpoint from the above training run.
    # checkpoint = result.get_best_result().checkpoint
    # Create new Algorithm and restore its state from the last checkpoint.
    algo = Algorithm.from_checkpoint(checkpoint_dir)

    # Create the env to do inference in.
    env = TradeEnv(config = eval_config)
    obs, info = env.reset()

    episode = 0
    num_episodes_during_inference = 1
    episode_reward = 0.0

    while episode < num_episodes_during_inference:
        # Compute an action (`a`).
        a = algo.compute_single_action(
            observation=obs,
            explore=False,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `a` to the env.
        obs, reward, done, truncated, info = env.step(a)
        episode_reward += reward
        # Is the episode `done`? -> Reset.
        if done:
            COLOR_BLUE = "\033[34m"
            COLOR_GREEN = "\033[32m"
            COLOR_YELLOW = "\033[33m"
            COLOR_CYAN = "\033[36m"
            COLOR_RESET = "\033[0m"
            print(f"Episode done: Total reward = {episode_reward}")
            print("Portfolio:")
            print(f"Trading:          {COLOR_BLUE}{eval_config['symbols']}{COLOR_RESET}")
            print(f"Start Date:       {COLOR_GREEN}{eval_config['start_date']}{COLOR_RESET}")
            print(f"End Date:         {COLOR_GREEN}{eval_config['end_date']}{COLOR_RESET}")
            print(f"Portfolio Value:  {COLOR_YELLOW}{info['portfolio_value']}{COLOR_RESET}")
            print(f"Traded using trained policy until {COLOR_CYAN}{config['end_date']}{COLOR_RESET}")

            obs, info = env.reset()
            episode += 1
            episode_reward = 0.0

    algo.stop()





    ray.shutdown()
    wandb.finish()