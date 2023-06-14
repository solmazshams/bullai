from json import load

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm

from envs.trade_env import TradeEnv

with open("./experiments/evaluation/eval_config.json", "r", encoding='utf-8') as f:
# with open("./experiments/default/config.json", "r", encoding='utf-8') as f:
    eval_config = load(f)

checkpoint_dir = None


def evaluate(eval_config=eval_config, 
             checkpoint_dir=checkpoint_dir):

    algo = Algorithm.from_checkpoint(checkpoint_dir)

    # Create the env to do inference in.
    env = TradeEnv(config = eval_config)
    obs, info = env.reset()

    episode = 0
    num_episodes_during_inference = 1
    episode_reward = 0.0

    while episode < num_episodes_during_inference:
        a = algo.compute_single_action(
            observation=obs,
            explore=False,
            policy_id="default_policy",  # <- default value
        )
        obs, reward, done, truncated, info = env.step(a)
        episode_reward += reward
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

            obs, _ = env.reset()
            episode += 1
            episode_reward = 0.0

    algo.stop()
    return info["portfolio_value"]
