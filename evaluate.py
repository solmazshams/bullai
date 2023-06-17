from json import load

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
import matplotlib.pyplot as plt
from envs.trade_env import TradeEnv
import numpy as np


COLOR_BLUE = "\033[34m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_CYAN = "\033[36m"
COLOR_RESET = "\033[0m"

with open("./experiments/evaluation/eval_config.json", "r", encoding='utf-8') as f:
# with open("./experiments/default/config.json", "r", encoding='utf-8') as f:
    eval_config = load(f)

def evaluate(eval_config = eval_config,
             checkpoint_dir=None,
             render = True,
             iteration = 0):

    eval_env = TradeEnv(config = eval_config)
    obs, info = eval_env.reset()
    done = False
    episode_reward = 0.0

    all_portfolio_values = []
    time_idx = 0
    df = eval_env.df
    buy_signals = []
    sell_signals = []
    all_portfolio_values.append(eval_config["initial_balance"])

    algo = Algorithm.from_checkpoint(checkpoint_dir)
    while not done:
        # [TODO] add multiple episode evaluation
        action = algo.compute_single_action(
            observation=obs,
            explore=False,
            policy_id="default_policy",
        )
        if action == 1:
            buy_signals.append(time_idx)
        if action == 2:
            sell_signals.append(time_idx)
        obs, reward, done, _, info = eval_env.step(action)
        episode_reward += reward
        all_portfolio_values.append(info["portfolio_value"])

        time_idx += 1


    print(f"\033[4mEpisode done: Total reward = {episode_reward}\033[0m")
    print("Portfolio:")
    print(f"Trading:          {COLOR_BLUE}{eval_config['symbols']}{COLOR_RESET}")
    print(f"Start Date:       {COLOR_GREEN}{eval_config['start_date']}{COLOR_RESET}")
    print(f"End Date:         {COLOR_GREEN}{eval_config['end_date']}{COLOR_RESET}")
    print(f"Portfolio Value:  {COLOR_YELLOW}{info['portfolio_value']}{COLOR_RESET}")

    if render:
        default_investment = eval_env.df['Close']/eval_env.df['Close'][0]  * eval_config["initial_balance"]

        plt.figure(figsize=(10, 10), dpi = 300)
        ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3, colspan=1)
        ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2, colspan=1, sharex=ax1)


        ax1.fill_between(
            x=df.index,
            y1=eval_config["initial_balance"],
            y2=all_portfolio_values,
            where = np.array(all_portfolio_values) > eval_config["initial_balance"],
            facecolor='forestgreen', alpha=0.5)
        ax1.fill_between(df.index, eval_config["initial_balance"], all_portfolio_values,
                        where = np.array(all_portfolio_values) < eval_config["initial_balance"],
                facecolor='orangered', alpha=0.5)
        ax1.set_title(eval_env.symbols[0])
        ax1.plot(df.index, all_portfolio_values, color = 'gray', linewidth = 0.5)
        ax1.plot(df.index, default_investment, color = 'purple', linewidth = 1, linestyle = '--')
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.grid(color = 'gray', linewidth = 0.5, alpha = 0.5)
        ax2.plot(df.index, df["Close"])
        ax2.tick_params(axis='x', labelbottom=False)
        for signal in buy_signals:
            ax2.arrow(
                df.index[signal], df["Close"][signal]*0.9, 0, df["Close"][signal]*0.025,
            width=2, color='green', alpha = 0.25, linewidth=0)

        for signal in sell_signals:
            ax2.arrow(
                df.index[signal], df["Close"][signal]*1.1, 0, -df["Close"][signal]*0.025,
            width=2, color='red', alpha = 0.25, linewidth=0)
        ax2.grid(color = 'gray', linewidth = 0.5, alpha = 0.5)
        plt.savefig(f"plots/portfolio_value_{iteration}.png", dpi=300)
        plt.close('all')

    return {'portfolio_value' : info["portfolio_value"],
            'episode_reward' : episode_reward}
