from json import load

from ray.rllib.algorithms.algorithm import Algorithm
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
COLOR_BLUE = "\033[34m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_CYAN = "\033[36m"
COLOR_RESET = "\033[0m"

with open("./experiments/evaluation/eval_config.json", "r", encoding='utf-8') as f:
# with open("./experiments/default/config.json", "r", encoding='utf-8') as f:
    eval_config = load(f)

def evaluate(eval_env,
             checkpoint_dir=None,
             render = True,
             iteration = 0):


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

        plt.figure(figsize=(12, 10), dpi = 300)


        
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2, colspan=1)
        ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=2, colspan=1, sharex=ax1)

        ax1.fill_between(
            x=df.index,
            y1=eval_config["initial_balance"],
            y2=all_portfolio_values,
            where = np.array(all_portfolio_values) > eval_config["initial_balance"],
            facecolor='forestgreen', alpha=0.5)
        ax1.fill_between(df.index, eval_config["initial_balance"], all_portfolio_values,
                        where = np.array(all_portfolio_values) < eval_config["initial_balance"],
                facecolor='tomato', alpha=0.5)
        ax1.set_title(eval_env.symbols[0])
        ax1.plot(df.index, all_portfolio_values, color = 'k', linewidth = 1)
        ax1.fill_between(df.index, eval_config["initial_balance"], default_investment,
                facecolor='silver', alpha=0.25)
        ax1.plot(df.index, default_investment,
                color='gray', linewidth = 0.5)
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.grid(color = 'olive', linewidth = 0.5, alpha = 0.25, linestyle = ':')
        ax2.plot(df.index, df["Close"], linewidth = 0.5, color = 'black')
        ax2.plot(df.index, df["wma_short"], color = 'black', linestyle = '--',linewidth = 0.5, alpha = 0.25, zorder = -1)
        ax2.plot(df.index, df["wma_long"], color = 'black', linewidth = 0.5, linestyle = '-.', alpha = 0.25, zorder = -1)

        for signal in buy_signals:
            ax2.scatter(
                df.index[signal],
                df["Close"][signal]*0.9,
                marker = '^',
                s = 40,
                color='forestgreen',
                alpha = 0.25,
                linewidths = 0)

        for signal in sell_signals:
            ax2.scatter(
                df.index[signal],
                df["Close"][signal]*1.1,
                marker = 'v',
                s = 40,
                color='tomato',
                alpha = 0.25,
                linewidths = 0)

        ax2.grid(color = 'olive', linewidth = 0.5, alpha = 0.25, linestyle = ':')
        plt.tight_layout()
        plt.savefig(f"plots/portfolio_value_{iteration}.png", dpi=300)
        plt.close('all')
        
        if iteration == 0:
            fig, ax = plt.subplots(figsize=(12, 8), dpi = 300)
            cmap = plt.cm.get_cmap('tab20')
            num_colors = 20
            custom_colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
            ax.set_prop_cycle(cycler(color=custom_colors))

            for indicator in eval_config["obs_components"]:
                if indicator in ["wma_short", "wma_long", "Close", "Open", "High", "Low", "bollinger_l", "bollinger_h"]:
                    scale = 1/df["wma_long"]
                else:
                    scale = 1
                if indicator == "obv":
                    scale = 1/df["Volume"]/20
                if indicator in [ "rsi_short", "rsi_long", "roc_long" , "adx", "stoch_osc", "mfi"]:
                    scale = 1/100
                if indicator in ["cci_long", "cci_short"]:
                    scale = 1/500
                if indicator=="macd":
                    scale = 1/25
                
                print(indicator, " : ", (scale * df[indicator]).max())
                ax.plot(scale * df[indicator], label = indicator, linewidth = 0.5)
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"plots/scaled_features.png", dpi = 300)

    return {'eval_portfolio_value' : info["portfolio_value"],
            'eval_episode_reward' : episode_reward}
