""" This is the evaluate function and script """
from json import load

from ray.rllib.algorithms.algorithm import Algorithm
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


with open("./experiments/evaluation/eval_config.json", "r", encoding='utf-8') as f:
    eval_config = load(f)
def sharpe(values, initial_value = 10000):
    final_return = values[-1]
    if final_return > initial_value:
        R_p = (final_return/initial_value)**(1/len(values)) - 1
        excess_return = initial_value*(1 + R_p)**np.arange(len(values))/values - 1
        s_p = np.std(excess_return)
        return R_p/s_p
    else:
        return 0
def evaluate(eval_env,
             checkpoint_dir=None,
             render = True,
             iteration = 0):
    """
    Evaluate the performance of a trading algorithm using a given evaluation environment.

    Args:
    eval_env (gym.Env): Evaluation environment for the trading algorithm.
    checkpoint_dir (str, optional): Directory path for loading model checkpoints. Default is None.
    render (bool, optional): Whether to render the evaluation process. Default is True.
    iteration (int, optional): Iteration number or identifier for the evaluation. Default is 0.

    Returns:
    float: Total episode reward earned during the evaluation.

    Notes:
    - The evaluation environment should have a compatible observation and actions
    - If `checkpoint_dir` is provided, the model will be loaded from it
    - If `render` is True, the evaluation process will be visualized.
    """

    obs, info = eval_env.reset()
    done = False
    episode_reward = 0.0

    all_portfolio_values = []
    time_idx = 0
    data = eval_env.df
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
    print(f"Trading:          \033[34m{eval_config['symbols']}\033[0m")
    print(f"Start Date:       \033[32m{eval_config['start_date']}\033[0m")
    print(f"End Date:         \033[32m{eval_config['end_date']}\033[0m")
    print(f"Portfolio Value:  \033[33m{info['portfolio_value']}\033[0m")

    if render:
        default_investment = data['Close']/data['Close'][0]  * eval_config["initial_balance"]
        sharpe_ratio = sharpe(all_portfolio_values)
        plt.figure(figsize=(10, 6), dpi = 300)



        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2, colspan=1)
        ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=2, colspan=1, sharex=ax1)

        ax1.fill_between(
            x=data.index,
            y1=eval_config["initial_balance"],
            y2=all_portfolio_values,
            where = np.array(all_portfolio_values) > eval_config["initial_balance"],
            facecolor='forestgreen', alpha=0.5)
        ax1.fill_between(data.index, eval_config["initial_balance"], all_portfolio_values,
                        where = np.array(all_portfolio_values) < eval_config["initial_balance"],
                facecolor='tomato', alpha=0.5)
        ax1.set_title(f"{eval_env.symbols[0]} with Sharpe ratio : {sharpe_ratio:0.3f}")
        ax1.plot(data.index, all_portfolio_values, color = 'k', linewidth = 1)
        ax1.fill_between(data.index, eval_config["initial_balance"], default_investment,
                facecolor='silver', alpha=0.25)
        ax1.plot(data.index, default_investment,
                color='gray', linewidth = 0.5)
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.grid(color = 'olive', linewidth = 0.5, alpha = 0.25, linestyle = ':')
        ax2.plot(data.index, data["Close"], linewidth = 0.5, color = 'black')
        ax2.plot(data.index, data["wma_short"],\
            color='black', linestyle='--',
            linewidth=0.5, alpha=0.25, zorder=-1)

        ax2.plot(data.index, data["wma_long"],
            color='black', linestyle='-.',
            linewidth=0.5, alpha=0.25, zorder=-1)


        for signal in buy_signals:
            ax2.scatter(
                data.index[signal],
                data["Close"][signal]*0.9,
                marker = '^',
                s = 60,
                color='forestgreen',
                alpha = 0.5,
                linewidths = 0)

        for signal in sell_signals:
            ax2.scatter(
                data.index[signal],
                data["Close"][signal]*1.1,
                marker = 'v',
                s = 60,
                color='tomato',
                alpha = 0.5,
                linewidths = 0)

        ax2.grid(color = 'olive', linewidth = 0.5, alpha = 0.25, linestyle = ':')
        plt.tight_layout()
        plt.savefig(f"plots/portfolio_value_{iteration}.png", dpi=300)
        plt.close('all')

        if iteration == 0:
            _, axes = plt.subplots(figsize=(12, 8), dpi = 300)
            cmap = plt.cm.get_cmap('tab20')
            num_colors = 20
            custom_colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
            axes.set_prop_cycle(cycler(color=custom_colors))

            for indicator in eval_config["obs_components"]:
                scale = eval_env.stocks[eval_config['symbols'][0]].normalization_info[indicator][0]
                bias = eval_env.stocks[eval_config['symbols'][0]].normalization_info[indicator][1]

                print(f"{indicator:15s}",
                    f"\n\t max:  {scale * data[indicator].max() + bias:.2f}",
                    f"\n\t min:  {scale * data[indicator].min() + bias:.2f}",
                    f"\n\t mean: {scale * data[indicator].mean() + bias:.2f}"
                    )

                axes.plot(scale * data[indicator] + bias, label = indicator, linewidth = 0.5)
            axes.legend()
            plt.tight_layout()
            plt.savefig("plots/scaled_features.png", dpi = 300)

    return {'eval_portfolio_value' : info["portfolio_value"],
            'eval_episode_reward' : episode_reward}
