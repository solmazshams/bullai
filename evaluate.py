""" This is the evaluate function and script """
from datetime import datetime
from json import load

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from ray.rllib.algorithms.algorithm import Algorithm

matplotlib.use("Agg")
with open("./experiments/evaluation/eval_config.json", "r", encoding="utf-8") as f:
    config = load(f)


def evaluate(env, checkpoint_dir=None, render=False, iteration=0):
    """
    Evaluate the performance of a trading algorithm using a given evaluation environment.

    Args:
    env (gym.Env): Evaluation environment for the trading algorithm.
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

    obs, info = env.reset()
    done = False
    episode_reward = 0.0

    portfolio_values = []
    time_idx = 0
    buy_signals = []
    sell_signals = []

    algo = Algorithm.from_checkpoint(checkpoint_dir)
    while not done:
        # [TODO] add multiple episode evaluation
        portfolio_values.append(info["portfolio_value"])
        action = algo.compute_single_action(
            observation=obs,
            explore=False,
            policy_id="default_policy",
        )
        if action == 1 and info["num_shares"] == 0:
            buy_signals.append(time_idx)
        if action == 2 and info["num_shares"] > 0:
            sell_signals.append(time_idx)
        obs, reward, done, _, info = env.step(action)
        episode_reward += reward
        time_idx += 1

    portfolio_values.append(info["portfolio_value"])
    portfolio_values = np.array(portfolio_values)

    print(f"\033[4mEpisode done: Total reward = {episode_reward}\033[0m")
    print("Portfolio:")
    print(f"Trading:          \033[34m{config['symbols']}\033[0m")
    print(f"Start Date:       \033[32m{config['start_date']}\033[0m")
    print(f"End Date:         \033[32m{config['end_date']}\033[0m")
    print(f"Portfolio Value:  \033[33m{info['portfolio_value']}\033[0m")

    if render:
        data = env.data.copy()
        data.set_index("Date", inplace=True)
        data.index = pd.to_datetime(data.index, unit='s')
        default_investment = (
            data["Close"] / data["Close"][0] * config["initial_balance"]
        )
        sharpe_ratio = info["sharpe_ratio"]
        # data.set_index('Date', inplace = True)
        plt.figure(figsize=(10, 6), dpi=300)
        plt.style.use("dark_background")
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2, colspan=1)
        ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=2, colspan=1, sharex=ax1)

        ax1.fill_between(
            x=data.index,
            y1=config["initial_balance"],
            y2=portfolio_values,
            where=np.array(portfolio_values) > config["initial_balance"],
            facecolor="forestgreen",
            alpha=0.5,
            zorder=1,
        )
        ax1.fill_between(
            data.index,
            config["initial_balance"],
            portfolio_values,
            where=np.array(portfolio_values) < config["initial_balance"],
            facecolor="tomato",
            alpha=0.75,
            zorder=1,
        )
        ax1.set_title(f"{env.symbols[0]} with Sharpe ratio : {sharpe_ratio:0.3f}")
        ax1.plot(data.index, portfolio_values, color="w", linewidth=1)
        ax1.fill_between(
            data.index,
            config["initial_balance"],
            default_investment,
            facecolor="silver",
            alpha=0.75,
            zorder=0,
        )
        ax1.plot(data.index, default_investment, color="gray", linewidth=0.5)
        ax1.tick_params(axis="x", labelbottom=False)
        ax1.grid(color="silver", linewidth=0.5, alpha=0.5, linestyle="--")
        ax2.plot(data.index, data["Close"], linewidth=0.5, color="white")
        ax2.plot(
            data.index,
            data["ema_short"],
            color="white",
            linestyle="--",
            linewidth=0.5,
            alpha=0.25,
            zorder=-1,
        )

        ax2.plot(
            data.index,
            data["ema_long"],
            color="white",
            linestyle="-.",
            linewidth=0.5,
            alpha=0.75,
            zorder=-1,
        )
        ax2_ = ax2.twinx()
        ax2_.plot(
            data.index,
            data["rsi_long"],
            color="cyan",
            linewidth=0.5,
            alpha=0.75,
            zorder=-1,
        )
        ax2_.plot(
            data.index,
            data["macd"],
            color="purple",
            linestyle="--",
            linewidth=0.5,
            alpha=0.75,
            zorder=-1,
        )
        data_index = np.zeros((len(data.index)))

        for signal in buy_signals:
            data_index[signal:] += +1
            ax2.scatter(
                data.index[signal],
                data["Close"][signal] * 0.9,
                marker="^",
                s=60,
                color="forestgreen",
                alpha=0.5,
                linewidths=0,
            )

        for signal in sell_signals:
            data_index[signal:] += -1
            ax2.scatter(
                data.index[signal],
                data["Close"][signal] * 1.1,
                marker="v",
                s=60,
                color="tomato",
                alpha=0.5,
                linewidths=0,
            )

        long_indices = data_index > 0
        short_indices = data_index == 0
        ax2.plot(
            data.index[long_indices],
            data["Close"][long_indices],
            color="g",
            linewidth=0,
            marker=".",
            markersize=4,
            alpha=0.5,
        )
        ax2.plot(
            data.index[short_indices],
            data["Close"][short_indices],
            color="r",
            linewidth=0,
            marker=".",
            markersize=2,
            alpha=0.5,
        )

        # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        # Format the date ticks
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Change the format as needed
        # plt.xticks(fontsize = 7, rotation = 45)
        plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=7)
        ax2.grid(color="silver", linewidth=0.5, alpha=0.25, linestyle=":")
        plt.tight_layout()
        plt.savefig(f"plots/portfolio_value_{iteration}.png", dpi=300)
        plt.close("all")

    return {
        "eval_portfolio_value": info["portfolio_value"],
        "eval_episode_reward": episode_reward,
        "eval_sharpe_ratio": info["sharpe_ratio"],
    }
