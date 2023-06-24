"""
    This contains the trade environment using gym.Env
    It brings DRL trading engine design

"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import random
import pandas as pd
from envs.stock import Stock


class TradeEnv(gym.Env):
    """
    Custom Gym environment for trading.

    Attributes:
        symbols (list): List of ticker symbols for the stocks.
        init_balance (float): Initial balance for the trading environment.
        start_date (str): Start date in 'YYYY-MM-DD' format for fetching historical data.
        end_date (str): End date in 'YYYY-MM-DD' format for fetching historical data.
        action (list): List to store actions taken in the environment.
        stocks (dict): Dictionary to store Stock objects for each symbol.
    """

    def __init__(self, config):
        self.config = config
        self.obs_components = config["obs_components"]
        self.action_type = config["action_type"]
        self.obs_interval = config["obs_interval"]
        self.symbols = [random.choice(config['symbols'])]
        self.init_balance = config["initial_balance"]
        self.balance = config["initial_balance"]
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.action = []
        self.cost = {}
        self.stocks = {}

        for symbol in config['symbols']:
            self.stocks[symbol] = Stock(
                symbol=symbol, start_date=self.start_date, end_date=self.end_date
            )
        self.time_idx = 0
        self.episode_length = len(self.stocks[config["symbols"][0]].data)
        self.num_symbols = len(self.symbols)

        if self.action_type == "buy_sell_hold":
            self.num_actions = 3
            self.action_space = Discrete(self.num_actions)
        elif self.action_type == "portions":
            self.num_actions = self.num_symbols + 1
            self.action_space = Box(0.0, 1.0, shape=(self.num_actions,), dtype=np.float32)

        self.num_states = 1 + (len(self.obs_components) * self.obs_interval) * self.num_symbols

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.num_states,), dtype=np.float32
        )
        self.portfolio_value = self.init_balance
        self.prev_portfolio_value = self.init_balance
        self.df = None
        self.portfolio = {key: 0 for key in self.symbols}

        self.reset()

    def reset(self, *, seed = None, options = None):
        """
        Resets the environment to an initial internal state, returning an initial observation and info.
        """
        # self.symbols = [random.choice(self.config["symbols"])]
        self.df = self.stocks[self.symbols[0]].data
        self.episode_length = len(self.df)
        self.portfolio = {key: 0 for key in self.symbols}
        self.balance = self.init_balance
        self.portfolio_value = self.init_balance
        self.prev_portfolio_value = self.init_balance
        self.time_idx = 0
        if self.action_type == "buy_sell_hold":
            self.action = [0]
            self.prev_action = [0, 0]
        elif self.action_type == "portions":
            self.action = [0 for _ in range(len(self.symbols))] + [1]

        info = self._get_info()
        observation = self._get_obs()

        return observation, info

    def _set_balance(self, amount):
        self.balance = amount
    
    def step(self, action):
        """Run one timestep of the trade environment using the agent actions."""
        self._get_portfolio_value()

        reward = 0
        if self.action_type == "buy_sell_hold":
            if action == 0:
                # hold
                pass
            if action == 1:
                # buy
                for symbol in self.symbols:
                    # if self.portfolio[symbol] > 0:
                    #     reward -= 1/self.episode_length
                    self.portfolio[symbol] = (
                        self.portfolio_value
                        /self.df["Close"][self.time_idx]
                        /len(self.symbols)
                        )
                    self.cost[symbol] = self.df["Close"][self.time_idx]
                self._set_balance(0)
            if action == 2:
                # sell
                for symbol in self.symbols:
                    # if self.portfolio[symbol] == 0:
                    #     reward -= 1/self.episode_length
                    self.portfolio[symbol] = 0
                self._set_balance(self.portfolio_value)

        elif self.action_type == "portions":
            self.action = [a / (np.sum(action) + 0.0001) for a in action]
            self._get_portfolio_value()
            self._set_balance(self.portfolio_value)
            for i, symbol in enumerate(self.symbols):
                self.portfolio[symbol] = (
                    self.portfolio_value
                    * self.action[i]
                    / self.df["Close"][self.time_idx]
                )
                self.balance -= self.portfolio_value * self.action[i]
            self.action[len(self.symbols)] = (
                self.balance / self.portfolio_value
            )

        self.prev_portfolio_value = self.portfolio_value
        obs = self._get_obs()
        # move to next time
        self.time_idx += 1
        self._get_portfolio_value()
        reward += (self.portfolio_value - self.prev_portfolio_value)/self.episode_length

        info = self._get_info()
        done = truncated = self.time_idx == self.episode_length - 1
        return obs, reward, done, truncated, info

    def render(self):
        pass

    def _get_portfolio_value(self):
        self.portfolio_value = 0
        for symbol in self.symbols:
            self.portfolio_value += (
                self.portfolio[symbol] * self.df["Close"][self.time_idx]
            )
        self.portfolio_value += self.balance
        self.portfolio_value = max(self.portfolio_value, 0)

    def _get_obs(self):
        obs = []
        for symbol in self.symbols:
            if self.portfolio[symbol] > 0:
                obs.append(self.cost[symbol]/self.df["wma_long"][self.time_idx])
            else:
                obs.append(-1)
            for time_id in range(self.time_idx - self.obs_interval + 1, self.time_idx + 1):
                for indicator in self.obs_components:
                    if time_id >= 0:
                        # [TODO] seperate the normalization
                        if indicator in ["wma_short", "wma_long", "Close", "Open", "High", "Low", "bollinger_l", "bollinger_h"]:
                            scale = 1/self.df["wma_long"][self.time_idx]
                            bias = -1
                        else:
                            scale = 1
                            bias = 0
                        if indicator == "obv":
                            scale = 1/self.df["Volume"][time_id]/20
                            bias = 0
                        if indicator in [ "rsi_short", "rsi_long", "roc_long" , "adx", "stoch_osc", "mfi"]:
                            scale = 1/100
                            bias = -0.5
                            if indicator == "roc_long":
                                bias = 0
                            if indicator == "adx":
                                bias = -0.2
                        if indicator in ["cci_long", "cci_short"]:
                            scale = 1/500
                            bias = 0
                        if indicator=="macd":
                            scale = 1/25
                            bias = 0

                        obs.append(scale * self.df[indicator][time_id] + bias)
                    else:
                        obs.append(-1)
        return obs

    def _get_info(self):
        return {
            "sharpe_ratio": 1,
            "portfolio_value": self.portfolio_value,
            "symbols": self.symbols
        }
