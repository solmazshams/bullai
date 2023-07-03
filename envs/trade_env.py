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

def sharpe(values, initial_value = 10000):
    final_return = values[-1]
    if final_return > initial_value:
        R_p = (final_return/initial_value)**(1/len(values)) - 1
        excess_return = initial_value*(1 + R_p)**np.arange(len(values))/values - 1
        s_p = np.std(excess_return)
        return R_p/s_p
    else:
        return 0
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
                symbol=symbol, start_date=self.start_date, end_date=self.end_date, indicators=self.obs_components
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
        self.values = []
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
        self.values.append(self.portfolio_value)
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


        done = truncated = self.time_idx == self.episode_length - 1
        info = self._get_info(done=done)
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
                obs.append(0)
            for time_id in range(self.time_idx - self.obs_interval + 1, self.time_idx + 1):
                for indicator in self.obs_components:
                    if time_id >= 0:
                        scale = self.stocks[self.symbols[0]].normalization_info[indicator][0]
                        bias = self.stocks[self.symbols[0]].normalization_info[indicator][1]
                        obs.append(self.df[indicator][time_id] * scale + bias)
                    else:
                        obs.append(-1)
        return obs

    def _get_info(self, done=False):
        return {
            "sharpe_ratio": sharpe(self.values, self.init_balance) if done else 0,
            "portfolio_value": self.portfolio_value,
            "symbols": self.symbols
        }
