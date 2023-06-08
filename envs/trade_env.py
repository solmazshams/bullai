"""
    This contains the trade environment using gym.Env
    It brings DRL trading engine design

"""

import yfinance as yf
from ta import trend, momentum
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

class Stock:
    """ StockData class """
    def __init__(self, symbol, start_date, end_date):
        """
        Initialize the StockData object.

        Args:
        symbol (str): Ticker symbol for the stock.
        start_date (str): Start date in 'YYYY-MM-DD' format for fetching historical data.
        end_date (str): End date in 'YYYY-MM-DD' format for fetching historical data.
        """
        self.symbol = symbol
        self.data = yf.download(
            symbol, start=start_date, end=end_date
        )  # load data of symbol

        self.data["rsi"] = momentum.rsi(self.data["Close"], 14, False)

        self.data["adx"] = trend.ADXIndicator(
            self.data["High"], self.data["Low"], self.data["Close"], 20, False
        ).adx()

        self.data["macd"] = trend.MACD(
            self.data["Close"],
            window_slow=26,
            window_fast=12,
            window_sign=9,
            fillna=False,
        ).macd()

        self.data.fillna(0, inplace=True)


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
        self.symbols = config["symbols"]
        self.init_balance = config["initial_balance"]
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.action = []
        self.stocks = {}

        for symbol in self.symbols:
            self.stocks[symbol] = Stock(
                symbol=symbol, start_date=self.start_date, end_date=self.end_date
            )

        self.time_idx = 0
        self.episode_length = len(self.stocks[self.symbols[0]].data)
        self.num_symbols = len(self.symbols)
        self.num_states = 4 * self.num_symbols + 1
        self.num_actions = self.num_symbols + 1

        self.action_space = Box(0.0, 1.0, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.num_states,), dtype=np.float32
        )
        self.portfolio_value = self.init_balance
        self.prev_portfolio_value = self.init_balance
        self.reset()

    def reset(self, *, seed = None, options = None):
        """
        Resets the environment to an initial internal state, returning an initial observation and info.
        """

        self.portfolio = {key: 0 for key in self.symbols}
        self.portfolio["balance"] = self.init_balance
        self.portfolio_value = self.init_balance
        self.prev_portfolio_value = self.init_balance
        self.time_idx = 0
        self.action = [0 for _ in range(len(self.symbols))] + [1]

        info = self._get_info()
        observation = self._get_obs()

        return observation, info

    def step(self, action):
        """Run one timestep of the trade environment using the agent actions."""

        self.action = [a / (np.sum(action) + 0.0001) for a in action]

        self._get_portfolio_value()
        self.portfolio["balance"] = self.portfolio_value
        for i, symbol in enumerate(self.symbols):
            self.portfolio[symbol] = (
                self.portfolio_value
                * self.action[i]
                / self.stocks[symbol].data["Close"][self.time_idx]
            )
            self.portfolio["balance"] -= self.portfolio_value * self.action[i]
        self.action[len(self.symbols)] = (
            self.portfolio["balance"] / self.portfolio_value
        )

        self.prev_portfolio_value = self.portfolio_value

        obs = self._get_obs()
        # [TODO] fix the reward
        self.time_idx += 1
        self._get_portfolio_value()
        reward = self.portfolio_value - self.prev_portfolio_value
        info = self._get_info()

        done = truncated = self.time_idx == self.episode_length - 1

        return obs, reward, done, truncated, info

    def render(self):
        pass

    def _get_portfolio_value(self):
        self.portfolio_value = 0
        for symbol in self.symbols:
            self.portfolio_value += (
                self.portfolio[symbol] * self.stocks[symbol].data["Close"][self.time_idx]
            )
        self.portfolio_value += self.portfolio["balance"]
        self.portfolio_value = max(self.portfolio_value, 0)

    def _get_obs(self):
        obs = []
        # obs.extend(self.action[:])
        for i, symbol in enumerate(self.symbols):
            obs.append(self.action[i])
        obs.append(self.action[len(self.symbols)])

        for symbol in self.symbols:
            obs.append(self.stocks[symbol].data["rsi"][self.time_idx])
            obs.append(self.stocks[symbol].data["adx"][self.time_idx])
            obs.append(self.stocks[symbol].data["macd"][self.time_idx])
        return obs

    def _get_info(self):
        return {
            "sharpe_ratio": 1,
            "portfolio_value": self.portfolio_value,
            "portions": self.action,
            "stocks": np.sum(self.action)
        }
