"""
    This contains the trade environment using gym.Env
    It brings DRL trading engine design

"""

import yfinance as yf
from ta import trend, momentum, volatility, volume
import ta
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import random
import pandas as pd
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
            symbol, start=start_date, end=end_date)  # load data of symbol
        long_window = 28
        short_window = 14
        self.data["rsi_short"] = momentum.rsi(
            self.data["Close"], 
            window=short_window,
            fillna=False)

        self.data["rsi_long"] = momentum.rsi(
            self.data["Close"], 
            window=long_window,
            fillna=False)

        self.data["cci_short"] = trend.cci(
            self.data["High"],
            self.data["Low"],
            self.data["Close"],
            window=short_window,
            fillna=False)

        self.data["cci_long"] = trend.cci(
            self.data["High"],
            self.data["Low"],
            self.data["Close"],
            window=long_window,
            fillna=False)

        self.data["roc_short"] = momentum.roc(
            self.data["Close"], 
            window=short_window,
            fillna=False)

        self.data["roc_long"] = momentum.roc(
            self.data["Close"], 
            window=long_window,
            fillna=False)

        self.data["adx_short"] = trend.ADXIndicator(
            self.data["High"], 
            self.data["Low"], 
            self.data["Close"], 
            window=short_window, 
            fillna=False
        ).adx()

        self.data["adx_long"] = trend.ADXIndicator(
            self.data["High"], 
            self.data["Low"], 
            self.data["Close"], 
            window=long_window, 
            fillna=False
        ).adx()

        self.data["adx"] = trend.ADXIndicator(
            self.data["High"], 
            self.data["Low"], 
            self.data["Close"], 
            window=20, 
            fillna=False
        ).adx()

        self.data["macd"] = trend.MACD(
            self.data["Close"],
            window_slow=26,
            window_fast=12,
            window_sign=9,
            fillna=False,
        ).macd()
        
        self.data["wma_short"] = trend.wma_indicator(
            self.data["Close"],
            window = short_window,
            fillna = False
        )

        self.data["wma_long"] = trend.wma_indicator(
            self.data["Close"],
            window = long_window,
            fillna = False
        )
        
        self.data["bollinger"] = volatility.bollinger_hband(
            self.data["Close"], 
            window = short_window, 
            window_dev = 2,
            fillna = False
        )
        
        self.data["donchian"] = volatility.donchian_channel_wband(
            self.data["High"],
            self.data["Low"],
            self.data["Close"], 
            window = short_window, 
            offset = 0,
            fillna = False
        )
        
        self.data["ulcer"] = volatility.ulcer_index(
            self.data["Close"],
            window = short_window,
            fillna = False            
        )

        self.data["ease_of_movement"] = volume.ease_of_movement(
            self.data["High"], 
            self.data["Low"],
            self.data["Volume"],
            window = short_window, 
            fillna = False
        )
        
        self.data["mfi"] = volume.money_flow_index(
            self.data["High"], 
            self.data["Low"],
            self.data["Close"],
            self.data["Volume"],
            window = short_window, 
            fillna = False
        )
        
        self.data.fillna(-1, inplace=True)


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
        self.obs_interval = config["obs_interval"]
        self.symbols = [random.choice(config['symbols'])]
        self.init_balance = config["initial_balance"]
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.action = []
        self.stocks = {}

        for symbol in config['symbols']:
            try:
                self.stocks[symbol] = Stock(
                    symbol=symbol, start_date=self.start_date, end_date=self.end_date
                )
            except:
                print('\033[91m Could not load %s from yfinance successfully\033[0m'%symbol)
                if symbol in self.stocks:
                    del self.stocks[symbol]
        self.time_idx = 0
        self.episode_length = len(self.stocks[config["symbols"][0]].data)
        self.num_symbols = len(self.symbols)
        self.num_states = 2 + (2 + len(self.obs_components) * self.obs_interval) * self.num_symbols
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
        self.symbols = [random.choice(self.config["symbols"])]
        # self.stocks = {}
        # for symbol in self.symbols:
        #     self.stocks[symbol] = Stock(
        #         symbol=symbol, start_date=self.start_date, end_date=self.end_date
        #     )
        self.portfolio = {key: 0 for key in self.symbols}
        self.portfolio["balance"] = self.init_balance
        self.portfolio_value = self.init_balance
        self.prev_portfolio_value = self.init_balance
        self.time_idx = 0
        self.action = [0 for _ in range(len(self.symbols))] + [1]
        self.prev_action = [[0 for _ in range(len(self.symbols))] + [1], 
                            [0 for _ in range(len(self.symbols))] + [1]]
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
        self.prev_action.append(self.action)
        self.prev_action.pop(0)
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
        # obs.append(self.portfolio_value)
        for symbol in self.symbols:
            for time_id in range(self.time_idx - self.obs_interval + 1, self.time_idx + 1):
                for col in self.obs_components:
                    if time_id >= 0:
                        if col in ["wma_short", "wma_long", "Close"]:
                            scale = 1/self.stocks[symbol].data["wma_long"][self.time_idx]
                        else:
                            scale = 1
                        obs.append(scale * self.stocks[symbol].data[col][time_id])
                    else:
                        obs.append(-1)
        obs.extend(self.prev_action[0])
        obs.extend(self.prev_action[1])
        return obs

    def _get_info(self):
        return {
            "sharpe_ratio": 1,
            "portfolio_value": self.portfolio_value,
            "portions": self.action,
            "symbols": self.symbols
        }
