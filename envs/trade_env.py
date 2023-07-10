"""
    This contains the trade environment using gym.Env
    It brings DRL trading engine design

"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from envs.stock import Stock
np.seterr(divide='ignore', invalid='ignore')
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

    def __init__(self, config, normalization_info=None):
        self.config         = config
        self.symbols        = config["symbols"]
        self.indicators     = config["indicators"]
        self.time_window    = config["time_window"]
        self.init_balance   = config["initial_balance"]
        self.balance        = config["initial_balance"]
        self.start_date     = config["start_date"]
        self.end_date       = config["end_date"]
        self.cost           = {}
        self.stocks         = []
        self.normalization_info = normalization_info

        for symbol in self.symbols:
            self.stocks.append(
                Stock(
                    symbol=symbol,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    indicators=self.indicators,
                    normalization_info=self.normalization_info
                    )
            )

        self.time_idx = 0
        self.episode_length = len(self.stocks[0].data)
        self.num_symbols = len(config["symbols"])
        self.portfolio_value = self.init_balance
        self.prev_portfolio_value = self.init_balance
        self.data = None
        self.num_shares = {key: 0 for key in self.symbols}
        self.num_actions = 3
        self.num_states = 1 + (len(self.indicators) * self.time_window) * self.num_symbols

        self.action_space = Discrete(self.num_actions)

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.num_states,), dtype=np.float32
        )
        self.portfolio_values = [self.init_balance]

        self.reset()

    def reset(self, *, seed = None, options = None):
        """
        Resets the environment to an initial internal state,
        """
        # [TODO] this works for single stock now
        self.data = self.stocks[0].data.copy()
        self.data["portfolio_values"] = self.init_balance
        self.episode_length = len(self.data)
        self.num_shares = {key: 0 for key in self.symbols}
        self.balance = self.init_balance
        self.portfolio_value = self.init_balance
        self.prev_portfolio_value = self.init_balance
        self.time_idx = 0
        self.cost = {}
        info = self._get_info()
        observation = self._get_obs()
        return observation, info

    def _set_balance(self, amount):
        self.balance = amount

    def step(self, action):
        """Run one timestep of the trade environment using the agent actions."""
        self._update_portfolio_value()
        for symbol in self.symbols:
            if action == 1: # buy
                self.num_shares[symbol] = (
                    self.portfolio_value/self.data["Close"][self.time_idx]
                    )
                self.cost[symbol] = self.data["Close"][self.time_idx]
                self._set_balance(0.0)
            elif action == 2:
                self.num_shares[symbol] = 0.0
                self._set_balance(self.portfolio_value)

        self.time_idx += 1
        self._update_portfolio_value()
        self.data.loc[self.time_idx, "portfolio_values"] = self.portfolio_value
        obs = self._get_obs()
        # reward = (
        #     self.portfolio_value - self.prev_portfolio_value
        #     )/self.episode_length/10
        reward = self.portfolio_value/self.prev_portfolio_value - 1

        done = truncated = self.time_idx == self.episode_length - 1
        info = self._get_info()
        return obs, reward, done, truncated, info

    def _update_portfolio_value(self):
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_value = 0
        for symbol in self.symbols:
            self.portfolio_value += (
                self.num_shares[symbol] * self.data["Close"][self.time_idx]
            )
        self.portfolio_value += self.balance
        self.portfolio_value = max(self.portfolio_value, 0.0)

    def _get_obs(self):
        obs = []
        for symbol in self.symbols:
            if self.num_shares[symbol] > 0:
                obs.append(self.cost[symbol]/self.data["ema_long"][self.time_idx])
            else:
                obs.append(0)
            for time_id in range(self.time_idx - self.time_window + 1, self.time_idx + 1):
                for indicator in self.indicators:
                    if time_id >= 0:
                        # scale = self.stocks[0].normalization_info[indicator][0]
                        # bias = self.stocks[0].normalization_info[indicator][1]
                        scale = 1
                        bias = 0
                        if indicator in ["Close", "Open", "Low", "High", "ema_short", "ema_long", "bollinger_h", "bollinger_l"]:
                            scale = 1/self.data.loc[self.time_idx, "ema_long"]
                            bias = 0
                        obs.append(self.data.loc[time_id, indicator] * scale + bias)
                    else:
                        obs.append(-1)
        return np.array(obs)

    def _get_info(self):
        return {
            "sharpe_ratio": self._sharpe(),
            "portfolio_value": self.portfolio_value,
            "num_shares" : self.num_shares[self.symbols[0]],
        }

    def render(self):
        pass

    def _sharpe(self):
        return_rates = self.data["portfolio_values"].pct_change()
        avg_return = return_rates.mean()
        risk_free_rate = 0.0
        std_dev = return_rates.std()
        # 252 trading days in a year
        sharpe_ratio = (avg_return - risk_free_rate) / std_dev * np.sqrt(252)
        return sharpe_ratio
