import pandas as pd
import yfinance as yf
from ta import trend
from ta import momentum
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
from ray.rllib.env.env_context import EnvContext

class Stock:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.data = yf.download(symbol, start=start_date, end=end_date)# load data of symbol
        

class TradeEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.symbols = config.get("symbols", ["TQQQ", "SOXL"])
        self.initial_balance = config.get("initial_balance", 10000)
        self.start_date = config.get("start_date", "2016-01-01")
        self.end_date = config.get("end_date", " 2023-01-01")      
        self.action = []
        self.stocks = {}
        scaler = MinMaxScaler()
        for s in self.symbols:
            self.stocks[s] = Stock(
                symbol = s, 
                start_date = self.start_date, 
                end_date = self.end_date
                )
            self.stocks[s].data['rsi'] = momentum.rsi(self.stocks[s].data['Close'], 14, False)
            self.stocks[s].data['adx'] = trend.ADXIndicator(self.stocks[s].data['High'], self.stocks[s].data['Low'], self.stocks[s].data['Close'], 20, False).adx()
            self.stocks[s].data['macd'] = trend.MACD(self.stocks[s].data['Close'], window_slow = 26, window_fast= 12, window_sign = 9, fillna = False).macd()

            self.stocks[s].data.fillna(0, inplace = True)
        
        self.time_idx = 0
        self.episode_length = len(self.stocks[self.symbols[0]].data)
        self.num_states = 4 * len(self.symbols) + 1
        self.action_space = Box(0.0, 1.0, shape=(len(self.symbols) + 1,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_states,), dtype=np.float32)

        self.reset()
        
    def reset(self, *, seed=None, options=None):
        self.balance = self.initial_balance
        self.portfolio = {key: 0 for key in self.symbols}
        self.portfolio["balance"] = self.balance
        self.portfolio_value = self.balance
        self.time_idx = 0
        self.action = [0 for _ in range(len(self.symbols))] + [1]
        info = self._get_info()
        observation = self._get_obs()
        return observation, info

    
    def step(self, action):
        self.action = [a/(np.sum(action) + 0.0001) for a in action]

        self._get_portfolio_value()
        self.portfolio["balance"] = self.portfolio_value
        for i, s in enumerate(self.symbols):
            self.portfolio[s] = self.portfolio_value * self.action[i]/self.stocks[s].data['Close'][self.time_idx]
            self.portfolio["balance"] -= self.portfolio_value * self.action[i]
        self.action[len(self.symbols)] = self.portfolio["balance"]/self.portfolio_value

        self.prev_portfolio_value = self.portfolio_value
                
        obs = self._get_obs()
        # [TODO] fix the reward 
        self.time_idx += 1
        self._get_portfolio_value()
        reward = self.portfolio_value - self.prev_portfolio_value
        info = self._get_info()
        
        done = truncated = self.time_idx == self.episode_length - 1

        return obs, reward, done, truncated, info
            
    def _get_portfolio_value(self):
        self.portfolio_value = 0
        for s in self.symbols:
            self.portfolio_value += self.portfolio[s] * self.stocks[s].data['Close'][self.time_idx]
        self.portfolio_value += self.portfolio["balance"]
        self.portfolio_value = max(self.portfolio_value, 0)
    
    def _get_obs(self):
        obs = []
        # obs.extend(self.action[:])
        for i, s in enumerate(self.symbols):
            obs.append(self.action[i])
        obs.append(self.action[len(self.symbols)])

        for s in self.symbols:
            obs.append(self.stocks[s].data['rsi'][self.time_idx])
            obs.append(self.stocks[s].data['adx'][self.time_idx])
            obs.append(self.stocks[s].data['macd'][self.time_idx])
        return obs
    
    def _get_info(self):
        return {"sharpe_ratio" : 1,
                "portfolio_value" : self.portfolio_value,
                "portions" : self.action,
                "all_portions" : np.sum(self.action)}
    
    
    
