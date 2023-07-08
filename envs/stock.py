""" StockData class """

# import yfinance as yf
from ta import trend, momentum, volatility, volume
import numpy as np
import pandas as pd
class Stock:
    """ StockData class """
    def __init__(self, symbol, start_date, end_date,
                 indicators = None,
                 offset_history = 200,
                 normalization_info = None):
        """
        Initialize the StockData object.

        Args:
        symbol (str): Ticker symbol for the stock.
        start_date (str): Start date in 'YYYY-MM-DD' format for fetching historical data.
        end_date (str): End date in 'YYYY-MM-DD' format for fetching historical data.
        """
        self.symbol = symbol
        self.offset_history = offset_history
        self.indicators = indicators
        self.normalization_info = normalization_info
        # self.data = yf.download(
        #     symbol, start=start_date, end=end_date)  # load data of symbol
        # self.data.to_csv(f'./data/{self.symbol}.csv', index = True)
        self.data = pd.read_csv(f'./data/{self.symbol}.csv')
        long_window = 26
        short_window = 12
        self.data.loc[:, "rsi_short"] = momentum.rsi(
            self.data["Close"],
            window=6,
            fillna=False)

        self.data.loc[:, "rsi_long"] = momentum.rsi(
            self.data["Close"],
            window=long_window,
            fillna=False)

        self.data.loc[:, "cci_short"] = trend.cci(
            self.data["High"],
            self.data["Low"],
            self.data["Close"],
            window=short_window,
            fillna=False)

        self.data.loc[:, "cci_long"] = trend.cci(
            self.data["High"],
            self.data["Low"],
            self.data["Close"],
            window=long_window,
            fillna=False)

        self.data.loc[:, "roc_short"] = momentum.roc(
            self.data["Close"],
            window=short_window,
            fillna=False)

        self.data.loc[:, "roc_long"] = momentum.roc(
            self.data["Close"],
            window=long_window,
            fillna=False)

        self.data.loc[:, "adx_short"] = trend.ADXIndicator(
            self.data["High"],
            self.data["Low"],
            self.data["Close"],
            window=short_window,
            fillna=False
        ).adx()

        self.data.loc[:, "adx_long"] = trend.ADXIndicator(
            self.data["High"],
            self.data["Low"],
            self.data["Close"],
            window=long_window,
            fillna=False
        ).adx()

        self.data.loc[:, "adx"] = trend.ADXIndicator(
            self.data["High"],
            self.data["Low"],
            self.data["Close"],
            window=20,
            fillna=False
        ).adx()

        self.data.loc[:, "macd"] = trend.MACD(
            self.data["Close"],
            window_slow=26,
            window_fast=12,
            window_sign=9,
            fillna=False,
        ).macd()

        self.data.loc[:, "wma_short"] = trend.wma_indicator(
            self.data["Close"],
            window = 50,
            fillna = False
        )

        self.data.loc[:, "wma_long"] = trend.wma_indicator(
            self.data["Close"],
            window = 200,
            fillna = False
        )

        self.data.loc[:, "bollinger_h"] = volatility.bollinger_hband(
            self.data["Close"],
            window = 200,
            fillna = False
        )
        self.data.loc[:, "bollinger_l"] = volatility.bollinger_lband(
            self.data["Close"],
            window = 200,
            fillna = False
        )
        self.data.loc[:, "donchian"] = volatility.donchian_channel_wband(
            self.data["High"],
            self.data["Low"],
            self.data["Close"],
            window = short_window,
            offset = 0,
            fillna = False
        )

        self.data.loc[:, "ulcer"] = volatility.ulcer_index(
            self.data["Close"],
            window = short_window,
            fillna = False
        )

        self.data.loc[:, "ease_of_movement"] = volume.ease_of_movement(
            self.data["High"],
            self.data["Low"],
            self.data["Volume"],
            window = short_window,
            fillna = False
        )

        self.data.loc[:, "stoch_osc"] = momentum.stoch(
            self.data['High'],
            self.data['Low'],
            self.data["Close"],
            window=short_window,
            smooth_window=3,
            fillna=False
        )
        self.data.loc[:, "obv"] = volume.on_balance_volume(
            self.data["Close"],
            self.data["Volume"],
            fillna=False
        )
        self.data.loc[:, "mfi"] = volume.money_flow_index(
            self.data["High"],
            self.data["Low"],
            self.data["Close"],
            self.data["Volume"],
            window = short_window,
            fillna = False
        )

        self.data = self.data.iloc[self.offset_history:]
        self.data = self.data.fillna(-1)

        # self.data.to_csv(f'./data/{self.symbol}_TA.csv', index = True)
        self.data = self.data.loc[
            (self.data['Date'] >= start_date) & (self.data['Date'] <= end_date)].copy()
        if self.normalization_info is None:
            print("initial normalization on training data!")
            self._normalize()
        self.data.reset_index(drop = True, inplace=True)

    def _normalize(self):
        self.normalization_info = {}

        for indicator in self.indicators:
            min_value = np.min(self.data[indicator])
            max_value = np.max(self.data[indicator])
            scale_factor = 2/(max_value - min_value)
            bias_factor = - min_value * scale_factor - 1
            self.normalization_info[indicator] = (scale_factor, bias_factor)
