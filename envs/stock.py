""" StockData class """

import yfinance as yf
from ta import trend, momentum, volatility, volume

class Stock:
    """ StockData class """
    def __init__(self, symbol, start_date, end_date, offset_history = 200):
        """
        Initialize the StockData object.

        Args:
        symbol (str): Ticker symbol for the stock.
        start_date (str): Start date in 'YYYY-MM-DD' format for fetching historical data.
        end_date (str): End date in 'YYYY-MM-DD' format for fetching historical data.
        """
        self.symbol = symbol
        self.offset_history = offset_history
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

        # self.data["adx_short"] = trend.ADXIndicator(
        #     self.data["High"],
        #     self.data["Low"],
        #     self.data["Close"],
        #     window=short_window,
        #     fillna=False
        # ).adx()

        # self.data["adx_long"] = trend.ADXIndicator(
        #     self.data["High"],
        #     self.data["Low"],
        #     self.data["Close"],
        #     window=long_window,
        #     fillna=False
        # ).adx()

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
            window = 50,
            fillna = False
        )

        self.data["wma_long"] = trend.wma_indicator(
            self.data["Close"],
            window = 200,
            fillna = False
        )

        self.data["bollinger_h"] = volatility.bollinger_hband(
            self.data["Close"],
            window = 200,
            fillna = False
        )
        self.data["bollinger_l"] = volatility.bollinger_lband(
            self.data["Close"],
            window = 200,
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

        self.data["stoch_osc"] = momentum.stoch(
            self.data['High'],
            self.data['Low'],
            self.data["Close"],
            window=short_window,
            smooth_window=3,
            fillna=False
        )
        self.data["obv"] = volume.on_balance_volume(
            self.data["Close"],
            self.data["Volume"],
            fillna=False
        )
        self.data["mfi"] = volume.money_flow_index(
            self.data["High"],
            self.data["Low"],
            self.data["Close"],
            self.data["Volume"],
            window = short_window,
            fillna = False
        )

        self.data = self.data.iloc[self.offset_history:]
        self.data = self.data.fillna(-1)

        self.data.to_csv(f'./data/{self.symbol}.csv', index = True)
