# from typing import List, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
# import mplfinance as mpf
import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter
# from ta.momentum import RSIIndicator
from alpha_vantage.timeseries import TimeSeries

"""
# Calculate the 14-day Relative Strength Index (RSI)
rsi = RSIIndicator(data['Close']).rsi()

# Combine the stock price data and RSI data into a single DataFrame
data['RSI'] = rsi
"""

# Replace 'YOUR_API_KEY' with your actual API key
api_key = 'G83PZ65IL4NMNGDC'

# Replace 'AAPL' with the desired stock symbol
symbol = 'TQQQ'

# Initialize Alpha Vantage API client
ts = TimeSeries(key=api_key, output_format='pandas')

# Retrieve 5-minute intraday data
data, meta_data = ts.get_intraday(symbol=symbol, interval='1min', outputsize='full')
data = data.rename(columns={
   '1. open': 'Open',
   '2. high': 'High',
   '3. low': 'Low',
   '4. close': 'Close',
   '5. volume': ' Volume',
})
# Download ticker data
ticker = "TQQQ"  # Ticker symbol (e.g., "AAPL" for Apple Inc.)
start_time = "2021-01-01"
end_time = "2023-05-22"
# Define backtesting parameters
INITIAL_CAPITAL = 20000.0  # Initial capital (USD)

short_window = 50
long_window = 200
def backtest(data, ticker = "TQQQ",
             start_time = start_time,
             end_time = end_time,
             short_window = short_window,
             long_window = long_window):

    positions = {"cash" : INITIAL_CAPITAL, ticker : 0}
    portfolio_values = []
    # data: pd.DataFrame = yf.download(ticker, start=start_time, end=end_time)

    # Calculate short- and long-term MA
    data['SMA_Short'] = data['Close'].rolling(window=50).mean()
    data['LMA_Long'] = data['Close'].rolling(window=155).mean()

    # Check for crossover signals
    data['Signal'] = 0
    data.loc[data['SMA_Short'] > data['LMA_Long'], 'Signal'] = 1
    data.loc[data['SMA_Short'] < data['LMA_Long'], 'Signal'] = -1


    # Backtesting loop
    for i in range(len(data)):
        # Check for a buy signal
        # if data['RSI'][i] < 30:
        if data["Signal"][i] == 1:
            buy_price = data['Close'][i]
            if positions["cash"] > buy_price:
                n = positions["cash"]//buy_price
                positions[ticker] += n
                positions["cash"] -= n * buy_price

        # Check for a sell signal
        # elif data['RSI'][i] > 70:
        elif data["Signal"][i] == -1:
            sell_price = data['Close'][i]
            if positions[ticker] > 0:
                positions["cash"] += sell_price * positions[ticker]
                positions[ticker] = 0
        if positions[ticker] * data['Close'][i] + positions["cash"] < 0:
            print('Error')
        portfolio_values.append(positions[ticker] * data['Close'][i] + positions["cash"])
    data["portfolio_values"] = portfolio_values
    returns = data["portfolio_values"].pct_change()

    def sharpe_ratio(R, risk_free_rate=0):
        excess_returns = R - risk_free_rate
        mean_excess_returns = excess_returns.mean()
        std_excess_returns = excess_returns.std()
        ratio = mean_excess_returns / std_excess_returns
        return ratio

    sharpe_ratio_value = sharpe_ratio(returns) #, risk_free_rate = data['Close'].pct_change())
    return sharpe_ratio_value * 100

_plot = False

for short_window in range(10, 100, 20):
    for long_window in range(20, 200, 20):
        if long_window > short_window:
            sharpe_ratio = backtest(data, short_window = short_window,
                                    long_window = long_window)
            print("[%d, %d, %0.2f]"%(short_window, long_window, sharpe_ratio))

if _plot:
    fig = plt.figure(figsize=(8, 8), dpi = 300)

    # # Candlestick chart with RSI subplot
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2, colspan=1)
    ax2 = plt.subplot2grid((5, 1), (2, 0), rowspan=2, colspan=1, sharex=ax1)
    ax3 = plt.subplot2grid((5, 1), (4, 0), rowspan=1, colspan=1, sharex=ax2)

    # ax1.plot(data.index, portfolio_values, linewidth = 0.5)
    ax1.axhline(INITIAL_CAPITAL, linewidth = 2, alpha = 0.4, color = 'gray')
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.fill_between(data.index, INITIAL_CAPITAL, data.portfolio_values,
                    where = np.array(data.portfolio_values) > INITIAL_CAPITAL,
                    facecolor='forestgreen', alpha=0.2)
    ax1.fill_between(data.index, INITIAL_CAPITAL, data.portfolio_values,
                    where = np.array(data.portfolio_values) < INITIAL_CAPITAL,
                    facecolor='orangered', alpha=0.2)
    # ax2.plot(data.index, data["Open"], "bo-", label="Open")
    # ax2.plot(data.index, data["High"], "ro-", label="High")
    # ax2.plot(data.index, data["Low"], "go-", label="Low")
    # ax2.plot(data.index, data["Close"], "ko-", label="Close")


    ax2.plot(data.index, data['Close'], linewidth = 1, color = 'k')
    ax2.plot(data.index, data['SMA_Short'], linestyle = 'dashed', color = 'orchid', linewidth = 0.5)
    ax2.plot(data.index, data['LMA_Long'], linestyle = 'dashed', color = 'plum', linewidth = 0.5)
    ax2.fill_between(data.index, 0, data['Close'],
                    where = data["Signal"] == 1,
                    facecolor='lime', alpha=0.2)
    ax2.fill_between(data.index, 0, data['Close'],
                    where = data["Signal"] == -1,
                    facecolor='crimson', alpha=0.2)
    # ax2.axvline(data[data['Signal'] == 1].index, facecolor='green', alpha=0.2)
    # ax2.axvline(data[data['Signal'] == -1].index, facecolor='indianred', alpha=0.2)
    # ax2.plot(data.index, data['Signal']*20, color = 'yellow', linewidth = 5, alpha = 0.5)
    ax2.tick_params(axis='x', labelbottom=False)
    # # Plot the RSI
    ax3.plot(data.index, data['RSI'], color='purple', linewidth=1)
    ax3.axhline(30, color='blue', linestyle='--', linewidth=0.8)
    ax3.axhline(70, color='red', linestyle='--', linewidth=0.8)
    ax3.fill_between(data.index, 30, 70, where=(data['RSI'] >= 70), facecolor='red', alpha=0.2)
    ax3.fill_between(data.index, 30, 70, where=(data['RSI'] <= 30), facecolor='green', alpha=0.2)
    ax3.set_ylabel('RSI')
    ax3.set_xlabel('Date')

    # # Position account value subplot
    # ax3 = plt.subplot2grid((6, 1), (4, 0), rowspan=1, colspan=1, sharex=ax1)
    # ax3.plot(positions_df['Date'], positions_df['Profit'].cumsum(), color='orange', linewidth=1)
    # ax3.set_ylabel('Account Value')

    # # Adjust spacing between subplots
    # plt.subplots_adjust(hspace=0.5)

    # # Show the plot
    plt.savefig("%s_lamb.png", dpi = 300)%ticker