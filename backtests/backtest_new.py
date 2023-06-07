import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from alpha_vantage.timeseries import TimeSeries

symbol = "TQQQ"  # symbol symbol (e.g., "AAPL" for Apple Inc.)
start_time = "2021-01-01"
end_time = "2023-05-22"

data: pd.DataFrame = yf.download(symbol, start=start_time, end=end_time)

# Define backtesting parameters
INITIAL_CAPITAL = 20000.0  # Initial capital (USD)

short_window = 50
long_window = 200

positions = {"cash" : INITIAL_CAPITAL, symbol : 0}
portfolio_values = []

# Calculate the 14-day Relative Strength Index (RSI)
rsi = RSIIndicator(data['Close']).rsi()

# Combine the stock price data and RSI data into a single DataFrame
data['RSI'] = rsi

# Calculate short- and long-term MA
data['SMA_Short'] = data['Close'].rolling(window=50).mean()
data['LMA_Long'] = data['Close'].rolling(window=200).mean()

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
            positions[symbol] += n
            positions["cash"] -= n * buy_price

    # Check for a sell signal
    # elif data['RSI'][i] > 70:
    elif data["Signal"][i] == -1:
        sell_price = data['Close'][i]
        if positions[symbol] > 0:
            positions["cash"] += sell_price * positions[symbol]
            positions[symbol] = 0
    if positions[symbol] * data['Close'][i] + positions["cash"] < 0:
        print('Error')
    portfolio_values.append(positions[symbol] * data['Close'][i] + positions["cash"])
data["portfolio_values"] = portfolio_values
returns = data["portfolio_values"].pct_change()

def sharpe_ratio(R, risk_free_rate=0):
    excess_returns = R - risk_free_rate
    mean_excess_returns = excess_returns.mean()
    std_excess_returns = excess_returns.std()
    ratio = mean_excess_returns / std_excess_returns
    print(ratio)
    return ratio

sharpe_ratio_value = sharpe_ratio(returns)


fig = plt.figure(figsize=(8, 8), dpi = 300)
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


ax2.plot(data.index, data['Close'], linewidth = 1, color = 'k')
ax2.plot(data.index, data['SMA_Short'], linestyle = 'dashed', color = 'orchid', linewidth = 0.5)
ax2.plot(data.index, data['LMA_Long'], linestyle = 'dashed', color = 'plum', linewidth = 0.5)
ax2.fill_between(data.index, 0, data['Close'],
                where = data["Signal"] == 1,
                facecolor='lime', alpha=0.2)
ax2.fill_between(data.index, 0, data['Close'],
                where = data["Signal"] == -1,
                facecolor='crimson', alpha=0.2)
ax2.tick_params(axis='x', labelbottom=False)

# Plot the RSI
ax3.plot(data.index, data['RSI'], color='purple', linewidth=1)
ax3.axhline(30, color='blue', linestyle='--', linewidth=0.8)
ax3.axhline(70, color='red', linestyle='--', linewidth=0.8)
ax3.fill_between(data.index, 30, 70, where=(data['RSI'] >= 70), facecolor='red', alpha=0.2)
ax3.fill_between(data.index, 30, 70, where=(data['RSI'] <= 30), facecolor='green', alpha=0.2)
ax3.set_ylabel('RSI')
ax3.set_xlabel('Date')

# Show the plot
plt.savefig("logs/%s_backtest.png"%symbol, dpi = 300)