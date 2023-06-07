import numpy as np
import matplotlib.pyplot as plt
from ta import trend
from ta import momentum
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import envs.trade_env as trade_env


symbol = "TQQQ" 
start_time = "2018-01-01"
end_time = "2023-05-22"

data_s = trade_env.Stock(symbol, start_time, end_time).data

def metric(data, n_train):
    rsi = momentum.rsi(data['Close'], 14, False)
    #Average Directional Index
    adx = trend.ADXIndicator(data['High'], data['Low'], data['Close'], 20, False).adx()
    macd = trend.MACD(data['Close'], window_slow = 26, window_fast= 12, window_sign = 9, fillna = False).macd()

    data['rsi'] = rsi
    data['adx'] = adx
    data['macd'] = macd
    data['target'] = [1 * (data['Close'][i] > data['Close'][i-1]) for i in range(1, len(data))] + [0]
    x = data[['rsi', 'adx', 'macd']].values
    y = data['target'].values.reshape(-1,)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
    X_train, X_test, y_train, y_test = x[:n_train], x[n_train:], y[:n_train], y[n_train:]
   
    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return predictions

first_money = 100000
predictions = metric(data_s, n_train = 500)
data_pred = data_s[-len(predictions):]

def backtest(data, symbol):
    positions = {"cash" : first_money, symbol : 0}
    portfolio_values = []
    for i in range(len(predictions)):
        if predictions[i] == 0:
            buy_price = data['Close'][i]
            if positions["cash"] > buy_price:
                n = positions["cash"]//buy_price
                positions[symbol] += n
                positions["cash"] -= n * buy_price
        else:
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
        return ratio

    sharpe_ratio_value = sharpe_ratio(returns) #, risk_free_rate = data['Close'].pct_change())
    return sharpe_ratio_value * 100

sharpe_ratio = backtest(data_pred, symbol)

_plot = False
if _plot:
    fig = plt.figure(figsize=(8, 8), dpi = 300)

    # # Candlestick chart with RSI subplot
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2, colspan=1)

    ax2 = plt.subplot2grid((5, 1), (2, 0), rowspan=2, colspan=1, sharex=ax1)
    ax3 = plt.subplot2grid((5, 1), (4, 0), rowspan=1, colspan=1, sharex=ax2)

    # ax1.plot(data.index, portfolio_values, linewidth = 0.5)
    ax1.axhline(first_money, linewidth = 2, alpha = 0.4, color = 'gray')
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.fill_between(data_pred.index, first_money, data_pred.portfolio_values,
                    where = np.array(data_pred.portfolio_values) > first_money,
                    facecolor='forestgreen', alpha=0.5)
    ax1.fill_between(data_pred.index, first_money, data_pred.portfolio_values,
                    where = np.array(data_pred.portfolio_values) < first_money,
                    facecolor='orangered', alpha=0.5)
    ax1.set_title("Backtest of %s Trading"%symbol)
    ax1.grid()
    
    # ax2.plot(data.index, data["Open"], "bo-", label="Open")
    # ax2.plot(data.index, data["High"], "ro-", label="High")
    # ax2.plot(data.index, data["Low"], "go-", label="Low")
    # ax2.plot(data.index, data["Close"], "ko-", label="Close")


    ax2.plot(data_pred.index, data_pred['Close'], linewidth = 1, color = 'k')
    # ax2.plot(data_pred.index, data_pred['SMA_Short'], linestyle = 'dashed', color = 'orchid', linewidth = 0.5)
    # ax2.plot(data_pred.index, data_pred['LMA_Long'], linestyle = 'dashed', color = 'plum', linewidth = 0.5)
    # ax2.fill_between(data_pred.index, 0, data['Close'],
    #                 where = data_pred["Signal"] == 1,
    #                 facecolor='lime', alpha=0.2)
    # ax2.fill_between(data_pred.index, 0, data['Close'],
    #                 where = data_pred["Signal"] == -1,
    #                 facecolor='crimson', alpha=0.2)
    # ax2.axvline(data[data['Signal'] == 1].index, facecolor='green', alpha=0.2)
    # ax2.axvline(data[data['Signal'] == -1].index, facecolor='indianred', alpha=0.2)
    # ax2.plot(data.index, data['Signal']*20, color = 'yellow', linewidth = 5, alpha = 0.5)
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.set_ylabel('%s _ Close Price'%symbol)
    # # Plot the RSI
    ax3.plot(data_pred.index, data_pred['rsi'], color='purple', linewidth=1)
    ax3.axhline(30, color='blue', linestyle='--', linewidth=0.8)
    ax3.axhline(70, color='red', linestyle='--', linewidth=0.8)
    # ax3.fill_between(data_pred.index, 30, 70, where=(data_pred['RSI'] >= 70), facecolor='red', alpha=0.2)
    # ax3.fill_between(data_pred.index, 30, 70, where=(data_pred['RSI'] <= 30), facecolor='green', alpha=0.2)
    ax3.set_ylabel('RSI')
    ax3.set_xlabel('Date')

    # # Position account value subplot
    # ax3 = plt.subplot2grid((6, 1), (4, 0), rowspan=1, colspan=1, sharex=ax1)
    # ax3.plot(positions_df['Date'], positions_df['Profit'].cumsum(), color='orange', linewidth=1)
    # ax3.set_ylabel('Account Value')

    # # Adjust spacing between subplots
    # plt.subplots_adjust(hspace=0.5)

    # # Show the plot
    plt.savefig("%s_lamb.png"%symbol, dpi = 300)
