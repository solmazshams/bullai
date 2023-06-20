import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
import pandas_datareader as pdr

# Define the stock symbol and timeframe
symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2022-12-31'

# Download the stock data
df = yf.download(
            symbol, start=start_date, end=end_date)  # load data of symbol
# Calculate the moving averages
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()
df = df[200:].copy()
# Add buy and sell signals
# df['BuySignal'] = df['Close'].rolling(window=20).mean() > df['Close'].rolling(window=50).mean()
# df['SellSignal'] = df['Close'].rolling(window=20).mean() < df['Close'].rolling(window=50).mean()
# [df['BuySignal']]
# Prepare the buy and sell signals as markers
# buy_markers = [mpf.make_addplot(df['Close'] + 20, type='scatter', color='g', markersize=100)]
# sell_markers = [mpf.make_addplot(df[df['SellSignal']]['Close'], type='scatter', color='r', markersize=100)]
# Add buy and sell signals
df['BuySignal'] = df['MA200'] < df['MA50']
df['SellSignal'] = df['MA200'] > df['MA50']
marker_df = pd.DataFrame(index=df.index)
marker_df['BuyMarkers'] = df[df['BuySignal']]['Close'] - 25
marker_df['SellMarkers'] = df[df['SellSignal']]['Close'] + 25
# plt.figure(figsize=(14, 9), dpi = 300)
# ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3, colspan=1)
# ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2, colspan=1)
apds = [
    mpf.make_addplot(df[['MA200', 'MA50']]),
    mpf.make_addplot(marker_df['BuyMarkers'], type='scatter', color='g', markersize=25, marker = '^', alpha = 0.5),
    mpf.make_addplot(marker_df['SellMarkers'], type='scatter', color='r', markersize=25, marker = 'v', alpha = 0.5)
]

mpf.plot(df, type='candle', volume=False, style = 'charles', title=symbol, addplot=apds, savefig = 'AAPL')
plt.close('all')

        # plt.figure(figsize=(14, 9), dpi = 300)
        # ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3, colspan=1)
        # ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2, colspan=1)
        
                # ax1.fill_between(
        #     x=df.index,
        #     y1=eval_config["initial_balance"],
        #     y2=all_portfolio_values,
        #     where = np.array(all_portfolio_values) > eval_config["initial_balance"],
        #     facecolor='forestgreen', alpha=0.5)
        # ax1.fill_between(df.index, eval_config["initial_balance"], all_portfolio_values,
        #                 where = np.array(all_portfolio_values) < eval_config["initial_balance"],
        #         facecolor='orangered', alpha=0.5)
        # ax1.set_title(eval_env.symbols[0])
        # ax1.plot(df.index, all_portfolio_values, color = 'k', linewidth = 1)
        # ax1.fill_between(df.index, eval_config["initial_balance"], default_investment,
        #         facecolor='yellow', alpha=0.25)
                # ax1.tick_params(axis='x', labelbottom=False)
        # ax1.grid(color = 'olive', linewidth = 0.5, alpha = 0.5)
        # ax2.plot(df.index, df["Close"])
        # colors = mpf.make_marketcolors(up='forestgreen', down='indianred', edge='inherit', wick='inherit')
        # buy = mpf.make_addplot(df[df["buy"] == 1]["Close"], type = "scatter", marker = "^")
        # Create a style based on the custom colors
        # style = mpf.make_mpf_style(marketcolors=colors)
        # apds = [mpf.make_addplot(signals['signal'].map({'buy': df['Close'], 'sell': df['Close']}), scatter=True, markersize=100)]
                # ax2.tick_params(axis='x', labelbottom=False)
        # for signal in buy_signals:
        #     ax2.arrow(
        #         df.index[signal], df["Close"][signal]*0.9, 0, df["Close"][signal]*0.025,
        #     width=4, color='green', alpha = 0.25, linewidth=0)

        # for signal in sell_signals:
        #     ax2.arrow(
        #         df.index[signal], df["Close"][signal]*1.1, 0, -df["Close"][signal]*0.025,
        #     width=4, color='red', alpha = 0.25, linewidth=0)
        # ax2.grid(color = 'gray', linewidth = 0.5, alpha = 0.5)
        # plt.savefig(f"plots/portfolio_value_{iteration}.png", dpi=300)
        
        
        
        
        
        