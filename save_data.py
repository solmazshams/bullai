import yfinance as yf
import pandas as pd

symbol = 'QQQ'
start_date = '2000-01-01'
data = yf.download(
    symbol, start=start_date)  # load data of symbol
data.to_csv(f'./data/{symbol}.csv', index = True)

