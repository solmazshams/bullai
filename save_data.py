"""_This script download the data for the training_
"""
import yfinance as yf

symbol = 'QQQ'
start_date = '1999-01-01'
data = yf.download(
    symbol, start=start_date)  # load data of symbol
data.to_csv(f'./data/{symbol}.csv', index = True)

