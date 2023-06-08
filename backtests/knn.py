""" This is module for the knn pine code from tradingview into python """

import numpy as np
import ta
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import CCIIndicator

import mplfinance as mpf
import yfinance as yf
# Preset Dates
start_date = np.datetime64('2000-01-01')
stop_date = np.datetime64('2025-12-31')

# Inputs
Indicator = 'All'  # Specify the desired indicator
short_window = 14
long_window = 28

base_k = 252
filter = False
bars = 300

# Constants
BUY = 1
SELL = -1
CLEAR = 0
k = int(np.floor(np.sqrt(base_k)))

symbol = "AAPL"

df = yf.download(symbol, period='100d')



rsi_long = RSIIndicator(df['Close'], window=long_window)
rsi_short = RSIIndicator(df['Close'], window=short_window)
cci_long = CCIIndicator(df['high'], df['low'], df['close'], window = long_window)
cci_short = CCIIndicator(df['high'], df['low'], df['close'], window = short_window)
roc_long = ROCIndicator(df['close'], window=long_window)
roc_short = ROCIndicator(df['close'], window=short_window)
vol_short = ta.highest(df['volume'], window=short_window)
vol_long = ta.highest()
# Logic
window = (StartDate <= time) & (time <= StopDate)

# Calculate indicator values
rs = ta.rsi(close, LongWindow)
rf = ta.rsi(close, ShortWindow)
cs = ta.cci(close, LongWindow)
cf = ta.cci(close, ShortWindow)
os = ta.roc(close, LongWindow)
of = ta.roc(close, ShortWindow)
vs = ta.highest(volume, LongWindow) - ta.lowest(volume, LongWindow)
vf = ta.highest(volume, ShortWindow) - ta.lowest(volume, ShortWindow)

# Calculate feature values
f1 = np.mean([rs, cs, os, vs])
f2 = np.mean([rf, cf, of, vf])

# Classification data
class_label = np.sign(close[-1] - close[-2])

# Store training data if within the window
if window:
    feature1.append(f1)
    feature2.append(f2)
    directions.append(class_label)

# Core logic of the algorithm
size = len(directions)
maxdist = -999.0
for i in range(size):
    d = np.sqrt((f1 - feature1[i]) ** 2 + (f2 - feature2[i]) ** 2)
    if d > maxdist:
        maxdist = d
        if len(predictions) >= k:
            predictions = predictions[1:]
        predictions.append(directions[i])

# Get the overall prediction of k nearest neighbors
prediction = sum(predictions)

# Apply filter
filter = ta.atr(10) > ta.atr(40) if Filter else True

# Determine trading signals
long = prediction > 0 and filter
short = prediction < 0 and filter
clear = not (long or short)

if bars[0] == Bars:
    signal = CLEAR
    bars[0] = 0
else:
    bars[0] += 1

signal = BUY if long else SELL if short else CLEAR if clear else signal[1]

changed = ta.change(signal)
startLongTrade = changed and signal == BUY
startShortTrade = changed and signal == SELL
clear_condition = changed and signal == CLEAR

maxpos = ta.highest(high, 10)
minpos = ta.lowest(low, 10)

# Visuals (plotting omitted)

# Notification (alertcondition omitted)

# Backtesting (commented out)

# Cumulative return calculation (commented out)
