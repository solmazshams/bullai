import ta
import talib
import math
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
# import mplfinance as mpf
from ta.momentum import RSIIndicator

# Download ticker data
ticker: str  = "TQQQ"  # Ticker symbol (e.g., "AAPL" for Apple Inc.)
data: pd.DataFrame = yf.download(ticker, start="2020-01-01", end="2021-12-31")
data['HLC'] = (data['High'] + data['Low'] + data['Close']) /3

# Calculate the 14-day Relative Strength Index (RSI)
delta: pd.Series = data['Close'].diff()
gain: pd.Series = delta.copy()
loss: pd.Series = delta.copy()
gain[gain < 0] = 0
loss[loss > 0] = 0
avg_gain: pd.Series = gain.rolling(window=14).mean()
avg_loss: pd.Series = abs(loss.rolling(window=14).mean())
rs: pd.Series = avg_gain / avg_loss
rsi: pd.Series = 100 - (100 / (1 + rs))

delta_hlc: pd.Series = data['HLC'].diff()
gain_hlc: pd.Series = delta_hlc.copy()
loss_hlc: pd.Series = delta_hlc.copy()
gain_hlc[gain_hlc < 0] = 0
loss_hlc[loss_hlc > 0] = 0
avg_gain_hlc: pd.Series = gain_hlc.rolling(window=14).mean()
avg_loss_hlc: pd.Series = abs(loss_hlc.rolling(window=14).mean())
ad: pd.Series = avg_gain_hlc / avg_loss_hlc
adx: pd.Series = 100 - (100 / (1 + ad))
data['ADX']: pd.Series = adx

rsi_n = 2
adx_n = 1
def get_lorentzian_distance(i):
    rsi_n = np.m 
    return (math.log(1 + abs(rsi_n - rsi[i])) + math.log(1 + abs(adx_n - adx[i])))

# ==== Inputs ====  
neigh_count = 8
maxbarback = 2000
feture_count = 5
colorcomp = 1
showexit = False
usedynamicexit = False

showTradeStats = True 
useWorstCase = False 

# Settings object for user-defined settings
useVolatilityFilter = True
useRegimeFilter = True
useAdxFilter = False
regimeThreshold = -0.1
adxThreshold = 20

# Filter object for filtering the ML predictions
volatility = True
regime = True
adx = False

#Feature Variables: User-Defined Inputs for calculating Feature Series. 
f1_str, f1_A, f1_B = 'RSI', 14, 1
f2_str, f2_A, f2_B = "ADX", 20, 2

# Label Object: Used for classifying historical data as training data for the ML Model
long=1
short=-1
neutral=0

# Derived from General Settings
last_bar_index = 15  #??
maxBarsBackIndex = last_bar_index - maxbarback if last_bar_index >= maxbarback else 0

# EMA Settings 
useEmaFilter = False
emaPeriod = 200
isEmaUptrend = 'close' > ta.trend.ema_indicator('close', emaPeriod) if useEmaFilter else True
isEmaDowntrend = 'close' < ta.trend.ema_indicator('close', emaPeriod) if useEmaFilter else True
useSmaFilter = False
smaPeriod = 200
isSmaUptrend = 'close' > ta.trend.sma_indicator('close', smaPeriod) if useSmaFilter else True
isSmaDowntrend = 'close' < ta.trend.sma_indicator('close', smaPeriod) if useSmaFilter else True

# Nadaraya-Watson Kernel Regression Settings
useKernelFilter = True
showKernelEstimate = True
useKernelSmoothing = False
h = 8
r = 8
x = 25
lag = 2

# Display Settings
showBarColors = True
showBarPredictions = True
useAtrOffset = False
barPredictionsOffset = 0

# ==== Next Bar Classification ====

src = data.columns
y_train_series = short if src[4] < src[0] else long if src[4] > src[0] else neutral
y_train_array = []
predictions = []
prediction = 0.0
signal = neutral
distances = []

y_train_array.append(y_train_series)

# ====  Core ML Logic  ====
lastDistance = -1.0
size = min(maxbarback-1, len(y_train_array)-1)
sizeLoop = min(maxbarback-1, size)


bar_index = 4
if bar_index >= maxBarsBackIndex:
    for i in range(sizeLoop):
        d = get_lorentzian_distance(i, 5)
        if d >= lastDistance and i % 4:
            lastDistance = d
            distances.append(d)
            predictions.append(round(y_train_array[i]))
            if len(predictions) > neigh_count:
                lastDistance = distances[round(neigh_count*3/4)]
                distances = distances[1:]
                predictions = predictions[1:]

    prediction = sum(predictions)

# ==== Prediction Filters ====

if prediction > 0:
    signal = long
elif prediction < 0:
    signal = short
else:
    signal = neutral #[1] if len(signal) > 0 else direction.neutral
   
# Bar-Count Filters: Represents strict filters based on a pre-defined holding period of 4 bars
barsHeld = 0
if np.abs(signal):
    barsHeld = 0
else:
    barsHeld += 1

isHeldFourBars = barsHeld == 4
isHeldLessThanFourBars = 0 < barsHeld < 4


# Fractal Filters: Derived from relative appearances of signals in a given time series fractal/segment with a default length of 4 bars
isDifferentSignalType = np.abs(signal)
isEarlySignalFlip = np.abs(signal) and (np.abs(signal)) # or np.diff(signal[2]) or np.diff(signal[3]))
isBuySignal = signal == long and isEmaUptrend and isSmaUptrend
isSellSignal = signal == short and isEmaDowntrend and isSmaDowntrend
isLastSignalBuy = signal == long and isEmaUptrend and isSmaUptrend
isLastSignalSell = signal == short and isEmaDowntrend and isSmaDowntrend
isNewBuySignal = isBuySignal and isDifferentSignalType
isNewSellSignal = isSellSignal and isDifferentSignalType

# Kernel Regression Filters: Filters based on Nadaraya-Watson Kernel Regression using the Rational Quadratic Kernel
c_green = (0, 153, 136, 20)
c_red = (204, 51, 17, 20)
transparent = (0, 0, 0, 100)

def rationalQuadratic(x, y, alpha, beta):
    squared_distance = np.sum((x - y) ** 2)
    kernel_value = (1 + squared_distance / (2 * alpha * beta)) ** (-alpha)
    return kernel_value

yhat1 = []
for i in range(len(data.columns)):
    yh = rationalQuadratic(i, h, r, x)
    yhat1.append(yh)

def gaussianKernel(x, y, sigma):
    squared_distance = np.sum((x - y) ** 2)
    kernel_value = np.exp(-squared_distance / (2 * sigma ** 2))
    return kernel_value

yhat2 = []
for i in range(len(data.columns)):
    yh = gaussianKernel(0, h-lag, x)
    yhat2.append(yh)



kernelEstimate = yhat1

# Kernel Rates of Change
wasBearishRate = yhat1[2] > yhat1[1]
wasBullishRate = yhat1[2] < yhat1[1]
isBearishRate = yhat1[1] > yhat1[2]
isBullishRate = yhat1[1] < yhat1[2]
isBearishChange = isBearishRate and wasBullishRate
isBullishChange = isBullishRate and wasBearishRate

def crossover(arr1, arr2):
    crossover_points = np.logical_and(arr1[:-1] < arr2[:-1], arr1[1:] >= arr2[1:])
    return crossover_points

def crossunder(arr1, arr2):
    crossunder_points = np.logical_and(arr1[:-1] > arr2[:-1], arr1[1:] <= arr2[1:])
    return crossunder_points


# Kernel Crossovers
isBullishCrossAlert = crossover(yhat2, yhat1)
isBearishCrossAlert = crossunder(yhat2, yhat1)
isBullishSmooth = yhat2[0] >= yhat1[0]
isBearishSmooth = yhat2[0] <= yhat1[0]


# Kernel Colors
# colorByCross = c_green if isBullishSmooth else c_red
# colorByRate = c_green if isBullishRate else c_red
# plotColor = colorByCross if showKernelEstimate and useKernelSmoothing else colorByRate if showKernelEstimate else transparent
# plt.plot(kernelEstimate, color=plotColor, linewidth=2, title="Kernel Regression Estimate")

# Alert Variables
alertBullish = isBullishCrossAlert if useKernelSmoothing else isBullishChange
alertBearish = isBearishCrossAlert if useKernelSmoothing else isBearishChange

isBullish = isBullishSmooth if useKernelFilter and useKernelSmoothing else isBullishRate if useKernelFilter else True
isBearish = isBearishSmooth if useKernelFilter and useKernelSmoothing else isBearishRate if useKernelFilter else True

# ==== Entries and Exits ====

# I added:
use_dynamic_exits = False

# Entry Conditions: Booleans for ML Model Position Entries
startLongTrade = isNewBuySignal and isBullish and isEmaUptrend and isSmaUptrend
startShortTrade = isNewSellSignal and isBearish and isEmaDowntrend and isSmaDowntrend


# Dynamic Exit Conditions: Booleans for ML Model Position Exits based on Fractal Filters and Kernel Regression Filters
def barssince(condition):
    indices = np.where(condition)[0]
    return np.arange(len(condition)) - indices[np.searchsorted(indices, np.arange(len(condition)))]


lastSignalWasBullish = barssince(startLongTrade) < barssince(startShortTrade)
lastSignalWasBearish = barssince(startShortTrade) < barssince(startLongTrade)
barsSinceRedEntry = barssince(startShortTrade)
barsSinceRedExit = barssince(alertBullish)
barsSinceGreenEntry = barssince(startLongTrade)
barsSinceGreenExit = barssince(alertBearish)
isValidShortExit = barsSinceRedExit > barsSinceRedEntry
isValidLongExit = barsSinceGreenExit > barsSinceGreenEntry
endLongTradeDynamic = isBearishChange and isValidLongExit[1]
endShortTradeDynamic = isBullishChange and isValidShortExit[1]


# Fixed Exit Conditions: Booleans for ML Model Position Exits based on a Bar-Count Filters
endLongTradeStrict = ((isHeldFourBars and isLastSignalBuy) or (isHeldLessThanFourBars and isNewSellSignal and isLastSignalBuy)) and startLongTrade[4]
endShortTradeStrict = ((isHeldFourBars and isLastSignalSell) or (isHeldLessThanFourBars and isNewBuySignal and isLastSignalSell)) and startShortTrade[4]
isDynamicExitValid = not useEmaFilter and not useSmaFilter and not useKernelSmoothing
endLongTrade = endLongTradeDynamic if use_dynamic_exits and isDynamicExitValid else endLongTradeStrict
endShortTrade = endShortTradeDynamic if use_dynamic_exits and isDynamicExitValid else endShortTradeStrict
