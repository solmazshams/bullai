# import ml
import math
import numpy as np
import ta
# import kernels

class Settings:
    def __init__(self, source, neighborsCount, maxBarsBack, featureCount, colorCompression, showExit, useDynamicExits):
        self.source: float = 0.0
        self.neighborsCount: int = 0
        self.maxBarsBack: int = 0
        self.featureCount: int = 0
        self.colorCompression: int = 0
        self.showExits: bool = False
        self.useDynamicExits: bool = False

class Label:
    def __init__(self, long, short, neutral):
        self.long: int = 0
        self.short:int = 0
        self.neutral: int = 0

class FeatureArrays:
    def __init__(self, f1, f2, f3, f4, f5):
        self.f1: list[float] = []
        self.f2: list[float] = []
        self.f3: list[float] = []
        self.f4: list[float] = []
        self.f5: list[float] = []
        
class FeatureSeries:
    def __init__(self, f1, f2, f3, f4, f5):
        self.f1: float = 0.0 
        self.f2: float = 0.0
        self.f3: float = 0.0 
        self.f4: float = 0.0
        self.f5: float = 0.0

class MLModel:
    def __init__(self, firstBarIndex, trainingLabels, loopSize, lastDistance, distancesArray, predictionsArray, prediction):
        self.firstBarIndex: int = 0
        self.trainingLabels: list[int] = []
        self.loopSize: int = 0
        self.lastDistance: float = 0.0
        self.distancesArray: list[float] = []
        self.predictionsArray: list[int] = []
        self.prediction: int = 0

class FilterSettings:
     def __init__(self, useVolatilityFilter, useRegimeFilter, useAdxFilter, regimeThreshold, adxThreshold):
        self.useVolatilityFilter: bool = False
        self.useRegimeFilter: bool = False
        self.useAdxFilter: bool = False
        self.regimeThreshold: float = 0.0
        self.adxThreshold: int = 0

class Filter:
    def __init__(self, volatility, regime, adx):
        self.volatility: bool = False
        self.regime: bool = False
        self.adx: bool = False 

# ==== Helper Functions ====
def series_from(feature_string, _close, _high, _low, _hlc3, f_paramA, f_paramB):
    if feature_string == "RSI":
        # return ml.n_rsi(_close, f_paramA, f_paramB)
        return np.array(_close)
    elif feature_string == "WT":
        # return ml.n_wt(_hlc3, f_paramA, f_paramB)
        return np.array(_hlc3)  #ta.momentum.RSIIndicator
    elif feature_string == "CCI":
        # return ml.n_cci(_close, f_paramA, f_paramB)
        return np.array(_close)
    elif feature_string == "ADX":
        # return ml.n_adx(_high, _low, _close, f_paramA)
        return np.array(_high)


def get_lorentzian_distance(i, featureCount, featureSeries, featureArrays):
    if featureCount == 5:
        return (
            math.log(1 + abs(featureSeries.f1 - featureArrays.f1[i])) +
            math.log(1 + abs(featureSeries.f2 - featureArrays.f2[i])) +
            math.log(1 + abs(featureSeries.f3 - featureArrays.f3[i])) +
            math.log(1 + abs(featureSeries.f4 - featureArrays.f4[i])) +
            math.log(1 + abs(featureSeries.f5 - featureArrays.f5[i]))
        )
    elif featureCount == 4:
        return (
            math.log(1 + abs(featureSeries.f1 - featureArrays.f1[i])) +
            math.log(1 + abs(featureSeries.f2 - featureArrays.f2[i])) +
            math.log(1 + abs(featureSeries.f3 - featureArrays.f3[i])) +
            math.log(1 + abs(featureSeries.f4 - featureArrays.f4[i]))
        )
    elif featureCount == 3:
        return (
            math.log(1 + abs(featureSeries.f1 - featureArrays.f1[i])) +
            math.log(1 + abs(featureSeries.f2 - featureArrays.f2[i])) +
            math.log(1 + abs(featureSeries.f3 - featureArrays.f3[i]))
        )
    elif featureCount == 2:
        return (
            math.log(1 + abs(featureSeries.f1 - featureArrays.f1[i])) +
            math.log(1 + abs(featureSeries.f2 - featureArrays.f2[i]))
        )

# ==== Inputs ====  
settings = Settings('close', 8, 2000, 5, 1, False, False)
# (source, neighborsCount, maxBarsBack, featureCount, colorCompression, showExit, useDynamicExits)

# Trade Stats Settings
showTradeStats = True 
useWorstCase = False 

# Settings object for user-defined settings
filterSettings = FilterSettings(True, True, False, -0.1, 20)
# (useVolatilityFilter, useRegimeFilter, useAdxFilter, regimeThreshold, adxThreshold)

# Filter object for filtering the ML predictions
# filter = Filter(ml.filter_volatility(1, 10, filterSettings.useVolatilityFilter),
#                 ml.regime_filter(ohlc4, filterSettings.regimeThreshold, filterSettings.useRegimeFilter),
#                 ml.filter_adx(settings.source, 14, filterSettings.adxThreshold, filterSettings.useADXFilter))

filter = Filter(True, True, np.array(settings.source))
# (volatility, regime, adx)

#Feature Variables: User-Defined Inputs for calculating Feature Series. 
f1_string = 'RSI'
f1_paramA = 14
f1_paramB = 1
f2_string = "WT"
f2_paramA = 10
f2_paramB = 11
f3_string = "CCI"
f3_paramA = 20
f3_paramB = 1
f4_string = "ADX"
f4_paramA = 20
f4_paramB = 2
f5_string = "RSI"
f5_paramA = 9
f5_paramB = 1

# FeatureSeries Object: Calculated Feature Series based on Feature Variables
featureSeries = FeatureSeries(series_from(f1_string, 'close', 'high', 'low', 'hlc3', f1_paramA, f1_paramB),  # f1
                             series_from(f2_string, 'close', 'high', 'low', 'hlc3', f2_paramA, f2_paramB),  # f2
                             series_from(f3_string, 'close', 'high', 'low', 'hlc3', f3_paramA, f3_paramB),  # f3
                             series_from(f4_string, 'close', 'high', 'low', 'hlc3', f4_paramA, f4_paramB),  # f4
                             series_from(f5_string, 'close', 'high', 'low', 'hlc3', f5_paramA, f5_paramB))  # f5


# FeatureArrays Variables: Storage of Feature Series as Feature Arrays Optimized for ML
# Note: These arrays cannot be dynamically created within the FeatureArrays Object Initialization and thus must be set-up in advance.

f1Array = []
f2Array = []
f3Array = []
f4Array = []
f5Array = []

f1Array.append(featureSeries.f1)
f2Array.append(featureSeries.f2)
f3Array.append(featureSeries.f3)
f4Array.append(featureSeries.f4)
f5Array.append(featureSeries.f5)


# FeatureArrays Object: Storage of the calculated FeatureArrays into a single object
featureArrays = FeatureArrays(f1Array, f2Array, f3Array, f4Array, f5Array)

# Label Object: Used for classifying historical data as training data for the ML Model
direction = Label(long=1, short=-1, neutral=0)

# Derived from General Settings
last_bar_index = 15  #??
maxBarsBackIndex = last_bar_index - settings.maxBarsBack if last_bar_index >= settings.maxBarsBack else 0

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

src = settings.source
y_train_series = direction.short #if src[4] < src[0] else direction.long if src[4] > src[0] else direction.neutral
y_train_array = []
predictions = []
prediction = 0.0
signal = direction.neutral
distances = []

y_train_array.append(y_train_series)

# ====  Core ML Logic  ====
lastDistance = -1.0
size = min(settings.maxBarsBack-1, len(y_train_array)-1)
sizeLoop = min(settings.maxBarsBack-1, size)

# bar_index = 4
if bar_index >= maxBarsBackIndex:
    for i in range(sizeLoop):
        d = get_lorentzian_distance(i, settings.featureCount, featureSeries, featureArrays)
        if d >= lastDistance and i % 4:
            lastDistance = d
            distances.append(d)
            predictions.append(round(y_train_array[i]))
            if len(predictions) > settings.neighborsCount:
                lastDistance = distances[round(settings.neighborsCount*3/4)]
                distances = distances[1:]
                predictions = predictions[1:]

    prediction = sum(predictions)

# ==== Prediction Filters ====

# User Defined Filters: Used for adjusting the frequency of the ML Model's predictions
filter_all = filter.volatility and filter.regime and filter.adx

# Filtered Signal: The model's prediction of future price movement direction with user-defined filters applied
if prediction > 0 and filter_all:
    signal = direction.long
elif prediction < 0 and filter_all:
    signal = direction.short
else:
    signal = signal#[1] if len(signal) > 0 else direction.neutral
   
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
isBuySignal = signal == direction.long and isEmaUptrend and isSmaUptrend
isSellSignal = signal == direction.short and isEmaDowntrend and isSmaDowntrend
isLastSignalBuy = signal == direction.long and isEmaUptrend and isSmaUptrend
isLastSignalSell = signal == direction.short and isEmaDowntrend and isSmaDowntrend
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
yhat1 = rationalQuadratic(settings.source, h, r, x)
print(yhat1)
def gaussianKernel(x, y, sigma):
    squared_distance = np.sum((x - y) ** 2)
    kernel_value = np.exp(-squared_distance / (2 * sigma ** 2))
    return kernel_value
yhat2 = gaussianKernel(settings.source, h-lag, x)

kernelEstimate = yhat1

# Kernel Rates of Change
wasBearishRate = yhat1#[2] > yhat1[1]
wasBullishRate = yhat1#[2] < yhat1[1]
isBearishRate = yhat1#[1] > yhat1
isBullishRate = yhat1#[1] < yhat1
isBearishChange = isBearishRate and wasBullishRate
isBullishChange = isBullishRate and wasBearishRate

# Kernel Crossovers
isBullishCrossAlert = ta.crossover(yhat2, yhat1)
isBearishCrossAlert = ta.crossunder(yhat2, yhat1)
isBullishSmooth = yhat2 >= yhat1
isBearishSmooth = yhat2 <= yhat1


# Kernel Colors
colorByCross = c_green if isBullishSmooth else c_red
colorByRate = c_green if isBullishRate else c_red
plotColor = colorByCross if showKernelEstimate and useKernelSmoothing else colorByRate if showKernelEstimate else transparent
plot(kernelEstimate, color=plotColor, linewidth=2, title="Kernel Regression Estimate")

# Alert Variables
alertBullish = isBullishCrossAlert if useKernelSmoothing else isBullishChange
alertBearish = isBearishCrossAlert if useKernelSmoothing else isBearishChange

isBullish = isBullishSmooth if useKernelFilter and useKernelSmoothing else isBullishRate if useKernelFilter else True
isBearish = isBearishSmooth if useKernelFilter and useKernelSmoothing else isBearishRate if useKernelFilter else True

# ==== Entries and Exits ====
# Entry Conditions: Booleans for ML Model Position Entries

startLongTrade = isNewBuySignal and isBullish and isEmaUptrend and isSmaUptrend
startShortTrade = isNewSellSignal and isBearish and isEmaDowntrend and isSmaDowntrend


# Dynamic Exit Conditions: Booleans for ML Model Position Exits based on Fractal Filters and Kernel Regression Filters
lastSignalWasBullish = ta.barssince(startLongTrade) < ta.barssince(startShortTrade)
lastSignalWasBearish = ta.barssince(startShortTrade) < ta.barssince(startLongTrade)
barsSinceRedEntry = ta.barssince(startShortTrade)
barsSinceRedExit = ta.barssince(alertBullish)
barsSinceGreenEntry = ta.barssince(startLongTrade)
barsSinceGreenExit = ta.barssince(alertBearish)
isValidShortExit = barsSinceRedExit > barsSinceRedEntry
isValidLongExit = barsSinceGreenExit > barsSinceGreenEntry
endLongTradeDynamic = isBearishChange and isValidLongExit[1]
endShortTradeDynamic = isBullishChange and isValidShortExit[1]


# Fixed Exit Conditions: Booleans for ML Model Position Exits based on a Bar-Count Filters
endLongTradeStrict = ((isHeldFourBars and isLastSignalBuy) or (isHeldLessThanFourBars and isNewSellSignal and isLastSignalBuy)) and startLongTrade[4]
endShortTradeStrict = ((isHeldFourBars and isLastSignalSell) or (isHeldLessThanFourBars and isNewBuySignal and isLastSignalSell)) and startShortTrade[4]
isDynamicExitValid = not useEmaFilter and not useSmaFilter and not useKernelSmoothing
endLongTrade = endLongTradeDynamic if settings.useDynamicExits and isDynamicExitValid else endLongTradeStrict
endShortTrade = endShortTradeDynamic if settings.useDynamicExits and isDynamicExitValid else endShortTradeStrict



# ==== Plotting Labels ====

import matplotlib.pyplot as plt

# Buy signal
if startLongTrade:
    plt.plot(bar_index, low, 'g^', label='Buy', markersize=4)

# Sell signal
if startShortTrade:
    plt.plot(bar_index, high, 'rv', label='Sell', markersize=4)

# Stop Buy signal
if endLongTrade and settings.showExits:
    plt.plot(bar_index, high, 'kx', label='StopBuy', markersize=3)

# Stop Sell signal
if endShortTrade and settings.showExits:
    plt.plot(bar_index, low, 'kx', label='StopSell', markersize=3)

plt.legend()
plt.show()

# ==== Alerts ====

# Separate Alerts for Entries and Exits
# Open Long alert
if startLongTrade:
    print("Open Long ▲ | {ticker}@{close} | ({interval})")

# Close Long alert
if endLongTrade:
    print("Close Long ▲ | {ticker}@{close} | ({interval})")

# Open Short alert
if startShortTrade:
    print("Open Short ▼ | {ticker}@{close} | ({interval})")

# Close Short alert
if endShortTrade:
    print("Close Short ▼ | {ticker}@{close} | ({interval})")

# Combined Open Position alert
if startLongTrade or startShortTrade:
    print("Open Position ▲▼ | {ticker}@{close} | ({interval})")

# Combined Close Position alert
if endLongTrade or endShortTrade:
    print("Close Position ▲▼ | {ticker}@[{close}] | ({interval})")

# Kernel Bullish Color Change alert
if alertBullish:
    print("Kernel Bullish ▲ | {ticker}@{close} | ({interval})")

# Kernel Bearish Color Change alert
if alertBearish:
    print("Kernel Bearish ▼ | {ticker}@{close} | ({interval})")


# ==== Display Signals ==== 
import matplotlib.pyplot as plt
import numpy as np

atrSpaced = ta.atr(1) if useAtrOffset else np.nan
compressionFactor = settings.neighborsCount / settings.colorCompression

c_pred = np.where(prediction > 0, plt.cm.get_cmap('RdYlGn')(prediction / compressionFactor), np.where(prediction <= 0, plt.cm.get_cmap('YlOrRd')(-prediction / compressionFactor), np.nan))

c_label = c_pred if showBarPredictions else np.nan
c_bars = c_pred if showBarColors else np.nan

x_val = np.array(bar_index)
y_val = np.where(useAtrOffset, np.where(prediction > 0, high + atrSpaced, low - atrSpaced), np.where(prediction > 0, high + hl2 * barPredictionsOffset / 20, low - hl2 * barPredictionsOffset / 30))

fig, ax = plt.subplots()
ax.bar(x_val, y_val, color=c_bars, width=1)

for i in range(len(x_val)):
    ax.text(x_val[i], y_val[i], str(prediction[i]), ha='left', va='center', color=c_label[i])

plt.show()


# ==== Backtesting ====

# The following can be used to stream signals to a backtest adapter
import matplotlib.pyplot as plt

backTestStream = np.select([startLongTrade, endLongTrade, startShortTrade, endShortTrade], [1, 2, -1, -2], 0)

plt.plot(bar_index, backTestStream, label='Backtest Stream')
plt.legend()
plt.show()


# The following can be used to display real-time trade stats. This can be a useful mechanism for obtaining real-time feedback during Feature Engineering. This does NOT replace the need to properly backtest.
# Note: In this context, a "Stop-Loss" is defined instances where the ML Signal prematurely flips directions before an exit signal can be generated.
import numpy as np
import pandas as pd

def init_table():
    c_transparent = "black"
    table_data = {
        "Trade Stats": ["Winrate", "Trades", "WL Ratio", "Early Signal Flips"],
        "Value": ["", "", "", ""],
    }
    table = pd.DataFrame(table_data)
    return table

def update_table(table, tradeStatsHeader, totalTrades, totalWins, totalLosses, winLossRatio, winRate, stopLosses):
    table.loc[table["Trade Stats"] == "Winrate", "Value"] = f"{totalWins / totalTrades:.1%}"
    table.loc[table["Trade Stats"] == "Trades", "Value"] = f"{totalTrades} ({totalWins}|{totalLosses})"
    table.loc[table["Trade Stats"] == "WL Ratio", "Value"] = f"{totalWins / totalLosses:.2f}"
    table.loc[table["Trade Stats"] == "Early Signal Flips", "Value"] = f"{totalEarlySignalFlips}"
    return table

if showTradeStats:
    tbl = init_table()
    if barstate.islast:
        tbl = update_table(tbl, tradeStatsHeader, totalTrades, totalWins, totalLosses, winLossRatio, winRate, totalEarlySignalFlips)
