import pandas as pd
import numpy as np

# ------------------------------------------------
# BASIC CANDLE PATTERN DETECTION
# ------------------------------------------------
def detect_candle_pattern(df):
    """
    Returns one candle pattern name or 'None'
    """

    last = df.iloc[-1]
    prev = df.iloc[-2]

    open_ = last["Open"]
    close = last["Close"]
    high = last["High"]
    low = last["Low"]

    body = abs(close - open_)
    range_ = high - low

    # Avoid division by zero
    if range_ == 0:
        return "None"

    # -------- Doji --------
    if body <= (0.1 * range_):
        return "Doji"

    # -------- Bullish Engulfing --------
    if close > open_ and prev["Close"] < prev["Open"] and \
       close > prev["Open"] and open_ < prev["Close"]:
        return "Bullish Engulfing"

    # -------- Bearish Engulfing --------
    if close < open_ and prev["Close"] > prev["Open"] and \
       close < prev["Open"] and open_ > prev["Close"]:
        return "Bearish Engulfing"

    # -------- Hammer --------
    if body <= (0.3 * range_) and (open_ - low > 2 * body):
        return "Hammer"

    # -------- Shooting Star --------
    if body <= (0.3 * range_) and (high - close > 2 * body):
        return "Shooting Star"

    return "None"


# ------------------------------------------------
# SIMPLE CHART PATTERN: DOUBLE TOP/BOTTOM
# ------------------------------------------------
def detect_chart_pattern(df, window=20):
    """
    Detects simple double top / double bottom patterns.
    """
    data = df.tail(window)["Close"].values

    max1 = np.max(data[:window//2])
    max2 = np.max(data[window//2:])
    min1 = np.min(data[:window//2])
    min2 = np.min(data[window//2:])

    # Double Top
    if abs(max1 - max2) <= (0.02 * max1):
        return "Double Top"

    # Double Bottom
    if abs(min1 - min2) <= (0.02 * min1):
        return "Double Bottom"

    return "None"
