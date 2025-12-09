import pandas as pd
import ta

# ------------------------------------
# ADD TECHNICAL INDICATORS
# ------------------------------------
def add_indicators(df):
    df = df.copy()

    df["rsi"] = ta.momentum.rsi(df["Close"])
    df["macd"] = ta.trend.macd(df["Close"])
    df["ema20"] = ta.trend.ema_indicator(df["Close"], 20)
    df["ema50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["bb_high"] = ta.volatility.bollinger_hband(df["Close"])
    df["bb_low"] = ta.volatility.bollinger_lband(df["Close"])

    df = df.fillna(0)
    return df
