import yfinance as yf
import pandas as pd
import numpy as np
import re

# --------------------------------------------
# CLEAN YAHOO FINANCE DATA (handles multi-index)
# --------------------------------------------
def fix_yahoo_data(raw):
    # Flatten multi-index columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    clean = pd.DataFrame()

    for col in df.columns:
        s = df[col].astype(str)

        # Extract first number from any messy string
        s = s.apply(
            lambda x: float(re.findall(r"[-+]?\d*\.\d+|\d+", x)[0])
            if re.findall(r"[-+]?\d*\.\d+|\d+", x)
            else np.nan
        )
        clean[col] = s

    clean.dropna(inplace=True)
    clean = clean.astype(float)
    return clean


# --------------------------------------------
# DOWNLOAD DATA FOR SYMBOL
# --------------------------------------------
def load_symbol(symbol, market):
    if market.upper() == "NSE":
        raw = yf.download(symbol + ".NS", period="2y", interval="1d", auto_adjust=False)
    elif market.upper() == "US":
        raw = yf.download(symbol, period="2y", interval="1d", auto_adjust=False)
    elif market.upper() == "CRYPTO":
        raw = yf.download(symbol + "-USD", period="2y", interval="1d", auto_adjust=False)
    else:
        raise ValueError("Invalid market")

    return fix_yahoo_data(raw)
