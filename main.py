from fastapi import FastAPI
from pydantic import BaseModel

from data_fetch import load_symbol
from indicators import add_indicators
from patterns import detect_candle_pattern, detect_chart_pattern
from risk import calculate_risk
from options import get_option_chain
from autotrade import auto_buy, auto_sell

from model import predict_next_price

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = FastAPI(title="AstraTrade AI Backend")

# -----------------------------------------------------------
# MODELS FOR REQUESTS
# -----------------------------------------------------------
class AutoTradeRequest(BaseModel):
    api_key: str
    api_secret: str
    symbol: str
    quantity: float


class RiskRequest(BaseModel):
    entry: float
    stoploss: float
    capital: float


# -----------------------------------------------------------
# PREDICTION ROUTE
# -----------------------------------------------------------
@app.get("/predict")
def predict(symbol: str, market: str):
    df = load_symbol(symbol, market)
    df = add_indicators(df)

    # Normalize data
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Need last 40 rows for model input
    window = df_scaled.tail(40).values

    predicted_price = predict_next_price(window)

    # Candle + chart patterns
    candle = detect_candle_pattern(df)
    chart = detect_chart_pattern(df)

    return {
        "symbol": symbol.upper(),
        "market": market.upper(),
        "prediction": predicted_price,
        "candle_pattern": candle,
        "chart_pattern": chart
    }


# -----------------------------------------------------------
# RISK MANAGEMENT
# -----------------------------------------------------------
@app.post("/risk")
def risk(req: RiskRequest):
    result = calculate_risk(req.entry, req.stoploss, req.capital)
    return result


# -----------------------------------------------------------
# OPTION CHAIN
# -----------------------------------------------------------
@app.get("/option_chain")
def option_chain(symbol: str):
    return get_option_chain(symbol)


# -----------------------------------------------------------
# AUTO BUY (CRYPTO)
# -----------------------------------------------------------
@app.post("/autobuy")
def auto_buy_route(req: AutoTradeRequest):
    return auto_buy(req.api_key, req.api_secret, req.symbol, req.quantity)


# -----------------------------------------------------------
# AUTO SELL (CRYPTO)
# -----------------------------------------------------------
@app.post("/autosell")
def auto_sell_route(req: AutoTradeRequest):
    return auto_sell(req.api_key, req.api_secret, req.symbol, req.quantity)

@app.get("/debug_model")
def debug_model():
    import torch
    state = torch.load("model.pth", map_location="cpu")
    return {k: list(v.shape) for k,v in state.items()}


# -----------------------------------------------------------
# HOME
# -----------------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "message": "AstraTrade AI Backend Active"
    }

