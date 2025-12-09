from binance.client import Client

# -----------------------------------------------------
# AUTO TRADING (CRYPTO ONLY) — Binance API
# -----------------------------------------------------
# NOTE:
# User must provide their own API key + secret key
# We do NOT store them in backend. They are sent from app.


def create_client(api_key, api_secret):
    try:
        client = Client(api_key, api_secret)
        return client
    except Exception as e:
        return None


# -----------------------------------------------------
# MARKET BUY
# -----------------------------------------------------
def auto_buy(api_key, api_secret, symbol, quantity):
    try:
        client = create_client(api_key, api_secret)
        if client is None:
            return {"status": "error", "message": "Invalid API keys"}

        order = client.order_market_buy(
            symbol=symbol.upper(),
            quantity=quantity
        )
        return {"status": "success", "order": order}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# -----------------------------------------------------
# MARKET SELL
# -----------------------------------------------------
def auto_sell(api_key, api_secret, symbol, quantity):
    try:
        client = create_client(api_key, api_secret)
        if client is None:
            return {"status": "error", "message": "Invalid API keys"}

        order = client.order_market_sell(
            symbol=symbol.upper(),
            quantity=quantity
        )
        return {"status": "success", "order": order}

    except Exception as e:
        return {"status": "error", "message": str(e)}
