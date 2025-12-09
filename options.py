import requests
import numpy as np

# ---------------------------------------------------
# NSE OPTION CHAIN FETCHER (NO LOGIN, PUBLIC API)
# ---------------------------------------------------
def get_option_chain(symbol):
    try:
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol.upper()}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }

        session = requests.Session()
        data = session.get(url, headers=headers).json()

        records = data["records"]["data"]
        calls = []
        puts = []

        for row in records:
            if "CE" in row:
                ce = row["CE"]
                calls.append({
                    "strike": ce.get("strikePrice"),
                    "oi": ce.get("openInterest"),
                    "change_oi": ce.get("changeinOpenInterest"),
                    "iv": ce.get("impliedVolatility")
                })

            if "PE" in row:
                pe = row["PE"]
                puts.append({
                    "strike": pe.get("strikePrice"),
                    "oi": pe.get("openInterest"),
                    "change_oi": pe.get("changeinOpenInterest"),
                    "iv": pe.get("impliedVolatility")
                })

        return {
            "calls": calls,
            "puts": puts,
            "status": "success"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
