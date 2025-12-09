# ---------------------------------------
# RISK MANAGEMENT CALCULATOR
# ---------------------------------------

def calculate_risk(entry, stoploss, capital):
    """
    entry: entry price
    stoploss: stoploss price
    capital: user total capital
    
    Returns:
        risk_amount
        position_size
        risk_reward_if_target_x2
    """

    risk_per_share = abs(entry - stoploss)
    if risk_per_share == 0:
        risk_per_share = 0.0001  # avoid divide by zero

    # Risk 1% of capital
    max_risk_amount = capital * 0.01

    # Position size = (max amount willing to lose) / (loss per share)
    qty = max_risk_amount / risk_per_share
    qty = int(qty)

    # Target is 2x reward
    target = entry + (risk_per_share * 2)

    return {
        "risk_amount": round(max_risk_amount, 2),
        "position_size": qty,
        "stoploss": stoploss,
        "target": round(target, 2),
        "risk_reward": "1:2"
    }
