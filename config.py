CONFIG = {
    # API Keys - leave empty, Railway uses environment variables
    "api_key":    "B9kW2hgYLUfB3zjJePo8w0pzpIAxLWeucz36IaiglAwE5OizCeNeTGfuYForiILK",
    "api_secret": "Cvz8svfgCNfxR9f8ukUuLJIoxpi2I9fYsePiDvMv5MaaQdRXMnJIDS7wdqjnYqut",
    "testnet": True,

    # Trading
    "symbols":  ["SOLUSDT"],
    "interval": "15m",

    # Risk
    "risk_pct":           0.02,
    "stop_loss_pct":      0.015,
    "take_profit_pct":    0.03,
    "min_usdt_trade":     10.0,
    "max_daily_loss_pct": 3.0,
    "max_trades_per_day": 50,

    # AI
    "confidence_threshold":   0.52,   # Lower = more trades
    "retrain_every_n_cycles": 8,     # Retrain every 30 cycles

    # Timing
    "sleep_seconds": 60,
}
