CONFIG = {
    # API Keys - leave empty, Railway uses environment variables
    "api_key":    "DoTY2EYdHLuYr0dCvwJO4NPFsu0vICwnKxCF4kgWyINmyWmHzdAovSVulA3xpLq5",
    "api_secret": "WYWyG7KpFHvC0HbDytyq7UooqvInvbQoZQcvZKJ3G46xyAkdLypbVURizUf1Xo4C",
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
