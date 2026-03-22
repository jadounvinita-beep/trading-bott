"""
============================================================
  AI Trading Bot v3 - ULTRA SMART EDITION
  Configuration File
============================================================
"""

CONFIG = {
    # ── API Keys ──────────────────────────────────────────
    "api_key":    "7ioIS33xEr3yGpAOQ2jW4KnMcQtPfKB7kjKMAtUpGAAYKeIFcUSgy5TQhXrE0wfF",   # Leave empty - use environment variables
    "api_secret": "SrXedQwF0ZwUUIi2FcdvqJlIQqi8vvF3e1ytTWyjKZ3kE9TN0T1dIrPDJlolTlA4",   # Leave empty - use environment variables

    # ── Trading ───────────────────────────────────────────
    "symbol":   "SOLUSDT",   # Coin to trade
    "interval": "1m",        # 1 minute candles for scalping

    # ── Safety ────────────────────────────────────────────
    "testnet": True,          # True = demo, False = real money

    # ── Risk Management ───────────────────────────────────
    "risk_pct":           0.90,   # 90% of balance per trade
    "stop_loss_pct":      0.3,    # Stop loss 0.3%
    "take_profit_pct":    0.5,    # Take profit 0.5%
    "max_daily_loss_pct": 3.0,    # Stop if down 3% today
    "max_trades_per_day": 50,     # Max 50 trades per day
    "min_usdt_trade":     5.0,    # Minimum $5 per trade

    # ── AI Settings ───────────────────────────────────────
    "confidence_threshold":   0.25,  # Combined signal threshold
    "retrain_every_n_cycles": 30,    # Retrain every 30 cycles

    # ── Timing ────────────────────────────────────────────
    "sleep_seconds": 30,             # Check every 30 seconds
}

"""
============================================================
  HOW THIS BOT IS SMARTER:

  1. GRADIENT BOOSTING MODEL
     - More accurate than Random Forest
     - Learns complex market patterns
     - 40+ technical indicators

  2. MARKET REGIME DETECTION
     - Trending Up  → aggressive buying
     - Trending Down → stay out
     - Sideways     → mean reversion

  3. DYNAMIC POSITION SIZING
     - Strong signal = bigger trade
     - High accuracy = bigger trade
     - Extreme fear  = bigger trade

  4. TRAILING STOP LOSS
     - Moves stop to breakeven after 70% of TP
     - Locks in profit automatically!

  5. SMART SIGNAL WEIGHTS
     - Adjusts weights based on regime
     - Trusts technical more if accurate
     - Considers all 3 signals together
============================================================
"""
