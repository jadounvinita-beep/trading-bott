"""
============================================================
  AI Trading Bot v3 - Auto Retraining Fix
  Always trains on CURRENT market data on startup
============================================================
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException
import ta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

from config import CONFIG

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Binance Client ────────────────────────────────────────
def get_client():
    api_key    = os.getenv("BINANCE_API_KEY",    CONFIG["api_key"])
    api_secret = os.getenv("BINANCE_API_SECRET", CONFIG["api_secret"])
    if CONFIG["testnet"]:
        client = Client(api_key, api_secret, testnet=True)
        client.API_URL = "https://testnet.binance.vision/api"
        log.info("[TESTNET] Connected to Binance Testnet")
    else:
        client = Client(api_key, api_secret)
        log.info("[LIVE] Connected to Binance LIVE")
    return client


# ── Fetch Data ────────────────────────────────────────────
def fetch_ohlcv(client, symbol, interval="1m", limit=1000):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","qav","trades","tbbav","tbqav","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df[["open","high","low","close","volume"]]


# ── Features ──────────────────────────────────────────────
def add_features(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["ema_9"]       = ta.trend.EMAIndicator(c, 9).ema_indicator()
    df["ema_21"]      = ta.trend.EMAIndicator(c, 21).ema_indicator()
    df["ema_50"]      = ta.trend.EMAIndicator(c, 50).ema_indicator()
    macd              = ta.trend.MACD(c)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()
    df["adx"]         = ta.trend.ADXIndicator(h, l, c).adx()
    df["rsi"]         = ta.momentum.RSIIndicator(c, 14).rsi()
    df["stoch_k"]     = ta.momentum.StochasticOscillator(h, l, c).stoch()
    df["cci"]         = ta.trend.CCIIndicator(h, l, c).cci()
    df["roc"]         = ta.momentum.ROCIndicator(c, 10).roc()
    bb                = ta.volatility.BollingerBands(c)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_width"]    = bb.bollinger_wband()
    df["bb_pct"]      = bb.bollinger_pband()
    df["atr"]         = ta.volatility.AverageTrueRange(h, l, c).average_true_range()
    df["obv"]         = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["cmf"]         = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v).chaikin_money_flow()
    df["returns_1"]   = c.pct_change(1)
    df["returns_3"]   = c.pct_change(3)
    df["returns_5"]   = c.pct_change(5)
    df["hl_ratio"]    = (h - l) / c
    df["oc_ratio"]    = (c - df["open"]) / df["open"]
    df.dropna(inplace=True)
    return df


FEATURES = [
    "ema_9","ema_21","ema_50","macd","macd_signal","macd_diff","adx",
    "rsi","stoch_k","cci","roc","bb_upper","bb_lower","bb_width",
    "bb_pct","atr","obv","cmf","returns_1","returns_3","returns_5",
    "hl_ratio","oc_ratio"
]


# ── AI Model ──────────────────────────────────────────────
def train_model(df, symbol, path="model.pkl"):
    """Always trains on CURRENT data — fixes stale model problem"""
    log.info(f"[AI] Training on CURRENT {symbol} data...")
    log.info(f"[AI] Price range in training: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Label: price goes up 0.3% in next 3 candles
    df = df.copy()
    df["target"] = (df["close"].shift(-3) / df["close"] - 1 > 0.003).astype(int)
    df = df.dropna()

    buy_count  = df["target"].sum()
    sell_count = len(df) - buy_count
    log.info(f"[AI] Training samples - BUY: {buy_count} | SELL: {sell_count}")

    X  = df[FEATURES]
    y  = df["target"]
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    Xt, Xv, yt, yv = train_test_split(Xs, y, test_size=0.2, shuffle=False)

    # GradientBoosting is better than RandomForest for scalping
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(Xt, yt)
    acc = model.score(Xv, yv)
    log.info(f"[AI] Accuracy: {acc:.2%}")

    joblib.dump({"model": model, "scaler": sc, "symbol": symbol, "trained_at": datetime.now().isoformat()}, path)
    log.info(f"[AI] Model saved! Trained on current {symbol} prices")
    return model, sc


def get_tech_signal(df, model, scaler, threshold=0.55):
    latest = df[FEATURES].iloc[-1:]
    scaled = scaler.transform(latest)
    prob   = model.predict_proba(scaled)[0]
    buy_p, sell_p = prob[1], prob[0]
    log.info(f"[TECH] BUY: {buy_p:.2%} | SELL: {sell_p:.2%}")
    if buy_p  >= threshold: return "BUY",  buy_p
    if sell_p >= threshold: return "SELL", sell_p
    return "HOLD", max(buy_p, sell_p)


# ── Fear & Greed ──────────────────────────────────────────
def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        d = r.json()["data"][0]
        score = int(d["value"])
        label = d["value_classification"]
        log.info(f"[F&G] Score: {score} ({label})")
        if score <= 25:   return "BUY",  score
        elif score <= 45: return "BUY",  score
        elif score <= 55: return "HOLD", score
        elif score <= 75: return "SELL", score
        else:             return "SELL", score
    except:
        return "HOLD", 50


# ── News Sentiment ────────────────────────────────────────
def get_news(symbol):
    try:
        coin = symbol.replace("USDT","")
        url  = f"https://min-api.cryptocompare.com/data/v2/news/?categories={coin}&limit=10"
        r    = requests.get(url, timeout=10)
        arts = r.json().get("Data", [])

        pos = ["surge","rally","bull","gain","rise","high","record",
               "growth","buy","upgrade","positive","boost","strong"]
        neg = ["crash","drop","bear","loss","fall","low","ban","sell",
               "fear","decline","negative","warning","risk","weak"]

        p = n = 0
        for a in arts[:10]:
            text = (a.get("title","") + " " + a.get("body","")[:200]).lower()
            p += sum(1 for w in pos if w in text)
            n += sum(1 for w in neg if w in text)

        total = p + n
        score = (p - n) / total * 100 if total > 0 else 0
        sent  = "POSITIVE" if score > 20 else "NEGATIVE" if score < -20 else "NEUTRAL"
        log.info(f"[NEWS] {coin}: {sent} ({score:.1f})")

        if sent == "POSITIVE": return "BUY",  score
        if sent == "NEGATIVE": return "SELL", score
        return "HOLD", score
    except:
        return "HOLD", 0


# ── Combined Signal ───────────────────────────────────────
def combined_signal(tech_sig, tech_conf, fg_sig, news_sig):
    def s(x): return 1 if x=="BUY" else -1 if x=="SELL" else 0
    score = s(tech_sig)*tech_conf*0.60 + s(fg_sig)*0.25 + s(news_sig)*0.15
    log.info(f"[SIGNAL] Tech:{tech_sig}({tech_conf:.0%}) F&G:{fg_sig} News:{news_sig} Score:{score:.3f}")
    if score >=  0.20: return "BUY",  score
    if score <= -0.20: return "SELL", abs(score)
    return "HOLD", abs(score)


# ── Balance & Orders ──────────────────────────────────────
def get_usdt_balance(client):
    return float(client.get_asset_balance(asset="USDT")["free"])


def get_asset_balance(client, asset):
    return float(client.get_asset_balance(asset=asset)["free"])


def calculate_qty(client, symbol, usdt):
    info = client.get_symbol_info(symbol)
    step = None
    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            step = float(f["stepSize"])
            break
    price = float(client.get_symbol_ticker(symbol=symbol)["price"])
    qty   = usdt / price
    if step:
        prec = len(str(step).rstrip("0").split(".")[-1])
        qty  = round(qty - (qty % step), prec)
    return qty


class TradeManager:
    def __init__(self, client):
        self.client       = client
        self.positions    = {}
        self.daily_trades = 0
        self.daily_pnl    = 0.0
        self.trade_log    = []

    def buy(self, symbol, usdt):
        if symbol in self.positions:
            return False
        if self.daily_trades >= CONFIG["max_trades_per_day"]:
            log.warning("[LIMIT] Max daily trades reached")
            return False
        qty = calculate_qty(self.client, symbol, usdt)
        if qty <= 0:
            return False
        try:
            order = self.client.order_market_buy(symbol=symbol, quantity=qty)
            price = float(order["fills"][0]["price"])
            self.positions[symbol] = {"price": price, "qty": float(order["executedQty"])}
            self.daily_trades += 1
            log.info(f"[BUY] {symbol} x{qty} @ ${price:.4f}")
            self._log("BUY", symbol, qty, price)
            return True
        except BinanceAPIException as e:
            log.error(f"[ERROR] BUY: {e}")
            return False

    def sell(self, symbol):
        if symbol not in self.positions:
            return False
        asset = symbol.replace("USDT","")
        qty   = round(get_asset_balance(self.client, asset) * 0.999, 6)
        try:
            order = self.client.order_market_sell(symbol=symbol, quantity=qty)
            price = float(order["fills"][0]["price"])
            entry = self.positions[symbol]["price"]
            pnl   = (price - entry) / entry * 100
            self.daily_pnl += pnl
            log.info(f"[SELL] {symbol} @ ${price:.4f} PnL:{pnl:.2f}%")
            self._log("SELL", symbol, qty, price, pnl)
            del self.positions[symbol]
            return True
        except BinanceAPIException as e:
            log.error(f"[ERROR] SELL: {e}")
            return False

    def check_sl_tp(self, symbol, price):
        if symbol not in self.positions:
            return
        entry = self.positions[symbol]["price"]
        chg   = (price - entry) / entry * 100
        if chg <= -CONFIG["stop_loss_pct"]:
            log.warning(f"[SL] {symbol} {chg:.2f}%")
            self.sell(symbol)
        elif chg >= CONFIG["take_profit_pct"]:
            log.info(f"[TP] {symbol} +{chg:.2f}%")
            self.sell(symbol)

    def _log(self, side, symbol, qty, price, pnl=None):
        self.trade_log.append({
            "time": datetime.now().isoformat(),
            "side": side, "symbol": symbol,
            "qty": qty, "price": price, "pnl": pnl
        })
        pd.DataFrame(self.trade_log).to_csv("trades.csv", index=False)


# ── Main Bot ──────────────────────────────────────────────
def run_bot():
    log.info("=" * 55)
    log.info("  AI Trading Bot v3 - Auto Retraining")
    log.info("=" * 55)

    client  = get_client()
    symbols = CONFIG["symbols"]
    interval = CONFIG["interval"]
    models  = {}
    retrain_counter = 0

    # ALWAYS train fresh model on startup!
    log.info("[AI] Training fresh model on CURRENT market data...")
    for symbol in symbols:
        df = fetch_ohlcv(client, symbol, interval)
        df = add_features(df)
        current_price = df["close"].iloc[-1]
        log.info(f"[AI] Current {symbol} price: ${current_price:.4f}")
        model, scaler = train_model(df.copy(), symbol, f"model_{symbol}.pkl")
        models[symbol] = {"model": model, "scaler": scaler}

    trader = TradeManager(client)
    log.info(f"[BOT] Trading: {', '.join(symbols)}")
    log.info("-" * 55)

    while True:
        try:
            usdt = get_usdt_balance(client)
            usdt_per = usdt / len(symbols)
            log.info(f"[BALANCE] ${usdt:.2f} | Per coin: ${usdt_per:.2f}")

            # Daily loss check
            if trader.daily_pnl <= -CONFIG["max_daily_loss_pct"]:
                log.warning("[LIMIT] Daily loss limit! Sleeping 1hr...")
                time.sleep(3600)
                continue

            # Fear & Greed
            fg_sig, fg_score = get_fear_greed()

            for symbol in symbols:
                log.info(f"\n--- {symbol} ---")
                df = fetch_ohlcv(client, symbol, interval)
                df = add_features(df)
                price = df["close"].iloc[-1]
                log.info(f"[PRICE] ${price:.4f}")

                # Check SL/TP
                trader.check_sl_tp(symbol, price)

                # Get signals
                m = models[symbol]
                t_sig, t_conf = get_tech_signal(df, m["model"], m["scaler"], CONFIG["confidence_threshold"])
                n_sig, _      = get_news(symbol)

                # Final decision
                final, strength = combined_signal(t_sig, t_conf, fg_sig, n_sig)
                log.info(f"[FINAL] {symbol}: {final} (strength:{strength:.3f})")

                if final == "BUY" and usdt_per >= CONFIG["min_usdt_trade"]:
                    trader.buy(symbol, usdt_per * CONFIG["risk_pct"])
                elif final == "SELL":
                    trader.sell(symbol)

            # Retrain periodically with fresh data
            retrain_counter += 1
            if retrain_counter >= CONFIG["retrain_every_n_cycles"]:
                log.info("[AI] Retraining with latest data...")
                for symbol in symbols:
                    df = fetch_ohlcv(client, symbol, interval)
                    df = add_features(df)
                    model, scaler = train_model(df.copy(), symbol, f"model_{symbol}.pkl")
                    models[symbol] = {"model": model, "scaler": scaler}
                retrain_counter = 0

            log.info(f"[WAIT] Sleeping {CONFIG['sleep_seconds']}s...\n")
            time.sleep(CONFIG["sleep_seconds"])

        except BinanceAPIException as e:
            log.error(f"[ERROR] Binance: {e}")
            time.sleep(60)
        except KeyboardInterrupt:
            log.info("[STOP] Bot stopped.")
            break
        except Exception as e:
            log.error(f"[ERROR] {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    run_bot()
