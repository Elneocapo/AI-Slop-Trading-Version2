import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_20d"] = close.pct_change(20)
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["ema_20"] = close.ewm(span=20, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    tr = pd.concat([out["High"] - out["Low"], (out["High"] - close.shift()).abs(), (out["Low"] - close.shift()).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    out["historical_vol_20"] = out["return_1d"].rolling(20).std() * np.sqrt(252)
    out["volume_change"] = out["Volume"].pct_change()
    return out


def option_liquidity_score(bid: float, ask: float, volume: int, open_interest: int) -> float:
    mid = (bid + ask) / 2
    if mid <= 0 or ask < bid:
        return 0.0
    spread = (ask - bid) / mid * 100
    spread_score = max(0.0, 1.0 - spread / 20.0)
    activity = min(1.0, np.log1p(max(volume, 0) + max(open_interest, 0)) / np.log1p(10000))
    return round(0.6 * spread_score + 0.4 * activity, 4)
