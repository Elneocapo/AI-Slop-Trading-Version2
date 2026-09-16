from datetime import datetime
import pandas as pd
import yfinance as yf


def get_historical_prices(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No market data returned for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing market columns: {missing}")
    return data[required].dropna().copy()


def get_stock_quote(ticker: str) -> dict:
    quote = yf.Ticker(ticker).fast_info
    return {
        "ticker": ticker.upper(),
        "price": float(quote["last_price"]),
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
    }


def get_options_chain(ticker: str, expiration: str | None = None):
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        return None
    selected = expiration or expirations[0]
    if selected not in expirations:
        raise ValueError(f"Expiration {selected} unavailable for {ticker}")
    return stock.option_chain(selected)
