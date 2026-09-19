from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from app.environment.options_env import DTE_DAYS, STRIKE_OFFSETS

ET = "America/New_York"
DATA_BASE = "https://data.alpaca.markets/v1beta1"
TRADING_BASE = "https://paper-api.alpaca.markets/v2"


def _headers(api_key: str, secret_key: str) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Accept": "application/json",
    }


def _get_json(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    params: dict,
    *,
    max_retries: int = 6,
) -> dict:
    for attempt in range(max_retries):
        response = session.get(url, headers=headers, params=params, timeout=60)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(2.0 ** attempt, 15.0)
            time.sleep(delay)
            continue
        if not response.ok:
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(
                f"Alpaca API error {response.status_code} for {url}: {detail}"
            )
        return response.json()
    raise RuntimeError(f"Alpaca API rate limit persisted after {max_retries} retries: {url}")


def _load_hourly_data(ticker: str, period: str) -> pd.DataFrame:
    # Imported lazily so this downloader never creates an import cycle when
    # training_v2 imports the real-options environment.
    from app.training_v2 import load_hourly_data

    return load_hourly_data(ticker, period)


def _fetch_option_contracts(
    session: requests.Session,
    api_headers: dict[str, str],
    underlying: str,
    expiration_start: pd.Timestamp,
    expiration_end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict] = []

    for status in ("active", "inactive"):
        page_token = None
        while True:
            params = {
                "underlying_symbols": underlying,
                "status": status,
                "expiration_date_gte": expiration_start.date().isoformat(),
                "expiration_date_lte": expiration_end.date().isoformat(),
                "limit": 1000,
            }
            if page_token:
                params["page_token"] = page_token

            payload = _get_json(
                session,
                f"{TRADING_BASE}/options/contracts",
                api_headers,
                params,
            )
            contracts = payload.get("option_contracts") or payload.get("contracts") or []
            rows.extend(contracts)

            page_token = payload.get("next_page_token")
            if not page_token:
                break

    if not rows:
        raise RuntimeError(f"No Alpaca option contracts found for {underlying}.")

    contracts = pd.DataFrame(rows)
    required = {"symbol", "underlying_symbol", "type", "strike_price", "expiration_date"}
    missing = required - set(contracts.columns)
    if missing:
        raise RuntimeError(
            f"Alpaca contract response is missing fields: {sorted(missing)}"
        )

    contracts = contracts.drop_duplicates("symbol").copy()
    contracts["symbol"] = contracts["symbol"].astype(str)
    contracts["underlying_symbol"] = contracts["underlying_symbol"].astype(str)
    contracts["type"] = contracts["type"].astype(str).str.upper()
    contracts["strike_price"] = pd.to_numeric(
        contracts["strike_price"], errors="coerce"
    )
    contracts["expiration_date"] = pd.to_datetime(
        contracts["expiration_date"], errors="coerce"
    )
    contracts = contracts[
        contracts["underlying_symbol"].str.upper() == underlying.upper()
    ]
    contracts = contracts[
        contracts["type"].isin(["CALL", "PUT"])
        & contracts["strike_price"].notna()
        & contracts["expiration_date"].notna()
    ].copy()
    contracts["expiration_date"] = contracts["expiration_date"].dt.date

    return contracts.reset_index(drop=True)


def _select_daily_contracts(
    contracts: pd.DataFrame,
    trade_date: pd.Timestamp,
    spot: float,
) -> dict[int, str]:
    day = trade_date.date()
    expiration_days = (
        pd.to_datetime(contracts["expiration_date"]).dt.date
        - day
    ).map(lambda x: x.days)

    eligible = contracts.assign(dte=expiration_days)
    eligible = eligible[(eligible["dte"] >= 1) & (eligible["dte"] <= max(DTE_DAYS) + 7)]
    if eligible.empty:
        return {}

    selected: dict[int, str] = {}
    per_type = len(STRIKE_OFFSETS) * len(DTE_DAYS)

    for option_type_idx, option_type in enumerate(["CALL", "PUT"]):
        typed = eligible[eligible["type"] == option_type]
        if typed.empty:
            continue

        for strike_idx, offset in enumerate(STRIKE_OFFSETS):
            target_strike = spot * (1.0 + float(offset))
            for dte_idx, target_dte in enumerate(DTE_DAYS):
                score = (
                    (typed["strike_price"] - target_strike).abs() / max(spot, 1.0)
                    + (typed["dte"] - target_dte).abs() * 0.01
                )
                best = score.nsmallest(1)
                if best.empty:
                    continue
                symbol = str(typed.loc[best.index[0], "symbol"])
                candidate_idx = (
                    option_type_idx * per_type
                    + strike_idx * len(DTE_DAYS)
                    + dte_idx
                )
                selected[candidate_idx] = symbol

    return selected


def _fetch_option_bars(
    session: requests.Session,
    api_headers: dict[str, str],
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for offset in range(0, len(symbols), 100):
        chunk = symbols[offset : offset + 100]
        page_token = None

        while True:
            params = {
                "symbols": ",".join(chunk),
                "timeframe": "1Hour",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": 10000,
                "sort": "asc",
                "feed": "indicative",
            }
            if page_token:
                params["page_token"] = page_token

            payload = _get_json(
                session,
                f"{DATA_BASE}/options/bars",
                api_headers,
                params,
            )
            bars_by_symbol = payload.get("bars") or {}
            page_rows = []

            for symbol, bars in bars_by_symbol.items():
                for bar in bars:
                    page_rows.append(
                        {
                            "symbol": str(symbol),
                            "timestamp": bar.get("t"),
                            "open": bar.get("o"),
                            "high": bar.get("h"),
                            "low": bar.get("l"),
                            "close": bar.get("c"),
                        }
                    )

            if page_rows:
                frames.append(pd.DataFrame(page_rows))

            page_token = payload.get("next_page_token")
            if not page_token:
                break

        # Stay comfortably below Alpaca's Basic historical request-rate ceiling.
        time.sleep(0.35)

    if not frames:
        raise RuntimeError("Alpaca returned no historical option bars.")

    bars = pd.concat(frames, ignore_index=True)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars["timestamp"] = bars["timestamp"].dt.tz_convert(ET)
    for column in ("open", "high", "low", "close"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.dropna(
        subset=["symbol", "timestamp", "high", "low", "close"]
    )
    bars = bars[(bars["close"] > 0) & (bars["high"] > 0) & (bars["low"] > 0)]
    bars = bars[bars["high"] >= bars["low"]]
    bars = bars.drop_duplicates(["symbol", "timestamp"], keep="last")
    return bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _proxy_bid_ask(close: float, high: float, low: float) -> tuple[float, float]:
    """Conservative bid/ask proxy; Alpaca Basic historical options has no NBBO history."""
    close = float(close)
    range_ = max(float(high) - float(low), 0.0)
    half_spread = max(0.01, range_ * 0.25)
    half_spread = min(half_spread, max(close * 0.50, 0.01))
    bid = max(close - half_spread, 0.0)
    ask = max(close + half_spread, 0.01)
    bid = round(bid, 2)
    ask = round(ask, 2)
    if bid > 0 and ask <= bid:
        ask = round(bid + 0.01, 2)
    return bid, ask


def build_alpaca_options_panel(
    ticker: str,
    period: str,
    output_path: Path,
    api_key: str | None = None,
    secret_key: str | None = None,
) -> Path:
    api_key = api_key or os.getenv("ALPACA_API_KEY")
    secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. In PowerShell use "
            "$env:ALPACA_API_KEY='YOUR_KEY' and "
            "$env:ALPACA_SECRET_KEY='YOUR_SECRET'."
        )

    underlying = _load_hourly_data(ticker, period)
    timestamps = pd.DatetimeIndex(underlying["timestamp"])
    local_dates = pd.Series(timestamps.date, index=underlying.index)
    trade_dates = sorted(local_dates.unique())
    if not trade_dates:
        raise RuntimeError(f"No trading dates found for {ticker}.")

    first_day = pd.Timestamp(trade_dates[0], tz=ET)
    last_day = pd.Timestamp(trade_dates[-1], tz=ET)
    contracts = None

    session = requests.Session()
    api_headers = _headers(api_key, secret_key)
    contracts = _fetch_option_contracts(
        session,
        api_headers,
        ticker,
        first_day,
        last_day + pd.Timedelta(days=max(DTE_DAYS) + 7),
    )

    selections: list[dict] = []
    for trade_date in trade_dates:
        mask = local_dates == trade_date
        day_data = underlying.loc[mask]
        day_ts = pd.DatetimeIndex(day_data["timestamp"])
        regular = (day_ts.hour * 60 + day_ts.minute >= 570) & (
            day_ts.hour * 60 + day_ts.minute <= 960
        )
        regular_ts = day_ts[regular]
        if len(regular_ts) == 0:
            continue

        first_regular = day_data.loc[regular, "Close"].iloc[0]
        spot = float(first_regular)
        if not np.isfinite(spot) or spot <= 0:
            continue

        selected = _select_daily_contracts(
            contracts,
            pd.Timestamp(trade_date, tz=ET),
            spot,
        )
        for candidate_idx, symbol in selected.items():
            meta = contracts[contracts["symbol"] == symbol]
            if meta.empty:
                continue
            row = meta.iloc[0]
            expiry_date = pd.Timestamp(row["expiration_date"], tz=ET)
            option_type = str(row["type"]).upper()
            strike = float(row["strike_price"])
            for timestamp in regular_ts:
                selections.append(
                    {
                        "timestamp": pd.Timestamp(timestamp),
                        "trade_date": trade_date,
                        "candidate_idx": int(candidate_idx),
                        "symbol": symbol,
                        "strike": strike,
                        "expiry": expiry_date.isoformat(),
                        "option_type": option_type,
                    }
                )

    base = pd.DataFrame(selections)
    if base.empty:
        raise RuntimeError("Could not select any Alpaca option candidates.")

    symbols = sorted(base["symbol"].unique())
    bars = _fetch_option_bars(
        session,
        api_headers,
        symbols,
        first_day + pd.Timedelta(hours=9, minutes=30),
        last_day + pd.Timedelta(days=1),
    )
    bars_by_symbol = {
        symbol: group.sort_values("timestamp").reset_index(drop=True)
        for symbol, group in bars.groupby("symbol", sort=False)
    }

    output_rows: list[dict] = []
    for symbol, requested in base.groupby("symbol", sort=False):
        quotes = bars_by_symbol.get(symbol)
        if quotes is None or quotes.empty:
            continue
        merged = pd.merge_asof(
            requested.sort_values("timestamp"),
            quotes[["timestamp", "open", "high", "low", "close"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        merged = merged.dropna(subset=["close", "high", "low"])
        for row in merged.itertuples(index=False):
            bid, ask = _proxy_bid_ask(row.close, row.high, row.low)
            if ask <= 0 or ask < bid:
                continue
            output_rows.append(
                {
                    "timestamp": pd.Timestamp(row.timestamp).isoformat(),
                    "candidate_idx": int(row.candidate_idx),
                    "symbol": str(row.symbol),
                    "strike": float(row.strike),
                    "expiry": str(row.expiry),
                    "option_type": str(row.option_type),
                    "bid": float(bid),
                    "ask": float(ask),
                    "quote_source": "alpaca_indicative_1h_proxy",
                }
            )

    panel = pd.DataFrame(output_rows)
    if panel.empty:
        raise RuntimeError(
            "Alpaca returned bars but no usable candidate quotes could be built."
        )

    panel = panel.drop_duplicates(
        ["timestamp", "candidate_idx"], keep="last"
    ).sort_values(["timestamp", "candidate_idx"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if str(output_path).endswith(".gz") else None
    panel.to_csv(output_path, index=False, compression=compression)

    print(
        f"Built Alpaca indicative option panel: {len(panel):,} rows, "
        f"{panel['symbol'].nunique():,} contracts."
    )
    print(f"Quote source: Alpaca Indicative 1h bars + conservative proxy spread.")
    print(f"Saved: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build historical NVDA options panel from Alpaca Basic/Indicative data."
    )
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--period", default="730d")
    parser.add_argument(
        "--output",
        default="data/nvda_alpaca_indicative_options.csv.gz",
    )
    args = parser.parse_args()

    build_alpaca_options_panel(
        args.ticker.upper(),
        args.period,
        Path(args.output),
    )


if __name__ == "__main__":
    main()
