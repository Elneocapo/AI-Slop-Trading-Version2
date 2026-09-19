from __future__ import annotations

import argparse
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import databento as db
import numpy as np
import pandas as pd
import yfinance as yf

from app.training_v2 import DTE_DAYS, LOOKBACK, STRIKE_OFFSETS, load_hourly_data

ET = ZoneInfo("America/New_York")
DATASET = "OPRA.PILLAR"
# Databento bills historical usage in USD; keep the default below the user's €12 budget.
DEFAULT_MAX_COST_USD = 1.0


def _option_type(value: object, raw_symbol: object) -> str | None:
    text = str(value).upper()
    if text in {"C", "CALL"} or "CALL" in text:
        return "CALL"
    if text in {"P", "PUT"} or "PUT" in text:
        return "PUT"
    raw = str(raw_symbol)
    if len(raw) >= 9:
        marker = raw[-9].upper()
        if marker == "C":
            return "CALL"
        if marker == "P":
            return "PUT"
    return None


def _first_regular_spot(day_data: pd.DataFrame) -> float | None:
    ts = pd.DatetimeIndex(day_data["timestamp"])
    regular = (ts.hour * 60 + ts.minute >= 570) & (ts.hour * 60 + ts.minute <= 960)
    regular_data = day_data.loc[regular]
    if regular_data.empty:
        return None
    return float(regular_data["Close"].iloc[0])


def _select_daily_contracts(
    definitions: pd.DataFrame,
    trade_date: pd.Timestamp,
    spot: float,
) -> dict[int, str]:
    defs = definitions.copy()
    if "ts_event" in defs.columns:
        defs["ts_event"] = pd.to_datetime(defs["ts_event"], utc=True, errors="coerce")
        cutoff = trade_date.tz_convert("UTC") if trade_date.tzinfo is not None else trade_date.tz_localize(ET).tz_convert("UTC")
        defs = defs[defs["ts_event"] <= cutoff]
    defs["option_type"] = [
        _option_type(a, b) for a, b in zip(defs["instrument_class"], defs["raw_symbol"])
    ]
    defs = defs[defs["option_type"].isin(["CALL", "PUT"])]
    defs["expiration"] = pd.to_datetime(defs["expiration"], utc=True, errors="coerce").dt.tz_convert(ET)
    defs["strike_price"] = pd.to_numeric(defs["strike_price"], errors="coerce")
    defs = defs.dropna(subset=["expiration", "strike_price", "raw_symbol"])

    day = trade_date.date()
    dte = (defs["expiration"].dt.date - day).map(lambda x: x.days)
    defs = defs[(dte >= 1) & (dte <= max(DTE_DAYS) + 7)]
    if defs.empty:
        return {}

    selected: dict[int, str] = {}
    for option_type_idx, option_type in enumerate(["CALL", "PUT"]):
        typed = defs[defs["option_type"] == option_type]
        for strike_idx, offset in enumerate(STRIKE_OFFSETS):
            target_strike = spot * (1.0 + offset)
            for dte_idx, target_dte in enumerate(DTE_DAYS):
                if typed.empty:
                    continue
                score = (
                    (typed["strike_price"] - target_strike).abs() / max(spot, 1.0)
                    + (dte.loc[typed.index] - target_dte).abs() * 0.01
                )
                best_index = score.nsmallest(1).index
                if len(best_index) == 0:
                    continue
                candidate_idx = option_type_idx * len(STRIKE_OFFSETS) * len(DTE_DAYS) + strike_idx * len(DTE_DAYS) + dte_idx
                selected[candidate_idx] = str(typed.loc[best_index[0], "raw_symbol"])
    return selected


def _fetch_quotes(
    client: db.Historical,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    data = client.timeseries.get_range(
        dataset=DATASET,
        schema="cbbo-1m",
        stype_in="raw_symbol",
        symbols=symbols,
        start=start,
        end=end,
    )
    df = data.to_df().reset_index()
    if df.empty:
        return df
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True).dt.tz_convert(ET)
    df["bid"] = pd.to_numeric(df["bid_px_00"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask_px_00"], errors="coerce")
    df = df[(df["bid"] >= 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])]
    return df[["ts_recv", "symbol", "bid", "ask"]].sort_values(["symbol", "ts_recv"])


def build_real_options_panel(
    ticker: str,
    period: str,
    output_path: Path,
    api_key: str | None = None,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
) -> Path:
    api_key = api_key or os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DATABENTO_API_KEY is missing. Create a Databento account/API key and set "
            "$env:DATABENTO_API_KEY='YOUR_KEY' in PowerShell."
        )

    underlying = load_hourly_data(ticker, period)
    timestamps = pd.DatetimeIndex(underlying["timestamp"])
    local_dates = pd.Series(timestamps.date, index=underlying.index)
    trade_dates = pd.Index(sorted(local_dates.unique()))

    if max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be greater than 0.")
    client = db.Historical(api_key)
    definition_start = pd.Timestamp(trade_dates.min(), tz=ET) - pd.Timedelta(days=max(DTE_DAYS) + 15)
    definition_end = pd.Timestamp(trade_dates.max(), tz=ET) + pd.Timedelta(days=1)

    estimated_definition_cost = float(
        client.metadata.get_cost(
            dataset=DATASET,
            symbols=[f"{ticker}.OPT"],
            schema="definition",
            start=definition_start,
            end=definition_end,
            stype_in="parent",
        )
    )
    if not np.isfinite(estimated_definition_cost) or estimated_definition_cost < 0:
        raise RuntimeError(
            f"Databento returned an invalid definition cost estimate: {estimated_definition_cost}"
        )
    if estimated_definition_cost > max_cost_usd:
        raise RuntimeError(
            "Databento cost guard stopped before downloading option definitions. "
            f"Estimated definition cost: ${estimated_definition_cost:,.4f}; "
            f"hard limit: ${max_cost_usd:,.2f}."
        )
    print(
        f"[cost] definitions: ${estimated_definition_cost:,.4f} | "
        f"remaining quote-data budget ${max_cost_usd - estimated_definition_cost:,.4f}"
    )

    definitions = client.timeseries.get_range(
        dataset=DATASET,
        schema="definition",
        stype_in="parent",
        symbols=f"{ticker}.OPT",
        start=definition_start,
        end=definition_end,
    ).to_df().reset_index()
    if definitions.empty:
        raise RuntimeError(f"No OPRA option definitions returned for {ticker}.")

    rows: list[dict] = []
    estimated_cost_usd = estimated_definition_cost
    cost_days = 0
    for trade_date in trade_dates:
        mask = local_dates == trade_date
        day_data = underlying.loc[mask]
        spot = _first_regular_spot(day_data)
        if spot is None or not np.isfinite(spot):
            continue

        day_ts = pd.DatetimeIndex(day_data["timestamp"])
        regular = (day_ts.hour * 60 + day_ts.minute >= 570) & (day_ts.hour * 60 + day_ts.minute <= 960)
        regular_ts = day_ts[regular]
        if len(regular_ts) == 0:
            continue

        selected = _select_daily_contracts(definitions, pd.Timestamp(trade_date, tz=ET), spot)
        if not selected:
            continue

        day_start = pd.Timestamp(trade_date, tz=ET) + pd.Timedelta(hours=9, minutes=30)
        day_end = pd.Timestamp(trade_date, tz=ET) + pd.Timedelta(hours=16)
        quote_symbols = sorted(set(selected.values()))

        # Metadata pricing is free, so enforce the hard ceiling before any
        # billable historical quote bytes are requested.
        estimated_day_cost = float(
            client.metadata.get_cost(
                dataset=DATASET,
                symbols=quote_symbols,
                schema="cbbo-1m",
                start=day_start,
                end=day_end,
                stype_in="raw_symbol",
            )
        )
        if not np.isfinite(estimated_day_cost) or estimated_day_cost < 0:
            raise RuntimeError(
                f"Databento returned an invalid cost estimate for {trade_date}: {estimated_day_cost}"
            )
        if estimated_cost_usd + estimated_day_cost > max_cost_usd:
            raise RuntimeError(
                "Databento cost guard stopped the download before requesting the next "
                f"billable batch. Estimated cumulative cost: ${estimated_cost_usd:,.4f}; "
                f"next batch: ${estimated_day_cost:,.4f}; hard limit: ${max_cost_usd:,.2f}. "
                "No output panel was written. Reduce --period or narrow the request."
            )
        estimated_cost_usd += estimated_day_cost
        cost_days += 1
        print(
            f"[cost] {trade_date}: ${estimated_day_cost:,.4f} | "
            f"cumulative ${estimated_cost_usd:,.4f} / ${max_cost_usd:,.2f}"
        )

        quotes = _fetch_quotes(client, quote_symbols, day_start, day_end)

        quote_by_symbol = {
            symbol: group for symbol, group in quotes.groupby("symbol", sort=False)
        } if not quotes.empty else {}

        for candidate_idx, symbol in selected.items():
            q = quote_by_symbol.get(symbol)
            if q is None or q.empty:
                continue
            base = pd.DataFrame({"timestamp": regular_ts})
            merged = pd.merge_asof(
                base.sort_values("timestamp"),
                q.rename(columns={"ts_recv": "quote_ts"}).sort_values("quote_ts"),
                left_on="timestamp",
                right_on="quote_ts",
                direction="backward",
            )
            meta = definitions[definitions["raw_symbol"].astype(str) == symbol]
            if meta.empty:
                continue
            definition = meta.iloc[-1]
            expiry = pd.to_datetime(definition["expiration"], utc=True).tz_convert(ET)
            strike = float(definition["strike_price"])
            option_type = _option_type(definition["instrument_class"], symbol)
            if option_type is None:
                continue
            for ts, bid, ask in zip(merged["timestamp"], merged["bid"], merged["ask"]):
                if not (np.isfinite(bid) and np.isfinite(ask) and ask > 0 and ask >= bid):
                    continue
                rows.append({
                    "timestamp": ts.isoformat(),
                    "candidate_idx": int(candidate_idx),
                    "symbol": symbol,
                    "strike": strike,
                    "expiry": expiry.isoformat(),
                    "option_type": option_type,
                    "bid": float(bid),
                    "ask": float(ask),
                })

    print(
        f"Databento estimated quote-data cost before download batches: "
        f"${estimated_cost_usd:,.4f} across {cost_days} trading days."
    )
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RuntimeError("No usable historical OPRA quotes were returned.")
    panel = panel.drop_duplicates(["timestamp", "candidate_idx"]).sort_values(["timestamp", "candidate_idx"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if str(output_path).endswith(".gz") else None
    panel.to_csv(output_path, index=False, compression=compression)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real historical NVDA options NBBO data from Databento OPRA.")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--period", default="730d")
    parser.add_argument("--output", default="data/nvda_real_options.csv.gz")
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=DEFAULT_MAX_COST_USD,
        help="Hard cumulative Databento historical quote-data ceiling in USD (default: $1).",
    )
    args = parser.parse_args()
    path = build_real_options_panel(
        args.ticker.upper(),
        args.period,
        Path(args.output),
        max_cost_usd=args.max_cost_usd,
    )
    print(f"Saved real options panel: {path}")


if __name__ == "__main__":
    main()
