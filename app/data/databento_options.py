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
DEFAULT_MAX_COST_USD = 20.0


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



def _effective_definition(
    definitions: pd.DataFrame,
    raw_symbol: str,
    trade_date: pd.Timestamp,
) -> pd.Series | None:
    """Return the definition effective on the requested trading date."""
    meta = definitions[
        definitions["raw_symbol"].astype(str) == str(raw_symbol)
    ].copy()
    if meta.empty:
        return None
    if "ts_event" in meta.columns:
        events = pd.to_datetime(meta["ts_event"], utc=True, errors="coerce")
        cutoff = trade_date.tz_convert("UTC")
        meta = meta.loc[events.notna() & (events <= cutoff)].copy()
        if meta.empty:
            return None
        meta["ts_event"] = pd.to_datetime(
            meta["ts_event"], utc=True, errors="coerce"
        )
        meta = meta.sort_values("ts_event")
    return meta.iloc[-1]


def _fetch_quotes(
    client: db.Historical,
    symbols: list[int],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    # Raw-symbol input is supported; Databento returns instrument_id by
    # default, which is the valid raw_symbol -> instrument_id combination.
    data = client.timeseries.get_range(
        dataset=DATASET,
        schema="cbbo-1m",
        stype_in="instrument_id",
        symbols=symbols,
        start=start,
        end=end,
    )
    df = data.to_df().reset_index()
    if df.empty:
        return df
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True).dt.tz_convert(ET)
    df["instrument_id"] = pd.to_numeric(
        df["instrument_id"], errors="coerce"
    )
    df["bid"] = pd.to_numeric(df["bid_px_00"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask_px_00"], errors="coerce")
    df = df[
        df["instrument_id"].notna()
        & (df["bid"] >= 0)
        & (df["ask"] > 0)
        & (df["ask"] >= df["bid"])
    ].copy()
    df["instrument_id"] = df["instrument_id"].astype(int)
    return df[["ts_recv", "instrument_id", "bid", "ask"]].sort_values(
        ["instrument_id", "ts_recv"]
    )


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

    # Databento availability is schema-specific. Clamp requests to the
    # currently available exclusive end returned by the metadata API.
    dataset_range = client.metadata.get_dataset_range(dataset=DATASET)
    schema_ranges = dataset_range.get("schema", {}) if isinstance(dataset_range, dict) else {}
    definition_range = schema_ranges.get("definition", dataset_range)
    quote_range = schema_ranges.get("cbbo-1m", dataset_range)
    definition_available_start = pd.Timestamp(definition_range["start"], tz="UTC")
    definition_available_end = pd.Timestamp(definition_range["end"], tz="UTC")
    quote_available_end = pd.Timestamp(quote_range["end"], tz="UTC")

    requested_definition_start = (
        pd.Timestamp(trade_dates.min(), tz=ET) - pd.Timedelta(days=max(DTE_DAYS) + 15)
    ).tz_convert("UTC").floor("D")
    requested_definition_end = (
        pd.Timestamp(trade_dates.max(), tz=ET) + pd.Timedelta(days=1)
    ).tz_convert("UTC")
    definition_start = max(requested_definition_start, definition_available_start)
    definition_end = min(requested_definition_end, definition_available_end)

    if definition_start >= definition_end:
        raise RuntimeError(
            "No usable Databento definition range overlaps the requested period."
        )

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

        day_start = (
            pd.Timestamp(trade_date, tz=ET) + pd.Timedelta(hours=9, minutes=30)
        ).tz_convert("UTC")
        requested_day_end = (
            pd.Timestamp(trade_date, tz=ET) + pd.Timedelta(hours=16)
        ).tz_convert("UTC")
        day_end = min(requested_day_end, quote_available_end)
        if day_start >= day_end:
            continue
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

        # Resolve OCC raw symbols to instrument IDs for this exact
        # historical day. Instrument IDs are only guaranteed to be unique
        # within a given day.
        resolution = client.symbology.resolve(
            dataset=DATASET,
            symbols=quote_symbols,
            stype_in="raw_symbol",
            stype_out="instrument_id",
            start_date=pd.Timestamp(trade_date).date().isoformat(),
            end_date=pd.Timestamp(trade_date).date().isoformat(),
        )
        resolved = resolution.get("result", {}) if isinstance(resolution, dict) else {}
        symbol_to_instrument: dict[str, int] = {}
        for raw_symbol, intervals in resolved.items():
            if not intervals:
                continue
            interval = intervals[-1]
            try:
                symbol_to_instrument[str(raw_symbol)] = int(interval["s"])
            except (KeyError, TypeError, ValueError):
                continue

        instrument_ids = sorted(set(symbol_to_instrument.values()))
        if not instrument_ids:
            raise RuntimeError(
                f"Databento could not resolve any selected NVDA option symbols on {trade_date}."
            )

        quotes = _fetch_quotes(client, instrument_ids, day_start, day_end)

        quote_by_instrument = (
            {
                int(instrument_id): group
                for instrument_id, group in quotes.groupby(
                    "instrument_id", sort=False
                )
            }
            if not quotes.empty
            else {}
        )

        for candidate_idx, symbol in selected.items():
            if str(symbol) not in symbol_to_instrument:
                continue
            instrument_id = symbol_to_instrument[str(symbol)]
            q = quote_by_instrument.get(instrument_id)
            if q is None or q.empty:
                continue

            base = pd.DataFrame({"timestamp": regular_ts})
            base["merge_ts_ns"] = (
                pd.to_datetime(base["timestamp"], utc=True).astype("int64")
            )
            quote_frame = q.rename(columns={"ts_recv": "quote_ts"}).copy()
            quote_frame["merge_ts_ns"] = (
                pd.to_datetime(quote_frame["quote_ts"], utc=True).astype("int64")
            )
            if quote_frame["merge_ts_ns"].empty:
                continue
            merged = pd.merge_asof(
                base.sort_values("merge_ts_ns"),
                quote_frame.sort_values("merge_ts_ns"),
                on="merge_ts_ns",
                direction="backward",
            )
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
        raise RuntimeError(
            "No usable historical OPRA quotes were returned after symbol/time matching. "
            "The request itself succeeded; check contract availability and quote coverage."
        )
    panel = panel.drop_duplicates(["timestamp", "candidate_idx"]).sort_values(["timestamp", "candidate_idx"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if str(output_path).endswith(".gz") else None
    panel.to_csv(output_path, index=False, compression=compression)
    return output_path



def _read_batch_data_files(raw_dir: Path) -> pd.DataFrame:
    files = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and (path.name.endswith(".csv") or path.name.endswith(".csv.zst"))
        and path.name not in {"metadata.csv", "symbology.csv"}
    )
    if not files:
        raise RuntimeError(
            f"No Databento batch CSV files found in {raw_dir}."
        )

    frames = []
    for path in files:
        compression = "zstd" if path.name.endswith(".zst") else None
        frame = pd.read_csv(path, compression=compression)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError(
            "Databento batch files were downloaded but contained no records."
        )
    return pd.concat(frames, ignore_index=True)


def _wait_for_batch_job(
    client: db.Historical,
    job_id: str,
    poll_seconds: float = 3.0,
    max_polls: int = 600,
) -> dict:
    for _ in range(max_polls):
        details = client.batch.get_job_details(job_id)
        state = str(details.get("state", "")).lower()
        progress = details.get("progress")
        if progress is None:
            print(f"[batch] state={state}")
        else:
            print(f"[batch] state={state} progress={progress}%")

        if state == "done":
            return details
        if state in {"expired", "failed", "cancelled"}:
            raise RuntimeError(
                f"Databento batch {job_id} ended in state={state}: {details}"
            )
        time.sleep(poll_seconds)

    raise TimeoutError(
        f"Databento batch {job_id} did not finish within the polling window."
    )


def _load_json_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[cache] Ignoring unreadable manifest {path}: {exc}")
        return None


def _save_json_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def build_real_options_panel(
    ticker: str,
    period: str,
    output_path: Path,
    api_key: str | None = None,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
    cache_dir: Path | None = None,
) -> Path:
    api_key = api_key or os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DATABENTO_API_KEY is missing. In PowerShell use "
            "$env:DATABENTO_API_KEY='YOUR_KEY'."
        )
    if max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be greater than 0.")

    # The processed panel is the first cache layer: training can run entirely
    # offline once this file exists.
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[cache] Using existing local panel: {output_path}")
        return output_path

    cache_dir = cache_dir or Path(DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    underlying = load_hourly_data(ticker, period)
    timestamps = pd.DatetimeIndex(underlying["timestamp"])
    local_dates = pd.Series(timestamps.date, index=underlying.index)
    trade_dates = sorted(local_dates.unique())
    if not trade_dates:
        raise RuntimeError(f"No trading dates found for {ticker}.")

    client = db.Historical(api_key)
    dataset_range = client.metadata.get_dataset_range(dataset=DATASET)

    def_range_data = (
        dataset_range.get("schema", {}).get("definition", dataset_range)
        if isinstance(dataset_range, dict)
        else dataset_range
    )
    quote_range_data = (
        dataset_range.get("schema", {}).get("cbbo-1m", dataset_range)
        if isinstance(dataset_range, dict)
        else dataset_range
    )
    definition_available_start = pd.Timestamp(
        def_range_data["start"], tz="UTC"
    )
    definition_available_end = pd.Timestamp(
        def_range_data["end"], tz="UTC"
    )
    quote_available_start = pd.Timestamp(
        quote_range_data["start"], tz="UTC"
    )
    quote_available_end = pd.Timestamp(
        quote_range_data["end"], tz="UTC"
    )

    requested_definition_start = (
        pd.Timestamp(trade_dates[0], tz=ET)
        - pd.Timedelta(days=max(DTE_DAYS) + 15)
    ).tz_convert("UTC").floor("D")
    requested_definition_end = (
        pd.Timestamp(trade_dates[-1], tz=ET)
        + pd.Timedelta(days=max(DTE_DAYS) + 7)
    ).tz_convert("UTC")
    definition_start = max(
        requested_definition_start,
        definition_available_start,
    )
    definition_end = min(
        requested_definition_end,
        definition_available_end,
    )
    if definition_start >= definition_end:
        raise RuntimeError(
            "The requested period does not overlap the available "
            "Databento definition range."
        )

    definition_cache = cache_dir / f"{ticker.lower()}_definitions.csv"
    if definition_cache.exists() and definition_cache.stat().st_size > 0:
        definitions = pd.read_csv(definition_cache)
        definition_cost_usd = 0.0
        print(f"[cache] Using local definitions: {definition_cache}")
    else:
        definition_cost_usd = float(
            client.metadata.get_cost(
                dataset=DATASET,
                symbols=[f"{ticker}.OPT"],
                schema="definition",
                start=definition_start,
                end=definition_end,
                stype_in="parent",
            )
        )
        if not np.isfinite(definition_cost_usd) or definition_cost_usd < 0:
            raise RuntimeError(
                f"Invalid Databento definition cost estimate: {definition_cost_usd}"
            )
        if definition_cost_usd > max_cost_usd:
            raise RuntimeError(
                "Cost guard stopped before downloading definitions: "
                f"estimated {definition_cost_usd:.4f} USD exceeds "
                f"the {max_cost_usd:.2f} USD limit."
            )

        print(
            f"[cost] definitions: {definition_cost_usd:.4f} USD | "
            f"remaining budget {max_cost_usd - definition_cost_usd:.4f} USD"
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
            raise RuntimeError(
                f"No OPRA option definitions returned for {ticker}."
            )
        definitions.to_csv(definition_cache, index=False)

    selected_by_day: dict[object, dict[int, str]] = {}
    all_symbols: set[str] = set()

    for trade_date in trade_dates:
        day_data = underlying.loc[local_dates == trade_date]
        spot = _first_regular_spot(day_data)
        if spot is None or not np.isfinite(spot):
            continue

        selected = _select_daily_contracts(
            definitions,
            pd.Timestamp(trade_date, tz=ET),
            spot,
        )
        if selected:
            selected_by_day[trade_date] = selected
            all_symbols.update(selected.values())

    if not all_symbols:
        raise RuntimeError(
            f"No selectable OPRA option contracts were found for {ticker}."
        )

    quote_start = max(
        pd.Timestamp(trade_dates[0], tz=ET).tz_convert("UTC"),
        quote_available_start,
    )
    quote_end = min(
        (
            pd.Timestamp(trade_dates[-1], tz=ET)
            + pd.Timedelta(days=1)
        ).tz_convert("UTC"),
        quote_available_end,
    )
    if quote_start >= quote_end:
        raise RuntimeError(
            "The requested period does not overlap the available OPRA quote range."
        )

    manifest_path = (
        cache_dir / f"{ticker.lower()}_{period}_opra_batch_cache.json"
    )
    raw_dir = (
        cache_dir / f"{ticker.lower()}_{period}_opra_batch_cache"
    )
    manifest = _load_json_manifest(manifest_path)

    if manifest and manifest.get("job_id"):
        job_id = str(manifest["job_id"])
        print(f"[cache] Reusing Databento batch job: {job_id}")
        details = client.batch.get_job_details(job_id)
        state = str(details.get("state", "")).lower()
        if state != "done":
            details = _wait_for_batch_job(client, job_id)

        local_data_files = [
            path for path in raw_dir.rglob("*")
            if path.is_file()
            and (path.name.endswith(".csv") or path.name.endswith(".csv.zst"))
        ]
        if not local_data_files:
            client.batch.download(
                job_id=job_id,
                output_dir=raw_dir,
                keep_zip=False,
            )
            print(f"[cache] Downloaded cached batch into {raw_dir}")
    else:
        symbols = sorted(all_symbols)
        estimated_quote_cost = float(
            client.metadata.get_cost(
                dataset=DATASET,
                symbols=symbols,
                schema="cbbo-1m",
                start=quote_start,
                end=quote_end,
                stype_in="raw_symbol",
                stype_out="instrument_id",
            )
        )
        if not np.isfinite(estimated_quote_cost) or estimated_quote_cost < 0:
            raise RuntimeError(
                f"Invalid Databento quote cost estimate: {estimated_quote_cost}"
            )

        estimated_total = definition_cost_usd + estimated_quote_cost
        print(
            f"[cost] one OPRA batch: {len(symbols):,} contracts | "
            f"quotes {estimated_quote_cost:.4f} USD | "
            f"estimated total {estimated_total:.4f} / {max_cost_usd:.2f} USD"
        )
        if estimated_total > max_cost_usd:
            raise RuntimeError(
                "Databento cost guard stopped before the batch was submitted. "
                f"Definitions {definition_cost_usd:.4f} USD + "
                f"quotes {estimated_quote_cost:.4f} USD = "
                f"{estimated_total:.4f} USD, above the {max_cost_usd:.2f} USD limit."
            )

        job = client.batch.submit_job(
            dataset=DATASET,
            symbols=symbols,
            schema="cbbo-1m",
            start=quote_start,
            end=quote_end,
            encoding="csv",
            compression="zstd",
            pretty_px=True,
            pretty_ts=True,
            map_symbols=True,
            split_duration="day",
            stype_in="raw_symbol",
            stype_out="instrument_id",
        )
        job_id = str(job["id"])
        manifest = {
            "job_id": job_id,
            "ticker": ticker,
            "period": period,
            "symbols": symbols,
            "quote_start": quote_start.isoformat(),
            "quote_end": quote_end.isoformat(),
            "estimated_definition_cost_usd": definition_cost_usd,
            "estimated_quote_cost_usd": estimated_quote_cost,
            "estimated_total_cost_usd": estimated_total,
            "max_cost_usd": max_cost_usd,
            "dataset": DATASET,
            "schema": "cbbo-1m",
            "encoding": "csv",
            "compression": "zstd",
            "cache_dir": str(raw_dir),
        }
        _save_json_manifest(manifest_path, manifest)
        print(f"[batch] Submitted: {job_id}")

        details = _wait_for_batch_job(client, job_id)
        actual_cost = details.get("cost_usd")
        if actual_cost is not None:
            actual_cost = float(actual_cost)
            print(f"[batch] Actual batch cost: {actual_cost:.4f} USD")
        manifest["actual_batch_cost_usd"] = actual_cost
        _save_json_manifest(manifest_path, manifest)

        raw_dir.mkdir(parents=True, exist_ok=True)
        client.batch.download(
            job_id=job_id,
            output_dir=raw_dir,
            keep_zip=False,
        )
        print(f"[cache] Raw batch saved locally in {raw_dir}")

    raw_df = _read_batch_data_files(raw_dir)
    required = {"symbol", "ts_recv", "bid_px_00", "ask_px_00"}
    missing = required - set(raw_df.columns)
    if missing:
        raise RuntimeError(
            f"Databento cbbo-1m batch is missing fields: {sorted(missing)}"
        )

    raw_df["symbol"] = raw_df["symbol"].astype(str)
    raw_df["timestamp"] = pd.to_datetime(
        raw_df["ts_recv"], utc=True, errors="coerce"
    ).dt.tz_convert(ET)
    raw_df["bid"] = pd.to_numeric(
        raw_df["bid_px_00"], errors="coerce"
    )
    raw_df["ask"] = pd.to_numeric(
        raw_df["ask_px_00"], errors="coerce"
    )
    raw_df = raw_df.dropna(
        subset=["symbol", "timestamp", "bid", "ask"]
    )
    raw_df = raw_df[
        (raw_df["bid"] >= 0)
        & (raw_df["ask"] > 0)
        & (raw_df["ask"] >= raw_df["bid"])
    ].sort_values(["symbol", "timestamp"])

    # The selected symbols are already limited to the contracts the RL
    # environment can actually choose. Everything else in the batch is
    # ignored when constructing the compact panel.
    selected_symbols = all_symbols
    raw_df = raw_df[raw_df["symbol"].isin(selected_symbols)]
    if raw_df.empty:
        raise RuntimeError(
            "Databento batch contains no usable quotes for the selected contracts."
        )

    panel_rows: list[dict] = []

    for trade_date, selected in selected_by_day.items():
        day_date = pd.Timestamp(trade_date).date()
        day_start = pd.Timestamp(
            day_date, tz=ET
        ) + pd.Timedelta(hours=9, minutes=30)
        day_end = pd.Timestamp(
            day_date, tz=ET
        ) + pd.Timedelta(hours=16)

        day_quotes = raw_df[
            (raw_df["timestamp"] >= day_start)
            & (raw_df["timestamp"] <= day_end)
        ]
        if day_quotes.empty:
            continue

        base = pd.DataFrame(
            {
                "timestamp": pd.DatetimeIndex(
                    underlying.loc[
                        local_dates == trade_date, "timestamp"
                    ]
                )
            }
        )
        base = base[
            (base["timestamp"] >= day_start)
            & (base["timestamp"] <= day_end)
        ].sort_values("timestamp")
        if base.empty:
            continue

        for candidate_idx, symbol in selected.items():
            q = day_quotes[
                day_quotes["symbol"] == str(symbol)
            ][["timestamp", "bid", "ask"]].sort_values("timestamp")
            if q.empty:
                continue

            merged = pd.merge_asof(
                base,
                q,
                on="timestamp",
                direction="backward",
            )
            merged = merged.dropna(subset=["bid", "ask"])
            if merged.empty:
                continue

            definition = _effective_definition(
                definitions,
                symbol,
                pd.Timestamp(trade_date, tz=ET),
            )
            if definition is None:
                continue

            expiry = pd.to_datetime(
                definition["expiration"],
                utc=True,
                errors="coerce",
            )
            if pd.isna(expiry):
                continue
            expiry = expiry.tz_convert(ET)
            strike = float(definition["strike_price"])
            option_type = _option_type(
                definition.get("instrument_class", ""),
                symbol,
            )
            if option_type is None:
                continue

            for row in merged.itertuples(index=False):
                panel_rows.append(
                    {
                        "timestamp": pd.Timestamp(
                            row.timestamp
                        ).isoformat(),
                        "candidate_idx": int(candidate_idx),
                        "symbol": str(symbol),
                        "strike": strike,
                        "expiry": expiry.isoformat(),
                        "option_type": option_type,
                        "bid": float(row.bid),
                        "ask": float(row.ask),
                    }
                )

    panel = pd.DataFrame(panel_rows)
    if panel.empty:
        raise RuntimeError(
            "The cached OPRA batch was downloaded, but no selected "
            "historical quotes could be mapped to the hourly timeline."
        )

    panel = panel.drop_duplicates(
        ["timestamp", "candidate_idx"], keep="last"
    ).sort_values(["timestamp", "candidate_idx"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if str(output_path).endswith(".gz") else None
    panel.to_csv(
        output_path,
        index=False,
        compression=compression,
    )

    print(
        f"[cache] Saved processed panel: {output_path} | "
        f"{len(panel):,} rows | {panel['symbol'].nunique():,} contracts"
    )
    print(
        "[cache] Future runs with this panel use local files and do not "
        "request quote data from Databento."
    )
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
        help="Hard cumulative Databento historical quote-data ceiling in USD (default: $20).",
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
