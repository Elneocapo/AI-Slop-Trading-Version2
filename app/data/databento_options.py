from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import databento as db
import numpy as np
import pandas as pd

from app.training_v2 import DTE_DAYS, STRIKE_OFFSETS, load_hourly_data

ET = ZoneInfo("America/New_York")
DATASET = "OPRA.PILLAR"
# Databento bills historical usage in USD; the hard program limit is $20.
DEFAULT_MAX_COST_USD = 20.0
DEFAULT_CACHE_DIR = "data/databento_cache"


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


def _normalize_option_expiration(values: pd.Series) -> pd.Series:
    """Convert Databento's UTC-midnight expiration date to 16:00 ET."""
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    # Keep this vectorized: a full-year NVDA definition file can contain
    # hundreds of thousands of rows, making a Python loop unnecessarily slow.
    dates = parsed.dt.normalize()
    return (dates + pd.Timedelta(hours=16)).dt.tz_convert(ET)


def _prepare_definitions(definitions: pd.DataFrame) -> pd.DataFrame:
    """Normalize the large definition cache once, not once per trading day."""
    defs = definitions.copy()
    if "ts_event" in defs.columns:
        defs["ts_event"] = pd.to_datetime(
            defs["ts_event"], utc=True, errors="coerce"
        )
    defs["option_type"] = [
        _option_type(a, b)
        for a, b in zip(defs["instrument_class"], defs["raw_symbol"])
    ]
    defs["expiration"] = _normalize_option_expiration(defs["expiration"])
    defs["strike_price"] = pd.to_numeric(defs["strike_price"], errors="coerce")
    return defs[
        defs["option_type"].isin(["CALL", "PUT"])
        & defs["raw_symbol"].notna()
        & defs["expiration"].notna()
        & defs["strike_price"].notna()
    ].copy()


def _select_daily_contracts(
    definitions: pd.DataFrame,
    trade_date: pd.Timestamp,
    spot: float,
) -> dict[int, str]:
    day = trade_date.date()
    defs = definitions
    if "ts_event" in defs.columns:
        cutoff = trade_date.tz_convert("UTC")
        defs = defs[(defs["ts_event"].isna()) | (defs["ts_event"] <= cutoff)]
    if defs.empty:
        return {}

    dte = (defs["expiration"].dt.tz_convert(ET).dt.date - day).map(
        lambda value: value.days
    )
    defs = defs[(dte >= 1) & (dte <= max(DTE_DAYS) + 7)]
    if defs.empty:
        return {}

    selected: dict[int, str] = {}
    normalized_spot = max(float(spot), 1.0)
    for option_type_idx, option_type in enumerate(["CALL", "PUT"]):
        typed = defs[defs["option_type"] == option_type]
        if typed.empty:
            continue
        typed_dte = dte.loc[typed.index]
        for strike_idx, offset in enumerate(STRIKE_OFFSETS):
            target_strike = spot * (1.0 + offset)
            strike_distance = (
                typed["strike_price"] - target_strike
            ).abs() / normalized_spot
            for dte_idx, target_dte in enumerate(DTE_DAYS):
                score = strike_distance + (
                    typed_dte - target_dte
                ).abs() * 0.01
                best_index = score.nsmallest(1).index
                if len(best_index) == 0:
                    continue
                candidate_idx = (
                    option_type_idx * len(STRIKE_OFFSETS) * len(DTE_DAYS)
                    + strike_idx * len(DTE_DAYS)
                    + dte_idx
                )
                selected[candidate_idx] = str(
                    typed.loc[best_index[0], "raw_symbol"]
                )
    return selected


def _effective_definition(
    definitions: pd.DataFrame,
    raw_symbol: str,
    trade_date: pd.Timestamp,
) -> pd.Series | None:
    """Return the definition effective on the requested trading date."""
    meta = definitions[definitions["raw_symbol"].astype(str) == str(raw_symbol)]
    if meta.empty:
        return None
    if "ts_event" in meta.columns:
        cutoff = trade_date.tz_convert("UTC")
        meta = meta[meta["ts_event"].notna() & (meta["ts_event"] <= cutoff)]
        if meta.empty:
            return None
        return meta.sort_values("ts_event").iloc[-1]
    return meta.iloc[-1]


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
    force_rebuild: bool = False,
) -> Path:
    # Reuse the processed panel by default. A forced rebuild is useful after
    # changing preprocessing logic while keeping the downloaded OPRA batch.
    if output_path.exists() and output_path.stat().st_size > 0:
        if not force_rebuild:
            print(f"[cache] Using existing local panel: {output_path}")
            return output_path
        print(
            f"[cache] Rebuilding local panel from cached OPRA data: {output_path}"
        )

    api_key = api_key or os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DATABENTO_API_KEY is missing. In PowerShell use "
            "$env:DATABENTO_API_KEY='YOUR_KEY'."
        )
    if max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be greater than 0.")

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

    definition_cache = cache_dir / (
        f"{ticker.lower()}_{period}_definitions.csv"
    )
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
        # A full-year OPRA definition stream can be very large and may appear
        # stalled for a long time. Download it in bounded chunks so progress
        # is visible and each request is smaller.
        chunks: list[pd.DataFrame] = []
        chunk_start = definition_start
        total_span_days = max(
            1,
            int((definition_end - definition_start).total_seconds() // 86400),
        )
        chunk_number = 0
        while chunk_start < definition_end:
            chunk_end = min(
                chunk_start + pd.Timedelta(days=30),
                definition_end,
            )
            chunk_number += 1
            elapsed_days = max(
                0,
                int((chunk_start - definition_start).total_seconds() // 86400),
            )
            print(
                f"[definitions] chunk {chunk_number} "
                f"~{elapsed_days}/{total_span_days} days "
                f"({chunk_start.date()} -> {chunk_end.date()})"
            )
            frame = (
                client.timeseries.get_range(
                    dataset=DATASET,
                    schema="definition",
                    stype_in="parent",
                    symbols=f"{ticker}.OPT",
                    start=chunk_start,
                    end=chunk_end,
                )
                .to_df()
                .reset_index()
            )
            if not frame.empty:
                chunks.append(frame)
                print(
                    f"[definitions] chunk {chunk_number} received "
                    f"{len(frame):,} rows"
                )
            chunk_start = chunk_end

        if not chunks:
            raise RuntimeError(
                f"No OPRA option definitions returned for {ticker}."
            )
        definitions = pd.concat(chunks, ignore_index=True).drop_duplicates()
        definitions.to_csv(definition_cache, index=False)
        print(
            f"[definitions] Saved {len(definitions):,} definition rows "
            f"to {definition_cache}"
        )

    definitions = _prepare_definitions(definitions)
    print(f"[definitions] Prepared {len(definitions):,} usable definition rows")

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

            # Pandas merge_asof requires identical datetime64 precision.
            # Yahoo-derived hourly timestamps can be second-resolution while
            # Databento quote timestamps are typically nanosecond-resolution.
            merge_base = base.copy()
            merge_quotes = q.copy()
            merge_base["timestamp"] = pd.to_datetime(
                merge_base["timestamp"], utc=True, errors="coerce"
            )
            merge_quotes["timestamp"] = pd.to_datetime(
                merge_quotes["timestamp"], utc=True, errors="coerce"
            )
            merge_base["timestamp"] = merge_base["timestamp"].astype(
                "datetime64[ns, UTC]"
            )
            merge_quotes["timestamp"] = merge_quotes["timestamp"].astype(
                "datetime64[ns, UTC]"
            )

            merged = pd.merge_asof(
                merge_base,
                merge_quotes,
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

            expiry = _normalize_option_expiration(
                pd.Series([definition["expiration"]], index=[0])
            ).iloc[0]
            if pd.isna(expiry):
                continue
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
                        ).tz_convert(ET).isoformat(),
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
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help=(
            "Rebuild the processed panel from the existing local Databento "
            "batch without submitting a new data request."
        ),
    )
    args = parser.parse_args()
    path = build_real_options_panel(
        args.ticker.upper(),
        args.period,
        Path(args.output),
        max_cost_usd=args.max_cost_usd,
        force_rebuild=args.force_rebuild,
    )
    print(f"Saved real options panel: {path}")


if __name__ == "__main__":
    main()