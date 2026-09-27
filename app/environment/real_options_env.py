from __future__ import annotations

from bisect import bisect_right
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.environment.options_env import (
    CALL,
    CLOSE,
    DTE_DAYS,
    OPEN_LONG,
    OPEN_SHORT,
    Position,
    STRIKE_OFFSETS,
    OptionsTradingEnv,
)


class RealOptionsTradingEnv(OptionsTradingEnv):
    """OptionsTradingEnv backed by historical OPRA NBBO quotes.

    The underlying and feature columns remain the same as the base environment.
    Candidate option premiums and position marks come from the supplied
    Databento-derived panel rather than Black-Scholes prices.
    """

    def __init__(self, data, option_panel: pd.DataFrame, **kwargs):
        super().__init__(data, **kwargs)
        required = {"timestamp", "candidate_idx", "symbol", "strike", "expiry", "option_type", "bid", "ask"}
        missing = required - set(option_panel.columns)
        if missing:
            raise ValueError(f"Real option panel missing columns: {sorted(missing)}")

        panel = option_panel.copy()
        panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce")
        panel["expiry"] = pd.to_datetime(panel["expiry"], errors="coerce")
        panel["candidate_idx"] = pd.to_numeric(panel["candidate_idx"], errors="coerce").astype("Int64")
        panel["bid"] = pd.to_numeric(panel["bid"], errors="coerce")
        panel["ask"] = pd.to_numeric(panel["ask"], errors="coerce")
        panel["strike"] = pd.to_numeric(panel["strike"], errors="coerce")
        panel = panel.dropna(subset=["timestamp", "candidate_idx", "symbol", "expiry", "strike", "bid", "ask"])
        panel = panel[(panel["bid"] >= 0) & (panel["ask"] > 0) & (panel["ask"] >= panel["bid"])]
        # Use an absolute UTC nanosecond key instead of isoformat(). This keeps
        # quote lookup stable when one side is represented as ET and the other
        # as UTC (or uses a different but equivalent timezone offset).
        panel["timestamp_key"] = panel["timestamp"].map(self._timestamp_key)

        # Normalize legacy candidate_idx values from the actual quote contract.
        # This is preprocessing only and never changes the causal time boundary.
        underlying = self.data[["timestamp", "Close"]].copy()
        underlying["timestamp"] = pd.to_datetime(underlying["timestamp"], errors="coerce")
        underlying = underlying.dropna(subset=["timestamp", "Close"]).sort_values("timestamp")

        # pandas may retain different datetime resolutions (s/us/ns) after
        # reading the Yahoo and OPRA-derived sources. Use one integer UTC key
        # for the preprocessing join so merge_asof sees identical dtypes.
        panel_for_alignment = panel.sort_values("timestamp").copy()
        panel_for_alignment["_timestamp_align_ns"] = panel_for_alignment["timestamp"].map(
            self._timestamp_key
        )
        underlying["_timestamp_align_ns"] = underlying["timestamp"].map(
            self._timestamp_key
        )
        underlying = underlying.sort_values("_timestamp_align_ns")

        aligned = pd.merge_asof(
            panel_for_alignment,
            underlying[["_timestamp_align_ns", "Close"]],
            on="_timestamp_align_ns",
            direction="backward",
        )
        aligned = aligned.drop(columns=["_timestamp_align_ns"])
        spot_series = pd.to_numeric(aligned["Close"], errors="coerce")
        valid = spot_series.notna() & (spot_series > 0)
        aligned = aligned.loc[valid].copy()
        spot = spot_series.loc[valid].to_numpy(dtype=float)

        rel = aligned["strike"].to_numpy(dtype=float) / spot - 1.0
        strike_dist = np.abs(
            rel[:, None] - np.asarray(STRIKE_OFFSETS, dtype=float)[None, :]
        )
        strike_idx = strike_dist.argmin(axis=1)

        dte = (
            aligned["expiry"].dt.normalize()
            - aligned["timestamp"].dt.normalize()
        ).dt.days.to_numpy(dtype=float)
        dte_dist = np.abs(
            dte[:, None] - np.asarray(DTE_DAYS, dtype=float)[None, :]
        )
        dte_idx = dte_dist.argmin(axis=1)

        type_idx = np.where(
            aligned["option_type"].astype(str).str.upper().isin(["CALL", "C", "0"]),
            0,
            1,
        )
        aligned["candidate_idx"] = (
            type_idx * len(STRIKE_OFFSETS) * len(DTE_DAYS)
            + strike_idx * len(DTE_DAYS)
            + dte_idx
        ).astype(int)
        aligned["canonical_distance"] = (
            strike_dist[np.arange(len(aligned)), strike_idx]
            + 0.01 * dte_dist[np.arange(len(aligned)), dte_idx]
        )

        panel = (
            aligned.sort_values(["timestamp_key", "canonical_distance"])
            .drop_duplicates(["timestamp_key", "candidate_idx"], keep="first")
        )
        panel["symbol"] = panel["symbol"].astype(str)

        self.option_panel = panel
        self._expiry_cache: dict[str, int | None] = {}
        self._candidate_quotes: dict[tuple[int, int], dict[str, Any]] = {}
        self._candidate_times: dict[int, list[int]] = {}
        self._timestamp_candidates: dict[int, list[dict[str, Any]]] = {}
        self._symbol_quotes: dict[tuple[int, str], tuple[float, float]] = {}
        for row in panel.itertuples(index=False):
            expiry_t = self._find_expiry_index(row.expiry)
            record = {
                "symbol": str(row.symbol),
                "strike": float(row.strike),
                "expiry_t": expiry_t,
                "expiry_ts": row.expiry.isoformat(),
                "bid": float(row.bid),
                "ask": float(row.ask),
                "mid": (float(row.bid) + float(row.ask)) / 2.0,
                "option_type": (CALL if str(row.option_type).upper() in {"CALL", "C", "0"} else 1),
            }
            candidate_idx = int(row.candidate_idx)
            timestamp_key = int(row.timestamp_key)
            record["candidate_idx"] = candidate_idx
            self._candidate_quotes[(timestamp_key, candidate_idx)] = record
            self._candidate_times.setdefault(candidate_idx, []).append(timestamp_key)
            self._timestamp_candidates.setdefault(timestamp_key, []).append(record)
            self._symbol_quotes[(timestamp_key, str(row.symbol))] = (float(row.bid), float(row.ask))

        for candidate_idx in self._candidate_times:
            self._candidate_times[candidate_idx] = sorted(set(self._candidate_times[candidate_idx]))
        self._all_quote_times = sorted(self._timestamp_candidates)
        self._data_timestamp_keys = [self._timestamp_key(v) for v in self.data["timestamp"]]
        self._candidate_lookup_cache: dict[tuple[int, int, int, int], dict[str, Any] | None] = {}

        self.real_option_mode = True

    def reset(self, *, seed=None, options=None):
        """Reset on a causally covered bar without scanning option candidates."""
        super().reset(seed=seed)
        if not self.real_option_mode:
            return self._observation(), {}

        initial_t = int(self.t)
        max_t = min(
            len(self.data) - self.episode_hours - 1,
            initial_t + 10 * 24 * 7,
        )
        if not self._all_quote_times:
            raise ValueError("Real options panel contains no usable quote timestamps.")

        quote_pos = bisect_right(
            self._all_quote_times,
            self._data_timestamp_keys[initial_t - 1],
        ) - 1
        if quote_pos < 0:
            quote_pos = 0

        selected_t = None
        pos = initial_t
        while pos <= max_t:
            decision_key = self._data_timestamp_keys[pos - 1]
            while (
                quote_pos + 1 < len(self._all_quote_times)
                and self._all_quote_times[quote_pos + 1] <= decision_key
            ):
                quote_pos += 1
            quote_key = self._all_quote_times[quote_pos]
            age_ns = decision_key - quote_key
            if (
                self.is_regular_session(pos - 1)
                and 0 <= age_ns <= 60 * 60 * 1_000_000_000
            ):
                selected_t = pos
                break
            pos += 1

        if selected_t is None:
            raise ValueError(
                "Real options panel has no causally available entry quote inside "
                "the current episode window."
            )

        self.t = int(selected_t)
        self.end_t = self.t + self.episode_hours
        self._candidate_observation_cache.clear()
        self._candidate_lookup_cache.clear()
        return self._observation(), {}

    @staticmethod
    def _timestamp_key(value) -> int:
        """Normalize timestamps to an absolute UTC key for quote lookup."""
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("America/New_York")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.value)

    def _find_expiry_index(self, expiry: pd.Timestamp) -> int | None:
        key = expiry.date().isoformat()
        if key in getattr(self, "_expiry_cache", {}):
            return self._expiry_cache[key]
        dates = pd.DatetimeIndex(self.data["timestamp"])
        local_dates = dates.date
        matches = np.flatnonzero(local_dates == expiry.date())
        value = int(matches[-1]) if len(matches) else None
        self._expiry_cache[key] = value
        return value

    def _get_candidate_contract(self, t: int, option_type: int, strike_idx: int, dte_idx: int) -> dict | None:
        """Return the executable real quote using only information available at t-1."""
        decision_t = max(int(t) - 1, 0)
        cache_key = (decision_t, int(option_type), int(strike_idx), int(dte_idx))
        if cache_key in self._candidate_lookup_cache:
            return self._candidate_lookup_cache[cache_key]

        timestamp_key = self._data_timestamp_keys[decision_t]
        per_type = len(STRIKE_OFFSETS) * len(DTE_DAYS)
        candidate_idx = (
            int(option_type) * per_type
            + int(strike_idx) * len(DTE_DAYS)
            + int(dte_idx)
        )

        exact = self._candidate_quotes.get((timestamp_key, candidate_idx))
        if exact is not None:
            self._candidate_lookup_cache[cache_key] = exact
            return exact

        times = self._candidate_times.get(candidate_idx, [])
        if times:
            pos = bisect_right(times, timestamp_key) - 1
            if pos >= 0:
                quote_key = times[pos]
                if timestamp_key - quote_key <= 60 * 60 * 1_000_000_000:
                    candidate = self._candidate_quotes.get((quote_key, candidate_idx))
                    if candidate is not None:
                        self._candidate_lookup_cache[cache_key] = candidate
                        return candidate

        self._candidate_lookup_cache[cache_key] = None
        return None

    def _position_bid_ask(self, t: int) -> tuple[float, float]:
        if self.position is None or self.position.symbol is None:
            return 0.0, 0.0
        start = min(max(int(t), 0), len(self._data_timestamp_keys) - 1)
        for idx in range(start, max(start - 40, -1), -1):
            key = self._data_timestamp_keys[idx]
            quote = self._symbol_quotes.get((key, self.position.symbol))
            if quote is not None:
                return quote
        return 0.0, 0.0

    def _open(self, operation: int, option_type: int, strike_idx: int, dte_idx: int, size_idx: int):
        if self.position is not None:
            return
        candidate = self._get_candidate_contract(self.t, option_type, strike_idx, dte_idx)
        if candidate is None or candidate["ask"] <= 0:
            return
        from app.environment.options_env import CONTRACT_SIZES
        contracts = int(CONTRACT_SIZES[int(size_idx)])
        execution_price = candidate["ask"] * (1.0 + self.slippage)
        total = execution_price * self.multiplier * contracts + self.transaction_cost
        if operation == OPEN_LONG:
            if total > self.cash or candidate["expiry_t"] is None:
                return
            self.cash -= total
            self.total_transaction_costs += self.transaction_cost
            self.position = Position(
                1 if int(option_type) == CALL else -1,
                candidate["strike"],
                candidate["expiry_t"],
                execution_price,
                contracts,
                entry_t=self.t,
                symbol=candidate["symbol"],
                expiry_ts=candidate["expiry_ts"],
            )
            self.position.entry_spot = float(self.data.loc[max(self.t - 1, 0), "Close"])
        elif operation == OPEN_SHORT:
            return

    def _mark(self, t: int) -> float:
        bid, ask = self._position_bid_ask(t)
        if bid <= 0 and ask <= 0:
            return 0.0
        return (bid + ask) / 2.0

    def _equity(self, t: int) -> float:
        if self.position is None:
            return self.cash
        bid, ask = self._position_bid_ask(t)
        if self.position.kind in (1, -1):
            return self.cash + bid * self.multiplier * self.position.contracts
        return self.cash + self.position.collateral - ask * self.multiplier * self.position.contracts

    def _close(self, mark_t: int | None = None, reason: str = "close"):
        if self.position is None:
            return
        position = self.position
        # Use the requested mark time for forced risk-stop liquidation. For
        # normal policy closes, t-1 remains the last observed bar and avoids
        # introducing look-ahead into the execution price.
        if reason == "risk_stop":
            decision_t = self.t if mark_t is None else int(mark_t)
        else:
            decision_t = max(self.t - 1, 0) if mark_t is None else int(mark_t)
        decision_t = min(max(decision_t, 0), len(self.data) - 1)
        bid, ask = self._position_bid_ask(decision_t)
        if bid <= 0 and ask <= 0:
            return
        if position.kind in (1, -1):
            execution_price = bid * max(1.0 - self.slippage, 0.0)
            proceeds = execution_price * self.multiplier * position.contracts
            self.cash += proceeds - self.transaction_cost
            self.total_transaction_costs += self.transaction_cost
            pnl = (execution_price - position.entry_price) * self.multiplier * position.contracts - (2.0 * self.transaction_cost)
        else:
            execution_price = ask * (1.0 + self.slippage)
            buyback = execution_price * self.multiplier * position.contracts + self.transaction_cost
            self.cash += position.collateral - buyback
            self.total_transaction_costs += self.transaction_cost
            pnl = (position.entry_price - execution_price) * self.multiplier * position.contracts - (2.0 * self.transaction_cost)

        self.trade_log.append({
            "entry_t": position.entry_t,
            "exit_t": decision_t,
            "kind": position.kind,
            "strike": position.strike,
            "contracts": position.contracts,
            "entry_price": position.entry_price,
            "exit_price": execution_price,
            "pnl": pnl,
            "reason": reason,
            "transaction_costs": 2.0 * self.transaction_cost,
            "symbol": position.symbol,
            "entry_spot": float(getattr(position, "entry_spot", np.nan)),
            "entry_moneyness": (
                float(position.strike) / float(getattr(position, "entry_spot", np.nan)) - 1.0
                if float(getattr(position, "entry_spot", np.nan)) > 0 else np.nan
            ),
        })
        self.position = None

    def is_regular_session(self, t: int) -> bool:
        ts = pd.Timestamp(self.data.loc[int(t), "timestamp"])
        minutes = ts.hour * 60 + ts.minute
        return 570 <= minutes <= 960

    def has_entry_quotes(self, t: int) -> bool:
        decision_t = max(int(t) - 1, 0)
        timestamp_key = self._timestamp_key(self.data.loc[decision_t, "timestamp"])
        return any(key[0] == timestamp_key and value["ask"] > 0 for key, value in self._candidate_quotes.items())
