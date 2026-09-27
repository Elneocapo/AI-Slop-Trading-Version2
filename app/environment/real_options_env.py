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
        panel["candidate_idx"] = panel["candidate_idx"].astype(int)
        panel["symbol"] = panel["symbol"].astype(str)
        panel = panel.drop_duplicates(["timestamp_key", "candidate_idx"], keep="last")

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
                "option_type": CALL if str(row.option_type).upper() == "CALL" else 1,
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

        self.real_option_mode = True

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
        decision_t = max(int(t) - 1, 0)
        timestamp_key = self._timestamp_key(self.data.loc[decision_t, "timestamp"])
        per_type = len(STRIKE_OFFSETS) * len(DTE_DAYS)
        candidate_idx = (
            int(option_type) * per_type
            + int(strike_idx) * len(DTE_DAYS)
            + int(dte_idx)
        )
        exact = self._candidate_quotes.get((timestamp_key, candidate_idx))
        if exact is not None:
            return exact

        # Yahoo hourly bars and OPRA resampled quotes can differ slightly in
        # timestamp. Use only the latest quote at or before the decision bar;
        # never use a future quote.
        times = self._candidate_times.get(candidate_idx, [])
        if not times:
            return None
        pos = bisect_right(times, timestamp_key) - 1
        if pos < 0:
            return None
        quote_key = times[pos]
        # Do not carry a stale quote across more than one trading hour.
        if timestamp_key - quote_key > 60 * 60 * 1_000_000_000:
            quote_key = None
        else:
            candidate = self._candidate_quotes.get((quote_key, candidate_idx))
            if candidate is not None:
                return candidate

        # The processed panel assigns candidate_idx using the contract selected
        # at the start of each trading day. Some historical quote streams can
        # lose that exact mapping after timestamp normalization/resampling.
        # Fall back to the real quote closest to the requested strike/DTE at the
        # latest available quote timestamp <= decision_t. This remains strictly
        # causal: no future quote can enter the candidate selection.
        all_times = sorted(self._timestamp_candidates)
        pos = bisect_right(all_times, timestamp_key) - 1
        if pos < 0:
            return None
        fallback_key = all_times[pos]
        if timestamp_key - fallback_key > 60 * 60 * 1_000_000_000:
            return None

        decision_t = max(int(t) - 1, 0)
        spot = float(self.data.loc[decision_t, "Close"])
        target_strike = self._strike_from_offset(
            spot, STRIKE_OFFSETS[int(strike_idx)]
        )
        target_dte = int(DTE_DAYS[int(dte_idx)])
        decision_date = pd.Timestamp(
            self.data.loc[decision_t, "timestamp"]
        ).date()

        best = None
        best_score = float("inf")
        for record in self._timestamp_candidates.get(fallback_key, []):
            if int(record.get("option_type", -1)) != int(option_type):
                continue
            if float(record.get("ask", 0.0)) <= 0:
                continue
            expiry_t = record.get("expiry_t")
            if expiry_t is None:
                continue
            expiry_ts = record.get("expiry_ts")
            try:
                expiry_date = pd.Timestamp(expiry_ts).date()
            except Exception:
                continue
            dte = max((expiry_date - decision_date).days, 0)
            strike = float(record.get("strike", 0.0))
            score = (
                abs(strike - target_strike) / max(abs(spot), 1.0)
                + abs(dte - target_dte) * 0.01
            )
            if score < best_score:
                best_score = score
                best = record
        return best

    def _position_bid_ask(self, t: int) -> tuple[float, float]:
        if self.position is None or self.position.symbol is None:
            return 0.0, 0.0
        timestamps = pd.DatetimeIndex(self.data["timestamp"])
        start = min(max(int(t), 0), len(timestamps) - 1)
        for idx in range(start, max(start - 40, -1), -1):
            key = self._timestamp_key(timestamps[idx])
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
