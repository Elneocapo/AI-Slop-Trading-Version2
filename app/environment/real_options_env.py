from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.environment.options_env import CALL, CLOSE, OPEN_LONG, OPEN_SHORT, Position, OptionsTradingEnv


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
        panel["timestamp_key"] = panel["timestamp"].map(lambda x: x.isoformat())
        panel["candidate_idx"] = panel["candidate_idx"].astype(int)
        panel["symbol"] = panel["symbol"].astype(str)
        panel = panel.drop_duplicates(["timestamp_key", "candidate_idx"], keep="last")

        self.option_panel = panel
        self._candidate_quotes: dict[tuple[str, int], dict[str, Any]] = {}
        self._symbol_quotes: dict[tuple[str, str], tuple[float, float]] = {}
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
            self._candidate_quotes[(row.timestamp_key, int(row.candidate_idx))] = record
            self._symbol_quotes[(row.timestamp_key, str(row.symbol))] = (float(row.bid), float(row.ask))

        self.real_option_mode = True
        self._expiry_cache: dict[str, int | None] = {}

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
        timestamp = pd.Timestamp(self.data.loc[decision_t, "timestamp"]).isoformat()
        candidate_idx = int(option_type) * 54 + int(strike_idx) * 6 + int(dte_idx)
        return self._candidate_quotes.get((timestamp, candidate_idx))

    def _position_bid_ask(self, t: int) -> tuple[float, float]:
        if self.position is None or self.position.symbol is None:
            return 0.0, 0.0
        timestamps = pd.DatetimeIndex(self.data["timestamp"])
        start = min(max(int(t), 0), len(timestamps) - 1)
        for idx in range(start, max(start - 40, -1), -1):
            key = pd.Timestamp(timestamps[idx]).isoformat()
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
        contracts = int(self.CONTRACT_SIZES[int(size_idx)]) if hasattr(self, "CONTRACT_SIZES") else None
        if contracts is None:
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

    def _close(self):
        if self.position is None:
            return
        position = self.position
        decision_t = max(self.t - 1, 0)
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
            "exit_t": self.t,
            "kind": position.kind,
            "strike": position.strike,
            "contracts": position.contracts,
            "entry_price": position.entry_price,
            "exit_price": execution_price,
            "pnl": pnl,
            "reason": "close",
            "transaction_costs": 2.0 * self.transaction_cost,
            "symbol": position.symbol,
        })
        self.position = None

    def is_regular_session(self, t: int) -> bool:
        ts = pd.Timestamp(self.data.loc[int(t), "timestamp"])
        minutes = ts.hour * 60 + ts.minute
        return 570 <= minutes <= 960

    def has_entry_quotes(self, t: int) -> bool:
        decision_t = max(int(t) - 1, 0)
        timestamp = pd.Timestamp(self.data.loc[decision_t, "timestamp"]).isoformat()
        return any(key[0] == timestamp and value["ask"] > 0 for key, value in self._candidate_quotes.items())
