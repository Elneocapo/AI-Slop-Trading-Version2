    def _get_candidate_contract(self, t: int, option_type: int, strike_idx: int, dte_idx: int) -> dict | None:
        """Return a real quote using only information available at t-1."""
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

        # The panel is canonicalized during initialization, so this path is
        # only a safety net for small timestamp gaps. It never scans history.
        result = None
        if times:
            pos = bisect_right(times, timestamp_key) - 1
            if pos >= 0:
                quote_key = times[pos]
                if timestamp_key - quote_key <= 60 * 60 * 1_000_000_000:
                    result = self._candidate_quotes.get((quote_key, candidate_idx))

        self._candidate_lookup_cache[cache_key] = result
        return result

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
