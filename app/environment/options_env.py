from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt

import gymnasium as gym
import numpy as np
from gymnasium import spaces


HOLD = 0
OPEN_LONG = 1
OPEN_SHORT = 2
CLOSE = 3

CALL = 0
PUT = 1

STRIKE_OFFSETS = (-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10)
DTE_DAYS = (1, 3, 5, 7, 14, 30)
CONTRACT_SIZES = (1, 2, 3, 5, 10)


@dataclass
class Position:
    kind: int
    strike: float
    expiry_t: int
    entry_price: float
    contracts: int
    collateral: float = 0.0
    entry_t: int = 0


class OptionsTradingEnv(gym.Env):
    """Hourly options environment with broker-style contract selection.

    The underlying candles are historical market data. Option quotes are still
    synthetic Black-Scholes marks, so this is not a live options market.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data,
        initial_cash: float = 500.0,
        lookback: int = 60,
        episode_hours: int = 145 * 7,
        fixed_start: int | None = None,
        transaction_cost: float = 0.75,
        slippage: float = 0.0025,
        contract_multiplier: int = 100,
    ):
        super().__init__()
        self.data = data.reset_index(drop=True).copy()
        self.initial_cash = float(initial_cash)
        self.lookback = int(lookback)
        self.episode_hours = int(episode_hours)
        self.fixed_start = fixed_start
        self.transaction_cost = float(transaction_cost)
        self.slippage = float(slippage)
        self.multiplier = int(contract_multiplier)

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.prices = self.data["Close"].astype(float).to_numpy()
        self.opens = self.data["Open"].astype(float).to_numpy()
        self.highs = self.data["High"].astype(float).to_numpy()
        self.lows = self.data["Low"].astype(float).to_numpy()
        self.volumes = self.data["Volume"].astype(float).to_numpy()

        feature_cols = [
            "return_1h", "return_6h", "return_24h", "sma24_gap", "sma72_gap",
            "volatility_24h", "volume_z", "volume_ratio_24h", "rsi_14", "atr_pct",
            "time_sin", "time_cos", "weekday_sin", "weekday_cos", "is_regular_session",
            "minutes_since_open", "minutes_to_close", "near_open", "near_close",
            "pre_market", "after_hours", "session_return", "session_high_gap",
            "session_low_gap", "session_range_position", "range_24h_position",
            "high_24h_gap", "low_24h_gap", "bar_return", "bar_range_pct",
            "gap_from_prev_close",
        ]
        self.feature_cols = [c for c in feature_cols if c in self.data.columns]
        self.features = (
            self.data[self.feature_cols]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy()
        )
        self.n_features = len(self.feature_cols) + 1

        self.candidate_count = 2 * len(STRIKE_OFFSETS) * len(DTE_DAYS)
        self.context_size = 8 + 8 + 5 + self.candidate_count * 5
        self.action_space = spaces.MultiDiscrete(
            [4, 2, len(STRIKE_OFFSETS), len(DTE_DAYS), len(CONTRACT_SIZES)]
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback * self.n_features + self.context_size,),
            dtype=np.float32,
        )

        self.t = self.lookback
        self.end_t = self.lookback + self.episode_hours
        self.cash = self.initial_cash
        self.position: Position | None = None
        self.equity = self.initial_cash
        self.previous_equity = self.initial_cash
        self.peak_equity = self.initial_cash
        self.trade_log: list[dict] = []
        self.total_transaction_costs = 0.0

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return exp(-0.5 * x * x) / sqrt(2.0 * np.pi)

    @staticmethod
    def _round_option_price(price: float) -> float:
        """Round a synthetic equity-option premium to a valid penny quote."""
        if not np.isfinite(price):
            return 0.01
        return max(round(float(price) / 0.01) * 0.01, 0.01)

    @staticmethod
    def _strike_from_offset(spot: float, offset: float) -> float:
        """Map relative strike requests onto a realistic $2.50 strike grid."""
        raw = max(float(spot) * (1.0 + float(offset)), 0.01)
        return max(round(raw / 2.5) * 2.5, 2.5)

    @staticmethod
    def _option_bid_ask(mid: float) -> tuple[float, float]:
        """Create a conservative synthetic bid/ask around the BS mid price."""
        mid = max(float(mid), 0.0)
        if mid <= 0.0:
            return 0.0, 0.0
        # Synthetic full spread: at least $0.02, otherwise 5% of premium.
        # Round both sides to the $0.01 option tick.
        spread = max(0.02, mid * 0.05)
        bid = max(np.floor((mid - spread / 2.0) * 100.0) / 100.0, 0.0)
        ask = max(np.ceil((mid + spread / 2.0) * 100.0) / 100.0, 0.01)
        return float(bid), float(ask)

    def _option_price(self, spot: float, strike: float, tau_hours: float, vol: float, call: bool) -> float:
        if tau_hours <= 0:
            intrinsic = max(spot - strike, 0.0) if call else max(strike - spot, 0.0)
            return self._round_option_price(intrinsic) if intrinsic > 0 else 0.0
        # Episodes use roughly 7 hourly trading bars per regular-session day.
        # Convert bar-hours to trading years consistently with DTE/expiry_t.
        tau = tau_hours / (7.0 * 252.0)
        vol = max(float(vol), 0.15)
        d1 = (log(max(spot, 1e-9) / max(strike, 1e-9)) + 0.5 * vol * vol * tau) / (vol * sqrt(tau))
        d2 = d1 - vol * sqrt(tau)
        nd1 = self._norm_cdf(d1)
        nd2 = self._norm_cdf(d2)
        theoretical = spot * nd1 - strike * nd2 if call else strike * (1.0 - nd2) - spot * (1.0 - nd1)
        return self._round_option_price(theoretical)

    def _option_greeks(self, spot: float, strike: float, tau_hours: float, vol: float, call: bool) -> tuple[float, float, float, float, float]:
        if tau_hours <= 0:
            intrinsic = max(spot - strike, 0.0) if call else max(strike - spot, 0.0)
            delta = 1.0 if call and spot > strike else -1.0 if (not call and spot < strike) else 0.0
            return intrinsic, delta, 0.0, 0.0, 0.0
        # Episodes use roughly 7 hourly trading bars per regular-session day.
        # Convert bar-hours to trading years consistently with DTE/expiry_t.
        tau = tau_hours / (7.0 * 252.0)
        vol = max(float(vol), 0.15)
        sqrt_tau = sqrt(tau)
        d1 = (log(max(spot, 1e-9) / max(strike, 1e-9)) + 0.5 * vol * vol * tau) / (vol * sqrt_tau)
        d2 = d1 - vol * sqrt_tau
        price = self._option_price(spot, strike, tau_hours, vol, call)
        pdf = self._norm_pdf(d1)
        delta = self._norm_cdf(d1) if call else self._norm_cdf(d1) - 1.0
        gamma = pdf / (max(spot, 1e-9) * vol * sqrt_tau)
        theta_hour = (-(spot * pdf * vol) / (2.0 * sqrt_tau)) / (7.0 * 252.0)
        vega = spot * pdf * sqrt_tau
        return price, delta, gamma, theta_hour, vega

    def _vol(self, t: int) -> float:
        if "volatility_24h" in self.data.columns:
            v = float(self.data.loc[t, "volatility_24h"])
            if np.isfinite(v) and v > 0:
                return v
        return 0.15

    def _mark(self, t: int) -> float:
        if self.position is None:
            return 0.0
        spot = float(self.prices[t])
        tau = max(self.position.expiry_t - t, 0)
        call = self.position.kind in (1, 2)
        return self._option_price(spot, self.position.strike, tau, self._vol(t), call)

    def _open(self, operation: int, option_type: int, strike_idx: int, dte_idx: int, size_idx: int):
        if self.position is not None:
            return
        # The observation contains data only through t-1. Use that last
        # observed bar for entries instead of the unseen bar at t.
        decision_t = max(self.t - 1, 0)
        spot = float(self.prices[decision_t])
        offset = STRIKE_OFFSETS[int(strike_idx)]
        strike = self._strike_from_offset(spot, offset)
        dte_days = DTE_DAYS[int(dte_idx)]
        contracts = CONTRACT_SIZES[int(size_idx)]
        call = int(option_type) == CALL
        expiry_t = min(decision_t + dte_days * 7, self.end_t)
        theoretical = self._option_price(
            spot, strike, expiry_t - decision_t, self._vol(decision_t), call
        )
        cost = self.transaction_cost

        if operation == OPEN_LONG:
            _, ask = self._option_bid_ask(theoretical)
            execution_price = ask * (1.0 + self.slippage)
            total = execution_price * self.multiplier * contracts + cost
            if total > self.cash:
                return
            self.cash -= total
            self.total_transaction_costs += cost
            kind = 1 if call else -1
            self.position = Position(kind, strike, expiry_t, execution_price, contracts, entry_t=self.t)
        elif operation == OPEN_SHORT:
            bid, _ = self._option_bid_ask(theoretical)
            execution_price = bid * max(1.0 - self.slippage, 0.0)
            collateral = spot * self.multiplier * contracts * 0.50
            premium = execution_price * self.multiplier * contracts
            net_cash_needed = collateral - premium + cost
            if net_cash_needed > self.cash:
                return
            self.cash -= net_cash_needed
            self.total_transaction_costs += cost
            kind = 2 if call else -2
            self.position = Position(kind, strike, expiry_t, execution_price, contracts, collateral, self.t)

    def _close(self):
        if self.position is None:
            return
        position = self.position
        decision_t = max(self.t - 1, 0)
        mark = self._mark(decision_t)
        value = mark * self.multiplier * position.contracts
        if position.kind in (1, -1):
            bid, _ = self._option_bid_ask(mark)
            execution_price = bid * max(1.0 - self.slippage, 0.0)
            proceeds = execution_price * self.multiplier * position.contracts
            self.cash += proceeds - self.transaction_cost
            self.total_transaction_costs += self.transaction_cost
            pnl = (execution_price - position.entry_price) * self.multiplier * position.contracts - (2.0 * self.transaction_cost)
        else:
            _, ask = self._option_bid_ask(mark)
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
        })
        self.position = None

    def _settle_expiry(self):
        if self.position is None or self.t < self.position.expiry_t:
            return
        position = self.position
        spot = float(self.prices[self.t])
        call = position.kind in (1, 2)
        intrinsic = max(spot - position.strike, 0.0) if call else max(position.strike - spot, 0.0)
        value = intrinsic * self.multiplier * position.contracts
        if position.kind in (1, -1):
            self.cash += value
            pnl = (intrinsic - position.entry_price) * self.multiplier * position.contracts - self.transaction_cost
        else:
            self.cash += position.collateral - value
            pnl = (position.entry_price - intrinsic) * self.multiplier * position.contracts - self.transaction_cost

        self.trade_log.append({
            "entry_t": position.entry_t,
            "exit_t": self.t,
            "kind": position.kind,
            "strike": position.strike,
            "contracts": position.contracts,
            "entry_price": position.entry_price,
            "exit_price": intrinsic,
            "pnl": pnl,
            "reason": "expiry",
            "transaction_costs": self.transaction_cost,
        })
        self.position = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        max_start = len(self.prices) - self.episode_hours - 1
        if max_start < self.lookback:
            raise ValueError(f"Need at least {self.lookback + self.episode_hours + 1} hourly candles")
        if self.fixed_start is not None:
            if self.fixed_start < self.lookback or self.fixed_start + self.episode_hours >= len(self.prices):
                raise ValueError("fixed_start does not leave enough hourly data for the episode")
            self.t = int(self.fixed_start)
        else:
            self.t = int(self.np_random.integers(self.lookback, max_start + 1))
        self.end_t = self.t + self.episode_hours
        self.cash = self.initial_cash
        self.position = None
        self.equity = self.initial_cash
        self.previous_equity = self.initial_cash
        self.peak_equity = self.initial_cash
        self.trade_log = []
        self.total_transaction_costs = 0.0
        return self._observation(), {}

    def _observation(self):
        start = self.t - self.lookback
        market = []
        for i in range(start, self.t):
            close = self.prices[i]
            market.extend(self.features[i].tolist())
            market.append(float(self.highs[i] - self.lows[i]) / max(close, 1e-9))

        decision_t = max(self.t - 1, 0)
        equity = self._equity(decision_t)
        drawdown = max(0.0, (self.peak_equity - equity) / max(self.peak_equity, 1e-9))
        position_flag = 0.0 if self.position is None else float(self.position.kind)
        position_pnl = 0.0
        if self.position is not None:
            direction = 1.0 if self.position.kind in (1, -1) else -1.0
            position_pnl = direction * (self._mark(decision_t) - self.position.entry_price) * self.multiplier * self.position.contracts

        timestamp = self.data.loc[decision_t, "timestamp"] if "timestamp" in self.data.columns else None
        if timestamp is not None:
            minutes = timestamp.hour * 60 + timestamp.minute
            session_open = 9 * 60 + 30
            session_close = 16 * 60
            session_elapsed = np.clip(minutes - session_open, 0, 390) / 390.0
            time_to_close = np.clip(session_close - minutes, 0, 390) / 390.0
            time_sin = float(np.sin(2 * np.pi * minutes / (24 * 60)))
            time_cos = float(np.cos(2 * np.pi * minutes / (24 * 60)))
            regular = 1.0 if session_open <= minutes <= session_close else 0.0
            near_open = 1.0 if session_open <= minutes < session_open + 30 else 0.0
            near_close = 1.0 if session_close - 30 <= minutes <= session_close else 0.0
        else:
            session_elapsed = time_to_close = time_sin = time_cos = 0.0
            regular = near_open = near_close = 0.0

        spot = float(self.prices[decision_t])
        vol = self._vol(decision_t)
        current_position_option = [0.0] * 5
        if self.position is not None:
            call = self.position.kind in (1, 2)
            greeks = self._option_greeks(spot, self.position.strike, max(self.position.expiry_t - decision_t, 0), vol, call)
            current_position_option = [
                greeks[0] / max(spot, 1e-9), greeks[1],
                greeks[2] * spot, greeks[3] / max(spot, 1e-9),
                greeks[4] / max(spot, 1e-9),
            ]

        candidates = []
        for call in (True, False):
            for offset in STRIKE_OFFSETS:
                strike = self._strike_from_offset(spot, offset)
                for dte_days in DTE_DAYS:
                    q = self._option_greeks(spot, strike, dte_days * 7, vol, call)
                    candidates.extend([
                        q[0] / max(spot, 1e-9), q[1], q[2] * spot,
                        q[3] / max(spot, 1e-9), q[4] / max(spot, 1e-9),
                    ])

        portfolio = [
            self.cash / self.initial_cash,
            equity / self.initial_cash,
            drawdown,
            position_flag,
            position_pnl / self.initial_cash,
            1.0 if self.position is not None else 0.0,
            float(self.t - start) / max(self.episode_hours, 1),
            1.0,
        ]
        current_context = [
            session_elapsed, time_to_close, time_sin, time_cos,
            regular, near_open, near_close,
            0.0 if self.position is None else max(self.position.expiry_t - self.t, 0) / (30.0 * 7),
        ]
        return np.asarray(market + portfolio + current_context + current_position_option + candidates, dtype=np.float32)

    def _equity(self, t: int) -> float:
        if self.position is None:
            return self.cash
        mark = self._mark(t) * self.multiplier * self.position.contracts
        if self.position.kind in (1, -1):
            return self.cash + mark
        return self.cash + self.position.collateral - mark

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if len(action) != 5:
            raise ValueError(f"Expected 5 action values, got {action}")
        operation, option_type, strike_idx, dte_idx, size_idx = [int(x) for x in action]

        # Reward must include the complete effect of the action, including
        # transaction costs and execution slippage. The old implementation
        # reset the reward baseline after the action, effectively hiding those
        # costs from the learning signal.
        decision_t = max(self.t - 1, 0)
        pre_action_equity = self._equity(decision_t)

        if operation in (OPEN_LONG, OPEN_SHORT):
            self._open(operation, option_type, strike_idx, dte_idx, size_idx)
        elif operation == CLOSE:
            self._close()

        self.t += 1
        self._settle_expiry()
        terminated = self.t >= self.end_t
        self.equity = self._equity(self.t)
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = max(0.0, (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9))
        reward = (self.equity - pre_action_equity) / self.initial_cash
        reward -= drawdown * 0.02
        self.previous_equity = self.equity
        return self._observation(), float(reward), terminated, False, {
            "equity": self.equity,
            "drawdown": drawdown,
            "trade_count": len(self.trade_log),
        }
