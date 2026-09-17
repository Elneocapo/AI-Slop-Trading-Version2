from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# The policy now chooses the operation plus the contract characteristics.
# Parameters are ignored for HOLD/CLOSE, but keeping one fixed MultiDiscrete
# action makes the policy compatible with PPO and easy to translate to a broker.
HOLD = 0
OPEN_LONG = 1
OPEN_SHORT = 2
CLOSE = 3

CALL = 0
PUT = 1

# Strike is expressed as a percentage of spot.  Negative = ITM, zero = ATM,
# positive = OTM for calls; for puts the same offset still identifies a concrete
# strike, giving the agent a broad but finite option chain to choose from.
STRIKE_OFFSETS = (-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10)
DTE_DAYS = (1, 3, 5, 7, 14, 30)
CONTRACT_SIZES = (1, 2, 3, 5, 10)


@dataclass
class Position:
    kind: int  # 1 call long, -1 put long, 2 call short, -2 put short
    strike: float
    expiry_t: int
    entry_price: float
    contracts: int
    collateral: float = 0.0


class OptionsTradingEnv(gym.Env):
    """Hourly options environment with trader-style contract selection.

    This is still a synthetic options market: the underlying candles are real
    historical data while option marks are theoretical Black-Scholes values.
    The action space is deliberately broker-friendly: operation, call/put,
    strike bucket, expiry bucket and contract size.
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

        # 8 portfolio + 8 current time/session + 5 current-position option
        # values + 108 candidate contracts * 5 quote/Greek values.
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

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return exp(-0.5 * x * x) / sqrt(2.0 * np.pi)

    def _option_price(self, spot: float, strike: float, tau_hours: float, vol: float, call: bool) -> float:
        if tau_hours <= 0:
            return max(spot - strike, 0.0) if call else max(strike - spot, 0.0)
        tau = tau_hours / (24.0 * 252.0)
        vol = max(float(vol), 0.15)
        d1 = (log(max(spot, 1e-9) / max(strike, 1e-9)) + 0.5 * vol * vol * tau) / (vol * sqrt(tau))
        d2 = d1 - vol * sqrt(tau)
        nd1 = self._norm_cdf(d1)
        nd2 = self._norm_cdf(d2)
        if call:
            return spot * nd1 - strike * nd2
        return strike * (1.0 - nd2) - spot * (1.0 - nd1)

    def _option_greeks(self, spot: float, strike: float, tau_hours: float, vol: float, call: bool) -> tuple[float, float, float, float, float]:
        if tau_hours <= 0:
            intrinsic = max(spot - strike, 0.0) if call else max(strike - spot, 0.0)
            delta = 1.0 if call and spot > strike else -1.0 if (not call and spot < strike) else 0.0
            return intrinsic, delta, 0.0, 0.0, 0.0
        tau = tau_hours / (24.0 * 252.0)
        vol = max(float(vol), 0.15)
        sqrt_tau = sqrt(tau)
        d1 = (log(max(spot, 1e-9) / max(strike, 1e-9)) + 0.5 * vol * vol * tau) / (vol * sqrt_tau)
        d2 = d1 - vol * sqrt_tau
        price = self._option_price(spot, strike, tau_hours, vol, call)
        pdf = self._norm_pdf(d1)
        delta = self._norm_cdf(d1) if call else self._norm_cdf(d1) - 1.0
        gamma = pdf / (max(spot, 1e-9) * vol * sqrt_tau)
        theta_hour = (-(spot * pdf * vol) / (2.0 * sqrt_tau)) / (24.0 * 252.0)
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
        spot = float(self.prices[self.t])
        offset = STRIKE_OFFSETS[int(strike_idx)]
        strike = max(spot * (1.0 + offset), 0.01)
        dte_days = DTE_DAYS[int(dte_idx)]
        contracts = CONTRACT_SIZES[int(size_idx)]
        call = int(option_type) == CALL
        expiry_t = min(self.t + dte_days * 7, self.end_t)
        price = self._option_price(spot, strike, expiry_t - self.t, self._vol(self.t), call)
        cost = self.transaction_cost

        if operation == OPEN_LONG:
            total = price * self.multiplier * contracts + cost
            if total > self.cash:
                return
            self.cash -= total
            kind = 1 if call else -1
            self.position = Position(kind, strike, expiry_t, price, contracts)
        elif operation == OPEN_SHORT:
            collateral = spot * self.multiplier * contracts * 0.50
            premium = price * self.multiplier * contracts
            net_cash_needed = collateral - premium + cost
            if net_cash_needed > self.cash:
                return
            self.cash -= net_cash_needed
            kind = 2 if call else -2
            self.position = Position(kind, strike, expiry_t, price, contracts, collateral)

    def _close(self):
        if self.position is None:
            return
        mark = self._mark(self.t)
        value = mark * self.multiplier * self.position.contracts
        if self.position.kind in (1, -1):
            self.cash += max(value * (1 - self.slippage) - self.transaction_cost, 0.0)
        else:
            buyback = value * (1 + self.slippage) + self.transaction_cost
            self.cash += self.position.collateral - buyback
        self.position = None

    def _settle_expiry(self):
        if self.position is None or self.t < self.position.expiry_t:
            return
        spot = float(self.prices[self.t])
        call = self.position.kind in (1, 2)
        intrinsic = max(spot - self.position.strike, 0.0) if call else max(self.position.strike - spot, 0.0)
        value = intrinsic * self.multiplier * self.position.contracts
        if self.position.kind in (1, -1):
            self.cash += value
        else:
            self.cash += self.position.collateral - value
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
        return self._observation(), {}

    def _observation(self):
        start = self.t - self.lookback
        market = []
        for i in range(start, self.t):
            close = self.prices[i]
            market.extend(self.features[i].tolist())
            market.append(float(self.highs[i] - self.lows[i]) / max(close, 1e-9))

        equity = self._equity(self.t)
        drawdown = max(0.0, (self.peak_equity - equity) / max(self.peak_equity, 1e-9))
        position_flag = 0.0 if self.position is None else float(self.position.kind)
        position_pnl = 0.0
        if self.position is not None:
            direction = 1.0 if self.position.kind in (1, -1) else -1.0
            position_pnl = direction * (self._mark(self.t) - self.position.entry_price) * self.multiplier * self.position.contracts

        timestamp = self.data.loc[self.t, "timestamp"] if "timestamp" in self.data.columns else None
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

        spot = float(self.prices[self.t])
        vol = self._vol(self.t)
        current_position_option = [0.0] * 5
        if self.position is not None:
            call = self.position.kind in (1, 2)
            greeks = self._option_greeks(
                spot, self.position.strike,
                max(self.position.expiry_t - self.t, 0), vol, call
            )
            current_position_option = [
                greeks[0] / max(spot, 1e-9), greeks[1],
                greeks[2] * spot, greeks[3] / max(spot, 1e-9),
                greeks[4] / max(spot, 1e-9),
            ]

        # Quote/Greek matrix for every contract the policy can choose.
        # Ordering is deterministic: call then put, strike bucket, DTE bucket.
        candidates = []
        for call in (True, False):
            for offset in STRIKE_OFFSETS:
                strike = max(spot * (1.0 + offset), 0.01)
                for dte_days in DTE_DAYS:
                    tau_hours = dte_days * 7
                    q = self._option_greeks(spot, strike, tau_hours, vol, call)
                    candidates.extend([
                        q[0] / max(spot, 1e-9),
                        q[1],
                        q[2] * spot,
                        q[3] / max(spot, 1e-9),
                        q[4] / max(spot, 1e-9),
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

        if operation in (OPEN_LONG, OPEN_SHORT):
            self._open(operation, option_type, strike_idx, dte_idx, size_idx)
        elif operation == CLOSE:
            self._close()

        self.previous_equity = self._equity(self.t)
        self.t += 1
        self._settle_expiry()
        terminated = self.t >= self.end_t
        self.equity = self._equity(self.t)
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = max(0.0, (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9))
        reward = (self.equity - self.previous_equity) / self.initial_cash
        reward -= drawdown * 0.02
        return self._observation(), float(reward), terminated, False, {"equity": self.equity, "drawdown": drawdown}
