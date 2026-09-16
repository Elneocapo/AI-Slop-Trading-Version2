from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt

import gymnasium as gym
import numpy as np
from gymnasium import spaces


HOLD = 0
BUY_CALL = 1
BUY_PUT = 2
SELL_CALL = 3
SELL_PUT = 4
CLOSE = 5


@dataclass
class Position:
    kind: int
    strike: float
    expiry_t: int
    entry_price: float
    contracts: int
    collateral: float = 0.0


class OptionsTradingEnv(gym.Env):
    """Single-position hourly options environment using Black-Scholes marks."""

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
            "return_1h",
            "return_6h",
            "return_24h",
            "sma24_gap",
            "sma72_gap",
            "volatility_24h",
            "volume_z",
            "rsi_14",
            "atr_pct",
        ]
        self.feature_cols = [c for c in feature_cols if c in self.data.columns]
        self.features = self.data[self.feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        self.n_features = len(self.feature_cols) + 1

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback * self.n_features + 8,),
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
        call = self.position.kind == 1
        return self._option_price(spot, self.position.strike, tau, self._vol(t), call)

    def _open(self, kind: int):
        if self.position is not None:
            return
        spot = float(self.prices[self.t])
        call = kind == 1
        strike = spot * (1.01 if call else 0.99)
        expiry_t = min(self.t + 120, self.end_t)
        price = self._option_price(spot, strike, expiry_t - self.t, self._vol(self.t), call)
        if kind in (1, -1):
            total = price * self.multiplier + self.transaction_cost
            if total > self.cash:
                return
            self.cash -= total
            self.position = Position(kind=kind, strike=strike, expiry_t=expiry_t, entry_price=price, contracts=1)
        else:
            collateral = spot * self.multiplier * 0.50
            total = price * self.multiplier + self.transaction_cost
            if collateral + total > self.cash:
                return
            self.cash -= total + collateral
            self.position = Position(kind=kind, strike=strike, expiry_t=expiry_t, entry_price=price, contracts=1, collateral=collateral)

    def _close(self):
        if self.position is None:
            return
        mark = self._mark(self.t)
        if self.position.kind in (1, -1):
            self.cash += max(mark * self.multiplier * (1 - self.slippage) - self.transaction_cost, 0.0)
        else:
            buyback = mark * self.multiplier * (1 + self.slippage) + self.transaction_cost
            self.cash += self.position.collateral - buyback
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
        position_pnl = 0.0 if self.position is None else float((self._mark(self.t) - self.position.entry_price) * self.multiplier)
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
        return np.asarray(market + portfolio, dtype=np.float32)

    def _equity(self, t: int) -> float:
        if self.position is None:
            return self.cash
        mark = self._mark(t)
        if self.position.kind in (1, -1):
            return self.cash + mark * self.multiplier
        return self.cash + self.position.collateral - mark * self.multiplier

    def step(self, action):
        action = int(action)
        if action == BUY_CALL:
            self._open(1)
        elif action == BUY_PUT:
            self._open(-1)
        elif action == SELL_CALL:
            self._open(2)
        elif action == SELL_PUT:
            self._open(-2)
        elif action == CLOSE:
            self._close()

        self.previous_equity = self._equity(self.t)
        self.t += 1
        terminated = self.t >= self.end_t
        self.equity = self._equity(self.t)
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = max(0.0, (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9))
        reward = (self.equity - self.previous_equity) / self.initial_cash
        reward -= drawdown * 0.02
        return self._observation(), float(reward), terminated, False, {"equity": self.equity, "drawdown": drawdown}
