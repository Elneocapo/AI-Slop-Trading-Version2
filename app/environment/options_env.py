from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class OptionPosition:
    kind: int  # +1 long call, -1 long put, +2 short call, -2 short put
    strike: float
    expiry_index: int
    contracts: int
    entry_price: float
    collateral: float = 0.0


class OptionsTradingEnv(gym.Env):
    """Hourly options-learning environment.

    The agent sees the previous 60 hourly candles and portfolio state, then
    chooses an action for the next hourly candle. Each episode spans 145
    trading days (configurable), starting from a random historical point.

    Historical hourly underlying candles are real. Option quotes are synthetic
    Black-Scholes marks because Yahoo does not provide a complete historical
    hourly option-chain archive. The synthetic layer is isolated so a real
    historical option dataset can replace it later.
    """

    metadata = {"render_modes": []}

    HOLD = 0
    BUY_CALL = 1
    BUY_PUT = 2
    SELL_CALL = 3
    SELL_PUT = 4
    CLOSE = 5

    def __init__(
        self,
        prices: pd.DataFrame,
        initial_cash: float = 500.0,
        lookback: int = 60,
        episode_hours: int = 145 * 7,
        contract_multiplier: int = 100,
        transaction_cost: float = 0.75,
        slippage: float = 0.0025,
        option_horizon_hours: int = 120,
    ):
        super().__init__()
        self.prices = prices.reset_index(drop=True).copy()
        self.initial_cash = float(initial_cash)
        self.lookback = int(lookback)
        self.episode_hours = int(episode_hours)
        self.multiplier = int(contract_multiplier)
        self.transaction_cost = float(transaction_cost)
        self.slippage = float(slippage)
        self.option_horizon_hours = int(option_horizon_hours)

        self.action_space = spaces.Discrete(6)
        # 60 candles x 10 normalized market features + 8 portfolio features.
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(608,), dtype=np.float32)
        self.t = self.lookback
        self.end_t = self.t + self.episode_hours
        self.cash = self.initial_cash
        self.position: OptionPosition | None = None
        self.equity = self.initial_cash
        self.peak_equity = self.initial_cash

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _option_price(self, spot: float, strike: float, tau_years: float, vol: float, kind: int) -> float:
        if tau_years <= 0:
            return max((spot - strike) if kind > 0 else (strike - spot), 0.0)
        vol = max(float(vol), 0.05)
        sigma_sqrt_t = vol * math.sqrt(tau_years)
        d1 = (math.log(max(spot, 1e-9) / strike) + 0.5 * vol * vol * tau_years) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t
        if kind > 0:
            return spot * self._norm_cdf(d1) - strike * self._norm_cdf(d2)
        return strike * self._norm_cdf(-d2) - spot * self._norm_cdf(-d1)

    def _market_features(self, i: int) -> np.ndarray:
        row = self.prices.iloc[i]
        return np.asarray([
            np.clip(float(row.return_1h) * 20, -5, 5),
            np.clip(float(row.return_6h) * 8, -5, 5),
            np.clip(float(row.return_24h) * 4, -5, 5),
            np.clip(float(row.sma24_gap) * 20, -5, 5),
            np.clip(float(row.sma72_gap) * 20, -5, 5),
            np.clip(float(row.volatility_24h) * 5, 0, 5),
            np.clip(float(row.volume_z) / 5, -5, 5),
            np.clip((float(row.rsi_14) - 50) / 10, -5, 5),
            np.clip(float(row.atr_pct) * 20, 0, 5),
            np.clip((float(row.High) - float(row.Low)) / max(float(row.Close), 1e-9) * 20, 0, 5),
        ], dtype=np.float32)

    def _position_mark(self) -> float:
        if self.position is None:
            return 0.0
        spot = float(self.prices.Close.iloc[self.t])
        vol = max(float(self.prices.volatility_24h.iloc[self.t]), 0.15)
        tau = max(self.position.expiry_index - self.t, 0) / (24.0 * 365.0)
        kind = 1 if abs(self.position.kind) == 1 else -1
        return self._option_price(spot, self.position.strike, tau, vol, kind)

    def _position_value(self) -> float:
        if self.position is None:
            return 0.0
        mark = self._position_mark() * self.position.contracts * self.multiplier
        return mark if abs(self.position.kind) == 1 else -mark

    def _equity(self) -> float:
        if self.position is None:
            return self.cash
        if abs(self.position.kind) == 1:
            return self.cash + self._position_value()
        # Short-option collateral is already removed from cash.
        return self.cash - self._position_value()

    def _observation(self) -> np.ndarray:
        window = np.stack([self._market_features(i) for i in range(self.t - self.lookback + 1, self.t + 1)])
        if self.position is None:
            portfolio = np.zeros(8, dtype=np.float32)
        else:
            mark = self._position_mark()
            pnl = ((mark - self.position.entry_price) * self.position.contracts * self.multiplier
                   * (1 if abs(self.position.kind) == 1 else -1))
            portfolio = np.asarray([
                np.clip(self.position.kind / 2, -1, 1),
                np.clip(pnl / self.initial_cash, -5, 5),
                np.clip(self.position.strike / max(float(self.prices.Close.iloc[self.t]), 1e-9) - 1, -1, 1),
                np.clip((self.position.expiry_index - self.t) / (24 * 30), 0, 5),
                np.clip(self.cash / self.initial_cash, -5, 5),
                np.clip((self.equity / max(self.peak_equity, 1e-9)) - 1, -5, 0),
                np.clip(self.equity / self.initial_cash, -10, 10),
                1.0,
            ], dtype=np.float32)
        return np.concatenate([window.reshape(-1), portfolio]).astype(np.float32)

    def _open(self, side: int) -> None:
        if self.position is not None:
            return
        spot = float(self.prices.Close.iloc[self.t])
        vol = max(float(self.prices.volatility_24h.iloc[self.t]), 0.15)
        is_call = side in (self.BUY_CALL, self.SELL_CALL)
        is_long = side in (self.BUY_CALL, self.BUY_PUT)
        strike = round(spot * (1.01 if is_call else 0.99), 2)
        expiry = min(self.t + self.option_horizon_hours, self.end_t - 1, len(self.prices) - 1)
        tau = max(expiry - self.t, 1) / (24.0 * 365.0)
        kind = 1 if is_call else -1
        premium = self._option_price(spot, strike, tau, vol, kind)
        if premium <= 0:
            return
        premium *= (1 + self.slippage) if is_long else (1 - self.slippage)
        notional = premium * self.multiplier
        if is_long:
            total = notional + self.transaction_cost
            if total > self.cash:
                return
            self.cash -= total
            self.position = OptionPosition(kind, strike, expiry, 1, premium)
        else:
            # Defined simulation margin: reserve 50% of underlying notional plus premium.
            collateral = spot * self.multiplier * 0.50
            total = collateral + self.transaction_cost
            if total > self.cash:
                return
            self.cash -= total
            self.position = OptionPosition(-2 if is_call else -2, strike, expiry, 1, premium, collateral)

    def _close(self) -> None:
        if self.position is None:
            return
        mark = self._position_mark()
        if abs(self.position.kind) == 1:
            self.cash += max(mark * self.multiplier * (1 - self.slippage) - self.transaction_cost, 0.0)
        else:
            # Buy back the short option and release reserved collateral.
            buyback = mark * self.multiplier * (1 + self.slippage) + self.transaction_cost
            self.cash += self.position.collateral - buyback
        self.position = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        max_start = len(self.prices) - self.episode_hours - 1
        if max_start <= self.lookback:
            raise ValueError(f"Need more than {self.lookback + self.episode_hours + 1} hourly candles")
        self.t = int(self.np_random.integers(self.lookback, max_start + 1))
        self.end_t = self.t + self.episode_hours
        self.cash = self.initial_cash
        self.position = None
        self.equity = self.initial_cash
        self.peak_equity = self.initial_cash
        return self._observation(), {}

    def step(self, action: int):
        previous_equity = self.equity
        if action == self.BUY_CALL or action == self.BUY_PUT:
            self._open(action)
        elif action == self.SELL_CALL or action == self.SELL_PUT:
            self._open(action)
        elif action == self.CLOSE:
            self._close()

        self.t += 1
        terminated = self.t >= self.end_t or self.t >= len(self.prices) - 1
        if self.position and self.t >= self.position.expiry_index:
            self._close()
        self.equity = self._equity()
        self.peak_equity = max(self.peak_equity, self.equity)

        # Primary reward: make money. Small drawdown penalty discourages ruinous paths.
        reward = (self.equity - previous_equity) / self.initial_cash
        drawdown = max(0.0, (self.peak_equity - self.equity) / self.initial_cash)
        reward -= drawdown * 0.02

        if terminated and self.position:
            self._close()
            self.equity = self.cash
        info = {
            "equity": float(self.equity),
            "cash": float(self.cash),
            "pnl": float(self.equity - self.initial_cash),
            "return_pct": float((self.equity / self.initial_cash - 1) * 100),
            "drawdown": float(self.equity / max(self.peak_equity, 1e-9) - 1),
            "hours_elapsed": int(self.t - (self.end_t - self.episode_hours)),
        }
        return self._observation(), float(reward), terminated, False, info
