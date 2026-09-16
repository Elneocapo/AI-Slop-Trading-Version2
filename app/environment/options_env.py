from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class OptionPosition:
    kind: int  # 1 call, -1 put
    strike: float
    expiry_index: int
    contracts: int
    entry_price: float


class OptionsTradingEnv(gym.Env):
    """Hourly synthetic-options market for reinforcement learning.

    The underlying is real historical hourly OHLCV data. Option prices are
    generated consistently from the underlying, an estimated volatility and
    Black-Scholes. This is deliberately a training simulator, not a source of
    historical option quotes.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.DataFrame,
        initial_cash: float = 500.0,
        max_contracts: int = 3,
        contract_multiplier: int = 100,
        transaction_cost: float = 0.75,
        slippage: float = 0.0025,
        max_holding_hours: int = 120,
    ):
        super().__init__()
        self.prices = prices.reset_index(drop=True).copy()
        self.initial_cash = float(initial_cash)
        self.max_contracts = int(max_contracts)
        self.multiplier = int(contract_multiplier)
        self.transaction_cost = float(transaction_cost)
        self.slippage = float(slippage)
        self.max_holding_hours = int(max_holding_hours)

        # 0 hold, 1 buy call, 2 buy put, 3 close position.
        self.action_space = spaces.Discrete(4)
        # Price/returns/technical state + portfolio state.
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(18,), dtype=np.float32)

        self.t = 0
        self.cash = self.initial_cash
        self.position: OptionPosition | None = None
        self.equity = self.initial_cash
        self.peak_equity = self.initial_cash

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _option_price(self, spot: float, strike: float, tau_years: float, vol: float, kind: int) -> float:
        if tau_years <= 0:
            return max(kind * (spot - strike), 0.0)
        vol = max(float(vol), 0.05)
        sigma_sqrt_t = vol * math.sqrt(tau_years)
        d1 = (math.log(max(spot, 1e-9) / strike) + 0.5 * vol * vol * tau_years) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t
        if kind == 1:
            return spot * self._norm_cdf(d1) - strike * self._norm_cdf(d2)
        return strike * self._norm_cdf(-d2) - spot * self._norm_cdf(-d1)

    def _features(self) -> np.ndarray:
        row = self.prices.iloc[self.t]
        close = float(row.Close)
        ret1 = float(row.return_1h)
        ret6 = float(row.return_6h)
        ret24 = float(row.return_24h)
        sma24_gap = float(row.sma24_gap)
        sma72_gap = float(row.sma72_gap)
        vol = float(row.volatility_24h)
        volume_z = float(row.volume_z)
        rsi = float(row.rsi_14)
        atr = float(row.atr_pct)
        position_kind = 0.0 if self.position is None else float(self.position.kind)
        position_pnl = 0.0 if self.position is None else self._position_pnl()
        drawdown = (self.equity / max(self.peak_equity, 1e-9)) - 1.0
        cash_ratio = self.cash / self.initial_cash
        moneyness = 0.0 if self.position is None else (close / self.position.strike - 1.0)
        dte_hours = 0.0 if self.position is None else max(self.position.expiry_index - self.t, 0) / 24.0
        return np.asarray([
            np.clip(ret1 * 20, -5, 5), np.clip(ret6 * 8, -5, 5),
            np.clip(ret24 * 4, -5, 5), np.clip(sma24_gap * 20, -5, 5),
            np.clip(sma72_gap * 20, -5, 5), np.clip(vol * 5, 0, 5),
            np.clip(volume_z / 5, -5, 5), np.clip((rsi - 50) / 10, -5, 5),
            np.clip(atr * 20, 0, 5), np.clip(position_kind, -1, 1),
            np.clip(position_pnl / self.initial_cash, -5, 5), np.clip(drawdown * 5, -5, 0),
            np.clip(cash_ratio, -5, 5), np.clip(moneyness * 10, -5, 5),
            np.clip(dte_hours / 30, 0, 5), np.clip((close / max(float(self.prices.Close.iloc[0]), 1e-9)) - 1, -5, 5),
            np.clip(self.t / max(len(self.prices), 1), 0, 1),
            np.clip(self.equity / self.initial_cash, -10, 10),
        ], dtype=np.float32)

    def _option_mark(self) -> float:
        if self.position is None:
            return 0.0
        spot = float(self.prices.Close.iloc[self.t])
        vol = max(float(self.prices.volatility_24h.iloc[self.t]), 0.15)
        tau = max(self.position.expiry_index - self.t, 0) / (24.0 * 365.0)
        return self._option_price(spot, self.position.strike, tau, vol, self.position.kind)

    def _position_pnl(self) -> float:
        if self.position is None:
            return 0.0
        return (self._option_mark() - self.position.entry_price) * self.position.contracts * self.multiplier

    def _equity(self) -> float:
        return self.cash + (self._option_mark() * self.position.contracts * self.multiplier if self.position else 0.0)

    def _execute_open(self, kind: int) -> None:
        if self.position is not None:
            return
        spot = float(self.prices.Close.iloc[self.t])
        vol = max(float(self.prices.volatility_24h.iloc[self.t]), 0.15)
        # Slightly OTM/ATM candidates are enough for the first learning stage.
        strike = round(spot * (1.01 if kind == 1 else 0.99), 2)
        expiry = min(self.t + self.max_holding_hours, len(self.prices) - 1)
        tau = max(expiry - self.t, 1) / (24.0 * 365.0)
        premium = self._option_price(spot, strike, tau, vol, kind)
        premium *= 1.0 + self.slippage
        total = premium * self.multiplier + self.transaction_cost
        if total > self.cash or premium <= 0:
            return
        self.cash -= total
        self.position = OptionPosition(kind, strike, expiry, 1, premium)

    def _execute_close(self) -> None:
        if self.position is None:
            return
        mark = self._option_mark() * (1.0 - self.slippage)
        proceeds = max(mark * self.position.contracts * self.multiplier - self.transaction_cost, 0.0)
        self.cash += proceeds
        self.position = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = int(self.np_random.integers(72, max(73, len(self.prices) - 2)))
        self.cash = self.initial_cash
        self.position = None
        self.equity = self.initial_cash
        self.peak_equity = self.initial_cash
        return self._features(), {}

    def step(self, action: int):
        previous_equity = self.equity
        if action == 1:
            self._execute_open(1)
        elif action == 2:
            self._execute_open(-1)
        elif action == 3:
            self._execute_close()

        self.t += 1
        terminated = self.t >= len(self.prices) - 1
        if self.position and self.t >= self.position.expiry_index:
            self._execute_close()
        self.equity = self._equity()
        self.peak_equity = max(self.peak_equity, self.equity)
        reward = (self.equity - previous_equity) / self.initial_cash
        reward -= max(0.0, (self.peak_equity - self.equity) / self.initial_cash) * 0.02
        if terminated and self.position:
            self._execute_close()
            self.equity = self.cash
        info = {
            "equity": float(self.equity),
            "cash": float(self.cash),
            "pnl": float(self.equity - self.initial_cash),
            "drawdown": float(self.equity / max(self.peak_equity, 1e-9) - 1.0),
        }
        return self._features(), float(reward), terminated, False, info
