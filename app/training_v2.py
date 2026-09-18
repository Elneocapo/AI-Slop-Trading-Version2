from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor

from app.environment.options_env import CALL, CONTRACT_SIZES, DTE_DAYS, STRIKE_OFFSETS, OptionsTradingEnv

EPISODE_HOURS = 145 * 7
LOOKBACK = 60
DEFAULT_TIMESTEPS = 5_000_000
MAX_TRADE_RISK_PCT = 0.10
INVALID_ACTION_PENALTY = 0.01
NO_POSITION_CLOSE_PENALTY = 0.002


class RiskManagedPPOEnv(gym.Wrapper):
    """Expose a compact joint action space and enforce hard account risk."""

    OPEN_ACTIONS = 2 * len(STRIKE_OFFSETS) * len(DTE_DAYS)
    ACTION_COUNT = 2 + OPEN_ACTIONS

    def __init__(self, env: OptionsTradingEnv, max_trade_risk_pct: float = MAX_TRADE_RISK_PCT):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(self.ACTION_COUNT)
        self.max_trade_risk_pct = float(max_trade_risk_pct)
        self.last_rejected = False

    @property
    def trade_log(self):
        return self.env.trade_log

    @property
    def equity(self):
        return self.env.equity

    @property
    def position(self):
        return self.env.position

    def action_masks(self):
        """Return the state-dependent valid actions for MaskablePPO."""
        mask = np.zeros(self.ACTION_COUNT, dtype=bool)
        mask[0] = True  # HOLD is always valid.
        if self.env.position is not None:
            mask[1] = True  # CLOSE is valid only while a position is open.
        else:
            mask[2:] = True  # OPEN_CALL / OPEN_PUT choices.
        return mask

    def _decode(self, action: int):
        action = int(np.asarray(action).item())
        if action == 0:
            return 0, 0, 0, 0
        if action == 1:
            return 2, 0, 0, 0
        idx = action - 2
        per_type = len(STRIKE_OFFSETS) * len(DTE_DAYS)
        option_type = idx // per_type
        rem = idx % per_type
        strike_idx = rem // len(DTE_DAYS)
        dte_idx = rem % len(DTE_DAYS)
        return 1, option_type, strike_idx, dte_idx

    def _cheapest_affordable(self, option_type: int, risk_budget: float):
        spot = float(self.env.prices[self.env.t])
        vol = self.env._vol(self.env.t)
        best = None
        for strike_idx, offset in enumerate(STRIKE_OFFSETS):
            strike = max(spot * (1.0 + offset), 0.01)
            for dte_idx, dte_days in enumerate(DTE_DAYS):
                expiry_t = min(self.env.t + dte_days * 7, self.env.end_t)
                theoretical = self.env._option_price(
                    spot, strike, expiry_t - self.env.t, vol, option_type == CALL
                )
                execution_price = theoretical * (1.0 + self.env.slippage)
                required = execution_price * self.env.multiplier + self.env.transaction_cost
                if required <= risk_budget and required <= float(self.env.cash):
                    if best is None or required < best[0]:
                        best = (required, strike_idx, dte_idx)
        return best

    def _translate(self, action):
        operation, option_type, strike_idx, dte_idx = self._decode(action)
        self.last_rejected = False
        self.last_invalid_reason = None

        if operation == 0:
            return np.array([0, 0, 0, 0, 0], dtype=np.int64)

        if operation == 2:
            if self.env.position is None:
                self.last_invalid_reason = "close_without_position"
                return np.array([0, 0, 0, 0, 0], dtype=np.int64)
            return np.array([3, 0, 0, 0, 0], dtype=np.int64)

        if self.env.position is not None:
            self.last_rejected = True
            self.last_invalid_reason = "open_while_position_open"
            return np.array([0, option_type, strike_idx, dte_idx, 0], dtype=np.int64)

        equity = max(float(self.env._equity(self.env.t)), 0.0)
        risk_budget = equity * self.max_trade_risk_pct
        if risk_budget <= self.env.transaction_cost:
            self.last_rejected = True
            self.last_invalid_reason = "risk_budget_too_small"
            return np.array([0, option_type, strike_idx, dte_idx, 0], dtype=np.int64)

        spot = float(self.env.prices[self.env.t])
        strike = max(spot * (1.0 + STRIKE_OFFSETS[strike_idx]), 0.01)
        expiry_t = min(self.env.t + DTE_DAYS[dte_idx] * 7, self.env.end_t)
        theoretical = self.env._option_price(
            spot, strike, expiry_t - self.env.t, self.env._vol(self.env.t), option_type == CALL
        )
        execution_price = theoretical * (1.0 + self.env.slippage)

        allowed_size_idx = None
        for idx, contracts in enumerate(CONTRACT_SIZES):
            required = execution_price * self.env.multiplier * contracts + self.env.transaction_cost
            if required <= risk_budget and required <= float(self.env.cash):
                allowed_size_idx = idx

        if allowed_size_idx is None:
            fallback = self._cheapest_affordable(option_type, risk_budget)
            if fallback is None:
                self.last_rejected = True
                self.last_invalid_reason = "no_affordable_contract"
                return np.array([0, option_type, strike_idx, dte_idx, 0], dtype=np.int64)
            _, strike_idx, dte_idx = fallback
            allowed_size_idx = 0

        return np.array([1, option_type, strike_idx, dte_idx, allowed_size_idx], dtype=np.int64)

    def step(self, action):
        translated = self._translate(action)
        obs, reward, terminated, truncated, info = self.env.step(translated)

        if self.last_rejected:
            reward -= INVALID_ACTION_PENALTY
        elif self.last_invalid_reason == "close_without_position":
            reward -= NO_POSITION_CLOSE_PENALTY

        info = dict(info)
        info["risk_rejected"] = bool(self.last_rejected)
        info["invalid_reason"] = self.last_invalid_reason
        info["decoded_action"] = self._decode(action)
        return obs, float(reward), terminated, truncated, info


def load_hourly_data(ticker: str, period: str = "730d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1h", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No hourly data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    timestamps = pd.DatetimeIndex(df.index)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("America/New_York")
    else:
        timestamps = timestamps.tz_convert("America/New_York")
    df["timestamp"] = timestamps

    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)
    df["return_1h"] = close.pct_change()
    df["return_6h"] = close.pct_change(6)
    df["return_24h"] = close.pct_change(24)
    df["sma24_gap"] = close / close.rolling(24).mean() - 1
    df["sma72_gap"] = close / close.rolling(72).mean() - 1
    # Yahoo 1h data is approximately 7 regular-session bars per trading day.
    # Annualize using trading-session hours, not 24 calendar hours.
    df["volatility_24h"] = df["return_1h"].rolling(24).std() * np.sqrt(7 * 252)
    vm = volume.rolling(48).mean()
    vs = volume.rolling(48).std().replace(0, np.nan)
    df["volume_z"] = (volume - vm) / vs
    df["volume_ratio_24h"] = volume / volume.rolling(24).mean().replace(0, np.nan)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["atr_pct"] = tr.rolling(14).mean() / close

    minutes = timestamps.hour * 60 + timestamps.minute
    frac = minutes / (24 * 60)
    df["time_sin"] = np.sin(2 * np.pi * frac)
    df["time_cos"] = np.cos(2 * np.pi * frac)
    df["weekday_sin"] = np.sin(2 * np.pi * timestamps.dayofweek / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * timestamps.dayofweek / 7)
    open_min, close_min = 570, 960
    session_minutes = minutes - open_min
    regular = (session_minutes >= 0) & (session_minutes <= 390)
    df["is_regular_session"] = regular.astype(float)
    df["minutes_since_open"] = np.clip(session_minutes, 0, 390) / 390.0
    df["minutes_to_close"] = np.clip(close_min - minutes, 0, 390) / 390.0
    df["near_open"] = ((minutes >= open_min) & (minutes < open_min + 30)).astype(float)
    df["near_close"] = ((minutes >= close_min - 30) & (minutes <= close_min)).astype(float)
    df["pre_market"] = (minutes < open_min).astype(float)
    df["after_hours"] = (minutes > close_min).astype(float)

    local_date = pd.Series(timestamps.date, index=df.index)
    regular_open = open_.where(regular).groupby(local_date).transform("first")
    df["session_return"] = (close / regular_open - 1).where(regular, 0.0)
    session_high = high.where(regular).groupby(local_date).cummax()
    session_low = low.where(regular).groupby(local_date).cummin()
    df["session_high_gap"] = (close / session_high - 1).where(regular, 0.0)
    df["session_low_gap"] = (close / session_low - 1).where(regular, 0.0)
    rng = (session_high - session_low).replace(0, np.nan)
    df["session_range_position"] = ((close - session_low) / rng).where(regular, 0.0)
    rh, rl = high.rolling(24).max(), low.rolling(24).min()
    rr = (rh - rl).replace(0, np.nan)
    df["range_24h_position"] = (close - rl) / rr
    df["high_24h_gap"] = close / rh - 1
    df["low_24h_gap"] = close / rl - 1
    df["bar_return"] = close / open_ - 1
    df["bar_range_pct"] = (high - low) / close.replace(0, np.nan)
    daily_close = close.groupby(local_date).last()
    prev = local_date.map(daily_close.shift(1))
    df["gap_from_prev_close"] = open_ / prev - 1
    return df.dropna().reset_index(drop=True)


def evaluate(model, data: pd.DataFrame, report_dir: Path | None = None) -> dict:
    env = RiskManagedPPOEnv(
        OptionsTradingEnv(
            data,
            initial_cash=500.0,
            lookback=LOOKBACK,
            episode_hours=EPISODE_HOURS,
            fixed_start=LOOKBACK,
        )
    )
    obs, _ = env.reset(seed=123)
    terminated = False
    actions = []
    rejected = 0
    invalid_reasons = {}
    while not terminated:
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        action = int(np.asarray(action).item())
        actions.append(action)
        obs, _, terminated, _, info = env.step(action)
        rejected += int(info.get("risk_rejected", False))
        reason = info.get("invalid_reason")
        if reason:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    trades = list(env.trade_log)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    action_counts = {
        "hold": sum(a == 0 for a in actions),
        "close": sum(a == 1 for a in actions),
        "open_call": sum(2 <= a < 2 + len(STRIKE_OFFSETS) * len(DTE_DAYS) for a in actions),
        "open_put": sum(a >= 2 + len(STRIKE_OFFSETS) * len(DTE_DAYS) for a in actions),
    }
    trade_pnls = [float(t["pnl"]) for t in trades]
    max_trade_risk = max(
        (
            float(t["entry_price"]) * env.env.multiplier * int(t["contracts"]) + env.env.transaction_cost
            for t in trades
            if t["kind"] in (1, -1)
        ),
        default=0.0,
    )

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        if trades:
            audit = pd.DataFrame(trades)
            audit["option_type"] = audit["kind"].map({1: "CALL", -1: "PUT", 2: "SHORT_CALL", -2: "SHORT_PUT"})
            audit.to_csv(report_dir / "oos_trade_audit.csv", index=False)
        pd.DataFrame([{
            "hold": action_counts["hold"],
            "close": action_counts["close"],
            "open_call": action_counts["open_call"],
            "open_put": action_counts["open_put"],
            "risk_rejected": rejected,
            "trade_count": len(trades),
            "win_count": len(wins),
            "loss_count": len(losses),
            "final_equity": float(env.equity),
            "max_drawdown_pct": float(-info["drawdown"] * 100),
            "best_trade": float(max(trade_pnls, default=0.0)),
            "worst_trade": float(min(trade_pnls, default=0.0)),
            **{f"invalid_{k}": v for k, v in invalid_reasons.items()},
        }]).to_csv(report_dir / "oos_action_audit.csv", index=False)

    return {
        "final": float(env.equity),
        "pnl": float(env.equity - 500.0),
        "return_pct": float((env.equity / 500.0 - 1) * 100),
        "max_drawdown_pct": float(-info["drawdown"] * 100),
        "trades": trades,
        "actions": actions,
        "trade_count": len(trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate_pct": len(wins) / len(trades) * 100 if trades else 0.0,
        "long_count": sum(t["kind"] in (1, -1) for t in trades),
        "short_count": sum(t["kind"] in (2, -2) for t in trades),
        "call_count": sum(t["kind"] in (1, 2) for t in trades),
        "put_count": sum(t["kind"] in (-1, -2) for t in trades),
        "total_trade_pnl": float(sum(t["pnl"] for t in trades)),
        "best_trade": float(max(trade_pnls, default=0.0)),
        "worst_trade": float(min(trade_pnls, default=0.0)),
        "open_position": env.position is not None,
        "open_position_unrealized_pnl": (
            float((env.env._mark(env.env.t) - env.position.entry_price) * env.env.multiplier * env.position.contracts)
            if env.position is not None else 0.0
        ),
        "action_counts": action_counts,
        "risk_rejected": rejected,
        "invalid_reasons": invalid_reasons,
        "max_entry_notional": max_trade_risk,
    }


def train(ticker: str, timesteps: int = DEFAULT_TIMESTEPS, period: str = "730d", resume: bool = False) -> Path:
    data = load_hourly_data(ticker, period)
    if len(data) <= EPISODE_HOURS + LOOKBACK + 100:
        raise ValueError("Not enough hourly history for training")
    split = len(data) - EPISODE_HOURS - 1
    train_data = data.iloc[:split].reset_index(drop=True)
    test_data = data.iloc[split - LOOKBACK:].reset_index(drop=True)

    train_env = Monitor(
        RiskManagedPPOEnv(
            OptionsTradingEnv(train_data, initial_cash=500.0, lookback=LOOKBACK, episode_hours=EPISODE_HOURS)
        )
    )
    eval_env = Monitor(
        RiskManagedPPOEnv(
            OptionsTradingEnv(
                test_data,
                initial_cash=500.0,
                lookback=LOOKBACK,
                episode_hours=EPISODE_HOURS,
                fixed_start=LOOKBACK,
            )
        )
    )
    out = Path("models")
    out.mkdir(exist_ok=True)
    logs = Path("training_eval")
    logs.mkdir(exist_ok=True)
    best = out / "best"
    best.mkdir(exist_ok=True)
    path = out / f"ppo_options_{ticker.lower()}"
    callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(best),
        log_path=str(logs),
        eval_freq=50_000,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )

    if resume:
        raise ValueError("--resume is disabled for the new joint Discrete action space. Start a fresh model.")
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.05,
        clip_range=0.2,
        verbose=1,
        seed=42,
        device="auto",
    )
    print(f"Starting NEW joint-action PPO model for {timesteps:,} timesteps.")
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    model.save(path)

    report_dir = Path("training_eval") / "latest_oos"
    r = evaluate(model, test_data, report_dir=report_dir)

    print("\n=== 145-DAY OUT-OF-SAMPLE TEST ===")
    print(f"Ticker: {ticker}\nLookback: {LOOKBACK} hourly candles\nEpisode: {EPISODE_HOURS} hourly steps (~145 trading days)")
    print("Initial capital: €500.00")
    print(f"Final equity: €{r['final']:,.2f}\nP&L: €{r['pnl']:,.2f}\nReturn: {r['return_pct']:.2f}%\nMax drawdown: {r['max_drawdown_pct']:.2f}%")
    print(f"Closed trades: {r['trade_count']}\nWin rate: {r['win_rate_pct']:.2f}% ({r['win_count']}W / {r['loss_count']}L)")
    print(f"Long trades: {r['long_count']} | Short trades: {r['short_count']}\nCalls: {r['call_count']} | Puts: {r['put_count']}")
    print(f"Best trade P&L: €{r['best_trade']:,.2f}\nWorst trade P&L: €{r['worst_trade']:,.2f}\nSum of closed-trade P&L: €{r['total_trade_pnl']:,.2f}")
    print(
        f"Action distribution: HOLD {r['action_counts']['hold']} | CLOSE {r['action_counts']['close']} | "
        f"OPEN_CALL {r['action_counts']['open_call']} | OPEN_PUT {r['action_counts']['open_put']}"
    )
    print(f"Risk-rejected actions: {r['risk_rejected']}")
    print(f"Invalid-action reasons: {r['invalid_reasons']}")
    print(f"Max entry cost observed: €{r['max_entry_notional']:,.2f}")
    print(f"Open position at test end: {'YES' if r['open_position'] else 'NO'}")
    if r["open_position"]:
        print(f"Open-position unrealized P&L: €{r['open_position_unrealized_pnl']:,.2f}")
    print(f"Risk limit: {MAX_TRADE_RISK_PCT * 100:.0f}% of current equity per new position")
    print(f"OOS audit files: {report_dir}")
    print("No closed trades recorded in the out-of-sample test." if not r["trades"] else "Trade log recorded.")
    print(f"Model saved: {path}.zip")
    print(f"Best checkpoint: {best / 'best_model.zip'}")
    return path


def main():
    p = argparse.ArgumentParser(description="Train the NVDA hourly options RL agent.")
    p.add_argument("--ticker", default="NVDA")
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p.add_argument("--period", default="730d")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    train(a.ticker.upper(), a.timesteps, a.period, a.resume)


if __name__ == "__main__":
    main()
