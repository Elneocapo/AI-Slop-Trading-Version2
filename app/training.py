from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from app.environment.options_env import OptionsTradingEnv


EPISODE_HOURS = 145 * 7
LOOKBACK = 60
DEFAULT_TIMESTEPS = 5_000_000


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

    df["return_1h"] = close.pct_change(1)
    df["return_6h"] = close.pct_change(6)
    df["return_24h"] = close.pct_change(24)
    df["sma24_gap"] = close / close.rolling(24).mean() - 1
    df["sma72_gap"] = close / close.rolling(72).mean() - 1
    df["volatility_24h"] = df["return_1h"].rolling(24).std() * np.sqrt(24 * 252)
    volume_mean = volume.rolling(48).mean()
    volume_std = volume.rolling(48).std().replace(0, np.nan)
    df["volume_z"] = (volume - volume_mean) / volume_std
    df["volume_ratio_24h"] = volume / volume.rolling(24).mean().replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_pct"] = tr.rolling(14).mean() / close

    minutes_of_day = timestamps.hour * 60 + timestamps.minute
    day_fraction = minutes_of_day / (24 * 60)
    df["time_sin"] = np.sin(2 * np.pi * day_fraction)
    df["time_cos"] = np.cos(2 * np.pi * day_fraction)
    df["weekday_sin"] = np.sin(2 * np.pi * timestamps.dayofweek / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * timestamps.dayofweek / 7)

    session_open_min = 9 * 60 + 30
    session_close_min = 16 * 60
    session_minutes = minutes_of_day - session_open_min
    regular = (session_minutes >= 0) & (session_minutes <= (session_close_min - session_open_min))
    df["is_regular_session"] = regular.astype(float)
    df["minutes_since_open"] = np.clip(session_minutes, 0, session_close_min - session_open_min) / 390.0
    df["minutes_to_close"] = np.clip(session_close_min - minutes_of_day, 0, 390) / 390.0
    df["near_open"] = ((minutes_of_day >= session_open_min) & (minutes_of_day < session_open_min + 30)).astype(float)
    df["near_close"] = ((minutes_of_day >= session_close_min - 30) & (minutes_of_day <= session_close_min)).astype(float)
    df["pre_market"] = (minutes_of_day < session_open_min).astype(float)
    df["after_hours"] = (minutes_of_day > session_close_min).astype(float)

    local_date = timestamps.date
    regular_open = open_.where(regular).groupby(local_date).transform("first")
    session_high = high.where(regular).groupby(local_date).cummax()
    session_low = low.where(regular).groupby(local_date).cummin()
    df["session_return"] = close / regular_open - 1
    df["session_high_gap"] = close / session_high - 1
    df["session_low_gap"] = close / session_low - 1
    session_range = (session_high - session_low).replace(0, np.nan)
    df["session_range_position"] = (close - session_low) / session_range

    rolling_high_24 = high.rolling(24).max()
    rolling_low_24 = low.rolling(24).min()
    rolling_range_24 = (rolling_high_24 - rolling_low_24).replace(0, np.nan)
    df["range_24h_position"] = (close - rolling_low_24) / rolling_range_24
    df["high_24h_gap"] = close / rolling_high_24 - 1
    df["low_24h_gap"] = close / rolling_low_24 - 1
    df["bar_return"] = close / open_ - 1
    df["bar_range_pct"] = (high - low) / close.replace(0, np.nan)

    daily_close = close.groupby(local_date).last()
    previous_day_close = pd.Series(local_date, index=df.index).map(daily_close.shift(1))
    df["gap_from_prev_close"] = open_ / previous_day_close - 1

    return df.dropna().reset_index(drop=True)


def evaluate(model: PPO, data: pd.DataFrame, label: str) -> dict:
    env = OptionsTradingEnv(
        data,
        initial_cash=500.0,
        lookback=LOOKBACK,
        episode_hours=EPISODE_HOURS,
        fixed_start=LOOKBACK,
    )
    obs, _ = env.reset(seed=123)
    terminated = False
    actions = []
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        actions.append(action.tolist())
        obs, _, terminated, _, info = env.step(action)
    result = {
        "label": label,
        "initial": 500.0,
        "final": float(env.equity),
        "pnl": float(env.equity - 500.0),
        "return_pct": float((env.equity / 500.0 - 1) * 100),
        "max_drawdown_pct": float(-info["drawdown"] * 100),
        "actions": actions,
    }
    return result


def train(
    ticker: str,
    timesteps: int = DEFAULT_TIMESTEPS,
    period: str = "730d",
    resume: bool = False,
) -> Path:
    data = load_hourly_data(ticker, period)
    if len(data) <= EPISODE_HOURS + LOOKBACK + 100:
        raise ValueError("Not enough hourly history for a 60-candle lookback and 145-day episodes")
    split = len(data) - EPISODE_HOURS - 1
    train_data = data.iloc[:split].reset_index(drop=True)
    test_data = data.iloc[split - LOOKBACK:].reset_index(drop=True)

    train_env = Monitor(OptionsTradingEnv(
        train_data, initial_cash=500.0, lookback=LOOKBACK, episode_hours=EPISODE_HOURS
    ))
    eval_env = Monitor(OptionsTradingEnv(
        test_data, initial_cash=500.0, lookback=LOOKBACK,
        episode_hours=EPISODE_HOURS, fixed_start=LOOKBACK
    ))

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    eval_dir = Path("training_eval")
    eval_dir.mkdir(exist_ok=True)
    best_dir = out_dir / "best"
    best_dir.mkdir(exist_ok=True)
    path = out_dir / f"ppo_options_{ticker.lower()}"

    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(eval_dir),
        eval_freq=50_000,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )

    if resume:
        model_file = Path(f"{path}.zip")
        if not model_file.exists():
            raise FileNotFoundError(
                f"Cannot resume: existing model not found at {model_file}. Run once without --resume to create it."
            )
        print(f"Resuming existing trader-action model: {model_file}")
        model = PPO.load(path, env=train_env, device="auto")
        print(f"Continuing training for {timesteps:,} additional timesteps.")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            verbose=1,
            seed=42,
            device="auto",
        )
        print(f"Starting new trader-action PPO model for {timesteps:,} timesteps.")

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=not resume,
    )
    model.save(path)
    result = evaluate(model, test_data, "final")

    print("\n=== 145-DAY OUT-OF-SAMPLE TEST ===")
    print(f"Ticker: {ticker}")
    print(f"Lookback: {LOOKBACK} hourly candles")
    print(f"Episode: {EPISODE_HOURS} hourly steps (~145 trading days)")
    print("Initial capital: €500.00")
    print(f"Final equity: €{result['final']:,.2f}")
    print(f"P&L: €{result['pnl']:,.2f}")
    print(f"Return: {result['return_pct']:.2f}%")
    print(f"Model saved: {path}.zip")
    print(f"Best checkpoint: {best_dir / 'best_model.zip'}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the options RL agent on hourly candles.")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--period", default="730d")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a model created with this same trader-action space. Do not use with older 6-action models.",
    )
    args = parser.parse_args()
    train(args.ticker.upper(), args.timesteps, args.period, args.resume)


if __name__ == "__main__":
    main()
