from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from app.environment.options_env import OptionsTradingEnv


def load_hourly_data(ticker: str, period: str = "730d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1h", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No hourly data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    close = df["Close"].astype(float)
    df["return_1h"] = close.pct_change(1)
    df["return_6h"] = close.pct_change(6)
    df["return_24h"] = close.pct_change(24)
    df["sma24_gap"] = close / close.rolling(24).mean() - 1
    df["sma72_gap"] = close / close.rolling(72).mean() - 1
    df["volatility_24h"] = df["return_1h"].rolling(24).std() * np.sqrt(24 * 252)
    volume_mean = df.Volume.rolling(48).mean()
    volume_std = df.Volume.rolling(48).std().replace(0, np.nan)
    df["volume_z"] = (df.Volume - volume_mean) / volume_std
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    tr = pd.concat([
        df.High - df.Low,
        (df.High - close.shift()).abs(),
        (df.Low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_pct"] = tr.rolling(14).mean() / close
    return df.dropna().reset_index(drop=True)


def train(ticker: str, timesteps: int, period: str = "730d") -> Path:
    data = load_hourly_data(ticker, period)
    split = int(len(data) * 0.8)
    train_data = data.iloc[:split].reset_index(drop=True)
    test_data = data.iloc[split:].reset_index(drop=True)

    train_env = Monitor(OptionsTradingEnv(train_data, initial_cash=500.0))
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        seed=42,
    )
    model.learn(total_timesteps=timesteps, progress_bar=True)

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"ppo_options_{ticker.lower()}"
    model.save(path)

    # Chronological out-of-sample evaluation, never used during training.
    eval_env = OptionsTradingEnv(test_data, initial_cash=500.0)
    obs, _ = eval_env.reset(seed=123)
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, _ = eval_env.step(int(action))
    print(f"Ticker: {ticker}")
    print(f"Training bars: {len(train_data):,}")
    print(f"Test bars: {len(test_data):,}")
    print(f"Initial capital: €500.00")
    print(f"Out-of-sample equity: €{eval_env.equity:,.2f}")
    print(f"Out-of-sample P&L: €{eval_env.equity - 500.0:,.2f}")
    print(f"Model saved: {path}.zip")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the options trading RL agent on hourly candles.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--period", default="730d")
    args = parser.parse_args()
    train(args.ticker.upper(), args.timesteps, args.period)


if __name__ == "__main__":
    main()
