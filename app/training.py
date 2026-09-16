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
        actions.append(int(action))
        obs, _, terminated, _, info = env.step(int(action))
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
    # Keep the final 145-day block completely untouched for the final test.
    if len(data) <= EPISODE_HOURS + LOOKBACK + 100:
        raise ValueError("Not enough hourly history for a 60-candle lookback and 145-day episodes")
    split = len(data) - EPISODE_HOURS
    train_data = data.iloc[:split].reset_index(drop=True)
    test_data = data.iloc[split - LOOKBACK:].reset_index(drop=True)

    train_env = Monitor(OptionsTradingEnv(
        train_data,
        initial_cash=500.0,
        lookback=LOOKBACK,
        episode_hours=EPISODE_HOURS,
    ))
    eval_env = Monitor(OptionsTradingEnv(
        test_data,
        initial_cash=500.0,
        lookback=LOOKBACK,
        episode_hours=EPISODE_HOURS,
        fixed_start=LOOKBACK,
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
                f"Cannot resume: existing model not found at {model_file}. "
                "Run once without --resume to create it."
            )
        print(f"Resuming existing model: {model_file}")
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
        print(f"Starting new PPO model for {timesteps:,} timesteps.")

    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True, reset_num_timesteps=not resume)
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
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--period", default="730d")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training the existing saved model instead of starting from scratch.",
    )
    args = parser.parse_args()
    train(args.ticker.upper(), args.timesteps, args.period, args.resume)


if __name__ == "__main__":
    main()
