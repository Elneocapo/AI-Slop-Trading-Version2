import numpy as np
import pandas as pd

from app.environment.options_env import CALL, OPEN_LONG, OptionsTradingEnv


def _sample_data(rows: int = 120) -> pd.DataFrame:
    idx = pd.date_range(
        "2026-01-02 09:30",
        periods=rows,
        freq="1h",
        tz="America/New_York",
    )
    close = np.linspace(180.0, 190.0, rows)
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(rows, 1_000_000.0),
            "timestamp": idx,
        }
    )


def test_episode_end_liquidates_open_position_on_last_visible_bar():
    env = OptionsTradingEnv(
        _sample_data(),
        initial_cash=500.0,
        lookback=60,
        episode_hours=2,
        fixed_start=60,
    )
    env.reset(seed=42)
    env._open(OPEN_LONG, CALL, 4, 2, 0)

    assert env.position is not None

    env.step(np.array([0, 0, 0, 0, 0], dtype=np.int64))
    _, _, terminated, _, _ = env.step(np.array([0, 0, 0, 0, 0], dtype=np.int64))

    assert terminated is True
    assert env.position is None
    assert len(env.trade_log) == 1
    assert env.trade_log[0]["reason"] == "episode_end"
    assert env.trade_log[0]["exit_t"] == 60
