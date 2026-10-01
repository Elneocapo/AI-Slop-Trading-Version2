import numpy as np
import pandas as pd
import torch

from app.environment.options_env import OptionsTradingEnv
from app.models.structured_options_extractor import StructuredOptionsExtractor


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


def test_structured_observation_and_extractor_shapes():
    env = OptionsTradingEnv(
        _sample_data(),
        initial_cash=70.0,
        lookback=60,
        episode_hours=20,
        fixed_start=60,
    )
    obs, _ = env.reset(seed=42)

    assert set(obs) == {"market", "portfolio", "context", "position", "candidates"}
    assert obs["market"].shape == env.observation_space["market"].shape
    assert obs["portfolio"].shape == (8,)
    assert obs["context"].shape == (8,)
    assert obs["position"].shape == (9,)
    assert obs["candidates"].shape == (108, 10)

    extractor = StructuredOptionsExtractor(env.observation_space)
    batch = {
        key: torch.as_tensor(value[None, ...], dtype=torch.float32)
        for key, value in obs.items()
    }
    features = extractor(batch)

    assert features.shape == (1, 696)
    assert torch.isfinite(features).all()
