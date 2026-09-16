import pandas as pd

from app.analysis import add_indicators, option_liquidity_score


def test_indicators_are_created():
    index = pd.date_range("2025-01-01", periods=80, freq="D")
    df = pd.DataFrame({"Open": 100, "High": 102, "Low": 98, "Close": range(100, 180), "Volume": 1000}, index=index)
    out = add_indicators(df)
    assert "rsi_14" in out
    assert "historical_vol_20" in out


def test_liquidity_score_penalizes_wide_spread():
    assert option_liquidity_score(9.9, 10.0, 1000, 1000) > option_liquidity_score(5, 8, 1000, 1000)
