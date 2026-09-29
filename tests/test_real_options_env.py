import pandas as pd

from app.environment.options_env import Position
from app.environment.real_options_env import RealOptionsTradingEnv


def test_put_expiry_uses_put_intrinsic_value():
    env = RealOptionsTradingEnv.__new__(RealOptionsTradingEnv)
    env.data = pd.DataFrame(
        [{"timestamp": pd.Timestamp("2026-01-02 16:00", tz="America/New_York"), "Close": 110.0}]
    )
    env.position = Position(
        kind=-1,  # long PUT
        strike=100.0,
        expiry_t=0,
        entry_price=2.0,
        contracts=1,
        entry_t=0,
        symbol="TEST_PUT",
    )
    env.cash = 0.0
    env.multiplier = 100
    env.transaction_cost = 0.25
    env.trade_log = []

    settled = env._settle_expiry_causal(0)

    assert settled is True
    assert env.position is None
    assert env.cash == 0.0
    assert len(env.trade_log) == 1
    assert env.trade_log[0]["exit_price"] == 0.0
    assert env.trade_log[0]["pnl"] == -200.25
