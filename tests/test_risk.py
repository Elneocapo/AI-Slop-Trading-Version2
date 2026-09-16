from app.config import Settings
from app.domain import Signal
from app.risk import RiskManager


def test_trading_disabled_rejects():
    s = Settings(trading_enabled=False)
    signal = Signal("NVDA", None, "BUY", .8, .1, 10, 10, 12, "test")
    result = RiskManager(s).check(signal, 10000, 0, 0)
    assert not result.approved
    assert any("disabled" in reason.lower() for reason in result.reasons)
