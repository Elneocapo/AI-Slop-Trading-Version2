from datetime import date, timedelta
from app.domain import OptionContract


def test_spread_percentage():
    option = OptionContract("X", "NVDA", 100, date.today() + timedelta(days=30), "call", 4, 5, 100, 500)
    assert option.spread_percentage == 22.22222222222222
    assert option.dte == 30
