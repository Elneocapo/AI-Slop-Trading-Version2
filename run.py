import argparse

from app.analysis import add_indicators
from app.broker import PaperBroker
from app.config import settings
from app.data import get_historical_prices
from app.domain import PaperOrder
from app.risk import RiskManager
from app.strategy import generate_signal


def main() -> None:
    parser = argparse.ArgumentParser(description="BrokerIA paper trading cycle")
    parser.add_argument("--once", action="store_true", help="run exactly one cycle")
    args = parser.parse_args()
    settings.validate_safety()
    market = add_indicators(get_historical_prices(settings.ticker))
    signal = generate_signal(settings.ticker, market, option=None, threshold=settings.model_threshold)
    print(f"MODE=PAPER | ticker={settings.ticker} | direction={signal.direction} | confidence={signal.confidence:.3f}")
    risk = RiskManager(settings).check(signal, settings.initial_cash, 0, 0.0)
    print(f"RISK approved={risk.approved} reasons={'; '.join(risk.reasons) or 'none'}")
    if risk.approved:
        order = PaperOrder(settings.ticker, "buy", 1, signal.suggested_entry or 0)
        print(PaperBroker().submit_order(order))
    if not args.once:
        print("Development mode: no infinite loop. Use --once for a single cycle.")


if __name__ == "__main__":
    main()
