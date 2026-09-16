from app.broker import PaperBroker
from app.domain import PaperOrder


def test_paper_broker_records_only_paper_order():
    broker = PaperBroker()
    result = broker.submit_order(PaperOrder("NVDA", "buy", 1, 10.0))
    assert result["status"] == "filled"
    assert result["mode"] == "paper"
    assert len(broker.orders) == 1
