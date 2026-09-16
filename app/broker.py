from abc import ABC, abstractmethod
from dataclasses import asdict
from .domain import PaperOrder


class BrokerInterface(ABC):
    @abstractmethod
    def submit_order(self, order: PaperOrder) -> dict: ...


class PaperBroker(BrokerInterface):
    """In-memory paper broker. There is deliberately no live order endpoint."""

    def __init__(self):
        self.orders: list[dict] = []

    def submit_order(self, order: PaperOrder) -> dict:
        if order.quantity <= 0 or order.price <= 0:
            raise ValueError("Invalid paper order")
        record = asdict(order) | {"status": "filled", "mode": "paper"}
        self.orders.append(record)
        return record
