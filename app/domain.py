from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    underlying: str
    strike: float
    expiration: date
    option_type: Literal["call", "put"]
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_percentage(self) -> float:
        mid = self.midpoint
        return 0.0 if mid <= 0 else (self.ask - self.bid) / mid * 100

    @property
    def dte(self) -> int:
        return (self.expiration - date.today()).days


@dataclass(frozen=True)
class Signal:
    ticker: str
    option_contract: OptionContract | None
    direction: Literal["BUY", "HOLD", "SELL"]
    confidence: float
    expected_move: float
    max_loss: float
    suggested_entry: float | None
    suggested_exit: float | None
    reasoning: str


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PaperOrder:
    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    price: float
    option_symbol: str | None = None
