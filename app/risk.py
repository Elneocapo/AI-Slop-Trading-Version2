from .config import Settings
from .domain import RiskDecision, Signal


class RiskManager:
    def __init__(self, settings: Settings):
        self.s = settings

    def check(self, signal: Signal, portfolio_value: float, open_positions: int, daily_pnl: float, quantity: int = 1) -> RiskDecision:
        reasons: list[str] = []
        if not self.s.trading_enabled:
            reasons.append("Trading disabled by emergency/safety switch")
        if self.s.trading_mode.lower() != "paper" or not self.s.alpaca_paper:
            reasons.append("Non-paper execution is forbidden")
        if signal.direction != "BUY":
            reasons.append("Signal is not BUY")
        amount = (signal.suggested_entry or 0) * quantity * (100 if signal.option_contract else 1)
        if amount > self.s.max_trade_amount:
            reasons.append("Trade amount exceeds max_trade_amount")
        if portfolio_value and amount / portfolio_value > self.s.max_portfolio_exposure:
            reasons.append("Portfolio exposure limit exceeded")
        if open_positions >= self.s.max_open_positions:
            reasons.append("Maximum open positions reached")
        if portfolio_value and daily_pnl <= -(portfolio_value * self.s.max_daily_loss):
            reasons.append("Daily loss limit reached")
        option = signal.option_contract
        if option:
            if option.spread_percentage > self.s.max_spread_percent:
                reasons.append("Option spread too wide")
            if option.volume < self.s.min_option_volume:
                reasons.append("Option volume too low")
            if option.open_interest < self.s.min_open_interest:
                reasons.append("Option open interest too low")
            if not self.s.min_dte <= option.dte <= self.s.max_dte:
                reasons.append("Option DTE outside allowed range")
        return RiskDecision(approved=not reasons, reasons=tuple(reasons))
