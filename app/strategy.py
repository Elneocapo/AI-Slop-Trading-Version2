from dataclasses import asdict
import pandas as pd

from .domain import OptionContract, Signal
from .ml import FEATURES, make_training_frame, train_model


def generate_signal(ticker: str, market: pd.DataFrame, option: OptionContract | None, threshold: float = 0.60) -> Signal:
    frame = make_training_frame(market)
    if len(frame) < 100:
        return Signal(ticker, option, "HOLD", 0.0, 0.0, 0.0, None, None, "Insufficient training history")
    train = frame.iloc[:-20]
    latest = frame.iloc[[-1]]
    model = train_model(train)
    probability = float(model.predict_proba(latest[FEATURES])[0, 1])
    direction = "BUY" if probability >= threshold else "HOLD"
    price = option.midpoint if option else float(market["Close"].iloc[-1])
    max_loss = price * 100 if option else 0.0
    return Signal(
        ticker=ticker,
        option_contract=option,
        direction=direction,
        confidence=probability,
        expected_move=probability - 0.5,
        max_loss=max_loss,
        suggested_entry=price if direction == "BUY" else None,
        suggested_exit=price * 1.25 if direction == "BUY" else None,
        reasoning=f"Random Forest P(up)={probability:.3f}; quantitative gate only.",
    )
