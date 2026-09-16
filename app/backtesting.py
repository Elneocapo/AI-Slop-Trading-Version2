import pandas as pd


def backtest_directional(df: pd.DataFrame, predictions: pd.Series, initial_cash: float = 10000.0, commission: float = 1.0, slippage_percent: float = 0.5):
    cash = initial_cash
    shares = 0
    equity = []
    trades = 0
    for timestamp, row in df.iterrows():
        price = float(row["Close"])
        pred = predictions.get(timestamp, 0)
        if pred == 1 and shares == 0:
            execution = price * (1 + slippage_percent / 100)
            shares = int(cash / execution)
            if shares > 0:
                cash -= shares * execution + commission
                trades += 1
        elif pred == 0 and shares > 0:
            execution = price * (1 - slippage_percent / 100)
            cash += shares * execution - commission
            shares = 0
            trades += 1
        equity.append(cash + shares * price)
    curve = pd.Series(equity, index=df.index, name="equity")
    returns = curve.pct_change().dropna()
    drawdown = curve / curve.cummax() - 1
    return {
        "equity": curve,
        "final_value": float(curve.iloc[-1]),
        "return_pct": float((curve.iloc[-1] / initial_cash - 1) * 100),
        "max_drawdown_pct": float(drawdown.min() * 100),
        "trades": trades,
        "sharpe": float((returns.mean() / returns.std()) * (252 ** 0.5)) if returns.std() else 0.0,
    }
