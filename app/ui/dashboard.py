import streamlit as st

from app.analysis import add_indicators
from app.config import settings
from app.data import get_historical_prices, get_stock_quote
from app.strategy import generate_signal

st.set_page_config(page_title="BrokerIA", page_icon="📈", layout="wide")
st.title("BrokerIA")
st.warning("PAPER TRADING — No real-money orders are implemented.")

with st.sidebar:
    ticker = st.text_input("Ticker", settings.ticker).upper().strip()
    period = st.selectbox("Historical period", ["6mo", "1y", "2y", "5y"], index=2)
    st.caption(f"Trading enabled: {settings.trading_enabled}")

if st.button("Run analysis"):
    try:
        quote = get_stock_quote(ticker)
        data = add_indicators(get_historical_prices(ticker, period=period))
        signal = generate_signal(ticker, data, None, settings.model_threshold)
        c1, c2, c3 = st.columns(3)
        c1.metric("Price", f"${quote['price']:.2f}")
        c2.metric("Signal", signal.direction)
        c3.metric("Model P(up)", f"{signal.confidence:.1%}")
        st.subheader("Quantitative reasoning")
        st.write(signal.reasoning)
        st.line_chart(data["Close"])
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
else:
    st.info("Choose a ticker and run one paper/research analysis cycle.")
