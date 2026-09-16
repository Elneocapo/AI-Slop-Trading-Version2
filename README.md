# BrokerIA — AI-assisted options research & paper trading

> **PAPER TRADING ONLY.** This repository does not place real-money orders. It is a research/education project and makes no promise of profitability.

BrokerIA is a modular Python application for quantitative options research, backtesting and paper trading. The architecture deliberately keeps the strategy separate from the risk manager and broker adapter.

## Architecture

`Market Data → Analysis → ML → Strategy → Risk Manager → Paper Broker → Monitor → Database`

The LLM layer, if added later, is auxiliary: it may summarize news or explain model output, but it must not directly decide or bypass quantitative risk controls.

## Safety defaults

- `TRADING_MODE=paper`
- `ALPACA_PAPER=true`
- Real-money broker execution is intentionally not implemented in this first version.
- The risk manager is a hard gate before every order.
- `TRADING_ENABLED=false` stops new orders.
- Secrets belong in `.env`, never in source control.

## Quick start

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python run.py --once
```

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python run.py --once
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py --once
```

Dashboard:

```bash
streamlit run app/ui/dashboard.py
```

Tests:

```bash
pytest
```

## Configuration

Copy `.env.example` to `.env`. The default configuration uses simulation and does not require broker credentials. Alpaca credentials are only needed when using the paper broker adapter.

## Current scope

1. Modular application foundation.
2. Market data abstraction with Yahoo Finance as auxiliary historical data.
3. Technical/volatility analysis.
4. Options contract filtering and liquidity metrics.
5. Explicit ML target and chronological evaluation.
6. Realistic-ish research backtesting with spread/slippage/commission assumptions.
7. Hard risk gate.
8. Paper broker adapter.
9. SQLite persistence and structured logging.
10. Streamlit dashboard.

Historical options-chain availability varies by provider; the project does not pretend that current option chains are equivalent to a complete historical options database.
