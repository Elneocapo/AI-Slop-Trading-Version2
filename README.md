# BrokerIA — AI-assisted options research & paper trading

> **PAPER TRADING ONLY.** This repository does not place real-money orders. It is a research/education project and makes no promise of profitability.

BrokerIA is a modular Python desktop application for quantitative options research, backtesting and paper trading. The strategy remains separate from the risk manager and broker adapter.

## Architecture

`Market Data → Analysis → ML → Strategy → Risk Manager → Paper Broker → Monitor → Database`

The desktop interface is built with Python/Tkinter and calls the existing application modules directly. Slow market-data/model work runs in a background thread so the interface stays responsive.

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
python app.py
```

Or explicitly:

```powershell
python run_gui.py
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Tests:

```bash
pytest
```

## Historical options data with Alpaca Basic

The RL backtest can use Alpaca's free Indicative historical options data from February 2024 onward. This is not historical OPRA NBBO; the downloader uses 1-hour option bars and creates an explicitly conservative bid/ask proxy from each bar's range.

In PowerShell, after creating an Alpaca account and API keys:

```powershell
$env:ALPACA_API_KEY="YOUR_KEY"
$env:ALPACA_SECRET_KEY="YOUR_SECRET"

python -m app.data.alpaca_options --ticker NVDA --period 730d
python train_ai.py --ticker NVDA --timesteps 100000 --data-source alpaca --options-file data/nvda_alpaca_indicative_options.csv.gz
```

For a saved-model evaluation only:

```powershell
python train_ai.py --ticker NVDA --eval-only --data-source alpaca --options-file data/nvda_alpaca_indicative_options.csv.gz
```

The historical options data is intended for research/backtesting and does not imply live-market execution quality.

The legacy Streamlit dashboard is no longer the primary interface. The desktop app is the supported UI.

## Desktop UI

The application provides:

- ticker and historical-period selection;
- non-blocking **Run analysis** execution;
- current price, model probability and signal display;
- quantitative reasoning output;
- an embedded price-history chart;
- a run log and visible error reporting;
- paper-trading safety status.

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
10. Tkinter desktop interface.

Historical options-chain availability varies by provider; the project does not pretend that current option chains are equivalent to a complete historical options database.

## Databento historical options cache

The OPRA downloader is cache-first. It estimates the cost before any billable request, submits one batch for the selected NVDA contracts, stores the raw batch under `data/databento_cache/`, and builds the processed panel at `data/nvda_real_options.csv.gz`.

Once the processed panel exists, training and evaluation use the local file and do not call Databento for quote data.

Databento batch downloads are intended for repeated access after the initial batch download, and the project additionally keeps a local copy so the RL workflow does not need Databento at all after preprocessing.

Run the first download with the hard $20 ceiling:

```powershell
$env:DATABENTO_API_KEY="YOUR_KEY"
python -m app.data.databento_options --ticker NVDA --period 30d --max-cost-usd 20
```

After that, use the local panel for RL training:

```powershell
python train_ai.py --ticker NVDA --timesteps 100000 --data-source real --options-file data/nvda_real_options.csv.gz
```

For later runs with the same period, if `data/nvda_real_options.csv.gz` exists, the downloader exits immediately and uses the local panel without needing the Databento API key.

