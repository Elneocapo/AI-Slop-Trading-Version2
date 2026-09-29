# AI-Slop-Trading-Version2 — Agent Handoff / Project Context

## 1. Purpose

This repository contains an experimental reinforcement-learning system for trading NVDA options with PPO/MaskablePPO.

The main objective is to build a model that can identify directional and contract opportunities on NVDA and eventually support small-account trading. The project is research/backtesting first. It is not currently safe or implemented as a live-money broker system.

The current development target is a small-account profile starting from €70, because the intended use case is to make individual option trades large enough to materially move a small account.

Do not promise profitability. Historical backtests are evidence about the tested historical period only.

Repository:
https://github.com/Elneocapo/AI-Slop-Trading-Version2

Typical local Windows path:
C:\Users\neofe\AI-Slop-Trading-Version2

## 2. Non-negotiable project rules

1. NVDA only for the RL project.
2. Use the existing local historical options panel at data\nvda_real_options.csv.gz.
3. Do not redownload Databento unless explicitly requested.
4. Avoid lookahead completely.
5. Keep the final 145-day OOS block untouched.
6. Do not choose a model using the final OOS results.
7. Do not silently replace a contract at live execution time just because the requested quote is unavailable.
8. Keep risk controls outside the neural network.
9. Before any real-money connection, implement and test a proper live order/execution layer.
10. Never assume that a backtest return translates directly into live profitability.

## 3. RL architecture

The main RL flow is:

historical NVDA data -> observation -> MaskablePPO -> masked action -> risk wrapper -> historical option environment -> reward

The model uses MaskablePPO from sb3-contrib.

At each decision point the observation contains:

- the previous 60 hourly candles;
- technical, volatility and time features;
- current portfolio state;
- current option position state when a position exists;
- a compact panel of candidate real option quotes.

The policy is intended to see only information through t-1.

## 4. Action space

The underlying environment exposes:

MultiDiscrete([4, 2, 9, 6, 5])

The dimensions represent:

- operation = HOLD / OPEN_LONG / OPEN_SHORT / CLOSE
- option type = CALL / PUT
- strike offset = 9 choices
- DTE bucket = 6 choices
- contract size = 5 choices

The risk wrapper exposes a compact Discrete action space:

- action 0 = HOLD
- action 1 = CLOSE
- actions 2..109 = 108 open combinations

The 108 open combinations are:

CALL/PUT × 9 strike offsets × 6 DTE buckets.

The current real-mode wrapper forces 1 contract while learning direction, strike and DTE. Contract sizing is therefore not genuinely learned yet.

## 5. Strike offsets and DTE

Current strike offsets:

(-10%, -5%, -2%, -1%, ATM, +1%, +2%, +5%, +10%)

Current DTE buckets:

(1, 3, 5, 7, 14, 30)

Real-mode policy learning excludes the 1-DTE bucket. The current minimum entry DTE index is 1, corresponding to the 3-DTE bucket.

The strike is a real Databento contract strike. Yahoo is not the source of the strike value. Yahoo spot is used to calculate relative target offsets and to help choose which real contract is closest to the requested offset.

## 6. Underlying market data

The RL trainer currently gets hourly NVDA OHLCV from Yahoo Finance through yfinance.

Features include:

- 1h, 6h and 24h returns;
- 24h and 72h moving-average gaps;
- rolling volatility;
- volume z-score and volume ratio;
- RSI;
- ATR percentage;
- time-of-day encodings;
- weekday encodings;
- regular-session flags;
- session return and session range features;
- 24h range position;
- bar return and bar range;
- previous-close gap.

The underlying is historical data used both for model features and for price-relative contract mapping.

## 7. Historical options data

The real RL environment uses:

data\nvda_real_options.csv.gz

This is a locally cached processed panel derived from Databento OPRA historical data.

Important fields include:

- timestamp
- symbol
- strike
- expiry
- option_type
- bid
- ask
- candidate_idx

The panel is intended to represent real historical option quotes rather than Black-Scholes prices.

Observed current alignment from recent runs:

- 2,472 hourly underlying candles;
- window from 2025-04-22 10:30 ET to 2026-09-18 15:30 ET;
- quote coverage about 85.7% of aligned regular-session underlying timestamps.

The quote coverage is not perfect. The historical environment masks unavailable entries and can use the most recent prior candidate quote within a bounded causal window.

## 8. Databento panel construction

The Databento downloader is in app/data/databento_options.py.

It uses OPRA.PILLAR and the cbbo-1m schema.

The pipeline:

1. load historical hourly NVDA data;
2. determine daily target contracts around the underlying spot;
3. use Databento definitions for actual option symbols, strikes, expirations and types;
4. retrieve historical OPRA NBBO quote data;
5. map quotes onto the hourly timeline;
6. save the processed panel locally.

The downloader is cache-first.

If data\nvda_real_options.csv.gz already exists, normal training and evaluation should use it directly.

Raw Databento batch data is also cached under data\databento_cache\.

Do not pay for or redownload the historical quote data just to retrain the model.

## 9. Yahoo + Databento methodology

Using two sources is intentional:

- Yahoo supplies the historical NVDA underlying and derived features.
- Databento supplies the real option contract definitions and historical bid/ask quotes.

The option strike itself always comes from the Databento contract definition.

A known methodological limitation is that the underlying and options are not sourced from exactly the same feed. During panel construction the Yahoo spot is used to choose which real contracts best match the relative strike targets.

For a stronger final/live system, investigate synchronizing the underlying and option timestamps using a common market-data source.

Do not describe the current system as having a completely homogeneous single-feed dataset.

## 10. Causality and no-lookahead

Causality is critical.

The intended decision boundary is:

decision_t = t - 1

The observation uses historical data through t-1.

The real option environment is intended to execute entries and normal closes using the quote available at t-1.

The strict causal real step was introduced after an earlier version accidentally valued positions using the next bar.

The current flow should be:

1. PPO observes information through t-1.
2. PPO chooses an action.
3. Entry or normal close uses information available at t-1.
4. The environment advances.
5. Equity and reward remain based on the causal information boundary.

Any future agent modifying execution, observation, marking or reward must preserve this property.

## 11. Reward design

The wrapper uses a hybrid reward.

Primary component:

- realized trade P&L transformed with tanh.

Additional shaping:

- mild loss-streak penalty;
- mild OTM penalty;
- small causal mark-to-market signal while holding;
- entry penalty to discourage churn;
- invalid or rejected action penalties.

Important current constants:

TRADE_REWARD_WEIGHT = 1.0

POSITION_MARK_REWARD_WEIGHT = 0.15

ENTRY_REWARD_PENALTY = 0.012

LOSS_STREAK_PENALTY = 0.003

OTM_PENALTY_START = 0.05

OTM_PENALTY_RATE = 0.10

MAX_OTM_PENALTY = 0.02

The current small-account reward scale is:

REWARD_PNL_SCALE = INITIAL_CASH * 0.10

For the €70 profile this is €7.

Previous experiments found that a sparse realized-P&L signal was more useful than blindly relying on dense equity reward. This still needs to be validated on the new small-account training.

## 12. Risk controls

Current small-account profile:

INITIAL_CASH = 70.0

MAX_TRADE_RISK_PCT = 0.35

Therefore the current maximum initial trade budget is approximately:

€70 × 35% = €24.50

This was chosen because earlier €500-profile runs observed entry costs around €24.31. The idea is to preserve roughly the same absolute one-contract position size while making P&L materially larger as a percentage of the small account.

Other controls:

- max account drawdown circuit breaker = 25%;
- one position at a time;
- one contract in current real-mode learning;
- naked shorts disabled in real mode;
- 3-DTE-and-longer entries during policy learning;
- round-trip cost filter;
- forced exit before expiration;
- executable action masks.

The 35% per-trade capital allocation is deliberately aggressive. It is a research profile derived from the desired small-account trade magnitude, not a claim that 35% is a safe live allocation.

## 13. Transaction cost model

Current real-mode transaction-cost assumption:

REAL_TRANSACTION_COST = 0.25

That represents €0.25 per side in the research environment, or €0.50 round-trip for one contract.

The model also rejects candidates when the round-trip transaction cost is more than 8% of option premium notional.

Entry is modeled at ask plus slippage.

Normal exit is modeled at bid minus slippage.

These are research assumptions. Before live deployment, use the exact broker's commission schedule, regulatory fees, exchange fees, FX costs, minimum order costs and fill behavior.

## 14. Current training profile

Current small-account profile:

- initial capital = €70;
- max trade budget = 35%;
- maximum one-contract position cost around €24.50;
- one contract;
- 1,000,000 default PPO timesteps;
- 50,000-step checkpoints;
- model tag = _small70;
- checkpoint directory = models/checkpoints/small70_v1.

The old €500 checkpoints are intentionally kept separate from the new small70 profile.

Do not mix them.

## 15. Training and data split

The final OOS block is:

145 trading days, approximately 1,015 hourly steps.

It remains untouched by training and checkpoint selection.

The validation block is:

60 × 7 = 420 hourly steps.

Validation is split into 3 contiguous segments.

Training episodes are:

20 × 7 = 140 hourly steps.

These shorter random training episodes are intended to expose PPO to more temporal starting locations during training.

Important distinction:

1,000,000 PPO timesteps does not mean 1,000,000 independent new market hours. The model repeatedly samples episodes from the available historical training region.

This creates a real risk of historical overfitting.

## 16. Checkpoint selection

The checkpoint selector uses only the pre-OOS validation block.

Current requirements:

- at least 4 total validation trades;
- at least 2 of 3 validation segments have trades.

The current selection score emphasizes:

worst validation-segment return - 0.50 × worst validation max drawdown

This was changed because an earlier selector allowed a positive mean return to hide a strongly negative validation segment.

Do not use the final OOS period to choose a checkpoint.

The audit is written to:

training_eval\validation_selection\checkpoint_selection.csv

## 17. Previous important bugs and fixes

### Real-option mask failure

At one point the available action mask was effectively 1 / 110, with zero accepted entries.

The root cause was candidate and timestamp mapping, not simply transaction fees.

This was fixed with:

- UTC-normalized integer timestamp keys;
- canonical candidate mapping;
- prior-quote matching;
- bounded lookup windows.

### Future quote leak

A previous real-environment step advanced time and then valued the position at the newly advanced timestamp.

That introduced lookahead.

The strict causal step was created to fix this.

### Best-checkpoint evaluation bug

An earlier version could evaluate the latest model instead of the checkpoint selected by validation.

This was fixed so the selected checkpoint is the model used for final OOS evaluation.

### Validation split issue

The final 145-day holdout was separated from a dedicated pre-OOS validation block.

### OTM reward-direction bug

The OTM penalty originally handled calls/puts incorrectly.

It was changed to be direction-aware so calls and puts are penalized according to their own moneyness.

### Expiry PUT bug

The real environment once used logic that could classify a long PUT as a CALL at expiration.

The current correct classification is:

call = position.kind in (1, 2)

A regression test exists in tests/test_real_options_env.py.

## 18. Previous results worth knowing

These results are historical context, not guarantees.

### Old €500 strict-causal V3 experiment

Approximately:

- final €599.75;
- return +19.95%;
- max drawdown -19.05%;
- 76 trades;
- 18 winners / 58 losers;
- 73 calls / 3 puts.

Segments:

- +45.84%;
- -2.44%;
- -11.70%.

### Old hybrid experiment with a future leak

One experiment produced approximately +30.21%.

It contained a known future-data leak at that point in project history and must not be treated as valid evidence.

### €500 V4 checkpoint 100k

Approximately:

- final €507.75;
- return +1.55%;
- max drawdown -13.28%;
- 25 trades.

The three OOS segments in that evaluation were negative.

### €500 V4 checkpoint 550k

Approximately:

- final €441.70;
- return -11.66%;
- max drawdown -25.52%;
- 72 trades;
- 14 winners / 58 losers;
- 62 calls / 10 puts.

Segments:

- +10.56%;
- -2.67%;
- -16.24%.

### €500 V4 checkpoint 150k

Approximately:

- final €548.93;
- return +9.79%;
- max drawdown -13.30%;
- 42 trades;
- 12 winners / 30 losers;
- 36 calls / 6 puts.

Segments:

- +18.75%;
- +3.74%;
- -5.55%.

These results show substantial sensitivity to training progress and market regime.

## 19. Current research conclusion

The biggest unresolved issue is generalization.

The project has already solved several implementation problems, but checkpoint results can change materially from one regime to another.

A future agent should not simply increase PPO timesteps until a high OOS result appears.

Prefer:

- walk-forward testing;
- multiple chronological OOS windows;
- stability across regimes;
- transaction-cost robustness;
- realistic execution modeling;
- risk-controlled small-account evaluation.

## 20. Small-account objective

The intended use case is that a relatively small trade, such as one option contract costing roughly €20-25, should be capable of moving a €70 account substantially.

The desired outcome is not to hard-code a target such as €70 -> €170 into the reward.

€70 -> €170 would be a +142.86% return.

The model should instead learn from actual trade outcomes and be evaluated on whether it can produce a robust positive expectancy.

Do not modify the reward just to chase a fixed target account value.

## 21. Why 50-70 € is difficult

A standard US equity option contract generally represents 100 shares.

Therefore:

premium × 100 = contract cost before other effects

A €0.20-equivalent option is roughly €20 of premium exposure before currency, fees and slippage.

With a €70 account, one contract can represent a very large fraction of the account.

This is why the small70 profile intentionally uses:

€70 capital
35% maximum trade budget
approximately €24.50 one-contract budget

The model is still high-risk by small-account standards.

## 22. Current broker status

The repository is not currently a live-money execution system.

app/broker.py contains a PaperBroker.

The general application safety configuration also keeps the first release in paper mode.

The general desktop application and the RL trainer are separate systems.

The general application stack is:

Market Data -> Analysis -> ML -> Strategy -> Risk Manager -> Paper Broker -> Monitor -> Database

The RL stack is:

training_v2.py -> RiskManagedPPOEnv -> RealOptionsTradingEnv -> MaskablePPO

Do not assume that the desktop application automatically executes the PPO model.

## 23. Live execution architecture that should eventually be built

A real deployment should be:

PPO
|
v
trade proposal
|
v
LIVE RISK GATE
|
v
ORDER MANAGER
|
v
BROKER API
|
v
fill / reject / partial fill
|
v
real position state
|
v
monitor / reconciliation
|
+--> next model observation

The PPO should propose.

The live risk engine should have authority to reject.

The order manager should execute and reconcile.

The network must never bypass the live risk gate.

## 24. Live issues not yet solved

### EUR/USD

The research account is expressed in EUR while US option prices are in USD.

A live system needs explicit:

EUR -> USD buying power -> USD P&L -> EUR equity

including real broker FX conversion behavior and costs.

### Live chain discovery

The historical panel is a preselected candidate universe.

A live system must discover and validate the currently tradable contracts.

### Contract identity

If PPO selects contract A, the live engine must either execute A or reject the trade.

Do not silently replace A with contract B.

The historical training wrapper currently has a fallback mechanism for missing real candidates. That is acceptable for research continuity but should not be used in live execution without explicit contract substitution logic and model awareness.

### Fills

Live orders can be:

- filled;
- partially filled;
- rejected;
- delayed;
- canceled.

The live order manager must handle all of these.

### Market microstructure

Live deployment must account for:

- spread;
- liquidity;
- latency;
- stale quotes;
- market open/close;
- halts;
- missing data;
- connection loss;
- unexpected broker position state.

## 25. Commands for the user on Windows PowerShell

From:

C:\Users\neofe\AI-Slop-Trading-Version2

Activate virtual environment:

.venv\Scripts\activate

Compile:

python -m py_compile app\training_v2.py app\environment\real_options_env.py

Run tests:

python -m pytest

Fresh small-account training:

python train_ai.py --ticker NVDA --timesteps 1000000 --data-source real --options-file data\nvda_real_options.csv.gz

Select checkpoint without retraining:

python train_ai.py --ticker NVDA --select-best --data-source real --options-file data\nvda_real_options.csv.gz

Evaluate selected checkpoint:

python train_ai.py --ticker NVDA --eval-only --data-source real --options-file data\nvda_real_options.csv.gz

Risk sweep:

python train_ai.py --ticker NVDA --risk-sweep --data-source real --options-file data\nvda_real_options.csv.gz

Validation risk sweep:

python train_ai.py --ticker NVDA --validation-risk-sweep --data-source real --options-file data\nvda_real_options.csv.gz

Compare two checkpoints:

python train_ai.py --ticker NVDA --compare-checkpoints 50000 100000 --data-source real --options-file data\nvda_real_options.csv.gz

## 26. Files an agent should inspect first

RL core:

- app/training_v2.py
- app/environment/options_env.py
- app/environment/real_options_env.py
- app/data/databento_options.py
- train_ai.py

Tests:

- tests/test_analysis.py
- tests/test_broker.py
- tests/test_options.py
- tests/test_risk.py
- tests/test_real_options_env.py

General application:

- app/broker.py
- app/risk.py
- app/strategy.py
- app/ml.py
- app/config.py
- run.py

## 27. Recommended order for the next agent

1. Read the current GitHub code before changing anything.
2. Confirm small70 constants and paths.
3. Run compile and tests.
4. Train the new small70 model if it has not already been trained.
5. Inspect the complete checkpoint-selection CSV rather than only the selected row.
6. Evaluate the selected checkpoint on untouched OOS.
7. Evaluate several chronological windows.
8. Implement walk-forward validation.
9. Fix EUR/USD accounting.
10. Build a real-time paper-trading bridge.
11. Compare simulated fills with paper fills.
12. Only then consider real-money deployment.

## 28. Source of truth

This README is a handoff document.

The actual GitHub code is the source of truth.

Constants, model paths, selected checkpoints, data windows and experimental results can change.

A future agent should inspect the current branch before relying on any value documented here.

Never treat a historical OOS result as proof of live profitability.

Never redownload the Databento dataset just to repeat an experiment.
