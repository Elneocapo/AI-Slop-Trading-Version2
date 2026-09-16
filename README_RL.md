# AI Options Trainer

The core experiment is now a reinforcement-learning agent that learns simulated options decisions from **1-hour candles**.

## Current training setup

- Starting capital: **€500**
- Market data: hourly OHLCV
- Agent: PPO (Stable-Baselines3)
- Actions: hold, buy call, buy put, close position
- Reward: change in simulated equity, normalized by the €500 starting capital, with a small drawdown penalty
- Training/test split: chronological 80/20 split
- Test data is never used for training
- Model output: `models/ppo_options_<ticker>.zip`

## Important data model

The current Yahoo Finance source provides historical hourly **underlying** prices, but not a complete historical hourly options-chain archive. Therefore the training environment uses real historical hourly underlying candles and generates synthetic option prices with a Black-Scholes model and estimated volatility. This makes the first trainer reproducible, but it is **not equivalent to training on historical option quotes**.

The synthetic layer is intentionally isolated in `app/environment/options_env.py` so a historical options dataset can replace it later without changing the PPO agent interface.

## Run

Install the RL dependencies:

```bash
pip install -r requirements-rl.txt
```

Train for 200,000 steps on SPY:

```bash
python train_ai.py --ticker SPY --timesteps 200000
```

You can use another ticker, for example:

```bash
python train_ai.py --ticker NVDA --timesteps 200000
```

The environment starts each episode with €500 and the agent's reward is tied to changes in simulated equity. The final chronological test period is reported separately after training.

## Safety

There is no broker/API execution path in this trainer. It is simulation and research only.
