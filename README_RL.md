# AI Options Trainer

The core experiment is a reinforcement-learning agent that learns simulated options decisions from **1-hour candles**.

## Training design

At every decision point the agent receives the **previous 60 hourly candles** plus its portfolio state. It then chooses the action for the next hourly step:

- `HOLD`
- `BUY_CALL`
- `BUY_PUT`
- `SELL_CALL` (open a short call)
- `SELL_PUT` (open a short put)
- `CLOSE`

Each episode is approximately **145 trading days (~1,015 hourly steps)** and starts with **€500**. The environment advances one hourly candle at a time until the episode ends.

The reward is primarily the change in account equity. A small drawdown penalty is included so the policy is not rewarded purely for taking extreme risk. Short options use simulated margin so the €500 account cannot create unlimited leverage.

## Training until a strong policy is found

The default local run is **5,000,000 PPO timesteps**. Evaluation is performed every 50,000 steps and the best checkpoint is saved to:

```text
models/best/best_model.zip
```

The final model is saved to:

```text
models/ppo_options_<ticker>.zip
```

Run locally from scratch:

```bash
pip install -r requirements-rl.txt
python train_ai.py --ticker SPY --timesteps 5000000
```

For a longer run, simply increase `--timesteps`, for example `10000000` or `20000000`.

## Incremental / cumulative training

The trainer can continue from an existing model instead of resetting its learned policy. This preserves the PPO policy and optimizer state and adds more training experience on top of the existing model.

After a model has already been created, continue training it with:

```bash
python train_ai.py --ticker SPY --timesteps 1000000 --resume
```

The `--timesteps` value is **additional** training, not the new lifetime total. For example, a model trained for 5M steps and then resumed for another 1M has received approximately 6M PPO timesteps in total.

This is useful when new market data becomes available: the model can be updated without throwing away everything it previously learned. The final 145-day chronological holdout remains excluded from training on each run.

For safety, `--resume` refuses to run if the saved model does not exist. A normal run without `--resume` starts a fresh model.

## 145-day test

The final **145-day block of historical data is held out completely from training**. After training, the model is released into that untouched chronological block, starting with €500 and the same 60-candle lookback. The program reports final equity, P&L and return.

This is the important test: the model does not get to train on the candles it is subsequently tested on.

## Data limitation

Yahoo Finance provides historical hourly **underlying** OHLCV, but not a complete historical hourly options-chain archive. The simulator therefore prices options with Black-Scholes using the historical underlying and estimated volatility. Those are **synthetic option quotes**, not claimed historical option transactions.

The synthetic options layer is isolated in `app/environment/options_env.py`, so a proper historical options dataset can replace it later without changing the PPO interface.

## No broker connection

There is no broker/API execution path in this trainer. It is simulation and research only.
