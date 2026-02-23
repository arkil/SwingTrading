# Day Trade Backtester

Config-driven Python backtesting package for intraday strategies using Yahoo Finance (`yfinance`) data.

## Features
- Extensible strategy interface (`BaseStrategy`)
- Config-based runs (symbol, interval, strategy, risk model)
- Yahoo intraday data fetcher
- Console summary with win/loss, P&L, R-multiples, and trend bifurcation
- Example strategy: Bollinger Band + RSI mean-reversion (SPY 5m)

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install .
dtb --config configs/bollinger_rsi_spy.yaml
```

## Add a new strategy
1. Create a file in `src/daytrade_backtester/strategies/` implementing `BaseStrategy`.
2. Register it in `src/daytrade_backtester/strategies/registry.py`.
3. Point your config `strategy.name` to that key and add any strategy params.

## Notes
- Yahoo 5m/1m data has lookback limits. Use `period` values Yahoo supports for intraday.
- This package uses an estimated option-return model from underlying SPY movement. It is not exact options tape replay.
