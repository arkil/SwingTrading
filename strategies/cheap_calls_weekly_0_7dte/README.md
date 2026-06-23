# cheap_calls_weekly_0_7dte

**Type:** Options strategy — short-dated OTM calls  
**Status:** Iterate phase (build complete, sweep done, parameter tuning in progress)  
**Universe:** ~130 high-momentum equities (SP500 + NDQ100 + growth)  
**Backtest period:** 2023-01-01 → present

---

## What It Does

Buys cheap OTM calls (premium ≤ $1.50, delta 0.05–0.45) expiring within 0–7 calendar days when the weekly options scanner fires on a ticker. The core bet: if the GO Score correctly identifies pre-move setups (volume surge, relative strength vs SPY, RSI in momentum zone), the underlying move in the next 0–7 days should overcome theta decay and generate a 2× return on the option.

Option prices are simulated via Black-Scholes using historical OHLCV + a 30d historical vol × 1.2 IV proxy (no real chain data required).

---

## Entry Conditions

| Condition | Value |
|-----------|-------|
| Premium | ≤ $1.50/share ($150/contract) |
| Delta | 0.05 – 0.45 (OTM to near-ATM) |
| Strike | Within ±15% of spot |
| Relative strength vs SPY | ≥ 0% outperformance on signal day |
| Volume surge vs 63d avg | ≥ 1.3× |
| RSI-14 | 45 – 65 (mid-trend, not exhausted) |
| GO Score (simplified, 0–5) | ≥ threshold (sweep parameter) |

### GO Score Components (OHLCV-only)

| Component | Threshold | Points |
|-----------|-----------|--------|
| Rel strength vs SPY | ≥ 2.0% outperformance | 2 |
| Rel strength vs SPY | ≥ 0.5% outperformance | 1 |
| Volume surge | ≥ 2.0× 63d avg | 2 |
| Volume surge | ≥ 1.3× 63d avg | 1 |
| RSI-14 | In 45–65 | 1 |

> The live scanner's full GO Score (0–10) also includes V/OI ratio and P/C ratio from the live options chain. The backtest uses the OHLCV-only subset (0–5) — results are a lower bound on live filter quality.

---

## Exit Conditions

| Exit | Trigger |
|------|---------|
| Take profit | Option doubles (100% gain on premium) |
| Stop loss | Option loses 50% of premium paid |
| Time exit | Hold to expiry; intrinsic value = max(spot − strike, 0) |

Exits checked daily at close in order: stop → target → expiry.

---

## Sweep Grid

The backtest sweeps all combinations of:

- **GO Score threshold:** 0, 1, 2, 3, 4, 5
- **DTE at entry:** 0, 1, 2, 3, 5, 7

36 cells total. Trade files are saved to `trades/trades_go{N}_dte{D}.csv`.

---

## How to Run

```bash
cd strategies/cheap_calls_weekly_0_7dte
pip install -r requirements.txt

# Full 2D sweep (all GO thresholds × all DTEs)
python backtest.py

# Single cell: GO score ≥ 3, 7 DTE
python backtest.py --go 3 --dte 7

# Unfiltered baseline (GO = 0, DTE = 7)
python backtest.py --baseline

# Win-rate iteration sweep at GO ≥ 2, DTE = 5
python backtest.py --iterate
```

---

## File Structure

```
cheap_calls_weekly_0_7dte/
├── backtest.py          # 2D sweep runner — main entry point
├── strategy.py          # Signal generation wrapper
├── config.yaml          # Position sizing, risk, DTE/premium limits
├── requirements.txt
├── src/
│   ├── data_loader.py   # yfinance fetcher + parquet cache
│   ├── features.py      # RSI, vol_ratio, rel_str, EMA-20
│   ├── signals.py       # GO Score computation + signal filter
│   └── options_sim.py   # Black-Scholes pricer + daily P&L simulation
├── Data/                # Cached OHLCV (gitignored after generation)
├── trades/              # Per-cell trade logs (gitignored)
└── experiments/         # Sweep result CSVs (gitignored)
```

---

## Config Parameters (`config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_dte` | 7 | Max calendar days to expiry |
| `min_dte` | 0 | Min calendar days to expiry |
| `max_premium` | 1.50 | Max option premium per share |
| `delta_min` | 0.05 | Min delta (OTM floor) |
| `delta_max` | 0.45 | Max delta (not too deep ITM) |
| `moneyness_pct` | 15 | Strike within ±15% of spot |
| `go_score_min` | 0.0 | GO Score threshold (sweep this) |
| `iv_assumption` | 0.45 | Fallback IV for BS pricing |
| `fixed_amount` | 150 | Max spend per trade ($) |
| `max_positions` | 5 | Max concurrent trades |
| `stop_loss` | 50% | Exit if option loses 50% |
| `take_profit` | 100% | Exit if option doubles |

---

## Caveats

1. **IV is approximated** — 30d historical vol × 1.2 proxy. Actual IV on signal days may differ. A sensitivity test on `iv_assumption` is warranted.
2. **Simplified GO Score** — backtest uses 0–5 pts (OHLCV only). Live scanner scores 0–10 (includes options flow). Backtest is a lower bound.
3. **No bid/ask spread on options** — BS mid used. Real fills are worse by ~5–10%.
4. **Entry timing** — backtest enters at next open after signal. Live trading enters intraday.
