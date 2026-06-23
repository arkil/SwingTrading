# daily_alerts_guardrails

**Type:** Swing trading — US equities, long/short signals  
**Status:** Build complete — best params found, pending full 600-ticker validation  
**Universe:** ~600 symbols (SP500 + NDQ100 + High-Growth + Trending)  
**Backtest period:** 2023-01-01 → present  
**Best result (30-ticker quick run):** 57.7% WR, +0.270R expectancy, Sharpe 3.96, MaxDD -5.9%

---

## What It Does

The Daily Alerts engine fires signals on hundreds of tickers daily but can enter overextended stocks with poor risk/reward (e.g. ARM at +53% above 50MA with RSI ~70 → -11% loss). This strategy adds **hard entry guardrails** validated by a vectorised backtest to ensure only high-conviction, well-positioned setups are taken.

Entries are taken at next open. Exits use ATR-based bracket orders (1.5× ATR stop, 2.0× ATR target).

---

## Guardrail Filters

All filters must pass for a trade to be entered:

| Filter | Best Value | Rationale |
|--------|-----------|-----------|
| `min_score` | ≥ 8 (STRONG only) | Score 8 beats score 7 by ~6pp win rate in sweep |
| `max_ext_pct` | ≤ 30% above 50MA | Blocks ARM-style overextended entries |
| `rsi_bull_min/max` | 42 – 70 (BUY) | Mid-trend entries; avoids overbought exhaustion |
| `rsi_bear_min/max` | 35 – 58 (SELL) | Appropriate for short-side entries |
| `min_adx` | ≥ 20 | Trend must be established, not ranging |
| `min_rr` | ≥ 1.0 | ATR bracket gives ~1.33 R/R; must set ≤ 1.33 |
| `min_vol_ratio` | ≥ 1.2× avg | Some institutional participation required |
| `require_trigger` | true | At least 1 TRIGGER signal (not just background) active |

### Sweep Findings

| Experiment | Setup | Result |
|------------|-------|--------|
| Sweep 1 — 8 combos, 30 tickers | score=8 vs 7, ext=20% vs 30%, rsi_max=65 vs 70 | Best: score=8, ext=30%, rsi_max=70 → 58.8% WR, +0.295R |
| Sweep 2 — full backtest, 30 tickers, 800 days | Best combo applied | 52 trades, **57.7% WR**, +0.270R expectancy, **Sharpe 3.96**, MaxDD -5.9% |

> Key insight: `max_ext_pct=30` beats `20` because the extension filter (not RSI) is the real ARM-prevention guardrail. RSI cap at 70 (vs 65) adds more trades without degrading quality.

---

## Entry Logic

```
for each daily alert signal:
    if score < 8 → skip
    if ext_pct > 30% → skip (too extended above 50MA)
    if RSI outside [42, 70] for BUY or [35, 58] for SELL → skip
    if ADX < 20 → skip
    if vol_ratio < 1.2 → skip
    if no TRIGGER signal active → skip
    if R/R (ATR T2 vs stop) < 1.0 → skip
    → enter at next market open
```

---

## Exit Logic

| Exit | Trigger |
|------|---------|
| Stop loss | Entry − 1.5 × ATR(14) |
| Take profit (T2) | Entry + 2.0 × ATR(14) |
| Time exit | After `max_hold_days` (default 20) |
| Signal exit (live) | EMA9 cross + MACD cross + below EMA50 (any 2 of 3) |

---

## How to Run

```bash
cd strategies/daily_alerts_guardrails
pip install -r ../../requirements.txt   # uses root requirements

# Full backtest with config.yaml guardrails
python run_backtest.py

# Quick test — small universe, default guardrails
python run_backtest.py --quick

# Parameter sweep across guardrail combos
python run_backtest.py --sweep

# Sweep then apply best params automatically
python run_backtest.py --sweep --apply-best

# Custom guardrails
python run_backtest.py --min-score 8 --max-ext 20 --rsi-max 65 --min-adx 20

# Specific tickers
python run_backtest.py --tickers NVDA AAPL MSFT TSLA --days 500
```

---

## File Structure

```
daily_alerts_guardrails/
├── run_backtest.py      # CLI entry point — all modes
├── config.yaml          # Guardrail params + account/risk config
├── IDEA.md              # Original motivation and design
├── src/
│   ├── indicators.py    # Vectorised: RSI, ATR, ADX, EMA, vol_ratio, ext_pct
│   ├── signals.py       # Guardrail scoring (applies all filters)
│   ├── backtest.py      # Numba-accelerated trade simulation
│   ├── sweep.py         # Polars parameter sweep across guardrail combos
│   └── fetcher.py       # yfinance OHLCV fetcher + parquet cache
└── Data/
    └── README.md        # Notes on data sourcing
```

---

## Config Parameters (`config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_score` | 8 | Minimum alerts engine score (STRONG conviction) |
| `max_ext_pct` | 30.0 | Max % price extension above 50MA |
| `rsi_bull_min/max` | 42 / 70 | RSI range for BUY signals |
| `rsi_bear_min/max` | 35 / 58 | RSI range for SELL signals |
| `min_adx` | 20.0 | Minimum ADX (trend strength) |
| `min_rr` | 1.0 | Minimum R/R ratio (ATR T2 vs stop) |
| `min_vol_ratio` | 1.2 | Minimum volume vs average |
| `require_trigger` | true | Require active TRIGGER signal |
| `percent_per_trade` | 1.0% | Risk 1% of equity per trade |
| `max_positions` | 5 | Max concurrent open trades |
| `stop_loss (ATR)` | 1.5× | ATR multiplier for stop |
| `take_profit (ATR)` | 2.0× | ATR multiplier for target |

---

## Pending Work

- [ ] Run full 600-ticker universe sweep to validate results at scale
- [ ] Test SELL-side signals (currently 0 SELL trades in backtest)
- [ ] Add earnings filter (avoid entry within 5 days of earnings)
- [ ] Test trailing stop vs fixed ATR exit
- [ ] Deploy best params to live `alerts_live_runner.py` as pre-trade guardrail

---

## Motivation

Real paper account context: $95,329 equity, -$1,768 unrealized (-3.54%) at strategy creation. ARM position was -11.1% — entered at 53% above 50MA with RSI ~70. These guardrails are specifically designed to prevent that class of entry.
