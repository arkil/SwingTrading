# Breakout Screener

**File:** `breakout_screener.py`  
**Dashboard:** `?scanner=breakout`  
**Method:** 6 research-backed breakout strategies

---

## What It Does

Runs every stock through 6 distinct breakout pattern detectors in a single pass. Each strategy is independently sourced and backtested in the literature. Results show which patterns fired, trade levels, and a composite signal count so you can prioritise the highest-conviction setups.

---

## The 6 Strategies

### 1. 52-Week High Breakout
**Source:** Minervini SEPA, George/Hwang (2004)  
**Win rate:** 68–76% in bull markets when volume-confirmed

- Price closes above its 52-week high
- Volume ≥ 1.5× 50-day average on the breakout bar
- Relative strength vs SPY positive over the past 52 bars
- Not more than 5% extended past the 52-week high (avoid chasing)

### 2. Volume Surge Breakout
**Source:** IBD/CAN SLIM volume studies

- Price closes above 20-bar SMA resistance with volume ≥ 1.5× 50-day avg
- 3 consecutive closes above 50-day EMA
- RS vs SPY positive

### 3. NR7 Breakout
**Source:** Toby Crabel, *Day Trading with Short-Term Price Patterns* (1990)

- Today's range is the narrowest of the last 7 bars (NR7)
- After NR7, price closes above the NR7 high (volatility expansion signal)
- Price above 50-day EMA
- Volume on the expansion bar ≥ 1.2× average

### 4. Bollinger Band Squeeze Breakout
**Source:** John Bollinger, *Bollinger on Bollinger Bands*

- BB bandwidth (20-period, 2σ) at a 6-month low → "squeeze" condition
- Price then closes above the upper Bollinger Band
- Volume ≥ 1.3× average on the breakout bar

### 5. Inside Bar Breakout
**Source:** QuantifiedStrategies backtests, price action methodology

- Today's high and low are fully inside the prior bar's range (inside bar)
- Next bar (or current bar) closes above the mother bar high
- Volume picks up on the outside close

### 6. MA Reclaim Breakout
**Source:** Minervini, O'Neil — 50-day EMA as institutional demand line

- Price was below 50-day EMA and has now crossed above it
- Volume ≥ 1.4× 50-day average on the reclaim bar
- 50-day EMA itself is sloping upward (not just a dead-cat bounce)

---

## Output Columns

| Column | Description |
|--------|-------------|
| Ticker | Symbol |
| Signals | Number of the 6 strategies triggered (higher = more conviction) |
| 52W High BO | 52-week high breakout fired |
| Vol Surge BO | Volume surge breakout fired |
| NR7 BO | NR7 volatility expansion fired |
| BB Squeeze BO | Bollinger squeeze breakout fired |
| Inside Bar BO | Inside bar breakout fired |
| MA Reclaim BO | 50-day EMA reclaim fired |
| Entry | Suggested entry price |
| Stop | ATR-based stop (1.5× ATR below entry) |
| Target | 2× ATR target |
| RS vs SPY | 1-year relative strength vs SPY |
| % from 52W H | Distance from 52-week high |

Results sorted by signal count descending (most strategies firing = highest conviction).

---

## How to Run

```bash
# Default universe
python breakout_screener.py

# Full S&P 500
python breakout_screener.py --universe sp500

# Both indices
python breakout_screener.py --universe both

# Specific tickers
python breakout_screener.py --tickers NVDA AAPL TSLA

# Only show tickers with 2+ signals firing
python breakout_screener.py --min-signals 2

# Save to CSV
python breakout_screener.py --output breakouts.csv
```

---

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--universe` | `default` | `default` / `sp500` / `nasdaq100` / `both` / `watchlist` |
| `--tickers` | — | Override with specific symbols |
| `--min-signals` | 1 | Minimum number of strategies that must fire |
| `--output` | — | Save results to CSV |

---

## Signal Conviction Guide

| Signals Firing | Interpretation |
|----------------|----------------|
| 5–6 | Extremely high conviction — multiple independent methods agree |
| 3–4 | High conviction — prioritise these |
| 2 | Moderate — worth watching |
| 1 | Low — single signal, verify manually |
