# MACD Screener

**File:** `macd_screener.py`  
**Dashboard:** `?scanner=macd`  
**Parameters:** fast=12, slow=26, signal=9 (classic MACD)

---

## What It Does

Scans for three distinct MACD-based signal types, ordered from earliest (most leading) to latest (highest quality). Optionally detects MACD divergence for early reversal warning.

---

## The 3 Signal Types

### 1. Histogram Flip (earliest signal — most leading)
- **HIST_FLIP_BULL** — histogram turns from negative to positive (first bar ≥ 0 after being < 0)
- **HIST_FLIP_BEAR** — histogram turns from positive to negative

Fires *before* the signal line cross. More false signals — best used with a trend filter. Good for early positioning.

### 2. Signal Line Cross (classic MACD signal)
- **BULLISH_CROSS** — MACD line crosses above signal line
- **BEARISH_CROSS** — MACD line crosses below signal line

Optional filter: require both lines below zero for bullish (stronger setup — more room to run) or above zero for bearish.

### 3. Zero Line Cross (latest — highest quality)
- **ZERO_BULL** — MACD line crosses above zero (trend has definitively shifted bullish)
- **ZERO_BEAR** — MACD line crosses below zero

Later signal, fewer whipsaws, higher win rate. Best for position trades.

---

## Divergence Detection (bonus)

| Type | Condition |
|------|-----------|
| Bullish divergence | Price makes a lower low, MACD makes a higher low over `div_lookback` bars |
| Bearish divergence | Price makes a higher high, MACD makes a lower high |

Divergence flags early momentum exhaustion before price confirmation.

---

## Volume Filter

All signals can be filtered by volume confirmation: volume ≥ `vol_mult` × 20-bar average. Default 1.3×. Reduces false signals significantly.

---

## Trade Levels

| Level | Formula |
|-------|---------|
| Stop | Entry ± 1.5 × ATR(14) |
| Target | Entry ± 2.5 × ATR(14) |
| R/R | ~1.67:1 |

---

## Output Columns

| Column | Description |
|--------|-------------|
| Ticker | Symbol |
| Signal | Signal type |
| Direction | BULL / BEAR |
| MACD | MACD line value |
| Signal Line | Signal line value |
| Histogram | Histogram value |
| Divergence | BULL / BEAR / — |
| Above EMA50 | Price above 50-day EMA |
| Vol vs Avg | Volume vs 20-bar average |
| Entry / Stop / Target | Trade levels |

---

## How to Run

```bash
# Default universe
python macd_screener.py

# S&P 500
python macd_screener.py --universe sp500

# Bullish signals only
python macd_screener.py --direction bull

# Only zero-line crosses (highest quality)
python macd_screener.py --signal-type zero

# Volume filter at 1.5×
python macd_screener.py --vol-mult 1.5

# Specific tickers
python macd_screener.py --tickers NVDA AAPL TSLA
```

---

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--universe` | `default` | `default` / `sp500` / `nasdaq100` / `both` / `watchlist` |
| `--tickers` | — | Override with specific symbols |
| `--direction` | both | `bull` / `bear` / `both` |
| `--signal-type` | all | `hist` / `cross` / `zero` / `all` |
| `--vol-mult` | 1.3 | Volume confirmation multiplier |
| `--div-lookback` | 20 | Bars to look back for divergence |
| `--output` | — | Save to CSV |
