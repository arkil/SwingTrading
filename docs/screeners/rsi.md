# RSI Screener

**File:** `rsi_screener.py`  
**Dashboard:** `?scanner=rsi`  
**Default period:** RSI(14)

---

## What It Does

Finds three types of RSI-based trade setups across a stock universe: classic oversold/overbought reversals, momentum divergence, and trend-shift signals — each with different timing and quality characteristics.

---

## The 3 Signal Types

### 1. Oversold / Overbought (mean-reversion)
- **RSI_OVERSOLD** — RSI(14) crosses below the `oversold` threshold (default 30) → bullish reversal candidate
- **RSI_OVERBOUGHT** — RSI(14) crosses above the `overbought` threshold (default 70) → bearish reversal candidate

Confirmation requirements:
- Long: price must close above EMA-50 (don't catch falling knives in a downtrend)
- Short: price must close below EMA-50
- Volume ≥ `min_vol_mult` × 20-bar average

### 2. RSI Divergence (early reversal)
- **BULL_DIV** — price makes a lower low, RSI makes a higher low over the past `div_lookback` bars
- **BEAR_DIV** — price makes a higher high, RSI makes a lower high

Divergence fires before price confirms the reversal. More leading, more false signals. Best combined with a counter-trend setup or tight stop.

### 3. RSI Trend Momentum (momentum shift)
- **RSI_CROSS_50_BULL** — RSI(14) crosses above 50 from below → bullish momentum shift
- **RSI_CROSS_50_BEAR** — RSI(14) crosses below 50 from above → bearish momentum shift

Filtered by ADX > `adx_threshold` (default 20) — only fires in trending conditions, not ranges.

---

## Signal Quality Ranking

| Signal | Timing | Quality |
|--------|--------|---------|
| Divergence | Earliest | Lowest (most leading, most whipsaws) |
| Oversold/Overbought cross | Mid | Medium — confirmed reversal |
| 50-level cross | Latest | Highest — trend has shifted |

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
| RSI | Current RSI(14) value |
| Direction | BULL / BEAR |
| Divergence | BULL / BEAR / — |
| Above EMA50 | Price above 50-day EMA |
| ADX | ADX(14) trend strength |
| Vol vs Avg | Volume vs 20-bar average |
| Entry / Stop / Target | ATR trade levels |

---

## How to Run

```bash
# Default universe
python rsi_screener.py

# S&P 500
python rsi_screener.py --universe sp500

# Bullish signals only
python rsi_screener.py --direction bull

# Custom oversold level
python rsi_screener.py --oversold 25 --overbought 75

# Include divergence signals
python rsi_screener.py --include-divergence

# Specific tickers
python rsi_screener.py --tickers NVDA AAPL TSLA
```

---

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--universe` | `default` | `default` / `sp500` / `nasdaq100` / `both` / `watchlist` |
| `--tickers` | — | Override with specific symbols |
| `--direction` | both | `bull` / `bear` / `both` |
| `--oversold` | 30 | RSI oversold threshold |
| `--overbought` | 70 | RSI overbought threshold |
| `--include-divergence` | false | Also surface divergence signals |
| `--adx-threshold` | 20 | Min ADX for momentum shift signals |
| `--vol-mult` | 1.2 | Volume confirmation multiplier |
| `--div-lookback` | 20 | Bars for divergence detection |
| `--output` | — | Save to CSV |
