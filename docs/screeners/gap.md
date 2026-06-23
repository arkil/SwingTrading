# Gap Scanner

**File:** `gap_screener.py`  
**Dashboard:** `?scanner=gap`

---

## What It Does

Detects and classifies significant opening price gaps across a stock universe, distinguishing between high-conviction tradeable gaps and low-quality noise. Also tracks fill status of past gaps.

---

## The 4 Gap Types

### BREAKAWAY GAP (highest conviction — trade it)
- Gap occurs out of a **consolidation base**: Bollinger Band width is below its 6-month median (squeeze condition)
- Price gaps above a key resistance / prior pivot high
- Volume surge **≥ 2× average**
- Interpretation: institutional accumulation; these rarely fill quickly

### CONTINUATION GAP (runaway gap — add or new entry)
- Gap occurs **mid-trend**: price already above EMA-50 and trending
- Volume ≥ 1.5× average
- Interpretation: trend acceleration; typically does not fill in the near term

### EXHAUSTION GAP (reversal warning — avoid or fade)
- Gap occurs after an **extended move**: price ≥ 20% above 52-week low AND RSI > 70
- Volume spike but weak follow-through (close near open, upper wick)
- Interpretation: high probability of gap fill — avoid new longs or consider short

### COMMON GAP (noise — filtered out)
- Small gap (< `min_gap_pct`) occurring within a consolidation range
- Usually fills within 1–5 days
- Not surfaced by default

---

## Gap Fill Tracker

For each gap detected in the last `lookback_days`:
- **Fill %** — how much of the gap has been retraced by current price
  - 0% = gap fully open
  - 100% = gap completely filled (price traded back through the gap)
- **Status** — Open / Partial / Filled

---

## Trade Setup

| Gap Direction | Entry | Stop | Target |
|---------------|-------|------|--------|
| Gap Up (long) | Buy near gap open or close | Gap low (prior close) | Gap close + 2.0 × gap size |
| Gap Down (short) | Short near gap open or close | Gap high (prior close) | Gap close − 2.0 × gap size |

---

## Output Columns

| Column | Description |
|--------|-------------|
| Ticker | Symbol |
| Gap Type | BREAKAWAY / CONTINUATION / EXHAUSTION / COMMON |
| Gap Date | Date the gap occurred |
| Gap % | Size of the gap as % of prior close |
| Direction | UP / DOWN |
| Vol Surge | Volume vs 20-day average on gap day |
| Fill % | How much of the gap has been retraced |
| BB Squeeze | Whether gap occurred from a Bollinger squeeze |
| RSI | RSI on gap day |
| Above EMA50 | Whether price was above 50-day EMA at gap |
| Entry / Stop / Target | Trade levels |

---

## How to Run

```bash
# Default universe
python gap_screener.py

# S&P 500 universe
python gap_screener.py --universe sp500

# Both indices
python gap_screener.py --universe both

# Only gap-up signals
python gap_screener.py --direction up

# Only breakaway and continuation gaps (exclude exhaustion)
python gap_screener.py --types breakaway continuation

# Look back 20 days for recent gaps
python gap_screener.py --lookback 20

# Minimum gap size filter
python gap_screener.py --min-gap 1.5

# Specific tickers
python gap_screener.py --tickers NVDA AAPL TSLA
```

---

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--universe` | `default` | `default` / `sp500` / `nasdaq100` / `both` / `watchlist` |
| `--tickers` | — | Override with specific symbols |
| `--direction` | both | `up` / `down` / `both` |
| `--types` | all | `breakaway` `continuation` `exhaustion` (space-separated) |
| `--min-gap` | 1.0 | Minimum gap size in % |
| `--lookback` | 10 | Days to look back for gaps |
| `--output` | — | Save results to CSV |
