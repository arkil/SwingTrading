# IBD Scanner — Near Buy Zone Detection

**File:** `ibd_scanner.py`  
**Dashboard:** `?scanner=ibd`  
**Method:** IBD (Investor's Business Daily) buy zone patterns

---

## What It Does

Identifies stocks approaching or sitting inside IBD-style buy zones. Rather than waiting for a full breakout, this scanner finds stocks early — while still in base or pulling back to key moving averages — giving you time to plan entries before the crowd.

---

## The 6 Buy Zone Patterns

### In Base
Stock is consolidating within 15% of its 52-week high for 5+ weeks. This is the coiling-spring phase before a breakout. Watch list candidate — not an actionable entry yet, but building tension.

### Pullback to 10-Week MA
Stock in an uptrend that has pulled back to its 10-week (50-day) moving average on **light volume**. Classic O'Neil follow-on entry for existing holders and new buyers. Volume drying up signals institutional holding, not distribution.

### Wedge Tightening
Weekly price ranges are narrowing near the 52-week high — each bar's range tighter than the one before. Supply is drying up. Classic VCP precursor.

### Short Stroke
Tight, narrow weekly close following a shakeout (a sharp intraweek dip that recovers by Friday). Signals the shakeout was a bear trap; strong hands held. High-conviction add point.

### Crossing 10-Week MA (Xing 10W)
Price has just crossed above the 10-week MA from below. Follow-on buy if in confirmed uptrend with volume pickup. Lower quality than a base breakout but useful for adding exposure.

### High Tight Flag (HTF)
Stock gained 100%+ in 8 weeks or fewer, then formed a tight flag (< 20% deep, 3–5 weeks). Rare and powerful. Historically one of the most reliable high-octane patterns.

---

## Scoring

Each ticker gets a composite "My Points" score:
- RS Rating (percentile vs SPY, IBD-style weighted return)
- Pattern bonuses (+points per buy zone pattern detected)
- Proximity to 52-week high bonus
- IBD list membership (if user-supplied)

Results sorted by score descending.

---

## Output Columns

| Column | Description |
|--------|-------------|
| Ticker | Symbol |
| Pattern | Buy zone pattern(s) detected |
| My Points | Composite score |
| RS Rating | 1–99 percentile vs SPY |
| Price | Current close |
| % from 52W H | Distance from 52-week high |
| 10W MA | 10-week moving average |
| Vol vs Avg | Today's volume vs average |
| Weeks in Base | Weeks consolidating near highs |
| Entry / Stop / Target | ATR-based trade levels |
| Sector | Sector label |

---

## How to Run

```bash
# Default universe
python ibd_scanner.py

# Nasdaq-100 + watchlist
python ibd_scanner.py --universe nasdaq100

# Both indices
python ibd_scanner.py --universe both

# Only In Base and Pullback patterns
python ibd_scanner.py --patterns "In Base" "Pullback"

# Specific tickers
python ibd_scanner.py --tickers NVDA AAPL TSLA META

# High Tight Flag only (rare but powerful)
python ibd_scanner.py --patterns HTF
```

---

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--universe` | `default` | `default` / `sp500` / `nasdaq100` / `both` / `watchlist` |
| `--tickers` | — | Override with specific symbols |
| `--patterns` | all | Space-separated pattern names to filter |
| `--min-rs` | 60 | Minimum RS Rating |
| `--output` | — | Save to CSV |
