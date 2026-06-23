# Livermore Pivotal Point Screener

**File:** `livermore_pivotal_screener.py`  
**Dashboard:** `?scanner=livermore`  
**Method:** Jesse Livermore's Market Method (adapted from *How to Trade in Stocks*, 1940)

---

## What It Does

Scans a stock universe for Jesse Livermore's three types of pivotal points — the exact price levels that, when broken after a proper counter-move and confirmed by volume, signal the start of a large directional move.

---

## The Three Signal Types

### UPWARD_PIVOT (Bullish)
Price rallies to a new swing high → pulls back at least `min_reaction_pct` → then closes **above** the prior swing high. The prior high was the "pivotal point." Livermore saw this breakout as the trigger for a major advance.

### DOWNWARD_PIVOT (Bearish)
Price drops to a new swing low → bounces at least `min_reaction_pct` → then closes **below** the prior swing low. Signals a major decline.

### CONTINUATION (follow-on)
After an initial pivot breakout, price consolidates near the breakout level and then thrusts again in the same direction within `lookback_bars * 3` bars. Flagged as an add-to-position signal.

---

## Detection Logic

1. **Swing highs/lows** — centred rolling window of `2 × swing_window + 1` bars. A swing high is the bar whose `High` equals the rolling max; same for lows.
2. **Counter-move check** — between the swing pivot and the breakout bar, the price must have retraced at least `min_reaction_pct` in the opposite direction. No reaction = no pivot.
3. **Volume expansion** — optional: the breakout bar's volume must exceed the 20-bar average. Livermore emphasised volume as confirmation.
4. **One signal per pivot level** — once a pivot fires, it is reset so the same level doesn't re-fire on consecutive bars.

---

## Output Columns

| Column | Description |
|--------|-------------|
| Signal | `UPWARD_PIVOT` or `DOWNWARD_PIVOT` |
| Signal Date | Date the pivot fired |
| Bars Ago | How many bars ago the signal appeared |
| Pivot Level | The swing high/low price that was broken |
| Close | Current price |
| % from Pivot | How far price is from the pivot level now |
| Vol Expansion | Volume above 20-bar avg on breakout bar |
| Continuation | Follow-on thrust detected after initial signal |
| % from 52W High | Position relative to 52-week high |
| Trend (EMA) | UPTREND / DOWNTREND per 50/200 EMA |

---

## How to Run

```bash
# Default universe (SP500 watchlist + custom names)
python livermore_pivotal_screener.py

# Scan full S&P 500 + watchlist
python livermore_pivotal_screener.py --universe sp500

# Nasdaq-100 only
python livermore_pivotal_screener.py --universe nasdaq100

# Both indices merged
python livermore_pivotal_screener.py --universe both

# Specific tickers
python livermore_pivotal_screener.py --tickers NVDA AAPL TSLA MSFT

# Bullish pivots only, trend-aligned (50 EMA > 200 EMA), volume required
python livermore_pivotal_screener.py --signal UPWARD_PIVOT --trend-aligned --vol-required

# Look back 20 bars for recent signals, save to CSV
python livermore_pivotal_screener.py --recent-bars 20 --output pivots.csv

# Custom settings: tighter reaction filter, wider swing window
python livermore_pivotal_screener.py --swing-window 8 --min-reaction 3.5
```

---

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--universe` | `default` | `default` / `sp500` / `nasdaq100` / `both` / `watchlist` |
| `--tickers` | — | Space-separated list (overrides universe) |
| `--file` | — | Text file with one ticker per line |
| `--period` | 252 | Lookback in trading days |
| `--swing-window` | 5 | Bars on each side to confirm a swing high/low |
| `--min-reaction` | 2.0 | Min % counter-move between pivot and breakout |
| `--recent-bars` | 10 | Only return signals fired in the last N bars |
| `--signal` | ALL | `ALL` / `UPWARD_PIVOT` / `DOWNWARD_PIVOT` |
| `--trend-aligned` | false | Only show signals aligned with 50/200 EMA trend |
| `--vol-required` | false | Only show signals with above-average breakout volume |
| `--output` | — | Save results to CSV path |

---

## Custom Watchlist

The watchlist (always included in every universe) covers thematic sectors: optical networking, AI infrastructure, nuclear/power, defense tech, robotics, GLP-1, cybersecurity, quantum computing, grid modernisation, and onshoring plays. Edit `WATCHLIST_TICKERS` in the file to customise.

---

## Trade Setup

Results are sorted: upward pivots by proximity to 52-week high (stronger = closer to high), then downward pivots. The pivot level itself is the natural stop reference — buy above the pivot, stop below it.
