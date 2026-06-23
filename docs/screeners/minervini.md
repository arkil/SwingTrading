# Minervini SEPA Screener

**File:** `minervini_screener.py`  
**Dashboard:** `?scanner=minervini`  
**Method:** Mark Minervini's SEPA — Specific Entry Point Analysis  
**Sources:** *Trade Like a Stock Market Wizard* (2013), *Think & Trade Like a Champion* (2017), US Investing Championship wins (1997, 2021)

---

## What It Does

Finds stocks in a Stage 2 uptrend that have formed a Volatility Contraction Pattern (VCP) and are near or at the breakout pivot. These are Minervini's ideal entries: strong trend + tightening base + volume dry-up + breakout on big volume.

---

## The Four-Layer Filter

### 1. Trend Template — all 8 must pass for Stage 2

| ID | Condition |
|----|-----------|
| T1 | Price > 150-day SMA |
| T2 | Price > 200-day SMA |
| T3 | 150-day SMA > 200-day SMA |
| T4 | 200-day SMA trending UP for ≥ 1 month |
| T5 | 50-day SMA > 150-day SMA AND 50-day > 200-day SMA |
| T6 | Price > 50-day SMA |
| T7 | Price ≥ 30% above 52-week low |
| T8 | Price within 25% of 52-week high |

Score 8/8 → **Stage 2 ✅**. Score 6–7 → Stage 2 (partial). Screener default requires ≥ 7.

### 2. Relative Strength Rating (RS)

IBD-style weighted 12-month return vs SPY: most recent quarter weighted 2×.
```
RS = (Q4 × 2 + Q3 × Q2 + Q1) / 5   (relative to SPY)
```
Percentile-ranked across the scanned universe. Default cutoff: **RS ≥ 70** (top 30%).

### 3. VCP — Volatility Contraction Pattern

Looks back 60 bars for a base with:
- ≥ 2 swing high-to-low contractions, each shallower than the last
- Volume drying up in the latest contraction vs the first (< 85% of earlier avg volume)
- Tightest contraction depth < 20%

**Pivot** = high of the last (tightest) contraction.

### 4. Breakout Check

| State | Condition |
|-------|-----------|
| Near Pivot | Price within 5% below pivot (buyable soon) |
| Broken Out | Price 0–5% above pivot AND volume ≥ 1.4× 50-day avg |

---

## Output Columns

| Column | Description |
|--------|-------------|
| Stage | Stage 2 / Stage 2 (partial) / Stage 1 / Stage 3 / Stage 4 |
| Trend Score | 0–8 conditions passed |
| RS Rating | Percentile vs universe (0–100) |
| VCP | ✅ if pattern detected |
| Contractions | Number of tightening swings |
| VCP Depth % | Depth of the tightest contraction |
| Vol Dry-Up | ✅ if volume declining in contractions |
| Pivot | Breakout price level |
| % from Pivot | Distance of current price from pivot |
| Near Pivot / Broken Out | Entry state flags |
| Vol vs Avg50 | Today's volume vs 50-day average |
| Entry / Stop / Target (3:1) | Trade levels: entry just above pivot, 8% stop, 3:1 target |

---

## How to Run

```bash
# Nasdaq-100 (default for this screener)
python minervini_screener.py

# S&P 500 scan
python minervini_screener.py --universe sp500

# Both indices
python minervini_screener.py --universe both

# Require VCP detection (stricter filter)
python minervini_screener.py --require-vcp

# Lower trend score threshold (more results)
python minervini_screener.py --min-score 6

# Higher RS filter (top 20% only)
python minervini_screener.py --min-rs 80
```

---

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--universe` | `nasdaq100` | `default` / `sp500` / `nasdaq100` / `both` / `watchlist` |
| `--min-score` | 7 | Min trend template score (7 = near-Stage 2, 8 = full Stage 2) |
| `--min-rs` | 70.0 | Min RS Rating percentile |
| `--require-vcp` | false | Only return stocks with a detected VCP |

---

## What Minervini Avoids

- Stage 1 (basing), Stage 3 (topping), Stage 4 (downtrend) stocks
- Wide and loose bases (lack of tightness = weak institutional support)
- Low RS stocks (laggards rarely become leaders)
- Chasing extended stocks (> 5% past the pivot)

---

## Position Sizing (Minervini's Rules)

- Max loss per trade: 7–10% below pivot (hard stop)
- Risk per trade: 0.5–2% of total portfolio
- Profit target: minimum 3:1 reward-to-risk
- Partial sells: +10%, +20%, let remainder run

---

## Sorting Logic

Results sorted: broken-out stocks first → near-pivot stocks → rest, then by RS Rating descending within each group.
