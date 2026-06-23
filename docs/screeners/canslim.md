# O'Neil CAN SLIM Scanner

**File:** `oneil_canslim_scanner.py`  
**Dashboard:** `?scanner=canslim`  
**Method:** William O'Neil's CAN SLIM framework  
**Source:** *How to Make Money in Stocks* (O'Neil, 1988/2009)

---

## What It Does

Scores each stock across all 7 CAN SLIM criteria, awards points per criterion, and ranks the output as STRONG BUY / BUY / WATCH / PASS. Combines fundamental factors (earnings, growth, institutional ownership) with technical factors (relative strength, supply/demand, market direction).

---

## The 7 Criteria

| Letter | Criterion | What's Checked |
|--------|-----------|----------------|
| **C** | Current quarterly EPS | ≥ 25% acceleration vs prior quarter |
| **A** | Annual earnings & revenue | Multi-year growth record, EPS trend |
| **N** | New highs | Price within 5% of 52-week high; new price highs in recent bars |
| **S** | Supply & demand | Up-volume > down-volume; volume surge; float analysis |
| **L** | Leader | RS Rating ≥ 80 (top 20% vs market); IBD-style weighted 12-month return |
| **I** | Institutional sponsorship | 30–85% institutional ownership; increasing (not topping out) |
| **M** | Market direction | SPY above its 50-day MA (only buy in confirmed uptrends) |

---

## Scoring

Each criterion awards multiple points based on strength. Total out of 18:

| Score | Verdict |
|-------|---------|
| ≥ 14 | **STRONG BUY** |
| ≥ 10 | **BUY** |
| ≥ 7 | **WATCH** |
| < 7 | PASS |

---

## RS Rating Calculation

IBD-style weighted 12-month return vs SPY:
- Most recent quarter: 40% weight
- Quarters 2, 3, 4: 20% each
- Percentile-ranked across the scanned universe (1–99)
- RS ≥ 80 awards full L-criterion points

---

## How to Run

```bash
# Default universe
python oneil_canslim_scanner.py

# Full S&P 500
python oneil_canslim_scanner.py --universe sp500

# Nasdaq-100 + watchlist
python oneil_canslim_scanner.py --universe nasdaq100

# Both indices
python oneil_canslim_scanner.py --universe both

# Only show STRONG BUY and BUY
python oneil_canslim_scanner.py --min-score 10

# Specific tickers
python oneil_canslim_scanner.py --tickers NVDA META AAPL MSFT
```

---

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--universe` | `default` | `default` / `sp500` / `nasdaq100` / `both` / `watchlist` |
| `--tickers` | — | Override universe with specific symbols |
| `--min-score` | 7 | Minimum score to show (7 = WATCH and above) |

---

## O'Neil's Key Rules

- Only buy market leaders (RS ≥ 80), never laggards
- Avoid stocks with excessive institutional ownership (>85%) — already discovered
- The M (market) condition gates everything — don't buy individual stocks in a downtrending market
- EPS acceleration in the most recent quarter is the single most important factor
