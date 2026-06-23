# Data Requirements

## Source
Daily OHLCV fetched live via `yfinance` — no files needed to drop here.
The backtest runner fetches and caches data automatically.

## Universe (~600 symbols)
- S&P 500 (503 symbols)
- Nasdaq-100 (101 symbols)
- High-Growth Tech (~50 curated)
- Trending (dynamic, top movers)
- Custom Watchlist

## Cache Location
`Data/cache/` — parquet files per ticker, refreshed if older than 24h.

## Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `Open` | float | Open price |
| `High` | float | High price |
| `Low` | float | Low price |
| `Close` | float | Close price (adjusted) |
| `Volume` | float | Trading volume |

## Backtest Period
- Start: 2023-01-01
- End: latest available
- Minimum history per ticker: 420 days (for 200MA + RS calculation)

## Validation
- [ ] No gaps > 5 trading days
- [ ] No zero/null Close prices
- [ ] Volume > 0 on all trading days
- [ ] Sufficient history for all indicators (200MA needs 200+ bars)
