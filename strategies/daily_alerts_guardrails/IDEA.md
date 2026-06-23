# Strategy Idea: Daily Alerts Guardrails

## The Idea

Multi-indicator swing trading alerts on US equities (S&P 500 + Nasdaq-100 + High-Growth + Trending).
Current system generates signals but enters overextended stocks with poor risk/reward.
Goal: add hard entry filters (guardrails) and validate signal quality via vectorised backtest.

## Why It Should Work

Minervini/O'Neil research shows best entries cluster near:
- Price within 20% of 50MA (not too extended)
- RSI 42-65 at entry (momentum but not overbought)
- Volume confirmation (1.5× avg) on breakout
- ADX > 20 (trend has direction)

Current failures (ARM -11%, AMD -3.5%): entered at 53% above 50MA with RSI ~70.

## Entry Logic (refined)

**Required (all must pass):**
1. Score ≥ 8 (STRONG conviction only)
2. Price within 20% of 50MA (`ext_pct <= 20`)
3. RSI at entry: 42–65 for BUY, 35–58 for SELL
4. ADX ≥ 20 (trending market)
5. R/R ≥ 1.5 (T2 vs stop)
6. Vol ratio ≥ 1.2 (some volume confirmation)
7. NOT within 1 week of earnings
8. At least 1 TRIGGER signal active

**Soft filters (penalise score if failing):**
- RS rating ≥ 60 for BUY
- Minervini ≥ 6/8
- Not near 52W high (within 2%) unless volume surges 2×

## Exit Logic

**Auto-exits (bracket orders via Alpaca):**
- Stop: Entry − 1.5 × ATR(14)
- T2 target: Entry + 2.0 × ATR(14)

**Signal-based exits (monitor loop):**
- URGENT: EMA9 cross + MACD cross + below EMA50 (any 2 of 3)
- TIME: close after max_hold_days (default 20)

## Data Needed

- Daily OHLCV via yfinance (already integrated)
- Universe: ~600 symbols (SP500 + NDQ100 + High-Growth + Trending)
- Backtest period: 2 years minimum (2023-01-01 to 2025-12-31)

## Guardrails to Backtest

| Filter | Value | Hypothesis |
|--------|-------|------------|
| max ext_pct | 20% | Reduces snap-back risk |
| RSI cap | 65 | Avoids overbought entries |
| min R/R | 1.5 | Only take positive-expectancy setups |
| min score | 8 | STRONG conviction only |
| min ADX | 20 | Trend must be established |
| min vol ratio | 1.2 | Some institutional participation |

## Notes

- Current paper account: $95,329 equity, -$1,768 unrealized (-3.54%)
- ARM position is the main drag (-11.1%) — perfect case study for guardrails
- Backtest should simulate: signal → next-open entry → ATR-based stop/T2 exit
- Benchmark: SPY buy-and-hold over same period

---
*Run `/cbt:discover` to formalize into complete strategy specification.*
