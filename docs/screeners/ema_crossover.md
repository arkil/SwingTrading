# EMA Crossover Screener

**File:** `ema_crossover_screener.py`  
**Dashboard:** `?scanner=ema`  
**Sources:** QuantifiedStrategies, StockCharts, OpoFinance, AlphaExCapital EMA crossover research

---

## What It Does

Scans for EMA crossover entries with a full filter stack — trend filter, ADX confirmation, volume confirmation, and RSI range guard — to eliminate the low-quality crossovers that plague naive implementations. Two presets: aggressive (9/21) and conservative (20/50).

---

## Primary Preset — 9/21 EMA with 55 EMA Trend Filter

### Bullish Entry (all 5 must pass)
1. EMA-9 crosses **above** EMA-21 (confirmed on close)
2. Price close **above** EMA-55 (trades in direction of trend)
3. ADX(14) > 20 (market is trending, not ranging)
4. Volume > 1.5× 20-bar average (institutional participation)
5. RSI(14) between 30 and 70 (avoids overbought entries, not deeply oversold)

### Bearish Entry (mirror)
1. EMA-9 crosses **below** EMA-21
2. Price close **below** EMA-55
3. ADX(14) > 20
4. Volume > 1.5× 20-bar average
5. RSI(14) between 30 and 70

---

## Secondary Preset — Conservative 20/50/200

- Fast = 20, Slow = 50, Trend = 200 EMA
- Same filter stack
- Fewer signals, higher win rate
- Best for large-caps and indices (SPY, QQQ)
- Only take longs when price > EMA-200, shorts when < EMA-200

---

## Pullback Mode (optional — higher quality entries)

Instead of entering on the crossover bar, wait for:
1. EMA-9 has crossed EMA-21 (crossover already happened)
2. Price pulls back and touches/closes near EMA-21
3. Next candle closes back in the trend direction

Reduces whipsaws significantly by avoiding immediate entries into already-extended crossovers.

---

## Trade Levels

| Level | Formula |
|-------|---------|
| Stop | Entry ± 1.5 × ATR(14) |
| Target 1 | Entry ± 2.0 × ATR(14) — partial exit (1.33:1 R/R) |
| Target 2 | Entry ± 3.0 × ATR(14) — full exit (2.0:1 R/R) |

---

## Output Columns

| Column | Description |
|--------|-------------|
| Ticker | Symbol |
| Signal | BULLISH_CROSS / BEARISH_CROSS |
| Preset | 9/21/55 or 20/50/200 |
| EMA Fast / Slow / Trend | Current EMA values |
| ADX | ADX(14) value |
| RSI | RSI(14) value |
| Vol vs Avg | Volume vs 20-bar average |
| Above EMA200 | 200 EMA alignment |
| Pullback Mode | Signal is a pullback re-entry (not fresh cross) |
| Entry / Stop / T1 / T2 | Trade levels |

---

## How to Run

```bash
# Default universe, 9/21 preset
python ema_crossover_screener.py

# S&P 500
python ema_crossover_screener.py --universe sp500

# Conservative preset (20/50/200)
python ema_crossover_screener.py --preset conservative

# Bullish only
python ema_crossover_screener.py --direction bull

# Pullback mode (higher quality entries)
python ema_crossover_screener.py --pullback

# Both indices, bullish, pullback mode
python ema_crossover_screener.py --universe both --direction bull --pullback

# Specific tickers
python ema_crossover_screener.py --tickers NVDA AAPL TSLA META
```

---

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--universe` | `default` | `default` / `sp500` / `nasdaq100` / `both` / `watchlist` |
| `--tickers` | — | Override with specific symbols |
| `--preset` | `aggressive` | `aggressive` (9/21/55) or `conservative` (20/50/200) |
| `--direction` | both | `bull` / `bear` / `both` |
| `--pullback` | false | Enable pullback mode (wait for retest of slow EMA) |
| `--adx-min` | 20 | Minimum ADX for trend confirmation |
| `--vol-mult` | 1.5 | Volume confirmation multiplier |
| `--output` | — | Save to CSV |
