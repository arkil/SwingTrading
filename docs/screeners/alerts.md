# Daily Alerts Engine

**File:** `alerts_engine.py`  
**Dashboard:** `?scanner=alerts`  
**Live runner:** `alerts_live_runner.py`  
**Paper trader:** `stock_paper_trader.py`

---

## What It Does

Runs every stock through 7 indicator systems simultaneously and produces a single composite score (0–12) with a BUY / SELL / WATCH verdict plus precise ATR-based entry, stop, and three targets. It is the central signal source that feeds both the dashboard and the live Alpaca paper trader.

---

## Scoring System (0–12 per direction)

| Component | Max Points | What's Measured |
|-----------|-----------|-----------------|
| TREND | 3 | EMA stack (9/21/50/200), Minervini 8-condition template |
| MOMENTUM | 3 | RSI(14) zone/cross/divergence, MACD histogram, RS rating vs SPY |
| TRIGGER | 3 | At least 1 required; caps at 3. 52W-high breakout, NR7, Bollinger squeeze, inside-bar, MA-reclaim, Livermore pivot, gap continuation |
| VOLUME | 2 | 1.5× avg = +1; 3× avg = +2 |
| EXTRA | 1 | Livermore pivot bonus or CAN SLIM fundamental bonus |

**Verdict thresholds:**

| Score | Verdict |
|-------|---------|
| ≥ 8 | **STRONG** BUY / SELL |
| ≥ 6 | **BUY** / SELL |
| ≥ 4 | **WATCH** |
| < 4 | no alert |

---

## Signal Sources Detail

### Trend (0–3)
- **+1** — EMA9 > EMA21 > EMA50 (short-term stack aligned)
- **+1** — EMA50 > EMA200 (medium-term trend up)
- **+1** — Minervini trend template score ≥ 6/8

### Momentum (0–3)
- **+1** — RSI in bullish zone (45–70) or recent oversold bounce
- **+1** — MACD histogram positive and expanding
- **+1** — RS Rating vs SPY > 60th percentile

### Trigger (0–3, at least 1 required)
Any of:
- 52-week high close
- NR7 volatility expansion
- Bollinger Band squeeze breakout
- Inside bar close outside mother bar
- 50-day EMA reclaim with volume
- Livermore upward/downward pivotal point
- Gap-up or gap-down continuation

### Volume (0–2)
- **+1** — Today's volume ≥ 1.5× 20-day average
- **+2** — Today's volume ≥ 3× 20-day average (replaces +1)

### Extra (0–1)
- **+1** — Livermore pivot confirmed on this bar OR CAN SLIM score ≥ 10

---

## Price Levels (ATR-based)

All levels derived from ATR(14):

| Level | Formula | Use |
|-------|---------|-----|
| Entry | Current close | Market-on-close entry |
| Stop | Entry − 1.5 × ATR | Hard stop, position sizing |
| T1 | Entry + 1.0 × ATR | Quick partial exit (take 1/3) |
| T2 | Entry + 2.0 × ATR | Main target (take 1/3) |
| T3 | Entry + 3.0 × ATR | Runner — let ride |
| R/R | T2 dist / Stop dist | Should be ≥ 1.5 to take trade |

---

## How to Run

The engine is primarily used programmatically by the dashboard and live runner, but can be called directly:

```python
from alerts_engine import run_alerts

alerts = run_alerts(
    tickers=["NVDA", "AAPL", "TSLA"],
    min_score=6,        # only BUY and STRONG
    max_workers=8,
)
print(alerts)
```

**Via dashboard:** `http://localhost:8501/?scanner=alerts`

**Live on Alpaca paper:**
```bash
python alerts_live_runner.py       # watches for signals and places orders
python stock_paper_trader.py       # daemon mode (runs continuously)
```

---

## Live Trading Integration

The live runner (`alerts_live_runner.py`) polls the engine every market hour, filters signals by `daily_alerts_guardrails` parameters (score ≥ 8, ext_pct ≤ 30%, RSI ≤ 70, ADX ≥ 20), then submits bracket orders to Alpaca with the ATR stop and T2 target.

See `configs/alerts_live.yaml` for live config and `strategies/daily_alerts_guardrails/` for the guardrail backtest.
