# Strategy Discovery: cheap_calls_weekly_0_7dte

**Date:** 2026-06-12
**Phase:** Discovery Complete
**Engine:** fast (Polars + NumPy + Numba)
**Project Type:** indicator

---

## Core Hypothesis

When the cheap_calls / weekly_opts scanner fires on a ticker — low-premium OTM call,
momentum confirmation, stock outperforming SPY — the underlying move over the next
0–7 days is large enough to overcome theta decay and make the call profitable.
The GO Score (0–10) should act as a quality filter: higher scores should produce
meaningfully better win rates and expectancy.

### Market Behavior Exploited
Short-dated OTM calls have convex payoff profiles. A 3–5% move in the underlying
can triple or quadruple the option value. If the scanner correctly identifies
pre-move setups (volume surge, relative strength, RSI in momentum zone), the
win rate × avg winner should exceed the high theta cost of 0–7 DTE options.

### Theoretical Basis
- Volume surges ahead of moves are well-documented (order-flow leading price)
- Relative strength vs SPY filters for stocks with sector/idiosyncratic tailwinds
- RSI 45–65 captures mid-trend, avoiding overbought exhaustion entries
- The 0–7 DTE window forces a quick verdict — no slow bleed, quick resolution

---

## Entry Conditions

| Condition | Description | Data Required |
|-----------|-------------|---------------|
| Premium ≤ $1.50/share | Max $150 per contract | BS-simulated option price |
| Delta 0.05–0.45 | OTM to near-ATM calls only | BS delta |
| Strike within ±15% of spot | Moneyness filter | OHLCV close |
| Rel strength vs SPY ≥ 0 | Stock not lagging market | OHLCV + SPY close |
| Volume surge ≥ 1.3× 3M avg | Above-average activity | OHLCV volume |
| RSI-14 in 45–65 | Mid-trend, not exhausted | OHLCV close |
| Simplified GO Score ≥ threshold | Composite conviction filter | See below |

### GO Score (OHLCV-derivable components only)

The live scanner's full GO Score (0–10) includes V/OI and P/C ratios from the
options chain. Historical chain data is unavailable without Polygon. The backtest
uses a **simplified GO Score (0–5)** from OHLCV only:

| Component | Signal | Points |
|-----------|--------|--------|
| Rel strength vs SPY | ≥ 2.0% outperformance | 2 |
| Rel strength vs SPY | ≥ 0.5% outperformance | 1 |
| Volume surge vs 3M avg | ≥ 2.0× | 2 |
| Volume surge vs 3M avg | ≥ 1.3× | 1 |
| RSI-14 in 45–65 | In range | 1 |

**Max: 5 pts.** The sweep will test thresholds 0, 1, 2, 3, 4, 5.

### Entry Signal Logic
```
for each trading day D, for each ticker T in universe:
    compute: rel_str = T.chg_pct(D) - SPY.chg_pct(D)
    compute: vol_surge = T.volume(D) / T.avg_volume_63d(D)
    compute: rsi14(D)
    compute: simplified_go_score(D)

    if go_score >= threshold:
        for dte_target in [0, 1, 2, 3, 5, 7]:
            find nearest expiry at dte_target calendar days out
            price OTM call at spot * 1.05 via Black-Scholes
              (IV = 30d historical vol × 1.2 scaling factor)
            if premium <= $1.50 and delta in [0.05, 0.45]:
                enter trade at open next day D+1
```

---

## Exit Conditions

### Take Profit
Option value doubles (100% gain on premium paid). Evaluated at each day's close
using BS repricing with updated spot price and reduced DTE.

### Stop Loss
Option loses 50% of premium paid. Evaluated daily at close.

### Time Exit
Hold to expiry. At expiry: intrinsic value = max(spot - strike, 0).

### Priority
Stop → Target → Expiry (checked in this order each day).

---

## Data Requirements

| Dataset | Resolution | Source | Size Estimate | Status |
|---------|------------|--------|---------------|--------|
| OHLCV universe (~130 tickers) | Daily | yfinance | ~65k rows | Auto-fetched |
| SPY daily | Daily | yfinance | ~600 rows | Auto-fetched |
| Simulated option prices | Derived | Black-Scholes | — | Computed |

### Data Scale
- **Estimated rows:** ~65,000 (small dataset)
- **Engine:** fast (Polars + NumPy + Numba — chosen by user; overkill for this size but zero downside)
- **Rationale:** Fast engine adds no complexity and future-proofs if universe expands

### Data Validation Checklist
- [ ] No gaps in timestamps
- [ ] Adjusted closes used (splits/dividends)
- [ ] ≥ 252 trading days of history per ticker
- [ ] SPY available for all dates in universe

---

## Build Plan

**Complexity Level:** Medium (multi-feature scoring + option simulation + 2D param sweep)

| Step | Description | Output |
|------|-------------|--------|
| 1 | Fetch & cache daily OHLCV for full universe + SPY via yfinance | `Data/ohlcv_universe.parquet`, `Data/spy_daily.parquet` |
| 2 | Feature engineering: RSI-14, vol_ratio (63-day avg), rel_str vs SPY, EMA-20 | Features added to OHLCV dataframe |
| 3 | Signal generation: compute simplified GO Score per ticker per day | `signals.parquet` |
| 4 | Option simulation: BS-price OTM call for each signal day × DTE target | `simulated_options.parquet` |
| 5 | Trade simulation: enter next open, exit on 50% stop / 100% target / expiry | `trades.parquet` |
| 6 | 2D sweep: GO Score threshold (0–5) × DTE at entry (0,1,2,3,5,7) | `experiments/sweep_results.parquet` |
| 7 | Analysis: win rate, avg W/L, expectancy, equity curve per cell in grid | `experiments/sweep_summary.csv` |
| 8 | Plots: heat map (GO Score × DTE → win rate), equity curve for best config | `plots/` |

---

## Success Criteria

- [ ] Win rate > 40% at the best GO Score threshold
- [ ] Avg winner ≥ 2× avg loser (positive expectancy at ~33% win rate)
- [ ] GO Score ≥ 3 outperforms GO Score = 0 (filter adds value)
- [ ] At least 1 DTE tier shows positive expectancy

## Account Rules

**Account Type:** Personal — no external rules applied.

---

## Kill Criteria

Abandon strategy if:
- [ ] Win rate < 30% across all GO Score thresholds (even unfiltered entries lose)
- [ ] Avg winner < 1.5× avg loser (theta decay dominates even on winners)
- [ ] GO Score filter adds no lift — threshold 0 and threshold 4 perform equally
- [ ] Every DTE tier loses money (move timing is wrong, not just filter quality)

---

## Important Caveats

1. **Simplified GO Score:** The backtest can only score rel_str + vol_surge + RSI (0–5 pts). The live scanner also scores V/OI ratio and P/C ratio (options flow) — 5 more pts. The backtest will be a lower bound on live scanner quality.
2. **IV assumption:** Using 30d historical vol × 1.2 as IV proxy. Actual IV on signal days may be higher (event premium) or lower. Sensitivity test on IV scaling factor warranted.
3. **Execution:** Entering at open the day after signal. Live trading enters intraday. Slippage of $0.05/share assumed.
4. **No bid/ask spread on options:** BS price used as mid. Real fills will be worse. Results are optimistic by ~5–10%.

---

## Questions for Research Phase

1. What is the empirically observed win rate for buying OTM calls 0–7 DTE on high-volume breakout days?
2. Does RSI 45–65 at entry materially affect short-dated call outcomes vs RSI outside that range?
3. What IV scaling factor best approximates real short-dated IV vs 30d historical vol?

---

*Generated by CBT Framework /cbt:discover*
