# Qullamaggie Breakout — Research & Backtest Report

_Last run: 2026-09-08 · `backtest.py` · 514-symbol universe · 2016–2026 (10.7 yrs)_

---

## 1. The methodology (from qullamaggie.com)

Kristjan Kullamägi ("Qullamaggie") trades three setups. This project implements and
backtests the **Breakout**; the dashboard scanner also covers the **Episodic Pivot**.

### Breakout — the rules as stated
| Element | Qullamaggie's rule | How we model it (daily bars) |
|---|---|---|
| Universe | "the 1 or 2 % of stocks up the most over 1-, 3- **and** 6-month" | cross-sectional percentile rank of `max(1m,3m,6m) return`, keep ≥ 90th pct |
| Prior move | 30–100 %+ over days-to-weeks | implied by the momentum-percentile filter |
| Trend | price "surfing" a rising 10 > 20 > 50-day MA | `close > SMA10 > SMA20 > SMA50`, each SMA rising over 5 days |
| Location | near the highs | within 25 % of the 52-week high |
| Base | "2 weeks to 2 months, higher lows and tightening range" | 12-day window: regression slope of lows ≥ 0, recent-half range < prior-half range, base volume < pre-base volume |
| Entry | break of the consolidation high on the opening-range (1/5/60-min) high | the day price trades through the base's **pivot high**; fill at `max(open, pivot)` + 5 bps |
| Stop | "lows of the day … not wider than the ADR/ATR of the stock" | `entry − min(entry−day_low, 1·ADR)`, floored at 0.5·ADR |
| Trim | "sell 1/3–1/2 after 3–5 days, move stop to break-even" | sell 50 % at the close of day 4, stop → entry |
| Trail | "trail with the 10- or 20-day MA … first CLOSE below the 10-day" | exit the remainder on the first close < SMA10 |
| Risk | "0.25–1 % per trade, rarely > 1 %; positions 10–20 %; ≤ 30 % overnight" | 0.5 % equity risk/trade, position ≤ 20 % equity, no leverage |
| Regime | bullish markets pay 10–20R; he sizes down in bad tape | new entries only while SPY > its 200-day SMA (toggle) |

### Episodic Pivot (scanner only, not backtested)
Gap ≥ 10 % on ≥ 3× average volume, out of a quiet non-extended base (6-mo move
< 100 %), catalyst-driven (earnings/FDA/regulatory). Entry = opening-range high
(daily proxy: gap-day high), stop = low of day, same trail. Not backtested here
because it needs the catalyst + intraday data to model honestly.

---

## 2. Bugs found & fixed in the dashboard scanner

| # | Bug | Effect | Fix |
|---|---|---|---|
| 1 | Pivot high was `max(High)` over a window that **included the current bar** | on any new-high day `today_high ≥ pivot` was trivially true → "🔥 TRIGGERED" for ~160/750 names | pivot now taken from the `N` bars strictly **before** today |
| 2 | Momentum gate was an absolute `%` threshold | let through hundreds of mid-momentum names; not "top 1–2 %" | cross-sectional **percentile rank** across the scanned universe |
| 3 | "Tight flag" test was an arbitrary `range ≤ ADR·len·0.6` | passed loose, sloppy bases | higher-lows (regression slope ≥ 0) **and** tightening (recent half < prior half) **and** drying volume |
| 4 | Stop = `min(low of last 2 days, flag low)` — could sit **at or above entry** | negative "risk %" (e.g. VLO −0.5 %), absurd share counts | stop clamped to 0.5–1.0 ADR and always strictly below entry |
| 5 | `triggered` used `price ≥ pivot·0.985` (circular with bug 1) | see #1 | fresh-breakout logic: `prev_close < pivot ≤ today_high`, plus an `⏰ EXTENDED` state when price is already > 1 ADR past the pivot |
| 6 | EMA 10/20 for the trend stack | Qullamaggie's charts use simple 10/20/50 | switched to SMA 10/20/50 |
| 7 | Table `.format()` strings applied to possibly-empty cells | Streamlit `ValueError` risk | numeric-guarded formatter lambdas |
| 8 | EP: `🔥 GAP TODAY` shown even when price had faded far below the entry | misleading — no clean entry left | added `📉 GAP FADED` when `price < entry·0.97` |

Result: the breakout scan went from **163 candidates → ~25**, "triggered" from 44 → ~13,
and every stop/risk figure is now sane.

---

## 3. Backtest results — base case

```
Period               10.7 yrs
Final equity         $169,859   (from $100,000)
CAGR                    5.1 %    (SPY buy & hold 15.0 %)
Max drawdown          -11.2 %    (SPY -33.7 %)
Sharpe (daily)          0.57
Trades                  422
Win rate               34.1 %
Expectancy            +0.31 R
Avg win / avg loss   +2.94R / -1.05R
Profit factor           1.36
Best / worst trade   +22.1R / -4.2R
Avg holding             5 bars
```

### By year
| Year | Strat % | Strat maxDD | Trades | Win % | Exp R | SPY % |
|---|---|---|---|---|---|---|
| 2016 | +3.9 | -2.2 | 12 | 42 | +0.74 | +9 |
| 2017 | +0.2 | -2.9 | 6 | 33 | +0.03 | +19 |
| 2018 | +6.8 | -3.0 | 30 | 30 | +0.55 | -6 |
| 2019 | -1.6 | -4.7 | 23 | 26 | -0.14 | +29 |
| 2020 | +20.0 | -8.8 | 66 | 36 | +0.67 | +16 |
| 2021 | +3.6 | -7.5 | 53 | 28 | +0.20 | +27 |
| 2022 | -3.5 | -6.0 | 30 | 27 | -0.18 | -18 |
| 2023 | +6.3 | -4.5 | 20 | 60 | +0.68 | +24 |
| 2024 | +14.9 | -5.8 | 46 | 35 | +0.64 | +23 |
| 2025 | +2.5 | -7.5 | 65 | 37 | +0.14 | — |
| 2026 YTD | +2.7 | -9.9 | 71 | 32 | +0.10 | — |

Equity curve: `equity.png` · every trade: `trades.csv`

---

## 4. Parameter sensitivity (one knob at a time)

| Knob | Value | CAGR | MaxDD | Trades | Win% | Exp R | Sharpe |
|---|---|---|---|---|---|---|---|
| **cons_len** | 8 | 4.2 | -11.7 | 485 | 35 | +0.23 | 0.49 |
| | **12** | **5.1** | **-11.2** | **422** | **34** | **+0.31** | **0.57** |
| | 16 | 4.8 | -12.9 | 327 | 37 | +0.36 | 0.60 |
| | 20 | 5.2 | -7.9 | 262 | 38 | +0.48 | 0.68 |
| **mom_pctile** | 0.80 | 5.6 | -15.9 | 484 | 34 | +0.30 | 0.60 |
| | **0.90** | 5.1 | -11.2 | 422 | 34 | +0.31 | 0.57 |
| | 0.95 | 4.9 | -11.8 | 350 | 34 | +0.36 | 0.59 |
| | 0.98 | 4.8 | -11.2 | 209 | 35 | +0.56 | 0.74 |
| **adr_min** | 2.5 | 5.8 | -15.0 | 906 | 33 | +0.23 | 0.53 |
| | **3.5** | 5.1 | -11.2 | 422 | 34 | +0.31 | 0.57 |
| | 5.0 | 3.0 | -8.8 | 136 | 37 | +0.50 | 0.56 |
| **partial_day** | 3 | 6.2 | -11.5 | 438 | 37 | +0.36 | 0.70 |
| | **4** | 5.1 | -11.2 | 422 | 34 | +0.31 | 0.57 |
| | 5 | 5.6 | -11.3 | 419 | 33 | +0.34 | 0.60 |
| | 8 | 6.0 | -14.0 | 406 | 28 | +0.37 | 0.61 |
| **cons_range_mult** | 5.0 | 4.9 | -11.7 | 346 | 34 | +0.36 | 0.61 |
| | **8.0** | 5.1 | -11.2 | 422 | 34 | +0.31 | 0.57 |
| | 12.0 | 5.0 | -11.2 | 423 | 34 | +0.31 | 0.57 |
| **regime_filter** | **True** | 5.1 | -11.2 | 422 | 34 | +0.31 | 0.57 |
| | False | 5.8 | -13.6 | 520 | 35 | +0.29 | 0.60 |

Dropping the SPY-200 **regime filter** raises CAGR to 5.8 % but deepens the drawdown
to -13.6 % and adds 100 marginal trades at lower expectancy — keep the filter on.
`cons_range_mult` is inert between 8 and 12 (the higher-lows / tightening tests already
bind first).

**Read:** expectancy is **positive in every single configuration** (+0.23R to +0.56R).
The trade-off is monotonic — more selective (longer base, higher momentum percentile,
higher ADR floor) → higher expectancy and Sharpe, fewer trades, similar-or-lower CAGR.
Nothing is fragile; there is no cliff.

---

## 5. Verdict

**The edge is real but small, and the standalone equity curve lags the index.**

* Positive, statistically meaningful expectancy (+0.31R over 422 trades, robust to
  every parameter) with a **right-skewed** payoff — 34 % win rate, +2.94R average
  winner, capped ~1R losers. This is a textbook trend-continuation profile.
* Drawdown is **1/3 of buy-and-hold** (-11 % vs -34 %) and the worst years are shallow
  (2022: -3.5 % while SPY -18 %).
* **But** CAGR is only ~5 % vs SPY's 15 %, because the portfolio is ~80 % in cash:
  0.5 % risk/trade + a 5-bar average hold means rarely more than 1–2 positions on at
  once. The system extracts its edge in bursts (2020 +20 %, 2024 +15 %) and treads
  water the rest of the time, missing the bull runs of 2017/2019/2021.

### How to actually use it
1. **Signal generator (recommended).** Use the dashboard scanner for the daily
   candidate list; size discretionarily. The backtest confirms the checklist selects
   names with a genuine forward edge.
2. **Low-drawdown satellite sleeve.** Allocate 20–40 % of capital; accept the lower
   absolute return for the drawdown protection and low correlation to a core index
   position.
3. **Scale up risk** to 1–1.5 %/trade and raise `max_positions` if you will tolerate
   ~20–25 % drawdowns — moves CAGR toward the low teens but no longer "sleep at night".
4. Best single tweak found: **`partial_day = 3`** (trim on day 3, not 4) → CAGR 6.2 %,
   Sharpe 0.70, same drawdown.

---

## 6. Caveats (do not skip)

1. **Survivorship bias — inflates results.** The universe is *today's* S&P 500 +
   Nasdaq 100. Names that blew up and were delisted never enter the test. Real-world
   returns would be **lower** than shown.
2. **No intraday data.** Qullamaggie enters on the 1/5/60-min opening-range high; we
   proxy with the daily pivot and fill at `max(open, pivot)`. This misses the "wait
   for the ORH" timing that both cuts losers faster and gets worse fills on gaps.
3. **Partial & trail fills are modelled at the close**, not intraday into strength —
   mildly optimistic on the trims, mildly pessimistic on the trails.
4. Slippage 5 bps/side; **no commissions, no hard-to-borrow, no dividends on the
   short side** (breakout is long-only so the latter two don't bite here).
5. EP setup is **not** backtested — needs catalyst + premarket volume data.
6. 2015 is warm-up only (indicators need ~1 yr); "10.7 yrs" effectively starts 2016.

---

## 7. Files

| File | What |
|---|---|
| `backtest.py` | the simulator — `python backtest.py [--sensitivity] [--no-regime] [--risk 0.01] [--refresh]` |
| `data/prices.pkl` | cached daily OHLCV (delete or `--refresh` to rebuild) |
| `trades.csv` | every trade: entry/exit dates & prices, bars held, exit reason, R multiple, P&L |
| `equity.csv` / `equity.png` | daily equity curve vs SPY + drawdown |
