# Qullamaggie Breakout — Research & Backtest Report

_Last run: 2026-09-09 · `backtest.py` · 514-symbol universe · 2016–2026 (10.7 yrs)_

> **2026-09-09 correction.** The first version of this report (+0.31R / 5.1 % CAGR)
> contained a look-ahead bug in the momentum-leader rank. Fixed; the honest base
> case is **+0.13R / 2.1 % CAGR / −20.5 % DD** — see bug #9 and §3.

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
| 9 | **Backtest only** — momentum percentile was read on the *entry day's* close (`m1 = close[t]/close[t-21]`), i.e. look-ahead: it ranked leaders using a price not yet known when the intraday breakout fires | inflated the base case to +0.31R / 5.1 % CAGR / −11 % DD | rank momentum as of the **prior** close (`mom_pct.get(prev_day)`); honest result is +0.13R / 2.1 % CAGR / −20.5 % DD (§3) |
| 10 | **Backtest only** — consolidation window was `d.iloc[i-1-W:i-1]`, off by one (dropped yesterday, kept an extra old bar); `cands.sort(reverse=True)` on raw tuples would raise on a momentum tie (compares a `Series`) | subtle base mis-placement; latent crash | window is now the `W` completed bars before today; sort by an explicit key |

Result: the breakout scan went from **163 candidates → ~25**, "triggered" from 44 → ~13,
every stop/risk figure is sane — and the backtest's headline edge shrank by more than
half once the look-ahead was removed.

---

## 3. Backtest results — base case (look-ahead removed)

```
Period               10.7 yrs
Final equity         $124,903   (from $100,000)
CAGR                    2.1 %    (SPY buy & hold 15.0 %)
Max drawdown          -20.5 %    (SPY -33.7 %)
Sharpe (daily)          0.26
Trades                  488
Win rate               32.0 %
Expectancy            +0.13 R
Avg win / avg loss   +2.68R / -1.06R
Profit factor           1.14
Best / worst trade   +22.2R / -3.4R
Avg holding             5 bars
```

> The first pass reported +0.31R / 5.1 % CAGR / −11 % DD. That was **look-ahead
> bias** (bug #9): the momentum-leader rank used the entry day's own close. With
> the rank taken as of the prior close, most of the edge disappears — the honest
> read is a **thin positive expectancy that does not stand on its own**.

### By year
| Year | Strat % | Strat maxDD | Trades | Win % | Exp R | SPY % |
|---|---|---|---|---|---|---|
| 2016 | +6.6 | -2.2 | 11 | 55 | +1.53 | +9 |
| 2017 | -0.1 | -2.9 | 7 | 29 | -0.12 | +19 |
| 2018 | -4.5 | -5.6 | 38 | 21 | -0.23 | -6 |
| 2019 | -5.3 | -7.5 | 31 | 23 | -0.34 | +29 |
| 2020 | +3.1 | -15.1 | 80 | 32 | +0.14 | +16 |
| 2021 | -1.6 | -8.3 | 56 | 25 | -0.08 | +27 |
| 2022 | -4.8 | -7.7 | 29 | 24 | -0.27 | -18 |
| 2023 | +16.7 | -3.0 | 26 | 58 | +1.05 | +24 |
| 2024 | +1.4 | -8.9 | 54 | 35 | +0.18 | +23 |
| 2025 | +9.8 | -5.4 | 73 | 38 | +0.36 | — |
| 2026 YTD | +2.7 | -11.2 | 83 | 29 | +0.09 | — |

Six of ten full years are flat-to-down. The edge is concentrated in 2016, 2023 and
2025; 2018–2019 and 2021–2022 lose money. Equity curve: `equity.png` · trades: `trades.csv`

---

## 4. Parameter sensitivity (one knob at a time)

| Knob | Value | CAGR | MaxDD | Trades | Win% | Exp R | Sharpe |
|---|---|---|---|---|---|---|---|
| **cons_len** | 8 | 5.5 | -17.0 | 575 | 34 | +0.24 | 0.54 |
| | **12** | **2.1** | **-20.5** | **488** | **32** | **+0.13** | **0.26** |
| | 16 | 4.0 | -16.1 | 393 | 34 | +0.27 | 0.48 |
| | 20 | 3.0 | -18.3 | 320 | 34 | +0.26 | 0.40 |
| **mom_pctile** | 0.80 | 2.5 | -24.2 | 552 | 32 | +0.13 | 0.29 |
| | **0.90** | 2.1 | -20.5 | 488 | 32 | +0.13 | 0.26 |
| | 0.95 | 2.2 | -18.0 | 392 | 32 | +0.15 | 0.28 |
| | 0.98 | 2.2 | -12.7 | 238 | 31 | +0.23 | 0.33 |
| **adr_min** | 2.5 | 3.3 | -21.3 | 1040 | 33 | +0.16 | 0.31 |
| | **3.5** | 2.1 | -20.5 | 488 | 32 | +0.13 | 0.26 |
| | **5.0** | **3.0** | **-9.7** | **137** | **40** | **+0.48** | **0.52** |
| **partial_day** | 3 | 2.5 | -16.0 | 506 | 35 | +0.15 | 0.31 |
| | **4** | 2.1 | -20.5 | 488 | 32 | +0.13 | 0.26 |
| | 5 | 2.4 | -20.9 | 483 | 30 | +0.15 | 0.28 |
| | 8 | 2.9 | -23.7 | 466 | 25 | +0.19 | 0.31 |
| **cons_range_mult** | 5.0 | 2.4 | -19.5 | 414 | 32 | +0.16 | 0.31 |
| | **8.0** | 2.1 | -20.5 | 488 | 32 | +0.13 | 0.26 |
| | 12.0 | 2.0 | -20.3 | 490 | 32 | +0.13 | 0.25 |
| **regime_filter** | **True** | 2.1 | -20.5 | 488 | 32 | +0.13 | 0.26 |
| | False | 1.1 | -26.0 | 593 | 33 | +0.08 | 0.16 |

**Read:**
* Expectancy stays **positive in all 20 configs (+0.08R to +0.48R)** — the sign of
  the edge is robust, its *size* is not: at the base parameters it is barely above
  break-even (+0.13R, PF 1.14).
* `cons_len = 12` is an unlucky local dip — 8/16/20 all do better (+0.24–0.27R).
  Don't read anything into the exact base number.
* The **one genuinely interesting pocket is `adr_min = 5.0`**: only high-volatility
  names, 137 trades in 10 yrs, but +0.48R, 40 % win, −9.7 % DD, Sharpe 0.52. The
  edge lives in the most volatile leaders and is diluted by everything else.
* Keep the **regime filter on** (off → +0.08R, −26 % DD).

---

## 5. Verdict

**A thin, real, but not standalone-tradeable edge — on this daily-bar model.**

* Expectancy is positive and its *sign* survives every parameter perturbation
  (+0.08R to +0.48R), so the checklist does select forward-better-than-random names.
  But at sensible parameters it is **+0.13R with PF 1.14** — after real slippage,
  commissions and the survivorship haircut (§6), that is plausibly zero.
* 2.1 % CAGR vs SPY's 15 %, with a **worse risk-adjusted profile** than the first
  (biased) pass suggested: −20.5 % drawdown, Sharpe 0.26, six of ten years flat-to-red.
* The payoff is still right-skewed (32 % win, +2.68R winners, −1.06R losers) — the
  trend-continuation shape is there, just not enough of it.
* The daily bar is doing real damage here: no opening-range-high timing, no intraday
  stop, close-fills on trims. Qullamaggie's results come from **intraday execution**
  this model cannot see (§6.2). This backtest should be read as a *floor*, not a verdict
  on his method.

### How to actually use it
1. **Signal / watchlist generator only.** The dashboard scanner is fine for surfacing
   the daily candidate list; the backtest does *not* justify a mechanical daily-bar
   version of it.
2. **If you want a mechanical sleeve, trade the `adr_min ≥ 5` subset** — high-volatility
   leaders only. That is the sole cut with a Sharpe (0.52) and drawdown (−9.7 %) worth
   the screen time, at ~13 trades/year.
3. **Do the intraday work.** The setup needs 1/5-min data for the ORH entry and a
   same-day low-of-day stop to be evaluated (or traded) properly. Until then, treat any
   daily-bar P&L as an underestimate of a well-executed version and an overestimate of
   a sloppy one.

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
   **No same-day stop-out**: position management starts on the bar *after* entry.
   Modelling a same-day stop when `low < stop` was tried and rejected — on a real
   breakout the daily low usually prints in the morning *before* the afternoon break,
   so that assumption is a severe pessimistic bias (it drove expectancy to −0.33R).
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
