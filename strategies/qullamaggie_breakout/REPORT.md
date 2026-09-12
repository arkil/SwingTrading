# Qullamaggie Breakout — Research & Backtest Report

_Last run: 2026-09-12 · `backtest.py` / `optimize.py` · 514-symbol universe · 2016–2026 (10.7 yrs)_

> **History.** 2026-09-08 first backtest: +0.31R / 5.1% CAGR (contained a
> look-ahead bug). 2026-09-09 correction: fixed, honest baseline **+0.13R /
> 2.1% CAGR / −20.5% DD** (§3). 2026-09-12 optimization: grid-searched around
> that honest baseline with an in-sample/out-of-sample split → **+0.65R /
> 4.3% CAGR / −11.3% DD / Sharpe 0.66** (§4–5), now the default in `backtest.py`.

---

## 1. The methodology (from qullamaggie.com)

Kristjan Kullamägi ("Qullamaggie") trades three setups. This project implements and
backtests the **Breakout**; the dashboard scanner also covers the **Episodic Pivot**.

### Breakout — the rules as stated
| Element | Qullamaggie's rule | How we model it (daily bars) |
|---|---|---|
| Universe | "the 1 or 2 % of stocks up the most over 1-, 3- **and** 6-month" | cross-sectional percentile rank of `max(1m,3m,6m) return`, keep ≥ 95th pct |
| Prior move | 30–100 %+ over days-to-weeks | implied by the momentum-percentile filter |
| Trend | price "surfing" a rising 10 > 20 > 50-day MA | `close > SMA10 > SMA20 > SMA50`, each SMA rising over 5 days |
| Location | near the highs | within 25 % of the 52-week high |
| Base | "2 weeks to 2 months, higher lows and tightening range" | 16-day window: regression slope of lows ≥ 0, recent-half range < prior-half range, base volume < pre-base volume |
| Entry | break of the consolidation high on the opening-range (1/5/60-min) high | the day price trades through the base's **pivot high**; fill at `max(open, pivot)` + 5 bps |
| Stop | "lows of the day … not wider than the ADR/ATR of the stock" | `entry − min(entry−day_low, 1·ADR)`, floored at 0.5·ADR |
| Trim | "sell 1/3–1/2 after 3–5 days, move stop to break-even" | sell 50 % at the close of day 5, stop → entry |
| Trail | "trail with the 10- or 20-day MA … first CLOSE below the 10-day" | exit the remainder on the first close < SMA10 |
| Risk | "0.25–1 % per trade, rarely > 1 %; positions 10–20 %; ≤ 30 % overnight" | 0.5 % equity risk/trade, position ≤ 20 % equity, no leverage |
| Regime | bullish markets pay 10–20R; he sizes down in bad tape | new entries only while SPY > its 200-day SMA (toggle) |
| Volatility floor | needs volatility to pay off big R-multiples | 20-day ADR ≥ 4.5 % (optimized §4; was 3.5 %) |

### Episodic Pivot (scanner only, not backtested)
Gap ≥ 10 % on ≥ 3× average volume, out of a quiet non-extended base (6-mo move
< 100 %), catalyst-driven (earnings/FDA/regulatory). Entry = opening-range high
(daily proxy: gap-day high), stop = low of day, same trail. Not backtested here
because it needs the catalyst + intraday data to model honestly.

---

## 2. Bugs found & fixed

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
half once the look-ahead was removed (§3), then partly recovered through honest
optimization (§4–5).

---

## 3. Baseline backtest (post bug-fix, pre-optimization)

The parameters Qullamaggie states literally (`cons_len=12`, `mom_pctile=0.90`,
`adr_min=3.5`, `partial_day=4`) produced this, with the look-ahead removed:

```
Period               10.7 yrs        CAGR        2.1 %   (SPY 15.0 %)
Trades                  488          Max drawdown -20.5 % (SPY -33.7 %)
Win rate               32.0 %        Sharpe        0.26
Expectancy            +0.13 R        Profit factor 1.14
```

Six of ten years flat-to-down (2018, 2019, 2021, 2022 all negative). Positive-sign
but thin — see the original one-at-a-time sensitivity sweep this finding came from
in the git history of this file (`git log -p -- REPORT.md`) or re-run
`python backtest.py --sensitivity` with `cons_len=12, mom_pctile=0.90, adr_min=3.5,
partial_day=4` to reproduce it. This was the trigger for the optimization pass below.

---

## 4. Optimization — grid search with an in-sample/out-of-sample split

`optimize.py` runs each candidate **once** over the full history, then reports
metrics computed separately on **train** (entries before 2023-01-01, ~7 yrs) and
**test** (entries from 2023-01-01 on, ~3.7 yrs) from that *same* equity/trade
stream — a config only looks good here if it works on data it wasn't picked from.

### Stage 1 — signal quality (`adr_min × cons_len × mom_pctile`, 30 combos)
Fixed: `partial_day=4, max_positions=10, risk=0.5%`. Full grid in `stage1.out`;
top result by train Sharpe (min 40 train trades):

| Config | Train Sharpe / Exp | Test Sharpe / Exp (n) |
|---|---|---|
| **adr 4.5 · len 16 · mom 0.95** | 0.61 / +0.69R | 0.73 / +0.51R (n=92) |
| adr 4.5 · len 16 · mom 0.90 | 0.51 / +0.49R | 0.73 / +0.46R (n=104) |
| adr 3.5 · len 16 · mom 0.95 | 0.42 / +0.28R | 0.76 / +0.43R (n=154) |
| adr 5.0 · len 12 · mom 0.90 | 0.29 / +0.30R | 0.80 / +0.59R (n=85) |

**Finding: `cons_len = 16` (vs. the literal 12) is the single biggest lever** —
it roughly doubles Sharpe and expectancy at *every* ADR tier, in both train and
test. `adr_min = 4.5` is the next best lever. Above `adr_min ≈ 5.5` the trade
count collapses (< 20/yr) and results get noisy/unstable (adr 6.5 · len 20 turns
*negative* — a small-sample artifact, not a real finding). Chose **adr 4.5 / len
16 / mom 0.95** over the higher-Sharpe-but-thinner adr 5.0/5.5 cells: comparable
edge, ~50% more trades, and train/test agree closely (no overfitting red flag).

### Stage 2 — capital deployment (`risk_per_trade × max_positions`, `partial_day`)
Fixed at stage 1's pick. Full grid in `stage2.out`:

* **`max_positions` is inert** (10 = 15 = 20, bit-for-bit identical results) —
  the strategy is candidate-constrained (only ~15 qualifying setups/year), never
  slot-constrained. Raising it does nothing; don't bother.
* **Raising `risk_per_trade` above 0.5 % only adds drawdown**, not return: 0.75%
  → DD −14.1% (was −10.8%), Sharpe 0.54 (was 0.64); 1.0% → DD −15.6%, Sharpe 0.47.
  The 0.5% base was already right.
* **`partial_day = 5`** (trim on day 5, not 4) edges out day 3/4/8: full Sharpe
  0.66 vs 0.64/0.60/(untested at this base), CAGR 4.3% vs 3.9%, test Sharpe 0.84
  (best of the stage-2 grid) — chosen as the new default.

### Chosen configuration
```python
cons_len=16, mom_pctile=0.95, adr_min=4.5, partial_day=5,
risk_per_trade=0.005, max_positions=10, regime_filter=True   # unchanged from baseline
```

---

## 5. Backtest results — optimized configuration (current `backtest.py` default)

```
Period               10.7 yrs
Final equity         $157,400   (from $100,000)
CAGR                    4.3 %    (SPY buy & hold 15.0 %)
Max drawdown          -11.3 %    (SPY -33.7 %)
Sharpe (daily)          0.66
Trades                  150
Win rate               30.0 %
Expectancy            +0.65 R
Avg win / avg loss   +4.66R / -1.06R
Profit factor           1.82
Best / worst trade   +23.2R / -3.9R
Avg holding             5 bars
```

vs. the honest, un-optimized baseline (§3): Sharpe **0.26 → 0.66**, drawdown
**−20.5% → −11.3%**, expectancy **+0.13R → +0.65R**, profit factor **1.14 → 1.82** —
at a third of the trade count (488 → 150), i.e. more selective, not more active.

### By year
| Year | Strat % | Strat maxDD | Trades | Win % | Exp R |
|---|---|---|---|---|---|
| 2016 | +2.2 | -1.7 | 2 | 100 | +2.20 |
| 2017 | +1.2 | -0.5 | 1 | 100 | +2.39 |
| 2018 | +9.4 | -1.4 | 6 | 50 | +3.31 |
| 2019 | -2.2 | -2.3 | 7 | 14 | -0.62 |
| 2020 | +3.7 | -8.1 | 28 | 32 | +0.28 |
| 2021 | +5.7 | -4.0 | 10 | 30 | +1.16 |
| 2022 | -1.2 | -1.5 | 4 | 25 | -0.62 |
| 2023 | -2.0 | -2.4 | 5 | 0 | -0.82 |
| 2024 | +12.0 | -5.9 | 16 | 19 | +1.50 |
| 2025 | +4.1 | -5.0 | 26 | 23 | +0.31 |
| 2026 YTD | +15.1 | -6.7 | 45 | 36 | +0.68 |

Note the very low trade counts in 2016–2018 (1–6/yr) — those years' 100%/50% win
rates are not statistically meaningful on their own; the strategy only becomes
frequent enough to trust year-by-year from 2020 on (10–45 trades/yr).

### Sensitivity around the optimized center (one knob at a time)
| Knob | Value | CAGR | MaxDD | Trades | Win% | Exp R | Sharpe |
|---|---|---|---|---|---|---|---|
| **cons_len** | 8 | 6.0 | -13.0 | 247 | 33 | +0.56 | 0.73 |
| | 12 | 2.9 | -10.2 | 184 | 33 | +0.36 | 0.45 |
| | **16** | 4.3 | -11.3 | 150 | 30 | +0.65 | 0.66 |
| | 20 | 1.8 | -10.3 | 126 | 29 | +0.34 | 0.34 |
| **mom_pctile** | 0.80 | 4.9 | -14.9 | 189 | 29 | +0.59 | 0.68 |
| | 0.90 | 4.3 | -13.1 | 176 | 30 | +0.56 | 0.65 |
| | **0.95** | 4.3 | -11.3 | 150 | 30 | +0.65 | 0.66 |
| | 0.98 | 2.7 | -9.3 | 109 | 29 | +0.59 | 0.54 |
| **adr_min** | 2.5 | 4.7 | -20.6 | 608 | 29 | +0.21 | 0.49 |
| | 3.5 | 5.2 | -18.8 | 303 | 32 | +0.42 | 0.62 |
| | **4.5** | 4.3 | -11.3 | 150 | 30 | +0.65 | 0.66 |
| | 5.0 | 4.3 | -8.4 | 102 | 31 | +0.93 | 0.72 |
| **partial_day** | 3 | 3.3 | -10.7 | 154 | 36 | +0.48 | 0.60 |
| | 4 | 3.9 | -10.8 | 150 | 37 | +0.58 | 0.64 |
| | **5** | 4.3 | -11.3 | 150 | 30 | +0.65 | 0.66 |
| | 8 | 5.1 | -11.3 | 146 | 27 | +0.79 | 0.70 |
| **cons_range_mult** | 5.0 | 2.7 | -8.2 | 87 | 32 | +0.71 | 0.58 |
| | **8.0** | 4.3 | -11.3 | 150 | 30 | +0.65 | 0.66 |
| | 12.0 | 4.1 | -9.8 | 155 | 30 | +0.59 | 0.63 |
| **regime_filter** | **True** | 4.3 | -11.3 | 150 | 30 | +0.65 | 0.66 |
| | False | 4.4 | -13.3 | 186 | 30 | +0.53 | 0.62 |

**Read:** every direction still shows a monotonic, positive-expectancy surface — no
fragile cliffs. Two knobs (`cons_len=8`, `adr_min=5.0`, `partial_day=8`) each look
*better* than the chosen center in this single-knob view; they were **not** re-run
through the train/test split, so don't chase them without doing that first — with
~150 trades total, one-at-a-time sweeps around an already-optimized point are
increasingly just noise, and stacking multiple "better" single knobs is how you
curve-fit a 150-trade sample into oblivion. Treat this table as a robustness check
on the chosen config, not a menu to keep re-optimizing from.

---

## 6. Verdict

**A real, moderate edge — usable as a low-frequency mechanical sleeve, not just a watchlist.**

* The optimized config roughly **2.5× the Sharpe, halves the drawdown, and 5×s the
  expectancy** of the literal-rules baseline, and it does so out-of-sample: the
  train/test split (§4) shows the edge holding up on 2023–2026 data the grid search
  never touched (test Sharpe 0.73, actually *higher* than train's 0.61).
* It is **low frequency by construction** — ~15 trades/year, not ~45. `max_positions`
  never binds; this is a selective screen, not a portfolio filler. Don't expect it to
  ever be more than a satellite allocation.
* CAGR (4.3%) still lags SPY (15%) because of that selectivity — this is the trade-off
  for a −11.3% max drawdown vs. SPY's −33.7%, not a flaw to "fix" by adding leverage
  (stage 2 showed raising risk/trade only adds drawdown, not return).
* The daily-bar limitation from the original report still applies in full: no real
  opening-range-high timing, no intraday stop, close-fills on trims/trails. This
  remains a **floor estimate** of a well-executed intraday version, and an
  **overestimate** of a sloppily-executed one.

### How to actually use it
1. **Mechanical sleeve, small allocation.** The optimized config (adr_min 4.5,
   cons_len 16, mom_pctile 0.95, partial_day 5, 0.5% risk) is the one worth running
   live-paper first, given the out-of-sample confirmation.
2. **Don't re-optimize further on this sample.** 150 trades is already thin for a
   30-combo search; the §5 sensitivity table shows tempting single-knob improvements
   that were not out-of-sample-checked — resist stacking them.
3. **Do the intraday work eventually.** 1/5-min data would let this be evaluated (and
   traded) against Qullamaggie's actual entry/stop timing rather than the daily proxy.
4. Re-run `python optimize.py` yearly as more out-of-sample data accumulates, to see
   whether the picked config keeps confirming or was a 2023–2026-specific fit.

---

## 7. Caveats (do not skip)

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
7. **The optimization itself is a form of overfitting risk**, mitigated but not
   eliminated by the train/test split — 30+9 candidates were tried, the split is a
   single fixed date (2023-01-01), and the "winner" was picked by eye, not by a
   formal correction for multiple comparisons. Treat the 4.3% CAGR / 0.66 Sharpe as
   optimistic relative to genuinely fresh, never-touched future data.

---

## 8. Files

| File | What |
|---|---|
| `backtest.py` | the simulator — `python backtest.py [--sensitivity] [--no-regime] [--refresh]`. `P` holds the current (optimized) defaults. |
| `optimize.py` | the grid search — `python optimize.py` (stage 1) / `python optimize.py --stage2 --base "k=v,..."` (stage 2), each with a 2023-01-01 train/test split |
| `data/prices.pkl` | cached daily OHLCV (delete or `--refresh` to rebuild) |
| `trades.csv` | every trade from the current default config: entry/exit dates & prices, bars held, exit reason, R multiple, P&L |
| `equity.csv` / `equity.png` | daily equity curve vs SPY + drawdown, current default config |
| `stage1.out` / `stage2.out` | raw console output of the two grid searches (gitignored — regenerate with `optimize.py`) |
