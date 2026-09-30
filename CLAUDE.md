# SwingTrading — Claude Code Context

Primary working directory: `/Users/arkilthakkar/workplace/Scripts/SwingTrading`
Git remote: `https://github.com/arkil/SwingTrading` (local is source of authority — push freely)

## Environment Setup

```bash
source .venv/bin/activate          # always activate before running anything
pip install -e .                   # installs dtb / dtb-live / dtb-pyramid CLIs
```

`.env` must exist with Alpaca keys for any live/paper trading tool:
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

---

## How to Run the Dashboard

```bash
# Main hub (all scanners accessible from here)
streamlit run home.py --server.port 8501
# or
./launch_home.sh

# Full trading dashboard
streamlit run dashboard.py
# or
./launch_dashboard.sh
```

Open `http://localhost:8501` — use `?scanner=<name>` to jump to a specific tool:

The 4 short-DTE options scanners (`weekly_opts`, `cheap_calls`, `weekly_puts`, `zero_dte`)
now live under one sidebar entry, `options_scanner` ("⚡ Options Scanner (0-7 DTE)"),
as radio sub-tabs (`render_options_unified` in dashboard.py). The old `?scanner=weekly_opts`
etc. URLs still work — they redirect into `options_scanner` with the matching sub-tab
pre-selected (see `_OPTIONS_MODES`/`_url_scanner` handling in dashboard.py). Each sub-tab
shows a walk-forward backtest badge (see "Standalone Strategies" below); the "Sell Premium"
0DTE mode shows a KILLED warning instead, since that has already been backtested against
real data and found to have no edge.

| URL param | Tool |
|-----------|------|
| `?scanner=weekly_opts` | Weekly Options 0-7 DTE (now a sub-tab of Options Scanner) |
| `?scanner=cheap_calls` | Cheap Calls <$1.50 (now a sub-tab of Options Scanner) |
| `?scanner=weekly_puts` | Weekly Puts 0-3 DTE (now a sub-tab of Options Scanner) |
| `?scanner=zero_dte` | 0DTE Scanner (now a sub-tab of Options Scanner) |
| `?scanner=swing_opts` | Swing Options 45–60 DTE |
| `?scanner=crash_hedge` | Crash Hedge — 2-month bearish put debit spread + VIX tail sleeve |
| `?scanner=ravish_theta` | Ravish Theta Hub (`ravish_hub_module.py`) — one page, four views picked by a radio / `&view=`. Old `?scanner=theta_wheel` / `?scanner=spx_rsi_put` URLs redirect here with the matching view. |
| `…&view=machine` | Ravish Theta Machine — overnight earnings iron fly, theta-machine calendar, post-earnings iron condor (`ravish_theta_module.py`; logic, live scan + real-Alpaca-print backtester in `~/workplace/strategies/ravish_theta/`, scan cache `Data/scan_latest.json` there) |
| `…&view=calendar_em` | EM Calendar — Ravish's calendar-spread video (GuI-hH_jhlg): sell ~35 DTE, buy the same strike +7 d; neutral ATM call, bullish call at spot+EM, bearish put at spot−EM, double (both). PASS = ATM IV below its own 1-yr median (Alpaca option prints, cached `Data/cache/ativ_<sym>_35.parquet`), no earnings before the back expiry, 7±2 d gap, tight legs. Model max profit / profit range at the short expiry (`calendar_em_module.py`; scan `scan.py --only calendar_em` + `src/calendar_em.py` in `~/workplace/strategies/ravish_theta/`, universe in its `config.yaml` `calendar_em:`) |
| `…&view=spx_put` | SPX RSI Bull Put — Ravish's RSI(14)<35 ATM 10-wide SPX put credit spread, 14 DTE, TP 50% (`spx_rsi_put_module.py`; SPX signal/chart from yfinance, SPY 1-wide live proxy from the ravish_theta scan; backtest `strategies/ravish_theta/src/rsi_put_spread.py`) |
| `…&view=wheel` | Theta Wheel — weekly cash-secured puts on bullish-weekly, fundamentally strong names (`theta_wheel_module.py`, cache `Data/theta_wheel/latest.json`) |
| `?scanner=daily_dashboard` | Daily Market Dashboard — copy of github.com/shthakkar/daily_dashboard: SPY 10/20 EMA direction, VIX level, SPY vs 52W high cards + Capitulation Fade, Qullamaggie MA tightness, Relative Strength, News Movers (`daily_dashboard_module.py`; screeners ported to `strategies/daily_dashboard/`, tests `tests/daily_dashboard/`; ~3-min Finviz+yfinance scan `python -m strategies.daily_dashboard.market` → `Data/daily_dashboard/latest.json`, auto-refreshed in the background when >12h old; needs `finvizfinance`) |
| `?scanner=gap` | Gap Scanner |
| `?scanner=breakout` | Breakout Screener |
| `?scanner=qullamaggie` | Qullamaggie Breakout + Episodic Pivot Screener |
| `?scanner=smallcap_momentum` | Small-Cap Momentum (Trader YC) — pre-market gapper watchlist with a WATCH/ARMED/TRIGGERED/SKIP status per name for the pre-market-high breakout (9:30–10:00), heat sizing from yesterday's doublers, rule replay with session charts |
| `?scanner=vwap_bias` | VWAP Bias (Xavi Comas) — small-cap gappers get a long/short bias from VWAP; buy dips above it, short bounces below it, cut on a VWAP close or no bounce in 5 min (`vwap_bias_module.py`; rules `strategies/vwap_bias_xavi/rules.py`, shared with the backtest; reuses the Small-Cap Momentum gapper scan and bar cache). 2-yr SIP backtest: no edge after costs (`strategies/vwap_bias_xavi/research/REPORT.md`) |
| `?scanner=parabolic_short` | Parabolic Short (Marius Stamatiou) — exhaustion-day short after a parabolic run: move vs market cap from the last 20-SMA base + huge expansion days, 2-day window, 1-min entries (first weakness / VWAP loss / VWAP kiss), stop HOD, target = his expected drop (`parabolic_short_module.py`; rules `strategies/parabolic_short_marius/rules.py`, shared with the backtest; full-market scan `python -m strategies.parabolic_short_marius.scan` after 17:00 ET → `Data/parabolic_short/latest.json`). Alerts only. 2-yr backtest: his intraday rules ≈ 0R; the daily filter alone shorted 09:35→close is positive both years (`strategies/parabolic_short_marius/research/REPORT.md`) |
| `?scanner=lag7` | Lag-7 — at each month-end buy the Mag-7 name with the worst trailing 3M/6M return, hold 1 month (`lag7_module.py`, JSON cache `Data/lag7/latest.json` 20h TTL; research `~/workplace/strategies/lag7/`) |
| `?scanner=combined` | Combined Strategy Scanner |
| `?scanner=ibd` | IBD Buy Zone |
| `?scanner=minervini` | Minervini SEPA |
| `?scanner=canslim` | O'Neil CAN SLIM |
| `?scanner=livermore` | Livermore Pivotal Points |
| `?scanner=macd` | MACD |
| `?scanner=rsi` | RSI |
| `?scanner=ema` | EMA Crossover |
| `?scanner=alerts` | Daily Alerts Engine |
| `?scanner=astro` | Financial Astrology (Gann) |
| `?scanner=economic_calendar` | Macro Events Calendar |

---

## Tool Map

### Screeners (run standalone or via dashboard)
- `ta_common.py` — shared `ema`/`sma`/`rsi_ema`/`rsi_sma`/`atr` imported by the screeners below
  (don't re-define them per file). Not named `indicators.py`: that collides with
  `vendor/volume_scanner/indicators/` on the dashboard's `sys.path`.
- `alerts_engine.py` — Daily alerts with entry/stop/target across all indicators
- `breakout_screener.py` — Minervini SEPA, Darvas Box, NR7, Bollinger Squeeze, IBD, QuantifiedStrategies
- `combined_screener.py` — Multi-factor scoring across all strategies
- `gap_screener.py` — Opening gap detection and classification
- `ibd_scanner.py` — IBD-style buy zone detection
- `livermore_pivotal_screener.py` — Jesse Livermore pivotal point method
- `macd_screener.py` — MACD crossover and histogram signals
- `minervini_screener.py` — SEPA entry point analysis
- `oneil_canslim_scanner.py` — O'Neil CAN SLIM framework
- `rsi_screener.py` — RSI oversold/overbought with trend filter
- `ema_crossover_screener.py` — EMA crossover with volume confirmation
- `swing_options_screener.py` — 45–60 DTE options screener (wraps `swing_options_45_60d` strategy)
- `render_qullamaggie()` in `dashboard.py` — Kristjan Kullamägi's setups: Breakout (momentum-leader
  flag continuation, opening-range-breakout proxy on daily bars) and Episodic Pivot (gap on catalyst).
  Scores each name on his checklist; outputs entry (pivot / gap-day high), stop (breakout-day low),
  1R target, and risk-based position size. No standalone module (inline in dashboard.py).
- `smallcap_momentum_module.py` — Streamlit page for Trader YC's small-cap day-trading method
  (youtube.com/watch?v=9PdfWdoyxxM). Logic lives in `strategies/smallcap_momentum_yc/` (package,
  relative imports — run its CLIs as `python -m strategies.smallcap_momentum_yc.<scanner|heat|replay|monitor>`).
  Alerts only, cash-only sizing (shorts alert-only). Scan cache `Data/smallcap_momentum/latest_scan.json`,
  heat history `strategies/smallcap_momentum_yc/data/heat_history.json`, Polygon grouped-daily cache
  `strategies/smallcap_momentum_yc/data/grouped/`. Pre-market (4:00–9:30 ET) the scan uses Yahoo batch
  pre-market quotes, not the Yahoo screener (which still shows yesterday's session then).
- `astro_scanner.py` — Financial astrology / Gann timing
- `economic_calendar.py` — Macro events calendar and news feed
- `stock_analyzer_module.py` — Deep multi-framework analysis on a single ticker

### Options Tools
- `options_paper_trader.py` — Daemon: runs 45–60 DTE strategy live on Alpaca paper
- `options_trade_log.py` — View open/closed paper option positions
- `options_backtest_runner.py` — Backtest runner for 45–60 DTE options
- `close_options_at_open.py` — One-shot: close all open option positions at market open

### Live Trading
- `alerts_live_runner.py` — Execute Daily Alerts signals on Alpaca paper/live
- `alpaca_trader.py` — Low-level Alpaca order management layer
- `stock_paper_trader.py` — Daemon: trades stock signals on Alpaca paper
- `spy_reversal_log.py` — Log and display SPY intraday reversal alerts

### Backtester CLI (`dtb`)
```bash
dtb --config configs/bollinger_rsi_spy_live.yaml      # backtest
dtb-live --config configs/bollinger_rsi_spy_live.yaml  # live runner
dtb-pyramid --config configs/bollinger_rsi_spy_pyramid_live.yaml  # pyramid live
```
Key configs in `configs/`: `bollinger_rsi_spy_live.yaml`, `bollinger_rsi_spy_improved.yaml`, `bollinger_rsi_spy_pure.yaml`, `bollinger_rsi_spy_pyramid_live.yaml`, `options_paper.yaml`, `stock_paper.yaml`, `alerts_live.yaml`

Source package: `src/daytrade_backtester/` — broker, live, engine, strategies, data, reporting, utils

### Standalone Strategies (`strategies/`)
- `_shared/` — shared OHLCV data loader / features / Black-Scholes pricer (calls + puts) used
  by the three short-DTE backtests below. Not a runnable strategy itself.
- `cheap_calls_weekly_0_7dte/` — Weekly cheap calls 0–7 DTE (also backs the `weekly_opts`
  scanner via a looser preset); run `python backtest.py` for the in-sample sweep or
  `python walkforward.py` for out-of-sample validation (writes `validated_config.json`,
  read by `dashboard.py`'s scanner captions via `backtest_badges.py`)
- `weekly_puts_0_3dte/` — Bearish mirror of the above for the `weekly_puts` scanner; same
  `backtest.py` / `walkforward.py` pattern
- `zero_dte_0_3dte/` — Buy Directional / Intraday Close long-option modes for the `zero_dte`
  scanner (1-3 DTE; true same-day 0DTE excluded, numerically unstable in a BS-simulated
  pricer). The Sell Premium (credit spread/iron condor) mode is **not** modeled here — see
  `zero_dte_backtest_runner.py` at the repo root, which surfaces the real-data finding from
  the separate `strategies/spy_0dte_iron_condor/` project (killed, no positive edge found).
- `daily_alerts_guardrails/` — Parameter sweep / guardrails for Daily Alerts; run `python run_backtest.py`

All three short-DTE walk-forward backtests use Black-Scholes-simulated option pricing from
historical OHLCV (no real historical option chain data) — see each folder's caveats in its
`validated_config.json`. Re-run `walkforward.py` periodically as more data accumulates;
`backtest_badges.py` just reads whatever `validated_config.json` last contained, it does not
re-run anything on dashboard load.

---

## Service Management

```bash
./manage_services.sh status              # show all services and last log lines
./manage_services.sh start [name]        # start one or all
./manage_services.sh stop  [name]        # stop one or all
./manage_services.sh restart [name]      # restart
./manage_services.sh log <name>          # tail live log

# Service names: dashboard | options | alerts | spy
```

---

## Git

- Remote: `https://github.com/arkil/SwingTrading`
- Local is authoritative — commit and push freely
- Gitignored: `.env`, `logs/`, `screener_output/`, `artifacts/`, strategy data caches and trade CSVs
