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

| URL param | Tool |
|-----------|------|
| `?scanner=weekly_opts` | Swing Options 45–60 DTE |
| `?scanner=gap` | Gap Scanner |
| `?scanner=breakout` | Breakout Screener |
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
- `cheap_calls_weekly_0_7dte/` — Weekly cheap calls 0–7 DTE; run `python backtest.py`
- `daily_alerts_guardrails/` — Parameter sweep / guardrails for Daily Alerts; run `python run_backtest.py`

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
