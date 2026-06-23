# SwingTrading

A suite of stock screeners, options tools, paper traders, and backtesting infrastructure for swing trading and short-term options plays.

---

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

**Launch the main dashboard:**
```bash
streamlit run home.py --server.port 8501
# or
./launch_home.sh
```

**Launch the trading dashboard:**
```bash
streamlit run dashboard.py
# or
./launch_dashboard.sh
```

---

## Dashboards

| File | Description |
|------|-------------|
| `home.py` | Main hub at `localhost:8501` — entry point that links all scanners and tools |
| `dashboard.py` | Full trading dashboard with all screeners embedded as tabs (`?scanner=<name>`) |

The dashboard supports a `?scanner=` URL param to jump directly to a scanner:

| `?scanner=` value | Scanner |
|-------------------|---------|
| `weekly_opts` | Swing Options (45–60 DTE) |
| `gap` | Gap Scanner |
| `breakout` | Breakout Screener |
| `combined` | Combined Strategy Scanner |
| `ibd` | IBD Buy Zone |
| `minervini` | Minervini SEPA |
| `canslim` | O'Neil CAN SLIM |
| `livermore` | Livermore Pivotal Points |
| `macd` | MACD Screener |
| `rsi` | RSI Screener |
| `ema` | EMA Crossover |
| `alerts` | Daily Alerts Engine |
| `astro` | Financial Astrology (Gann) |
| `economic_calendar` | Macro Events Calendar |

---

## Screeners

| File | Description |
|------|-------------|
| `alerts_engine.py` | **Daily Alerts Engine** — runs every stock through all indicator systems in one pass, outputs BUY/SELL alerts with entry, stop, and target prices |
| `breakout_screener.py` | **Breakout Screener** — 6 research-backed strategies: Minervini SEPA, Darvas Box, NR7, Bollinger Squeeze, IBD/CAN SLIM, QuantifiedStrategies |
| `combined_screener.py` | **Combined Scanner** — multi-factor scoring engine combining signals across all strategies into a single ranked buy list |
| `gap_screener.py` | **Gap Scanner** — detects and classifies significant opening price gaps for swing setups |
| `ibd_scanner.py` | **IBD Scanner** — identifies stocks near IBD-style buy zones (in base, pivot breakout, handle breakout) |
| `livermore_pivotal_screener.py` | **Livermore Pivotal Points** — Jesse Livermore's market method adapted for modern scanning |
| `macd_screener.py` | **MACD Screener** — momentum signals based on MACD crossovers and histogram divergence |
| `minervini_screener.py` | **Minervini SEPA** — Specific Entry Point Analysis from *Trade Like a Stock Market Wizard* |
| `oneil_canslim_scanner.py` | **O'Neil CAN SLIM** — screens against William O'Neil's CAN SLIM framework |
| `rsi_screener.py` | **RSI Screener** — oversold/overbought signals with trend filters |
| `ema_crossover_screener.py` | **EMA Crossover** — short/long EMA crossover setups with volume confirmation |
| `swing_options_screener.py` | **Swing Options Screener** — 45–60 DTE options plays; bridges dashboard ticker universe with the `swing_options_45_60d` strategy module |
| `astro_scanner.py` | **Financial Astrology Scanner** — Gann, Pesavento, Meridian & Prakash planetary timing techniques |
| `economic_calendar.py` | **Economic Calendar** — macro events feed with strategy-impact context |
| `stock_analyzer_module.py` | **Stock Analyzer** — multi-framework deep analysis on individual tickers |

---

## Options Tools

| File | Description |
|------|-------------|
| `options_paper_trader.py` | **Options Paper Trader** — daemon that runs the 45–60 DTE strategy live on an Alpaca paper account |
| `options_trade_log.py` | **Options Trade Log** — dashboard tab to review open/closed paper options positions |
| `options_backtest_runner.py` | **Options Backtest Runner** — runs the 45–60 DTE options backtest via config |
| `close_options_at_open.py` | **Close Options at Open** — one-shot script to close all remaining 45–60 DTE positions at market open |

---

## Live Trading & Alerts

| File | Description |
|------|-------------|
| `alerts_live_runner.py` | **Alerts Live Runner** — executes Daily Alerts signals live on an Alpaca paper/live account |
| `alpaca_trader.py` | **Alpaca Trader** — low-level integration layer for submitting and managing orders via Alpaca API |
| `stock_paper_trader.py` | **Stock Paper Trader** — daemon that trades Daily Alert stock signals on Alpaca paper |
| `spy_reversal_log.py` | **SPY Reversal Logger** — logs and displays SPY intraday reversal alerts |

---

## Backtester (`src/daytrade_backtester`)

Config-driven backtesting package for intraday strategies.

```bash
dtb --config configs/bollinger_rsi_spy_live.yaml
```

**Package structure:**

| Module | Description |
|--------|-------------|
| `cli.py` | `dtb` CLI entry point |
| `config/` | YAML config loader and Pydantic models |
| `data/` | Yahoo Finance fetcher, Polygon options data, cache layer |
| `engine/` | Core backtester loop and options enrichment |
| `strategies/` | Strategy implementations (`BaseStrategy` interface) |
| `broker/` | Alpaca broker integration for live execution |
| `live/` | Live bar buffer, position monitor, pyramid runner |
| `reporting/` | Console summary (P&L, win rate, R-multiples) and CSV export |
| `utils/` | Technical indicators |

**Configs in `configs/`:**

| Config | Description |
|--------|-------------|
| `bollinger_rsi_spy_live.yaml` | BB+RSI mean-reversion on SPY (live params) |
| `bollinger_rsi_spy_improved.yaml` | Improved variant with tighter filters |
| `bollinger_rsi_spy_pure.yaml` | Pure signal, no option overlay |
| `bollinger_rsi_spy_pyramid_live.yaml` | Pyramid scaling version |
| `bollinger_rsi_spy_oos_2025q1q2.yaml` | Out-of-sample test Q1/Q2 2025 |
| `bollinger_rsi_spy_highfreq.yaml` | High-frequency variant |
| `bollinger_rsi_qqq_highfreq.yaml` | Same on QQQ |
| `bollinger_rsi_spy_longtest.yaml` | Extended lookback test |
| `bollinger_rsi_spy_option_native.yaml` | Native options P&L model |
| `options_paper.yaml` | Options paper trading config |
| `stock_paper.yaml` | Stock paper trading config |
| `alerts_live.yaml` | Live alerts runner config |

**Add a new strategy:**
1. Create a file in `src/daytrade_backtester/strategies/` implementing `BaseStrategy`.
2. Register it in `src/daytrade_backtester/strategies/registry.py`.
3. Set `strategy.name` in your config.

---

## Standalone Strategies (`strategies/`)

| Strategy | Description |
|----------|-------------|
| `cheap_calls_weekly_0_7dte/` | Cheap weekly calls, 0–7 DTE, signal-driven entry; includes backtest, config, and signal modules |
| `daily_alerts_guardrails/` | Guardrails backtest for Daily Alerts — sweep-based parameter optimization |

---

## Services & Shell Scripts

| File | Description |
|------|-------------|
| `manage_services.sh` | Manage all launchd background services (`status / start / stop / restart / log`) |
| `launch_home.sh` | Launch `home.py` dashboard on port 8501 |
| `launch_dashboard.sh` | Launch `dashboard.py` |
| `options_daemon.sh` | Start the options paper trader as a background daemon |
| `run_livermore_screener.sh` | Run Livermore screener from CLI |

**Service names for `manage_services.sh`:** `dashboard` | `options` | `alerts` | `spy`

```bash
./manage_services.sh status
./manage_services.sh restart alerts
./manage_services.sh log options
```

---

## Environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env   # if present, otherwise create .env
```

Required for live/paper trading:
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

---

## Notes

- Yahoo 5m/1m data has lookback limits; use `period` values Yahoo supports for intraday.
- The backtester uses an estimated options return model from underlying SPY movement — not exact tape replay.
- Logs, screener output, and data caches are gitignored (regenerated at runtime).
