"""
Daily Alerts Live Runner
=========================
Runs the Daily Alerts swing-trading strategy live on Alpaca paper/live account.

Flow (each trading day)
-----------------------
1. Wait for market open (9:30 ET)
2. Run generate_daily_alerts() — same signals as the dashboard
3. Place bracket orders (limit entry + stop-loss + take-profit) for top N alerts
4. Monitor open positions every `monitor_interval_min` minutes:
   - Log unrealized P&L and days held per position
   - Check exit signals from alerts engine (URGENT → close)
   - Check max_hold_days (time exit → close)
   - Reconcile tracker when bracket orders auto-exit via SL/TP
5. At session end — print day summary
6. Sleep until next market open

Usage
-----
    # Make sure .env has ALPACA_API_KEY and ALPACA_SECRET_KEY
    python alerts_live_runner.py --config configs/alerts_live.yaml

    # One-shot scan + execute (no daily loop, no monitoring):
    python alerts_live_runner.py --once
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from alerts_engine import generate_daily_alerts, generate_exit_signals
from alpaca_trader import (
    make_client,
    get_account_summary,
    get_positions,
    get_open_orders,
    get_todays_trades,
    execute_alerts,
    cancel_all_orders,
    close_position,
    is_market_open_alpaca,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ── Config loader ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "broker": {
        "mode": "paper",
        "api_key": "",
        "secret_key": "",
    },
    "universe": {
        "preset": "High-Growth Tech",
        "custom_tickers": [],
    },
    "strategy": {
        "min_score":       7,
        "max_positions":   3,
        "risk_pct":        0.5,          # % of portfolio per trade (low for paper safety)
        "max_position_pct": 5.0,         # max single position size %
        "max_dollars_per_trade": 1000,   # hard cap: never spend more than this per entry
        "take_profit":     "T2",         # T1 | T2 | T3
        "direction":       "BUY",        # BUY | SELL | ALL
        "scan_time":       "09:35",      # ET — when to scan after open
    },
    "risk": {
        "max_hold_days":        20,
        "monitor_interval_min": 15,
        "urgent_exit":          True,    # close on URGENT exit signals
        "time_exit":            True,    # close when max_hold_days exceeded
    },
}

PRESETS = {
    "High-Growth Tech": [
        "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","AVGO","PLTR",
        "APP","ARM","CRWD","PANW","AXON","FICO","DDOG","NET","ZS","MRVL",
        "SNOW","TTD","HUBS","DUOL","CAVA","CELH","ONON","SHOP","COIN","HOOD",
    ],
    "Large Cap Leaders": [
        "AAPL","MSFT","AMZN","GOOGL","META","TSLA","NVDA","BRK-B","LLY",
        "V","MA","UNH","JPM","XOM","COST","WMT","PG","HD","JNJ","ABBV",
    ],
}


def load_config(path: str) -> dict:
    cfg = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    if path and os.path.exists(path):
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        for section, values in loaded.items():
            if isinstance(values, dict) and section in cfg:
                cfg[section] = {**cfg[section], **values}
            else:
                cfg[section] = values
    return cfg


# ── Market clock helpers ──────────────────────────────────────────────────────

def _now_et() -> datetime:
    return datetime.now(ET)


def _is_weekday() -> bool:
    return _now_et().weekday() < 5


def _seconds_until(hour: int, minute: int) -> float:
    now    = _now_et()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return max(0.0, (target - now).total_seconds())


def _parse_time(s: str):
    h, m = map(int, s.split(":"))
    return h, m


def _wait_for_market_open(scan_time: str) -> None:
    sh, sm = _parse_time(scan_time)
    while True:
        now = _now_et()
        if now.weekday() >= 5:
            while _now_et().weekday() >= 5:
                secs = _seconds_until(sh, sm)
                log.info("Weekend — sleeping %.0f minutes until %s ET Monday", secs / 60, scan_time)
                time.sleep(min(secs, 3600))
            continue
        now            = _now_et()
        target_today   = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
        if now < target_today:
            wait = (target_today - now).total_seconds()
            log.info("Waiting %.0f minutes until scan time %s ET", wait / 60, scan_time)
            time.sleep(wait)
            return
        elif now.hour < 16:
            return
        else:
            secs = _seconds_until(sh, sm)
            log.info("Session ended — sleeping %.0f hours until tomorrow %s ET", secs / 3600, scan_time)
            time.sleep(secs)


# ── Universe resolver ─────────────────────────────────────────────────────────

def _resolve_universe(cfg: dict) -> list:
    uni    = cfg.get("universe", {})
    custom = [t.strip().upper() for t in uni.get("custom_tickers", []) if t.strip()]
    if custom:
        return custom
    preset = uni.get("preset", "High-Growth Tech")
    try:
        from dashboard import _cached_sp500, _cached_ndq100, _cached_trending
        if preset == "S&P 500":     return _cached_sp500()
        if preset == "Nasdaq-100":  return _cached_ndq100()
        if preset == "🔥 Trending": return _cached_trending()
    except Exception:
        pass
    return PRESETS.get(preset, PRESETS["High-Growth Tech"])


# ── Position tracker ──────────────────────────────────────────────────────────

_TRACKER_PATH = os.path.join(os.path.dirname(__file__), "logs", "position_tracker.json")


def _load_tracker() -> dict:
    if os.path.exists(_TRACKER_PATH):
        try:
            with open(_TRACKER_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_tracker(tracker: dict) -> None:
    os.makedirs(os.path.dirname(_TRACKER_PATH), exist_ok=True)
    with open(_TRACKER_PATH, "w") as f:
        json.dump(tracker, f, indent=2, default=str)


def _sync_tracker(tracker: dict, positions: list) -> list:
    """
    Add new positions to tracker (with entry_time = now).
    Remove symbols no longer in positions (filled via SL/TP).
    Returns list of symbols that were auto-exited by Alpaca bracket orders.
    """
    open_syms  = {p["symbol"] for p in positions}
    now_str    = datetime.now().isoformat()

    # Detect auto-exits (position gone but was in tracker)
    auto_exited = [sym for sym in tracker if sym not in open_syms]
    for sym in auto_exited:
        del tracker[sym]

    # Register new positions
    for p in positions:
        if p["symbol"] not in tracker:
            tracker[p["symbol"]] = {
                "entry_time":  now_str,
                "entry_price": p["entry_price"],
                "qty":         p["qty"],
                "side":        str(p["side"]),
            }

    _save_tracker(tracker)
    return auto_exited


def _days_held(tracker: dict, symbol: str) -> int:
    entry = tracker.get(symbol, {}).get("entry_time", "")
    if not entry:
        return 0
    try:
        return (datetime.now() - datetime.fromisoformat(entry)).days
    except Exception:
        return 0


# ── Day runner ────────────────────────────────────────────────────────────────

def _run_day(client, cfg: dict) -> None:
    strat   = cfg["strategy"]
    now_str = _now_et().strftime("%Y-%m-%d %H:%M ET")

    log.info("=" * 60)
    log.info("DAILY ALERTS SCAN — %s", now_str)
    log.info("=" * 60)

    acct = get_account_summary(client)
    log.info("Portfolio: $%s  |  Cash: $%s  |  Buying Power: $%s",
             f"{acct['portfolio_value']:,.2f}",
             f"{acct['cash']:,.2f}",
             f"{acct['buying_power']:,.2f}")

    tickers   = _resolve_universe(cfg)
    min_score = int(strat.get("min_score", 7))
    direction = strat.get("direction", "BUY").upper()

    log.info("Scanning %d tickers (min_score=%d, direction=%s)…", len(tickers), min_score, direction)

    alerts_df = generate_daily_alerts(tickers=tickers, min_score=min_score, max_workers=12)

    if alerts_df.empty:
        log.info("No alerts met the criteria today — no orders placed.")
        return

    if direction != "ALL":
        alerts_df = alerts_df[alerts_df["Direction"] == direction]

    log.info("Alerts found: %d", len(alerts_df))
    for _, row in alerts_df.head(10).iterrows():
        log.info("  %-6s  %s  score=%d  entry=$%.2f  stop=$%.2f  T2=$%.2f  R/R=%.1f",
                 row["Symbol"], row["Direction"], row["Score"],
                 row["Entry"], row["Stop"], row["T2"], row["R/R"])

    use_t2        = strat.get("take_profit", "T2") != "T1"
    max_pos       = int(strat.get("max_positions", 3))
    risk_pct      = float(strat.get("risk_pct", 0.5)) / 100
    max_pct       = float(strat.get("max_position_pct", 5.0)) / 100
    max_dollars   = float(strat.get("max_dollars_per_trade", 1000))

    log.info("Placing up to %d bracket orders (risk=%.1f%%  cap=$%.0f/trade)…",
             max_pos, risk_pct * 100, max_dollars)

    results = execute_alerts(
        client            = client,
        alerts_df         = alerts_df,
        risk_pct          = risk_pct,
        max_position_pct  = max_pct,
        max_new_positions = max_pos,
        min_score         = min_score,
        use_t2_target     = use_t2,
        dry_run           = False,
        max_dollars_per_trade = max_dollars,
    )

    placed = 0
    for r in results:
        if r.get("ok"):
            placed += 1
            cost = r["qty"] * r["entry"]
            log.info("  ✅  ORDER  %-6s  %s  qty=%d  entry=$%.2f  stop=$%.2f  target=$%.2f  cost≈$%.0f",
                     r["symbol"], r["side"], r["qty"], r["entry"], r["stop"], r["target"], cost)
        else:
            log.warning("  ❌  FAILED %-6s  %s", r.get("symbol", "?"), r.get("error", ""))

    log.info("Orders placed: %d / %d attempted", placed, len(results))


# ── Position monitor ──────────────────────────────────────────────────────────

def _monitor_positions(client, cfg: dict, tracker: dict) -> None:
    """
    Loop every monitor_interval_min minutes until 16:00 ET:
      - Sync tracker (detect auto-exits via bracket SL/TP)
      - Log unrealized P&L and days held per position
      - Run exit signal analysis on held symbols
      - Close positions that hit URGENT signals or max_hold_days
    """
    risk         = cfg["risk"]
    interval_min = int(risk.get("monitor_interval_min", 15))
    max_hold     = int(risk.get("max_hold_days", 20))
    do_urgent    = bool(risk.get("urgent_exit", True))
    do_time      = bool(risk.get("time_exit", True))

    log.info("Monitor started — checking every %d min, max_hold=%d days, "
             "urgent_exit=%s, time_exit=%s",
             interval_min, max_hold, do_urgent, do_time)

    while True:
        now = _now_et()
        if now.hour >= 16:
            log.info("16:00 ET — session ended, stopping monitor.")
            break

        positions = get_positions(client)

        # ── Sync tracker ──────────────────────────────────────────────────────
        auto_exited = _sync_tracker(tracker, positions)
        for sym in auto_exited:
            log.info("  🎯  AUTO-EXIT  %s  (bracket SL/TP filled by Alpaca)", sym)

        if not positions:
            log.info("No open positions — waiting %d min…", interval_min)
            time.sleep(interval_min * 60)
            continue

        # ── Exit signal analysis ──────────────────────────────────────────────
        syms         = [p["symbol"] for p in positions]
        exit_df      = generate_exit_signals(syms)
        total_pl     = sum(p["unrealized_pl"] for p in positions)

        log.info("─" * 60)
        log.info("MONITOR  %s  |  %d positions  |  Total P&L: $%+.2f",
                 now.strftime("%H:%M ET"), len(positions), total_pl)

        for p in positions:
            sym       = p["symbol"]
            days      = _days_held(tracker, sym)
            ex_row    = exit_df.loc[sym] if sym in exit_df.index else None
            urgency   = ex_row["Exit Urgency"] if ex_row is not None else ""
            sigs      = ex_row["Exit Signals"] if ex_row is not None else []
            action    = ex_row["Exit Action"]  if ex_row is not None else ""

            status = urgency if urgency else "HOLD"
            log.info(
                "  %-8s  qty=%.0f  entry=$%.2f  curr=$%.2f  "
                "P&L=$%+.2f (%+.1f%%)  held=%dd  [%s]",
                sym, p["qty"], p["entry_price"], p["current_price"],
                p["unrealized_pl"], p["unrealized_pl%"], days, status,
            )
            for s in (sigs or [])[:4]:
                log.info("    ⚠️  %s", s)
            if action:
                log.info("    → %s", action)

            # ── Exit decisions ────────────────────────────────────────────────
            should_exit = False
            exit_reason = ""

            if do_time and days >= max_hold:
                should_exit = True
                exit_reason = f"TIME EXIT — held {days}d ≥ {max_hold}d limit"

            elif do_urgent and urgency == "URGENT":
                should_exit = True
                exit_reason = f"SIGNAL EXIT URGENT — {'; '.join((sigs or [])[:2])}"

            if should_exit:
                log.warning("  🔴  CLOSING %s — %s", sym, exit_reason)
                result = close_position(client, sym)
                if result.get("ok"):
                    log.info("  ✅  %s closed (order_id=%s)", sym, result.get("order_id", "?"))
                    tracker.pop(sym, None)
                    _save_tracker(tracker)
                else:
                    log.error("  ❌  Close failed for %s: %s", sym, result.get("error", ""))

        log.info("─" * 60)
        time.sleep(interval_min * 60)


# ── Day-end summary ───────────────────────────────────────────────────────────

def _print_day_summary(client) -> None:
    trades = get_todays_trades(client)
    filled = [t for t in trades if t["status"] in ("filled", "partially_filled")]
    acct   = get_account_summary(client)

    log.info("=" * 60)
    log.info("DAY SUMMARY — %s", _now_et().strftime("%Y-%m-%d"))
    log.info("=" * 60)
    log.info("Portfolio value: $%s", f"{acct['portfolio_value']:,.2f}")
    log.info("Filled orders:   %d", len(filled))
    for t in filled:
        log.info("  %-30s  %s  qty=%.0f  @$%.2f",
                 t["symbol"], t["side"], t["filled_qty"],
                 t["filled_price"] or 0)
    log.info("=" * 60)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    # Load .env
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    parser = argparse.ArgumentParser(description="Daily Alerts live paper trader")
    parser.add_argument("--config", default="configs/alerts_live.yaml",
                        help="Path to YAML config (default: configs/alerts_live.yaml)")
    parser.add_argument("--once", action="store_true",
                        help="Run one scan immediately and exit (no loop or monitoring)")
    parser.add_argument("--monitor-only", action="store_true",
                        help="Skip scan, only run the position monitor on existing positions")
    args = parser.parse_args()

    cfg = load_config(args.config)

    broker  = cfg.get("broker", {})
    api_key = broker.get("api_key") or os.environ.get("ALPACA_API_KEY", "")
    sec_key = broker.get("secret_key") or os.environ.get("ALPACA_SECRET_KEY", "")

    if not api_key or not sec_key:
        log.error("No Alpaca credentials. Set ALPACA_API_KEY / ALPACA_SECRET_KEY in .env or config.")
        sys.exit(1)

    paper     = str(broker.get("mode", "paper")).lower() == "paper"
    client    = make_client(api_key, sec_key, paper=paper)
    tracker   = _load_tracker()
    scan_time = cfg["strategy"].get("scan_time", "09:35")

    log.info("Daily Alerts Live Runner started")
    log.info("Broker: %s  |  Scan time: %s ET  |  Universe: %s",
             broker.get("mode", "paper"), scan_time,
             cfg.get("universe", {}).get("preset", "custom"))
    log.info("Risk cap: $%.0f/trade  |  Max positions: %d  |  Max hold: %d days",
             float(cfg["strategy"].get("max_dollars_per_trade", 1000)),
             int(cfg["strategy"].get("max_positions", 3)),
             int(cfg["risk"].get("max_hold_days", 20)))

    if args.once:
        _run_day(client, cfg)
        _print_day_summary(client)
        return

    if args.monitor_only:
        _monitor_positions(client, cfg, tracker)
        _print_day_summary(client)
        return

    # Continuous daily loop
    while True:
        _wait_for_market_open(scan_time)
        try:
            _run_day(client, cfg)
            _monitor_positions(client, cfg, tracker)
            _print_day_summary(client)
        except KeyboardInterrupt:
            log.info("Shutdown requested.")
            _print_day_summary(client)
            break
        except Exception as e:
            log.error("Day run error: %s", e, exc_info=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
