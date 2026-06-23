"""
Stock Daily Alerts — Paper Trader Daemon
=========================================
Full hands-off daemon that mirrors options_paper_trader.py for equity swing trades.

Flow (each trading day)
-----------------------
1. 09:35 ET — generate_daily_alerts() → place Alpaca bracket orders for top N signals
2. Every 15 min — check_exits(): close positions that hit URGENT exit signals or
   max-hold-days (bracket SL/TP legs handle price-based exits automatically on Alpaca)
3. 15:45 ET — print day summary

Exit criteria applied by this monitor
--------------------------------------
  URGENT   Two or more exit triggers, at least one critical:
             - EMA9 crossed below EMA21 (short-term trend broke)
             - MACD bearish cross (momentum reversing)
             - Broke below EMA50
             - Distribution day (heavy vol on down close)
             - Livermore downward pivot
  TIME     Position held longer than max_hold_days (swing default: 10)
  BRACKET  Alpaca stop-loss / take-profit legs fire independently at Alpaca;
           this monitor cancels orphaned legs on the above exits.

State is persisted to logs/stock_positions.json across restarts.

Usage
-----
    python stock_paper_trader.py --config configs/stock_paper.yaml
    python stock_paper_trader.py --now          # scan + exits immediately, no loop
    python stock_paper_trader.py --monitor-only # only check exits, no new entries
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from alerts_engine import generate_daily_alerts, generate_exit_signals
from alpaca_trader import (
    make_client,
    get_account_summary,
    get_positions,
    get_open_orders,
    place_bracket_order,
    close_position,
    cancel_order,
    _calc_shares,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ET         = ZoneInfo("America/New_York")
STATE_FILE = Path(__file__).parent / "logs" / "stock_positions.json"
LOG_DIR    = Path(__file__).parent / "logs" / "stock_paper"

DEFAULT_CONFIG = {
    "broker": {
        "mode":       "paper",
        "api_key":    "",
        "secret_key": "",
    },
    "universe": {
        "preset":         "Tech Leaders",
        "custom_tickers": [],
    },
    "strategy": {
        "scan_time":           "09:35",
        "min_score":           6,
        "max_positions":       5,
        "risk_pct":            1.0,    # % of portfolio risked per trade
        "max_position_pct":    10.0,   # max single position % of portfolio
        "max_dollars_per_trade": 2000,
        "use_t2_target":       True,   # True=T2, False=T1 as bracket TP
    },
    "exit": {
        "max_hold_days":    10,    # swing default; time exit regardless of P&L
        "monitor_interval": 15,   # minutes between exit checks during session
        "urgent_exit":      True, # exit on URGENT signal from generate_exit_signals
    },
}

PRESETS = {
    "Tech Leaders": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AMD", "TSLA", "AVGO", "CRM",
        "NOW", "SNOW", "DDOG", "CRWD", "NET", "PANW", "ADBE", "INTU", "AMAT", "KLAC",
    ],
    "High-Growth": [
        "NVDA", "AMD", "TSLA", "META", "AMZN", "SHOP", "CRWD", "PLTR", "COIN", "HOOD",
        "SMCI", "ARM", "TTD", "MELI", "SE", "GTLB", "MDB", "CFLT", "ZS", "OKTA",
    ],
    "S&P Leaders": [
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "V", "MA",
        "JPM", "UNH", "JNJ", "XOM", "WMT", "PG", "HD", "BAC", "ABBV", "LLY",
    ],
}


# ── Config ────────────────────────────────────────────────────────────────────

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


def _resolve_universe(cfg: dict) -> List[str]:
    uni    = cfg.get("universe", {})
    custom = [t.strip().upper() for t in (uni.get("custom_tickers") or []) if t.strip()]
    if custom:
        return custom
    return PRESETS.get(uni.get("preset", "Tech Leaders"), PRESETS["Tech Leaders"])


# ── Market helpers ────────────────────────────────────────────────────────────

def _now_et() -> datetime:
    return datetime.now(ET)


def _wait_for_scan_time(scan_time: str) -> None:
    sh, sm = map(int, scan_time.split(":"))
    while True:
        now = _now_et()
        if now.weekday() >= 5:
            secs = ((7 - now.weekday()) * 86400
                    - now.hour * 3600 - now.minute * 60 - now.second)
            log.info("Weekend — sleeping %d hours", secs // 3600)
            time.sleep(min(secs, 3600))
            continue
        target = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
        if now < target:
            wait = (target - now).total_seconds()
            log.info("Waiting %.0f min until scan time %s ET", wait / 60, scan_time)
            time.sleep(wait)
            return
        elif now.hour < 16:
            return
        else:
            tomorrow = now + timedelta(days=1)
            target   = tomorrow.replace(hour=sh, minute=sm, second=0, microsecond=0)
            wait     = (target - now).total_seconds()
            log.info("Session ended — sleeping %.0f hours until tomorrow", wait / 3600)
            time.sleep(min(wait, 3600))


# ── State helpers ─────────────────────────────────────────────────────────────

def _load_state() -> Dict:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"positions": []}


def _save_state(state: Dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _add_position(state: Dict, pos: Dict) -> None:
    state["positions"].append(pos)
    _save_state(state)


def _remove_position(state: Dict, symbol: str) -> None:
    state["positions"] = [p for p in state["positions"] if p["symbol"] != symbol]
    _save_state(state)


def _append_day_log(entry: Dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    d_str    = entry.get("date", date.today().isoformat())
    log_path = LOG_DIR / f"stock_paper_{d_str}.json"
    try:
        data = json.loads(log_path.read_text()) if log_path.exists() else {"date": d_str, "orders": []}
    except Exception:
        data = {"date": d_str, "orders": []}
    data["orders"].append(entry)
    log_path.write_text(json.dumps(data, indent=2, default=str))


# ── Alpaca position sync ──────────────────────────────────────────────────────

def _sync_closed_positions(client, state: Dict) -> None:
    """Remove positions from state that Alpaca no longer holds (bracket SL/TP fired)."""
    if not state["positions"]:
        return
    live_syms = {p["symbol"] for p in get_positions(client)}
    for pos in list(state["positions"]):
        if pos["symbol"] not in live_syms:
            log.info("  SYNC  %s no longer in Alpaca — bracket exit fired, removing from state",
                     pos["symbol"])
            _append_day_log({
                "role":       "BRACKET_EXIT",
                "date":       date.today().isoformat(),
                "time_et":    _now_et().strftime("%H:%M"),
                "symbol":     pos["symbol"],
                "side":       pos["side"],
                "qty":        pos["qty"],
                "exit_reason": "bracket_sl_or_tp",
                "days_held":  (date.today() - date.fromisoformat(pos["entry_date"])).days,
            })
            _remove_position(state, pos["symbol"])


# ── Entry scan ────────────────────────────────────────────────────────────────

def _run_entry_scan(client, cfg: dict, state: Dict) -> None:
    strat    = cfg["strategy"]
    tickers  = _resolve_universe(cfg)
    min_score        = int(strat.get("min_score", 6))
    max_pos          = int(strat.get("max_positions", 5))
    risk_pct         = float(strat.get("risk_pct", 1.0)) / 100.0
    max_pos_pct      = float(strat.get("max_position_pct", 10.0)) / 100.0
    max_dollars      = float(strat.get("max_dollars_per_trade", 2000))
    use_t2           = bool(strat.get("use_t2_target", True))

    open_syms = {p["symbol"] for p in state["positions"]}
    pending   = {o["symbol"] for o in get_open_orders(client)}
    skip_set  = open_syms | pending
    available = max_pos - len(state["positions"])

    if available <= 0:
        log.info("Max positions (%d) reached — skipping entry scan", max_pos)
        return

    log.info("Scanning %d tickers for stock signals (min_score=%d)…", len(tickers), min_score)

    try:
        alerts_df = generate_daily_alerts(tickers, min_score=min_score)
    except Exception as e:
        log.error("Alert generation error: %s", e)
        return

    if alerts_df is None or alerts_df.empty:
        log.info("No signals met criteria today.")
        return

    # BUY signals only for a long-only paper account
    alerts_df = alerts_df[alerts_df["Direction"] == "BUY"]
    alerts_df = alerts_df[~alerts_df["Symbol"].isin(skip_set)]
    alerts_df = alerts_df.head(available)

    log.info("Signals after filters: %d", len(alerts_df))
    for _, r in alerts_df.iterrows():
        log.info("  %-6s  score=%d  conviction=%s  R/R=%.1f",
                 r["Symbol"], r["Score"], r["Conviction"], r["R/R"])

    acct         = get_account_summary(client)
    buying_power = acct["buying_power"]

    for _, row in alerts_df.iterrows():
        sym    = row["Symbol"]
        entry  = float(row["Entry"])
        stop   = float(row["Stop"])
        target = float(row["T2"]) if use_t2 else float(row["T1"])

        shares = _calc_shares(
            buying_power     = buying_power,
            entry_price      = entry,
            stop_price       = stop,
            risk_pct         = risk_pct,
            max_position_pct = max_pos_pct,
        )
        if max_dollars > 0:
            shares = min(shares, max(1, int(max_dollars / entry)))

        if shares == 0:
            log.warning("  %-6s  Insufficient buying power — skipping", sym)
            continue

        log.info("  %-6s  entry=%.2f  stop=%.2f  target=%.2f  shares=%d",
                 sym, entry, stop, target, shares)

        result = place_bracket_order(client, sym, "BUY", entry, stop, target, shares)

        if result.get("ok"):
            log.info("  ✅  ORDER PLACED  %s  qty=%d  order_id=%s",
                     sym, shares, result["order_id"])
            pos = {
                "symbol":      sym,
                "side":        "BUY",
                "qty":         shares,
                "entry_date":  date.today().isoformat(),
                "entry_price": entry,
                "stop":        stop,
                "target":      target,
                "score":       int(row["Score"]),
                "conviction":  row["Conviction"],
                "rr":          float(row["R/R"]),
                "order_id":    result["order_id"],
                "t1":          float(row["T1"]),
                "t2":          float(row["T2"]),
                "t3":          float(row.get("T3", target)),
            }
            _add_position(state, pos)
            _append_day_log({
                "role":       "ENTRY",
                "date":       date.today().isoformat(),
                "time_et":    _now_et().strftime("%H:%M"),
                "symbol":     sym,
                "side":       "BUY",
                "qty":        shares,
                "entry":      entry,
                "stop":       stop,
                "target":     target,
                "score":      int(row["Score"]),
                "conviction": row["Conviction"],
                "rr":         float(row["R/R"]),
                "order_id":   result["order_id"],
            })
            buying_power -= entry * shares
        else:
            log.warning("  ❌  ORDER FAILED  %s  %s", sym, result.get("error", ""))


# ── Exit monitor ──────────────────────────────────────────────────────────────

def _get_current_price(client, symbol: str) -> Optional[float]:
    """Get current price of an equity position from Alpaca."""
    try:
        for p in get_positions(client):
            if p["symbol"] == symbol:
                return float(p["current_price"])
    except Exception:
        pass
    return None


def _update_trailing_stop(pos: Dict, cur_price: float) -> Optional[str]:
    """
    Raise the trailing stop based on price progress through targets.
    Returns a log message if stop was raised, else None.
    Mutates pos["stop"] in place (persisted on next _save_state call).

    Rules (ATR-based, mirrors signal levels):
      Price >= T2 → raise stop to T1 (lock in 1R)
      Price >= T1 → raise stop to entry (breakeven)
    """
    entry = float(pos["entry_price"])
    t1    = float(pos.get("t1", entry))
    t2    = float(pos.get("t2", entry))
    old_stop = float(pos["stop"])
    new_stop = old_stop

    if cur_price >= t2:
        new_stop = max(old_stop, t1)          # raise to T1
    elif cur_price >= t1:
        new_stop = max(old_stop, entry)        # raise to breakeven

    if new_stop > old_stop:
        pos["stop"] = round(new_stop, 2)
        return f"trailing stop raised {old_stop:.2f} → {new_stop:.2f}"
    return None


def _check_exits(client, cfg: dict, state: Dict) -> None:
    """Close positions that hit trailing stop, URGENT signals, or max-hold-days."""
    if not state["positions"]:
        return

    _sync_closed_positions(client, state)
    if not state["positions"]:
        return

    exit_cfg    = cfg["exit"]
    max_hold    = int(exit_cfg.get("max_hold_days", 10))
    urgent_exit = bool(exit_cfg.get("urgent_exit", True))

    held_syms = [p["symbol"] for p in state["positions"]]
    log.info("Checking exits for: %s", ", ".join(held_syms))

    exit_signals_df = None
    if urgent_exit:
        try:
            exit_signals_df = generate_exit_signals(held_syms)
        except Exception as e:
            log.warning("generate_exit_signals error: %s", e)

    to_close: List[tuple] = []

    for pos in list(state["positions"]):
        sym       = pos["symbol"]
        entry_d   = date.fromisoformat(pos["entry_date"])
        days_held = (date.today() - entry_d).days
        exit_reason = None

        # Get live price for trailing stop logic
        cur_price = _get_current_price(client, sym)

        # 0. Update trailing stop (raises stop as price moves through T1/T2)
        if cur_price:
            msg = _update_trailing_stop(pos, cur_price)
            if msg:
                log.info("  %-6s  %s", sym, msg)
                _save_state(state)

        # 1. Trailing stop hit (our software stop — bracket SL handles fast moves)
        if cur_price and cur_price <= float(pos["stop"]):
            exit_reason = "trailing_stop"
            log.info("  %-6s  TRAILING STOP HIT  cur=%.2f <= stop=%.2f",
                     sym, cur_price, pos["stop"])

        # 2. Time exit
        if exit_reason is None and days_held >= max_hold:
            exit_reason = f"max_hold_{max_hold}d"
            log.info("  %-6s  TIME EXIT — held %d days (max=%d)", sym, days_held, max_hold)

        # 3. URGENT technical exit
        if exit_reason is None and urgent_exit and exit_signals_df is not None:
            if sym in exit_signals_df.index:
                row     = exit_signals_df.loc[sym]
                urgency = str(row.get("Exit Urgency", ""))
                signals = str(row.get("Exit Signals", ""))
                if urgency == "URGENT":
                    exit_reason = "urgent_technical"
                    log.info("  %-6s  URGENT EXIT — %s", sym, signals[:120])

        if exit_reason:
            to_close.append((pos, exit_reason, days_held))

    for pos, reason, days_held in to_close:
        sym = pos["symbol"]
        log.info("  EXIT  %-6s  reason=%s  held=%d days", sym, reason, days_held)
        result = close_position(client, sym)

        if result["ok"]:
            log.info("  ✅  CLOSED  %s  order_id=%s", sym, result.get("order_id", ""))
        else:
            log.warning("  ❌  CLOSE FAILED  %s  %s", sym, result.get("error", ""))

        _append_day_log({
            "role":       "EXIT",
            "date":       date.today().isoformat(),
            "time_et":    _now_et().strftime("%H:%M"),
            "symbol":     sym,
            "side":       pos["side"],
            "qty":        pos["qty"],
            "exit_reason": reason,
            "days_held":  days_held,
            "entry":      pos["entry_price"],
            "order_id":   result.get("order_id", ""),
        })
        _remove_position(state, sym)


# ── Monitor loop ──────────────────────────────────────────────────────────────

def _monitor_loop(client, cfg: dict, state: Dict) -> None:
    interval_min = int(cfg["exit"].get("monitor_interval", 15))
    log.info("Monitoring exits every %d min until 15:45 ET…", interval_min)
    while True:
        now = _now_et()
        if now.hour > 15 or (now.hour == 15 and now.minute >= 45):
            break
        _check_exits(client, cfg, state)
        if state["positions"]:
            live = get_positions(client)
            live_map = {p["symbol"]: p for p in live}
            for pos in state["positions"]:
                sym   = pos["symbol"]
                lp    = live_map.get(sym, {})
                cur   = float(lp.get("current_price", 0) or 0)
                pnl   = float(lp.get("unrealized_pl", 0) or 0)
                entry_d  = date.fromisoformat(pos["entry_date"])
                days     = (date.today() - entry_d).days
                log.info("  %-6s  cur=$%.2f  pnl=%+.2f  held=%dd  target=%.2f  stop=%.2f",
                         sym, cur, pnl, days, pos["target"], pos["stop"])
        time.sleep(interval_min * 60)


# ── Day summary ───────────────────────────────────────────────────────────────

def _print_summary(client, state: Dict) -> None:
    acct = get_account_summary(client)
    log.info("=" * 65)
    log.info("STOCK PAPER TRADE SUMMARY — %s", _now_et().strftime("%Y-%m-%d"))
    log.info("=" * 65)
    log.info("Portfolio: $%.2f  |  Cash: $%.2f  |  Open Positions: %d",
             acct["portfolio_value"], acct["cash"], len(state["positions"]))
    live = {p["symbol"]: p for p in get_positions(client)}
    for pos in state["positions"]:
        lp  = live.get(pos["symbol"], {})
        pnl = float(lp.get("unrealized_pl", 0) or 0)
        pct = float(lp.get("unrealized_pl%", 0) or 0)
        entry_d = date.fromisoformat(pos["entry_date"])
        days    = (date.today() - entry_d).days
        log.info("  %-6s  %-4s  qty=%d  held=%dd  est_pnl=%+.2f (%.1f%%)",
                 pos["symbol"], pos["side"], pos["qty"], days, pnl, pct)
    log.info("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    parser = argparse.ArgumentParser(description="Stock daily alerts paper trader")
    parser.add_argument("--config",       default="configs/stock_paper.yaml")
    parser.add_argument("--now",          action="store_true",
                        help="Scan + place orders RIGHT NOW regardless of market hours")
    parser.add_argument("--monitor-only", action="store_true",
                        help="Skip entry scan — only check exits on open positions")
    args = parser.parse_args()

    cfg = load_config(args.config)

    api_key = cfg["broker"].get("api_key") or os.environ.get("ALPACA_API_KEY", "")
    sec_key = cfg["broker"].get("secret_key") or os.environ.get("ALPACA_SECRET_KEY", "")

    if not api_key or not sec_key:
        log.error("No Alpaca credentials. Set ALPACA_API_KEY / ALPACA_SECRET_KEY in .env or config.")
        sys.exit(1)

    client = make_client(api_key, sec_key)
    state  = _load_state()

    log.info("Stock Paper Trader started — universe: %s  |  mode: %s",
             cfg["universe"].get("preset", "custom"),
             cfg["broker"].get("mode", "paper"))

    if args.now:
        log.info("--now flag: running immediately (market hours ignored)")
        _check_exits(client, cfg, state)
        if not args.monitor_only:
            _run_entry_scan(client, cfg, state)
        _print_summary(client, state)
        return

    # Continuous daily loop
    scan_time = cfg["strategy"].get("scan_time", "09:35")
    while True:
        _wait_for_scan_time(scan_time)
        try:
            _check_exits(client, cfg, state)
            if not args.monitor_only:
                _run_entry_scan(client, cfg, state)
            _monitor_loop(client, cfg, state)
            _print_summary(client, state)
        except KeyboardInterrupt:
            log.info("Shutdown requested.")
            _print_summary(client, state)
            break
        except Exception as e:
            log.error("Day run error: %s", e, exc_info=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
