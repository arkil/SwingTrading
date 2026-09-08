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
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from alerts_engine import generate_daily_alerts, generate_exit_signals
from alpaca_trader import (
    make_client,
    get_account_summary,
    get_positions,
    get_todays_trades,
    execute_alerts,
    cancel_order,
    close_position,
    place_simple_order,
    place_stop_order,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

TOP_CONVICTION_N = 3  # only ever enter the top-N alerts by Conviction Score per scan

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
    "pyramid": {
        "enabled":          False,          # BUY-only scale-in/scale-out mode
        "dynamic_stop_pct": 2.0,            # % below T1 the stop moves to once armed
        "sell_mode":        "scale_50_25_25",  # only mode wired to live trading
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
            while wait > 0:
                chunk = min(wait, 3600)
                time.sleep(chunk)
                wait -= chunk
            return
        elif now.hour < 16:
            return
        else:
            secs = _seconds_until(sh, sm)
            log.info("Session ended — sleeping %.0f hours until tomorrow %s ET", secs / 3600, scan_time)
            while secs > 0:
                chunk = min(secs, 3600)
                time.sleep(chunk)
                secs -= chunk
                # re-check wall clock each hour in case a suspend/hibernate
                # event skewed our countdown vs. actual elapsed time
                if _now_et() >= _now_et().replace(hour=sh, minute=sm, second=0, microsecond=0):
                    return


# ── Universe resolver ─────────────────────────────────────────────────────────

def _backtest_universe() -> list:
    """
    Same ~583-ticker universe the pyramid strategy was backtested against:
    S&P 500 + Nasdaq 100 + High-Growth Tech + Watchlist, deduped. Built
    directly from the lightweight ticker-list helpers (not by importing
    dashboard.py, which executes the whole Streamlit script on import).
    """
    from livermore_pivotal_screener import get_sp500_tickers, get_nasdaq100_tickers, WATCHLIST_TICKERS
    seen, out = set(), []
    for t in (get_sp500_tickers() or []) + (get_nasdaq100_tickers() or []) \
             + PRESETS["High-Growth Tech"] + list(WATCHLIST_TICKERS):
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _resolve_universe(cfg: dict) -> list:
    uni    = cfg.get("universe", {})
    custom = [t.strip().upper() for t in uni.get("custom_tickers", []) if t.strip()]
    if custom:
        return custom
    preset = uni.get("preset", "High-Growth Tech")
    if preset == "Backtest Universe (583)":
        try:
            return _backtest_universe()
        except Exception:
            log.warning("Failed to build backtest universe — falling back to High-Growth Tech")
            return PRESETS["High-Growth Tech"]
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


_ENTRIES_TODAY_PATH = os.path.join(os.path.dirname(__file__), "logs", "entries_today.json")


def _entries_placed_today() -> int:
    """How many new positions this runner has already entered today —
    persisted so a service restart mid-day can't reset the top-3-per-day
    conviction cap and place another 3 on top of what already went in."""
    if os.path.exists(_ENTRIES_TODAY_PATH):
        try:
            with open(_ENTRIES_TODAY_PATH) as f:
                d = json.load(f)
            if d.get("date") == date.today().isoformat():
                return int(d.get("count", 0))
        except Exception:
            pass
    return 0


def _record_entries_placed(n: int) -> None:
    if n <= 0:
        return
    count = _entries_placed_today() + n
    os.makedirs(os.path.dirname(_ENTRIES_TODAY_PATH), exist_ok=True)
    with open(_ENTRIES_TODAY_PATH, "w") as f:
        json.dump({"date": date.today().isoformat(), "count": count}, f)


_DUST_VALUE_USD = 1.0  # positions worth less than this don't count as "held"


def _is_dust(position: dict) -> bool:
    """True for leftover sub-share positions (e.g. 0.0001 shares, $0.02)
    that Alpaca still lists after a partial-sell rounding drift. These
    aren't real trades — without this filter one can occupy a
    max_positions slot indefinitely and block all new entries."""
    return abs(position.get("market_value", 0)) < _DUST_VALUE_USD


# ── New cohort tracker ────────────────────────────────────────────────────────
# The 15 legacy positions open in the account before this feature (AFL, APA,
# BG, BMY, CASY, CTVA, EOG, FANG, FRT, JBHT, MO, OXY, PM, UPS, VTRS) are left
# completely alone — same monitoring/exits as always, but they no longer
# occupy a slot for new-entry purposes. This registry tracks a *separate*
# pool of up to NEW_COHORT_MAX positions that the daily top-conviction entry
# logic fills and reports on independently, so its P&L can be judged on its
# own instead of being mixed into (or blocked by) whatever's already open.

NEW_COHORT_MAX  = 15
_COHORT_PATH    = os.path.join(os.path.dirname(__file__), "logs", "new_cohort_positions.json")


def _load_cohort() -> dict:
    if os.path.exists(_COHORT_PATH):
        try:
            with open(_COHORT_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cohort(cohort: dict) -> None:
    os.makedirs(os.path.dirname(_COHORT_PATH), exist_ok=True)
    with open(_COHORT_PATH, "w") as f:
        json.dump(cohort, f, indent=2, default=str)


def _register_cohort_entries(cohort: dict, results: list) -> None:
    for r in results:
        if r.get("ok"):
            cohort[r["symbol"]] = {
                "entry_date":  date.today().isoformat(),
                "entry_price": r["entry"],
                "qty":         r["qty"],
                "confirmed":   False,
            }
    _save_cohort(cohort)


def _sync_cohort(cohort: dict, live_symbols: set) -> list:
    """Drop cohort symbols whose Alpaca position is gone (SL/TP filled, or
    closed by the exit-signal/time-exit logic) — frees their slot for the
    next day's top-conviction entries. Returns the symbols removed.

    Bracket entries are resting LIMIT orders, not market fills — a symbol
    can be absent from live_symbols for a while simply because the entry
    hasn't filled yet. Only treat "absent" as a real exit once we've seen
    the symbol confirmed live at least once; otherwise every fresh entry
    gets falsely marked auto-exited on the very next monitor tick.
    """
    auto_exited = []
    for sym in list(cohort.keys()):
        if sym in live_symbols:
            cohort[sym]["confirmed"] = True
        elif cohort[sym].get("confirmed"):
            auto_exited.append(sym)
            del cohort[sym]
        # else: entry order still resting/unfilled — leave it, recheck next tick
    if auto_exited:
        _save_cohort(cohort)
    return auto_exited


def _days_held(tracker: dict, symbol: str) -> int:
    entry = tracker.get(symbol, {}).get("entry_time", "")
    if not entry:
        return 0
    try:
        return (datetime.now() - datetime.fromisoformat(entry)).days
    except Exception:
        return 0


# ── Pyramid position state ─────────────────────────────────────────────────────
# Kept in its own file, separate from position_tracker.json — the generic
# tracker above blindly ingests every symbol in the shared Alpaca paper
# account (including option contracts opened by the separate options bot),
# so pyramid state only ever holds symbols this runner itself entered.

_PYRAMID_STATE_PATH = os.path.join(os.path.dirname(__file__), "logs", "pyramid_state.json")


def _load_pyramid_state() -> dict:
    if os.path.exists(_PYRAMID_STATE_PATH):
        try:
            with open(_PYRAMID_STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_pyramid_state(state: dict) -> None:
    os.makedirs(os.path.dirname(_PYRAMID_STATE_PATH), exist_ok=True)
    with open(_PYRAMID_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _pyramid_cancel_stop(client, st: dict) -> None:
    """Release the resting stop's held qty before any buy/sell touches this
    symbol — a resting stop-sell holds 100% of the shares as collateral and
    trips Alpaca's wash-trade guard on a BUY, or "insufficient qty available"
    on a SELL, until it's cancelled."""
    if st.get("stop_order_id"):
        cancel_order(client, st["stop_order_id"])
        st["stop_order_id"] = None


def _pyramid_replace_stop(client, sym: str, st: dict, qty: float, stop_price: float) -> None:
    """(Re)place the protective stop for `qty` shares at `stop_price`.

    Alpaca requires whole-share quantities on GTC orders ("fractional
    orders must be DAY orders") — a stop needs GTC to survive past the
    entry day, so floor to the nearest whole share. Add-leg buys size by
    notional and routinely leave a sub-share remainder (e.g. 13.2997);
    that dust is left unprotected, which is immaterial next to leaving
    the whole position unprotected because the stop request got rejected.
    """
    whole_qty = math.floor(qty)
    st["current_stop"] = stop_price
    if whole_qty <= 0:
        st["stop_order_id"] = None
        return
    result = place_stop_order(client, sym, whole_qty, stop_price)
    st["stop_order_id"] = result.get("order_id") if result.get("ok") else None
    if not result.get("ok"):
        log.warning("  ⚠️  %-6s  replacement stop order FAILED — position is UNPROTECTED  %s",
                    sym, result.get("error", ""))
    elif whole_qty < qty:
        log.info("  🛡️  %-6s  stop covers %d of %.4f shares (fractional remainder uncovered)",
                  sym, whole_qty, qty)


def _pyramid_add_leg(client, state: dict, sym: str, st: dict, leg_name: str) -> None:
    """Scale-in: buy another `leg_notional` worth of shares at add_level_1/2."""
    price = st["add_level_1"] if leg_name == "leg1" else st["add_level_2"]
    add_qty = round(st["leg_notional"] / price, 4)

    _pyramid_cancel_stop(client, st)

    result = place_simple_order(client, sym, "BUY", add_qty, order_type="market")
    if result.get("ok"):
        st["total_qty"] += add_qty
        st["added_leg1" if leg_name == "leg1" else "added_leg2"] = True
        log.info("  ➕  %-6s  ADD %s  qty=%.2f @ ~%.2f  total_qty=%.2f",
                 sym, leg_name, add_qty, price, st["total_qty"])
    else:
        log.warning("  ❌  %-6s  ADD %s FAILED  %s", sym, leg_name, result.get("error", ""))

    # Whether the buy succeeded or not, we cancelled the only protection
    # above — restore it for whatever qty is actually held now.
    _pyramid_replace_stop(client, sym, st, st["total_qty"], st["current_stop"])


def _pyramid_sell_t1(client, state: dict, sym: str, st: dict, dynamic_stop_pct: float) -> None:
    """Sell 50% of the (now-locked) position at T1, arm the dynamic stop for the rest."""
    sell_qty = round(0.5 * st["total_qty"], 4)

    _pyramid_cancel_stop(client, st)

    result = place_simple_order(client, sym, "SELL", sell_qty, order_type="market")
    if not result.get("ok"):
        log.warning("  ❌  %-6s  SELL@T1 FAILED  %s", sym, result.get("error", ""))
        # Sell failed after we released the stop — restore protection for
        # the full (unchanged) position immediately.
        _pyramid_replace_stop(client, sym, st, st["total_qty"], st["current_stop"])
        return

    st["sold_t1"] = True
    remaining = round(st["total_qty"] - sell_qty, 4)
    new_stop  = st["t1"] * (1 - dynamic_stop_pct / 100)

    _pyramid_replace_stop(client, sym, st, remaining, new_stop)

    log.info("  💰  %-6s  SELL 50%% @ T1 (%.2f)  qty=%.2f  new stop=%.2f", sym, st["t1"], sell_qty, new_stop)


def _pyramid_sell_partial(client, sym: str, qty: float, label: str) -> bool:
    result = place_simple_order(client, sym, "SELL", qty, order_type="market")
    if result.get("ok"):
        log.info("  💰  %-6s  SELL %s  qty=%.2f", sym, label, qty)
        return True
    log.warning("  ❌  %-6s  SELL %s FAILED  %s", sym, label, result.get("error", ""))
    return False


def _pyramid_close(client, state: dict, sym: str, st: dict, qty: float, label: str) -> None:
    """Sell whatever remains, cancel the resting stop order, drop from state.

    Uses close_position() rather than selling the caller's computed `qty` —
    that figure is derived from total_qty * (1 - sold_fractions) and drifts
    from the real Alpaca balance after a couple of rounded partial sells,
    leaving a sub-share dust position (e.g. PSX at 0.0001 shares) that then
    sits in the account forever, silently occupying a max_positions slot.
    close_position() always zeroes the actual live quantity.
    """
    _pyramid_cancel_stop(client, st)
    if qty > 0:
        result = close_position(client, sym)
        if result.get("ok"):
            log.info("  💰  %-6s  CLOSE %s  qty=%.4f", sym, label, qty)
        else:
            log.warning("  ❌  %-6s  CLOSE %s FAILED  %s", sym, label, result.get("error", ""))
            # Sell failed after we released the stop — restore protection
            # rather than leaving the position naked.
            _pyramid_replace_stop(client, sym, st, qty, st["current_stop"])
            return
    del state[sym]


def _manage_pyramid_positions(client, cfg: dict, state: dict) -> None:
    """
    Per pyramid symbol, every monitor tick:
      add_level_2 -> add_level_1 (scale-in, same $ as entry)
      T1 (sell 50%, arm dynamic stop) -> T2 (sell 25%) -> T3 (sell remaining)
      stop hit -> sell remaining
    Mirrors backtest_strategy.py's sell_mode="scale_50_25_25" branch exactly.
    """
    if not state:
        return

    dynamic_stop_pct = float(cfg.get("pyramid", {}).get("dynamic_stop_pct", 2.0))
    live = {p["symbol"]: p for p in get_positions(client)}

    for sym in list(state.keys()):
        st = state[sym]
        pos = live.get(sym)

        if pos is None:
            if not st.get("confirmed"):
                # Entry limit order hasn't filled yet (still resting) — this
                # is NOT a stop firing, just an unconfirmed entry. Leave it
                # in state and check again next tick.
                log.info("  ⏳  %-6s  entry still unfilled — checking again next tick", sym)
                continue
            # Was confirmed live before, now gone — the resting GTC stop
            # (or a manual/urgent close) must have fired.
            log.info("  🛑  %-6s  position no longer at Alpaca — resting stop fired, "
                     "removing from pyramid state", sym)
            del state[sym]
            _save_pyramid_state(state)
            continue

        if not st.get("confirmed"):
            st["confirmed"] = True

        # Safety net: entry filled (position exists) but the initial stop
        # attempt was rejected or deferred — place it now that the entry
        # order is no longer resting, so Alpaca's wash-trade check won't fire.
        # Sized off the live Alpaca qty, not total_qty (which stays fixed at
        # the original entry basis after T1/T2 partial sells and would
        # overstate what's actually left to protect).
        if not st.get("stop_order_id"):
            _pyramid_replace_stop(client, sym, st, float(pos["qty"]), st["current_stop"])
            if st.get("stop_order_id"):
                log.info("  🛡️  %-6s  protective stop placed (was missing) @ %.2f",
                         sym, st["current_stop"])
            else:
                log.warning("  ⚠️  %-6s  still unable to place protective stop", sym)

        cur_price = float(pos["current_price"])
        qty_live  = float(pos["qty"])
        # total_qty is the fixed basis for the 50/25/25 fractions once T1 has
        # fired — only resync it pre-T1, where live qty should track it 1:1
        # (partial sells afterward are *expected* to drop live qty below it).
        if not st["sold_t1"] and qty_live < st["total_qty"] - 0.01:
            st["total_qty"] = qty_live  # defensive resync

        # 1. Stop check first (worst case)
        if cur_price <= st["current_stop"]:
            log.info("  🔴  %-6s  STOP HIT  cur=%.2f <= stop=%.2f", sym, cur_price, st["current_stop"])
            _pyramid_close(client, state, sym, st, st["total_qty"], "STOP")
            _save_pyramid_state(state)
            continue

        # 2. Scale-in adds (below T1 only)
        if cur_price >= st["add_level_2"] and not st["added_leg2"]:
            if not st["added_leg1"]:
                _pyramid_add_leg(client, state, sym, st, "leg1")
            _pyramid_add_leg(client, state, sym, st, "leg2")
        elif cur_price >= st["add_level_1"] and not st["added_leg1"]:
            _pyramid_add_leg(client, state, sym, st, "leg1")

        # 3. Scale-out sells
        if cur_price >= st["t3"]:
            # Catch-up: honor any 50%/25% tranches this position gapped past
            # before selling whatever's left, same as the backtest's same-bar cascade.
            if not st["sold_t1"]:
                _pyramid_sell_t1(client, state, sym, st, dynamic_stop_pct)
            if not st["sold_t2"]:
                _pyramid_cancel_stop(client, st)
                qty_t2 = round(0.25 * st["total_qty"], 4)
                if _pyramid_sell_partial(client, sym, qty_t2, "T2"):
                    st["sold_t2"] = True
                else:
                    # Sell failed after we released the stop — restore
                    # protection for whatever's still held before the T3
                    # close call below (it re-cancels harmlessly if unsold).
                    _pyramid_replace_stop(client, sym, st, st["total_qty"], st["current_stop"])
            # Remaining reflects only tranches that actually sold (a failed
            # order leaves its flag False and its shares still open).
            sold_frac = 0.5 * float(st["sold_t1"]) + 0.25 * float(st["sold_t2"])
            remaining = round(st["total_qty"] * (1 - sold_frac), 4)
            _pyramid_close(client, state, sym, st, remaining, "T3")
        elif cur_price >= st["t2"] and not st["sold_t2"]:
            if not st["sold_t1"]:
                _pyramid_sell_t1(client, state, sym, st, dynamic_stop_pct)
            _pyramid_cancel_stop(client, st)
            qty_t2 = round(0.25 * st["total_qty"], 4)
            if _pyramid_sell_partial(client, sym, qty_t2, "T2"):
                st["sold_t2"] = True
            # Restore protection for whatever remains after this tick's sells
            # — this branch (unlike T3) keeps holding the rest of the position.
            sold_frac = 0.5 * float(st["sold_t1"]) + 0.25 * float(st["sold_t2"])
            remaining = round(st["total_qty"] * (1 - sold_frac), 4)
            _pyramid_replace_stop(client, sym, st, remaining, st["current_stop"])
        elif cur_price >= st["t1"] and not st["sold_t1"]:
            _pyramid_sell_t1(client, state, sym, st, dynamic_stop_pct)

        _save_pyramid_state(state)


# ── Day runner ────────────────────────────────────────────────────────────────

def _run_day(client, cfg: dict, cohort: dict) -> None:
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

    live_positions = get_positions(client)
    live_symbols   = {p["symbol"] for p in live_positions if not _is_dust(p)}

    cohort_auto_exited = _sync_cohort(cohort, live_symbols)
    for sym in cohort_auto_exited:
        log.info("  🎯  [new cohort] AUTO-EXIT  %s  (bracket SL/TP filled by Alpaca)", sym)

    # Entry-slot gating is scoped to the tracked new-position cohort only —
    # the legacy positions already open in the account (pre-dating this
    # feature) are monitored/exited as before but no longer occupy a slot.
    available = NEW_COHORT_MAX - len(cohort)
    if available <= 0:
        log.info("New cohort already at cap (%d/%d tracked positions) — skipping entry scan.",
                 len(cohort), NEW_COHORT_MAX)
        return

    entries_today   = _entries_placed_today()
    remaining_today = TOP_CONVICTION_N - entries_today
    if remaining_today <= 0:
        log.info("Already entered %d/%d top-conviction positions today — skipping entry scan "
                 "(more room opens tomorrow).", entries_today, TOP_CONVICTION_N)
        return

    tickers   = _resolve_universe(cfg)
    min_score = int(strat.get("min_score", 7))
    direction = strat.get("direction", "BUY").upper()

    # VIX regime: this account is long-biased (BUY is the default direction),
    # so there's no "switch to puts" equivalent — instead raise the bar or
    # pause new BUY entries when fear is elevated. Same regime source and
    # tiers as options_paper_trader.py's call/put switching.
    from market_context import get_regime_summary
    regime = get_regime_summary()
    if regime["label"] == "HIGH-VOL" and direction in ("BUY", "ALL"):
        log.info("VIX rank=%.0f — HIGH-VOL regime, pausing new BUY entries this run",
                 regime["vix"]["rank_1y"])
        if direction == "BUY":
            return
        direction = "SELL"
    elif regime["label"] == "RISK-OFF" and direction in ("BUY", "ALL"):
        min_score = max(min_score, min_score + 2)
        log.info("VIX rank=%.0f — RISK-OFF regime, raised min_score to %d for new BUY entries",
                 regime["vix"]["rank_1y"], min_score)

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
    risk_pct      = float(strat.get("risk_pct", 0.5)) / 100
    max_pct       = float(strat.get("max_position_pct", 5.0)) / 100
    max_dollars   = float(strat.get("max_dollars_per_trade", 1000))

    # Cap new entries to the top-conviction alerts by Conviction Score (Score +
    # R/R + RS + ADX + Vol Ratio + Minervini composite, see
    # alerts_engine._add_conviction_score), bounded by both the new-cohort
    # slot cap (`available`, NEW_COHORT_MAX total) and TOP_CONVICTION_N per
    # calendar day (`remaining_today`, restart-safe via _entries_placed_today).
    to_place = min(available, remaining_today)
    log.info("Placing up to %d bracket orders (top-conviction, %d/%d entered today, "
             "risk=%.1f%%  cap=$%.0f/trade)…",
             to_place, entries_today, TOP_CONVICTION_N, risk_pct * 100, max_dollars)

    results = execute_alerts(
        client            = client,
        alerts_df         = alerts_df,
        risk_pct          = risk_pct,
        max_position_pct  = max_pct,
        max_new_positions = to_place,
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

    _record_entries_placed(placed)
    _register_cohort_entries(cohort, results)
    log.info("Orders placed: %d / %d attempted  (%d/%d entered today  |  new cohort now %d/%d)",
              placed, len(results), entries_today + placed, TOP_CONVICTION_N,
              len(cohort), NEW_COHORT_MAX)


# ── Position monitor ──────────────────────────────────────────────────────────

def _monitor_positions(client, cfg: dict, tracker: dict, pyramid_state: dict = None,
                        cohort: dict = None) -> None:
    """
    Loop every monitor_interval_min minutes until 16:00 ET:
      - Run pyramid scale-in/scale-out management (if any pyramid positions open)
      - Sync tracker (detect auto-exits via bracket SL/TP)
      - Sync new-position cohort (detect auto-exits, free the slot)
      - Log unrealized P&L and days held per position
      - Run exit signal analysis on held symbols
      - Close positions that hit URGENT signals or max_hold_days
        (also tears down pyramid state/stop order for pyramid symbols,
        and drops the symbol from the cohort registry if it's tracked)
    """
    risk         = cfg["risk"]
    interval_min = int(risk.get("monitor_interval_min", 15))
    max_hold     = int(risk.get("max_hold_days", 20))
    do_urgent    = bool(risk.get("urgent_exit", True))
    do_time      = bool(risk.get("time_exit", True))
    pyramid_state = pyramid_state if pyramid_state is not None else {}
    cohort        = cohort if cohort is not None else {}

    log.info("Monitor started — checking every %d min, max_hold=%d days, "
             "urgent_exit=%s, time_exit=%s",
             interval_min, max_hold, do_urgent, do_time)

    while True:
        now = _now_et()
        if now.hour >= 16:
            log.info("16:00 ET — session ended, stopping monitor.")
            break

        _manage_pyramid_positions(client, cfg, pyramid_state)

        positions = get_positions(client)

        # ── Sync tracker ──────────────────────────────────────────────────────
        auto_exited = _sync_tracker(tracker, positions)
        for sym in auto_exited:
            log.info("  🎯  AUTO-EXIT  %s  (bracket SL/TP filled by Alpaca)", sym)

        live_symbols = {p["symbol"] for p in positions if not _is_dust(p)}
        cohort_auto_exited = _sync_cohort(cohort, live_symbols)
        for sym in cohort_auto_exited:
            log.info("  🎯  [new cohort] AUTO-EXIT  %s  (bracket SL/TP filled by Alpaca)", sym)

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
            tag    = "[NEW] " if sym in cohort else ""
            log.info(
                "  %s%-8s  qty=%.0f  entry=$%.2f  curr=$%.2f  "
                "P&L=$%+.2f (%+.1f%%)  held=%dd  [%s]",
                tag, sym, p["qty"], p["entry_price"], p["current_price"],
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
                # Pyramid symbols keep a resting GTC stop order — cancel it
                # before close_position(), else it's left dangling against a
                # position that's about to disappear.
                if sym in pyramid_state:
                    stop_id = pyramid_state[sym].get("stop_order_id")
                    if stop_id:
                        cancel_order(client, stop_id)
                result = close_position(client, sym)
                if result.get("ok"):
                    log.info("  ✅  %s closed (order_id=%s)", sym, result.get("order_id", "?"))
                    tracker.pop(sym, None)
                    _save_tracker(tracker)
                    if sym in pyramid_state:
                        del pyramid_state[sym]
                        _save_pyramid_state(pyramid_state)
                    if sym in cohort:
                        del cohort[sym]
                        _save_cohort(cohort)
                else:
                    log.error("  ❌  Close failed for %s: %s", sym, result.get("error", ""))

        log.info("─" * 60)
        time.sleep(interval_min * 60)


# ── Day-end summary ───────────────────────────────────────────────────────────

def _print_day_summary(client, cohort: dict = None) -> None:
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

    if cohort:
        live = {p["symbol"]: p for p in get_positions(client)}
        cohort_pl = sum(live[s]["unrealized_pl"] for s in cohort if s in live)
        log.info("-" * 60)
        log.info("New cohort:      %d/%d tracked  |  Unrealized P&L: $%+.2f",
                  len(cohort), NEW_COHORT_MAX, cohort_pl)
        for s in cohort:
            if s in live:
                p = live[s]
                log.info("  %-8s  qty=%.4f  entry=$%.2f  curr=$%.2f  P&L=$%+.2f",
                          s, p["qty"], p["entry_price"], p["current_price"], p["unrealized_pl"])
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

    paper         = str(broker.get("mode", "paper")).lower() == "paper"
    client        = make_client(api_key, sec_key, paper=paper)
    tracker       = _load_tracker()
    pyramid_state = _load_pyramid_state()
    cohort        = _load_cohort()
    scan_time     = cfg["strategy"].get("scan_time", "09:35")
    pyramid_on    = bool(cfg.get("pyramid", {}).get("enabled", False))

    log.info("Daily Alerts Live Runner started")
    log.info("Broker: %s  |  Scan time: %s ET  |  Universe: %s",
             broker.get("mode", "paper"), scan_time,
             cfg.get("universe", {}).get("preset", "custom"))
    log.info("New cohort: %d/%d tracked  |  Entries/day cap: %d  |  Max hold: %d days",
             len(cohort), NEW_COHORT_MAX, TOP_CONVICTION_N,
             int(cfg["risk"].get("max_hold_days", 20)))
    if pyramid_on:
        log.info("Pyramid mode ENABLED (managing %d legacy leg(s) only — "
                  "new entries use standard bracket orders)  |  sell_mode=%s  |  dynamic_stop_pct=%.1f%%",
                  len(pyramid_state),
                  cfg["pyramid"].get("sell_mode", "scale_50_25_25"),
                  float(cfg["pyramid"].get("dynamic_stop_pct", 2.0)))

    def _run_entry(c, cf):
        # New signals always go through the plain bracket-order path.
        # Pyramid mode only manages positions already in pyramid_state
        # (opened before this change) via _manage_pyramid_positions in the
        # monitor loop — no new pyramid entries are opened going forward.
        _run_day(c, cf, cohort)

    if args.once:
        _run_entry(client, cfg)
        _print_day_summary(client, cohort)
        return

    if args.monitor_only:
        _monitor_positions(client, cfg, tracker, pyramid_state, cohort)
        _print_day_summary(client, cohort)
        return

    # Continuous daily loop
    while True:
        _wait_for_market_open(scan_time)
        try:
            _run_entry(client, cfg)
            _monitor_positions(client, cfg, tracker, pyramid_state, cohort)
            _print_day_summary(client, cohort)
        except KeyboardInterrupt:
            log.info("Shutdown requested.")
            _print_day_summary(client, cohort)
            break
        except Exception as e:
            log.error("Day run error: %s", e, exc_info=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
