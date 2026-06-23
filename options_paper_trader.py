"""
Options 45-60 DTE Paper Trader
================================
Runs the swing_options_45_60d strategy live on an Alpaca paper account.

Flow (each trading day)
-----------------------
1. 09:35 ET — scan universe with run_swing_options_screener()
2. For each signal: find the right option contract on Alpaca, buy it
3. Every 15 min — check open positions for stop / take-profit / DTE / max-hold exits
4. 15:45 ET — print day summary, log to options_trade_log

State is persisted to logs/options_positions.json so exits survive restarts.

Usage
-----
    python options_paper_trader.py --config configs/options_paper.yaml
    python options_paper_trader.py --once          # one scan + exit, no loop
    python options_paper_trader.py --monitor-only  # only run exit checks
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
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from swing_options_screener import run_swing_options_screener, DEFAULT_PARAMS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ET          = ZoneInfo("America/New_York")
STATE_FILE  = Path(__file__).parent / "logs" / "options_positions.json"
LOG_DIR     = Path(__file__).parent / "logs" / "options_45_60d"

CONTRACTS_PER_100 = 100
COMMISSION        = 0.65   # per contract per leg

# ── Default config ────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "broker": {
        "mode": "paper",
        "api_key": "",
        "secret_key": "",
    },
    "universe": {
        "preset": "Tech Leaders",
        "custom_tickers": [],
    },
    "strategy": {
        "scan_time":        "09:35",
        "min_score":        7.0,
        "max_positions":    4,
        "pct_per_trade":    5.0,       # % of portfolio per position
        "dte_target":       50,        # target DTE at entry
        "iv_premium":       1.10,
        "direction":        "ALL",     # ALL | LONG | SHORT
    },
    "exit": {
        "tp_pct":           0.50,      # take profit at +50%
        "sl_pct":           0.25,      # stop loss at -25%
        "dte_exit":         21,        # exit when DTE <= this
        "max_hold_days":    21,
        "monitor_interval": 15,        # minutes between exit checks
    },
}

PRESETS = {
    "Tech Leaders":    ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AMD", "TSLA", "AVGO", "CRM"],
    "High-Growth":     ["NVDA", "AMD", "TSLA", "META", "AMZN", "SHOP", "CRWD", "PLTR", "COIN", "HOOD"],
    "S&P 500 Leaders": ["SPY",  "QQQ",  "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "V",   "MA"],
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


# ── Alpaca client ─────────────────────────────────────────────────────────────

def _make_client(api_key: str, secret_key: str):
    from alpaca.trading.client import TradingClient
    return TradingClient(api_key, secret_key, paper=True)


# ── Market clock helpers ──────────────────────────────────────────────────────

def _now_et() -> datetime:
    return datetime.now(ET)


def _parse_scan_time(s: str):
    h, m = map(int, s.split(":"))
    return h, m


def _wait_for_scan_time(scan_time: str) -> None:
    sh, sm = _parse_scan_time(scan_time)
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
            return   # already past scan time but still in session
        else:
            # after close — wait until tomorrow's scan time
            tomorrow = now + timedelta(days=1)
            target   = tomorrow.replace(hour=sh, minute=sm, second=0, microsecond=0)
            wait = (target - now).total_seconds()
            log.info("Session ended — sleeping %.0f hours until tomorrow", wait / 3600)
            time.sleep(min(wait, 3600))


# ── Option contract finder ─────────────────────────────────────────────────────

def _next_expiry_in_range(dte_target: int, dte_min: int = 40, dte_max: int = 65,
                          ) -> List[date]:
    """Return all Fridays (option expiry days) within the DTE window."""
    today    = date.today()
    expiries = []
    for delta in range(dte_min, dte_max + 1):
        d = today + timedelta(days=delta)
        if d.weekday() == 4:   # Friday
            expiries.append(d)
    if not expiries:
        # fallback: nearest Friday to dte_target
        target_d = today + timedelta(days=dte_target)
        # Walk to nearest Friday
        while target_d.weekday() != 4:
            target_d += timedelta(days=1)
        expiries = [target_d]
    return expiries


def _build_occ_symbol(underlying: str, expiry: date,
                      direction: str, strike: float) -> str:
    """Build OCC option symbol e.g. AAPL240315C00160000."""
    exp_str    = expiry.strftime("%y%m%d")
    cp         = "C" if direction == "call" else "P"
    # Strike in 8-digit format = strike * 1000 (e.g. $160.00 → 00160000)
    strike_int = round(strike * 1000)
    return f"{underlying}{exp_str}{cp}{strike_int:08d}"


def find_option_contract(client, underlying: str, direction: str,
                         target_strike: float, dte_target: int = 50) -> Optional[str]:
    """
    Find the best-matching option contract on Alpaca for the given params.
    Returns the OCC symbol string, or None if not found.

    Queries the full DTE window in one call and picks the contract with
    strike closest to target_strike. No Friday-detection fallback — only
    returns contracts that actually exist on Alpaca.
    """
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import ContractType

    today   = date.today()
    exp_min = today + timedelta(days=max(30, dte_target - 15))
    exp_max = today + timedelta(days=dte_target + 20)
    cp_type = ContractType.CALL if direction == "call" else ContractType.PUT

    # Search ±10% around target strike to get enough candidates
    strike_lo = round(target_strike * 0.90, 2)
    strike_hi = round(target_strike * 1.10, 2)

    best_sym   = None
    best_dist  = float("inf")

    try:
        req = GetOptionContractsRequest(
            underlying_symbols  = [underlying],
            type                = cp_type,
            expiration_date_gte = exp_min,
            expiration_date_lte = exp_max,
            strike_price_gte    = str(strike_lo),
            strike_price_lte    = str(strike_hi),
            limit               = 50,
        )
        resp      = client.get_option_contracts(req)
        contracts = resp.option_contracts or []

        for c in contracts:
            k    = float(c.strike_price)
            dist = abs(k - target_strike)
            if dist < best_dist:
                best_dist = dist
                best_sym  = c.symbol

    except Exception as e:
        log.warning("Contract lookup failed for %s: %s", underlying, e)

    if best_sym:
        log.debug("  Best contract for %s %s: %s (strike dist=%.2f)",
                  underlying, direction, best_sym, best_dist)
    else:
        log.warning("  No %s %s contracts found on Alpaca for %s (strike ~%.0f, DTE ~%d)",
                    direction, underlying, underlying, target_strike, dte_target)

    return best_sym


# ── Order placement ───────────────────────────────────────────────────────────

def place_option_buy(client, symbol: str, qty: int) -> Dict:
    """Place a market buy order for an option contract."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums    import OrderSide, TimeInForce
    try:
        req = MarketOrderRequest(
            symbol        = symbol,
            qty           = qty,
            side          = OrderSide.BUY,
            time_in_force = TimeInForce.DAY,
        )
        order = client.submit_order(req)
        return {"ok": True, "order_id": str(order.id), "symbol": symbol, "qty": qty}
    except Exception as e:
        return {"ok": False, "symbol": symbol, "error": str(e)}


def place_option_sell(client, symbol: str, qty: int, reason: str = "") -> Dict:
    """Place a market sell order for an option contract."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums    import OrderSide, TimeInForce
    try:
        req = MarketOrderRequest(
            symbol        = symbol,
            qty           = qty,
            side          = OrderSide.SELL,
            time_in_force = TimeInForce.DAY,
        )
        order = client.submit_order(req)
        return {"ok": True, "order_id": str(order.id), "symbol": symbol, "qty": qty,
                "reason": reason}
    except Exception as e:
        return {"ok": False, "symbol": symbol, "error": str(e)}


# ── Current option price via Alpaca positions ─────────────────────────────────

def _get_position_price(client, symbol: str) -> Optional[float]:
    """Get current market price for an open option position."""
    try:
        positions = client.get_all_positions()
        for p in positions:
            if str(p.symbol) == symbol:
                return float(p.current_price or 0) or None
    except Exception:
        pass
    return None


def _get_underlying_price(symbol: str) -> Optional[float]:
    """Get current underlying price via yfinance."""
    try:
        import yfinance as yf
        tk  = yf.Ticker(symbol)
        hist = tk.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def _get_account(client) -> Dict:
    try:
        acct = client.get_account()
        return {
            "equity":       float(acct.equity),
            "cash":         float(acct.cash),
            "buying_power": float(acct.buying_power),
        }
    except Exception:
        return {"equity": 0.0, "cash": 0.0, "buying_power": 0.0}


# ── State file ────────────────────────────────────────────────────────────────

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


def _remove_position(state: Dict, option_symbol: str) -> None:
    state["positions"] = [p for p in state["positions"]
                          if p["option_symbol"] != option_symbol]
    _save_state(state)


# ── Day log storage ────────────────────────────────────────────────────────────

def _append_day_log(entry: Dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    d_str = entry.get("date", date.today().isoformat())
    log_path = LOG_DIR / f"options_45_60d_{d_str}.json"
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text())
        except Exception:
            data = {"date": d_str, "orders": []}
    else:
        data = {"date": d_str, "orders": []}
    data["orders"].append(entry)
    log_path.write_text(json.dumps(data, indent=2, default=str))


# ── VIX regime ────────────────────────────────────────────────────────────────

def _get_vix() -> float:
    """Return current VIX level. Returns 20.0 on failure (neutral default)."""
    try:
        import yfinance as yf
        vix = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
        if not vix.empty:
            return float(vix["Close"].squeeze().iloc[-1])
    except Exception:
        pass
    return 20.0


# ── Entry scan ────────────────────────────────────────────────────────────────

def _run_entry_scan(client, cfg: dict, state: Dict) -> None:
    strat   = cfg["strategy"]
    tickers = _resolve_universe(cfg)
    min_score      = float(strat.get("min_score", 7.5))
    max_pos        = int(strat.get("max_positions", 4))
    pct_per        = float(strat.get("pct_per_trade", 5.0)) / 100.0
    dte_target     = int(strat.get("dte_target", 52))
    iv_premium     = float(strat.get("iv_premium", 1.10))
    direction      = strat.get("direction", "ALL").upper()
    vix_call_max   = float(strat.get("vix_call_max", 22.0))
    vix_puts_only  = float(strat.get("vix_puts_only", 25.0))

    # VIX regime: adjust allowed direction based on fear level
    vix = _get_vix()
    now_et = _now_et()
    in_q3  = now_et.month in (7, 8, 9)

    if vix >= vix_puts_only:
        direction = "PUT"
        log.info("VIX=%.1f ≥ %.0f — puts-only regime", vix, vix_puts_only)
    elif vix >= vix_call_max:
        if direction in ("ALL", "CALL", "LONG"):
            direction = "PUT"
        log.info("VIX=%.1f ≥ %.0f — calls blocked, puts only", vix, vix_call_max)
    elif in_q3 and direction in ("ALL", "CALL", "LONG"):
        direction = "PUT"
        log.info("Q3 call block active (Jul-Sep 33%% WR) — puts only")
    else:
        log.info("VIX=%.1f — normal regime, direction=%s", vix, direction)

    open_syms   = {p["underlying"] for p in state["positions"]}
    available   = max_pos - len(state["positions"])
    if available <= 0:
        log.info("Max positions (%d) reached — skipping entry scan", max_pos)
        return

    # Puts require higher conviction (backtest: 8.0 optimal for puts)
    effective_score = 8.0 if direction == "PUT" else min_score

    log.info("Scanning %d tickers for options signals (min_score=%.1f, vix=%.1f, dir=%s)…",
             len(tickers), effective_score, vix, direction)

    # Pass any Greeks overrides from config into the screener
    greeks_params = {k: v for k, v in strat.items()
                     if k in ("gamma_min", "gamma_max", "delta_min", "delta_max",
                              "theta_vega_ratio_max", "theta_max_daily_pct", "max_entry_sigma")}

    try:
        signals_df = run_swing_options_screener(
            tickers            = tickers,
            min_score          = effective_score,
            dte                = float(dte_target),
            iv_premium         = iv_premium,
            params             = greeks_params if greeks_params else None,
            progress_cb        = lambda pct, msg: log.debug("  [%.0f%%] %s", pct * 100, msg),
        )
    except Exception as e:
        log.error("Screener error: %s", e)
        return

    if signals_df is None or signals_df.empty:
        log.info("No signals met criteria today.")
        return

    total_before = len(signals_df)
    log.info("Raw signals: %d", total_before)
    for _, r in signals_df.iterrows():
        log.info("  %-6s  %-5s  score=%.1f  greeks=%s",
                 r["Symbol"], r["Direction"], r["Score"], r.get("Greeks OK", "?"))

    # ── Greeks filter: only trade when all 4 Greek conditions pass ──────────
    if "_passes_greeks" in signals_df.columns:
        signals_df = signals_df[signals_df["_passes_greeks"] == True]
    elif "Greeks OK" in signals_df.columns:
        signals_df = signals_df[signals_df["Greeks OK"] == "✅"]

    log.info("After Greeks filter: %d / %d", len(signals_df), total_before)

    if signals_df.empty:
        log.info("No signals passed Greek filter today — no orders placed.")
        return

    # Filter by direction (screener uses CALL/PUT; config uses LONG/SHORT or CALL/PUT)
    if direction in ("LONG", "CALL"):
        signals_df = signals_df[signals_df["Direction"] == "CALL"]
    elif direction in ("SHORT", "PUT"):
        signals_df = signals_df[signals_df["Direction"] == "PUT"]

    # Skip already open underlying
    signals_df = signals_df[~signals_df["Symbol"].isin(open_syms)]

    # Only take top N by score (up to available slots)
    signals_df = signals_df.head(available)

    log.info("Signals queued for entry: %d", len(signals_df))

    acct = _get_account(client)
    portfolio_value = acct["equity"]

    for _, row in signals_df.iterrows():
        underlying  = row["Symbol"]
        direction_r = "call" if row["Direction"] in ("LONG", "CALL") else "put"
        score       = float(row["Score"])
        close       = float(row["Close"])
        strike      = float(row["Strike"])
        premium_est = float(row["Premium"])

        log.info("  Signal: %-6s  %s  score=%.1f  strike=%.0f  est_premium=%.2f",
                 underlying, direction_r.upper(), score, strike, premium_est)

        # Find option contract on Alpaca
        opt_symbol = find_option_contract(
            client, underlying, direction_r, strike, dte_target
        )
        if not opt_symbol:
            log.warning("  ❌  No option contract found for %s — skipping", underlying)
            continue

        log.info("  Contract: %s", opt_symbol)

        # Position size: % of portfolio
        budget     = portfolio_value * pct_per
        # Use estimated premium; fall back to $5 if missing
        px         = premium_est if premium_est > 0.10 else 5.0
        n_contracts = max(1, int(budget / (px * CONTRACTS_PER_100)))

        # Cap at 35% of portfolio to prevent oversizing
        max_cost  = portfolio_value * 0.35
        while n_contracts > 1 and n_contracts * px * CONTRACTS_PER_100 > max_cost:
            n_contracts -= 1

        log.info("  Sizing: %d contracts @ est $%.2f  (budget $%.0f)",
                 n_contracts, px, budget)

        result = place_option_buy(client, opt_symbol, n_contracts)

        if result["ok"]:
            log.info("  ✅  ORDER PLACED  %s  qty=%d  order_id=%s",
                     opt_symbol, n_contracts, result["order_id"])

            # Compute stops using ATR from screener row
            atr_val = float(row.get("ATR", close * 0.02))
            if direction_r == "call":
                stop_underlying   = round(close - 2.0 * atr_val, 2)
                target_underlying = round(close + 4.0 * atr_val, 2)
            else:
                stop_underlying   = round(close + 3.0 * atr_val, 2)
                target_underlying = round(close - 4.0 * atr_val, 2)

            from options_trade_log import _parse_option_symbol
            opt_info = _parse_option_symbol(opt_symbol)
            expiry_str = opt_info["expiry"].isoformat() if opt_info else ""
            dte_actual = (opt_info["expiry"] - date.today()).days if opt_info else dte_target

            pos = {
                "option_symbol":     opt_symbol,
                "underlying":        underlying,
                "direction":         direction_r,
                "qty":               n_contracts,
                "entry_date":        date.today().isoformat(),
                "entry_price_est":   px,
                "underlying_entry":  close,
                "stop_underlying":   stop_underlying,
                "target_underlying": target_underlying,
                "dte_entry":         dte_actual,
                "expiry":            expiry_str,
                "signal_score":      score,
                "order_id":          result["order_id"],
                "days_held":         0,
                "tp_pct":            cfg["exit"].get("tp_pct", 0.50),
                "sl_pct":            cfg["exit"].get("sl_pct", 0.25),
            }
            _add_position(state, pos)

            _append_day_log({
                "role":          "ENTRY",
                "date":          date.today().isoformat(),
                "time_et":       _now_et().strftime("%H:%M"),
                "option_symbol": opt_symbol,
                "underlying":    underlying,
                "direction":     direction_r.upper(),
                "qty":           n_contracts,
                "dte":           dte_actual,
                "signal_score":  score,
                "entry_price_est": px,
                "order_id":      result["order_id"],
            })
        else:
            log.warning("  ❌  ORDER FAILED  %s  %s", opt_symbol, result.get("error", ""))


# ── Exit monitoring ────────────────────────────────────────────────────────────

def _check_exits(client, cfg: dict, state: Dict) -> None:
    """Check all open positions and exit those that hit stop/TP/DTE/max-hold."""
    if not state["positions"]:
        return

    exit_cfg   = cfg["exit"]
    tp_pct     = float(exit_cfg.get("tp_pct", 0.50))
    sl_pct     = float(exit_cfg.get("sl_pct", 0.25))
    dte_exit   = int(exit_cfg.get("dte_exit", 21))
    max_hold   = int(exit_cfg.get("max_hold_days", 21))

    to_close: List[tuple] = []   # (pos, exit_reason)

    for pos in list(state["positions"]):
        sym        = pos["option_symbol"]
        underlying = pos["underlying"]
        direction  = pos["direction"]
        entry_px   = float(pos.get("entry_price_est", 1.0))
        stop_ul    = float(pos.get("stop_underlying", 0))
        target_ul  = float(pos.get("target_underlying", 9999))
        expiry_str = pos.get("expiry", "")
        # Compute days_held from entry_date so multiple intra-day checks don't over-count
        try:
            entry_d  = date.fromisoformat(pos["entry_date"])
            days_held = (date.today() - entry_d).days
        except Exception:
            days_held = int(pos.get("days_held", 0))
        pos["days_held"] = days_held

        # DTE remaining
        dte_remaining = None
        if expiry_str:
            try:
                exp = date.fromisoformat(expiry_str)
                dte_remaining = (exp - date.today()).days
            except Exception:
                pass

        exit_reason = None

        # 1. DTE exit
        if dte_remaining is not None and dte_remaining <= dte_exit:
            exit_reason = "dte_exit"

        # 2. Max hold
        elif days_held >= max_hold:
            exit_reason = "max_hold"

        # 3. Underlying-based stop / target + trailing stop
        if exit_reason is None:
            ul_price = _get_underlying_price(underlying)
            if ul_price is not None:
                if direction == "call":
                    # Raise stop to entry after reaching halfway to target
                    halfway = pos.get("underlying_entry", stop_ul) + (target_ul - pos.get("underlying_entry", stop_ul)) * 0.5
                    if ul_price >= halfway:
                        new_stop = max(stop_ul, float(pos.get("underlying_entry", stop_ul)))
                        if new_stop > stop_ul:
                            log.info("  %s  trailing stop raised %.2f → %.2f (call at halfway)",
                                     sym, stop_ul, new_stop)
                            pos["stop_underlying"] = new_stop
                            stop_ul = new_stop
                    if ul_price <= stop_ul:
                        exit_reason = "stop_loss"
                    elif ul_price >= target_ul:
                        exit_reason = "take_profit"
                else:
                    halfway = pos.get("underlying_entry", stop_ul) - (pos.get("underlying_entry", stop_ul) - target_ul) * 0.5
                    if ul_price <= halfway:
                        new_stop = min(stop_ul, float(pos.get("underlying_entry", stop_ul)))
                        if new_stop < stop_ul:
                            log.info("  %s  trailing stop raised %.2f → %.2f (put at halfway)",
                                     sym, stop_ul, new_stop)
                            pos["stop_underlying"] = new_stop
                            stop_ul = new_stop
                    if ul_price >= stop_ul:
                        exit_reason = "stop_loss"
                    elif ul_price <= target_ul:
                        exit_reason = "take_profit"

        # 4. Option price-based stop / TP
        if exit_reason is None:
            cur_px = _get_position_price(client, sym)
            if cur_px is not None and entry_px > 0:
                pnl_pct = (cur_px - entry_px) / entry_px
                if pnl_pct <= -sl_pct:
                    exit_reason = "stop_loss"
                elif pnl_pct >= tp_pct:
                    exit_reason = "take_profit"

        if exit_reason:
            to_close.append((pos, exit_reason))

    for pos, reason in to_close:
        sym = pos["option_symbol"]
        qty = int(pos["qty"])
        log.info("  EXIT  %s  reason=%s  held=%d days", sym, reason, pos["days_held"])

        result = place_option_sell(client, sym, qty, reason=reason)

        cur_px    = _get_position_price(client, sym)
        entry_px  = float(pos.get("entry_price_est", 1.0))
        pnl_est   = round((cur_px - entry_px) * qty * CONTRACTS_PER_100, 2) \
                    if cur_px else None
        pnl_pct_r = round((cur_px - entry_px) / entry_px * 100, 1) \
                    if cur_px and entry_px else None

        if result["ok"]:
            log.info("  ✅  SOLD  %s  qty=%d  reason=%s  est_pnl=%s",
                     sym, qty, reason,
                     f"${pnl_est:+,.2f}" if pnl_est else "—")
        else:
            log.warning("  ❌  SELL FAILED  %s  %s", sym, result.get("error", ""))

        _append_day_log({
            "role":           "EXIT",
            "date":           date.today().isoformat(),
            "time_et":        _now_et().strftime("%H:%M"),
            "option_symbol":  sym,
            "underlying":     pos["underlying"],
            "direction":      pos["direction"].upper(),
            "qty":            qty,
            "exit_reason":    reason,
            "days_held":      pos["days_held"],
            "exit_price_est": cur_px,
            "entry_price_est": entry_px,
            "pnl_est":        pnl_est,
            "pnl_pct_est":    pnl_pct_r,
            "order_id":       result.get("order_id", ""),
        })

        _remove_position(state, sym)

    _save_state(state)


# ── Day summary ───────────────────────────────────────────────────────────────

def _print_summary(client, state: Dict) -> None:
    acct = _get_account(client)
    log.info("=" * 65)
    log.info("OPTIONS PAPER TRADE SUMMARY — %s", _now_et().strftime("%Y-%m-%d"))
    log.info("=" * 65)
    log.info("Portfolio: $%.2f  |  Cash: $%.2f  |  Open Positions: %d",
             acct["equity"], acct["cash"], len(state["positions"]))
    for pos in state["positions"]:
        cur = _get_position_price(client, pos["option_symbol"])
        ep  = float(pos.get("entry_price_est", 0))
        pnl = (cur - ep) * pos["qty"] * CONTRACTS_PER_100 if cur else None
        log.info("  %-30s  %-4s  qty=%d  held=%dd  est_pnl=%s",
                 pos["option_symbol"], pos["direction"].upper(),
                 pos["qty"], pos.get("days_held", 0),
                 f"${pnl:+,.2f}" if pnl is not None else "—")
    log.info("=" * 65)


# ── Main day runner ────────────────────────────────────────────────────────────

def _run_day(client, cfg: dict, state: Dict) -> None:
    now_str = _now_et().strftime("%Y-%m-%d %H:%M ET")
    log.info("=" * 65)
    log.info("OPTIONS 45-60 DTE PAPER TRADE SCAN — %s", now_str)
    log.info("=" * 65)

    acct = _get_account(client)
    log.info("Portfolio: $%.2f  |  Cash: $%.2f  |  Buying Power: $%.2f",
             acct["equity"], acct["cash"], acct["buying_power"])

    # First check exits for any existing open positions
    _check_exits(client, cfg, state)

    # Then enter new positions
    _run_entry_scan(client, cfg, state)

    log.info("Open positions after scan: %d", len(state["positions"]))


def _monitor_loop(client, cfg: dict, state: Dict) -> None:
    interval_min = int(cfg["exit"].get("monitor_interval", 15))
    log.info("Monitoring exits every %d min until 15:45 ET…", interval_min)
    while True:
        now = _now_et()
        if now.hour > 15 or (now.hour == 15 and now.minute >= 45):
            break
        _check_exits(client, cfg, state)
        if state["positions"]:
            total_pos = len(state["positions"])
            log.info("Open positions: %d", total_pos)
            for pos in state["positions"]:
                ul_px  = _get_underlying_price(pos["underlying"]) or 0
                cur_px = _get_position_price(client, pos["option_symbol"])
                ep     = float(pos.get("entry_price_est", 0))
                pnl    = (cur_px - ep) * pos["qty"] * CONTRACTS_PER_100 if cur_px else None
                log.info("  %-28s  ul=$%.2f  opt=%s  est_pnl=%s  held=%dd",
                         pos["option_symbol"], ul_px,
                         f"${cur_px:.2f}" if cur_px else "—",
                         f"${pnl:+,.2f}" if pnl is not None else "—",
                         pos.get("days_held", 0))
        time.sleep(interval_min * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

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

    parser = argparse.ArgumentParser(description="Options 45-60 DTE paper trader")
    parser.add_argument("--config",       default="configs/options_paper.yaml")
    parser.add_argument("--once",         action="store_true",
                        help="Run one scan immediately and exit (waits for market if before open)")
    parser.add_argument("--now",          action="store_true",
                        help="Scan + place orders RIGHT NOW regardless of market hours")
    parser.add_argument("--monitor-only", action="store_true",
                        help="Skip entry scan — only check exits")
    args = parser.parse_args()

    cfg = load_config(args.config)

    api_key = cfg["broker"].get("api_key") or os.environ.get("ALPACA_API_KEY", "")
    sec_key = cfg["broker"].get("secret_key") or os.environ.get("ALPACA_SECRET_KEY", "")

    if not api_key or not sec_key:
        log.error("No Alpaca credentials. Set ALPACA_API_KEY / ALPACA_SECRET_KEY in .env or config.")
        sys.exit(1)

    client = _make_client(api_key, sec_key)
    state  = _load_state()

    log.info("Options Paper Trader started — universe: %s  |  mode: %s",
             cfg["universe"].get("preset", "custom"),
             cfg["broker"].get("mode", "paper"))

    if args.now:
        # Immediate scan — no market-hours check, no loop
        log.info("--now flag: running scan immediately (market hours ignored)")
        _check_exits(client, cfg, state)
        if not args.monitor_only:
            _run_entry_scan(client, cfg, state)
        _print_summary(client, state)
        return

    if args.once:
        if not args.monitor_only:
            _run_day(client, cfg, state)
        else:
            _check_exits(client, cfg, state)
        _print_summary(client, state)
        return

    # Continuous daily loop
    scan_time = cfg["strategy"].get("scan_time", "09:35")
    while True:
        _wait_for_scan_time(scan_time)
        try:
            if not args.monitor_only:
                _run_day(client, cfg, state)
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
