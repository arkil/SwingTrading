"""
Options 45-60 DTE Trade Log
============================
Fetches, stores, and pairs option trades in the 40-70 DTE range
(covering the 45-60 DTE target window) from Alpaca.

Trade lifecycle
---------------
  ENTRY: buy option (simple or mleg)
  EXIT:  sell option (simple or mleg)

P&L per round-trip:
  simple : (exit_fill - entry_fill) × qty × 100
  mleg   : (exit_fill + entry_fill) × qty × 100  (entry_fill < 0)
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz

LOG_DIR = Path(__file__).parent / "logs" / "options_45_60d"
ET      = pytz.timezone("America/New_York")

DTE_MIN = 35   # lower tolerance below 45 for entry timing variance
DTE_MAX = 75   # upper tolerance above 60


# ── Option symbol helpers ──────────────────────────────────────────────────────

_OPT_RE = re.compile(r"^([A-Z]{1,5})(\d{6})([CP])(\d{8})$")


def _parse_option_symbol(sym: str) -> Optional[Dict]:
    """
    Parse an OCC option symbol (e.g. AAPL240315C00160000).
    Returns dict with: underlying, expiry (date), direction, strike (float).
    Returns None if not an option symbol.
    """
    m = _OPT_RE.match(sym or "")
    if not m:
        return None
    underlying, exp_str, cp, strike_str = m.groups()
    try:
        expiry = datetime.strptime(exp_str, "%y%m%d").date()
        strike = int(strike_str) / 1000.0
        return {
            "underlying": underlying,
            "expiry":     expiry,
            "direction":  "call" if cp == "C" else "put",
            "strike":     strike,
        }
    except Exception:
        return None


def _dte_at_order(symbol: str, order_date: date) -> Optional[int]:
    """Days to expiry from the order date, or None if not an option symbol."""
    info = _parse_option_symbol(symbol)
    if not info:
        return None
    return (info["expiry"] - order_date).days


def _is_45_60_dte(symbol: str, order_date: date) -> bool:
    """True if option DTE falls in our target window (DTE_MIN–DTE_MAX)."""
    dte = _dte_at_order(symbol, order_date)
    if dte is None:
        return False
    return DTE_MIN <= dte <= DTE_MAX


# ── Shared helpers (mirrors spy_reversal_log) ──────────────────────────────────

def _to_et_str(dt_str: str) -> str:
    if not dt_str:
        return "—"
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ET).strftime("%H:%M")
    except Exception:
        return "—"


def _to_date(dt_str: str) -> Optional[date]:
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ET).date()
    except Exception:
        return None


def _enum_val(v) -> str:
    if v is None:
        return ""
    return v.value if hasattr(v, "value") else str(v)


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Order parsing ─────────────────────────────────────────────────────────────

def _is_option_order(o) -> bool:
    """True if this order involves at least one equity option."""
    sym = str(getattr(o, "symbol", "") or "")
    if _parse_option_symbol(sym):
        return True
    for leg in (getattr(o, "legs", None) or []):
        if _parse_option_symbol(str(leg.symbol or "")):
            return True
    return False


def _is_45_60_order(o, order_date: date) -> bool:
    """True if any leg/symbol is in the 45-60 DTE window."""
    sym = str(getattr(o, "symbol", "") or "")
    if _is_45_60_dte(sym, order_date):
        return True
    for leg in (getattr(o, "legs", None) or []):
        if _is_45_60_dte(str(leg.symbol or ""), order_date):
            return True
    return False


def _describe_symbol(o) -> str:
    sym = str(getattr(o, "symbol", "") or "")
    if sym:
        return sym
    legs = getattr(o, "legs", None) or []
    if legs:
        underlyings = {_parse_option_symbol(str(l.symbol or "")) for l in legs}
        underlyings = {u["underlying"] for u in underlyings if u}
        root = "/".join(sorted(underlyings)) if underlyings else "OPT"
        return f"{root}-spread ({len(legs)} legs)"
    return "OPT"


def _direction_from_symbol(sym: str, side: str) -> str:
    """Infer trade direction from option symbol + side."""
    info = _parse_option_symbol(sym)
    if not info:
        return "?"
    if side == "buy":
        return "LONG" if info["direction"] == "call" else "SHORT"
    return "LONG" if info["direction"] == "put" else "SHORT"


def _infer_direction_mleg(legs: List[Dict]) -> str:
    """Infer direction from multi-leg order legs."""
    call_side = put_side = None
    for leg in legs:
        info = _parse_option_symbol(leg.get("symbol", ""))
        if not info:
            continue
        side = leg.get("side", "").lower()
        if info["direction"] == "call" and call_side is None:
            call_side = side
        if info["direction"] == "put" and put_side is None:
            put_side = side
    if put_side == "buy":
        return "SHORT"
    if call_side == "buy":
        return "LONG"
    return "?"


def parse_orders(raw_orders, dte_filter: bool = True) -> List[Dict[str, Any]]:
    """Convert raw Alpaca order objects → clean dicts (options 45-60 DTE only)."""
    result: List[Dict] = []
    for o in raw_orders:
        if not _is_option_order(o):
            continue

        submitted   = str(o.submitted_at) if o.submitted_at else ""
        order_date  = _to_date(submitted)
        if order_date is None:
            continue

        if dte_filter and not _is_45_60_order(o, order_date):
            continue

        filled_at   = str(o.filled_at) if o.filled_at else ""
        net_fill    = _safe_float(o.filled_avg_price)
        order_class = _enum_val(o.order_class) or "simple"
        sym         = str(o.symbol or "")
        side_val    = _enum_val(o.side)

        legs = []
        for leg in (getattr(o, "legs", None) or []):
            legs.append({
                "symbol":     str(leg.symbol or ""),
                "side":       _enum_val(leg.side),
                "qty":        float(leg.qty or 0),
                "fill_price": _safe_float(leg.filled_avg_price),
            })

        # Entry vs Exit
        if order_class == "mleg":
            role = "ENTRY" if (net_fill is not None and net_fill <= 0) else "EXIT"
        else:
            role = "ENTRY" if side_val == "buy" else "EXIT"

        # Direction
        if legs:
            direction = _infer_direction_mleg(legs)
        else:
            direction = _direction_from_symbol(sym, side_val)

        # DTE at order time
        dte_val = None
        info = _parse_option_symbol(sym)
        if info:
            dte_val = (info["expiry"] - order_date).days
        elif legs:
            for leg in legs:
                li = _parse_option_symbol(leg["symbol"])
                if li:
                    dte_val = (li["expiry"] - order_date).days
                    break

        # Underlying from symbol
        underlying = info["underlying"] if info else "?"
        if underlying == "?" and legs:
            li = _parse_option_symbol(legs[0]["symbol"])
            if li:
                underlying = li["underlying"]

        result.append({
            "id":           str(o.id),
            "date":         order_date.isoformat(),
            "time_et":      _to_et_str(submitted),
            "symbol":       _describe_symbol(o),
            "underlying":   underlying,
            "dte":          dte_val,
            "order_class":  order_class,
            "side":         side_val or "spread",
            "role":         role,
            "direction":    direction,
            "qty":          float(o.qty or 0),
            "filled_qty":   float(o.filled_qty or 0),
            "net_fill":     round(net_fill, 4) if net_fill is not None else None,
            "status":       str(o.status),
            "submitted_at": submitted,
            "filled_at":    filled_at,
            "legs":         legs,
            "pnl":          None,
            "pnl_pct":      None,
        })
    return result


# ── P&L pairing ───────────────────────────────────────────────────────────────

def pair_trades(orders: List[Dict]) -> List[Dict]:
    """
    Match ENTRY → EXIT on the same underlying + date to produce P&L.

    P&L calculation by order class:
      mleg   : entry_fill < 0 (debit), exit_fill > 0 (credit)
               net_per_contract = exit_fill + entry_fill
      simple : entry = buy price, exit = sell price (both positive)
               net_per_contract = exit_fill - entry_fill

    P&L = net_per_contract × qty × 100
    """
    sorted_orders = sorted(orders, key=lambda x: x["submitted_at"])
    open_entries: Dict[str, List[Dict]] = {}
    result: List[Dict] = []

    for rec in sorted_orders:
        rec  = dict(rec)
        d    = rec["date"]
        root = rec.get("underlying", rec.get("symbol", "")[:6])

        if rec["role"] == "ENTRY":
            key = (d, root)
            open_entries.setdefault(key, []).append(rec)
            result.append(rec)

        elif rec["role"] == "EXIT":
            key  = (d, root)
            pool = open_entries.get(key, [])
            entry = next(
                (e for e in pool
                 if e["qty"] == rec["qty"]
                 and not e.get("_matched")
                 and e["order_class"] == rec["order_class"]),
                None,
            )
            if entry and entry["net_fill"] is not None and rec["net_fill"] is not None:
                if rec["order_class"] == "mleg":
                    net_per_contract = rec["net_fill"] + entry["net_fill"]
                    cost_basis       = abs(entry["net_fill"])
                else:
                    net_per_contract = rec["net_fill"] - entry["net_fill"]
                    cost_basis       = abs(entry["net_fill"])

                total_pnl = round(net_per_contract * rec["qty"] * 100, 2)
                pnl_pct   = round(net_per_contract / cost_basis * 100, 1) \
                            if cost_basis > 0 else None

                rec["pnl"]     = total_pnl
                rec["pnl_pct"] = pnl_pct
                entry["_matched"]     = True
                entry["exit_time_et"] = rec["time_et"]

            result.append(rec)

    return result


# ── Storage ───────────────────────────────────────────────────────────────────

def _log_path(d: date) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"options_45_60d_{d.isoformat()}.json"


def store_logs(orders_paired: List[Dict]) -> int:
    by_date: Dict[str, List] = {}
    for rec in orders_paired:
        d = rec.get("date", "")
        if d:
            by_date.setdefault(d, []).append(rec)

    for d_str, recs in by_date.items():
        with open(_log_path(date.fromisoformat(d_str)), "w") as f:
            json.dump({"date": d_str, "orders": recs}, f, indent=2, default=str)

    return sum(len(v) for v in by_date.values())


def load_logs(start: date, end: date) -> List[Dict]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result: List[Dict] = []
    d = start
    while d <= end:
        p = _log_path(d)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                result.extend(data.get("orders", []))
            except Exception:
                pass
        d += timedelta(days=1)
    return result


def available_dates() -> List[date]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    dates = []
    for p in sorted(LOG_DIR.glob("options_45_60d_*.json"), reverse=True):
        try:
            dates.append(date.fromisoformat(p.stem.replace("options_45_60d_", "")))
        except ValueError:
            pass
    return dates


# ── Fetch + store convenience ─────────────────────────────────────────────────

def sync_from_alpaca(client, days_back: int = 90, dte_filter: bool = True) -> int:
    """Pull option orders from Alpaca, pair them, store per-day logs."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums   import QueryOrderStatus

    start_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
    try:
        raw = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=start_dt,
            limit=500,
        ))
    except Exception:
        return 0

    orders = parse_orders(raw, dte_filter=dte_filter)
    paired = pair_trades(orders)
    return store_logs(paired)


def get_records(client, start: date, end: date,
                use_cache: bool = True, dte_filter: bool = True) -> List[Dict]:
    """Return paired records for a date range (cache or live)."""
    if use_cache:
        return load_logs(start, end)

    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums   import QueryOrderStatus

    start_dt = datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt   = datetime.combine(end,   datetime.max.time()).replace(tzinfo=timezone.utc)
    try:
        raw = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=start_dt,
            until=end_dt,
            limit=500,
        ))
    except Exception:
        return []

    orders = parse_orders(raw, dte_filter=dte_filter)
    return pair_trades(orders)
