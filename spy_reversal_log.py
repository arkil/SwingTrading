"""
SPY Reversal Alert Logger
==========================
Fetches, stores, and pairs BB+RSI reversal trades from Alpaca.
dtb-live places multi-leg (mleg) spread orders and simple option orders.

Trade lifecycle
---------------
  ENTRY: mleg with negative net_fill (paid debit) OR simple buy
  EXIT:  mleg with positive net_fill (received credit) OR simple sell

P&L per round-trip = (exit_net_fill + entry_net_fill) × qty × 100
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz

LOG_DIR = Path(__file__).parent / "logs" / "spy_reversal"
ET      = pytz.timezone("America/New_York")


# ── Time helpers ──────────────────────────────────────────────────────────────

def _to_et_str(dt_str: str) -> str:
    """UTC datetime string → 'HH:MM ET'."""
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
    """UTC datetime string → date (ET)."""
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ET).date()
    except Exception:
        return None


# ── Order parsing ─────────────────────────────────────────────────────────────

def _spy_related(o) -> bool:
    """True if this Alpaca order involves SPY (stock or options)."""
    if o.symbol and str(o.symbol).startswith("SPY"):
        return True
    for leg in (getattr(o, "legs", None) or []):
        if leg.symbol and str(leg.symbol).startswith("SPY"):
            return True
    return False


def _enum_val(v) -> str:
    """Return the .value string of an Alpaca enum, or str() fallback."""
    if v is None:
        return ""
    return v.value if hasattr(v, "value") else str(v)


def _safe_float(v) -> Optional[float]:
    """Convert string or numeric fill price to float, None on error."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _infer_direction(legs: List[Dict]) -> str:
    """
    Determine bullish (LONG) or bearish (SHORT) from option legs.
    dtb-live's mleg structure:
      LONG (bullish): first P leg bought (put spread for downside protection paid)
      SHORT (bearish): first C leg bought
    Stored sides use raw enum .value ("buy"/"sell").
    """
    call_side = put_side = None
    for leg in legs:
        sym  = leg.get("symbol", "")
        side = leg.get("side", "").lower()     # already .value from parse_orders
        if re.search(r"C\d{8}$", sym) and call_side is None:
            call_side = side
        if re.search(r"P\d{8}$", sym) and put_side is None:
            put_side = side
    if put_side == "buy":
        return "SHORT"
    if call_side == "buy":
        return "LONG"
    return "?"


def _describe_symbol(o) -> str:
    """Human-friendly symbol for the order."""
    sym = getattr(o, "symbol", None)
    if sym:
        return str(sym)
    legs = getattr(o, "legs", None) or []
    if legs:
        # Show underlying + leg count
        return f"SPY-spread  ({len(legs)} legs)"
    return "SPY"


def _direction_from_symbol(sym: str) -> str:
    """Infer direction from a simple option symbol (e.g. SPY260318C00671000)."""
    if not sym:
        return "?"
    # Last letter before the 8-digit strike tells us Call or Put
    m = re.search(r"([CP])\d{8}$", sym)
    if m:
        return "LONG" if m.group(1) == "C" else "SHORT"
    return "?"


def parse_orders(raw_orders) -> List[Dict[str, Any]]:
    """Convert raw Alpaca order objects → clean dicts (SPY only)."""
    result: List[Dict] = []
    for o in raw_orders:
        if not _spy_related(o):
            continue

        legs = []
        for leg in (getattr(o, "legs", None) or []):
            legs.append({
                "symbol":     str(leg.symbol or ""),
                "side":       _enum_val(leg.side),           # "buy" or "sell"
                "qty":        float(leg.qty or 0),
                "fill_price": _safe_float(leg.filled_avg_price),
            })

        submitted   = str(o.submitted_at) if o.submitted_at else ""
        filled_at   = str(o.filled_at)    if o.filled_at    else ""
        net_fill    = _safe_float(o.filled_avg_price)
        order_class = _enum_val(o.order_class) or "simple"  # "mleg", "simple", "bracket" …
        sym         = str(o.symbol or "")

        # Entry = paid (negative net_fill for mleg) OR simple buy
        if order_class == "mleg":
            role = "ENTRY" if (net_fill is not None and net_fill <= 0) else "EXIT"
        else:
            side_val = _enum_val(o.side)
            role = "ENTRY" if side_val == "buy" else "EXIT"

        # Direction: from legs (mleg) or from option symbol (simple)
        direction = _infer_direction(legs) if legs else _direction_from_symbol(sym)

        result.append({
            "id":           str(o.id),
            "date":         _to_date(submitted).isoformat() if _to_date(submitted) else "",
            "time_et":      _to_et_str(submitted),
            "symbol":       _describe_symbol(o),
            "order_class":  order_class,
            "side":         _enum_val(o.side) if o.side else "spread",
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
    Match ENTRY → EXIT on the same date to produce P&L.

    P&L calculation by order class:
      mleg   : entry_fill is negative (paid debit), exit_fill is positive
               net_per_contract = exit_fill + entry_fill   (e.g. 2.05 + -0.98 = +1.07)
      simple : both fills are positive (buy price / sell price)
               net_per_contract = exit_fill - entry_fill   (e.g. 3.39 - 3.11 = +0.28)

    P&L = net_per_contract × qty × 100  (options multiplier)
    """
    sorted_orders = sorted(orders, key=lambda x: x["submitted_at"])

    # Pool unmatched entries per date, keyed by (date, symbol prefix)
    open_entries: Dict[str, List[Dict]] = {}
    result: List[Dict] = []

    for rec in sorted_orders:
        rec = dict(rec)  # copy
        d   = rec["date"]

        if rec["role"] == "ENTRY":
            open_entries.setdefault(d, []).append(rec)
            result.append(rec)

        elif rec["role"] == "EXIT":
            pool = open_entries.get(d, [])
            # Match by same qty and same underlying symbol root, FIFO
            entry = next(
                (e for e in pool
                 if e["qty"] == rec["qty"]
                 and not e.get("_matched")
                 and (e.get("symbol", "")[:10] == rec.get("symbol", "")[:10]
                      or e["order_class"] == rec["order_class"])),
                None,
            )
            if entry and entry["net_fill"] is not None and rec["net_fill"] is not None:
                order_class = rec.get("order_class", "simple")
                if order_class == "mleg":
                    # entry_fill < 0 (paid), exit_fill > 0 (received)
                    net_per_contract = rec["net_fill"] + entry["net_fill"]
                    cost_basis       = abs(entry["net_fill"])
                else:
                    # simple: entry = buy price, exit = sell price (both positive)
                    net_per_contract = rec["net_fill"] - entry["net_fill"]
                    cost_basis       = abs(entry["net_fill"])

                total_pnl = round(net_per_contract * rec["qty"] * 100, 2)
                pnl_pct   = round(net_per_contract / cost_basis * 100, 1) \
                            if cost_basis > 0 else None

                rec["pnl"]     = total_pnl
                rec["pnl_pct"] = pnl_pct
                entry["_matched"]    = True
                entry["exit_time_et"] = rec["time_et"]

            result.append(rec)

    return result


# ── Storage ───────────────────────────────────────────────────────────────────

def _log_path(d: date) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"spy_reversal_{d.isoformat()}.json"


def store_logs(orders_paired: List[Dict]) -> int:
    """Group paired orders by date and write one JSON file per day."""
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
    """Load stored records for a date range."""
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
    """Dates that have stored log files."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    dates = []
    for p in sorted(LOG_DIR.glob("spy_reversal_*.json"), reverse=True):
        try:
            dates.append(date.fromisoformat(p.stem.replace("spy_reversal_", "")))
        except ValueError:
            pass
    return dates


# ── Fetch + store convenience ─────────────────────────────────────────────────

def sync_from_alpaca(client, days_back: int = 90) -> int:
    """Pull SPY orders from Alpaca, pair them, store per-day logs. Returns record count."""
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

    orders = parse_orders(raw)
    paired = pair_trades(orders)
    return store_logs(paired)


def get_records(client, start: date, end: date, use_cache: bool = True) -> List[Dict]:
    """
    Return paired records for a date range.
    If use_cache: load from disk (faster); else fetch live from Alpaca.
    """
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

    orders = parse_orders(raw)
    return pair_trades(orders)
