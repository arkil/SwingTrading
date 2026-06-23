"""
Alpaca Paper Trading Integration
==================================
Executes Daily Alert signals on Alpaca paper trading account.

Features:
 - Bracket orders: entry limit + stop-loss + take-profit (T2 as main target)
 - Portfolio overview: equity, buying power, open P&L
 - Position & order management
 - Risk controls: max positions, min score, market-hours guard
"""

import os
import math
from typing import Optional, List, Dict, Any
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderClass,
    QueryOrderStatus,
)


# ── Client factory ────────────────────────────────────────────────────────────

def make_client(api_key: str, secret_key: str, paper: bool = True) -> TradingClient:
    return TradingClient(api_key, secret_key, paper=paper)


# ── Account ───────────────────────────────────────────────────────────────────

def get_account_summary(client: TradingClient) -> Dict[str, Any]:
    acct = client.get_account()
    return {
        "equity":        float(acct.equity),
        "cash":          float(acct.cash),
        "buying_power":  float(acct.buying_power),
        "portfolio_value": float(acct.portfolio_value),
        "daytrade_count": int(acct.daytrade_count),
        "status":        str(acct.status),
        "pattern_day_trader": bool(acct.pattern_day_trader),
    }


def is_market_open_alpaca(client: TradingClient) -> bool:
    clock = client.get_clock()
    return bool(clock.is_open)


# ── Positions ─────────────────────────────────────────────────────────────────

def get_positions(client: TradingClient) -> List[Dict[str, Any]]:
    positions = client.get_all_positions()
    result = []
    for p in positions:
        entry = float(p.avg_entry_price)
        curr  = float(p.current_price or 0)
        qty   = float(p.qty)
        side  = str(p.side)
        unrealized_pl     = float(p.unrealized_pl or 0)
        unrealized_plpc   = float(p.unrealized_plpc or 0) * 100
        result.append({
            "symbol":         p.symbol,
            "qty":            qty,
            "side":           side,
            "entry_price":    round(entry, 2),
            "current_price":  round(curr, 2),
            "market_value":   round(float(p.market_value or 0), 2),
            "unrealized_pl":  round(unrealized_pl, 2),
            "unrealized_pl%": round(unrealized_plpc, 2),
            "asset_id":       str(p.asset_id),
        })
    return result


def close_position(client: TradingClient, symbol: str) -> Dict:
    try:
        resp = client.close_position(symbol)
        return {"ok": True, "order_id": str(resp.id)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def close_all_positions(client: TradingClient) -> Dict:
    try:
        client.close_all_positions(cancel_orders=True)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Orders ────────────────────────────────────────────────────────────────────

def get_open_orders(client: TradingClient) -> List[Dict[str, Any]]:
    try:
        orders = client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)
        )
    except Exception:
        return []
    result = []
    for o in orders:
        result.append({
            "id":          str(o.id),
            "symbol":      o.symbol,
            "side":        str(o.side),
            "type":        str(o.type),
            "qty":         float(o.qty or 0),
            "filled_qty":  float(o.filled_qty or 0),
            "limit_price": float(o.limit_price) if o.limit_price else None,
            "stop_price":  float(o.stop_price)  if o.stop_price  else None,
            "status":      str(o.status),
            "created_at":  str(o.created_at)[:16],
            "order_class": str(o.order_class),
        })
    return result


def cancel_order(client: TradingClient, order_id: str) -> Dict:
    try:
        client.cancel_order_by_id(order_id)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def cancel_all_orders(client: TradingClient) -> Dict:
    try:
        client.cancel_orders()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Order placement ───────────────────────────────────────────────────────────

def _calc_shares(
    buying_power: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.01,
    max_position_pct: float = 0.10,
) -> int:
    """
    Calculate share qty using 1% risk rule, capped at 10% of portfolio.
    """
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0 or entry_price <= 0:
        return 0
    risk_dollars    = buying_power * risk_pct
    size_by_risk    = int(risk_dollars / risk_per_share)
    size_by_capital = int((buying_power * max_position_pct) / entry_price)
    shares = min(size_by_risk, size_by_capital)
    return max(1, shares)


def place_bracket_order(
    client:          TradingClient,
    symbol:          str,
    direction:       str,   # "BUY" or "SELL"
    entry_price:     float,
    stop_price:      float,
    take_profit_price: float,
    shares:          int,
    time_in_force:   TimeInForce = TimeInForce.DAY,
) -> Dict:
    """
    Place a bracket order: limit entry + stop-loss + take-profit.

    Returns dict with ok, order_id, or error.
    """
    side = OrderSide.BUY if direction == "BUY" else OrderSide.SELL

    # Round prices to 2 decimal places (Alpaca requirement)
    entry_r  = round(entry_price,      2)
    stop_r   = round(stop_price,       2)
    target_r = round(take_profit_price, 2)

    try:
        req = LimitOrderRequest(
            symbol         = symbol,
            qty            = shares,
            side           = side,
            time_in_force  = time_in_force,
            limit_price    = entry_r,
            order_class    = OrderClass.BRACKET,
            stop_loss      = StopLossRequest(stop_price=stop_r),
            take_profit    = TakeProfitRequest(limit_price=target_r),
        )
        order = client.submit_order(req)
        return {
            "ok":       True,
            "order_id": str(order.id),
            "symbol":   symbol,
            "qty":      shares,
            "entry":    entry_r,
            "stop":     stop_r,
            "target":   target_r,
            "side":     direction,
        }
    except Exception as e:
        return {"ok": False, "symbol": symbol, "error": str(e)}


# ── Execute alerts batch ──────────────────────────────────────────────────────

def execute_alerts(
    client:          TradingClient,
    alerts_df,                          # pd.DataFrame from generate_daily_alerts
    risk_pct:        float = 0.005,
    max_position_pct: float = 0.05,
    max_new_positions: int  = 5,
    min_score:       int    = 6,
    use_t2_target:   bool   = True,     # T2 = main target; False = T1
    dry_run:         bool   = False,
    max_dollars_per_trade: float = 1000.0,  # hard dollar cap per position
) -> List[Dict]:
    """
    Submit bracket orders for qualifying alerts.

    Args:
        alerts_df:         DataFrame from generate_daily_alerts()
        risk_pct:          Fraction of portfolio to risk per trade (default 1%)
        max_position_pct:  Max single-position size as fraction of portfolio
        max_new_positions: Maximum number of new orders to place this run
        min_score:         Minimum alert score required
        use_t2_target:     Use T2 as take-profit; False uses T1
        dry_run:           If True, simulate without placing real orders

    Returns:
        List of result dicts (one per attempted order).
    """
    import pandas as pd

    results = []

    acct = get_account_summary(client)
    buying_power = acct["buying_power"]

    # Existing positions — avoid doubling up
    existing = {p["symbol"] for p in get_positions(client)}
    pending  = {o["symbol"] for o in get_open_orders(client)}
    skip_set = existing | pending

    # Filter & rank
    df = alerts_df[
        (alerts_df["Score"] >= min_score) &
        (~alerts_df["Symbol"].isin(skip_set))
    ].sort_values(["Score", "R/R"], ascending=False).head(max_new_positions)

    for _, row in df.iterrows():
        sym     = row["Symbol"]
        direction = row["Direction"]
        entry   = float(row["Entry"])
        stop    = float(row["Stop"])
        target  = float(row["T2"]) if use_t2_target else float(row["T1"])

        shares = _calc_shares(
            buying_power    = buying_power,
            entry_price     = entry,
            stop_price      = stop,
            risk_pct        = risk_pct,
            max_position_pct= max_position_pct,
        )

        # Hard dollar cap — never spend more than max_dollars_per_trade per position
        if entry > 0 and max_dollars_per_trade > 0:
            max_by_dollars = int(max_dollars_per_trade / entry)
            shares = min(shares, max(1, max_by_dollars))

        if shares == 0:
            results.append({"ok": False, "symbol": sym,
                            "error": "Insufficient buying power"})
            continue

        if dry_run:
            results.append({
                "ok": True, "dry_run": True,
                "symbol": sym, "qty": shares,
                "entry": entry, "stop": stop, "target": target,
                "side": direction,
                "estimated_risk": round(abs(entry - stop) * shares, 2),
            })
        else:
            result = place_bracket_order(
                client, sym, direction, entry, stop, target, shares,
            )
            results.append(result)

        # Reduce remaining buying power estimate
        buying_power -= entry * shares

    return results


# ── Today's filled orders (closed trades) ────────────────────────────────────

def get_todays_trades(client: TradingClient) -> List[Dict[str, Any]]:
    """Return filled orders placed today."""
    from datetime import date, timezone
    import datetime as dt
    try:
        today_utc = dt.datetime.combine(date.today(), dt.time.min, tzinfo=timezone.utc)
        orders = client.get_orders(
            GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                after=today_utc,
                limit=100,
            )
        )
    except Exception:
        return []

    result = []
    for o in orders:
        filled_price = float(o.filled_avg_price) if o.filled_avg_price else None
        result.append({
            "id":           str(o.id),
            "symbol":       o.symbol,
            "side":         str(o.side),
            "type":         str(o.type),
            "qty":          float(o.qty or 0),
            "filled_qty":   float(o.filled_qty or 0),
            "filled_price": filled_price,
            "status":       str(o.status),
            "submitted_at": str(o.submitted_at)[:16] if o.submitted_at else "",
            "filled_at":    str(o.filled_at)[:16]    if o.filled_at    else "",
        })
    return result


def get_portfolio_history(client: TradingClient, period: str = "1D") -> Dict:
    """Fetch equity curve for today (intraday)."""
    try:
        hist = client.get_portfolio_history(period=period, timeframe="5Min", extended_hours=False)
        return {
            "equity":     list(hist.equity     or []),
            "profit_loss":list(hist.profit_loss or []),
            "timestamps": list(hist.timestamp  or []),
        }
    except Exception:
        return {"equity": [], "profit_loss": [], "timestamps": []}
