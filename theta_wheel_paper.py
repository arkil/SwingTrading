#!/usr/bin/env python3
"""
Theta Wheel — Alpaca PAPER forward test
=======================================
Trades the tested Theta Wheel rules (theta_wheel_module.py, strategies/theta_wheel/RESULTS.md)
on the Alpaca paper account so real fills replace the simulation.

    python theta_wheel_paper.py open      # Mon/Tue ~10:20 ET: sell this Friday's puts on SELL signals
    python theta_wheel_paper.py manage    # daily ~15:40 ET: penny buy-backs; Friday: close ITM puts
    python theta_wheel_paper.py status    # ledger + P&L
    add --dry-run to print what it would do without sending orders

Rules traded (same as the page):
  * signal + tier from the latest Theta Wheel scan (bullish weekly regime / stars, fundamentals
    not FAIL, no earnings before expiry)
  * strike re-checked on Alpaca's LIVE quote: >= 5% OTM and mid >= 0.5% of strike, else skip
  * sell-to-open LIMIT at the mid (day order); unfilled orders just lapse
  * Mon–Thu: buy back any put whose ask is <= $0.05
  * expiry day: put in the money (or within 1% of the strike) -> buy to close at the ask;
    otherwise let it expire worthless
  * any assigned shares from a ledger put are sold at the next `open` run (no wheeling);
    P&L is booked from the sale's actual fill

Safety:
  * refuses to run unless ALPACA_BASE_URL is the paper endpoint
  * CASH ONLY: total collateral of open puts <= min(SLEEVE_CASH, account cash); the account
    is a margin account, so buying_power is never used
  * one-shot runs with the state on disk (no in-memory positions to lose on SIGTERM)
  * close_options_at_open.py (the 9:35 expiry sweep) skips puts listed here as open, since
    this bot closes them itself with limit orders on Friday afternoon
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "Data", "theta_wheel", "paper_ledger.json")
LOG = os.path.join(HERE, "Data", "theta_wheel", "paper_log.jsonl")

SLEEVE_CASH = 50_000.0      # this bot's cash budget inside the shared paper account (~half its cash)
PER_NAME_PCT = 25.0         # full (3-star) allocation per name, % of the sleeve
MIN_OTM = 0.05
MIN_YIELD = 0.005
PENNY = 0.05                # buy back when ask <= this
NEAR_STRIKE = 0.01          # expiry day: close if spot within 1% above the strike


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #
def _load_env():
    p = os.path.join(HERE, ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _clients():
    _load_env()
    base = os.environ.get("ALPACA_BASE_URL", "")
    if "paper" not in base:
        sys.exit(f"refusing: ALPACA_BASE_URL is not the paper endpoint ({base!r})")
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    k, s = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    return (TradingClient(k, s, paper=True), OptionHistoricalDataClient(k, s),
            StockHistoricalDataClient(k, s))


def load_ledger() -> list:
    try:
        with open(LEDGER) as f:
            return json.load(f)
    except Exception:
        return []


def save_ledger(rows: list):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    tmp = LEDGER + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f, indent=1, default=str)
    os.replace(tmp, LEDGER)


def log(event: str, **kw):
    rec = {"at": datetime.now(timezone.utc).isoformat(), "event": event, **kw}
    print(json.dumps(rec, default=str), flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def occ(sym: str, exp: str, strike: float, right: str = "P") -> str:
    d = date.fromisoformat(exp)
    return f"{sym}{d:%y%m%d}{right}{int(round(strike * 1000)):08d}"


def tick(px: float) -> float:
    """Valid option price increment: $0.05 at/above $3, $0.01 below (penny-pilot names)."""
    return 0.05 if px >= 3 else 0.01


def round_px(px: float, up: bool) -> float:
    t = tick(px)
    n = px / t
    return round((math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)) * t, 2)


def quote(oc, symbol: str):
    from alpaca.data.requests import OptionLatestQuoteRequest
    q = oc.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=[symbol])).get(symbol)
    if q is None:
        return None, None
    return float(q.bid_price or 0), float(q.ask_price or 0)


def spot(sc, sym: str):
    from alpaca.data.requests import StockLatestTradeRequest
    t = sc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=[sym])).get(sym)
    return float(t.price) if t else None


def _order(tc, symbol, qty, side, intent, limit, dry):
    from alpaca.common.exceptions import APIError
    from alpaca.trading.enums import OrderSide, PositionIntent, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest
    if dry:
        return {"id": "DRY-RUN", "status": "dry_run", "limit": limit}

    def _submit(px):
        return tc.submit_order(LimitOrderRequest(
            symbol=symbol, qty=qty, limit_price=px, time_in_force=TimeInForce.DAY,
            side=OrderSide.SELL if side == "sell" else OrderSide.BUY,
            position_intent=PositionIntent(intent)))
    try:
        o = _submit(limit)
    except APIError as e:
        # names outside the penny program quote in $0.05 steps even below $3
        nickel = round((math.ceil(limit / 0.05 - 1e-9) if side == "buy" else math.floor(limit / 0.05 + 1e-9)) * 0.05, 2)
        if nickel == limit or nickel <= 0 or not any(w in str(e).lower() for w in ("increment", "tick", "price")):
            raise
        log("reprice_nickel", occ=symbol, limit=limit, retry=nickel, err=str(e)[:160])
        o, limit = _submit(nickel), nickel
    return {"id": str(o.id), "status": str(o.status.value if hasattr(o.status, "value") else o.status),
            "limit": limit}


# --------------------------------------------------------------------------- #
# sync ledger with the broker
# --------------------------------------------------------------------------- #
def sync(tc, ledger: list, sc=None):
    """Pending opens -> open/unfilled; pending closes -> closed; lapsed expiries -> expired/assigned."""
    from alpaca.common.exceptions import APIError
    positions = {p.symbol: p for p in tc.get_all_positions()}
    today = date.today()
    for r in ledger:
        legs = [("open_order", r.get("open_order")), ("close_order", r.get("close_order"))]
        legs += [("close_order", o) for o in r.get("close_orders_prev", [])]
        for leg, o in legs:
            if not o or o.get("final") or o["id"] == "DRY-RUN":
                continue
            try:
                od = tc.get_order_by_id(o["id"])
            except APIError as e:
                log("order_lookup_failed", occ=r["occ"], err=str(e))
                continue
            st = str(od.status.value if hasattr(od.status, "value") else od.status)
            o["status"] = st
            filled = float(od.filled_qty or 0)
            if st in ("filled", "canceled", "expired", "rejected", "done_for_day"):
                o["final"] = True
            if filled > 0 and od.filled_avg_price:
                o["filled_qty"], o["fill"] = filled, float(od.filled_avg_price)
            if leg == "open_order" and o.get("final"):
                if filled > 0:
                    r.update(status="open", qty=int(filled), credit=o["fill"])
                    log("opened", occ=r["occ"], qty=int(filled), credit=o["fill"])
                else:
                    r["status"] = "unfilled"
                    log("open_unfilled", occ=r["occ"], order_status=st)
            if leg == "close_order" and o.get("final") and filled > 0 and not o.get("booked"):
                o["booked"] = True
                n = int(filled)
                part = round((r["credit"] - o["fill"]) * 100 * n, 2)
                r["realized"] = round(r.get("realized", 0.0) + part, 2)
                r["closed_qty"] = r.get("closed_qty", 0) + n
                if n >= r["qty"]:
                    r.update(status="closed", debit=o["fill"], closed_on=today.isoformat(),
                             qty=r["closed_qty"], pnl=r["realized"])
                    log("closed", occ=r["occ"], debit=o["fill"], pnl=r["pnl"])
                else:                                   # partial: keep managing the rest
                    r["qty"] -= n
                    log("closed_partial", occ=r["occ"], closed=n, remaining=r["qty"], pnl_part=part)
        # assigned shares being sold: P&L from the actual fill price
        so = r.get("sell_order")
        if r["status"] == "selling_shares" and so and not so.get("final"):
            try:
                od = tc.get_order_by_id(so["id"])
            except APIError as e:
                log("order_lookup_failed", occ=r["occ"], err=str(e))
                od = None
            if od is not None:
                st = str(od.status.value if hasattr(od.status, "value") else od.status)
                so["status"] = st
                if st == "filled":
                    fill = float(od.filled_avg_price)
                    n = int(float(od.filled_qty))
                    so.update(final=True, fill=fill)
                    r.update(status="closed", closed_on=today.isoformat(), shares_sold_px=fill,
                             pnl=round(r.get("realized", 0.0) + r["credit"] * 100 * r["qty"]
                                       - (r["strike"] - fill) * n, 2))
                    log("sold_assigned", sym=r["sym"], shares=n, fill=fill, pnl=r["pnl"])
                elif st in ("canceled", "expired", "rejected", "done_for_day"):
                    so["final"] = True
                    r["status"] = "assigned"              # not sold: try again next open run
                    log("sell_assigned_failed", sym=r["sym"], order_status=st)
        # expiry passed and the option is gone from the account
        if r["status"] == "open" and date.fromisoformat(r["exp"]) < today and r["occ"] not in positions:
            # Assigned only if it settled in the money AND the shares are actually there.
            # Holding the stock alone isn't proof: another bot on this account may own it.
            settle = close_on(sc, r["sym"], date.fromisoformat(r["exp"])) if sc else None
            stock = positions.get(r["sym"])
            held = float(stock.qty) if stock is not None else 0.0
            if settle is not None and settle < r["strike"] and held >= 100 * r["qty"]:
                r.update(status="assigned", assigned_shares=int(100 * r["qty"]), settle=settle)
                log("assigned", occ=r["occ"], shares=r["assigned_shares"], settle=settle)
            elif settle is not None and settle < r["strike"]:
                r.update(status="check_manually", settle=settle)
                log("assignment_unclear", occ=r["occ"], settle=settle, held=held,
                    note="settled ITM but the shares aren't in the account — check Alpaca")
            else:
                prior = r.get("realized", 0.0)
                r.update(status="expired", closed_on=r["exp"], debit=0.0, settle=settle,
                         pnl=round(prior + r["credit"] * 100 * r["qty"], 2))
                log("expired_worthless", occ=r["occ"], pnl=r["pnl"])
    return ledger


def close_on(sc, sym: str, day: date):
    """Official daily close for `day` (the option's settlement reference)."""
    from datetime import timedelta
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    try:
        bars = sc.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=[sym], timeframe=TimeFrame.Day,
            start=datetime(day.year, day.month, day.day), end=datetime(day.year, day.month, day.day) + timedelta(days=1))).df
        return float(bars["close"].iloc[-1]) if not bars.empty else None
    except Exception as e:  # noqa: BLE001
        log("settle_lookup_failed", sym=sym, day=day.isoformat(), err=str(e)[:160])
        return None


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_open(dry: bool, rescan: bool):
    from theta_wheel_module import load_latest, build_scan
    tc, oc, sc = _clients()
    if not tc.get_clock().is_open:
        log("skip_open", reason="market closed")
        return
    ledger = sync(tc, load_ledger(), sc)

    # 1) sell assigned shares (tested rule: take the loss, don't wheel)
    for r in ledger:
        if r["status"] == "assigned":
            px = spot(sc, r["sym"])
            if dry:
                log("would_sell_assigned", sym=r["sym"], shares=r["assigned_shares"], px=px)
                continue
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest
            held = next((float(p.qty) for p in tc.get_all_positions() if p.symbol == r["sym"]), 0.0)
            n = int(min(r["assigned_shares"], held))
            if n <= 0:
                r["status"] = "check_manually"
                log("assigned_shares_missing", sym=r["sym"])
                continue
            o = tc.submit_order(MarketOrderRequest(symbol=r["sym"], qty=n,
                                                   side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
            r.update(status="selling_shares", shares_sold=n, spot_at_sell=px,
                     sell_order={"id": str(o.id), "status": "new"})
            log("selling_assigned", sym=r["sym"], shares=n, spot=px)
    sync(tc, ledger, sc)                                  # books the sale if it filled already

    # 2) fresh scan if the cached one isn't from today's session
    df, details, meta = load_latest()
    built = (meta or {}).get("built_at", "")
    if rescan or not built or datetime.fromisoformat(built).astimezone().date() != date.today():
        log("rescan", reason="cache not from today" if not rescan else "forced")
        build_scan()
        df, details, meta = load_latest()
    if df is None or df.empty:
        log("skip_open", reason="empty scan")
        return

    acct = tc.get_account()
    cash = float(acct.cash)
    reserved = sum(r["strike"] * 100 * r.get("qty", r.get("qty_req", 0)) for r in ledger
                   if r["status"] in ("open", "pending"))
    budget = min(SLEEVE_CASH, cash) - reserved
    log("budget", cash=cash, sleeve=SLEEVE_CASH, reserved=reserved, free=budget)
    active = {r["sym"] for r in ledger if r["status"] in ("open", "pending", "assigned", "selling_shares",
                                                           "check_manually")}

    for _, row in df.iterrows():
        if not str(row["Action"]).startswith("SELL"):
            continue
        sym = row["Ticker"]
        d = details.get(sym) or {}
        put = d.get("put") or {}
        if sym in active or not put:
            continue
        exp, K, tier = d["exp"], float(put["strike"]), float(d["tier"])
        symbol = occ(sym, exp, K)
        px = spot(sc, sym)
        bid, ask = quote(oc, symbol)
        if not px or not bid or not ask or ask <= bid:
            log("skip", sym=sym, occ=symbol, reason="no live two-sided quote", bid=bid, ask=ask)
            continue
        mid = (bid + ask) / 2
        otm, yld = 1 - K / px, mid / K
        if otm < MIN_OTM or yld < MIN_YIELD:
            log("skip", sym=sym, occ=symbol, reason="fails live re-check", otm=round(otm, 4),
                yld=round(yld, 5), spot=px, bid=bid, ask=ask)
            continue
        full = SLEEVE_CASH * PER_NAME_PCT / 100
        alloc = full * tier
        qty = int(min(alloc, budget) // (K * 100))
        if qty < 1 and K * 100 <= min(full, budget):
            qty = 1        # whole contracts: a starter tier is 1 contract if one fits the name's full allocation
        if qty < 1:
            log("skip", sym=sym, occ=symbol, reason="collateral exceeds tier/budget",
                need=K * 100, alloc=alloc, free=budget)
            continue
        limit = round_px(mid, up=False)
        if limit < bid:
            limit = round_px(bid, up=True)
        res = _order(tc, symbol, qty, "sell", "sell_to_open", limit, dry)
        budget -= qty * K * 100
        entry = {"sym": sym, "occ": symbol, "exp": exp, "strike": K, "qty_req": qty, "tier": tier,
                 "opened_on": date.today().isoformat(), "limit": limit, "spot_at_open": px,
                 "bid": bid, "ask": ask, "otm": round(otm, 4), "yield": round(yld, 5),
                 "status": "dry_run" if dry else "pending", "open_order": res}
        log("sell_to_open" if not dry else "would_sell_to_open", **{k: entry[k] for k in
            ("sym", "occ", "qty_req", "limit", "spot_at_open", "otm", "yield")})
        if not dry:
            ledger.append(entry)
            active.add(sym)
    save_ledger(ledger)


def cmd_manage(dry: bool):
    tc, oc, sc = _clients()
    ledger = sync(tc, load_ledger(), sc)
    if not tc.get_clock().is_open:
        save_ledger(ledger)
        log("manage_synced_only", reason="market closed")
        return
    today = date.today()
    for r in ledger:
        if r["status"] != "open":
            continue
        exp = date.fromisoformat(r["exp"])
        working = r.get("close_order") and not r["close_order"].get("final")
        if working and exp != today:
            continue
        if any(not o.get("final") for o in r.get("close_orders_prev", [])):
            log("close_cancel_pending", occ=r["occ"])   # never stack a second buy-back on top
            continue
        bid, ask = quote(oc, r["occ"])
        px = spot(sc, r["sym"])
        if working:
            # expiry day and the earlier buy-back hasn't filled: cancel it and re-send at the
            # current ask so an in-the-money put isn't left to be assigned at the bell
            if r["close_order"].get("limit") is not None and ask and r["close_order"]["limit"] >= ask:
                continue                               # already marketable, give it time
            if not dry:
                old = r["close_order"]
                try:
                    tc.cancel_order_by_id(old["id"])
                    od = tc.get_order_by_id(old["id"])
                    for _ in range(20):                    # pending_cancel usually clears in seconds
                        if str(getattr(od.status, "value", od.status)) in ("canceled", "filled"):
                            break
                        time.sleep(1)
                        od = tc.get_order_by_id(old["id"])
                except Exception as e:  # noqa: BLE001
                    log("cancel_failed", occ=r["occ"], err=str(e)[:160])
                    continue
                # hand the old order to sync so any fill it got (even one that raced the
                # cancel) is booked, then only re-send what is still open
                r.setdefault("close_orders_prev", []).append(old)
                r["close_order"] = None
                sync(tc, [r], sc)
                if r["status"] != "open":
                    continue
                if str(getattr(od.status, "value", od.status)) != "canceled":
                    # still not confirmed after ~20s: re-sending could double-buy; the 15:50 run retries
                    log("cancel_not_confirmed", occ=r["occ"], status=str(od.status))
                    continue
            log("reprice_close", occ=r["occ"], ask=ask)
        reason = None
        if ask is not None and 0 < ask <= PENNY:
            reason, limit = "penny buy-back", round_px(ask, up=True)
        elif exp == today and px is not None and px <= r["strike"] * (1 + NEAR_STRIKE):
            reason = "expiry day, in/near the money"
            limit = round_px(ask if ask and ask > 0 else max(r["strike"] - px, 0.05), up=True)
        if not reason:
            continue
        res = _order(tc, r["occ"], r["qty"], "buy", "buy_to_close", limit, dry)
        log("buy_to_close" if not dry else "would_buy_to_close", occ=r["occ"], reason=reason,
            limit=limit, spot=px, bid=bid, ask=ask)
        if not dry:
            r["close_order"] = res
    save_ledger(ledger)


def cmd_status():
    ledger = load_ledger()
    if not ledger:
        print("ledger empty")
        return
    done = [r for r in ledger if r["status"] in ("closed", "expired")]
    openp = [r for r in ledger if r["status"] in ("open", "pending", "assigned", "selling_shares")]
    pnl = sum(r.get("pnl", 0) for r in done)
    wins = sum(1 for r in done if r.get("pnl", 0) > 0)
    print(f"closed {len(done)} (wins {wins}) realized ${pnl:,.2f} · open {len(openp)} · "
          f"collateral ${sum(r['strike'] * 100 * r.get('qty', r.get('qty_req', 0)) for r in openp):,.0f}")
    for r in ledger[-25:]:
        print(f"  {r['opened_on']} {r['occ']:24s} {r['status']:9s} qty {r.get('qty', r.get('qty_req'))} "
              f"credit {r.get('credit', '-')} pnl {r.get('pnl', '-')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["open", "manage", "status"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescan", action="store_true", help="force a fresh scan before opening")
    a = ap.parse_args()
    if a.cmd == "open":
        if date.today().weekday() > 1 and not a.dry_run:
            log("skip_open", reason="entries are Mon/Tue only")
            return
        cmd_open(a.dry_run, a.rescan)
    elif a.cmd == "manage":
        cmd_manage(a.dry_run)
    else:
        cmd_status()


if __name__ == "__main__":
    main()
