"""
Pyramid position monitor.

Entry → scale into winners → partial exits.

  Entry  : 1 contract at option_price P0
  Scale1 : ret >= +scale1_pct  → buy 1 more (total 2)
  Scale2 : ret >= +scale2_pct  → buy 1 more (total 3)
  Stop   : ret <= -stop_pct    → close ALL (fires regardless of scale state)
  Exit1  : ret >= +target1_pct → sell 1 contract
  Exit2  : ret >= +target2_pct → sell 1 contract
  Exit3  : ret >= +target3_pct OR bars_elapsed >= hold_bars → close remaining
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from daytrade_backtester.broker.base import BrokerAdapter

log = logging.getLogger(__name__)


@dataclass
class PyramidPosition:
    contract_symbol: str
    underlying_symbol: str
    side: str                      # "long" | "short"
    entry_time: datetime
    entry_underlying_price: float
    entry_option_price: float      # price at first contract (P0 for all % calcs)
    qty: int                       # current total contracts held
    hold_bars: int
    scale1_pct: float              # +1.5%  → add 2nd contract
    scale2_pct: float              # +2.7%  → add 3rd contract
    stop_pct: float                # 0.10   → close ALL if ret <= -stop_pct
    target1_pct: float             # +3.4%  → sell 1
    target2_pct: float             # +6.4%  → sell 1
    target3_pct: float             # +11.2% → sell last
    bars_elapsed: int = 0
    scale1_done: bool = False
    scale2_done: bool = False
    exit1_done: bool = False
    exit2_done: bool = False


@dataclass
class PyramidEvent:
    position: PyramidPosition
    event_time: datetime
    option_price: float
    option_return_pct: float       # relative to P0 (entry_option_price)
    pnl_usd: float
    qty_affected: int
    reason: str


class PyramidPositionMonitor:
    """Tracks one open pyramid position and fires scale-in / exit actions."""

    def __init__(self, broker: BrokerAdapter, capital_per_trade: float = 1000.0) -> None:
        self._broker = broker
        self._capital = capital_per_trade
        self.open_position: PyramidPosition | None = None
        self.events: list[PyramidEvent] = []

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, position: PyramidPosition) -> None:
        self.open_position = position
        log.info(
            "PYRAMID registered: %s  P0=%.4f  qty=%d  side=%s",
            position.contract_symbol,
            position.entry_option_price,
            position.qty,
            position.side,
        )

    def has_position(self) -> bool:
        return self.open_position is not None

    # ── Per-bar ───────────────────────────────────────────────────────────────

    def on_bar_close(self, bar_time: datetime) -> list[PyramidEvent]:
        if self.open_position is None:
            return []

        pos = self.open_position
        pos.bars_elapsed += 1

        quote = self._broker.get_quote(pos.contract_symbol)
        if quote is None:
            log.warning("No quote for %s (bar %d)", pos.contract_symbol, pos.bars_elapsed)
            if pos.bars_elapsed >= pos.hold_bars:
                return self._time_exit(pos, bar_time, pos.entry_option_price, 0.0)
            return []

        mid = quote.mid
        P0  = pos.entry_option_price
        ret = (mid / P0) - 1.0

        log.info(
            "PYRAMID %s  bar=%d  qty=%d  mid=%.4f  ret=%+.2f%%",
            pos.contract_symbol, pos.bars_elapsed, pos.qty, mid, ret * 100,
        )

        fired: list[PyramidEvent] = []

        # ── Stop loss ─────────────────────────────────────────────────────────
        if ret <= -abs(pos.stop_pct):
            print(f"  STOP-LOSS  {pos.contract_symbol}  ret={ret*100:+.1f}%  closing {pos.qty} contracts")
            self._sell(pos, pos.qty)
            evt = PyramidEvent(
                position=pos, event_time=bar_time, option_price=mid,
                option_return_pct=ret, pnl_usd=(mid - pos.entry_option_price) * pos.qty * 100,
                qty_affected=pos.qty, reason="stop_loss",
            )
            self.events.append(evt)
            fired.append(evt)
            self.open_position = None
            return fired

        # ── Scale-in #1 ───────────────────────────────────────────────────────
        if not pos.scale1_done and ret >= pos.scale1_pct:
            print(f"  SCALE-IN #1  {pos.contract_symbol}  ret={ret*100:+.1f}%  adding 1 contract")
            order = self._broker.place_order(pos.contract_symbol, 1, "buy")
            if order.status.lower() not in {"rejected", "expired", "canceled", "cancelled"}:
                pos.qty += 1
                pos.scale1_done = True
                print(f"            → now holding {pos.qty} contracts")
            else:
                log.warning("Scale-in #1 order failed: %s", order.status)

        # ── Scale-in #2 ───────────────────────────────────────────────────────
        if not pos.scale2_done and pos.scale1_done and ret >= pos.scale2_pct:
            print(f"  SCALE-IN #2  {pos.contract_symbol}  ret={ret*100:+.1f}%  adding 1 contract")
            order = self._broker.place_order(pos.contract_symbol, 1, "buy")
            if order.status.lower() not in {"rejected", "expired", "canceled", "cancelled"}:
                pos.qty += 1
                pos.scale2_done = True
                print(f"            → now holding {pos.qty} contracts")
            else:
                log.warning("Scale-in #2 order failed: %s", order.status)

        # ── Partial exit #1 ───────────────────────────────────────────────────
        if not pos.exit1_done and ret >= pos.target1_pct and pos.qty >= 1:
            print(f"  EXIT-1  {pos.contract_symbol}  ret={ret*100:+.1f}%  selling 1 contract")
            self._sell(pos, 1)
            evt = PyramidEvent(
                position=pos, event_time=bar_time, option_price=mid,
                option_return_pct=ret, pnl_usd=(mid - pos.entry_option_price) * 1 * 100,
                qty_affected=1, reason="target1",
            )
            self.events.append(evt)
            fired.append(evt)
            pos.qty -= 1
            pos.exit1_done = True
            if pos.qty == 0:
                self.open_position = None
                return fired

        # ── Partial exit #2 ───────────────────────────────────────────────────
        if not pos.exit2_done and ret >= pos.target2_pct and pos.qty >= 1:
            print(f"  EXIT-2  {pos.contract_symbol}  ret={ret*100:+.1f}%  selling 1 contract")
            self._sell(pos, 1)
            evt = PyramidEvent(
                position=pos, event_time=bar_time, option_price=mid,
                option_return_pct=ret, pnl_usd=(mid - pos.entry_option_price) * 1 * 100,
                qty_affected=1, reason="target2",
            )
            self.events.append(evt)
            fired.append(evt)
            pos.qty -= 1
            pos.exit2_done = True
            if pos.qty == 0:
                self.open_position = None
                return fired

        # ── Final exit (target3 or time) ──────────────────────────────────────
        hit_target3 = pos.exit1_done and pos.exit2_done and ret >= pos.target3_pct
        hit_time    = pos.bars_elapsed >= pos.hold_bars

        if hit_target3 or hit_time:
            reason = "target3" if hit_target3 else "time_exit"
            print(f"  FINAL-EXIT ({reason})  {pos.contract_symbol}  ret={ret*100:+.1f}%  closing {pos.qty} contracts")
            self._sell(pos, pos.qty)
            evt = PyramidEvent(
                position=pos, event_time=bar_time, option_price=mid,
                option_return_pct=ret,
                pnl_usd=(mid - pos.entry_option_price) * pos.qty * 100,
                qty_affected=pos.qty, reason=reason,
            )
            self.events.append(evt)
            fired.append(evt)
            self.open_position = None
            return fired

        return fired

    def close_all(self, bar_time: datetime, reason: str = "day_end") -> list[PyramidEvent]:
        if self.open_position is None:
            return []
        pos = self.open_position
        quote = self._broker.get_quote(pos.contract_symbol)
        mid = quote.mid if quote else pos.entry_option_price
        ret = (mid / pos.entry_option_price) - 1.0 if pos.entry_option_price > 0 else 0.0
        self._sell(pos, pos.qty)
        evt = PyramidEvent(
            position=pos, event_time=bar_time, option_price=mid,
            option_return_pct=ret,
            pnl_usd=(mid - pos.entry_option_price) * pos.qty * 100,
            qty_affected=pos.qty, reason=reason,
        )
        self.events.append(evt)
        self.open_position = None
        return [evt]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sell(self, pos: PyramidPosition, qty: int) -> None:
        if qty <= 0:
            return
        result = self._broker.place_order(pos.contract_symbol, qty, "sell")
        if result and result.status.lower() not in {"rejected", "expired", "canceled", "cancelled"}:
            log.info("SELL %d × %s  order_id=%s  status=%s", qty, pos.contract_symbol, result.order_id, result.status)
        else:
            log.warning("SELL order %s (qty=%d) may have failed: %s", pos.contract_symbol, qty, result.status if result else "None")

    def _time_exit(
        self, pos: PyramidPosition, bar_time: datetime, mid: float, ret: float
    ) -> list[PyramidEvent]:
        self._sell(pos, pos.qty)
        evt = PyramidEvent(
            position=pos, event_time=bar_time, option_price=mid,
            option_return_pct=ret,
            pnl_usd=(mid - pos.entry_option_price) * pos.qty * 100,
            qty_affected=pos.qty, reason="time_exit_no_quote",
        )
        self.events.append(evt)
        self.open_position = None
        return [evt]

    # ── Summary ───────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        if not self.events:
            print("No trade events today.")
            return
        total_pnl = sum(e.pnl_usd for e in self.events)
        wins = [e for e in self.events if e.pnl_usd > 0]
        print("\n" + "=" * 64)
        print("  PYRAMID SESSION SUMMARY")
        print("=" * 64)
        print(f"  Events today  : {len(self.events)}")
        print(f"  Profitable    : {len(wins)}")
        print(f"  Net P&L       : ${total_pnl:+.2f}")
        print("-" * 64)
        for i, e in enumerate(self.events, 1):
            print(
                f"  #{i:<3} {e.position.contract_symbol:<28} "
                f"{e.reason:<14} "
                f"qty={e.qty_affected}  "
                f"ret={e.option_return_pct*100:+.1f}%  "
                f"P&L=${e.pnl_usd:+.2f}"
            )
        print("=" * 64)
