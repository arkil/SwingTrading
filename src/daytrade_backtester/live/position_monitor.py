from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from daytrade_backtester.broker.base import BrokerAdapter

log = logging.getLogger(__name__)


@dataclass
class LivePosition:
    """Represents one open option position managed by the live runner."""

    contract_symbol: str          # OCC symbol, e.g. "SPY260314C00685000"
    underlying_symbol: str
    side: str                     # "long" (call) | "short" (put)
    entry_time: datetime
    entry_underlying_price: float
    entry_option_price: float     # mid at fill time (bid+ask)/2
    qty: int                      # number of contracts
    hold_bars: int                # max N-min bars to hold
    early_exit_bar: int           # bar at which to check early exit (0 = disabled)
    early_exit_pct: float         # e.g. -0.03 means -3%; exit if option return <= this
    option_target_pct: float      # e.g. 0.05 means +5% option return = close
    bars_elapsed: int = 0


@dataclass
class ExitEvent:
    position: LivePosition
    exit_time: datetime
    exit_option_price: float
    option_return_pct: float
    pnl_usd: float
    exit_reason: str              # "profit_target" | "early_exit" | "time_exit" | "day_end_exit"


class PositionMonitor:
    """
    Tracks open option positions and triggers exits based on:
      - profit_target  : option return >= option_target_pct
      - early_exit     : option return <= early_exit_pct at exactly early_exit_bar
      - time_exit      : bars_elapsed >= hold_bars
      - day_end_exit   : called explicitly at session close

    Call `on_bar_close()` once per completed N-min bar.
    Exited positions are removed from `open_positions` and appended to `closed`.
    """

    def __init__(self, broker: BrokerAdapter, capital_per_trade: float = 1000.0) -> None:
        self._broker = broker
        self._capital = capital_per_trade
        self.open_positions: list[LivePosition] = []
        self.closed: list[ExitEvent] = []

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, position: LivePosition) -> None:
        self.open_positions.append(position)
        log.info(
            "Position registered: %s  entry_opt=%.4f  hold_bars=%d",
            position.contract_symbol,
            position.entry_option_price,
            position.hold_bars,
        )

    # ── Per-bar update ────────────────────────────────────────────────────────

    def on_bar_close(self, bar_time: datetime) -> list[ExitEvent]:
        """
        Called once per completed N-min bar. Returns list of exits that triggered
        this bar (may be empty).
        """
        exits: list[ExitEvent] = []
        still_open: list[LivePosition] = []

        for pos in self.open_positions:
            pos.bars_elapsed += 1
            event = self._check_exit(pos, bar_time)
            if event is not None:
                self._execute_exit(pos)
                self.closed.append(event)
                exits.append(event)
            else:
                still_open.append(pos)

        self.open_positions = still_open
        return exits

    def close_all(self, bar_time: datetime, reason: str = "day_end_exit") -> list[ExitEvent]:
        """Force-close all open positions (call at session end)."""
        exits: list[ExitEvent] = []
        for pos in list(self.open_positions):
            event = self._force_exit(pos, bar_time, reason)
            if event:
                exits.append(event)
                self.closed.append(event)
        self.open_positions = []
        return exits

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_exit(self, pos: LivePosition, bar_time: datetime) -> ExitEvent | None:
        if pos.entry_option_price <= 0:
            log.error("Position %s has zero entry price — force-closing to prevent zombie", pos.contract_symbol)
            return ExitEvent(
                position=pos,
                exit_time=bar_time,
                exit_option_price=0.0,
                option_return_pct=0.0,
                pnl_usd=0.0,
                exit_reason="bad_entry_price",
            )

        quote = self._broker.get_quote(pos.contract_symbol)
        if quote is None:
            log.warning("Quote unavailable for %s (bars=%d)", pos.contract_symbol, pos.bars_elapsed)
            # Still enforce time exit so a dead quote can never create a zombie position.
            if pos.bars_elapsed >= pos.hold_bars:
                return ExitEvent(
                    position=pos,
                    exit_time=bar_time,
                    exit_option_price=pos.entry_option_price,
                    option_return_pct=0.0,
                    pnl_usd=0.0,
                    exit_reason="time_exit_no_quote",
                )
            return None

        current_mid = quote.mid
        opt_return = (current_mid / pos.entry_option_price) - 1.0
        pnl = opt_return * self._capital * pos.qty

        log.debug(
            "%s  bars=%d  mid=%.4f  ret=%.2f%%  pnl=$%.2f",
            pos.contract_symbol, pos.bars_elapsed, current_mid, opt_return * 100, pnl,
        )

        # Profit target
        if opt_return >= pos.option_target_pct:
            return ExitEvent(
                position=pos,
                exit_time=bar_time,
                exit_option_price=current_mid,
                option_return_pct=opt_return,
                pnl_usd=pnl,
                exit_reason="profit_target",
            )

        # Early exit: >= so a single missed quote at exactly early_exit_bar doesn't disable it.
        use_early = pos.early_exit_bar > 0 and pos.early_exit_pct < 0.0
        if use_early and pos.bars_elapsed >= pos.early_exit_bar:
            if opt_return <= pos.early_exit_pct:
                return ExitEvent(
                    position=pos,
                    exit_time=bar_time,
                    exit_option_price=current_mid,
                    option_return_pct=opt_return,
                    pnl_usd=pnl,
                    exit_reason="early_exit",
                )

        # Time exit
        if pos.bars_elapsed >= pos.hold_bars:
            return ExitEvent(
                position=pos,
                exit_time=bar_time,
                exit_option_price=current_mid,
                option_return_pct=opt_return,
                pnl_usd=pnl,
                exit_reason="time_exit",
            )

        return None

    def _force_exit(
        self, pos: LivePosition, bar_time: datetime, reason: str
    ) -> ExitEvent | None:
        quote = self._broker.get_quote(pos.contract_symbol)
        current_mid = quote.mid if quote else pos.entry_option_price
        opt_return = (current_mid / pos.entry_option_price) - 1.0 if pos.entry_option_price > 0 else 0.0
        pnl = opt_return * self._capital * pos.qty
        self._execute_exit(pos)
        return ExitEvent(
            position=pos,
            exit_time=bar_time,
            exit_option_price=current_mid,
            option_return_pct=opt_return,
            pnl_usd=pnl,
            exit_reason=reason,
        )

    def _execute_exit(self, pos: LivePosition) -> None:
        result = self._broker.close_position(pos.contract_symbol)
        if result:
            log.info(
                "EXIT  %s  order_id=%s  status=%s",
                pos.contract_symbol, result.order_id, result.status,
            )
        else:
            log.warning("EXIT order for %s may have failed — check broker dashboard", pos.contract_symbol)

    # ── Summary ───────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        if not self.closed:
            print("No closed trades today.")
            return

        total_pnl = sum(e.pnl_usd for e in self.closed)
        wins = [e for e in self.closed if e.pnl_usd > 0]
        win_rate = len(wins) / len(self.closed) * 100

        print("\n" + "=" * 60)
        print("  LIVE SESSION SUMMARY")
        print("=" * 60)
        print(f"  Trades closed : {len(self.closed)}")
        print(f"  Win rate      : {win_rate:.1f}%  ({len(wins)}W / {len(self.closed) - len(wins)}L)")
        print(f"  Net P&L       : ${total_pnl:+.2f}")
        print("-" * 60)

        for i, e in enumerate(self.closed, 1):
            print(
                f"  #{i:<3} {e.position.contract_symbol:<28} "
                f"{e.exit_reason:<16} "
                f"ret={e.option_return_pct*100:+.1f}%  "
                f"P&L=${e.pnl_usd:+.2f}"
            )
        print("=" * 60)
