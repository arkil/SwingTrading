"""
Pyramid live trading runner — BB+RSI signal, scale into winners.

Flow
----
1. Load config, init AlpacaBroker + BarBuffer + PyramidPositionMonitor
2. Warmup: pull last 500 bars via REST so indicators are primed
3. WebSocket: stream 1-min bars → aggregate to N-min bars
4. On each completed N-min bar:
   a. PyramidMonitor.on_bar_close()  ← handle scale-ins / exits
   b. If no open position AND signal fires → enter 1 contract
5. Day end: close_all(), print_summary()

No guardrails: both calls and puts, no SMA filter, no ADX, no cooldown.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from daytrade_backtester.broker.alpaca import AlpacaBroker
from daytrade_backtester.config.loader import load_config
from daytrade_backtester.live.bar_buffer import BarBuffer
from daytrade_backtester.live.pyramid_monitor import PyramidPosition, PyramidPositionMonitor
from daytrade_backtester.strategies.registry import get_strategy

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _interval_minutes(interval_str: str) -> int:
    s = interval_str.strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(s[:-1]) * 60
    raise ValueError(f"Unsupported interval: {interval_str!r}")


def _alpaca_timeframe(interval_minutes: int) -> TimeFrame:
    return TimeFrame(interval_minutes, TimeFrameUnit.Minute)


def _in_session(now: datetime, cfg, tz: ZoneInfo) -> bool:
    loc = now.astimezone(tz)
    params = cfg.strategy.params
    avoid_open  = int(params.get("avoid_open_minutes",  0))
    avoid_close = int(params.get("avoid_close_minutes", 0))
    hh_s, mm_s = map(int, cfg.data.session_start.split(":"))
    hh_e, mm_e = map(int, cfg.data.session_end.split(":"))
    start_m = hh_s * 60 + mm_s + avoid_open
    end_m   = hh_e * 60 + mm_e - avoid_close
    now_m   = loc.hour * 60 + loc.minute
    return start_m <= now_m < end_m


def _past_eod_close(now: datetime, cfg, tz: ZoneInfo) -> bool:
    """True when it's time to force-close open positions before market end."""
    loc = now.astimezone(tz)
    eod_close = int(cfg.strategy.params.get("eod_close_minutes", 5))
    hh_e, mm_e = map(int, cfg.data.session_end.split(":"))
    cutoff_m = hh_e * 60 + mm_e - eod_close
    now_m    = loc.hour * 60 + loc.minute
    return now_m >= cutoff_m


def _warmup_bars(
    key: str, secret: str, symbol: str, interval_min: int, n_bars: int
) -> pd.DataFrame:
    from datetime import timedelta
    client = StockHistoricalDataClient(key, secret)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=max(10, n_bars // (390 // interval_min) + 5))
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=_alpaca_timeframe(interval_min),
        start=start,
        end=end,
        feed="iex",
    )
    bars = client.get_stock_bars(req)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0, axis=0)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1)
    df = df.rename(columns=str.lower)
    needed = {"open", "high", "low", "close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Warmup bars missing columns: {missing}")
    return df.tail(n_bars)


# ── Runner ────────────────────────────────────────────────────────────────────

class PyramidRunner:
    def __init__(self, config_path: str) -> None:
        self.cfg = load_config(config_path)
        self._tz = ZoneInfo(self.cfg.data.timezone)

        broker_cfg = getattr(self.cfg, "broker", {}) or {}
        paper  = str(broker_cfg.get("mode", "paper")).lower() == "paper"
        api_key = broker_cfg.get("api_key") or os.environ.get("ALPACA_API_KEY", "")
        secret  = broker_cfg.get("secret_key") or os.environ.get("ALPACA_SECRET_KEY", "")
        self._broker = AlpacaBroker(api_key=api_key, secret_key=secret, paper=paper)
        self._key, self._secret, self._paper = api_key, secret, paper

        self._strategy     = get_strategy(self.cfg.strategy.name)
        self._params       = self.cfg.strategy.params
        self._interval_min = _interval_minutes(self.cfg.data.interval)
        self._symbol       = self.cfg.data.symbol

        self._buffer = BarBuffer(
            interval_minutes=self._interval_min,
            max_bars=500,
        )

        risk = self.cfg.risk
        self._monitor = PyramidPositionMonitor(
            broker=self._broker,
            capital_per_trade=float(risk.capital_per_trade),
        )
        self._warmed_up = False

        # Pyramid thresholds from config (with diagram defaults)
        self._capital     = float(getattr(risk, "capital_per_trade", 1000.0))
        self._scale1_pct  = float(getattr(risk, "scale1_pct",  0.015))
        self._scale2_pct  = float(getattr(risk, "scale2_pct",  0.027))
        self._stop_pct    = float(getattr(risk, "stop_pct",    0.10))
        self._target1_pct = float(getattr(risk, "target1_pct", 0.034))
        self._target2_pct = float(getattr(risk, "target2_pct", 0.064))
        self._target3_pct = float(getattr(risk, "target3_pct", 0.112))
        self._hold_bars   = int(getattr(risk, "hold_bars", 12))

        mode_str = "PAPER" if paper else "LIVE"
        print(f"\n{'='*60}")
        print(f"  PYRAMID RUNNER — {self._symbol}  [{mode_str}]")
        print(f"  Scale-in  : +{self._scale1_pct*100:.1f}% / +{self._scale2_pct*100:.1f}%")
        print(f"  Stop      : -{self._stop_pct*100:.1f}%")
        print(f"  Targets   : +{self._target1_pct*100:.1f}% / +{self._target2_pct*100:.1f}% / +{self._target3_pct*100:.1f}%")
        print(f"  Hold bars : {self._hold_bars} × {self._interval_min}m = {self._hold_bars * self._interval_min}min")
        print(f"{'='*60}\n")

    # ── Startup ───────────────────────────────────────────────────────────────

    def _close_orphaned_positions(self) -> None:
        """Close any open SPY option positions left from a previous run."""
        try:
            positions = self._broker.get_open_positions()
            # asset_class may come back as "AssetClass.US_OPTION" or "us_option"
            orphans = [
                p for p in positions
                if p["symbol"].startswith(self._symbol + "26")
                and "option" in str(p["asset_class"]).lower()
            ]
            if not orphans:
                return
            print(f"  Found {len(orphans)} orphaned position(s) from previous run — closing:")
            for p in orphans:
                result = self._broker.close_position(p["symbol"])
                status = result.status if result else "unknown"
                print(f"    CLOSED {p['symbol']}  qty={p['qty']}  status={status}")
        except Exception as exc:
            log.warning("Orphan cleanup failed: %s", exc)

    def _close_all_alpaca_option_positions(self, reason: str) -> None:
        """Close ALL open SPY option positions in Alpaca — catches orphans the monitor missed."""
        try:
            positions = self._broker.get_open_positions()
            opts = [
                p for p in positions
                if p["symbol"].startswith(self._symbol + "26")
                and "option" in str(p["asset_class"]).lower()
            ]
            for p in opts:
                result = self._broker.close_position(p["symbol"])
                status = result.status if result else "unknown"
                log.info("EOD-ALPACA-CLOSE %s  qty=%s  reason=%s  status=%s",
                         p["symbol"], p["qty"], reason, status)
        except Exception as exc:
            log.warning("EOD Alpaca close failed: %s", exc)

    def _warmup(self) -> None:
        print(f"Warming up: fetching 500 bars for {self._symbol}...")
        try:
            df = _warmup_bars(self._key, self._secret, self._symbol, self._interval_min, 500)
            self._buffer.add_warmup_bars(df)
            print(f"Warmup complete: {self._buffer.num_completed()} bars loaded.")
            self._warmed_up = True
        except Exception as exc:
            log.error("Warmup failed: %s — will warm up from live stream", exc)

    # ── Per-bar ───────────────────────────────────────────────────────────────

    def _on_completed_bar(self, bar_time: datetime) -> None:
        # 1. Handle pyramid exits / scale-ins on existing position
        events = self._monitor.on_bar_close(bar_time)
        for ev in events:
            print(
                f"  EVENT  {ev.position.contract_symbol}  "
                f"{ev.reason}  qty={ev.qty_affected}  "
                f"ret={ev.option_return_pct*100:+.1f}%  "
                f"P&L=${ev.pnl_usd:+.2f}"
            )

        # 2. EOD force-close: close any open position 5 min before market end (3:55 PM)
        if self._monitor.has_position() and _past_eod_close(bar_time, self.cfg, self._tz):
            et_str = bar_time.astimezone(self._tz).strftime("%H:%M ET")
            log.info("EOD cutoff reached at %s — force-closing all positions", et_str)
            exits = self._monitor.close_all(bar_time, reason="eod_force_close")
            for ev in exits:
                print(f"  EOD-CLOSE  {ev.position.contract_symbol}  P&L=${ev.pnl_usd:+.2f}")
            self._monitor.print_summary()
            return

        # 3. If position still open (but before cutoff), nothing more to do
        if self._monitor.has_position():
            return

        # 4. Session guard — no new entries in last 45 min
        if not _in_session(bar_time, self.cfg, self._tz):
            return
        if not self._warmed_up:
            return

        # 4. Run BB+RSI signal
        df = self._buffer.get_dataframe()
        if len(df) < 30:
            return

        prepared = self._strategy.prepare(df, self._params).dropna()
        if prepared.empty or len(prepared) < 2:
            return

        last = prepared.iloc[-1]
        log.info(
            "BAR  %s  close=%.2f  rsi=%.1f  bb_lo=%.2f  bb_hi=%.2f",
            bar_time.astimezone(self._tz).strftime("%H:%M ET"),
            float(last["close"]),
            float(last["rsi"]),
            float(last["bb_lower"]),
            float(last["bb_upper"]),
        )

        sig = self._strategy.signal(prepared, len(prepared) - 1, self._params)
        if sig is None:
            return

        print(f"\n  SIGNAL  {self._symbol}  side={sig.side}  reason={sig.reason}  @{bar_time.astimezone(self._tz).strftime('%H:%M ET')}")

        # 5. Find option contract
        right = "C" if sig.side == "long" else "P"
        opts  = self.cfg.options
        contract = self._broker.find_option_contract(
            underlying=self._symbol,
            right=right,
            dte_target=opts.dte_target_days,
            otm_steps=opts.otm_steps,
        )
        if contract is None:
            log.warning("No suitable contract found — skipping trade")
            return

        # 6. Get entry mid quote
        quote = self._broker.get_quote(contract.symbol)
        if quote is None or quote.mid <= 0:
            log.warning("No valid quote for %s — skipping trade", contract.symbol)
            return
        entry_opt_price    = quote.mid
        entry_underlying   = self._broker.get_underlying_price(self._symbol)

        # 7. Size qty by capital: floor(capital / (P0 × 100)), minimum 1 contract
        cost_per_contract = entry_opt_price * 100
        qty_initial = max(1, int(self._capital / cost_per_contract))

        print(
            f"  ORDER  BTO {contract.symbol}  x{qty_initial}"
            f"  strike={contract.strike}  expiry={contract.expiration}"
            f"  mid={entry_opt_price:.2f}  cost=${qty_initial*cost_per_contract:.0f}"
        )
        order = self._broker.place_order(contract.symbol, qty_initial, "buy")
        if order.status.lower() in {"rejected", "expired", "canceled", "cancelled", ""}:
            log.error("Entry order failed (status=%s)", order.status)
            return

        # 8. Register pyramid position
        pos = PyramidPosition(
            contract_symbol=contract.symbol,
            underlying_symbol=self._symbol,
            side=sig.side,
            entry_time=bar_time,
            entry_underlying_price=entry_underlying,
            entry_option_price=entry_opt_price,
            qty=qty_initial,
            hold_bars=self._hold_bars,
            scale1_pct=self._scale1_pct,
            scale2_pct=self._scale2_pct,
            stop_pct=self._stop_pct,
            target1_pct=self._target1_pct,
            target2_pct=self._target2_pct,
            target3_pct=self._target3_pct,
        )
        self._monitor.register(pos)

    # ── WebSocket run loop ────────────────────────────────────────────────────

    async def run_async(self) -> None:
        from alpaca.data.live import StockDataStream
        from alpaca.data.enums import DataFeed

        self._close_orphaned_positions()
        self._warmup()

        stream = StockDataStream(self._key, self._secret, feed=DataFeed.IEX)
        _last_day: str | None = None

        async def on_bar(bar) -> None:
            nonlocal _last_day
            ts = bar.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            day_str = ts.astimezone(self._tz).strftime("%Y-%m-%d")
            if day_str != _last_day:
                is_transition = _last_day is not None
                if is_transition:
                    exits = self._monitor.close_all(datetime.now(timezone.utc), reason="day_end")
                    for ev in exits:
                        print(f"  DAY-END  {ev.position.contract_symbol}  P&L=${ev.pnl_usd:+.2f}")
                    self._monitor.print_summary()
                _last_day = day_str
                if not self._warmed_up and is_transition:
                    self._warmed_up = True

            completed = self._buffer.add(
                ts=ts,
                open_=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
            )
            if completed:
                self._on_completed_bar(ts)

        stream.subscribe_bars(on_bar, self._symbol)
        print(f"WebSocket live — streaming 1-min bars for {self._symbol}")
        print("Press Ctrl+C to stop.\n")
        await stream._run_forever()

    def run(self) -> None:
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            print("\nShutting down — closing all open positions...")
            exits = self._monitor.close_all(datetime.now(timezone.utc), reason="shutdown")
            for ev in exits:
                print(f"  CLOSED  {ev.position.contract_symbol}  P&L=${ev.pnl_usd:+.2f}")
            self._monitor.print_summary()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pyramid live runner — BB+RSI + scale-in")
    p.add_argument("--config", required=True, help="Path to pyramid YAML config")
    return p.parse_args()


def main() -> None:
    try:
        from dotenv import load_dotenv
        env_file = Path(__file__).resolve().parents[3] / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
    except ImportError:
        pass

    args = parse_args()
    PyramidRunner(args.config).run()


if __name__ == "__main__":
    main()
