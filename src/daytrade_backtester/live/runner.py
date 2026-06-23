"""
Live trading runner — entry point: `dtb-live --config <yaml>`

Flow
----
1. Load config (YAML — same schema as backtest, plus a `broker:` section)
2. Initialise AlpacaBroker, BarBuffer, PositionMonitor
3. Warm up: pull the last ~500 bars via Alpaca REST so indicators have data
4. Connect Alpaca WebSocket → subscribe to 1-min bars
5. On each 1-min bar:
   a. BarBuffer.add() → True when a completed N-min bar is sealed
   b. PositionMonitor.on_bar_close()   ← check exits first
   c. If signal fires AND conditions met → find contract → place BTO order
6. At session end: close_all() → print_summary()
"""
from __future__ import annotations

import argparse
import asyncio
import json
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
from daytrade_backtester.live.position_monitor import LivePosition, PositionMonitor
from daytrade_backtester.strategies.registry import get_strategy

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _interval_minutes(interval_str: str) -> int:
    """Parse '5m', '1h', '15m' → integer minutes."""
    s = interval_str.strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(s[:-1]) * 60
    raise ValueError(f"Unsupported interval: {interval_str!r}")


def _alpaca_timeframe(interval_minutes: int) -> TimeFrame:
    return TimeFrame(interval_minutes, TimeFrameUnit.Minute)


def _in_session(now: datetime, cfg, tz: ZoneInfo) -> bool:
    """True if current time is within the trading session window."""
    loc = now.astimezone(tz)
    params = cfg.strategy.params
    avoid_open = int(params.get("avoid_open_minutes", 0))
    avoid_close = int(params.get("avoid_close_minutes", 0))

    sess_start = cfg.data.session_start  # "09:30"
    sess_end   = cfg.data.session_end    # "16:00"
    hh_s, mm_s = map(int, sess_start.split(":"))
    hh_e, mm_e = map(int, sess_end.split(":"))

    start_m = hh_s * 60 + mm_s + avoid_open
    end_m   = hh_e * 60 + mm_e - avoid_close
    now_m   = loc.hour * 60 + loc.minute
    return start_m <= now_m < end_m


def _warmup_bars(broker_key: str, broker_secret: str, symbol: str, interval_min: int, n_bars: int) -> pd.DataFrame:
    """Fetch recent historical bars from Alpaca REST for indicator warmup."""
    client = StockHistoricalDataClient(broker_key, broker_secret)
    from datetime import timedelta
    end = datetime.now(timezone.utc)
    # Fetch extra days to account for weekends/holidays
    start = end - timedelta(days=max(10, n_bars // (390 // interval_min) + 5))
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=_alpaca_timeframe(interval_min),
        start=start,
        end=end,
        feed="iex",          # free feed; use "sip" for paid
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


# ── Position checkpoint helpers ───────────────────────────────────────────────

def _save_positions(positions: list, path: Path) -> None:
    """Persist open LivePosition list to JSON so they survive a crash/restart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "contract_symbol":        p.contract_symbol,
            "underlying_symbol":      p.underlying_symbol,
            "side":                   p.side,
            "entry_time":             p.entry_time.isoformat(),
            "entry_underlying_price": p.entry_underlying_price,
            "entry_option_price":     p.entry_option_price,
            "qty":                    p.qty,
            "hold_bars":              p.hold_bars,
            "early_exit_bar":         p.early_exit_bar,
            "early_exit_pct":         p.early_exit_pct,
            "option_target_pct":      p.option_target_pct,
            "bars_elapsed":           p.bars_elapsed,
        }
        for p in positions
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _restore_positions(path: Path, interval_min: int) -> list:
    """Reload positions saved before a crash; skip stale entries from a prior day."""
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        log.warning("Could not read position checkpoint %s: %s", path, exc)
        return []

    today = datetime.now(timezone.utc).date()
    restored = []
    for d in data:
        entry_time = datetime.fromisoformat(d["entry_time"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        if entry_time.date() != today:
            continue  # previous-day stale — skip
        # Recompute bars_elapsed from wall-clock time so time exits fire correctly.
        elapsed_secs = (datetime.now(timezone.utc) - entry_time).total_seconds()
        bars_elapsed = max(d.get("bars_elapsed", 0), int(elapsed_secs / (interval_min * 60)))
        restored.append(LivePosition(
            contract_symbol=d["contract_symbol"],
            underlying_symbol=d["underlying_symbol"],
            side=d["side"],
            entry_time=entry_time,
            entry_underlying_price=d["entry_underlying_price"],
            entry_option_price=d["entry_option_price"],
            qty=d["qty"],
            hold_bars=d["hold_bars"],
            early_exit_bar=d["early_exit_bar"],
            early_exit_pct=d["early_exit_pct"],
            option_target_pct=d["option_target_pct"],
            bars_elapsed=bars_elapsed,
        ))
    return restored


# ── Main runner ───────────────────────────────────────────────────────────────

class LiveRunner:
    def __init__(self, config_path: str) -> None:
        self.cfg = load_config(config_path)
        self._tz = ZoneInfo(self.cfg.data.timezone)

        # Broker
        broker_cfg = getattr(self.cfg, "broker", {}) or {}
        paper = str(broker_cfg.get("mode", "paper")).lower() == "paper"
        api_key = broker_cfg.get("api_key") or os.environ.get("ALPACA_API_KEY", "")
        secret  = broker_cfg.get("secret_key") or os.environ.get("ALPACA_SECRET_KEY", "")
        self._broker = AlpacaBroker(api_key=api_key, secret_key=secret, paper=paper)
        self._key = api_key
        self._secret = secret
        self._paper = paper

        # Strategy
        self._strategy = get_strategy(self.cfg.strategy.name)
        self._params = self.cfg.strategy.params
        self._interval_min = _interval_minutes(self.cfg.data.interval)
        self._symbol = self.cfg.data.symbol

        # Bar buffer (needs enough bars for all indicators + SMA warmup)
        sma_len = int(self._params.get("sma_length", 0))
        self._warmup_needed = max(500, sma_len + 50)
        self._buffer = BarBuffer(
            interval_minutes=self._interval_min,
            max_bars=self._warmup_needed,
        )

        # Position / risk tracking
        risk = self.cfg.risk
        self._monitor = PositionMonitor(
            broker=self._broker,
            capital_per_trade=risk.capital_per_trade,
        )
        self._trades_today = 0
        self._cooldown_bars_left = 0
        self._warmed_up = False

        # Crash-resilience: persist open positions so they survive launchd restarts.
        self._positions_path = Path(config_path).resolve().parent.parent / "logs" / "spy_positions.json"

    # ── Startup ───────────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        print(f"Warming up: fetching last {self._warmup_needed} bars for {self._symbol}...")
        try:
            df = _warmup_bars(self._key, self._secret, self._symbol, self._interval_min, self._warmup_needed)
            self._buffer.add_warmup_bars(df)
            print(f"Warmup complete: {self._buffer.num_completed()} bars loaded.")
            self._warmed_up = True
        except Exception as exc:
            log.error("Warmup failed: %s — will warm up from live stream", exc)

        # Restore any positions that were open when the process last crashed.
        for pos in _restore_positions(self._positions_path, self._interval_min):
            log.warning(
                "Restored position after crash: %s  entry_opt=%.4f  bars_elapsed=%d",
                pos.contract_symbol, pos.entry_option_price, pos.bars_elapsed,
            )
            self._monitor.register(pos)
            self._trades_today += 1

    # ── Per-bar logic ─────────────────────────────────────────────────────────

    def _on_completed_bar(self, bar_time: datetime) -> None:
        """Called once a new N-min bar is sealed. Core signal + exit logic."""

        # 1. Check exits first — start cooldown when a position closes (not at entry)
        exits = self._monitor.on_bar_close(bar_time)
        for ex in exits:
            print(
                f"  EXIT  {ex.position.contract_symbol}  "
                f"reason={ex.exit_reason}  "
                f"ret={ex.option_return_pct*100:+.1f}%  "
                f"P&L=${ex.pnl_usd:+.2f}"
            )
            self._cooldown_bars_left = int(getattr(self.cfg.risk, "cooldown_bars", 0))
        if exits:
            _save_positions(self._monitor.open_positions, self._positions_path)

        # 2. Cooldown decrement
        if self._cooldown_bars_left > 0:
            self._cooldown_bars_left -= 1

        # 3. Gate checks
        risk = self.cfg.risk
        if self._monitor.open_positions:
            return  # only one position at a time
        if self._cooldown_bars_left > 0:
            return
        max_day = int(getattr(risk, "max_trades_per_day", 0))
        if max_day > 0 and self._trades_today >= max_day:
            return
        if not _in_session(bar_time, self.cfg, self._tz):
            return
        if not self._warmed_up:
            return

        # 4. Run strategy signal
        df = self._buffer.get_dataframe()
        if len(df) < 30:
            return   # not enough bars yet

        prepared = self._strategy.prepare(df, self._params).dropna()
        if prepared.empty or len(prepared) < 2:
            return

        last = prepared.iloc[-1]
        log.info(
            "BAR  %s  close=%.2f  rsi=%.1f  bb_lower=%.2f  bb_upper=%.2f  below_band=%s",
            bar_time.astimezone(self._tz).strftime("%H:%M ET"),
            float(last["close"]),
            float(last["rsi"]),
            float(last["bb_lower"]),
            float(last["bb_upper"]),
            float(last["close"]) < float(last["bb_lower"]),
        )

        sig = self._strategy.signal(prepared, len(prepared) - 1, self._params)
        if sig is None:
            return

        print(f"\n  SIGNAL  {self._symbol}  side={sig.side}  reason={sig.reason}  @{bar_time.astimezone(self._tz).strftime('%H:%M ET')}")

        # 5. Find option contract
        right = "C" if sig.side == "long" else "P"
        opts = self.cfg.options
        contract = self._broker.find_option_contract(
            underlying=self._symbol,
            right=right,
            dte_target=opts.dte_target_days,
            otm_steps=opts.otm_steps,
        )
        if contract is None:
            log.warning("No suitable contract found — skipping trade")
            return

        # 6. Get entry option price (mid quote)
        quote = self._broker.get_quote(contract.symbol)
        if quote is None or quote.mid <= 0:
            log.warning("No valid quote for %s — skipping trade", contract.symbol)
            return
        entry_opt_price = quote.mid
        entry_underlying = self._broker.get_underlying_price(self._symbol)

        # 7. Place BTO order
        print(f"  ORDER   BTO {contract.symbol}  x1  strike={contract.strike}  expiry={contract.expiration}  opt_mid={entry_opt_price:.4f}")
        order = self._broker.place_order(symbol=contract.symbol, qty=1, side="buy")
        if order.status.lower() in {"rejected", "expired", "canceled", "cancelled", ""}:
            log.error("Order failed (status=%s): %s", order.status, order)
            return

        # 8. Register position
        pos = LivePosition(
            contract_symbol=contract.symbol,
            underlying_symbol=self._symbol,
            side=sig.side,
            entry_time=bar_time,
            entry_underlying_price=entry_underlying,
            entry_option_price=entry_opt_price,
            qty=1,
            hold_bars=risk.hold_bars,
            early_exit_bar=int(getattr(risk, "early_exit_bar", 0)),
            early_exit_pct=float(getattr(risk, "early_exit_pct", -1.0)),
            option_target_pct=float(getattr(risk, "option_target_pct", 0.05)),
        )
        self._monitor.register(pos)
        self._trades_today += 1
        _save_positions(self._monitor.open_positions, self._positions_path)

    # ── Day reset ─────────────────────────────────────────────────────────────

    def _reset_day(self, bar_time: datetime) -> None:
        loc = bar_time.astimezone(self._tz)
        log.debug("Day reset at %s", loc.strftime("%Y-%m-%d"))
        self._trades_today = 0
        self._cooldown_bars_left = 0

    # ── WebSocket run loop ────────────────────────────────────────────────────

    async def run_async(self) -> None:
        from alpaca.data.live import StockDataStream
        from alpaca.data.enums import DataFeed

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
                is_day_transition = _last_day is not None  # False on cold start
                if is_day_transition:
                    # Session end for previous day
                    prev_close = datetime.now(timezone.utc)
                    exits = self._monitor.close_all(prev_close, reason="day_end_exit")
                    for ex in exits:
                        print(f"  DAY-END-EXIT  {ex.position.contract_symbol}  P&L=${ex.pnl_usd:+.2f}")
                    _save_positions([], self._positions_path)  # clear stale checkpoint
                    self._monitor.print_summary()
                    self._reset_day(ts)
                _last_day = day_str
                # Only re-enable trading on a real day change (not cold-start).
                # On cold-start, REST warmup already sets _warmed_up; if it failed,
                # leave it False so the len(df)<30 guard and dropna() act as backstop.
                if not self._warmed_up and is_day_transition:
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
        print(f"\nWebSocket connected — streaming 1-min bars for {self._symbol}")
        print("Press Ctrl+C to stop.\n")
        await stream._run_forever()

    def run(self) -> None:
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            print("\nShutdown requested — closing all open positions...")
            exits = self._monitor.close_all(datetime.now(timezone.utc), reason="day_end_exit")
            for ex in exits:
                print(f"  CLOSED  {ex.position.contract_symbol}  P&L=${ex.pnl_usd:+.2f}")
            _save_positions([], self._positions_path)
            self._monitor.print_summary()


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live trading runner — Alpaca paper/live")
    p.add_argument("--config", required=True, help="Path to live YAML config")
    return p.parse_args()


def main() -> None:
    # Load .env from project root if present — lets keys live in .env instead of requiring export
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        env_file = Path(__file__).resolve().parents[3] / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
    except ImportError:
        pass

    args = parse_args()
    runner = LiveRunner(args.config)
    runner.run()


if __name__ == "__main__":
    main()
