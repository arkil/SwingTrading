from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger(__name__)


class BarBuffer:
    """
    Aggregates 1-minute WebSocket bars into N-minute OHLCV bars.

    Usage
    -----
    buf = BarBuffer(interval_minutes=5, max_bars=500)

    # On each 1-min bar from WebSocket:
    completed = buf.add(ts, open_, high, low, close, volume)
    if completed:
        df = buf.get_dataframe()   # feed to strategy.prepare()

    Design
    ------
    - Bars are bucketed by flooring the timestamp to the nearest N-minute boundary.
    - A bucket is "completed" when the first bar with a NEW bucket timestamp arrives,
      sealing the previous bucket.
    - The rolling window holds at most `max_bars` completed N-min bars.
    """

    def __init__(self, interval_minutes: int = 5, max_bars: int = 500) -> None:
        self.interval_minutes = interval_minutes
        self.max_bars = max_bars
        self._completed: deque[dict] = deque(maxlen=max_bars)

        # Current (in-progress) N-min bucket
        self._cur_bucket: str | None = None   # bucket key "YYYY-MM-DD HH:MM"
        self._cur_open: float | None = None
        self._cur_high: float | None = None
        self._cur_low: float | None = None
        self._cur_close: float | None = None
        self._cur_volume: float = 0.0
        self._cur_ts: datetime | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        ts: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> bool:
        """
        Add a 1-minute bar. Returns True if a new N-minute bar was completed.
        """
        bucket = self._bucket_key(ts)

        if self._cur_bucket is None:
            # Very first bar
            self._start_bucket(bucket, ts, open_, high, low, close, volume)
            return False

        if bucket == self._cur_bucket:
            # Same N-min window — update running OHLCV
            self._cur_high = max(self._cur_high, high)
            self._cur_low  = min(self._cur_low, low)
            self._cur_close = close
            self._cur_volume += volume
            self._cur_ts = ts
            return False

        # New bucket arrived — seal the previous one
        self._seal_bucket()
        self._start_bucket(bucket, ts, open_, high, low, close, volume)
        return True

    def add_warmup_bars(self, df: pd.DataFrame) -> None:
        """
        Load a historical OHLCV DataFrame (already at strategy interval) into
        the completed-bar window.  Used during startup warmup.

        df must have a DatetimeIndex and columns: open, high, low, close, volume.
        """
        for ts, row in df.iterrows():
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            self._completed.append({
                "ts": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low":  float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
        log.debug("BarBuffer: loaded %d warmup bars", len(df))

    def get_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame of completed N-min bars, ready for strategy.prepare()."""
        if not self._completed:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        rows = list(self._completed)
        df = pd.DataFrame(rows).set_index("ts")
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()

    def num_completed(self) -> int:
        return len(self._completed)

    def last_bar_time(self) -> datetime | None:
        if self._completed:
            return self._completed[-1]["ts"]
        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _bucket_key(self, ts: datetime) -> str:
        """Floor timestamp to the nearest N-minute boundary."""
        n = self.interval_minutes
        floored_minute = (ts.minute // n) * n
        return ts.strftime(f"%Y-%m-%d %H:") + f"{floored_minute:02d}"

    def _start_bucket(
        self,
        bucket: str,
        ts: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        self._cur_bucket = bucket
        self._cur_ts = ts
        self._cur_open = open_
        self._cur_high = high
        self._cur_low = low
        self._cur_close = close
        self._cur_volume = volume

    def _seal_bucket(self) -> None:
        if self._cur_bucket is None:
            return
        self._completed.append({
            "ts": self._cur_ts,
            "open":   self._cur_open,
            "high":   self._cur_high,
            "low":    self._cur_low,
            "close":  self._cur_close,
            "volume": self._cur_volume,
        })
        log.debug(
            "Bar sealed: %s  O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f",
            self._cur_bucket,
            self._cur_open, self._cur_high, self._cur_low, self._cur_close, self._cur_volume,
        )
