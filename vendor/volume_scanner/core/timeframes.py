"""
core/timeframes.py — Timeframe definitions and volume ratio logic

This module defines HOW each timeframe computes its volume ratio.

A ratio = current_period_volume / prior_period_volume

Each timeframe uses a different comparison window:
  1m  → last completed 1-min bar    vs  bar before it
  15m → last completed 15-min bar   vs  prior 15-min bar
  3h  → sum of last 3 hourly bars   vs  prior 3 hourly bars
  1d  → today's total volume        vs  20-day average daily volume

To add a new timeframe:
  1. Add an entry to TIMEFRAME_CONFIGS in config.py (or directly here)
  2. If it needs a custom ratio formula, add a case to compute_ratio()
  3. Optionally add a custom candle time range in get_candle_range()
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional
from core.fetcher import CandleData

logger = logging.getLogger(__name__)


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class VolumeRatio:
    """
    Result of a volume ratio computation for one timeframe.

    Fields:
        tf_id       : Timeframe identifier (e.g. "1m", "15m", "3h", "1d")
        tf_label    : Human-readable label (e.g. "1 MIN")
        ratio       : current_vol / prior_vol  (the key signal)
        current_vol : Volume in the current period
        prior_vol   : Volume in the comparison period (prior bar or avg)
        num_bars    : Number of bars used for current period
        ok          : False if computation failed (not enough data, etc.)
        note        : Optional explanation (e.g. "market closed", "no data")
    """
    tf_id:       str
    tf_label:    str
    ratio:       float = 0.0
    current_vol: int   = 0
    prior_vol:   int   = 0
    num_bars:    int   = 1
    ok:          bool  = False
    note:        str   = ""

    @property
    def is_unusual(self) -> bool:
        return self.ratio >= 1.8

    @property
    def is_high(self) -> bool:
        return self.ratio >= 3.0

    @property
    def is_extreme(self) -> bool:
        return self.ratio >= 5.0

    @property
    def signal_level(self) -> str:
        if self.ratio >= 5.0: return "EXTREME"
        if self.ratio >= 3.0: return "HIGH"
        if self.ratio >= 1.8: return "UNUSUAL"
        if self.ratio >= 1.2: return "ELEVATED"
        return "NORMAL"


# ── Core ratio computation ─────────────────────────────────────────────────────

def compute_ratio(candles: CandleData, tf_id: str, compare_bars: int = 1) -> VolumeRatio:
    """
    Compute volume ratio for a given timeframe from candle data.

    The formula varies by timeframe type:

    For bar-vs-bar (1m, 15m):
        ratio = last_completed_bar_vol / bar_before_it
        "last completed" = index -2 (index -1 is the in-progress current bar)

    For multi-bar window (3h using hourly bars):
        current_sum = sum(volumes[-4:-1])   # last 3 completed bars
        prior_sum   = sum(volumes[-7:-4])   # 3 bars before that
        ratio = current_sum / prior_sum

    For day-vs-average (1d):
        today_vol   = volumes[-1]           # today's volume so far
        avg_vol     = mean(volumes[:-1])    # average of prior N days
        ratio = today_vol / avg_vol

    Parameters:
        candles      : CandleData object from fetcher
        tf_id        : Timeframe ID ("1m", "15m", "3h", "1d")
        compare_bars : How many bars = one "period" for this TF

    Returns VolumeRatio dataclass.

    To add a custom ratio formula:
        Add a new elif branch below matching your new tf_id.
    """
    label_map = {"1m": "1 MIN", "15m": "15 MIN", "3h": "3 HOUR", "1d": "DAY"}
    label = label_map.get(tf_id, tf_id.upper())

    if not candles.ok or len(candles.volumes) < 3:
        return VolumeRatio(tf_id=tf_id, tf_label=label, ok=False,
                           note="insufficient data")

    vols = candles.volumes
    n = len(vols)

    try:
        if tf_id in ("1m", "15m"):
            # ── Bar-vs-bar comparison ──────────────────────────────────────────
            # Index -2 = last COMPLETED bar (index -1 is still forming)
            # Index -3 = bar before that
            cur_vol  = vols[-2]
            prev_vol = vols[-3] if n >= 3 else vols[-2]
            if prev_vol == 0:
                return VolumeRatio(tf_id=tf_id, tf_label=label, ok=False,
                                   note="prior bar volume is zero")
            return VolumeRatio(
                tf_id=tf_id, tf_label=label,
                ratio=cur_vol / prev_vol,
                current_vol=int(cur_vol),
                prior_vol=int(prev_vol),
                num_bars=1,
                ok=True,
            )

        elif tf_id == "3h":
            # ── Multi-bar window comparison ────────────────────────────────────
            # Using hourly bars: sum last 3 completed hourly bars vs prior 3
            # We skip index -1 (in-progress bar)
            completed = vols[:-1]   # exclude in-progress bar
            m = len(completed)
            if m < 6:
                return VolumeRatio(tf_id=tf_id, tf_label=label, ok=False,
                                   note=f"need 6 completed bars, got {m}")
            cur_sum  = sum(completed[max(0, m-3):m])
            prev_sum = sum(completed[max(0, m-6):max(0, m-3)])
            if prev_sum == 0:
                return VolumeRatio(tf_id=tf_id, tf_label=label, ok=False,
                                   note="prior window volume is zero")
            return VolumeRatio(
                tf_id=tf_id, tf_label=label,
                ratio=cur_sum / prev_sum,
                current_vol=int(cur_sum),
                prior_vol=int(prev_sum),
                num_bars=3,
                ok=True,
            )

        elif tf_id == "1d":
            # ── Today vs N-day average ─────────────────────────────────────────
            # Today's volume is the last bar (index -1)
            # Average is computed over all prior bars (exclude today)
            today_vol  = vols[-1]
            prior_vols = [v for v in vols[:-1] if v > 0]   # filter zero-volume days
            if not prior_vols:
                return VolumeRatio(tf_id=tf_id, tf_label=label, ok=False,
                                   note="no prior day volume data")
            avg_vol = sum(prior_vols) / len(prior_vols)
            if avg_vol == 0:
                return VolumeRatio(tf_id=tf_id, tf_label=label, ok=False,
                                   note="avg volume is zero")
            return VolumeRatio(
                tf_id=tf_id, tf_label=label,
                ratio=today_vol / avg_vol,
                current_vol=int(today_vol),
                prior_vol=int(avg_vol),
                num_bars=1,
                ok=True,
            )

        else:
            # ── Generic bar-vs-bar for any custom timeframe ────────────────────
            # Uses compare_bars to define period size
            completed = vols[:-1]
            m = len(completed)
            cb = compare_bars
            if m < cb * 2:
                return VolumeRatio(tf_id=tf_id, tf_label=label, ok=False,
                                   note=f"need {cb*2} bars, got {m}")
            cur_sum  = sum(completed[max(0, m-cb):m])
            prev_sum = sum(completed[max(0, m-(cb*2)):max(0, m-cb)])
            if prev_sum == 0:
                return VolumeRatio(tf_id=tf_id, tf_label=label, ok=False,
                                   note="prior period volume is zero")
            return VolumeRatio(
                tf_id=tf_id, tf_label=label,
                ratio=cur_sum / prev_sum,
                current_vol=int(cur_sum),
                prior_vol=int(prev_sum),
                num_bars=cb,
                ok=True,
            )

    except (IndexError, ZeroDivisionError) as e:
        logger.debug(f"Ratio compute error for {tf_id}: {e}")
        return VolumeRatio(tf_id=tf_id, tf_label=label, ok=False, note=str(e))


def get_candle_range(tf_id: str, config_tf: dict) -> tuple[int, int]:
    """
    Return (from_timestamp, to_timestamp) for fetching candles for a given timeframe.

    The lookback window is sized to give enough bars for ratio computation
    plus extra history for indicator calculation (RSI, MACD, etc. need ~30+ bars).

    Parameters:
        tf_id     : Timeframe ID
        config_tf : The timeframe config dict from config.TIMEFRAMES[tf_id]

    Returns (from_ts, to_ts) as Unix timestamps.
    """
    now = int(time.time())
    resolution = config_tf.get("resolution", "D")
    lookback   = config_tf.get("lookback_bars", 25)

    # Seconds per bar for each resolution
    bar_seconds = {
        "1":  60,
        "5":  300,
        "15": 900,
        "30": 1800,
        "60": 3600,
        "D":  86400,
        "W":  604800,
    }
    secs_per_bar = bar_seconds.get(resolution, 86400)

    # Add 50% buffer for weekends/holidays where markets are closed
    buffer_factor = 1.8 if resolution == "D" else 1.3
    from_ts = now - int(lookback * secs_per_bar * buffer_factor)

    return from_ts, now
