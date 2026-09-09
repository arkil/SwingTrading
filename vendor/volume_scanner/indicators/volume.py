"""
indicators/volume.py — Volume-based technical indicators

All functions accept pandas Series or lists of OHLCV data
and return a single scalar value (the latest reading).

To add a new indicator:
  1. Add a function following the pattern below
  2. Call it in composite.py's build_feature_dict()
  3. Add the feature name to models/features.py FEATURE_COLUMNS

Each function is documented with:
  - What it measures
  - Typical interpretation ranges
  - Parameters you can tune in config.INDICATOR_SETTINGS
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def safe_div(a, b, default=0.0):
    """Safe division returning default if b is zero."""
    return a / b if b and b != 0 else default


# ── Volume Z-Score ─────────────────────────────────────────────────────────────

def volume_zscore(volumes: list, window: int = 20) -> float:
    """
    How many standard deviations above/below average is current volume?

    Interpretation:
        > +2.0  : Extremely unusual, strong anomaly signal
        > +1.5  : Unusual volume
        +0 to +1: Normal range
        < -1.0  : Notably low volume (often pre-breakout coiling)

    Parameters (tune in config.INDICATOR_SETTINGS):
        window : Rolling window for mean/std calculation (default: 20)

    Returns:
        Z-score of the most recent volume bar. Returns 0.0 if insufficient data.
    """
    if len(volumes) < window + 1:
        return 0.0
    arr = np.array(volumes[-window-1:-1], dtype=float)  # exclude current bar
    mean = arr.mean()
    std  = arr.std()
    current = volumes[-1]
    return safe_div(current - mean, std)


# ── VWAP — Volume Weighted Average Price ───────────────────────────────────────

def vwap(highs: list, lows: list, closes: list, volumes: list) -> float:
    """
    Intraday Volume Weighted Average Price.

    VWAP = sum(typical_price * volume) / sum(volume)
    typical_price = (high + low + close) / 3

    Interpretation:
        Price above VWAP → bullish intraday momentum
        Price below VWAP → bearish intraday momentum
        Price crossing VWAP → potential reversal signal

    Used as a feature via price_vs_vwap = (current_price - vwap) / vwap * 100

    Returns:
        VWAP price level, or latest close if computation fails.
    """
    if not volumes or sum(volumes) == 0:
        return closes[-1] if closes else 0.0
    typical = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    num   = sum(tp * v for tp, v in zip(typical, volumes))
    denom = sum(volumes)
    return safe_div(num, denom, closes[-1] if closes else 0.0)


# ── OBV — On-Balance Volume ────────────────────────────────────────────────────

def obv(closes: list, volumes: list) -> list:
    """
    On-Balance Volume — cumulative volume flow.

    OBV adds volume when close > prior close, subtracts when close < prior close.
    Divergence between OBV and price is a powerful reversal signal.

    Interpretation:
        Rising OBV with rising price  → confirmed uptrend
        Rising OBV with falling price → bullish divergence (accumulation)
        Falling OBV with rising price → bearish divergence (distribution)

    Returns:
        List of OBV values (same length as input, first value = first volume).

    To use as a feature, compute obv_slope (linear regression slope of last N bars).
    """
    if len(closes) < 2:
        return list(volumes)
    result = [volumes[0]]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            result.append(result[-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            result.append(result[-1] - volumes[i])
        else:
            result.append(result[-1])
    return result


def obv_slope(closes: list, volumes: list, bars: int = 5) -> float:
    """
    Linear regression slope of OBV over the last `bars` periods.

    Positive slope → accumulation trend
    Negative slope → distribution trend

    Parameters (tune in config.INDICATOR_SETTINGS):
        bars : Number of recent bars to measure slope over (default: 5)

    Returns:
        Slope value normalized by average OBV magnitude.
        Positive = accumulation, negative = distribution.
    """
    obv_vals = obv(closes, volumes)
    if len(obv_vals) < bars:
        return 0.0
    recent = np.array(obv_vals[-bars:], dtype=float)
    x = np.arange(bars)
    # Normalize slope by mean absolute value to make it comparable across stocks
    slope = np.polyfit(x, recent, 1)[0]
    scale = np.abs(recent).mean()
    return safe_div(slope, scale)


# ── MFI — Money Flow Index ─────────────────────────────────────────────────────

def mfi(highs: list, lows: list, closes: list, volumes: list,
        period: int = 14) -> float:
    """
    Money Flow Index — volume-weighted RSI.

    MFI oscillates 0–100 and incorporates both price and volume.

    Interpretation:
        > 80   : Overbought (potential reversal down)
        < 20   : Oversold (potential reversal up)
        50–80  : Bullish zone
        20–50  : Bearish zone
        Divergence with price is a strong signal

    Parameters (tune in config.INDICATOR_SETTINGS):
        period : Lookback period (default: 14)

    Returns:
        MFI value 0–100, or 50.0 if insufficient data.
    """
    if len(closes) < period + 1:
        return 50.0

    typical = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    raw_mf  = [tp * v for tp, v in zip(typical, volumes)]

    pos_mf, neg_mf = 0.0, 0.0
    for i in range(-period, 0):
        if typical[i] > typical[i-1]:
            pos_mf += raw_mf[i]
        elif typical[i] < typical[i-1]:
            neg_mf += raw_mf[i]

    if neg_mf == 0:
        return 100.0
    mf_ratio = safe_div(pos_mf, neg_mf)
    return 100 - (100 / (1 + mf_ratio))


# ── CMF — Chaikin Money Flow ───────────────────────────────────────────────────

def cmf(highs: list, lows: list, closes: list, volumes: list,
        period: int = 20) -> float:
    """
    Chaikin Money Flow — measures buying vs selling pressure over N periods.

    CMF = sum(money_flow_volume) / sum(volume) over N periods
    money_flow_volume = ((close - low) - (high - close)) / (high - low) * volume

    Interpretation:
        > +0.25  : Strong buying pressure (bullish)
        0 to +0.25: Moderate accumulation
        -0.25 to 0: Moderate distribution
        < -0.25  : Strong selling pressure (bearish)

    Parameters (tune in config.INDICATOR_SETTINGS):
        period : Lookback period (default: 20)

    Returns:
        CMF value typically between -1 and +1.
    """
    if len(closes) < period:
        return 0.0

    mf_vol_sum = 0.0
    vol_sum    = 0.0
    for i in range(-period, 0):
        h, l, c, v = highs[i], lows[i], closes[i], volumes[i]
        hl_range = h - l
        if hl_range == 0:
            continue
        clv = ((c - l) - (h - c)) / hl_range  # close location value
        mf_vol_sum += clv * v
        vol_sum    += v

    return safe_div(mf_vol_sum, vol_sum)


# ── Accumulation/Distribution Line ────────────────────────────────────────────

def ad_line(highs: list, lows: list, closes: list, volumes: list) -> list:
    """
    Accumulation/Distribution Line — running total of money flow volume.

    Similar to OBV but uses close location value instead of raw up/down.

    Interpretation:
        Rising AD with rising price  → confirmed trend
        Rising AD with falling price → bullish divergence
        Falling AD with rising price → bearish divergence (warning sign)

    Returns:
        List of AD values. Use ad_slope() for the feature value.
    """
    result = []
    running = 0.0
    for h, l, c, v in zip(highs, lows, closes, volumes):
        hl = h - l
        if hl > 0:
            clv = ((c - l) - (h - c)) / hl
            running += clv * v
        result.append(running)
    return result


def ad_slope(highs: list, lows: list, closes: list, volumes: list,
             bars: int = 5) -> float:
    """
    Slope of the AD line over last `bars` periods.

    Returns normalized slope (positive = accumulation, negative = distribution).
    """
    ad = ad_line(highs, lows, closes, volumes)
    if len(ad) < bars:
        return 0.0
    recent = np.array(ad[-bars:], dtype=float)
    x = np.arange(bars)
    slope = np.polyfit(x, recent, 1)[0]
    scale = np.abs(recent).mean()
    return safe_div(slope, scale)


# ── Volume Moving Average Ratio ────────────────────────────────────────────────

def vol_ma_ratio(volumes: list, period: int = 20) -> float:
    """
    Current volume / N-period volume moving average.

    This is the simplest volume anomaly signal — just how much higher
    is today's volume vs the rolling average?

    Interpretation:
        > 2.0 : Volume is double average → significant anomaly
        > 3.0 : Triple average → strong signal
        1.0   : At average, no anomaly

    Parameters (tune in config.INDICATOR_SETTINGS):
        period : MA period (default: 20)

    Returns:
        Ratio value. Returns 1.0 if insufficient data.
    """
    if len(volumes) < period + 1:
        return 1.0
    prior_avg = np.mean(volumes[-period-1:-1])
    current   = volumes[-1]
    return safe_div(current, prior_avg, 1.0)


# ── Volume Acceleration ────────────────────────────────────────────────────────

def volume_acceleration(volumes: list, bars: int = 3) -> float:
    """
    Second derivative of volume — is the rate of change itself increasing?

    Computed as: (current_vol - prior_vol) vs (prior_vol - vol_before_that)

    Interpretation:
        > 0 : Volume acceleration (surge is getting stronger)
        < 0 : Volume deceleration (surge may be fading)
        0   : Stable

    Returns:
        Normalized acceleration value.
    """
    if len(volumes) < bars + 1:
        return 0.0
    recent = volumes[-bars-1:]
    diffs  = [recent[i] - recent[i-1] for i in range(1, len(recent))]
    if len(diffs) < 2:
        return 0.0
    accel = diffs[-1] - diffs[-2]
    scale = max(abs(diffs[-1]), abs(diffs[-2]), 1)
    return accel / scale
