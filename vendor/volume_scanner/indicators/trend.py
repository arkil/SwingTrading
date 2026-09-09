"""
indicators/trend.py — Trend and volatility indicators

These indicators tell you:
  - What direction is the trend?     (SMA, EMA, SuperTrend, ADX)
  - How strong is the trend?         (ADX)
  - How far is price from equilibrium? (Bollinger Bands)
  - What is the current volatility?  (ATR, BB Width)

Combined with volume anomalies:
  - High volume + price above BB upper → potential breakout (confirm with ADX)
  - High volume + price below BB lower → potential breakdown or reversal
  - High volume + ADX > 25 → trending market, breakout more reliable
  - High volume + ADX < 20 → ranging market, breakout may fail
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def safe_div(a, b, default=0.0):
    return a / b if b and b != 0 else default


# ── Simple Moving Average ─────────────────────────────────────────────────────

def sma(closes: list, period: int) -> float:
    """
    Simple Moving Average of closes over `period` bars.

    Returns the latest SMA value, or 0.0 if insufficient data.
    """
    if len(closes) < period:
        return 0.0
    return float(np.mean(closes[-period:]))


def price_vs_sma(closes: list, period: int) -> float:
    """
    Percentage distance of current price above/below SMA.

    +5.0 means price is 5% above the SMA → momentum
    -5.0 means price is 5% below the SMA → potential support

    Returns 0.0 if insufficient data.
    """
    s = sma(closes, period)
    if s == 0:
        return 0.0
    return safe_div(closes[-1] - s, s) * 100


# ── Exponential Moving Average ────────────────────────────────────────────────

def ema(closes: list, period: int) -> float:
    """
    Exponential Moving Average — weights recent prices more heavily.

    More responsive to recent price changes than SMA.
    Common configurations:
      9 EMA  = very short-term momentum
      21 EMA = short-term trend
      55 EMA = medium-term trend
      200 EMA = long-term trend (institutional baseline)

    Returns latest EMA value, or 0.0 if insufficient data.
    """
    if len(closes) < period:
        return 0.0
    k = 2 / (period + 1)
    result = closes[0]
    for v in closes[1:]:
        result = v * k + result * (1 - k)
    return result


def ema_cross(closes: list, fast: int = 9, slow: int = 21) -> float:
    """
    Fast EMA minus Slow EMA, normalized by slow EMA.

    Positive → fast EMA above slow EMA → bullish
    Negative → fast EMA below slow EMA → bearish
    Crossing zero → trend change signal

    Returns normalized spread. +0.02 means fast EMA is 2% above slow EMA.
    """
    e_fast = ema(closes, fast)
    e_slow = ema(closes, slow)
    return safe_div(e_fast - e_slow, e_slow)


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def bollinger_bands(closes: list, period: int = 20,
                    std_dev: float = 2.0) -> tuple[float, float, float]:
    """
    Bollinger Bands — volatility envelope around a moving average.

    Upper Band = SMA(period) + std_dev * std(period)
    Middle     = SMA(period)
    Lower Band = SMA(period) - std_dev * std(period)

    Returns (upper, middle, lower).

    Interpretation:
        Price touches/exceeds upper band → overbought or strong breakout
        Price touches/exceeds lower band → oversold or strong breakdown
        Bands narrowing (squeeze) → low volatility, breakout often imminent
        Bands widening → high volatility, trend in motion

    In the context of volume anomalies:
        High volume + price above upper BB → breakout confirmation
        High volume + price below lower BB → breakdown or capitulation reversal

    Parameters (tune in config.INDICATOR_SETTINGS):
        period  : SMA/std period (default: 20)
        std_dev : Number of standard deviations for bands (default: 2.0)
                  Use 1.5 for tighter bands, 2.5 for wider bands

    Returns (upper, middle, lower) or (0, 0, 0) if insufficient data.
    """
    if len(closes) < period:
        return 0.0, 0.0, 0.0
    window = closes[-period:]
    mid   = float(np.mean(window))
    sigma = float(np.std(window))
    return mid + std_dev * sigma, mid, mid - std_dev * sigma


def bb_position(closes: list, period: int = 20, std_dev: float = 2.0) -> float:
    """
    Where is current price within the Bollinger Bands? (0 to 1 scale)

    0.0 = at lower band
    0.5 = at middle (SMA)
    1.0 = at upper band
    > 1.0 = above upper band (breakout)
    < 0.0 = below lower band (breakdown)

    This is the %B indicator. Use as an ML feature directly.
    """
    upper, mid, lower = bollinger_bands(closes, period, std_dev)
    if upper == lower:
        return 0.5
    return safe_div(closes[-1] - lower, upper - lower, 0.5)


def bb_width(closes: list, period: int = 20, std_dev: float = 2.0) -> float:
    """
    Bollinger Band Width — normalized measure of volatility.

    Width = (upper - lower) / middle

    Low width → low volatility → potential breakout brewing
    High width → high volatility → trend in motion

    Returns normalized width, or 0.0 if insufficient data.
    """
    upper, mid, lower = bollinger_bands(closes, period, std_dev)
    return safe_div(upper - lower, mid)


# ── ATR — Average True Range ──────────────────────────────────────────────────

def atr(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """
    Average True Range — measures market volatility in price terms.

    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = exponential average of True Range over period

    Interpretation:
        ATR tells you the typical daily move in dollar terms.
        Use it to set stop-losses and price targets:
          Price target = current_price ± (N * ATR)
        Common N: 1.5× for conservative, 2.0× for normal, 3.0× for aggressive

    As a feature: atr_pct = ATR / current_price
        High atr_pct + high volume ratio → large expected move

    Parameters (tune in config.INDICATOR_SETTINGS):
        period : Smoothing period (default: 14)

    Returns:
        ATR in price units, or 0.0 if insufficient data.
    """
    if len(closes) < period + 1:
        return 0.0

    trs = []
    for i in range(1, len(closes)):
        h, l, pc = highs[i], lows[i], closes[i-1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)

    if len(trs) < period:
        return float(np.mean(trs)) if trs else 0.0

    # Wilder's smoothing (same as Wilder's RSI)
    result = np.mean(trs[:period])
    for tr in trs[period:]:
        result = (result * (period - 1) + tr) / period
    return result


def atr_pct(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """
    ATR as percentage of current price.

    More comparable across stocks than raw ATR.
    AAPL at $180 and NVDA at $900 have very different raw ATRs,
    but atr_pct gives a normalized volatility measure.

    Returns ATR as % of price (e.g. 0.025 = 2.5% daily range is typical).
    """
    a = atr(highs, lows, closes, period)
    price = closes[-1]
    return safe_div(a, price)


# ── ADX — Average Directional Index ──────────────────────────────────────────

def adx(highs: list, lows: list, closes: list, period: int = 14) -> tuple[float, float, float]:
    """
    Average Directional Index — measures trend strength (not direction).

    Returns (adx, plus_di, minus_di)

    ADX interpretation:
        > 50   : Extremely strong trend
        25–50  : Strong trend (breakouts reliable)
        20–25  : Weak trend or transition
        < 20   : No trend / ranging market (breakouts less reliable)

    +DI vs -DI:
        +DI > -DI → bullish trend
        -DI > +DI → bearish trend
        Crossover → trend change signal

    In volume anomaly context:
        High volume + ADX > 25 + price moving = breakout is reliable
        High volume + ADX < 20 = might be a false breakout in ranging market

    Parameters (tune in config.INDICATOR_SETTINGS):
        period : Smoothing period (default: 14)

    Returns (adx, plus_di, minus_di) or (0, 0, 0) if insufficient data.
    """
    if len(closes) < period * 2:
        return 0.0, 0.0, 0.0

    plus_dm_list, minus_dm_list, tr_list = [], [], []
    for i in range(1, len(closes)):
        up   = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        plus_dm_list.append(up if up > down and up > 0 else 0)
        minus_dm_list.append(down if down > up and down > 0 else 0)
        h, l, pc = highs[i], lows[i], closes[i-1]
        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))

    def wilder_smooth(data, p):
        result = [sum(data[:p])]
        for v in data[p:]:
            result.append(result[-1] - result[-1]/p + v)
        return result

    tr_s    = wilder_smooth(tr_list, period)
    pdm_s   = wilder_smooth(plus_dm_list, period)
    mdm_s   = wilder_smooth(minus_dm_list, period)

    pdi = [safe_div(p, t) * 100 for p, t in zip(pdm_s, tr_s)]
    mdi = [safe_div(m, t) * 100 for m, t in zip(mdm_s, tr_s)]
    dx  = [safe_div(abs(p - m), p + m) * 100 for p, m in zip(pdi, mdi)]

    if len(dx) < period:
        return 0.0, pdi[-1] if pdi else 0.0, mdi[-1] if mdi else 0.0

    adx_val = wilder_smooth(dx, period)
    return adx_val[-1], pdi[-1], mdi[-1]


# ── SuperTrend ────────────────────────────────────────────────────────────────

def supertrend(highs: list, lows: list, closes: list,
               period: int = 10, multiplier: float = 3.0) -> tuple[float, int]:
    """
    SuperTrend — ATR-based trend following indicator.

    Returns (supertrend_level, direction)
    direction: +1 = bullish (price above supertrend), -1 = bearish

    Interpretation:
        Direction flip from -1 to +1 → bullish signal (buy)
        Direction flip from +1 to -1 → bearish signal (sell)
        High volume + direction flip → strong trend initiation signal

    As an ML feature: use the direction (+1/-1) as a binary feature,
    and distance from supertrend level as a continuous feature.

    Parameters (tune in config.INDICATOR_SETTINGS):
        period     : ATR period (default: 10)
        multiplier : ATR multiplier for band width (default: 3.0)
                     Smaller = tighter, more signals (more noise)
                     Larger  = wider, fewer signals (less noise)

    Returns (level, direction) or (0.0, 0) if insufficient data.
    """
    if len(closes) < period + 1:
        return 0.0, 0

    atr_val = atr(highs, lows, closes, period)
    mid = (highs[-1] + lows[-1]) / 2

    upper_basic = mid + multiplier * atr_val
    lower_basic = mid - multiplier * atr_val

    # Simplified: use current bar only (full implementation needs full history)
    if closes[-1] >= lower_basic:
        return lower_basic, 1   # bullish
    else:
        return upper_basic, -1  # bearish
