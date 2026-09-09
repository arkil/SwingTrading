"""
indicators/momentum.py — Price momentum indicators

All functions return the latest scalar reading for use as ML features.

Momentum indicators answer: "Is price moving with or against volume?"
Combining high volume ratios with momentum confirmation greatly improves
signal quality:
  - High volume + rising RSI    → strong bullish signal
  - High volume + falling RSI   → potential distribution
  - High volume + MACD cross    → trend confirmation
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def safe_div(a, b, default=0.0):
    return a / b if b and b != 0 else default


# ── RSI — Relative Strength Index ─────────────────────────────────────────────

def rsi(closes: list, period: int = 14) -> float:
    """
    Relative Strength Index — measures speed and magnitude of price changes.

    RSI = 100 - (100 / (1 + avg_gain / avg_loss))

    Interpretation:
        > 70   : Overbought (consider bearish signal)
        50–70  : Bullish momentum
        30–50  : Bearish momentum
        < 30   : Oversold (consider bullish signal)
        50 line: Trend confirmation (above = bullish, below = bearish)

    In the context of volume anomalies:
        High volume + RSI > 70 → exhaustion top possible
        High volume + RSI < 30 → capitulation / reversal likely
        High volume + RSI crossing 50 → strong trend initiation

    Parameters (tune in config.INDICATOR_SETTINGS):
        period : Lookback period (default: 14). Shorter = more sensitive.
                 Common alternatives: 9 (day traders), 21 (swing), 25 (weekly)

    Returns:
        RSI value 0–100. Returns 50.0 if insufficient data.
    """
    if len(closes) < period + 1:
        return 50.0

    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas[-period:]]
    losses = [abs(min(d, 0)) for d in deltas[-period:]]

    avg_gain = np.mean(gains) if gains else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    if avg_loss == 0:
        return 100.0
    rs = safe_div(avg_gain, avg_loss)
    return 100 - (100 / (1 + rs))


# ── MACD ──────────────────────────────────────────────────────────────────────

def macd(closes: list, fast: int = 12, slow: int = 26,
         signal: int = 9) -> tuple[float, float, float]:
    """
    MACD — Moving Average Convergence Divergence.

    Returns (macd_line, signal_line, histogram)

    MACD Line   = EMA(fast) - EMA(slow)
    Signal Line = EMA(macd_line, signal)
    Histogram   = MACD Line - Signal Line

    Interpretation:
        Histogram > 0 and rising  → bullish momentum building
        Histogram > 0 and falling → bullish momentum fading
        Histogram < 0 and falling → bearish momentum building
        Histogram crossover zero  → trend change signal
        MACD line crosses signal  → entry/exit signal

    In the context of volume anomalies:
        High volume + positive and rising MACD histogram → strongest buy signal
        High volume + negative and falling histogram     → strong sell signal

    Parameters (tune in config.INDICATOR_SETTINGS):
        fast   : Fast EMA period (default: 12)
        slow   : Slow EMA period (default: 26)
        signal : Signal line EMA period (default: 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram). All 0.0 if insufficient data.
    """
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0

    def ema(data, period):
        k = 2 / (period + 1)
        result = [data[0]]
        for v in data[1:]:
            result.append(v * k + result[-1] * (1 - k))
        return result

    closes_arr = list(closes)
    ema_fast = ema(closes_arr, fast)
    ema_slow = ema(closes_arr, slow)
    macd_line = [f - s for f, s in zip(ema_fast[slow-fast:], ema_slow)]
    if len(macd_line) < signal:
        return 0.0, 0.0, 0.0
    signal_line = ema(macd_line, signal)
    histogram   = macd_line[-1] - signal_line[-1]
    return macd_line[-1], signal_line[-1], histogram


def macd_histogram(closes: list, fast: int = 12, slow: int = 26,
                   signal: int = 9) -> float:
    """Convenience function: returns just the MACD histogram value."""
    _, _, hist = macd(closes, fast, slow, signal)
    return hist


# ── Stochastic Oscillator ─────────────────────────────────────────────────────

def stochastic(highs: list, lows: list, closes: list,
               k_period: int = 14, d_period: int = 3) -> tuple[float, float]:
    """
    Stochastic Oscillator — shows where price is relative to recent range.

    %K = (current_close - lowest_low) / (highest_high - lowest_low) * 100
    %D = SMA(%K, d_period)

    Interpretation:
        > 80   : Overbought
        < 20   : Oversold
        %K crosses above %D in oversold zone → bullish signal
        %K crosses below %D in overbought zone → bearish signal

    In the context of volume anomalies:
        High volume + stochastic < 20 → strong oversold reversal signal
        High volume + stochastic > 80 → possible exhaustion

    Parameters (tune in config.INDICATOR_SETTINGS):
        k_period : %K lookback (default: 14)
        d_period : %D smoothing (default: 3)

    Returns:
        Tuple of (%K, %D). Returns (50.0, 50.0) if insufficient data.
    """
    if len(closes) < k_period + d_period:
        return 50.0, 50.0

    k_values = []
    for i in range(k_period, len(closes) + 1):
        window_h = max(highs[i-k_period:i])
        window_l = min(lows[i-k_period:i])
        hl_range = window_h - window_l
        k = safe_div((closes[i-1] - window_l), hl_range, 50.0) * 100
        k_values.append(k)

    if len(k_values) < d_period:
        return k_values[-1], k_values[-1]

    k = k_values[-1]
    d = np.mean(k_values[-d_period:])
    return k, d


# ── Rate of Change ────────────────────────────────────────────────────────────

def roc(closes: list, period: int = 10) -> float:
    """
    Rate of Change — percentage price change over N periods.

    ROC = (current_close - close_N_periods_ago) / close_N_periods_ago * 100

    Interpretation:
        > 0   : Price higher than N periods ago (bullish)
        < 0   : Price lower than N periods ago (bearish)
        Extreme readings suggest overbought/oversold

    Use cases with volume:
        High volume + strong positive ROC → trend breakout
        High volume + strong negative ROC → breakdown

    Parameters (tune in config.INDICATOR_SETTINGS):
        period : Lookback period (default: 10)

    Returns:
        Percentage rate of change. Returns 0.0 if insufficient data.
    """
    if len(closes) < period + 1:
        return 0.0
    prior = closes[-period-1]
    return safe_div((closes[-1] - prior), prior) * 100


def volume_roc(volumes: list, period: int = 10) -> float:
    """
    Rate of Change applied to volume instead of price.

    High positive volume ROC alongside price ROC confirms the move.
    Divergence (price ROC up, volume ROC down) suggests weakening momentum.
    """
    if len(volumes) < period + 1:
        return 0.0
    prior = volumes[-period-1]
    return safe_div((volumes[-1] - prior), prior) * 100


# ── CCI — Commodity Channel Index ────────────────────────────────────────────

def cci(highs: list, lows: list, closes: list, period: int = 20) -> float:
    """
    Commodity Channel Index — measures deviation from statistical mean.

    CCI = (typical_price - SMA_typical) / (0.015 * mean_deviation)
    typical_price = (high + low + close) / 3

    Interpretation:
        > +100  : Overbought (strong uptrend)
        -100 to +100: Normal range
        < -100  : Oversold (strong downtrend)
        ±200    : Extreme readings, high reversal probability

    Parameters (tune in config.INDICATOR_SETTINGS):
        period : Lookback period (default: 20)

    Returns:
        CCI value. Returns 0.0 if insufficient data.
    """
    if len(closes) < period:
        return 0.0
    typical = [(h + l + c) / 3 for h, l, c in zip(highs[-period:], lows[-period:], closes[-period:])]
    sma_tp  = np.mean(typical)
    mean_dev = np.mean([abs(t - sma_tp) for t in typical])
    return safe_div(typical[-1] - sma_tp, 0.015 * mean_dev)


# ── Williams %R ───────────────────────────────────────────────────────────────

def williams_r(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """
    Williams %R — shows closing price relative to high-low range.

    %R = (highest_high - current_close) / (highest_high - lowest_low) * -100

    Interpretation:
        -20 to 0    : Overbought
        -80 to -100 : Oversold
        Crossings of -50 line signal trend changes

    Note: Williams %R is essentially inverse Stochastic %K.

    Parameters (tune in config.INDICATOR_SETTINGS):
        period : Lookback period (default: 14)

    Returns:
        %R value (always negative, from 0 to -100). Returns -50.0 if insufficient.
    """
    if len(closes) < period:
        return -50.0
    hh = max(highs[-period:])
    ll = min(lows[-period:])
    hl = hh - ll
    return safe_div(-(hh - closes[-1]), hl) * 100
