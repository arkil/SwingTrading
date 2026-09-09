"""
indicators/composite.py — Feature builder combining all indicators

This module pulls together ALL indicator values into a single flat dict
that the ML model can consume as a feature vector.

To add a new indicator to the feature set:
  1. Import your function at the top
  2. Add a call in build_feature_dict() and store the result
  3. Add the key to models/features.py FEATURE_COLUMNS list
  4. Retrain: python main.py --mode retrain
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from core.fetcher import CandleData, QuoteData
from core.timeframes import VolumeRatio
from indicators.volume import (
    volume_zscore, vwap, obv_slope, mfi, cmf, ad_slope,
    vol_ma_ratio, volume_acceleration
)
from indicators.momentum import (
    rsi, macd_histogram, stochastic, roc, volume_roc, cci, williams_r
)
from indicators.trend import (
    sma, ema, price_vs_sma, ema_cross, bb_position, bb_width,
    atr_pct, adx, supertrend
)

logger = logging.getLogger(__name__)


@dataclass
class StockSignal:
    """
    Complete signal snapshot for one stock at one point in time.

    This is the core data structure passed between modules.

    Fields:
        symbol          : Ticker
        quote           : Real-time quote data
        tf_ratios       : Dict of tf_id → VolumeRatio
        daily_candles   : Daily OHLCV candles (for trend indicators)
        intraday_candles: Dict of tf_id → CandleData (for intraday indicators)
        features        : Flat dict of all computed feature values
        prediction      : Model output (set later by predictor.py)
        timestamp       : Unix timestamp when signal was captured
    """
    symbol:           str
    quote:            Optional[QuoteData]           = None
    tf_ratios:        dict[str, VolumeRatio]        = field(default_factory=dict)
    daily_candles:    Optional[CandleData]          = None
    intraday_candles: dict[str, CandleData]         = field(default_factory=dict)
    features:         dict[str, float]              = field(default_factory=dict)
    prediction:       Optional[dict]                = None
    timestamp:        int                           = 0

    @property
    def max_ratio(self) -> float:
        """Highest volume ratio across all timeframes."""
        ratios = [r.ratio for r in self.tf_ratios.values() if r.ok and r.ratio > 0]
        return max(ratios) if ratios else 0.0

    @property
    def is_flagged(self) -> bool:
        return self.max_ratio >= 1.8

    @property
    def best_tf(self) -> Optional[VolumeRatio]:
        """Timeframe with the highest ratio."""
        valid = [r for r in self.tf_ratios.values() if r.ok]
        return max(valid, key=lambda r: r.ratio) if valid else None


def build_feature_dict(signal: "StockSignal", cfg: dict) -> dict[str, float]:
    """
    Compute all indicator values from raw OHLCV data and return as feature dict.

    This function is the single place where all features are assembled.
    Every key added here becomes a potential model feature.

    Parameters:
        signal : StockSignal with candle data populated
        cfg    : config.INDICATOR_SETTINGS dict

    Returns:
        Dict mapping feature_name → float value.
        Missing values default to 0.0 (never None or NaN in this dict).

    ── HOW TO ADD A NEW FEATURE ──────────────────────────────────
    1. Import your function at the top of this file
    2. Call it here and add the result to `features`
    3. Add the key string to models/features.py FEATURE_COLUMNS
    4. Run: python main.py --mode retrain
    ─────────────────────────────────────────────────────────────
    """
    features: dict[str, float] = {}
    ind = cfg  # shorthand

    # ── Helper: safe extract from candles ─────────────────────────────────────
    def get_ohlcv(candles: Optional[CandleData]):
        if not candles or not candles.ok or len(candles) < 3:
            return [], [], [], [], []
        return candles.opens, candles.highs, candles.lows, candles.closes, candles.volumes

    daily = signal.daily_candles
    o_d, h_d, l_d, c_d, v_d = get_ohlcv(daily)

    intra_1m  = signal.intraday_candles.get("1m")
    intra_15m = signal.intraday_candles.get("15m")
    intra_3h  = signal.intraday_candles.get("3h")
    _, h_1m, l_1m, c_1m, v_1m     = get_ohlcv(intra_1m)
    _, h_15m, l_15m, c_15m, v_15m = get_ohlcv(intra_15m)
    _, h_3h, l_3h, c_3h, v_3h     = get_ohlcv(intra_3h)

    # ── 1. Volume ratio features (the core signal) ─────────────────────────────
    for tf_id, vr in signal.tf_ratios.items():
        features[f"vol_ratio_{tf_id}"] = vr.ratio if vr.ok else 0.0

    # ── 2. Volume Z-score ──────────────────────────────────────────────────────
    features["vol_zscore_daily"] = volume_zscore(
        v_d, window=ind.get("volume_zscore_window", 20)
    ) if v_d else 0.0

    features["vol_zscore_15m"] = volume_zscore(
        v_15m, window=10
    ) if v_15m else 0.0

    # ── 3. Volume Moving Average Ratio (daily) ─────────────────────────────────
    features["vol_ma_ratio"] = vol_ma_ratio(
        v_d, period=ind.get("vol_ma_period", 20)
    ) if v_d else 0.0

    # ── 4. OBV Slope ───────────────────────────────────────────────────────────
    features["obv_slope_daily"] = obv_slope(
        c_d, v_d, bars=ind.get("obv_slope_bars", 5)
    ) if (c_d and v_d) else 0.0

    features["obv_slope_15m"] = obv_slope(
        c_15m, v_15m, bars=5
    ) if (c_15m and v_15m) else 0.0

    # ── 5. Money Flow Index ────────────────────────────────────────────────────
    features["mfi_daily"] = mfi(
        h_d, l_d, c_d, v_d, period=ind.get("mfi_period", 14)
    ) if (h_d and v_d) else 50.0

    features["mfi_15m"] = mfi(
        h_15m, l_15m, c_15m, v_15m, period=ind.get("mfi_period", 14)
    ) if (h_15m and v_15m) else 50.0

    # ── 6. Chaikin Money Flow ──────────────────────────────────────────────────
    features["cmf_daily"] = cmf(
        h_d, l_d, c_d, v_d, period=ind.get("cmf_period", 20)
    ) if (h_d and v_d) else 0.0

    # ── 7. AD Line Slope ──────────────────────────────────────────────────────
    features["ad_slope_daily"] = ad_slope(
        h_d, l_d, c_d, v_d, bars=5
    ) if (h_d and v_d) else 0.0

    # ── 8. Volume Acceleration ─────────────────────────────────────────────────
    features["vol_acceleration"] = volume_acceleration(
        v_d, bars=3
    ) if v_d else 0.0

    # ── 9. VWAP distance (intraday) ────────────────────────────────────────────
    if h_1m and v_1m:
        vwap_val = vwap(h_1m, l_1m, c_1m, v_1m)
        current_price = c_1m[-1] if c_1m else 0.0
        features["price_vs_vwap_pct"] = (
            (current_price - vwap_val) / vwap_val * 100
        ) if vwap_val else 0.0
    else:
        features["price_vs_vwap_pct"] = 0.0

    # ── 10. RSI ────────────────────────────────────────────────────────────────
    features["rsi_daily"] = rsi(
        c_d, period=ind.get("rsi_period", 14)
    ) if c_d else 50.0

    features["rsi_15m"] = rsi(
        c_15m, period=ind.get("rsi_period", 14)
    ) if c_15m else 50.0

    # ── 11. MACD Histogram ─────────────────────────────────────────────────────
    features["macd_hist_daily"] = macd_histogram(
        c_d,
        fast=ind.get("macd_fast", 12),
        slow=ind.get("macd_slow", 26),
        signal=ind.get("macd_signal", 9),
    ) if c_d else 0.0

    # ── 12. Stochastic ─────────────────────────────────────────────────────────
    if h_d and l_d and c_d:
        stoch_k, stoch_d = stochastic(
            h_d, l_d, c_d,
            k_period=ind.get("stoch_k", 14),
            d_period=ind.get("stoch_d", 3),
        )
        features["stoch_k"] = stoch_k
        features["stoch_d"] = stoch_d
    else:
        features["stoch_k"] = 50.0
        features["stoch_d"] = 50.0

    # ── 13. Rate of Change ─────────────────────────────────────────────────────
    features["roc_daily"] = roc(c_d, period=ind.get("roc_period", 10)) if c_d else 0.0
    features["vol_roc"]   = volume_roc(v_d, period=ind.get("roc_period", 10)) if v_d else 0.0

    # ── 14. CCI ────────────────────────────────────────────────────────────────
    features["cci_daily"] = cci(
        h_d, l_d, c_d, period=ind.get("cci_period", 20)
    ) if (h_d and c_d) else 0.0

    # ── 15. Williams %R ────────────────────────────────────────────────────────
    features["williams_r"] = williams_r(
        h_d, l_d, c_d, period=ind.get("williams_r_period", 14)
    ) if (h_d and c_d) else -50.0

    # ── 16. Price vs SMA ───────────────────────────────────────────────────────
    for p in ind.get("sma_periods", [20, 50, 200]):
        features[f"price_vs_sma{p}"] = price_vs_sma(c_d, p) if c_d else 0.0

    # ── 17. EMA Cross ─────────────────────────────────────────────────────────
    features["ema_cross_9_21"] = ema_cross(c_d, 9, 21) if c_d else 0.0
    features["ema_cross_21_55"] = ema_cross(c_d, 21, 55) if c_d else 0.0

    # ── 18. Bollinger Bands ────────────────────────────────────────────────────
    features["bb_position"] = bb_position(
        c_d,
        period=ind.get("bb_period", 20),
        std_dev=ind.get("bb_std", 2.0),
    ) if c_d else 0.5

    features["bb_width"] = bb_width(
        c_d,
        period=ind.get("bb_period", 20),
        std_dev=ind.get("bb_std", 2.0),
    ) if c_d else 0.0

    # ── 19. ATR % ─────────────────────────────────────────────────────────────
    features["atr_pct"] = atr_pct(
        h_d, l_d, c_d, period=ind.get("atr_period", 14)
    ) if (h_d and c_d) else 0.0

    # ── 20. ADX ────────────────────────────────────────────────────────────────
    if h_d and l_d and c_d:
        adx_val, plus_di, minus_di = adx(h_d, l_d, c_d, period=ind.get("adx_period", 14))
        features["adx"]      = adx_val
        features["plus_di"]  = plus_di
        features["minus_di"] = minus_di
        features["di_diff"]  = plus_di - minus_di   # positive = bullish
    else:
        features["adx"] = 0.0
        features["plus_di"] = 0.0
        features["minus_di"] = 0.0
        features["di_diff"] = 0.0

    # ── 21. SuperTrend direction ───────────────────────────────────────────────
    if h_d and l_d and c_d:
        st_level, st_dir = supertrend(
            h_d, l_d, c_d,
            period=ind.get("supertrend_period", 10),
            multiplier=ind.get("supertrend_multiplier", 3.0),
        )
        features["supertrend_dir"]      = float(st_dir)   # +1 or -1
        features["price_vs_supertrend"] = (
            (c_d[-1] - st_level) / st_level * 100
        ) if st_level else 0.0
    else:
        features["supertrend_dir"] = 0.0
        features["price_vs_supertrend"] = 0.0

    # ── 22. Quote-derived features ─────────────────────────────────────────────
    if signal.quote and signal.quote.ok:
        q = signal.quote
        features["day_change_pct"] = q.change_pct
        features["gap_open_pct"]   = (
            (q.open - q.prev_close) / q.prev_close * 100
        ) if q.prev_close else 0.0
        features["day_range_pct"]  = (
            (q.high - q.low) / q.prev_close * 100
        ) if q.prev_close else 0.0
        # Price position within day range (0 = at low, 1 = at high)
        day_range = q.high - q.low
        features["price_day_position"] = (
            (q.price - q.low) / day_range
        ) if day_range > 0 else 0.5
    else:
        features["day_change_pct"]    = 0.0
        features["gap_open_pct"]      = 0.0
        features["day_range_pct"]     = 0.0
        features["price_day_position"] = 0.5

    # ── 23. Time-of-day context ────────────────────────────────────────────────
    import datetime
    now = datetime.datetime.now()
    # Normalize to 0-1 within market hours (9:30 AM - 4:00 PM ET)
    market_open_min  = 9 * 60 + 30
    market_close_min = 16 * 60
    current_min = now.hour * 60 + now.minute
    features["session_progress"] = max(0.0, min(1.0,
        (current_min - market_open_min) / (market_close_min - market_open_min)
    ))
    features["hour_of_day"] = float(now.hour)
    features["day_of_week"] = float(now.weekday())  # 0 = Monday, 4 = Friday

    # ── Replace any NaN or inf with 0.0 ───────────────────────────────────────
    import math
    for k, v in features.items():
        if v is None or math.isnan(v) or math.isinf(v):
            features[k] = 0.0

    return features
