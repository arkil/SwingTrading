"""
models/trainer.py — Fetch historical data and train the prediction model

Training process:
  1. For each symbol, fetch N days of daily + intraday candles
  2. For each historical bar, compute all indicator features (same as live)
  3. Compute the "forward return" (did price go up/down N hours later?)
  4. Label each sample as UP / DOWN / NEUTRAL
  5. Train the classifier on all labeled samples
  6. Save model to output/model.pkl

Run with:
    python main.py --mode retrain
    python main.py --mode retrain --symbol AAPL   (single symbol)
"""

import time
import logging
from typing import Optional

import numpy as np

from core.fetcher import FinnhubFetcher, CandleData
from indicators import volume as vol_ind
from indicators import momentum as mom_ind
from indicators import trend as trend_ind
from models.predictor import FEATURE_COLUMNS, UP_THRESHOLD, DOWN_THRESHOLD

logger = logging.getLogger(__name__)


def build_training_features(
    daily_candles: CandleData,
    i: int,
    cfg: dict,
) -> Optional[dict]:
    """
    Build a feature dict for training using data available up to bar index `i`.

    This simulates what the live system would see at time `i` —
    it only uses candles[0:i], never future data.

    Parameters:
        daily_candles : Full daily candle history
        i             : Current bar index (we use [0:i] for features)
        cfg           : config.INDICATOR_SETTINGS

    Returns:
        Feature dict or None if insufficient data.
    """
    if i < 30:  # need at least 30 bars for most indicators
        return None

    c = daily_candles.closes[:i]
    h = daily_candles.highs[:i]
    l = daily_candles.lows[:i]
    v = daily_candles.volumes[:i]

    if not c or len(c) < 20:
        return None

    ind = cfg
    features = {}

    # Volume ratios (daily only for historical training)
    # We can't go back and get historical intraday candles easily on free tier
    # so we estimate intraday ratios from daily data
    n = len(v)
    if n >= 3:
        features["vol_ratio_1d"]  = v[-1] / np.mean(v[-21:-1]) if np.mean(v[-21:-1]) > 0 else 1.0
        features["vol_ratio_3h"]  = v[-1] / v[-2] if v[-2] > 0 else 1.0   # proxy
        features["vol_ratio_15m"] = v[-1] / v[-2] if v[-2] > 0 else 1.0   # proxy
        features["vol_ratio_1m"]  = v[-1] / v[-2] if v[-2] > 0 else 1.0   # proxy
    else:
        for tf in ["1d", "3h", "15m", "1m"]:
            features[f"vol_ratio_{tf}"] = 1.0

    features["vol_zscore_daily"] = vol_ind.volume_zscore(v, window=ind.get("volume_zscore_window", 20))
    features["vol_zscore_15m"]   = features["vol_zscore_daily"]  # proxy
    features["vol_ma_ratio"]     = vol_ind.vol_ma_ratio(v, period=ind.get("vol_ma_period", 20))
    features["obv_slope_daily"]  = vol_ind.obv_slope(c, v, bars=ind.get("obv_slope_bars", 5))
    features["obv_slope_15m"]    = features["obv_slope_daily"]
    features["mfi_daily"]        = vol_ind.mfi(h, l, c, v, period=ind.get("mfi_period", 14))
    features["mfi_15m"]          = features["mfi_daily"]
    features["cmf_daily"]        = vol_ind.cmf(h, l, c, v, period=ind.get("cmf_period", 20))
    features["ad_slope_daily"]   = vol_ind.ad_slope(h, l, c, v, bars=5)
    features["vol_acceleration"] = vol_ind.volume_acceleration(v, bars=3)
    features["price_vs_vwap_pct"]= 0.0  # not available in daily training data

    features["rsi_daily"]        = mom_ind.rsi(c, period=ind.get("rsi_period", 14))
    features["rsi_15m"]          = features["rsi_daily"]
    features["macd_hist_daily"]  = mom_ind.macd_histogram(c, ind.get("macd_fast",12), ind.get("macd_slow",26), ind.get("macd_signal",9))
    stk, std = mom_ind.stochastic(h, l, c, ind.get("stoch_k",14), ind.get("stoch_d",3))
    features["stoch_k"]          = stk
    features["stoch_d"]          = std
    features["roc_daily"]        = mom_ind.roc(c, period=ind.get("roc_period",10))
    features["vol_roc"]          = mom_ind.volume_roc(v, period=ind.get("roc_period",10))
    features["cci_daily"]        = mom_ind.cci(h, l, c, period=ind.get("cci_period",20))
    features["williams_r"]       = mom_ind.williams_r(h, l, c, period=ind.get("williams_r_period",14))

    for p in ind.get("sma_periods", [20, 50, 200]):
        features[f"price_vs_sma{p}"] = trend_ind.price_vs_sma(c, p)

    features["ema_cross_9_21"]   = trend_ind.ema_cross(c, 9, 21)
    features["ema_cross_21_55"]  = trend_ind.ema_cross(c, 21, 55)
    features["bb_position"]      = trend_ind.bb_position(c, ind.get("bb_period",20), ind.get("bb_std",2.0))
    features["bb_width"]         = trend_ind.bb_width(c, ind.get("bb_period",20), ind.get("bb_std",2.0))
    features["atr_pct"]          = trend_ind.atr_pct(h, l, c, ind.get("atr_period",14))

    adx_val, plus_di, minus_di = trend_ind.adx(h, l, c, ind.get("adx_period",14))
    features["adx"]       = adx_val
    features["plus_di"]   = plus_di
    features["minus_di"]  = minus_di
    features["di_diff"]   = plus_di - minus_di

    st_level, st_dir = trend_ind.supertrend(h, l, c, ind.get("supertrend_period",10), ind.get("supertrend_multiplier",3.0))
    features["supertrend_dir"]       = float(st_dir)
    features["price_vs_supertrend"]  = (c[-1] - st_level) / st_level * 100 if st_level else 0.0

    # Quote-derived features from daily data
    features["day_change_pct"]     = (c[-1] - c[-2]) / c[-2] * 100 if c[-2] else 0.0
    features["gap_open_pct"]       = 0.0
    features["day_range_pct"]      = (h[-1] - l[-1]) / c[-2] * 100 if c[-2] else 0.0
    features["price_day_position"] = (c[-1] - l[-1]) / (h[-1] - l[-1]) if (h[-1] - l[-1]) > 0 else 0.5

    # Time features — use index as proxy (can't know exact time in historical)
    features["session_progress"] = 0.5
    features["hour_of_day"]      = 12.0
    features["day_of_week"]      = float(i % 5)

    # Clean NaN/inf
    import math
    for k in list(features.keys()):
        if features[k] is None or math.isnan(features[k]) or math.isinf(features[k]):
            features[k] = 0.0

    return features


def compute_label(closes: list, i: int, forward_days: int = 1,
                  up_thr: float = UP_THRESHOLD,
                  down_thr: float = DOWN_THRESHOLD) -> Optional[str]:
    """
    Compute the training label for bar i.

    Label is based on the price change `forward_days` bars after bar i.

    Parameters:
        closes      : List of closing prices
        i           : Current bar index
        forward_days: How many bars ahead to look for the label
        up_thr      : Minimum return to label as UP
        down_thr    : Minimum return to label as DOWN

    Returns:
        "UP", "DOWN", or "NEUTRAL", or None if not enough future data.
    """
    if i + forward_days >= len(closes):
        return None  # no future data available
    current  = closes[i]
    future   = closes[i + forward_days]
    if current == 0:
        return None
    ret = (future - current) / current
    if ret > up_thr:
        return "UP"
    elif ret < -down_thr:
        return "DOWN"
    else:
        return "NEUTRAL"


def collect_training_samples(
    symbols: list[str],
    fetcher: FinnhubFetcher,
    cfg: dict,
    training_days: int = 90,
) -> list[tuple[dict, str]]:
    """
    Fetch historical data for all symbols and build training samples.

    For each symbol:
      1. Fetch `training_days` of daily candles
      2. For each bar (with sufficient history), compute features
      3. Compute the forward-return label
      4. Collect as (features, label) pair

    Parameters:
        symbols      : List of ticker symbols
        fetcher      : Finnhub API client
        cfg          : Full config dict
        training_days: Days of history to use

    Returns:
        List of (feature_dict, label) tuples for training.
    """
    samples = []
    now = int(time.time())
    from_ts = now - 86400 * int(training_days * 1.5)  # buffer for weekends

    for idx, symbol in enumerate(symbols):
        logger.info(f"Collecting training data for {symbol} ({idx+1}/{len(symbols)})")
        try:
            candles = fetcher.get_candles(symbol, "D", from_ts, now)
            if not candles.ok or len(candles) < 35:
                logger.debug(f"Skipping {symbol}: insufficient history ({len(candles)} bars)")
                continue

            ind_cfg = cfg.get("INDICATOR_SETTINGS", {})
            forward = max(1, cfg.get("MODEL_FORWARD_HOURS", 4) // 6)  # approx daily bars

            for i in range(30, len(candles) - forward):
                features = build_training_features(candles, i, ind_cfg)
                if features is None:
                    continue
                label = compute_label(candles.closes, i, forward_days=forward)
                if label is None:
                    continue
                samples.append((features, label))

        except Exception as e:
            logger.warning(f"Error collecting data for {symbol}: {e}")
            continue

        # Brief pause to respect rate limits
        time.sleep(0.5)

    logger.info(f"Collected {len(samples)} training samples from {len(symbols)} symbols")
    return samples
