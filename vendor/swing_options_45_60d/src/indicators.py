"""
Indicator engine for the swing_options_45_60d strategy.

Public API
----------
    compute_all_indicators(df, params) -> pd.DataFrame

All calculations only use data up to and including the current bar (no
lookahead). Warmup rows (before the longest lookback window, e.g. EMA200)
will contain NaN and should be dropped/ignored by callers requiring
`len(df) >= 220` rows before calling this function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Building blocks ────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)


def _bollinger(close: pd.Series, period: int, num_std: float):
    mid = _sma(close, period)
    std = close.rolling(period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, width


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int):
    lowest_low = low.rolling(k_period, min_periods=k_period).min()
    highest_high = high.rolling(k_period, min_periods=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k.fillna(50.0), d.fillna(50.0)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).fillna(0).cumsum()


def _roc(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(period) * 100


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0):
    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    trend = pd.Series(1, index=close.index)  # 1 = up, -1 = down

    for i in range(1, len(close)):
        if pd.isna(atr.iloc[i]):
            continue
        # carry forward bands unless price breaks through
        if close.iloc[i - 1] <= final_upper.iloc[i - 1]:
            final_upper.iloc[i] = min(upper_band.iloc[i], final_upper.iloc[i - 1])
        else:
            final_upper.iloc[i] = upper_band.iloc[i]

        if close.iloc[i - 1] >= final_lower.iloc[i - 1]:
            final_lower.iloc[i] = max(lower_band.iloc[i], final_lower.iloc[i - 1])
        else:
            final_lower.iloc[i] = lower_band.iloc[i]

        if trend.iloc[i - 1] == 1 and close.iloc[i] < final_lower.iloc[i]:
            trend.iloc[i] = -1
        elif trend.iloc[i - 1] == -1 and close.iloc[i] > final_upper.iloc[i]:
            trend.iloc[i] = 1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    return trend


# ── Public API ─────────────────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Compute the full indicator set + boolean signal columns required by
    src/screener.py::screen_universe and Scripts/SwingTrading/swing_options_screener.py.

    Args:
        df: OHLCV DataFrame with columns Open, High, Low, Close, Volume
        params: strategy params dict (see swing_options_screener.DEFAULT_PARAMS)

    Returns:
        DataFrame (same index as df) with added indicator + signal columns:
        ema_fast, ema_medium, ema_slow, ema_trend, sma_50, sma_200, rsi,
        macd, macd_signal, macd_hist, bb_upper, bb_mid, bb_lower, bb_width,
        atr, adx, stoch_k, stoch_d, obv, roc, hist_vol, supertrend,
        ema_aligned, trend_200, macd_ok, vol_surge, supertrend_ok,
        bb_expansion, stoch_ok, obv_ok
    """
    out = df.copy()
    close, high, low, volume = out["Close"], out["High"], out["Low"], out["Volume"]

    out["ema_fast"] = _ema(close, params["ema_fast"])
    out["ema_medium"] = _ema(close, params["ema_medium"])
    out["ema_slow"] = _ema(close, params["ema_slow"])
    out["ema_trend"] = _ema(close, params["ema_trend"])
    out["sma_50"] = _sma(close, params["sma_50"])
    out["sma_200"] = _sma(close, params["sma_200"])

    out["rsi"] = _rsi(close, params["rsi_period"])

    macd_line, macd_signal, macd_hist = _macd(
        close, params["macd_fast"], params["macd_slow"], params["macd_signal"]
    )
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    bb_upper, bb_mid, bb_lower, bb_width = _bollinger(close, params["bb_period"], params["bb_std"])
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    out["bb_width"] = bb_width

    out["atr"] = _atr(high, low, close, params["atr_period"])
    out["adx"] = _adx(high, low, close, params["adx_period"])

    stoch_k, stoch_d = _stochastic(high, low, close, params["stoch_k"], params["stoch_d"])
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d

    out["obv"] = _obv(close, volume)
    out["roc"] = _roc(close, params["roc_period"])

    # Annualized historical volatility of log returns (21-day rolling)
    log_ret = np.log(close / close.shift(1))
    out["hist_vol"] = log_ret.rolling(21, min_periods=21).std() * np.sqrt(252)

    out["supertrend"] = _supertrend(high, low, close, period=10, multiplier=3.0)

    vol_avg = volume.rolling(20, min_periods=20).mean()

    # ── Boolean signal columns ──────────────────────────────────────────
    ema_bull = (out["ema_fast"] > out["ema_medium"]) & (out["ema_medium"] > out["ema_slow"])
    ema_bear = (out["ema_fast"] < out["ema_medium"]) & (out["ema_medium"] < out["ema_slow"])
    out["ema_aligned"] = ema_bull | ema_bear
    out["ema_bull"] = ema_bull
    out["ema_bear"] = ema_bear

    out["trend_200"] = close > out["ema_trend"]

    macd_bull = (out["macd"] > out["macd_signal"]) & (out["macd_hist"] > 0)
    macd_bear = (out["macd"] < out["macd_signal"]) & (out["macd_hist"] < 0)
    out["macd_ok"] = macd_bull | macd_bear
    out["macd_bull"] = macd_bull
    out["macd_bear"] = macd_bear

    out["vol_surge"] = volume > (vol_avg * params["volume_surge_multiplier"])

    out["supertrend_ok"] = out["supertrend"] != 0  # supertrend has a defined direction

    bb_width_avg = out["bb_width"].rolling(20, min_periods=20).mean()
    out["bb_expansion"] = out["bb_width"] > bb_width_avg

    stoch_bull = (out["stoch_k"] > out["stoch_d"]) & (out["stoch_k"] < 80)
    stoch_bear = (out["stoch_k"] < out["stoch_d"]) & (out["stoch_k"] > 20)
    out["stoch_ok"] = stoch_bull | stoch_bear
    out["stoch_bull"] = stoch_bull
    out["stoch_bear"] = stoch_bear

    obv_sma = out["obv"].rolling(20, min_periods=20).mean()
    obv_bull = out["obv"] > obv_sma
    obv_bear = out["obv"] < obv_sma
    out["obv_ok"] = obv_bull | obv_bear
    out["obv_bull"] = obv_bull
    out["obv_bear"] = obv_bear

    out["close"] = close  # lowercase alias used by screener/wrapper

    return out
