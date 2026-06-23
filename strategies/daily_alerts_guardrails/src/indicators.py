"""
Vectorized indicator computation over full OHLCV history.
All functions return full-length Series aligned to df.index.
No look-ahead bias: bar t uses only data up to and including t.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d  = close.diff()
    up = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up  = df["High"].diff()
    dn  = -df["Low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr_s = atr(df, n)
    pdi = 100 * pdm.ewm(span=n, adjust=False).mean() / atr_s.replace(0, np.nan)
    ndi = 100 * mdm.ewm(span=n, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx  = (abs(pdi - ndi) / (pdi + ndi).replace(0, np.nan)) * 100
    return dx.ewm(span=n, adjust=False).mean()


def macd(close: pd.Series, fast=12, slow=26, sig=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    line   = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


def vol_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    avg = df["Volume"].rolling(n, min_periods=n).mean()
    return df["Volume"] / avg.replace(0, np.nan)


def rs_score(close: pd.Series, spy_close: pd.Series) -> pd.Series:
    """
    Simplified RS score: 63-bar return of stock vs SPY, normalised 1-99.
    Computed for every bar (rolling 63-day momentum ratio).
    """
    idx = close.index.intersection(spy_close.index)
    if len(idx) < 70:
        return pd.Series(50.0, index=close.index)
    s = close.reindex(idx)
    m = spy_close.reindex(idx)
    ret_s = s.pct_change(63)
    ret_m = m.pct_change(63)
    diff  = (ret_s - ret_m).fillna(0)
    # Scale: ±20% outperformance → roughly RS 50 ± 50
    rs = (50 + diff * 250).clip(1, 99)
    return rs.reindex(close.index)


def minervini_score(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized Minervini 8-condition template score (0-8) per bar.
    """
    c = df["Close"]
    s50  = sma(c, 50)
    s150 = sma(c, 150)
    s200 = sma(c, 200)
    s200_22ago = s200.shift(22)
    h52  = df["High"].rolling(252, min_periods=200).max()
    l52  = df["Low"].rolling(252,  min_periods=200).min()

    sc = pd.DataFrame({
        "c1": (c > s150).astype(int),
        "c2": (c > s200).astype(int),
        "c3": (s150 > s200).astype(int),
        "c4": (s200 > s200_22ago).astype(int),
        "c5": ((s50 > s150) & (s50 > s200)).astype(int),
        "c6": (c > s50).astype(int),
        "c7": (c >= l52 * 1.30).astype(int),
        "c8": (c >= h52 * 0.75).astype(int),
    })
    return sc.sum(axis=1)


def ext_vs_50ma(df: pd.DataFrame) -> pd.Series:
    """% extension of Close above/below 50-bar SMA."""
    s50 = sma(df["Close"], 50)
    return (df["Close"] - s50) / s50.replace(0, np.nan) * 100


def build_indicator_frame(df: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators for one ticker's OHLCV DataFrame.
    Returns a DataFrame with one row per bar (aligned to df.index).
    """
    c = df["Close"]

    e9   = ema(c, 9)
    e21  = ema(c, 21)
    e50  = ema(c, 50)
    e200 = ema(c, 200)
    s50  = sma(c, 50)

    rsi14      = rsi(c, 14)
    atr14      = atr(df, 14)
    adx14      = adx(df, 14)
    macd_l, macd_sig, macd_h = macd(c)
    vr         = vol_ratio(df, 20)
    rs         = rs_score(c, spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0])
    mini       = minervini_score(df)
    ext        = ext_vs_50ma(df)

    h52  = df["High"].rolling(252, min_periods=200).max().shift(1)   # prior high (no look-ahead)
    l52  = df["Low"].rolling(252,  min_periods=200).min().shift(1)
    pct_52h = (c - h52) / h52.replace(0, np.nan) * 100

    # ── Trigger signals (boolean series) ──────────────────────────────────────
    # EMA9 × EMA21 bull cross (within last 5 bars — rolling window)
    e9_cross_bull = (e9.shift(1) <= e21.shift(1)) & (e9 > e21)
    e9_cross_bear = (e9.shift(1) >= e21.shift(1)) & (e9 < e21)

    # 52W high breakout with volume
    avg_v = df["Volume"].rolling(20, min_periods=10).mean()
    breakout_52w = (c > h52) & (df["Volume"] > avg_v * 1.3)

    # Volume breakout above recent 20-bar high
    res20 = df["High"].rolling(20, min_periods=10).max().shift(1)
    vol_breakout = (c > res20) & (df["Volume"] > avg_v * 1.5)

    # NR7 break
    rng  = df["High"] - df["Low"]
    nr7  = rng == rng.rolling(7, min_periods=7).min()
    # On day after NR7, check close > NR7 high
    nr7_break = nr7.shift(1) & (c > df["High"].shift(1))

    # BB squeeze breakout
    mid   = sma(c, 20)
    std20 = c.rolling(20, min_periods=20).std()
    upper_bb = mid + 2 * std20
    bb_width = (4 * std20) / mid.replace(0, np.nan)
    bb_squeeze = (bb_width <= bb_width.rolling(126, min_periods=60).min().shift(1) * 1.05) & \
                 (c > upper_bb) & (df["Volume"] > avg_v * 1.3)

    # MA reclaim (close crosses above EMA50)
    ma_reclaim = (c.shift(1) < e50.shift(1)) & (c > e50) & (df["Volume"] > avg_v * 1.2)

    # Inside bar breakout
    mother_h = df["High"].shift(2)
    mother_l = df["Low"].shift(2)
    inside_h = df["High"].shift(1)
    inside_l = df["Low"].shift(1)
    inside_bar = (inside_h < mother_h) & (inside_l > mother_l) & (c > mother_h * 0.998)

    # Gap up continuation (open > prev close by ≥0.5%, close > EMA50)
    gap_up = ((df["Open"] - c.shift(1)) / c.shift(1) * 100 >= 0.5) & (c > e50)

    # Gap down
    gap_down = ((c.shift(1) - df["Open"]) / c.shift(1) * 100 >= 0.5) & (c < e50)

    # MACD bull/bear cross
    macd_bull = (macd_l.shift(1) <= macd_sig.shift(1)) & (macd_l > macd_sig)
    macd_bear = (macd_l.shift(1) >= macd_sig.shift(1)) & (macd_l < macd_sig)

    frame = pd.DataFrame({
        "Close":       c,
        "Open":        df["Open"],
        "High":        df["High"],
        "Low":         df["Low"],
        "Volume":      df["Volume"],
        "EMA9":        e9,
        "EMA21":       e21,
        "EMA50":       e50,
        "EMA200":      e200,
        "SMA50":       s50,
        "RSI":         rsi14,
        "ATR":         atr14,
        "ADX":         adx14,
        "MACD_H":      macd_h,
        "MACD_H_prev": macd_h.shift(1),
        "VolRatio":    vr,
        "RS":          rs,
        "Minervini":   mini,
        "Ext50MA":     ext,
        "H52":         h52,
        "L52":         l52,
        "Pct52H":      pct_52h,
        # triggers
        "T_EMA9_BULL":    e9_cross_bull,
        "T_EMA9_BEAR":    e9_cross_bear,
        "T_52W_BREAK":    breakout_52w,
        "T_VOL_BREAK":    vol_breakout,
        "T_NR7":          nr7_break,
        "T_BB_SQ":        bb_squeeze,
        "T_MA_RECLAIM":   ma_reclaim,
        "T_INSIDE":       inside_bar,
        "T_GAP_UP":       gap_up,
        "T_GAP_DOWN":     gap_down,
        "T_MACD_BULL":    macd_bull,
        "T_MACD_BEAR":    macd_bear,
    }, index=df.index)

    return frame
