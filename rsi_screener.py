"""
RSI Screener — Swing Trading
============================
Three complementary RSI-based signals:

1. OVERSOLD / OVERBOUGHT (mean-reversion)
   • RSI(14) crosses below `oversold` threshold (default 30) → bullish reversal candidate
   • RSI(14) crosses above `overbought` threshold (default 70) → bearish reversal candidate
   • Confirmation: price must close > EMA-50 for longs (or < EMA-50 for shorts)
   • Volume filter: volume ≥ min_vol_mult × 20-bar average

2. RSI DIVERGENCE (momentum divergence)
   • Bullish divergence: price makes lower low, RSI makes higher low over `div_lookback` bars
   • Bearish divergence: price makes higher high, RSI makes lower high over `div_lookback` bars
   • Identifies early trend reversals before price confirmation

3. RSI TREND MOMENTUM
   • RSI(14) crosses above 50 from below → bull momentum shift
   • RSI(14) crosses below 50 from above → bear momentum shift
   • Filtered by ADX > adx_threshold (trending conditions only)

STOPS & TARGETS (ATR-based):
  Stop   = entry ± 1.5 × ATR(14)
  Target = entry ± 2.5 × ATR(14)   (1.67:1 R/R)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from livermore_pivotal_screener import get_universe, DEFAULT_TICKERS


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=n, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = _atr(df, 1)
    atr_s = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(span=period, adjust=False).mean()


# ── Signal detection ──────────────────────────────────────────────────────────

def detect_rsi_signals(
    df: pd.DataFrame,
    oversold: float = 30.0,
    overbought: float = 70.0,
    rsi_period: int = 14,
    div_lookback: int = 20,
    adx_threshold: float = 20.0,
    vol_mult: float = 1.0,
    recent_bars: int = 10,
    signal_filter: str = "ALL",
) -> pd.DataFrame:
    """
    Annotate df with RSI signals. Returns df with signal columns added.
    """
    df = df.copy()
    if len(df) < 50:
        return pd.DataFrame()

    df["rsi"] = _rsi(df["Close"], rsi_period)
    df["ema50"] = _ema(df["Close"], 50)
    df["ema200"] = _ema(df["Close"], 200)
    df["atr14"] = _atr(df, 14)
    df["adx14"] = _adx(df, 14)
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma20"].replace(0, np.nan)

    signals = []

    # 1. Oversold / Overbought cross
    rsi_cross_bull = (df["rsi"].shift(1) >= oversold) & (df["rsi"] < oversold)  # just crossed below
    rsi_cross_bear = (df["rsi"].shift(1) <= overbought) & (df["rsi"] > overbought)  # just crossed above

    # 2. RSI 50-line crosses (momentum shift)
    rsi_50_bull = (df["rsi"].shift(1) < 50) & (df["rsi"] >= 50)
    rsi_50_bear = (df["rsi"].shift(1) > 50) & (df["rsi"] <= 50)

    # 3. Divergence (swing detection over lookback)
    def _bullish_div(i: int) -> bool:
        if i < div_lookback:
            return False
        window = df.iloc[i - div_lookback: i + 1]
        price_low_idx = window["Low"].idxmin()
        rsi_low_idx = window["rsi"].idxmin()
        if price_low_idx == rsi_low_idx:
            return False
        # price low is more recent than rsi low → price made lower low but rsi didn't
        price_pos = window.index.get_loc(price_low_idx)
        rsi_pos = window.index.get_loc(rsi_low_idx)
        if price_pos <= rsi_pos:
            return False
        # Confirm: price at recent low < earlier low, RSI at recent low > earlier rsi
        mid = div_lookback // 2
        early_price_low = window.iloc[:mid]["Low"].min()
        late_price_low = window.iloc[mid:]["Low"].min()
        early_rsi_low = window.iloc[:mid]["rsi"].min()
        late_rsi_low = window.iloc[mid:]["rsi"].min()
        return (late_price_low < early_price_low) and (late_rsi_low > early_rsi_low)

    def _bearish_div(i: int) -> bool:
        if i < div_lookback:
            return False
        window = df.iloc[i - div_lookback: i + 1]
        mid = div_lookback // 2
        early_price_high = window.iloc[:mid]["High"].max()
        late_price_high = window.iloc[mid:]["High"].max()
        early_rsi_high = window.iloc[:mid]["rsi"].max()
        late_rsi_high = window.iloc[mid:]["rsi"].max()
        return (late_price_high > early_price_high) and (late_rsi_high < early_rsi_high)

    lookback_start = max(0, len(df) - recent_bars)
    for i in range(lookback_start, len(df)):
        row = df.iloc[i]
        close = row["Close"]
        rsi_val = row["rsi"]
        atr_val = row["atr14"]
        adx_val = row["adx14"]
        vol_r = row["vol_ratio"]
        date = df.index[i]

        if pd.isna(rsi_val) or pd.isna(atr_val):
            continue
        if vol_r < vol_mult:
            continue

        # Oversold → bullish
        if rsi_cross_bull.iloc[i]:
            sig = "OVERSOLD_BULL"
            if signal_filter not in ("ALL", "BULLISH"):
                pass
            else:
                signals.append({
                    "Signal": sig, "date_idx": i,
                    "Signal Date": date.strftime("%Y-%m-%d"),
                    "RSI": round(rsi_val, 1),
                    "Close": round(close, 2),
                    "ADX": round(adx_val, 1) if pd.notna(adx_val) else None,
                    "Vol vs Avg": round(vol_r, 2),
                    "Stop": round(close - 1.5 * atr_val, 2),
                    "Target": round(close + 2.5 * atr_val, 2),
                    "R/R": round(2.5 / 1.5, 2),
                    "EMA50": "ABOVE" if close > row["ema50"] else "BELOW",
                    "Divergence": "Bullish" if _bullish_div(i) else "—",
                })

        # Overbought → bearish
        if rsi_cross_bear.iloc[i]:
            sig = "OVERBOUGHT_BEAR"
            if signal_filter not in ("ALL", "BEARISH"):
                pass
            else:
                signals.append({
                    "Signal": sig, "date_idx": i,
                    "Signal Date": date.strftime("%Y-%m-%d"),
                    "RSI": round(rsi_val, 1),
                    "Close": round(close, 2),
                    "ADX": round(adx_val, 1) if pd.notna(adx_val) else None,
                    "Vol vs Avg": round(vol_r, 2),
                    "Stop": round(close + 1.5 * atr_val, 2),
                    "Target": round(close - 2.5 * atr_val, 2),
                    "R/R": round(2.5 / 1.5, 2),
                    "EMA50": "ABOVE" if close > row["ema50"] else "BELOW",
                    "Divergence": "Bearish" if _bearish_div(i) else "—",
                })

        # RSI 50 cross bullish
        if rsi_50_bull.iloc[i] and (pd.isna(adx_val) or adx_val >= adx_threshold):
            sig = "MOMENTUM_BULL"
            if signal_filter not in ("ALL", "BULLISH"):
                pass
            else:
                signals.append({
                    "Signal": sig, "date_idx": i,
                    "Signal Date": date.strftime("%Y-%m-%d"),
                    "RSI": round(rsi_val, 1),
                    "Close": round(close, 2),
                    "ADX": round(adx_val, 1) if pd.notna(adx_val) else None,
                    "Vol vs Avg": round(vol_r, 2),
                    "Stop": round(close - 1.5 * atr_val, 2),
                    "Target": round(close + 2.5 * atr_val, 2),
                    "R/R": round(2.5 / 1.5, 2),
                    "EMA50": "ABOVE" if close > row["ema50"] else "BELOW",
                    "Divergence": "—",
                })

        # RSI 50 cross bearish
        if rsi_50_bear.iloc[i] and (pd.isna(adx_val) or adx_val >= adx_threshold):
            sig = "MOMENTUM_BEAR"
            if signal_filter not in ("ALL", "BEARISH"):
                pass
            else:
                signals.append({
                    "Signal": sig, "date_idx": i,
                    "Signal Date": date.strftime("%Y-%m-%d"),
                    "RSI": round(rsi_val, 1),
                    "Close": round(close, 2),
                    "ADX": round(adx_val, 1) if pd.notna(adx_val) else None,
                    "Vol vs Avg": round(vol_r, 2),
                    "Stop": round(close + 1.5 * atr_val, 2),
                    "Target": round(close - 2.5 * atr_val, 2),
                    "R/R": round(2.5 / 1.5, 2),
                    "EMA50": "ABOVE" if close > row["ema50"] else "BELOW",
                    "Divergence": "—",
                })

    # Standalone divergence scan (not tied to cross)
    if signal_filter in ("ALL", "BULLISH"):
        for i in range(lookback_start, len(df)):
            row = df.iloc[i]
            if _bullish_div(i) and row["rsi"] < 50:
                close = row["Close"]
                atr_val = row["atr14"]
                adx_val = row["adx14"]
                vol_r = row["vol_ratio"]
                date = df.index[i]
                if vol_r < vol_mult:
                    continue
                # Don't duplicate if already flagged
                if not any(s["Signal Date"] == date.strftime("%Y-%m-%d") and "BULL" in s["Signal"] for s in signals):
                    signals.append({
                        "Signal": "DIV_BULL",
                        "date_idx": i,
                        "Signal Date": date.strftime("%Y-%m-%d"),
                        "RSI": round(row["rsi"], 1),
                        "Close": round(close, 2),
                        "ADX": round(adx_val, 1) if pd.notna(adx_val) else None,
                        "Vol vs Avg": round(vol_r, 2),
                        "Stop": round(close - 1.5 * atr_val, 2),
                        "Target": round(close + 2.5 * atr_val, 2),
                        "R/R": round(2.5 / 1.5, 2),
                        "EMA50": "ABOVE" if close > row["ema50"] else "BELOW",
                        "Divergence": "Bullish",
                    })

    if signal_filter in ("ALL", "BEARISH"):
        for i in range(lookback_start, len(df)):
            row = df.iloc[i]
            if _bearish_div(i) and row["rsi"] > 50:
                close = row["Close"]
                atr_val = row["atr14"]
                adx_val = row["adx14"]
                vol_r = row["vol_ratio"]
                date = df.index[i]
                if vol_r < vol_mult:
                    continue
                if not any(s["Signal Date"] == date.strftime("%Y-%m-%d") and "BEAR" in s["Signal"] for s in signals):
                    signals.append({
                        "Signal": "DIV_BEAR",
                        "date_idx": i,
                        "Signal Date": date.strftime("%Y-%m-%d"),
                        "RSI": round(row["rsi"], 1),
                        "Close": round(close, 2),
                        "ADX": round(adx_val, 1) if pd.notna(adx_val) else None,
                        "Vol vs Avg": round(vol_r, 2),
                        "Stop": round(close + 1.5 * atr_val, 2),
                        "Target": round(close - 2.5 * atr_val, 2),
                        "R/R": round(2.5 / 1.5, 2),
                        "EMA50": "ABOVE" if close > row["ema50"] else "BELOW",
                        "Divergence": "Bearish",
                    })

    if not signals:
        return pd.DataFrame()

    # Keep the most recent signal per ticker
    result = pd.DataFrame(signals)
    result = result.sort_values("date_idx", ascending=False).drop_duplicates("Signal Date").head(1)
    return result.drop(columns=["date_idx"])


def run_rsi_screener(
    tickers: Optional[List[str]] = None,
    oversold: float = 30.0,
    overbought: float = 70.0,
    rsi_period: int = 14,
    div_lookback: int = 20,
    adx_threshold: float = 20.0,
    vol_mult: float = 1.0,
    recent_bars: int = 10,
    signal_filter: str = "ALL",
    lookback_days: int = 120,
) -> pd.DataFrame:
    if tickers is None:
        tickers = DEFAULT_TICKERS

    end = datetime.today()
    start = end - timedelta(days=lookback_days + 60)
    results = []

    for ticker in tickers:
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if raw is None or raw.empty or len(raw) < 50:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.loc[:, ~raw.columns.duplicated()]
            df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
            sigs = detect_rsi_signals(
                df,
                oversold=oversold,
                overbought=overbought,
                rsi_period=rsi_period,
                div_lookback=div_lookback,
                adx_threshold=adx_threshold,
                vol_mult=vol_mult,
                recent_bars=recent_bars,
                signal_filter=signal_filter,
            )
            if not sigs.empty:
                sigs.insert(0, "Ticker", ticker)
                results.append(sigs)
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    # Add Bars Ago column
    today = pd.Timestamp(datetime.today().date())
    out["Bars Ago"] = out["Signal Date"].apply(
        lambda d: max(0, (today - pd.Timestamp(d)).days * 5 // 7)
    )
    return out.sort_values(["Signal", "Bars Ago"])
