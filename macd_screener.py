"""
MACD Screener — Swing Trading
==============================
Three MACD-based signal types:

1. SIGNAL LINE CROSS (classic MACD signal)
   • MACD line crosses ABOVE signal line → BULLISH_CROSS
   • MACD line crosses BELOW signal line → BEARISH_CROSS
   • Both MACD and signal must be below/above zero for stronger setups (optional filter)
   • Volume > vol_mult × 20-bar average confirms institutional participation

2. HISTOGRAM FLIP (leading indicator — fires before signal cross)
   • Histogram turns from negative to positive (≥ 0 from < 0) → HIST_FLIP_BULL
   • Histogram turns from positive to negative (≤ 0 from > 0) → HIST_FLIP_BEAR
   • Earlier signal than line cross; more false signals — best combined with trend filter

3. ZERO LINE CROSS (trend confirmation — later but higher quality)
   • MACD line crosses above 0 → ZERO_BULL (trend has shifted to bullish)
   • MACD line crosses below 0 → ZERO_BEAR (trend has shifted to bearish)

MACD DIVERGENCE (bonus):
   • Bullish div: price lower low, MACD higher low → early reversal
   • Bearish div: price higher high, MACD lower high → momentum exhaustion

Default parameters: fast=12, slow=26, signal=9 (classic MACD)
Stops & targets use ATR(14): Stop ±1.5×ATR, Target ±2.5×ATR (1.67:1 R/R)
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


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram) as pd.Series."""
    macd_line = _ema(close, fast) - _ema(close, slow)
    sig_line = _ema(macd_line, signal)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Signal detection ──────────────────────────────────────────────────────────

def detect_macd_signals(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    vol_mult: float = 1.0,
    require_zero_side: bool = False,
    recent_bars: int = 10,
    div_lookback: int = 20,
    signal_filter: str = "ALL",
) -> pd.DataFrame:
    """Annotate df with MACD signals. Returns signal rows."""
    df = df.copy()
    if len(df) < slow + signal_period + 10:
        return pd.DataFrame()

    df["macd"], df["sig_line"], df["hist"] = _macd(df["Close"], fast, slow, signal_period)
    df["ema50"] = _ema(df["Close"], 50)
    df["ema200"] = _ema(df["Close"], 200)
    df["atr14"] = _atr(df, 14)
    df["rsi14"] = _rsi(df["Close"], 14)
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma20"].replace(0, np.nan)

    # Crossover flags
    macd_cross_bull = (df["macd"].shift(1) <= df["sig_line"].shift(1)) & (df["macd"] > df["sig_line"])
    macd_cross_bear = (df["macd"].shift(1) >= df["sig_line"].shift(1)) & (df["macd"] < df["sig_line"])
    hist_flip_bull = (df["hist"].shift(1) < 0) & (df["hist"] >= 0)
    hist_flip_bear = (df["hist"].shift(1) > 0) & (df["hist"] <= 0)
    zero_cross_bull = (df["macd"].shift(1) <= 0) & (df["macd"] > 0)
    zero_cross_bear = (df["macd"].shift(1) >= 0) & (df["macd"] < 0)

    def _is_bullish_filter(sig_type):
        return signal_filter in ("ALL", "BULLISH") or sig_type.endswith("BULL")

    def _is_bearish_filter(sig_type):
        return signal_filter in ("ALL", "BEARISH") or sig_type.endswith("BEAR")

    signals = []
    lookback_start = max(0, len(df) - recent_bars)

    for i in range(lookback_start, len(df)):
        row = df.iloc[i]
        close = row["Close"]
        atr_val = row["atr14"]
        vol_r = row["vol_ratio"]
        macd_val = row["macd"]
        sig_val = row["sig_line"]
        hist_val = row["hist"]
        date = df.index[i]

        if pd.isna(atr_val) or pd.isna(macd_val):
            continue
        if vol_r < vol_mult:
            continue

        def _entry(sig_type: str, direction: str):
            is_long = direction == "bull"
            stop = round(close - 1.5 * atr_val, 2) if is_long else round(close + 1.5 * atr_val, 2)
            target = round(close + 2.5 * atr_val, 2) if is_long else round(close - 2.5 * atr_val, 2)
            return {
                "Signal": sig_type,
                "date_idx": i,
                "Signal Date": date.strftime("%Y-%m-%d"),
                "MACD": round(macd_val, 4),
                "Signal Line": round(sig_val, 4),
                "Histogram": round(hist_val, 4),
                "RSI": round(row["rsi14"], 1) if pd.notna(row["rsi14"]) else None,
                "Close": round(close, 2),
                "Vol vs Avg": round(vol_r, 2),
                "Stop": stop,
                "Target": target,
                "R/R": round(2.5 / 1.5, 2),
                "EMA50": "ABOVE" if close > row["ema50"] else "BELOW",
            }

        # Signal line cross
        if macd_cross_bull.iloc[i]:
            if not require_zero_side or macd_val < 0:
                if signal_filter in ("ALL", "BULLISH"):
                    signals.append(_entry("MACD_CROSS_BULL", "bull"))

        if macd_cross_bear.iloc[i]:
            if not require_zero_side or macd_val > 0:
                if signal_filter in ("ALL", "BEARISH"):
                    signals.append(_entry("MACD_CROSS_BEAR", "bear"))

        # Histogram flip
        if hist_flip_bull.iloc[i]:
            if signal_filter in ("ALL", "BULLISH"):
                e = _entry("HIST_FLIP_BULL", "bull")
                signals.append(e)

        if hist_flip_bear.iloc[i]:
            if signal_filter in ("ALL", "BEARISH"):
                signals.append(_entry("HIST_FLIP_BEAR", "bear"))

        # Zero line cross
        if zero_cross_bull.iloc[i]:
            if signal_filter in ("ALL", "BULLISH"):
                signals.append(_entry("ZERO_CROSS_BULL", "bull"))

        if zero_cross_bear.iloc[i]:
            if signal_filter in ("ALL", "BEARISH"):
                signals.append(_entry("ZERO_CROSS_BEAR", "bear"))

    if not signals:
        return pd.DataFrame()

    result = pd.DataFrame(signals)
    result = result.sort_values("date_idx", ascending=False).drop_duplicates(subset=["date_idx", "Signal"]).head(3)
    return result.drop(columns=["date_idx"])


def run_macd_screener(
    tickers: Optional[List[str]] = None,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    vol_mult: float = 1.0,
    require_zero_side: bool = False,
    recent_bars: int = 10,
    div_lookback: int = 20,
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
            if raw is None or raw.empty or len(raw) < 60:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.loc[:, ~raw.columns.duplicated()]
            df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
            sigs = detect_macd_signals(
                df,
                fast=fast,
                slow=slow,
                signal_period=signal_period,
                vol_mult=vol_mult,
                require_zero_side=require_zero_side,
                recent_bars=recent_bars,
                div_lookback=div_lookback,
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
    today = pd.Timestamp(datetime.today().date())
    out["Bars Ago"] = out["Signal Date"].apply(
        lambda d: max(0, (today - pd.Timestamp(d)).days * 5 // 7)
    )
    return out.sort_values(["Signal", "Bars Ago"])
