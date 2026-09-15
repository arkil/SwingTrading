"""
Data pipeline for the swing_options_45_60d strategy backtest.

Downloads (and caches to Data/) daily OHLCV for the strategy universe plus
VIX, for use by run_backtest.py. Mirrors the fetch/clean logic already used
by Scripts/SwingTrading/swing_options_screener.py's live wrapper
(_fetch_ohlcv / _fetch_vix_rank) so backtest and live behave consistently.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "Data"))
_OHLCV_DIR = os.path.join(_DATA_DIR, "ohlcv")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df = df[df["Close"] > 0]
    return df


def fetch_symbol(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV for a symbol via yfinance."""
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        return _clean(df)
    except Exception:
        return pd.DataFrame()


def load_or_fetch(symbol: str, start: str, end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Load a symbol's OHLCV from Data/ohlcv/{symbol}.csv, fetching+caching if missing."""
    os.makedirs(_OHLCV_DIR, exist_ok=True)
    path = os.path.join(_OHLCV_DIR, f"{symbol}.csv")

    if not refresh and os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if not df.empty:
            return df

    df = fetch_symbol(symbol, start=start, end=end)
    if not df.empty:
        df.to_csv(path)
    return df


def load_vix(start: str, end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Load VIX OHLC from Data/vix.csv, fetching+caching if missing."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    path = os.path.join(_DATA_DIR, "vix.csv")

    if not refresh and os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if not df.empty:
            return df

    df = fetch_symbol("^VIX", start=start, end=end)
    if not df.empty:
        df.to_csv(path)
    return df


def load_universe(universe: list[str], start: str, end: str | None = None, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Load OHLCV for every symbol in the universe (+ SPY, always needed for regime filter)."""
    symbols = list(dict.fromkeys(universe + ["SPY"]))  # de-dup, preserve order
    data = {}
    for sym in symbols:
        df = load_or_fetch(sym, start=start, end=end, refresh=refresh)
        if not df.empty:
            data[sym] = df
    return data


def vix_iv_rank_series(vix: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Rolling approximate IV rank (0-100) from VIX close, using a trailing
    `window`-day min/max — matches the live wrapper's 1y VIX approach but
    computed at every historical bar (no lookahead: window ends at t).
    """
    close = vix["Close"]
    roll_min = close.rolling(window, min_periods=20).min()
    roll_max = close.rolling(window, min_periods=20).max()
    rank = (close - roll_min) / (roll_max - roll_min + 1e-9) * 100
    return rank.clip(0, 100)
