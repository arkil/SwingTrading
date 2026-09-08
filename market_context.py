"""
Market Context — shared regime/macro layer
=============================================
Single source of truth for VIX, SPY trend, yield curve, breadth, and sector
rotation. Replaces the previously-duplicated VIX fetchers in dashboard.py,
swing_options_screener.py, and options_paper_trader.py.

Importable identically from Streamlit (dashboard.py) and from bare daemon
scripts (options_paper_trader.py, stock_paper_trader.py, alerts_live_runner.py,
v6_live_daemon.py) — deliberately has no st.cache_data dependency, just a
small local TTL cache, so behavior and caching are identical everywhere.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Tiny process-local TTL cache (works in Streamlit and bare scripts) ─────────

_cache: dict = {}


def _ttl_cached(key: str, ttl_seconds: float, fn: Callable):
    now = time.time()
    hit = _cache.get(key)
    if hit is not None and (now - hit[0]) < ttl_seconds:
        return hit[1]
    value = fn()
    _cache[key] = (now, value)
    return value


# ── Raw fetch helpers ────────────────────────────────────────────────────────

def _download(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


# ── VIX ──────────────────────────────────────────────────────────────────────

def _compute_vix() -> dict:
    df = _download("^VIX", period="1y")
    if df.empty:
        return {"level": 20.0, "rank_1y": 50.0}
    close = df["Close"]
    level = float(close.iloc[-1])
    rank = float((close < level).mean() * 100) if len(close) > 10 else 50.0
    return {"level": round(level, 2), "rank_1y": round(rank, 1)}


def get_vix() -> dict:
    """Current VIX level + 1-year percentile rank. {level, rank_1y}."""
    return _ttl_cached("vix", 300, _compute_vix)


# ── SPY trend ────────────────────────────────────────────────────────────────

def _compute_spy_trend() -> dict:
    df = _download("SPY", period="1y")
    if df.empty or len(df) < 200:
        return {"price": None, "chg_pct": 0.0, "sma50": None, "sma200": None, "trend": "NEUTRAL"}
    close = df["Close"]
    price = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else price
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    if price > sma50 > sma200:
        trend = "UPTREND"
    elif price < sma50 < sma200:
        trend = "DOWNTREND"
    else:
        trend = "NEUTRAL"
    return {
        "price": round(price, 2),
        "chg_pct": round((price - prev) / prev * 100, 2) if prev else 0.0,
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "trend": trend,
    }


def get_spy_trend() -> dict:
    """SPY price/trend snapshot. {price, chg_pct, sma50, sma200, trend}."""
    return _ttl_cached("spy_trend", 300, _compute_spy_trend)


# ── Yield curve ──────────────────────────────────────────────────────────────

def _compute_yield_curve() -> dict:
    y10_df = _download("^TNX", period="5d")
    m3_df = _download("^IRX", period="5d")
    if y10_df.empty or m3_df.empty:
        return {"y10": None, "m3": None, "spread": None, "inverted": False}
    y10 = float(y10_df["Close"].iloc[-1])
    m3 = float(m3_df["Close"].iloc[-1])
    spread = round(y10 - m3, 2)
    return {"y10": round(y10, 2), "m3": round(m3, 2), "spread": spread, "inverted": spread < 0}


def get_yield_curve() -> dict:
    """10Y (^TNX) vs 3M (^IRX) yields. {y10, m3, spread, inverted}."""
    return _ttl_cached("yield_curve", 900, _compute_yield_curve)


# ── Breadth ──────────────────────────────────────────────────────────────────

def _download_batch(tickers: tuple, period: str) -> dict:
    """
    One HTTP round-trip for many tickers (yfinance native batch download),
    instead of N separate requests — even N *threaded* requests still open
    N connections/DNS lookups, which is exactly what caused the earnings-
    calendar thread-exhaustion bug fixed earlier. Returns {ticker: df}.
    """
    if not tickers:
        return {}
    try:
        raw = yf.download(list(tickers), period=period, progress=False,
                          auto_adjust=True, group_by="ticker", threads=True)
    except Exception:
        return {}
    if raw is None or raw.empty:
        return {}

    # dropna(how="all") is not enough: yfinance's placeholder row for the
    # current calendar day (before that day's data has posted) often has a
    # real Volume figure while Open/High/Low/Close are all NaN — confirmed
    # live for XLK/XLF. "how=all" only drops a row when *every* column is
    # NaN, so that row survives and poisons every downstream % calc as NaN.
    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                sub = raw[t].dropna(subset=["Close"])
                if not sub.empty:
                    out[t] = sub
    elif len(tickers) == 1:
        out[tickers[0]] = raw.dropna(subset=["Close"])
    return out


def _compute_breadth(tickers: tuple) -> dict:
    data = _download_batch(tickers, period="1y")
    above_50 = above_200 = counted = 0
    for df in data.values():
        if df.empty or len(df) < 200 or "Close" not in df.columns:
            continue
        close = df["Close"]
        price = float(close.iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])
        counted += 1
        above_50 += int(price > sma50)
        above_200 += int(price > sma200)

    if counted == 0:
        return {"pct_above_50ma": None, "pct_above_200ma": None, "n": 0}
    return {
        "pct_above_50ma": round(above_50 / counted * 100, 1),
        "pct_above_200ma": round(above_200 / counted * 100, 1),
        "n": counted,
    }


def get_breadth(tickers: Optional[list] = None) -> dict:
    """% of a universe trading above its 50/200-day MA. {pct_above_50ma, pct_above_200ma, n}."""
    if tickers is None:
        from livermore_pivotal_screener import WATCHLIST_TICKERS
        tickers = WATCHLIST_TICKERS
    key = f"breadth:{hash(tuple(sorted(tickers)))}"
    return _ttl_cached(key, 600, lambda: _compute_breadth(tuple(tickers)))


# ── Sector rotation ──────────────────────────────────────────────────────────

_SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLY": "Consumer Disc.", "XLP": "Consumer Staples", "XLI": "Industrials",
    "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication",
}


def _compute_sector_performance() -> dict:
    data = _download_batch(tuple(_SECTOR_ETFS), period="3mo")
    result = {}
    for etf, df in data.items():
        if df.empty or len(df) < 22 or "Close" not in df.columns:
            continue
        close = df["Close"]
        price = float(close.iloc[-1])

        def _chg(bars_ago, close=close, price=price):
            if len(close) <= bars_ago:
                return None
            ref = float(close.iloc[-(bars_ago + 1)])
            return round((price - ref) / ref * 100, 2) if ref else None

        result[etf] = {"name": _SECTOR_ETFS[etf], "chg_1d": _chg(1), "chg_1w": _chg(5), "chg_1m": _chg(21)}
    return result


def get_sector_performance() -> dict:
    """1d/1w/1m % change for the 11 SPDR sector ETFs, keyed by ticker."""
    return _ttl_cached("sector_perf", 600, _compute_sector_performance)


# ── Macro calendar passthrough ──────────────────────────────────────────────

def _compute_next_high_impact_event() -> Optional[dict]:
    try:
        from economic_calendar import get_event_context
        ctx = get_event_context(days_ahead=14)
        events = ctx.get("next_events") or []
        return events[0] if events else None
    except Exception:
        return None


def next_high_impact_event() -> Optional[dict]:
    """
    Nearest upcoming HIGH-impact macro event, or None. Reuses economic_calendar.
    Cached here — economic_calendar's own ForexFactory fetch has no caching of
    its own, and this gets called from the sidebar on every page render.
    """
    return _ttl_cached("next_event", 900, _compute_next_high_impact_event)


# ── Composite regime ─────────────────────────────────────────────────────────

def get_regime_summary() -> dict:
    """
    Single composite read of current market regime — the one function
    everything else (sidebar strip, macro page, regime-aware filters) calls,
    so all consumers see the same label from the same underlying data.

    Returns:
        {vix, spy, yield_curve, next_event, label}
    label is one of: RISK-ON | NEUTRAL | RISK-OFF | HIGH-VOL
    """
    vix = get_vix()
    spy = get_spy_trend()
    yc = get_yield_curve()
    event = next_high_impact_event()

    if vix["rank_1y"] >= 80:
        label = "HIGH-VOL"
    elif vix["rank_1y"] >= 60 or spy["trend"] == "DOWNTREND":
        label = "RISK-OFF"
    elif spy["trend"] == "UPTREND" and vix["rank_1y"] < 40:
        label = "RISK-ON"
    else:
        label = "NEUTRAL"

    return {
        "vix": vix,
        "spy": spy,
        "yield_curve": yc,
        "next_event": event,
        "label": label,
    }
