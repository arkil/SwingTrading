"""
Gap Scanner — Swing Trading
============================
Detects and classifies significant opening price gaps for swing trade setups.

GAP TYPES
─────────────────────────────────────────────────────────────────────────────
BREAKAWAY GAP (highest conviction)
  • Gap occurs out of a consolidation or base (Bollinger Band squeeze → BB Width < 6-month median)
  • Price gaps above a key resistance / prior pivot high
  • Volume surge ≥ 2× average (institutional buying / news catalyst)
  • Rarely fills quickly — continuation expected

CONTINUATION GAP (runaway gap)
  • Gap occurs mid-trend: price already above EMA-50, trending up
  • Volume ≥ 1.5× average
  • Typically does NOT fill in the near term — add to position / new entry

EXHAUSTION GAP (reversal warning)
  • Gap occurs after an extended move: price ≥ 20% above 52-week low, RSI > 70
  • Volume spike but follow-through weak (close near open, upper wick)
  • High probability of gap fill — fade or avoid

COMMON GAP (noise)
  • Small gap (< min_gap_pct%) that occurs within a range
  • Usually fills within 1–5 days
  • Low trade value — filtered out by default

GAP FILL TRACKER
  • For gaps from up to `lookback_days` ago, shows whether price has since filled the gap
  • "Fill %" = how much of the gap has been retraced by current price

STOPS & TARGETS
  • Gap Up  long:  Stop = gap low (prev close), Target = gap close + 2.0 × gap size
  • Gap Down short: Stop = gap high (prev close), Target = gap close − 2.0 × gap size
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


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _bb_width(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    sma = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = sma + k * std
    lower = sma - k * std
    return (upper - lower) / sma.replace(0, np.nan) * 100


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=n, adjust=False).mean()


# ── Gap classification ────────────────────────────────────────────────────────

def _classify_gap(
    df: pd.DataFrame,
    i: int,
    direction: str,  # "UP" or "DOWN"
) -> str:
    """Classify gap type based on trend and volatility context."""
    close_series = df["Close"]
    rsi_series = df.get("rsi14", pd.Series(dtype=float))
    bb_w = df.get("bb_width", pd.Series(dtype=float))

    price = close_series.iloc[i]
    ema50_val = df["ema50"].iloc[i] if "ema50" in df.columns else None
    rsi_val = rsi_series.iloc[i] if len(rsi_series) > i else np.nan
    bb_val = bb_w.iloc[i] if len(bb_w) > i else np.nan

    # BB squeeze: current width < 6-month median
    bb_squeeze = False
    if not np.isnan(bb_val) and i >= 126:
        bb_6m_median = bb_w.iloc[max(0, i - 126):i].median()
        bb_squeeze = bb_val < bb_6m_median

    # 52-week context
    window_52w = close_series.iloc[max(0, i - 252):i + 1]
    pct_from_52w_low = (price / window_52w.min() - 1) * 100 if len(window_52w) > 0 else 0

    if direction == "UP":
        if bb_squeeze:
            return "BREAKAWAY"
        if ema50_val and price > ema50_val:
            if not np.isnan(rsi_val) and rsi_val > 70 and pct_from_52w_low > 25:
                return "EXHAUSTION"
            return "CONTINUATION"
        return "COMMON"
    else:  # DOWN
        if bb_squeeze:
            return "BREAKAWAY"
        if ema50_val and price < ema50_val:
            if not np.isnan(rsi_val) and rsi_val < 30 and pct_from_52w_low < 5:
                return "EXHAUSTION"
            return "CONTINUATION"
        return "COMMON"


def detect_gaps(
    df: pd.DataFrame,
    min_gap_pct: float = 0.5,
    vol_mult: float = 1.5,
    recent_bars: int = 10,
    gap_direction: str = "ALL",
    include_common: bool = False,
) -> pd.DataFrame:
    """
    Scan df for gap events. Returns one row per gap found.

    Parameters
    ----------
    min_gap_pct : minimum gap size as % of prev close
    vol_mult    : minimum volume ratio on gap day
    recent_bars : only look at last N bars
    gap_direction : "ALL", "UP", "DOWN"
    include_common : whether to include COMMON gap type in results
    """
    df = df.copy()
    if len(df) < 60:
        return pd.DataFrame()

    df["ema50"] = _ema(df["Close"], 50)
    df["ema200"] = _ema(df["Close"], 200)
    df["rsi14"] = _rsi(df["Close"], 14)
    df["bb_width"] = _bb_width(df["Close"], 20)
    df["atr14"] = _atr(df, 14)
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma20"].replace(0, np.nan)

    current_price = df["Close"].iloc[-1]
    results = []
    lookback_start = max(1, len(df) - recent_bars)

    for i in range(lookback_start, len(df)):
        prev_close = df["Close"].iloc[i - 1]
        open_price = df["Open"].iloc[i]
        close_price = df["Close"].iloc[i]
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        vol_r = df["vol_ratio"].iloc[i]
        atr_val = df["atr14"].iloc[i]
        date = df.index[i]

        if pd.isna(prev_close) or prev_close == 0:
            continue

        gap_pct = (open_price - prev_close) / prev_close * 100

        if abs(gap_pct) < min_gap_pct:
            continue
        if vol_r < vol_mult:
            continue

        direction = "UP" if gap_pct > 0 else "DOWN"
        if gap_direction != "ALL" and direction != gap_direction:
            continue

        gap_type = _classify_gap(df, i, direction)
        if gap_type == "COMMON" and not include_common:
            continue

        # Gap fill tracker: how much of the gap has been filled since gap day
        gap_open = open_price
        gap_bottom = min(open_price, prev_close)
        gap_top = max(open_price, prev_close)
        gap_size = gap_top - gap_bottom

        if direction == "UP":
            # Filled if current_price dropped back to prev_close
            fill_pct = max(0, min(100, (gap_top - current_price) / gap_size * 100)) if gap_size > 0 else 0
            stop = round(prev_close, 2)
            target = round(close_price + 2.0 * abs(gap_pct) / 100 * prev_close, 2)
        else:
            fill_pct = max(0, min(100, (current_price - gap_bottom) / gap_size * 100)) if gap_size > 0 else 0
            stop = round(prev_close, 2)
            target = round(close_price - 2.0 * abs(gap_pct) / 100 * prev_close, 2)

        # Body quality: close near high = strong gap day, close near low = weak
        bar_range = high - low
        if direction == "UP":
            body_quality = (close_price - low) / bar_range * 100 if bar_range > 0 else 50
        else:
            body_quality = (high - close_price) / bar_range * 100 if bar_range > 0 else 50

        results.append({
            "date_idx": i,
            "Signal": f"GAP_{direction}",
            "Gap Type": gap_type,
            "Signal Date": date.strftime("%Y-%m-%d"),
            "Gap %": round(gap_pct, 2),
            "Prev Close": round(prev_close, 2),
            "Open": round(open_price, 2),
            "Close": round(close_price, 2),
            "Vol vs Avg": round(vol_r, 2),
            "Body Quality %": round(body_quality, 1),
            "Gap Fill %": round(fill_pct, 1),
            "Stop": stop,
            "Target": target,
            "R/R": round(2.0, 2),
            "RSI": round(df["rsi14"].iloc[i], 1) if pd.notna(df["rsi14"].iloc[i]) else None,
            "EMA50": "ABOVE" if close_price > df["ema50"].iloc[i] else "BELOW",
        })

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    return out.drop(columns=["date_idx"])


def _extract_ticker_df(raw_all, ticker: str, n_tickers: int) -> pd.DataFrame:
    """Slice a single ticker out of a batch yf.download result."""
    try:
        if n_tickers == 1:
            df = raw_all.copy()
        else:
            df = raw_all[ticker].copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def run_gap_screener(
    tickers: Optional[List[str]] = None,
    min_gap_pct: float = 0.5,
    vol_mult: float = 1.5,
    recent_bars: int = 5,
    gap_direction: str = "ALL",
    include_common: bool = False,
    lookback_days: int = 60,
) -> pd.DataFrame:
    if tickers is None:
        tickers = DEFAULT_TICKERS

    end = datetime.today()
    start = end - timedelta(days=lookback_days + 60)

    # Batch download all tickers in one request (yfinance threads internally)
    raw_all = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )

    results = []
    for ticker in tickers:
        try:
            df = _extract_ticker_df(raw_all, ticker, len(tickers))
            if df.empty or len(df) < 30:
                continue
            sigs = detect_gaps(
                df,
                min_gap_pct=min_gap_pct,
                vol_mult=vol_mult,
                recent_bars=recent_bars,
                gap_direction=gap_direction,
                include_common=include_common,
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
    return out.sort_values(["Bars Ago", "Gap %"], ascending=[True, False])


def run_live_gap_screener(
    tickers: Optional[List[str]] = None,
    min_gap_pct: float = 1.0,
    vol_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Lightweight live screener: downloads only the last 25 days for all tickers
    in one batch call, then returns today's open vs prev close gap.
    Designed for the 15-min auto-refresh Live tab.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    raw_all = yf.download(
        tickers,
        period="25d",
        interval="1d",
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )

    rows = []
    for ticker in tickers:
        try:
            df = _extract_ticker_df(raw_all, ticker, len(tickers))
            if df.empty or len(df) < 3:
                continue

            prev_close = float(df["Close"].iloc[-2])
            today_open = float(df["Open"].iloc[-1])
            today_close = float(df["Close"].iloc[-1])
            today_vol = float(df["Volume"].iloc[-1])
            avg_vol = float(df["Volume"].iloc[-22:-1].mean()) if len(df) >= 22 else float(df["Volume"].mean())

            gap_pct = (today_open - prev_close) / prev_close * 100
            if abs(gap_pct) < min_gap_pct:
                continue

            vol_ratio = today_vol / avg_vol if avg_vol > 0 else 0
            if vol_ratio < vol_mult:
                continue

            # Intraday momentum: how much of the gap held (positive) or faded (negative)
            gap_size = abs(today_open - prev_close)
            if gap_pct > 0:
                momentum_pct = (today_close - today_open) / gap_size * 100 if gap_size > 0 else 0
            else:
                momentum_pct = (today_open - today_close) / gap_size * 100 if gap_size > 0 else 0

            rows.append({
                "Ticker": ticker,
                "Direction": "GAP UP ▲" if gap_pct > 0 else "GAP DOWN ▼",
                "Gap %": round(gap_pct, 2),
                "Prev Close": round(prev_close, 2),
                "Open": round(today_open, 2),
                "Current": round(today_close, 2),
                "Vol Ratio": round(vol_ratio, 2),
                "Momentum %": round(momentum_pct, 1),
                "Stop": round(prev_close, 2),
                "Target": round(
                    today_close + 2 * abs(gap_pct) / 100 * prev_close, 2
                ) if gap_pct > 0 else round(
                    today_close - 2 * abs(gap_pct) / 100 * prev_close, 2
                ),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    return df_out.sort_values("Gap %", ascending=False)
