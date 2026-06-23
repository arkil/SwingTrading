"""
IBD Scanner — Near Buy Zone Detection
=======================================
Identifies stocks near IBD-style buy zones using:
  • In Base          — consolidating within 15% of 52W high for 5+ weeks
  • Pullback         — pulling back to 10-week MA with light volume
  • Wedge Tightening — narrowing weekly price ranges near highs
  • Short Stroke     — tight week after a shakeout
  • Xing 10W         — price just crossed above 10-week MA
  • HTF              — High Tight Flag (100%+ gain then flag)

Cross-references user-supplied IBD list rankings and computes a
composite "My Points" score: RS Rating + bonuses for patterns + list memberships.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import warnings
warnings.filterwarnings("ignore")

_yf_lock = Lock()


def _normalize_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone from DatetimeIndex so tz-aware and tz-naive frames align."""
    if df.empty:
        return df
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    # Keep only the date part so daily granularity matches across sources
    df.index = pd.to_datetime(df.index.date)
    return df


# ── Sector label maps ─────────────────────────────────────────────────────────

_SECTOR_SHORT: Dict[str, str] = {
    "Technology":             "TECH",
    "Consumer Cyclical":      "CONSUMER",
    "Communication Services": "MEDIA",
    "Healthcare":             "MEDICAL",
    "Financial Services":     "FINANCE",
    "Industrials":            "INDUSTRIAL",
    "Basic Materials":        "MINING",
    "Energy":                 "ENERGY",
    "Consumer Defensive":     "CONSUMER DEF",
    "Real Estate":            "RLEST",
    "Utilities":              "UTIL",
}


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_stock_data(ticker: str, period_days: int = 420) -> Tuple[pd.DataFrame, dict]:
    """Return (daily_ohlcv, info_dict). Empty df on failure."""
    try:
        with _yf_lock:
            t = yf.Ticker(ticker)
            end   = datetime.today()
            start = end - timedelta(days=period_days)
            hist  = t.history(start=start, end=end, interval="1d", auto_adjust=True)

        if hist is None or hist.empty:
            return pd.DataFrame(), {}
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist = hist[["Open", "High", "Low", "Close", "Volume"]].dropna()
        hist = _normalize_idx(hist)

        try:
            fi = t.fast_info
            info_dict = {
                "longName": getattr(fi, "long_name", ticker) or ticker,
                "sector":   getattr(fi, "sector", "")        or "",
                "industry": getattr(fi, "industry", "")      or "",
            }
        except Exception:
            info_dict = {"longName": ticker, "sector": "", "industry": ""}

        return hist, info_dict
    except Exception:
        return pd.DataFrame(), {}


def _weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly (Friday close)."""
    return daily.resample("W-FRI").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    ).dropna()


# ── RS Rating ────────────────────────────────────────────────────────────────

def _weighted_return(returns: pd.Series) -> float:
    """IBD-style weighted 12-month return: 40% last q + 20% each prior q."""
    n = len(returns)
    if n == 0:
        return 0.0
    q = max(1, n // 4)
    r4 = float((1 + returns.iloc[-q:]).prod() - 1)
    r3 = float((1 + returns.iloc[max(0, -2*q):-q]).prod() - 1)
    r2 = float((1 + returns.iloc[max(0, -3*q):max(0, -2*q)]).prod() - 1)
    r1 = float((1 + returns.iloc[:max(0, -3*q)]).prod() - 1)
    return 0.40*r4 + 0.20*r3 + 0.20*r2 + 0.20*r1


def compute_rs_rating(tk_rets: pd.Series, spy_rets: pd.Series) -> float:
    """Relative Strength Rating 1–99 vs SPY (50 = market-parity)."""
    if len(tk_rets) < 10 or len(spy_rets) < 10:
        return 50.0
    raw = _weighted_return(tk_rets) - _weighted_return(spy_rets)
    # ±50 pp outperformance maps to ±49 on the scale; clamp to [1, 99]
    return float(max(1.0, min(99.0, 50.0 + raw * 98.0)))


# ── Pattern detectors (weekly data) ──────────────────────────────────────────

def detect_in_base(weekly: pd.DataFrame) -> bool:
    """Stock consolidating within 15% of 52-week high for ≥ 5 weeks."""
    if len(weekly) < 10:
        return False
    high_52w = float(weekly.tail(52)["High"].max())
    curr     = float(weekly["Close"].iloc[-1])
    if curr < high_52w * 0.85:
        return False
    last5_lows = weekly["Low"].iloc[-5:]
    return bool((last5_lows >= high_52w * 0.80).all())


def detect_pullback_to_10w(weekly: pd.DataFrame) -> bool:
    """Price within 5% of 10-week SMA, having been above it recently."""
    if len(weekly) < 15:
        return False
    w = weekly.copy()
    w["sma10"] = w["Close"].rolling(10).mean()
    w = w.dropna(subset=["sma10"])
    if len(w) < 6:
        return False
    curr  = float(w["Close"].iloc[-1])
    sma10 = float(w["sma10"].iloc[-1])
    pct   = abs(curr - sma10) / sma10
    was_above = bool((w["Close"].iloc[-6:-1] > w["sma10"].iloc[-6:-1]).any())
    return bool(pct <= 0.05 and was_above)


def detect_wedge_tightening(weekly: pd.DataFrame) -> bool:
    """3+ consecutive weeks with narrowing range near 52W highs."""
    if len(weekly) < 8:
        return False
    high_52w = float(weekly.tail(52)["High"].max())
    if float(weekly["Close"].iloc[-1]) < high_52w * 0.85:
        return False
    ranges = (weekly["High"] - weekly["Low"]).tail(5).values
    tightening = all(ranges[i] <= ranges[i-1] * 1.05 for i in range(1, 5))
    shrunk = ranges[-1] < ranges[0] * 0.70
    return bool(tightening and shrunk)


def detect_short_stroke(weekly: pd.DataFrame) -> bool:
    """Tight week (< 60% of prior range) closing in upper half, near highs."""
    if len(weekly) < 6:
        return False
    high_52w = float(weekly.tail(52)["High"].max())
    if float(weekly["Close"].iloc[-1]) < high_52w * 0.87:
        return False
    w       = weekly.tail(3)
    hi, lo  = float(w["High"].iloc[-1]), float(w["Low"].iloc[-1])
    cl      = float(w["Close"].iloc[-1])
    prev_r  = float(w["High"].iloc[-2] - w["Low"].iloc[-2])
    curr_r  = hi - lo
    close_pct = (cl - lo) / curr_r if curr_r > 0 else 0.5
    return bool(curr_r < prev_r * 0.60 and close_pct > 0.50)


def detect_xing_10w(weekly: pd.DataFrame) -> bool:
    """Price crossed above 10-week SMA in the last 1 week."""
    if len(weekly) < 15:
        return False
    w = weekly.copy()
    w["sma10"] = w["Close"].rolling(10).mean()
    w = w.dropna(subset=["sma10"])
    if len(w) < 3:
        return False
    curr_above = float(w["Close"].iloc[-1]) > float(w["sma10"].iloc[-1])
    prev_below = float(w["Close"].iloc[-2]) < float(w["sma10"].iloc[-2])
    return bool(curr_above and prev_below)


def detect_htf(daily: pd.DataFrame) -> bool:
    """High Tight Flag: ≥ 100% gain in 8 weeks, now flagging 10-25% below peak."""
    if len(daily) < 60:
        return False
    window = daily.tail(40)
    peak   = float(window["High"].max())
    trough = float(window["Low"].min())
    curr   = float(daily["Close"].iloc[-1])
    gained = (peak - trough) / max(trough, 0.01) >= 1.0
    pct_from_peak = (peak - curr) / max(peak, 0.01)
    return bool(gained and 0.05 <= pct_from_peak <= 0.25)


def detect_rs_line_new_high(daily: pd.DataFrame, spy_daily: pd.DataFrame) -> bool:
    """RS line (stock / SPY) at or within 0.5% of its 52-week high."""
    try:
        idx = daily.index.intersection(spy_daily.index)
        if len(idx) < 50:
            return False
        rs = daily.loc[idx, "Close"] / spy_daily.loc[idx, "Close"]
        return bool(rs.iloc[-1] >= rs.iloc[-252:].max() * 0.995)
    except Exception:
        return False


# ── Volume ────────────────────────────────────────────────────────────────────

def _avg_vol_50d_k(daily: pd.DataFrame) -> int:
    if len(daily) < 5:
        return 0
    return int(daily["Volume"].tail(50).mean() / 1000)


# ── Composite score ──────────────────────────────────────────────────────────

def _my_points(
    rs: float, rs_nh: bool,
    in_base: bool, pullback: bool, wt: bool,
    short_stroke: bool, xing_10w: bool, htf: bool,
    list_count: int,
) -> int:
    score = rs
    if rs_nh:        score += 5
    if in_base:      score += 3
    if pullback:     score += 4
    if wt:           score += 3
    if short_stroke: score += 3
    if xing_10w:     score += 4
    if htf:          score += 5
    score += list_count * 2
    return int(round(score))


# ── Per-ticker analysis ──────────────────────────────────────────────────────

def analyze_stock(
    ticker: str,
    spy_daily: pd.DataFrame,
    ibd_lists: Dict[str, List[str]],
) -> Optional[Dict]:
    daily, info = _fetch_stock_data(ticker)
    if daily.empty or len(daily) < 60:
        return None

    wk = _weekly(daily)
    if wk.empty or len(wk) < 15:
        return None

    # RS rating
    if not spy_daily.empty:
        idx = daily.index.intersection(spy_daily.index)
        if len(idx) > 20:
            n = min(len(idx), 252)
            tk_r  = daily.loc[idx, "Close"].pct_change().dropna().iloc[-n:]
            spy_r = spy_daily.loc[idx, "Close"].pct_change().dropna().iloc[-n:]
            mn    = min(len(tk_r), len(spy_r))
            rs    = compute_rs_rating(tk_r.iloc[-mn:], spy_r.iloc[-mn:])
        else:
            rs = 50.0
    else:
        rs = 50.0

    rs_nh        = detect_rs_line_new_high(daily, spy_daily)
    in_base      = detect_in_base(wk)
    pullback     = detect_pullback_to_10w(wk)
    wt           = detect_wedge_tightening(wk)
    short_stroke = detect_short_stroke(wk)
    xing_10w     = detect_xing_10w(wk)
    htf          = detect_htf(daily)

    list_ranks: Dict[str, Optional[int]] = {}
    list_count = 0
    for lname, ltickers in ibd_lists.items():
        upper = [t.strip().upper() for t in ltickers]
        if ticker.upper() in upper:
            list_ranks[lname] = upper.index(ticker.upper()) + 1
            list_count += 1
        else:
            list_ranks[lname] = None

    sector   = _SECTOR_SHORT.get(info.get("sector", ""), info.get("sector", ""))
    industry = info.get("industry", "")
    curr     = float(daily["Close"].iloc[-1])

    row: Dict = {
        "Symbol":        ticker.upper(),
        "Name":          info.get("longName", ticker),
        "Current Price": round(curr, 2),
        "RS Line NH":    "Yes" if rs_nh else "No",
        "Industry":      industry,
        "Sector":        sector,
        "50D Avg Vol K": _avg_vol_50d_k(daily),
        "RS Rating":     int(round(rs)),
        "In_Base":       1 if in_base      else None,
        "Pullback":      1 if pullback      else None,
        "WT":            1 if wt            else None,
        "Short Stroke":  1 if short_stroke  else None,
        "Xing 10W":      1 if xing_10w      else None,
        "HTF":           1 if htf           else None,
        "My Points":     _my_points(rs, rs_nh, in_base, pullback, wt,
                                    short_stroke, xing_10w, htf, list_count),
        "# Lists":       list_count,
    }
    row.update(list_ranks)
    return row


# ── Main entry point ─────────────────────────────────────────────────────────

def run_ibd_scanner(
    tickers: List[str],
    ibd_lists: Dict[str, List[str]] = None,
    min_rs: int = 60,
    require_pattern: bool = False,
    max_workers: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan tickers for IBD-style near-buy-zone setups.

    Args:
        tickers:        List of ticker symbols.
        ibd_lists:      Dict of {list_name: [tickers in order]}.
        min_rs:         Minimum RS Rating to include (0-99).
        require_pattern: Only include stocks with ≥ 1 pattern flag.
        max_workers:    Thread pool size.

    Returns:
        (df_main, df_lists) — two DataFrames matching the two tables in the image.
    """
    if ibd_lists is None:
        ibd_lists = {}

    # SPY benchmark
    try:
        with _yf_lock:
            spy_raw = yf.download(
                "SPY", period="2y", interval="1d",
                progress=False, auto_adjust=True,
            )
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = spy_raw.columns.get_level_values(0)
        spy_daily = _normalize_idx(spy_raw[["Close"]].dropna())
    except Exception:
        spy_daily = pd.DataFrame()

    tickers_clean = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))

    rows: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(analyze_stock, tk, spy_daily, ibd_lists): tk for tk in tickers_clean}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r is not None:
                    rows.append(r)
            except Exception:
                pass

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── Filters ────────────────────────────────────────────────────────────────
    df = df[df["RS Rating"] >= min_rs].copy()
    if require_pattern:
        pat_cols = ["In_Base", "Pullback", "WT", "Short Stroke", "Xing 10W", "HTF"]
        mask = df[[c for c in pat_cols if c in df.columns]].notna().any(axis=1)
        df = df[mask].copy()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df.sort_values("My Points", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Table 1: Near Buy Zone (main) ─────────────────────────────────────────
    t1_cols = [
        "Symbol", "Name", "Current Price", "RS Line NH",
        "Industry", "Sector", "50D Avg Vol K", "My Points",
        "In_Base", "Pullback", "WT", "Short Stroke", "Xing 10W", "HTF",
    ]
    df_main = df[[c for c in t1_cols if c in df.columns]].copy()

    # ── Table 2: List membership (cont) ──────────────────────────────────────
    list_cols = list(ibd_lists.keys())
    t2_rows = []
    for i, row in df.iterrows():
        r: Dict = {"No": int(i) + 1, "Symbol": row["Symbol"]}
        for lc in list_cols:
            r[lc] = row.get(lc)
        r["# Lists"] = int(row.get("# Lists", 0))
        t2_rows.append(r)
    df_lists = pd.DataFrame(t2_rows) if t2_rows else pd.DataFrame()

    return df_main, df_lists
