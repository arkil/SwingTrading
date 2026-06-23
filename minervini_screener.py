"""
Minervini SEPA Screener — Specific Entry Point Analysis
========================================================
Based on Mark Minervini's strategy from:
  • "Trade Like a Stock Market Wizard" (2013)
  • "Think & Trade Like a Champion" (2017)
  • US Investing Championship wins (1997, 2021)

══════════════════════════════════════════════════════
MINERVINI'S WINNING STRATEGY — SEPA
══════════════════════════════════════════════════════

1. TREND TEMPLATE (Stage 2 uptrend — ALL 8 must pass)
   ────────────────────────────────────────────────────
   T1. Price > 150-day SMA  (medium-term trend up)
   T2. Price > 200-day SMA  (long-term trend up)
   T3. 150-day SMA > 200-day SMA  (medium > long)
   T4. 200-day SMA trending UP for ≥ 1 month (slope > 0)
   T5. 50-day SMA > 150-day SMA AND 50-day > 200-day SMA
   T6. Price > 50-day SMA  (short-term trend up)
   T7. Price ≥ 30% above 52-week low  (escaped the basement)
   T8. Price within 25% of 52-week high  (near the top)

2. RELATIVE STRENGTH (RS)
   ────────────────────────────────────────────────────
   RS Rating ≥ 70 vs the S&P 500 (stock outperforming the market).
   Minervini only buys leaders, not laggards.

3. VCP — VOLATILITY CONTRACTION PATTERN
   ────────────────────────────────────────────────────
   After a strong uptrend, price forms a base with:
   • 2–6 contractions, each shallower than the last
   • Volume drying up during each contraction (supply drying up)
   • Tight price action (< 10% range) near the highs = pivot forming
   Classic counts: 3T (3 contractions), 4T, 2T

4. PIVOT / ENTRY
   ────────────────────────────────────────────────────
   • Buy on breakout above the pivot (top of the base / tightest point)
   • Breakout volume ≥ 40% above 50-day average volume
   • "Early entry" available on a tight close near highs w/ dry-up volume

5. POSITION SIZING & RISK
   ────────────────────────────────────────────────────
   • Maximum loss per trade: 7–10% below pivot (hard stop)
   • Risk per trade: 0.5–2% of total portfolio
   • Profit target: minimum 3:1 reward-to-risk
   • Partial sell at +10%, +20%, let remainder run

6. WHAT TO AVOID
   ────────────────────────────────────────────────────
   • Stage 1 (basing), Stage 3 (topping), Stage 4 (downtrend) stocks
   • Wide & loose bases (lack of tightness = weak institutional support)
   • Low RS stocks (laggards rarely become leaders)
   • Chasing extended stocks (> 5% past the pivot)

══════════════════════════════════════════════════════

Output columns
──────────────
  Ticker         | RS Rating (1-99) | Stage | Trend Score (0-8)
  VCP Detected   | Contractions     | Pivot | % from Pivot
  52W High       | 52W Low          | % from 52W H | % from 52W L
  SMA50/150/200  | Vol vs Avg50     | ADX   | ATR%
  Entry          | Stop             | Target (3:1)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import warnings
warnings.filterwarnings("ignore")

# yfinance has a known concurrency issue — serialize all downloads
_yf_lock = Lock()

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from livermore_pivotal_screener import get_universe, DEFAULT_TICKERS


# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=n, adjust=False).mean()

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr  = _atr(df, 1)
    up  = df["High"].diff()
    dn  = -df["Low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr_s  = pd.Series(tr).ewm(span=n, adjust=False).mean()
    pdm_s = pd.Series(pdm, index=df.index).ewm(span=n, adjust=False).mean()
    ndm_s = pd.Series(ndm, index=df.index).ewm(span=n, adjust=False).mean()
    pdi   = 100 * pdm_s / tr_s.replace(0, np.nan)
    ndi   = 100 * ndm_s / tr_s.replace(0, np.nan)
    dx    = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan))
    return dx.ewm(span=n, adjust=False).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d  = s.diff()
    up = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ─────────────────────────────────────────────────────────────────────────────
# Data fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(ticker: str, days: int = 400) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=days + 60)
    try:
        with _yf_lock:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


_spy_cache: Optional[pd.DataFrame] = None

def _get_spy(days: int = 400) -> pd.DataFrame:
    global _spy_cache
    if _spy_cache is not None and len(_spy_cache) > 0:
        return _spy_cache
    _spy_cache = _fetch("SPY", days)
    return _spy_cache


# ─────────────────────────────────────────────────────────────────────────────
# Trend Template (Minervini's 8 conditions)
# ─────────────────────────────────────────────────────────────────────────────

def check_trend_template(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns dict with individual condition results and total score (0-8).
    All 8 must be True for a Stage 2 stock.
    """
    if len(df) < 210:
        return {"score": 0, "conditions": {}, "stage": "Insufficient data"}

    price   = df["Close"].iloc[-1]
    sma50   = _sma(df["Close"], 50).iloc[-1]
    sma150  = _sma(df["Close"], 150).iloc[-1]
    sma200  = _sma(df["Close"], 200).iloc[-1]

    # 200-day SMA slope over 1 month (21 trading days)
    sma200_series = _sma(df["Close"], 200)
    sma200_1m_ago = sma200_series.iloc[-22] if len(sma200_series) >= 22 else np.nan
    sma200_slope_up = bool(sma200 > sma200_1m_ago) if not np.isnan(sma200_1m_ago) else False

    high52w = df["High"].rolling(252).max().iloc[-1]
    low52w  = df["Low"].rolling(252).min().iloc[-1]

    conds = {
        "T1_price_above_sma150":    bool(price > sma150),
        "T2_price_above_sma200":    bool(price > sma200),
        "T3_sma150_above_sma200":   bool(sma150 > sma200),
        "T4_sma200_trending_up":    sma200_slope_up,
        "T5_sma50_above_sma150_200":bool(sma50 > sma150 and sma50 > sma200),
        "T6_price_above_sma50":     bool(price > sma50),
        "T7_30pct_above_52w_low":   bool(price >= low52w * 1.30) if not np.isnan(low52w) else False,
        "T8_within_25pct_52w_high": bool(price >= high52w * 0.75) if not np.isnan(high52w) else False,
    }
    score = sum(conds.values())

    # Stage classification
    if score == 8:
        stage = "Stage 2 ✅"
    elif score >= 6:
        stage = "Stage 2 (partial)"
    elif bool(price > sma150 or price > sma200):
        stage = "Stage 1 (basing)"
    elif bool(price < sma150 and price < sma200 and sma150 < sma200):
        stage = "Stage 4 (decline)"
    else:
        stage = "Stage 3 (topping)"

    return {
        "score":      score,
        "conditions": conds,
        "stage":      stage,
        "sma50":      round(sma50,  2) if not np.isnan(sma50)  else None,
        "sma150":     round(sma150, 2) if not np.isnan(sma150) else None,
        "sma200":     round(sma200, 2) if not np.isnan(sma200) else None,
        "high52w":    round(high52w, 2) if not np.isnan(high52w) else None,
        "low52w":     round(low52w,  2) if not np.isnan(low52w)  else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Relative Strength Rating (vs SPY, percentile 1-99)
# ─────────────────────────────────────────────────────────────────────────────

def calc_rs_rating(df: pd.DataFrame, spy: pd.DataFrame) -> float:
    """
    Minervini uses IBD's RS Rating (1-99).
    We approximate: weighted return of stock vs SPY over 63/126/189/252 days.
    IBD formula weights the most recent quarter 2×:
        RS = (Q4 perf × 2 + Q3 perf + Q2 perf + Q1 perf) / 5
    Returns 0-100 raw score (compare across universe for percentile).
    """
    periods = [63, 126, 189, 252]
    weights = [2, 1, 1, 1]   # Q4 double-weighted per IBD

    if len(df) < 260 or len(spy) < 260:
        return 0.0

    def _perf(frame: pd.DataFrame, p: int) -> float:
        if len(frame) < p + 1:
            return 0.0
        c = frame["Close"].iloc[-1]
        p_ago = frame["Close"].iloc[-(p + 1)]
        return (c / p_ago - 1) * 100 if p_ago else 0.0

    stock_score = sum(w * _perf(df, p)  for p, w in zip(periods, weights)) / sum(weights)
    spy_score   = sum(w * _perf(spy, p) for p, w in zip(periods, weights)) / sum(weights)
    return round(stock_score - spy_score, 2)   # relative outperformance %; percentile done at universe level


# ─────────────────────────────────────────────────────────────────────────────
# VCP — Volatility Contraction Pattern
# ─────────────────────────────────────────────────────────────────────────────

def detect_vcp(df: pd.DataFrame, lookback: int = 60) -> Dict[str, Any]:
    """
    Simplified VCP detection:
      1. Look at the last `lookback` bars for a consolidation base.
      2. Identify swing highs/lows within the base.
      3. Count contractions (each swing high-to-low range < previous).
      4. Check volume dry-up in latest contraction vs base average.
      5. Determine pivot = highest high in the tightest contraction.

    Returns dict:
        detected      : bool
        contractions  : int   number of tightening swings found
        pivot         : float breakout level
        depth_pct     : float tightest contraction depth %
        vol_dryup     : bool  volume declining during contractions
        base_length   : int   bars in current base
    """
    if len(df) < lookback + 20:
        return {"detected": False, "contractions": 0, "pivot": None,
                "depth_pct": None, "vol_dryup": False, "base_length": 0}

    base = df.iloc[-lookback:].copy()

    # Find swing highs: local max over 5-bar window
    highs = base["High"].values
    lows  = base["Low"].values
    vols  = base["Volume"].values
    n     = len(base)

    swing_high_idx = []
    for i in range(5, n - 5):
        if highs[i] == max(highs[i-5:i+6]):
            swing_high_idx.append(i)

    swing_low_idx = []
    for i in range(5, n - 5):
        if lows[i] == min(lows[i-5:i+6]):
            swing_low_idx.append(i)

    if len(swing_high_idx) < 2 or len(swing_low_idx) < 2:
        return {"detected": False, "contractions": 0, "pivot": None,
                "depth_pct": None, "vol_dryup": False, "base_length": lookback}

    # Pair swing highs with their following swing lows to get contractions
    contractions = []
    for sh_i in swing_high_idx:
        # Find first swing low after this swing high
        following_lows = [sl for sl in swing_low_idx if sl > sh_i]
        if not following_lows:
            continue
        sl_i = following_lows[0]
        depth = (highs[sh_i] - lows[sl_i]) / highs[sh_i] * 100
        avg_vol = float(np.mean(vols[sh_i:sl_i + 1])) if sl_i > sh_i else float(vols[sh_i])
        contractions.append({
            "high_idx":  sh_i,
            "low_idx":   sl_i,
            "high_val":  highs[sh_i],
            "low_val":   lows[sl_i],
            "depth_pct": depth,
            "avg_vol":   avg_vol,
        })

    if len(contractions) < 2:
        return {"detected": False, "contractions": len(contractions), "pivot": None,
                "depth_pct": None, "vol_dryup": False, "base_length": lookback}

    # Check for tightening: each contraction shallower than the previous
    tightening_count = 0
    for i in range(1, len(contractions)):
        if contractions[i]["depth_pct"] < contractions[i-1]["depth_pct"]:
            tightening_count += 1

    # Volume dry-up: last contraction avg volume < first contraction avg volume
    vol_dryup = contractions[-1]["avg_vol"] < contractions[0]["avg_vol"] * 0.85

    # Pivot = high of the last (tightest) contraction
    pivot     = contractions[-1]["high_val"]
    depth_pct = contractions[-1]["depth_pct"]

    # Detect VCP if at least 2 tightening contractions and volume drying
    detected = tightening_count >= 1 and depth_pct < 20.0 and vol_dryup

    return {
        "detected":     detected,
        "contractions": tightening_count + 1,
        "pivot":        round(float(pivot), 2),
        "depth_pct":    round(float(depth_pct), 2),
        "vol_dryup":    vol_dryup,
        "base_length":  lookback,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Breakout quality check
# ─────────────────────────────────────────────────────────────────────────────

def check_breakout(df: pd.DataFrame, pivot: Optional[float]) -> Dict[str, Any]:
    """
    Check if the stock has broken out (or is near breakout) from the pivot.
    Minervini: buy on the day price closes above the pivot on big volume.
    """
    if pivot is None or df.empty:
        return {"near_pivot": False, "broken_out": False,
                "pct_from_pivot": None, "vol_vs_avg50": None}

    price     = df["Close"].iloc[-1]
    vol_today = df["Volume"].iloc[-1]
    vol_avg50 = _sma(df["Volume"], 50).iloc[-1]

    pct_from_pivot = (price / pivot - 1) * 100
    vol_vs_avg50   = vol_today / vol_avg50 if vol_avg50 else 0

    # "Near pivot" = within 5% below pivot (buyable soon)
    near_pivot  = bool(-5.0 <= pct_from_pivot <= 0)
    # "Broken out" = closed above pivot, within 5% (not extended)
    broken_out  = bool(0 < pct_from_pivot <= 5.0 and vol_vs_avg50 >= 1.40)

    return {
        "near_pivot":     near_pivot,
        "broken_out":     broken_out,
        "pct_from_pivot": round(pct_from_pivot, 2),
        "vol_vs_avg50":   round(vol_vs_avg50, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full analysis for one ticker
# ─────────────────────────────────────────────────────────────────────────────

def analyze_ticker(ticker: str, spy: pd.DataFrame) -> Optional[Dict[str, Any]]:
    df = _fetch(ticker)
    if df is None or len(df) < 60:
        return None

    price = df["Close"].iloc[-1]

    trend   = check_trend_template(df)
    rs_raw  = calc_rs_rating(df, spy)
    vcp     = detect_vcp(df, lookback=60)
    bo      = check_breakout(df, vcp.get("pivot"))

    atr14   = _atr(df, 14).iloc[-1]
    adx14   = _adx(df, 14).iloc[-1]
    rsi14   = _rsi(df["Close"], 14).iloc[-1]
    vol_avg50 = _sma(df["Volume"], 50).iloc[-1]
    vol_today = df["Volume"].iloc[-1]

    high52w = trend.get("high52w") or df["High"].rolling(252).max().iloc[-1]
    low52w  = trend.get("low52w")  or df["Low"].rolling(252).min().iloc[-1]

    pct_from_52h = round((price / high52w - 1) * 100, 2) if high52w else None
    pct_from_52l = round((price / low52w  - 1) * 100, 2) if low52w  else None

    pivot = vcp.get("pivot")

    # Risk / entry / target
    stop   = round(pivot * 0.92, 2) if pivot else round(price * 0.92, 2)   # 8% stop
    entry  = round(pivot * 1.005, 2) if pivot else round(price, 2)          # just above pivot
    risk   = entry - stop
    target = round(entry + risk * 3, 2)                                      # 3:1 R/R

    return {
        "Ticker":           ticker,
        "Price":            round(float(price), 2),
        "Stage":            trend["stage"],
        "Trend Score":      trend["score"],
        "RS (raw)":         rs_raw,          # percentile computed after full scan
        "VCP":              "✅" if vcp["detected"] else "—",
        "Contractions":     vcp["contractions"],
        "VCP Depth %":      vcp.get("depth_pct"),
        "Vol Dry-Up":       "✅" if vcp.get("vol_dryup") else "—",
        "Pivot":            pivot,
        "% from Pivot":     bo.get("pct_from_pivot"),
        "Near Pivot":       bo["near_pivot"],
        "Broken Out":       bo["broken_out"],
        "Vol vs Avg50":     bo.get("vol_vs_avg50"),
        "52W High":         high52w,
        "52W Low":          low52w,
        "% from 52W H":     pct_from_52h,
        "% from 52W L":     pct_from_52l,
        "SMA50":            trend.get("sma50"),
        "SMA150":           trend.get("sma150"),
        "SMA200":           trend.get("sma200"),
        "ADX":              round(float(adx14), 1) if not np.isnan(adx14) else None,
        "RSI":              round(float(rsi14), 1) if not np.isnan(rsi14) else None,
        "ATR%":             round(float(atr14 / price * 100), 2) if price else None,
        "Entry":            entry,
        "Stop":             stop,
        "Target (3:1)":     target,
        # raw conditions for detail view
        "_conditions":      trend["conditions"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run screener
# ─────────────────────────────────────────────────────────────────────────────

def run_minervini_screener(
    tickers: List[str],
    min_trend_score: int = 7,
    require_vcp: bool = False,
    min_rs_pct: float = 70.0,
    max_workers: int = 12,
) -> pd.DataFrame:
    """
    Scan universe and return stocks passing Minervini's SEPA criteria.

    Args:
        tickers         : list of ticker symbols
        min_trend_score : minimum trend template score (0-8); 7 or 8 = true Stage 2
        require_vcp     : if True only return stocks with detected VCP
        min_rs_pct      : RS percentile cutoff (0-100); Minervini uses ≥70
        max_workers     : concurrent download threads

    Returns:
        DataFrame sorted by RS Rating descending.
    """
    spy = _get_spy()

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(analyze_ticker, t, spy): t for t in tickers}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Convert RS raw scores to percentile ranking across this universe
    if "RS (raw)" in df.columns and len(df) > 1:
        from scipy.stats import percentileofscore
        df["RS Rating"] = df["RS (raw)"].apply(
            lambda x: round(percentileofscore(df["RS (raw)"].tolist(), x), 0)
        ).astype(int)
    else:
        df["RS Rating"] = 50

    # Apply filters
    df = df[df["Trend Score"] >= min_trend_score]
    if require_vcp:
        df = df[df["VCP"] == "✅"]
    df = df[df["RS Rating"] >= min_rs_pct]

    # Sort: broken-out first, then near-pivot, then by RS Rating
    df["_sort"] = df.apply(
        lambda r: 0 if r["Broken Out"] else (1 if r["Near Pivot"] else 2), axis=1
    )
    df = df.sort_values(["_sort", "RS Rating", "Trend Score"], ascending=[True, False, False])
    df = df.drop(columns=["_sort", "RS (raw)", "_conditions"], errors="ignore")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Minervini SEPA Screener")
    parser.add_argument("--universe", default="nasdaq100",
                        choices=["default", "sp500", "nasdaq100", "both", "watchlist"])
    parser.add_argument("--min-score",  type=int,   default=7)
    parser.add_argument("--min-rs",     type=float, default=70.0)
    parser.add_argument("--require-vcp", action="store_true")
    args = parser.parse_args()

    print(f"\nMinervini SEPA Scan — {args.universe} universe\n{'='*50}")
    tickers = get_universe(args.universe)
    print(f"Scanning {len(tickers)} tickers…\n")

    result_df = run_minervini_screener(
        tickers,
        min_trend_score=args.min_score,
        require_vcp=args.require_vcp,
        min_rs_pct=args.min_rs,
    )

    if result_df.empty:
        print("No stocks passed the SEPA criteria.")
    else:
        COLS = ["Ticker", "RS Rating", "Stage", "Trend Score",
                "VCP", "Contractions", "VCP Depth %", "Pivot",
                "% from Pivot", "% from 52W H", "Entry", "Stop", "Target (3:1)"]
        print(result_df[[c for c in COLS if c in result_df.columns]].to_string(index=False))
        print(f"\n{len(result_df)} stocks passed SEPA filters.")
