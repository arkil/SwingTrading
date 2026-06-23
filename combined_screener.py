"""
Combined Strategy Scanner — Straight Buy Signals
=================================================
Runs every stock through a multi-factor scoring engine that combines:

  1. TREND      — Is the stock in a healthy uptrend?
  2. MOMENTUM   — Is momentum accelerating?
  3. TRIGGER    — Has a concrete entry event fired?
  4. VOLUME     — Is institutional money participating?
  5. RISK       — Is the setup clean enough to trade?

Each dimension contributes points. The total score determines the rating:

  Score ≥ 7  →  ⚡ STRONG BUY  (high conviction, multiple confirmations)
  Score 5-6  →  ✅ BUY         (solid setup, act on a pullback or confirmation)
  Score 3-4  →  👀 WATCH       (developing, not ready yet)
  Score < 3  →  filtered out

═══════════════════════════════════════════════════════════════════════════════
SCORING MATRIX
═══════════════════════════════════════════════════════════════════════════════

TREND (max 3 pts)
  +1  Price above EMA-50          (short-term uptrend)
  +1  EMA-50 above EMA-200        (long-term uptrend confirmed)
  +1  Minervini Trend Score ≥ 6   (Stage 2 uptrend — all MAs stacked)

MOMENTUM (max 3 pts)
  +1  RSI(14) between 45–72       (momentum present, not overbought)
  +1  MACD histogram positive or just flipped positive (last 3 bars)
  +1  Relative Strength vs SPY ≥ 60th percentile (stock leads the market)

ENTRY TRIGGER (max 3 pts — must have ≥ 1 to generate a signal)
  +1  EMA-9 crossed above EMA-21 within last 5 bars
  +1  Breakout above 20-day high with volume surge
  +1  NR7 breakout (volatility contraction → expansion)
  +1  Bollinger squeeze breakout
  +1  Gap-up continuation/breakaway in last 3 bars
  (capped at 3)

VOLUME (max 2 pts)
  +1  Volume > 1.5× 20-day average  (unusual interest)
  +2  Volume > 3.0× 20-day average  (institutional surge)

RISK GATE (automatic disqualifiers — stock is skipped if any fail)
  ✗   Price < $5           (too illiquid/volatile for reliable signals)
  ✗   ATR% > 8%            (daily move too large for swing sizing)
  ✗   ADX < 15             (no trend — choppy market conditions)

═══════════════════════════════════════════════════════════════════════════════
ENTRY / STOP / TARGET
  Entry  = current close (or open next day)
  Stop   = close − 1.5 × ATR(14)   [below recent structure]
  Target = close + 3.0 × ATR(14)   [2:1 R/R minimum]
  Also shows: Pivot Target (first 52-week resistance) when applicable
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from threading import Lock
import concurrent.futures
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from livermore_pivotal_screener import get_universe, DEFAULT_TICKERS

_yf_lock = Lock()

# ── Shared indicators ─────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=n, adjust=False).mean()

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["High"].diff()
    dn = -df["Low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_s = _atr(df, n)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(span=n, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(span=n, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)) * 100
    return dx.ewm(span=n, adjust=False).mean()


_spy_cache: Optional[pd.DataFrame] = None

def _get_spy() -> pd.DataFrame:
    global _spy_cache
    if _spy_cache is not None and not _spy_cache.empty:
        return _spy_cache
    end = datetime.today()
    start = end - timedelta(days=400)
    try:
        with _yf_lock:
            raw = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        _spy_cache = raw[["Close"]].dropna()
    except Exception:
        _spy_cache = pd.DataFrame()
    return _spy_cache


def _rs_score(df: pd.DataFrame, spy: pd.DataFrame) -> float:
    """
    Approximate IBD RS Rating (0–100). Higher = stock outperforming SPY.
    Weighted: most recent quarter counts 2×.
    """
    periods = [63, 126, 189, 252]
    weights = [2, 1, 1, 1]
    if len(df) < 260 or len(spy) < 260:
        return 50.0
    def _perf(frame, p):
        if len(frame) < p + 1:
            return 0.0
        return frame["Close"].iloc[-1] / frame["Close"].iloc[-(p+1)] - 1
    try:
        stock_score = sum(w * _perf(df, p) for w, p in zip(weights, periods)) / sum(weights)
        spy_score   = sum(w * _perf(spy, p) for w, p in zip(weights, periods)) / sum(weights)
        return max(0.0, min(100.0, 50.0 + (stock_score - spy_score) * 500))
    except Exception:
        return 50.0


# ── Trigger detection (single-stock) ─────────────────────────────────────────

def _check_ema_cross(df: pd.DataFrame, recent: int = 5) -> bool:
    """EMA-9 crossed above EMA-21 in the last `recent` bars."""
    e9 = _ema(df["Close"], 9)
    e21 = _ema(df["Close"], 21)
    cross = (e9.shift(1) <= e21.shift(1)) & (e9 > e21)
    return bool(cross.iloc[-recent:].any())


def _check_breakout(df: pd.DataFrame, lookback: int = 20, vol_mult: float = 1.5) -> bool:
    """Close breaks above 20-day high with volume surge."""
    if len(df) < lookback + 2:
        return False
    vol20 = df["Volume"].rolling(20).mean()
    resistance = df["High"].rolling(lookback).max().shift(1)
    vol_ok = df["Volume"].iloc[-1] > vol20.iloc[-1] * vol_mult
    broke = df["Close"].iloc[-1] > resistance.iloc[-1]
    return bool(broke and vol_ok)


def _check_nr7(df: pd.DataFrame) -> bool:
    """NR7 breakout: narrowest range in 7 days, then close above that bar's high."""
    if len(df) < 10:
        return False
    bar_range = df["High"] - df["Low"]
    is_nr7 = bar_range == bar_range.rolling(7).min()
    # Check if NR7 occurred 1-3 bars ago and today closed above it
    for lag in range(1, 4):
        if len(df) > lag and is_nr7.iloc[-(lag+1)]:
            nr7_high = df["High"].iloc[-(lag+1)]
            if df["Close"].iloc[-1] > nr7_high:
                return True
    return False


def _check_bb_squeeze(df: pd.DataFrame) -> bool:
    """Bollinger squeeze: BB width at 6-month low, then close outside the band."""
    if len(df) < 130:
        return False
    mid = _sma(df["Close"], 20)
    std = df["Close"].rolling(20).std()
    bb_upper = mid + 2 * std
    bb_lower = mid - 2 * std
    bb_width = (bb_upper - bb_lower) / mid.replace(0, np.nan)
    bb_min_6m = bb_width.rolling(126).min()

    # Squeeze within last 5 bars
    in_squeeze = (bb_width.iloc[-6:-1] == bb_min_6m.iloc[-6:-1]).any()
    if not in_squeeze:
        return False
    vol20 = df["Volume"].rolling(20).mean()
    vol_ok = df["Volume"].iloc[-1] > vol20.iloc[-1] * 1.5
    return bool(df["Close"].iloc[-1] > bb_upper.iloc[-1] and vol_ok)


def _check_gap_up(df: pd.DataFrame, min_gap_pct: float = 0.3) -> bool:
    """Gap-up continuation/breakaway in last 3 bars."""
    if len(df) < 5:
        return False
    for i in range(-3, 0):
        prev_close = df["Close"].iloc[i - 1]
        open_p = df["Open"].iloc[i]
        if prev_close > 0 and (open_p - prev_close) / prev_close * 100 >= min_gap_pct:
            # Trend filter: above EMA50
            ema50 = _ema(df["Close"], 50).iloc[i]
            if df["Close"].iloc[i] > ema50:
                return True
    return False


def _minervini_trend_score(df: pd.DataFrame) -> int:
    """Fast Minervini trend template score (0-8)."""
    if len(df) < 210:
        return 0
    price  = df["Close"].iloc[-1]
    sma50  = _sma(df["Close"], 50).iloc[-1]
    sma150 = _sma(df["Close"], 150).iloc[-1]
    sma200 = _sma(df["Close"], 200).iloc[-1]
    sma200_series = _sma(df["Close"], 200)
    sma200_slope = sma200 > sma200_series.iloc[-22] if len(sma200_series) >= 22 else False
    high52w = df["High"].rolling(252).max().iloc[-1]
    low52w  = df["Low"].rolling(252).min().iloc[-1]
    checks = [
        price > sma150,
        price > sma200,
        sma150 > sma200,
        bool(sma200_slope),
        sma50 > sma150 and sma50 > sma200,
        price > sma50,
        price >= low52w * 1.30 if not np.isnan(low52w) else False,
        price >= high52w * 0.75 if not np.isnan(high52w) else False,
    ]
    return sum(bool(c) for c in checks)


# ── Main per-ticker scorer ────────────────────────────────────────────────────

def score_ticker(
    ticker: str,
    df: pd.DataFrame,
    spy: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Score a single ticker. Returns a result dict or None if filtered out.
    """
    if len(df) < 60:
        return None

    close  = df["Close"].iloc[-1]
    atr14  = _atr(df).iloc[-1]
    adx14  = _adx(df).iloc[-1]
    rsi14  = _rsi(df["Close"]).iloc[-1]
    ema50  = _ema(df["Close"], 50).iloc[-1]
    ema200 = _ema(df["Close"], 200).iloc[-1]
    vol20  = df["Volume"].rolling(20).mean().iloc[-1]
    vol    = df["Volume"].iloc[-1]
    vol_ratio = vol / vol20 if vol20 > 0 else 0

    # ── Risk gates ────────────────────────────────────────────────────────────
    if close < cfg.get("min_price", 5.0):
        return None
    atr_pct = (atr14 / close * 100) if close > 0 else 999
    if atr_pct > cfg.get("max_atr_pct", 8.0):
        return None
    if not np.isnan(adx14) and adx14 < cfg.get("min_adx", 15.0):
        return None

    # ── MACD ─────────────────────────────────────────────────────────────────
    fast_ema  = _ema(df["Close"], cfg.get("macd_fast", 12))
    slow_ema  = _ema(df["Close"], cfg.get("macd_slow", 26))
    macd_line = fast_ema - slow_ema
    sig_line  = _ema(macd_line, cfg.get("macd_signal", 9))
    histogram = macd_line - sig_line
    macd_bull = bool(
        histogram.iloc[-1] > 0 or
        (histogram.iloc[-3:] < 0).all() == False and histogram.iloc[-1] > histogram.iloc[-2]
    )
    macd_just_flipped = bool(histogram.iloc[-2] < 0 and histogram.iloc[-1] >= 0) if len(histogram) >= 2 else False

    # ── RS Score ─────────────────────────────────────────────────────────────
    rs = _rs_score(df, spy)

    # ══ SCORING ══════════════════════════════════════════════════════════════

    score = 0
    reasons = []

    # --- TREND (max 3) -------------------------------------------------------
    trend_pts = 0
    if close > ema50:
        trend_pts += 1
        reasons.append("Price > EMA50")
    if ema50 > ema200:
        trend_pts += 1
        reasons.append("EMA50 > EMA200")
    mv_score = _minervini_trend_score(df)
    if mv_score >= 6:
        trend_pts += 1
        reasons.append(f"Minervini {mv_score}/8")
    score += trend_pts

    # --- MOMENTUM (max 3) ----------------------------------------------------
    mom_pts = 0
    rsi_low  = cfg.get("rsi_low", 45)
    rsi_high = cfg.get("rsi_high", 72)
    if not np.isnan(rsi14) and rsi_low <= rsi14 <= rsi_high:
        mom_pts += 1
        reasons.append(f"RSI {rsi14:.1f}")
    if macd_bull:
        mom_pts += 1
        reasons.append("MACD+" + (" flip" if macd_just_flipped else ""))
    if rs >= cfg.get("min_rs", 60):
        mom_pts += 1
        reasons.append(f"RS {rs:.0f}")
    score += mom_pts

    # --- ENTRY TRIGGER (max 3, need ≥ 1) ------------------------------------
    triggers = []
    if _check_ema_cross(df, recent=cfg.get("ema_cross_bars", 5)):
        triggers.append("EMA9×21")
    if _check_breakout(df, lookback=20, vol_mult=cfg.get("breakout_vol_mult", 1.5)):
        triggers.append("BO-20d")
    if _check_nr7(df):
        triggers.append("NR7")
    if _check_bb_squeeze(df):
        triggers.append("BB-Squeeze")
    if _check_gap_up(df, min_gap_pct=cfg.get("min_gap_pct", 0.3)):
        triggers.append("Gap↑")

    if not triggers:
        return None  # No entry trigger = no signal regardless of score

    trig_pts = min(3, len(triggers))
    score += trig_pts
    reasons.extend(triggers)

    # --- VOLUME (max 2) ------------------------------------------------------
    if vol_ratio >= cfg.get("vol_high", 3.0):
        score += 2
        reasons.append(f"Vol {vol_ratio:.1f}×↑↑")
    elif vol_ratio >= cfg.get("vol_low", 1.5):
        score += 1
        reasons.append(f"Vol {vol_ratio:.1f}×")

    # ── Signal classification ─────────────────────────────────────────────────
    if score >= 7:
        signal = "⚡ STRONG BUY"
    elif score >= 5:
        signal = "✅ BUY"
    elif score >= 3:
        signal = "👀 WATCH"
    else:
        return None

    # ── Entry / Stop / Target ────────────────────────────────────────────────
    entry  = round(close, 2)
    stop   = round(close - 1.5 * atr14, 2)
    target = round(close + 3.0 * atr14, 2)
    rr     = round(3.0 / 1.5, 2)

    # Pivot target: distance to 52-week high (if within reach)
    high52w = df["High"].rolling(252).max().iloc[-1]
    pivot_target = round(high52w, 2) if not np.isnan(high52w) and high52w > close else None

    return {
        "Ticker":         ticker,
        "Signal":         signal,
        "Score":          score,
        "Why":            " · ".join(reasons),
        "Triggers":       " · ".join(triggers),
        "Entry":          entry,
        "Stop":           stop,
        "Target":         target,
        "R/R":            rr,
        "52W Target":     pivot_target,
        "RSI":            round(rsi14, 1) if not np.isnan(rsi14) else None,
        "MACD Hist":      round(histogram.iloc[-1], 4),
        "ADX":            round(adx14, 1) if not np.isnan(adx14) else None,
        "RS Score":       round(rs, 1),
        "Vol vs Avg":     round(vol_ratio, 2),
        "ATR%":           round(atr_pct, 2),
        "Minervini":      f"{mv_score}/8",
        "EMA Stack":      "ALIGNED" if close > ema50 > ema200 else "PARTIAL" if close > ema50 else "BELOW",
        "Trend Pts":      trend_pts,
        "Momentum Pts":   mom_pts,
        "Trigger Pts":    trig_pts,
    }


# ── Main screener ─────────────────────────────────────────────────────────────

def run_combined_screener(
    tickers: Optional[List[str]] = None,
    # Trend
    min_minervini: int = 5,
    # Momentum
    rsi_low: float = 45.0,
    rsi_high: float = 72.0,
    min_rs: float = 60.0,
    # MACD
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    # Entry triggers
    ema_cross_bars: int = 5,
    breakout_vol_mult: float = 1.5,
    min_gap_pct: float = 0.3,
    # Volume
    vol_low: float = 1.5,
    vol_high: float = 3.0,
    # Risk gates
    min_price: float = 5.0,
    max_atr_pct: float = 8.0,
    min_adx: float = 15.0,
    # Output
    min_score: int = 3,
    lookback_days: int = 400,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Run the combined buy signal scanner across all tickers.
    Returns a ranked DataFrame of BUY / STRONG BUY / WATCH signals.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    cfg = dict(
        min_price=min_price, max_atr_pct=max_atr_pct, min_adx=min_adx,
        rsi_low=rsi_low, rsi_high=rsi_high, min_rs=min_rs,
        macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
        ema_cross_bars=ema_cross_bars, breakout_vol_mult=breakout_vol_mult,
        min_gap_pct=min_gap_pct, vol_low=vol_low, vol_high=vol_high,
    )

    end   = datetime.today()
    start = end - timedelta(days=lookback_days)
    spy   = _get_spy()

    def _process(ticker: str):
        try:
            with _yf_lock:
                raw = yf.download(ticker, start=start, end=end,
                                  progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.loc[:, ~raw.columns.duplicated()]
            df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
            return score_ticker(ticker, df, spy, cfg)
        except Exception:
            return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, t): t for t in tickers}
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            if r and r["Score"] >= min_score:
                results.append(r)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    # Sort: STRONG BUY first, then by Score desc, then RS desc
    signal_order = {"⚡ STRONG BUY": 0, "✅ BUY": 1, "👀 WATCH": 2}
    out["_sig_order"] = out["Signal"].map(signal_order).fillna(3)
    out = out.sort_values(["_sig_order", "Score", "RS Score"],
                          ascending=[True, False, False]).drop(columns=["_sig_order"])
    return out.reset_index(drop=True)
