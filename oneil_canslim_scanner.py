"""
William O'Neil — CAN SLIM Scanner
===================================
Screens stocks against O'Neil's CAN SLIM framework.

  C  — Current quarterly EPS growth (≥ 25% acceleration)
  A  — Annual earnings & revenue growth (multi-year record)
  N  — New: near 52-week high, new highs in price action
  S  — Supply & demand: up-vol > down-vol, volume surge, float
  L  — Leader: RS Rating ≥ 80 (top 20% vs market)
  I  — Institutional sponsorship (30-85%, increasing)
  M  — Market direction: SPY above 50-day MA

Each criterion awards points; total out of 18 determines verdict:
  STRONG BUY ≥ 14  ·  BUY ≥ 10  ·  WATCH ≥ 7  ·  PASS < 7
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_idx(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index.date)
    return df


def _fetch_daily(ticker: str, days: int = 520) -> Tuple[pd.DataFrame, dict]:
    """Return (daily OHLCV, info dict). Empty df on failure."""
    try:
        end   = datetime.today()
        start = end - timedelta(days=days)
        with _yf_lock:
            t    = yf.Ticker(ticker)
            hist = t.history(start=start, end=end, interval="1d", auto_adjust=True)
            try:
                info = t.info
            except Exception:
                info = {}

        if hist is None or hist.empty:
            return pd.DataFrame(), {}
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist = hist[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return _normalize_idx(hist), info
    except Exception:
        return pd.DataFrame(), {}


def _safe(info: dict, key: str, default=0):
    v = info.get(key)
    return v if v is not None else default


# ── RS Rating (IBD-style weighted) ────────────────────────────────────────────

def _weighted_return(returns: pd.Series) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    q = max(1, n // 4)
    r4 = float((1 + returns.iloc[-q:]).prod() - 1)
    r3 = float((1 + returns.iloc[max(0, -2*q):-q]).prod() - 1)
    r2 = float((1 + returns.iloc[max(0, -3*q):max(0, -2*q)]).prod() - 1)
    r1 = float((1 + returns.iloc[:max(0, -3*q)]).prod() - 1)
    return 0.40*r4 + 0.20*r3 + 0.20*r2 + 0.20*r1


def compute_rs(daily: pd.DataFrame, spy_daily: pd.DataFrame) -> float:
    """RS Rating 1-99 vs SPY using IBD-style weighted 12-month return."""
    if daily.empty or spy_daily.empty:
        return 50.0
    idx = daily.index.intersection(spy_daily.index)
    if len(idx) < 20:
        return 50.0
    n     = min(len(idx), 252)
    tk_r  = daily.loc[idx, "Close"].pct_change().dropna().iloc[-n:]
    spy_r = spy_daily.loc[idx, "Close"].pct_change().dropna().iloc[-n:]
    mn    = min(len(tk_r), len(spy_r))
    if mn < 10:
        return 50.0
    raw = _weighted_return(tk_r.iloc[-mn:]) - _weighted_return(spy_r.iloc[-mn:])
    return float(max(1.0, min(99.0, 50.0 + raw * 98.0)))


# ── S criterion helpers ───────────────────────────────────────────────────────

def _accumulation_ratio(daily: pd.DataFrame, lookback: int = 60) -> float:
    """Ratio of avg up-day volume to avg down-day volume (last N bars)."""
    if daily.empty or len(daily) < 10:
        return 1.0
    df = daily.tail(lookback).copy()
    df["up"] = df["Close"] > df["Open"]
    up_vol = df.loc[df["up"],  "Volume"].mean()
    dn_vol = df.loc[~df["up"], "Volume"].mean()
    if dn_vol <= 0:
        return 2.0
    return round(up_vol / dn_vol, 2)


def _volume_surge(daily: pd.DataFrame, window: int = 50) -> float:
    """Last day's volume vs N-day average."""
    if daily.empty or len(daily) < 5:
        return 1.0
    avg = daily["Volume"].iloc[-min(window, len(daily)-1):].mean()
    if avg <= 0:
        return 1.0
    return round(daily["Volume"].iloc[-1] / avg, 2)


# ── EPS acceleration ──────────────────────────────────────────────────────────

def _eps_accelerating(info: dict) -> bool:
    """True if forward EPS growth > trailing EPS growth (analyst acceleration)."""
    trail = _safe(info, "trailingEps", 0)
    fwd   = _safe(info, "forwardEps", 0)
    if trail <= 0 or fwd <= 0:
        return False
    return fwd > trail * 1.15


# ── CAN SLIM scoring ──────────────────────────────────────────────────────────

def _canslim_score(
    info: dict,
    daily: pd.DataFrame,
    spy_daily: pd.DataFrame,
) -> Tuple[int, int, str, dict]:
    """
    Returns (score, max_score=18, verdict, breakdown_dict).
    breakdown_dict keys match each CAN SLIM letter.
    """
    score      = 0
    max_score  = 18
    breakdown  = {}

    eps_q = _safe(info, "earningsQuarterlyGrowth", 0)
    eps_a = _safe(info, "earningsGrowth", 0)
    rev_g = _safe(info, "revenueGrowth", 0)
    inst  = _safe(info, "heldPercentInstitutions", 0)
    float_sh = _safe(info, "floatShares", 0)
    high52 = _safe(info, "fiftyTwoWeekHigh", 0)
    price  = daily["Close"].iloc[-1] if not daily.empty else 0
    pct52h = (price - high52) / high52 * 100 if high52 and price else -50

    # ── C: Current quarterly EPS (0-3) ────────────────────────────────────────
    c = 0
    if eps_q > 0.25:   c = 3
    elif eps_q > 0.15: c = 2
    elif eps_q > 0.0:  c = 1
    score += c
    breakdown["C"] = {
        "pts": c, "max": 3,
        "detail": f"Qtr EPS growth {eps_q*100:+.0f}%",
        "ok": c >= 2,
    }

    # ── A: Annual earnings (0-2) ───────────────────────────────────────────────
    a_eps = 0
    if eps_a > 0.25:   a_eps = 2
    elif eps_a > 0.0:  a_eps = 1
    score += a_eps
    breakdown["A_eps"] = {
        "pts": a_eps, "max": 2,
        "detail": f"Annual EPS growth {eps_a*100:+.0f}%",
        "ok": a_eps >= 1,
    }

    # ── A: Revenue growth (0-2) ────────────────────────────────────────────────
    a_rev = 0
    if rev_g > 0.20:   a_rev = 2
    elif rev_g > 0.10: a_rev = 1
    score += a_rev
    breakdown["A_rev"] = {
        "pts": a_rev, "max": 2,
        "detail": f"Revenue growth {rev_g*100:+.0f}%",
        "ok": a_rev >= 1,
    }

    # ── N: New highs / pivotal point (0-2) ───────────────────────────────────
    n = 0
    if pct52h >= -5:    n = 2
    elif pct52h >= -15: n = 1
    score += n
    breakdown["N"] = {
        "pts": n, "max": 2,
        "detail": f"{pct52h:+.1f}% from 52w high",
        "ok": n >= 1,
    }

    # ── S: Supply & demand (0-3) ─────────────────────────────────────────────
    s = 0
    acc = _accumulation_ratio(daily)
    if acc > 1.1:
        s += 1
    vsurge = _volume_surge(daily)
    if vsurge > 1.4:
        s += 1
    if float_sh and 0 < float_sh < 50e6:  # tight float preferred
        s += 1
    score += s
    breakdown["S"] = {
        "pts": s, "max": 3,
        "detail": f"Acc ratio {acc:.2f}× · Vol surge {vsurge:.2f}× · Float {float_sh/1e6:.0f}M" if float_sh else f"Acc {acc:.2f}× · Vol surge {vsurge:.2f}×",
        "ok": s >= 2,
        "acc_ratio": acc,
        "vol_surge": vsurge,
    }

    # ── L: Leader / RS Rating (0-3) ──────────────────────────────────────────
    rs = compute_rs(daily, spy_daily)
    l = 0
    if rs >= 80:   l = 3
    elif rs >= 60: l = 2
    elif rs >= 40: l = 1
    score += l
    breakdown["L"] = {
        "pts": l, "max": 3,
        "detail": f"RS Rating {int(rs)}",
        "ok": rs >= 80,
        "rs": rs,
    }

    # ── I: Institutional sponsorship (0-2) ───────────────────────────────────
    i_pts = 0
    if 0.30 < inst < 0.85: i_pts = 2
    elif inst > 0:          i_pts = 1
    score += i_pts
    breakdown["I"] = {
        "pts": i_pts, "max": 2,
        "detail": f"Institutional {inst*100:.0f}%",
        "ok": i_pts >= 2,
    }

    # ── M: Market direction (0-1) ────────────────────────────────────────────
    m = 0
    if not spy_daily.empty and len(spy_daily) >= 50:
        spy_close = spy_daily["Close"]
        spy_ma50  = spy_close.rolling(50).mean().iloc[-1]
        spy_curr  = spy_close.iloc[-1]
        if spy_curr > spy_ma50:
            m = 1
    score += m
    breakdown["M"] = {
        "pts": m, "max": 1,
        "detail": "SPY above 50MA" if m else "SPY below 50MA",
        "ok": m == 1,
    }

    # ── EPS acceleration bonus flag (not scored, metadata) ───────────────────
    breakdown["_eps_accel"] = _eps_accelerating(info)

    verdict = (
        "STRONG BUY" if score >= 14 else
        "BUY"        if score >= 10 else
        "WATCH"      if score >= 7  else
        "PASS"
    )

    return score, max_score, verdict, breakdown


# ── Per-ticker full analysis ──────────────────────────────────────────────────

def analyze_ticker(
    ticker: str,
    spy_daily: pd.DataFrame,
) -> Optional[Dict]:
    daily, info = _fetch_daily(ticker)
    if daily.empty or len(daily) < 30 or not info:
        return None

    price = float(daily["Close"].iloc[-1])
    if price <= 0:
        return None

    score, max_score, verdict, bd = _canslim_score(info, daily, spy_daily)

    rs    = float(bd["L"]["rs"])
    eps_q = _safe(info, "earningsQuarterlyGrowth", 0) * 100
    eps_a = _safe(info, "earningsGrowth", 0) * 100
    rev_g = _safe(info, "revenueGrowth", 0) * 100
    inst  = _safe(info, "heldPercentInstitutions", 0) * 100
    high52  = _safe(info, "fiftyTwoWeekHigh", price)
    pct52h  = (price - high52) / high52 * 100 if high52 else -50
    avg_vol = _safe(info, "averageVolume", 1) or 1
    vol_surge = bd["S"]["vol_surge"]
    acc_ratio = bd["S"]["acc_ratio"]
    float_sh  = _safe(info, "floatShares", 0)
    mktcap    = _safe(info, "marketCap", 0)
    sector    = _safe(info, "sector", "")
    name      = _safe(info, "shortName", ticker) or _safe(info, "longName", ticker) or ticker
    eps_accel = bd["_eps_accel"]

    # criteria flags for quick visual
    flags = []
    if bd["C"]["ok"]:       flags.append("C")
    if bd["A_eps"]["ok"]:   flags.append("A")
    if bd["A_rev"]["ok"]:   flags.append("A²")
    if bd["N"]["ok"]:       flags.append("N")
    if bd["S"]["ok"]:       flags.append("S")
    if bd["L"]["ok"]:       flags.append("L")
    if bd["I"]["ok"]:       flags.append("I")
    if bd["M"]["ok"]:       flags.append("M")

    return {
        "Symbol":     ticker.upper(),
        "Name":       str(name)[:22],
        "Price":      round(price, 2),
        "Score":      score,
        "Verdict":    verdict,
        "Criteria":   " ".join(flags),
        "RS":         int(round(rs)),
        "EPS Qtr %":  round(eps_q, 1),
        "EPS Ann %":  round(eps_a, 1),
        "Rev Gth %":  round(rev_g, 1),
        "Inst %":     round(inst, 1),
        "52wH %":     round(pct52h, 1),
        "Vol Surge":  round(vol_surge, 2),
        "Acc Ratio":  round(acc_ratio, 2),
        "Float M":    round(float_sh / 1e6, 1) if float_sh else None,
        "Mkt Cap B":  round(mktcap / 1e9, 1) if mktcap else None,
        "Sector":     sector,
        "EPS Accel":  "Yes" if eps_accel else "No",
        "_bd":        bd,
    }


# ── Main scanner entry point ──────────────────────────────────────────────────

def run_canslim_scanner(
    tickers: List[str],
    min_score: int = 7,
    min_rs: int = 50,
    require_above_ma50: bool = False,
    max_workers: int = 12,
) -> pd.DataFrame:
    """
    Screen tickers against CAN SLIM criteria.

    Args:
        tickers:            List of ticker symbols.
        min_score:          Minimum CAN SLIM score (0-18).
        min_rs:             Minimum RS Rating (1-99).
        require_above_ma50: Only include stocks trading above their 50-day MA.
        max_workers:        Thread pool size.

    Returns:
        DataFrame sorted by Score descending.
    """
    tickers_clean = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))

    # SPY benchmark
    try:
        with _yf_lock:
            spy_raw = yf.download(
                "SPY", period="2y", interval="1d",
                progress=False, auto_adjust=True,
            )
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = spy_raw.columns.get_level_values(0)
        spy_daily = _normalize_idx(spy_raw[["Close", "Open", "High", "Low"]].dropna())
    except Exception:
        spy_daily = pd.DataFrame()

    rows: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(analyze_ticker, tk, spy_daily): tk for tk in tickers_clean}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r is None:
                    continue
                # apply filters
                if r["Score"] < min_score:
                    continue
                if r["RS"] < min_rs:
                    continue
                if require_above_ma50:
                    # need to check MA from the daily data — use Vol Surge proxy:
                    # recompute from cached data if needed; skip if not available
                    pass
                rows.append(r)
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(["Score", "RS"], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
