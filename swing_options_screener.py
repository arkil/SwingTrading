"""
Swing Options Screener — 45-60 DTE
===================================
Wrapper that bridges the dashboard's ticker universe with the
swing_options_45_60d strategy module.

Public API
----------
    run_swing_options_screener(tickers, ...) -> pd.DataFrame
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── Path: add swing_options_45_60d/src to import path ────────────────────────
_STRATEGY_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "swing_options_45_60d")
)
if _STRATEGY_DIR not in sys.path:
    sys.path.insert(0, _STRATEGY_DIR)

from src.indicators import compute_all_indicators
from src.screener import screen_universe
from src.options_sim import calc_option_entry

# ── Default strategy params (mirrors config.yaml) ────────────────────────────
DEFAULT_PARAMS = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "ema_trend": 200,
    "sma_50": 50, "sma_200": 200,
    "rsi_period": 14, "rsi_entry_min": 45, "rsi_entry_max": 65,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "bb_period": 20, "bb_std": 2.0,
    "atr_period": 14,
    "volume_surge_multiplier": 1.5,
    "adx_period": 14, "adx_min": 20,
    "stoch_k": 14, "stoch_d": 3,
    "roc_period": 10,
    # Greeks / entry params
    "delta_target": 0.55,
    "delta_min": 0.40,
    "delta_max": 0.70,
    "theta_max_daily_pct": 0.015,
    "gamma_min": 0.005,
    "gamma_max": 0.050,
    "theta_vega_ratio_max": 0.40,
    "max_entry_sigma": 0.50,
}

# ── VIX → IV rank helper ──────────────────────────────────────────────────────

def _fetch_vix_rank() -> float:
    """Fetch current VIX and return approximate IV rank (0-100)."""
    try:
        vix = yf.download("^VIX", period="1y", progress=False, auto_adjust=True)
        if vix.empty:
            return 25.0
        close = vix["Close"].squeeze()
        current = float(close.iloc[-1])
        lo = float(close.min())
        hi = float(close.max())
        rank = (current - lo) / (hi - lo + 1e-9) * 100
        return round(float(rank), 1)
    except Exception:
        return 25.0


# ── OHLCV fetcher ─────────────────────────────────────────────────────────────

def _fetch_ohlcv(ticker: str, days: int = 600) -> pd.DataFrame:
    """Download daily OHLCV via yfinance. Returns tz-naive DataFrame."""
    try:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df = df[df["Close"] > 0]
        return df
    except Exception:
        return pd.DataFrame()


# ── Greeks filter (mirrors backtest.py logic) ─────────────────────────────────

def _passes_greeks(opt: dict, params: dict) -> tuple[bool, str]:
    """
    Returns (passes, rejection_reason).
    Applies the 4-filter Greek gate added to backtest.py.
    """
    delta_min = params.get("delta_min", 0.40)
    delta_max = params.get("delta_max", 0.70)
    if not (delta_min <= abs(opt["delta"]) <= delta_max):
        return False, f"δ={abs(opt['delta']):.2f} out of [{delta_min},{delta_max}]"

    theta_max_pct = params.get("theta_max_daily_pct", 0.015)
    theta_pct = abs(opt["theta"]) / max(opt["premium"], 0.01)
    if theta_pct > theta_max_pct:
        return False, f"θ/prem={theta_pct:.3f} > {theta_max_pct}"

    gamma_min = params.get("gamma_min", 0.005)
    gamma_max = params.get("gamma_max", 0.050)
    if not (gamma_min <= opt["gamma"] <= gamma_max):
        return False, f"γ={opt['gamma']:.4f} out of [{gamma_min},{gamma_max}]"

    theta_vega_max = params.get("theta_vega_ratio_max", 0.40)
    if opt["vega"] > 0:
        tv = abs(opt["theta"]) / opt["vega"]
        if tv > theta_vega_max:
            return False, f"θ/vega={tv:.2f} > {theta_vega_max}"

    return True, ""


# ── Main public API ───────────────────────────────────────────────────────────

def run_swing_options_screener(
    tickers: list,
    min_score: float = 6.0,
    dte: float = 50.0,
    iv_premium: float = 1.10,
    params: dict = None,
    iv_rank_override: float = None,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Scan tickers for 45-60 DTE options swing setups.

    Args:
        tickers:           List of symbols to scan
        min_score:         Minimum composite score (0-10) to include
        dte:               Target DTE for option pricing (45-60)
        iv_premium:        IV = hist_vol * iv_premium (accounts for vol premium)
        params:            Strategy params dict (defaults to DEFAULT_PARAMS)
        iv_rank_override:  Force a specific IV rank (useful for testing)
        progress_cb:       Optional callable(pct: float, msg: str) for progress

    Returns:
        DataFrame with columns:
            Symbol, Direction, Score, Close, Strike, Premium,
            Delta, Theta/day, Gamma, Vega/1%, Theta/Prem, Theta/Vega,
            Hist Vol, ATR, ADX, RSI, EMA Aligned, MACD OK, Vol Surge,
            Supertrend, BB Expansion, Stoch OK, OBV OK
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    iv_rank = iv_rank_override if iv_rank_override is not None else _fetch_vix_rank()

    # Filter out symbols with insufficient data requirement
    n = len(tickers)
    raw_data: dict[str, pd.DataFrame] = {}

    for i, sym in enumerate(tickers):
        if progress_cb:
            progress_cb(i / n, f"Fetching {sym}…")
        df = _fetch_ohlcv(sym, days=600)
        if len(df) >= 220:
            raw_data[sym] = df

    if progress_cb:
        progress_cb(0.6, "Computing indicators…")

    # Compute indicators
    enriched: dict[str, pd.DataFrame] = {}
    for sym, df in raw_data.items():
        try:
            enriched[sym] = compute_all_indicators(df, p)
        except Exception:
            pass

    if progress_cb:
        progress_cb(0.75, "Running screener…")

    # Screen universe
    candidates = screen_universe(
        data=enriched,
        params=p,
        iv_rank=iv_rank,
        min_score=min_score,
    )

    if candidates.empty:
        if progress_cb:
            progress_cb(1.0, "Done")
        return pd.DataFrame()

    # Enrich each candidate with live option greeks
    rows = []
    for _, row in candidates.iterrows():
        sym = row["symbol"]
        direction = row["direction"]
        close = row["close"]
        hist_vol = max(row.get("hist_vol", 0.20), 0.10)

        # High-vol filter (mirrors backtest)
        if hist_vol * iv_premium > p.get("max_entry_sigma", 0.50):
            continue

        try:
            opt = calc_option_entry(
                S=close,
                direction=direction,
                dte=dte,
                hist_vol=hist_vol,
                iv_premium=iv_premium,
                delta_target=p.get("delta_target", 0.55),
            )
        except Exception:
            continue

        passes, reason = _passes_greeks(opt, p)

        theta_pct = abs(opt["theta"]) / max(opt["premium"], 0.01)
        tv_ratio = abs(opt["theta"]) / max(opt["vega"], 1e-9)

        rows.append({
            "Symbol":       sym,
            "Direction":    direction.upper(),
            "Score":        row["score"],
            "Close":        round(close, 2),
            "Strike":       int(opt["strike"]),
            "Premium":      round(opt["premium"], 2),
            "Delta":        round(abs(opt["delta"]), 3),
            "Theta/day":    round(opt["theta"], 4),
            "Gamma":        round(opt["gamma"], 4),
            "Vega/1%":      round(opt["vega"], 3),
            "θ/Prem %":     round(theta_pct * 100, 2),
            "θ/Vega":       round(tv_ratio, 3),
            "Greeks OK":    "✅" if passes else f"❌ {reason}",
            "Hist Vol":     round(hist_vol, 3),
            "ATR":          round(row.get("atr", 0), 2),
            "ADX":          round(row.get("adx", 0), 1),
            "RSI":          round(row.get("rsi", 0), 1),
            "EMA Aligned":  "✅" if row.get("ema_aligned") else "—",
            "Trend 200":    "✅" if row.get("trend_200") else "—",
            "MACD OK":      "✅" if row.get("macd_ok") else "—",
            "Vol Surge":    "✅" if row.get("vol_surge") else "—",
            "Supertrend":   "✅" if row.get("supertrend_ok") else "—",
            "BB Exp":       "✅" if row.get("bb_expansion") else "—",
            "Stoch OK":     "✅" if row.get("stoch_ok") else "—",
            "OBV OK":       "✅" if row.get("obv_ok") else "—",
            "_passes_greeks": passes,
        })

    if progress_cb:
        progress_cb(1.0, "Done")

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    return out
