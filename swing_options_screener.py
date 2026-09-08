"""
Swing Options Screener — 45-60 DTE
===================================
Wrapper that bridges the dashboard's ticker universe with the
swing_options_45_60d strategy module.

Public API
----------
    run_swing_options_screener(tickers, ...) -> pd.DataFrame
"""

from __future__ import annotations

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
    "adx_period": 14, "adx_min": 25,
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
    """Current VIX 1-year percentile rank (0-100). Shared source: market_context.get_vix()."""
    try:
        from market_context import get_vix
        return get_vix()["rank_1y"]
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


# ── Real option chain fetcher ─────────────────────────────────────────────────

def _fetch_real_option_data(
    symbol: str,
    direction: str,
    target_dte: float = 50.0,
    delta_target: float = 0.55,
    r: float = 0.045,
) -> dict | None:
    """
    Fetch a real option contract from yfinance nearest to target_dte and delta_target.
    Returns a dict with the same keys as calc_option_entry plus expiry/bid/ask/volume/OI.
    Returns None if no suitable contract found.
    """
    from datetime import date
    from scipy.stats import norm

    try:
        ticker = yf.Ticker(symbol)
        today = date.today()

        fi = ticker.fast_info
        S = float(fi.get("lastPrice") or fi.get("previousClose") or 0)
        if not S:
            return None

        # Find expiration closest to target_dte (search 35–75 DTE window)
        exps = ticker.options or []
        valid = [e for e in exps if 35 <= (date.fromisoformat(e) - today).days <= 75]
        if not valid:
            return None
        best_exp = min(valid, key=lambda e: abs((date.fromisoformat(e) - today).days - target_dte))
        dte_actual = (date.fromisoformat(best_exp) - today).days
        T = dte_actual / 365.0

        chain = ticker.option_chain(best_exp)
        df = chain.calls.copy() if direction.lower() == "call" else chain.puts.copy()
        if df.empty:
            return None

        # Compute delta for each row using real IV so we can pick the right strike
        def _compute_delta(row):
            iv = float(row.get("impliedVolatility") or 0)
            K = float(row.get("strike") or 0)
            if iv <= 0 or K <= 0 or T <= 0:
                return 0.0
            sqrtT = np.sqrt(T)
            try:
                d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * sqrtT)
                return float(norm.cdf(d1)) if direction.lower() == "call" else float(-norm.cdf(-d1))
            except Exception:
                return 0.0

        df["_delta"] = df.apply(_compute_delta, axis=1)
        df["_delta_diff"] = (df["_delta"].abs() - delta_target).abs()
        row = df.loc[df["_delta_diff"].idxmin()]

        iv = float(row.get("impliedVolatility") or 0)
        K = float(row["strike"])
        bid = float(row.get("bid") or 0)
        ask = float(row.get("ask") or 0)
        mid = (bid + ask) / 2 if bid and ask else float(row.get("lastPrice") or 0)
        volume = int(row.get("volume") or 0)
        oi = int(row.get("openInterest") or 0)
        contract = str(row.get("contractSymbol") or "")

        # Compute Greeks from real IV
        if iv > 0 and T > 0:
            sqrtT = np.sqrt(T)
            d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * sqrtT)
            d2 = d1 - iv * sqrtT
            phi_d1 = norm.pdf(d1)
            if direction.lower() == "call":
                delta = float(norm.cdf(d1))
                theta = (-S * phi_d1 * iv / (2 * sqrtT) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                delta = float(-norm.cdf(-d1))
                theta = (-S * phi_d1 * iv / (2 * sqrtT) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            gamma = float(phi_d1 / (S * iv * sqrtT))
            vega = float(S * phi_d1 * sqrtT / 100)
        else:
            delta = float(row["_delta"])
            theta = gamma = vega = 0.0

        return {
            "strike":         K,
            "premium":        round(mid, 2),
            "delta":          round(abs(delta), 3),
            "theta":          round(theta, 4),
            "gamma":          round(gamma, 4),
            "vega":           round(vega, 3),
            "sigma":          round(iv, 3),
            "dte":            dte_actual,
            "T":              T,
            "direction":      direction,
            "underlying":     S,
            "expiry":         best_exp,
            "bid":            round(bid, 2),
            "ask":            round(ask, 2),
            "volume":         volume,
            "open_interest":  oi,
            "contract":       contract,
        }
    except Exception:
        return None


# ── Liquidity filter ──────────────────────────────────────────────────────────

def _passes_liquidity(opt: dict, params: dict) -> tuple[bool, str]:
    """
    Hard gate on real option market quality.
    Eliminates illiquid / wide-spread contracts that are lottery-ticket territory.
    """
    min_oi  = params.get("min_oi",  200)
    min_vol = params.get("min_volume", 10)
    max_spread_pct = params.get("max_spread_pct", 0.12)  # 12% of mid

    oi  = opt.get("open_interest", 0) or 0
    vol = opt.get("volume", 0) or 0
    bid = opt.get("bid", 0) or 0
    ask = opt.get("ask", 0) or 0
    mid = opt.get("premium", 0) or 0.01

    if oi < min_oi:
        return False, f"OI={oi} < {min_oi}"
    if vol < min_vol:
        return False, f"vol={vol} < {min_vol}"
    spread_pct = (ask - bid) / max(mid, 0.01)
    if spread_pct > max_spread_pct:
        return False, f"spread={spread_pct:.0%} > {max_spread_pct:.0%}"
    return True, ""


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

        opt = _fetch_real_option_data(
            symbol=sym,
            direction=direction,
            target_dte=dte,
            delta_target=p.get("delta_target", 0.55),
        )
        if opt is None:
            continue

        passes, reason = _passes_greeks(opt, p)
        liq_ok, liq_reason = _passes_liquidity(opt, p)

        theta_pct = abs(opt["theta"]) / max(opt["premium"], 0.01)
        tv_ratio = abs(opt["theta"]) / max(opt["vega"], 1e-9)

        rows.append({
            "Symbol":       sym,
            "Direction":    direction.upper(),
            "Score":        row["score"],
            "Close":        round(close, 2),
            "Expiry":       opt.get("expiry", "—"),
            "DTE":          opt.get("dte", int(dte)),
            "Strike":       int(opt["strike"]),
            "Bid":          opt.get("bid", "—"),
            "Ask":          opt.get("ask", "—"),
            "Premium":      round(opt["premium"], 2),
            "Volume":       opt.get("volume", 0),
            "OI":           opt.get("open_interest", 0),
            "IV":           round(opt.get("sigma", 0) * 100, 1),
            "Delta":        round(abs(opt["delta"]), 3),
            "Theta/day":    round(opt["theta"], 4),
            "Gamma":        round(opt["gamma"], 4),
            "Vega/1%":      round(opt["vega"], 3),
            "θ/Prem %":     round(theta_pct * 100, 2),
            "θ/Vega":       round(tv_ratio, 3),
            "Greeks OK":    "✅" if passes else f"❌ {reason}",
            "Liq OK":       "✅" if liq_ok else f"❌ {liq_reason}",
            "_passes_liq":  liq_ok,
            "Contract":     opt.get("contract", ""),
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
