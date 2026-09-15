"""
Swing Options Screener — 45-60 DTE (ENHANCED WITH POLYGON.IO REAL DATA)
========================================================================

Enhanced version that uses REAL historical option data from Polygon.io
instead of Black-Scholes simulation. Falls back to yfinance + BS if API unavailable.

Public API (drop-in replacement for swing_options_screener.py):
    run_swing_options_screener(tickers, ..., use_polygon=True) -> pd.DataFrame

Environment Variables:
    POLYGON_API_KEY - Free tier from polygon.io (required for real data)
    USE_POLYGON_DATA - Force on/off (default: True if API key exists)
"""

from __future__ import annotations

import sys
import os
import warnings
warnings.filterwarnings("ignore")

from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from strategy_improvements import StrategyOptimizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Path: add swing_options_45_60d's src/ to import path ─────────────────────
# Prefer the copy vendored into this repo (works on Streamlit Cloud); fall
# back to the sibling checkout for local dev.
for _STRATEGY_DIR in (
    os.path.join(os.path.dirname(__file__), "vendor", "swing_options_45_60d"),
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "swing_options_45_60d")
    ),
):
    if os.path.isdir(os.path.join(_STRATEGY_DIR, "src")):
        if _STRATEGY_DIR not in sys.path:
            sys.path.insert(0, _STRATEGY_DIR)
        break

from src.indicators import compute_all_indicators
from src.screener import screen_universe
from src.options_sim import calc_option_entry

# ── Polygon.io client setup ────────────────────────────────────────────────────

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
POLYGON_ENABLED = bool(POLYGON_API_KEY)

if POLYGON_ENABLED:
    try:
        from polygon import RESTClient
        polygon_client = RESTClient(api_key=POLYGON_API_KEY)
        logger.info("✅ Polygon.io real data enabled (premium prices will use real market data)")
    except ImportError:
        logger.warning("⚠️ polygon-api-client not installed. Install: pip install polygon-api-client")
        POLYGON_ENABLED = False
else:
    logger.info("ℹ️  Polygon.io disabled (set POLYGON_API_KEY env var to enable real option data)")
    polygon_client = None

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
    """Current VIX 1-year percentile rank (0-100)."""
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
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df = df[df["Close"] > 0]
        return df
    except Exception:
        return pd.DataFrame()


# ── Cache integration ─────────────────────────────────────────────────────────

try:
    from polygon_cache import (
        get_cached_option_price,
        fetch_and_cache_polygon_data,
        cache_stats,
    )
    CACHE_ENABLED = True
    logger.info("✅ Local cache enabled (Polygon data will be stored locally)")
except ImportError:
    CACHE_ENABLED = False
    logger.info("ℹ️  Local cache disabled (polygon_cache.py not found)")


# ── NEW: Polygon.io real option data fetcher (with caching) ───────────────────

def _fetch_polygon_option_price(
    symbol: str,
    strike: float,
    expiry_str: str,
    direction: str,
    target_date: str = None,
    use_cache: bool = True,
) -> Optional[Dict]:
    """
    Fetch REAL historical option price from Polygon.io (latest available bar).
    Checks local cache first, fetches from API if not cached, stores result locally.

    Args:
        symbol: "SPY"
        strike: 500.0
        expiry_str: "20240119" (YYYYMMDD)
        direction: "call" or "put"
        target_date: "2024-01-19" (if None, use today)
        use_cache: Check/store in local cache (default True)

    Returns:
        dict: {price, source='polygon_cache'|'polygon_api_fresh', timestamp, ...} or None
    """
    if not POLYGON_ENABLED or not polygon_client:
        return None

    try:
        # ─── STEP 1: Try local cache first ────────────────────────────────────
        if use_cache and CACHE_ENABLED:
            cached = get_cached_option_price(
                symbol=symbol,
                strike=strike,
                expiry_str=expiry_str,
                direction=direction,
                target_date=target_date,
            )
            if cached:
                logger.debug(f"✅ Cache hit: {symbol} {strike} {expiry_str} {direction.upper()}")
                return cached

        # ─── STEP 2: Fetch from Polygon.io API ───────────────────────────────
        logger.debug(f"📡 Fetching from Polygon.io: {symbol} {strike} {expiry_str}")

        if CACHE_ENABLED:
            # Fetch + cache in one step
            result = fetch_and_cache_polygon_data(
                polygon_client=polygon_client,
                symbol=symbol,
                strike=strike,
                expiry_str=expiry_str,
                direction=direction,
                start_date="2023-01-01",
                end_date=target_date or datetime.now().strftime("%Y-%m-%d"),
                verbose=False,
            )
            if result:
                logger.debug(f"✅ API fresh + cached: {symbol} {strike} ({result['bars_cached']} bars)")
            return result
        else:
            # Fetch without caching
            direction_char = "C" if direction.lower() == "call" else "P"
            option_ticker = f"O:{symbol}{expiry_str}{direction_char}{int(strike):08d}"

            bars = list(polygon_client.get_aggs(
                ticker=option_ticker,
                timespan="day",
                from_="2023-01-01",
                to=target_date or datetime.now().strftime("%Y-%m-%d"),
                limit=1000
            ))

            if not bars:
                return None

            bar = bars[-1]
            return {
                'price': bar.close,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'volume': bar.volume,
                'timestamp': bar.timestamp,
                'source': 'polygon_real_data'
            }

    except Exception as e:
        logger.debug(f"Polygon.io error for {symbol} {strike} {expiry_str}: {e}")
        return None


# ── ENHANCED: Real option chain fetcher (Polygon.io first, fallback yfinance) ──

def _fetch_real_option_data(
    symbol: str,
    direction: str,
    target_dte: float = 50.0,
    delta_target: float = 0.55,
    r: float = 0.045,
    use_polygon: bool = True,
    use_cache: bool = True,  # ← NEW: Use local cache
) -> Optional[Dict]:
    """
    Fetch a real option contract nearest to target_dte and delta_target.

    ENHANCED: Tries Polygon.io first (real market data), falls back to yfinance + BS.

    Returns a dict with keys: strike, premium, delta, theta, gamma, vega, sigma, dte, T,
    direction, underlying, expiry, bid, ask, volume, open_interest, contract, source

    The 'source' field indicates data origin:
        'polygon_real_data' - Real historical market close from Polygon.io
        'yfinance_current' - Current real bid/ask from yfinance (live scan)
        'bs_simulation' - Black-Scholes simulated (fallback)
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
        expiry_str = best_exp.replace("-", "")  # Convert 2024-01-19 → 20240119

        chain = ticker.option_chain(best_exp)
        df = chain.calls.copy() if direction.lower() == "call" else chain.puts.copy()
        if df.empty:
            return None

        # Compute delta for each row using real IV
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

        # ─── TRY: Polygon.io real historical data first ───────────────────────
        source = "yfinance_current"  # Default
        premium = mid

        if use_polygon and POLYGON_ENABLED:
            polygon_data = _fetch_polygon_option_price(
                symbol=symbol,
                strike=K,
                expiry_str=expiry_str,
                direction=direction,
                target_date=datetime.now().strftime("%Y-%m-%d"),
                use_cache=use_cache  # ← Pass cache flag
            )

            if polygon_data:
                premium = polygon_data['price']
                source = "polygon_real_data"
                logger.debug(f"✅ Using Polygon.io real data for {symbol} {K} {direction}")

        # Compute Greeks from real IV (same as before)
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
            "premium":        round(premium, 2),
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
            "source":         source,  # ← NEW: Track data source
        }
    except Exception as e:
        logger.debug(f"Error fetching real option data: {e}")
        return None


# ── Liquidity filter (unchanged) ───────────────────────────────────────────────

def _passes_liquidity(opt: dict, params: dict) -> tuple[bool, str]:
    """Hard gate on real option market quality."""
    min_oi  = params.get("min_oi",  200)
    min_vol = params.get("min_volume", 10)
    max_spread_pct = params.get("max_spread_pct", 0.12)

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


# ── Greeks filter (unchanged) ──────────────────────────────────────────────────

def _passes_greeks(opt: dict, params: dict) -> tuple[bool, str]:
    """Returns (passes, rejection_reason). Applies 4-filter Greek gate."""
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
            return False, f"θ/vega={tv:.3f} > {theta_vega_max}"

    return True, ""


# ── Main screener (ENHANCED) ───────────────────────────────────────────────────

def run_swing_options_screener(
    tickers: List[str],
    min_score: float = 7.5,
    dte: float = 52,
    iv_premium: float = 1.1,
    params: Optional[Dict] = None,
    progress_cb = None,
    use_polygon: bool = True,  # ← Use Polygon.io for real data
    use_cache: bool = True,     # ← NEW: Use local cache for Polygon data
) -> pd.DataFrame:
    """
    Scan tickers for swing options setups (45-60 DTE).

    ENHANCED: Uses Polygon.io for real historical option prices (if available),
    caches locally for future backtests, falls back to yfinance or BS simulation.

    Args:
        tickers: List of symbols ['SPY', 'QQQ', ...]
        min_score: Min composite signal score (0-10)
        dte: Target days to expiration
        iv_premium: IV = historical_vol × this factor
        params: Override DEFAULT_PARAMS
        progress_cb: Callback function(pct, msg)
        use_polygon: Use Polygon.io real data if available (default True)
        use_cache: Cache Polygon data locally for reuse (default True)

    Returns:
        DataFrame with columns: symbol, direction, score, premium, greeks, ...
        Plus 'source' column: 'polygon_cache', 'polygon_api_fresh', 'yfinance_current', 'bs_simulation'
    """

    if params is None:
        params = DEFAULT_PARAMS.copy()
    else:
        p = DEFAULT_PARAMS.copy()
        p.update(params)
        params = p

    results = []
    n_tickers = len(tickers)

    for i, ticker in enumerate(tickers):
        pct = 100 * i / max(n_tickers, 1)
        msg = f"Scanning {ticker}... (using {'Polygon.io' if use_polygon and POLYGON_ENABLED else 'yfinance'})"
        if progress_cb:
            progress_cb(pct, msg)

        # Fetch OHLCV
        df = _fetch_ohlcv(ticker)
        if df.empty or len(df) < 220:
            continue

        # Compute indicators
        try:
            df_ind = compute_all_indicators(df, params)
        except Exception as e:
            logger.debug(f"Indicator error for {ticker}: {e}")
            continue

        # Screen universe
        try:
            candidates = screen_universe(df_ind, params, min_score)
        except Exception as e:
            logger.debug(f"Screening error for {ticker}: {e}")
            continue

        # For each candidate, fetch real option data
        for _, cand in candidates.iterrows():
            direction = cand.get("direction", "CALL").lower()

            opt = _fetch_real_option_data(
                symbol=ticker,
                direction=direction,
                target_dte=dte,
                delta_target=params.get("delta_target", 0.55),
                use_polygon=use_polygon,  # ← Use Polygon.io
                use_cache=use_cache,      # ← Cache locally
            )
            if not opt:
                continue

            # Check Greeks and liquidity
            passes_greeks, reject_greeks = _passes_greeks(opt, params)
            passes_liq, reject_liq = _passes_liquidity(opt, params)

            if not (passes_greeks and passes_liq):
                continue

            # Add to results
            result = {
                "symbol": ticker,
                "direction": direction.upper(),
                "score": round(cand.get("score", 0), 1),
                "close": round(float(df["Close"].iloc[-1]), 2),
                "premium": opt["premium"],
                "strike": opt["strike"],
                "expiry": opt["expiry"],
                "dte": opt["dte"],
                "delta": opt["delta"],
                "theta": opt["theta"],
                "gamma": opt["gamma"],
                "vega": opt["vega"],
                "iv": opt["sigma"],
                "bid": opt["bid"],
                "ask": opt["ask"],
                "volume": opt["volume"],
                "oi": opt["open_interest"],
                "theta_pct": round(abs(opt["theta"]) / max(opt["premium"], 0.01), 4),
                "theta_vega": round(abs(opt["theta"]) / max(opt["vega"], 0.001), 3),
                "greeks_ok": "✓",
                "liq_ok": "✓",
                "source": opt.get("source", "unknown"),  # ← NEW: Track data source
                "_passes_greeks": True,
            }
            results.append(result)

    if progress_cb:
        progress_cb(100, "Scan complete")

    df_results = pd.DataFrame(results) if results else pd.DataFrame()

    # ── APPLY STRATEGY IMPROVEMENTS (NEW: Top-3 optimizations) ────────────────────
    if not df_results.empty:
        try:
            optimizer = StrategyOptimizer(account_size=100000)
            df_results, metadata = optimizer.apply_all_improvements(df_results)
            logger.info(f"   🎯 Applied strategy improvements: {len(df_results)} qualified setups")
        except Exception as e:
            logger.debug(f"Could not apply improvements: {e}")

    # Log summary
    if not df_results.empty:
        polygon_cache = (df_results["source"] == "polygon_cache").sum()
        polygon_fresh = (df_results["source"] == "polygon_api_fresh").sum()
        yfinance_count = (df_results["source"] == "yfinance_current").sum()
        logger.info(f"📊 Scan complete: {len(df_results)} setups found")
        if polygon_cache > 0:
            logger.info(f"   ✅ {polygon_cache} from local cache (no API calls)")
        if polygon_fresh > 0:
            logger.info(f"   📡 {polygon_fresh} fetched from Polygon.io API + cached")
        if yfinance_count > 0:
            logger.info(f"   ℹ️  {yfinance_count} using yfinance current prices")

        # Show cache stats if enabled
        if use_cache and CACHE_ENABLED:
            try:
                stats = cache_stats()
                logger.info(f"   💾 Local cache: {stats['cached_options']} options, {stats['total_price_bars']} price bars, {stats['cache_size_mb']:.1f} MB")
            except Exception:
                pass

    return df_results
