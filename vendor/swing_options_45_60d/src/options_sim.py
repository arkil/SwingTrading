"""
Black-Scholes options simulator for backtesting the swing_options_45_60d
strategy (no historical option chain available, so entries are simulated).

Public API
----------
    calc_option_entry(underlying_price, direction, dte, hist_vol, iv_premium,
                       delta_target, r=0.045) -> dict | None

Greeks formulas mirror Scripts/SwingTrading/swing_options_screener.py's
_fetch_real_option_data so live and simulated backtest greeks are computed
identically.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _bs_delta(S: float, K: float, T: float, sigma: float, r: float, direction: str) -> float:
    if sigma <= 0 or K <= 0 or T <= 0:
        return 0.0
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    return float(norm.cdf(d1)) if direction == "call" else float(-norm.cdf(-d1))


def _solve_strike_for_delta(S: float, T: float, sigma: float, r: float, direction: str, delta_target: float) -> float:
    """Find strike K such that |delta(K)| ~= delta_target via bisection."""
    target = delta_target if direction == "call" else -delta_target

    def f(K):
        return _bs_delta(S, K, T, sigma, r, direction) - target

    lo, hi = S * 0.5, S * 1.8
    try:
        return float(brentq(f, lo, hi, xtol=1e-4))
    except ValueError:
        # Fallback: pick the strike in [lo, hi] closest to target delta by grid search
        grid = np.linspace(lo, hi, 400)
        deltas = np.array([_bs_delta(S, K, T, sigma, r, direction) for K in grid])
        idx = int(np.argmin(np.abs(deltas - target)))
        return float(grid[idx])


def price_option(S: float, K: float, T: float, sigma: float, r: float, direction: str) -> dict:
    """
    Black-Scholes price + greeks for a known strike/expiry, used to reprice
    an already-open simulated position as the underlying moves and time
    passes (sigma held constant at the entry IV — no vol-of-vol simulation).
    """
    direction = direction.lower()
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return {"premium": 0.0, "delta": 0.0, "theta": 0.0, "gamma": 0.0, "vega": 0.0}

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    phi_d1 = norm.pdf(d1)

    if direction == "call":
        delta = float(norm.cdf(d1))
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        theta = (-S * phi_d1 * sigma / (2 * sqrtT) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = float(-norm.cdf(-d1))
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        theta = (-S * phi_d1 * sigma / (2 * sqrtT) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    gamma = float(phi_d1 / (S * sigma * sqrtT))
    vega = float(S * phi_d1 * sqrtT / 100)

    return {
        "premium": round(float(max(price, 0.01)), 2),
        "delta": round(abs(delta), 3),
        "theta": round(float(theta), 4),
        "gamma": round(gamma, 4),
        "vega": round(vega, 3),
    }


def calc_option_entry(
    underlying_price: float,
    direction: str,
    dte: float,
    hist_vol: float,
    iv_premium: float = 1.10,
    delta_target: float = 0.55,
    r: float = 0.045,
) -> dict | None:
    """
    Simulate a delta-matched option entry using Black-Scholes.

    Args:
        underlying_price: current price of the underlying (S)
        direction: "call" or "put"
        dte: days to expiration at entry
        hist_vol: annualized historical volatility (e.g. 21-day realized vol)
        iv_premium: multiplier applied to hist_vol to approximate market IV
        delta_target: target absolute delta for strike selection
        r: risk-free rate

    Returns:
        dict with strike, premium, delta, theta, gamma, vega, sigma, dte, T,
        direction, underlying — same key set as the live wrapper's
        _fetch_real_option_data — or None if inputs are invalid.
    """
    direction = direction.lower()
    if direction not in ("call", "put"):
        return None
    S = float(underlying_price)
    if S <= 0 or dte <= 0 or hist_vol <= 0:
        return None

    T = dte / 365.0
    sigma = max(hist_vol * iv_premium, 0.05)

    K = _solve_strike_for_delta(S, T, sigma, r, direction, delta_target)
    if K <= 0:
        return None

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    phi_d1 = norm.pdf(d1)

    if direction == "call":
        delta = float(norm.cdf(d1))
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        theta = (-S * phi_d1 * sigma / (2 * sqrtT) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = float(-norm.cdf(-d1))
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        theta = (-S * phi_d1 * sigma / (2 * sqrtT) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    gamma = float(phi_d1 / (S * sigma * sqrtT))
    vega = float(S * phi_d1 * sqrtT / 100)

    return {
        "strike": round(float(K), 2),
        "premium": round(float(max(price, 0.01)), 2),
        "delta": round(abs(delta), 3),
        "theta": round(float(theta), 4),
        "gamma": round(gamma, 4),
        "vega": round(vega, 3),
        "sigma": round(sigma, 3),
        "dte": int(dte),
        "T": T,
        "direction": direction,
        "underlying": S,
    }
