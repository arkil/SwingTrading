"""
Black-Scholes option pricing utilities.
Vectorised NumPy — no Numba needed (called at signal generation time, not in the hot loop).
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm


def bs_call_price(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,   # years to expiry
    r: float,
    sigma: np.ndarray | float,
) -> np.ndarray:
    """Black-Scholes call price. Returns 0 where T <= 0 (expired)."""
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    with np.errstate(invalid="ignore", divide="ignore"):
        sqrtT = np.sqrt(np.where(T > 0, T, np.nan))
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    price = np.where(T <= 0, np.maximum(S - K, 0.0), price)
    return np.maximum(price, 0.0)


def bs_delta(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float,
    sigma: np.ndarray | float,
) -> np.ndarray:
    """Black-Scholes call delta N(d1)."""
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    with np.errstate(invalid="ignore", divide="ignore"):
        sqrtT = np.sqrt(np.where(T > 0, T, np.nan))
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    delta = norm.cdf(d1)
    delta = np.where(T <= 0, np.where(S > K, 1.0, 0.0), delta)
    return delta


def reprice_option(
    entry_spot:     float,
    current_spot:   float,
    strike:         float,
    entry_dte:      int,     # calendar days at entry
    days_held:      int,     # how many days since entry
    iv:             float,
    r:              float = 0.05,
) -> float:
    """Reprice a call option after `days_held` days."""
    remaining_T = max(entry_dte - days_held, 0) / 365.0
    return float(bs_call_price(current_spot, strike, remaining_T, r, iv))


def simulate_option_pnl(
    spot_series:  np.ndarray,  # daily closes from entry day onward
    strike:       float,
    entry_dte:    int,         # 0 = same-day expiry; priced with 6.5h of time value at entry
    iv:           float,
    max_premium:  float,
    stop_pct:     float = 0.50,   # exit if value drops to this fraction of entry premium
    target_pct:   float = 2.00,   # exit if value reaches this multiple of entry premium
    r:            float = 0.05,
) -> dict:
    """
    Simulate holding one call contract from day 0 through expiry (or stop/target).

    Returns dict with:
        entry_premium, exit_premium, return_pct, exit_reason, days_held
    """
    # For 0DTE, option still has ~6.5 market hours of time value at the open
    entry_T = max(entry_dte, 6.5 / 24) / 365.0
    entry_premium = float(bs_call_price(spot_series[0], strike, entry_T, r, iv))

    if entry_premium <= 0 or entry_premium > max_premium:
        return {"entry_premium": entry_premium, "exit_premium": np.nan,
                "return_pct": np.nan, "exit_reason": "filtered", "days_held": 0}

    stop_value   = entry_premium * stop_pct
    target_value = entry_premium * target_pct

    for day in range(1, len(spot_series)):
        remaining_dte = max(entry_dte - day, 0)
        value = float(bs_call_price(spot_series[day], strike, remaining_dte / 365.0, r, iv))

        if value <= stop_value:
            return {"entry_premium": entry_premium, "exit_premium": value,
                    "return_pct": (value / entry_premium - 1) * 100,
                    "exit_reason": "stop", "days_held": day}

        if value >= target_value:
            return {"entry_premium": entry_premium, "exit_premium": value,
                    "return_pct": (value / entry_premium - 1) * 100,
                    "exit_reason": "target", "days_held": day}

        if remaining_dte == 0:
            intrinsic = max(spot_series[day] - strike, 0.0)
            return {"entry_premium": entry_premium, "exit_premium": intrinsic,
                    "return_pct": (intrinsic / entry_premium - 1) * 100,
                    "exit_reason": "expiry", "days_held": day}

    # Ran out of spot_series before expiry — use last available price
    last_dte = max(entry_dte - (len(spot_series) - 1), 0)
    value = float(bs_call_price(spot_series[-1], strike, last_dte / 365.0, r, iv))
    return {"entry_premium": entry_premium, "exit_premium": value,
            "return_pct": (value / entry_premium - 1) * 100,
            "exit_reason": "end_of_data", "days_held": len(spot_series) - 1}
