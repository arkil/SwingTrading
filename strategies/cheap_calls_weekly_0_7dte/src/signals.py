"""
Signal generation: flatten 2-D feature arrays into a list of signal rows,
applying GO Score threshold and optional quality hard-filters.

Vectorised over the full (date, symbol) grid with NumPy boolean masking --
avoids a Python-level double loop, which dominates runtime when this is
called many times inside a walk-forward grid search (see walkforward.py).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.options_sim import bs_call_price, bs_delta, MIN_ENTRY_PREMIUM


def generate_signals(
    arrays: dict,
    features: dict,
    go_threshold: float,
    max_premium: float,
    delta_min: float,
    delta_max: float,
    otm_pct: float,
    dte_target: int,
    r: float = 0.05,
    # ── Hard entry filters (all default OFF = baseline behaviour) ────────────
    require_chg_positive: bool = False,   # stock must be up today
    require_above_ema20:  bool = False,   # stock must be above 20-day EMA
    require_spy_bull:     bool = False,   # SPY must be above its 20-day EMA
    min_mom5d:            float = -999.0, # min 5-day % return (e.g. 0 = up over last week)
    min_rsi:              float = 0.0,    # hard RSI floor
    max_rsi:              float = 100.0,  # hard RSI ceiling
    min_vol_ratio:        float = 0.0,    # min volume/avg-volume ratio (0 = no filter)
    min_rel_str:          float = -999.0, # min relative strength vs SPY (%)
) -> pd.DataFrame:
    """
    Scan every (date, symbol) pair where go_score >= go_threshold
    and all enabled hard filters pass, then price the simulated call.
    """
    close      = arrays["close"]
    dates      = arrays["dates"]
    symbols    = arrays["symbols"]
    go_score   = features["go_score"]
    iv_est     = features["iv_est"]
    vol_ratio  = features["vol_ratio"]
    rsi14      = features["rsi14"]
    rel_str    = features["rel_str"]
    chg_pct    = features["chg_pct"]
    ema20_flag = features["ema20_flag"]
    mom5       = features.get("mom5",     np.zeros_like(close))
    spy_bull   = features.get("spy_bull", np.ones_like(close))

    n_dates, n_syms = close.shape
    T = max(dte_target, 6.5 / 24) / 365.0   # floor for 0DTE

    with np.errstate(invalid="ignore"):
        mask = ~np.isnan(go_score) & (go_score >= go_threshold)
        mask[0, :] = False  # skip index 0 (no prior-day feature history)

        if require_chg_positive:
            mask &= ~np.isnan(chg_pct) & (chg_pct > 0)
        if require_above_ema20:
            mask &= ~np.isnan(ema20_flag) & (ema20_flag >= 1.0)
        if require_spy_bull:
            mask &= ~np.isnan(spy_bull) & (spy_bull >= 1.0)
        if min_mom5d > -999.0:
            mask &= ~np.isnan(mom5) & (mom5 >= min_mom5d)

        rsi_ok = np.isnan(rsi14) | ((rsi14 >= min_rsi) & (rsi14 <= max_rsi))
        mask &= rsi_ok

        if min_vol_ratio > 0:
            mask &= ~np.isnan(vol_ratio) & (vol_ratio >= min_vol_ratio)
        if min_rel_str > -999.0:
            mask &= ~np.isnan(rel_str) & (rel_str >= min_rel_str)

        mask &= ~np.isnan(close) & (close > 0) & ~np.isnan(iv_est)

    idx_i, idx_j = np.where(mask)
    if len(idx_i) == 0:
        return pd.DataFrame()

    spot = close[idx_i, idx_j]
    iv   = iv_est[idx_i, idx_j]
    strike = np.round(spot * (1.0 + otm_pct / 100.0), 2)

    # Premiums below MIN_ENTRY_PREMIUM are numerically unstable in a
    # BS-simulated pricer (no real bid/ask floor) -- a few-cent entry
    # premium turns a $0.02 move into a >1000% "return" artifact, not
    # real edge. See strategies/_shared/options_sim.py.
    prem  = bs_call_price(spot, strike, T, r, iv)
    delta = bs_delta(spot, strike, T, r, iv)

    ok = (prem >= MIN_ENTRY_PREMIUM) & (prem <= max_premium) & \
         (delta >= delta_min) & (delta <= delta_max)
    if not ok.any():
        return pd.DataFrame()

    idx_i, idx_j = idx_i[ok], idx_j[ok]
    spot, strike, iv, prem, delta = spot[ok], strike[ok], iv[ok], prem[ok], delta[ok]

    df = pd.DataFrame({
        "date":          dates[idx_i],
        "symbol":        symbols[idx_j],
        "spot":          np.round(spot, 4),
        "strike":        strike,
        "iv":            np.round(iv, 4),
        "dte":           dte_target,
        "entry_premium": np.round(prem, 4),
        "delta":         np.round(delta, 4),
        "go_score":      go_score[idx_i, idx_j],
        "vol_ratio":     vol_ratio[idx_i, idx_j],
        "rsi14":         rsi14[idx_i, idx_j],
        "rel_str":       rel_str[idx_i, idx_j],
        "chg_pct":       chg_pct[idx_i, idx_j],
        "mom5":          mom5[idx_i, idx_j],
        "date_idx":      idx_i,
        "sym_idx":       idx_j,
    })
    return df
