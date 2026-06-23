"""
Signal generation: flatten 2-D feature arrays into a list of signal rows,
applying GO Score threshold and optional quality hard-filters.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.options_sim import bs_call_price, bs_delta


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

    rows = []
    for i in range(1, n_dates):
        for j in range(n_syms):
            gs = go_score[i, j]
            if np.isnan(gs) or gs < go_threshold:
                continue

            # ── Hard filters ──────────────────────────────────────────────────
            if require_chg_positive:
                c = chg_pct[i, j]
                if np.isnan(c) or c <= 0:
                    continue

            if require_above_ema20:
                e = ema20_flag[i, j]
                if np.isnan(e) or e < 1.0:
                    continue

            if require_spy_bull:
                s = spy_bull[i, j]
                if np.isnan(s) or s < 1.0:
                    continue

            if min_mom5d > -999.0:
                m = mom5[i, j]
                if np.isnan(m) or m < min_mom5d:
                    continue

            rsi = rsi14[i, j]
            if not np.isnan(rsi) and not (min_rsi <= rsi <= max_rsi):
                continue

            if min_vol_ratio > 0:
                vr = vol_ratio[i, j]
                if np.isnan(vr) or vr < min_vol_ratio:
                    continue

            if min_rel_str > -999.0:
                rs = rel_str[i, j]
                if np.isnan(rs) or rs < min_rel_str:
                    continue

            spot = close[i, j]
            iv   = iv_est[i, j]
            if np.isnan(spot) or np.isnan(iv) or spot <= 0:
                continue

            strike = round(spot * (1.0 + otm_pct / 100.0), 2)
            prem   = float(bs_call_price(spot, strike, T, r, iv))
            if prem <= 0 or prem > max_premium:
                continue

            delta = float(bs_delta(spot, strike, T, r, iv))
            if not (delta_min <= delta <= delta_max):
                continue

            rows.append({
                "date":          dates[i],
                "symbol":        symbols[j],
                "spot":          round(float(spot), 4),
                "strike":        strike,
                "iv":            round(float(iv), 4),
                "dte":           dte_target,
                "entry_premium": round(prem, 4),
                "delta":         round(delta, 4),
                "go_score":      float(gs),
                "vol_ratio":     round(float(vol_ratio[i, j]), 3) if not np.isnan(vol_ratio[i, j]) else np.nan,
                "rsi14":         round(float(rsi), 1) if not np.isnan(rsi) else np.nan,
                "rel_str":       round(float(rel_str[i, j]), 3) if not np.isnan(rel_str[i, j]) else np.nan,
                "chg_pct":       round(float(chg_pct[i, j]), 3) if not np.isnan(chg_pct[i, j]) else np.nan,
                "mom5":          round(float(mom5[i, j]), 2) if not np.isnan(mom5[i, j]) else np.nan,
                "date_idx":      i,
                "sym_idx":       j,
            })

    return pd.DataFrame(rows)
