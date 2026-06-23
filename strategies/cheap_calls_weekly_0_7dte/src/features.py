"""
Feature engineering for cheap_calls_weekly_0_7dte.
Operates on 2-D NumPy arrays (n_dates × n_symbols).
All features computed from data available at EOD — no lookahead.
EOD scanner fires *after* the bar closes, so day-D features CAN use day-D close.
"""

from __future__ import annotations
import numpy as np


# ── Primitives ────────────────────────────────────────────────────────────────

def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """Rolling mean along axis-0 (dates). Returns same shape, NaN for early rows."""
    cs = np.cumsum(arr, axis=0)
    out = np.full_like(arr, np.nan)
    out[w - 1:] = (cs[w - 1:] - np.concatenate([np.zeros((1, arr.shape[1])), cs[:-(w)]], axis=0)) / w
    return out


def _ema_2d(arr: np.ndarray, span: int) -> np.ndarray:
    """EMA along axis-0. arr shape: (n_dates, n_symbols)."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, arr.shape[0]):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi_2d(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI-14 along axis-0."""
    delta = np.diff(close, axis=0, prepend=close[[0]])
    gains  = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    n, m = close.shape
    avg_gain = np.full((n, m), np.nan)
    avg_loss = np.full((n, m), np.nan)

    avg_gain[period] = np.nanmean(gains[1:period + 1], axis=0)
    avg_loss[period] = np.nanmean(losses[1:period + 1], axis=0)

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _chg_pct(close: np.ndarray) -> np.ndarray:
    """Day-over-day % change. Index 0 is NaN."""
    out = np.full_like(close, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = (close[1:] - close[:-1]) / close[:-1] * 100.0
    return out


def _rolling_vol_ratio(volume: np.ndarray, window: int = 63) -> np.ndarray:
    """volume / rolling_mean(volume, window). Values < window are NaN."""
    avg = _rolling_mean(volume, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        return volume / np.where(avg == 0, np.nan, avg)


def _hist_vol(close: np.ndarray, window: int = 21) -> np.ndarray:
    """Annualised historical volatility over `window` days."""
    log_ret = np.full_like(close, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_ret[1:] = np.log(close[1:] / np.where(close[:-1] == 0, np.nan, close[:-1]))
    roll_std = np.full_like(close, np.nan)
    for i in range(window, close.shape[0]):
        roll_std[i] = np.nanstd(log_ret[i - window + 1: i + 1], axis=0)
    return roll_std * np.sqrt(252)


# ── Main feature builder ──────────────────────────────────────────────────────

def compute_features(
    arrays: dict,
    spy_close: np.ndarray,       # shape (n_dates,) — SPY close aligned to universe dates
    iv_scaling: float = 1.2,
) -> dict:
    """
    Compute all features needed for GO Score and option simulation.

    Args:
        arrays: output of data_loader.pivot_to_wide()
        spy_close: SPY close prices aligned to the same dates
        iv_scaling: multiply 30d realized vol by this to estimate IV

    Returns dict of 2-D arrays (n_dates × n_symbols):
        chg_pct, spy_chg_pct, rel_str, vol_ratio, rsi14,
        ema20_flag, hist_vol_30, iv_est, go_score
    """
    close  = arrays["close"]    # (n_dates, n_symbols)
    volume = arrays["volume"]

    n = close.shape[0]

    # SPY daily % change, broadcast to (n_dates, n_symbols)
    spy_chg = _chg_pct(spy_close.reshape(-1, 1))  # (n, 1) → broadcasts

    chg_pct  = _chg_pct(close)                     # (n, n_symbols)
    rel_str  = chg_pct - spy_chg                    # outperformance vs SPY
    vol_ratio = _rolling_vol_ratio(volume, 63)
    rsi14    = _rsi_2d(close, 14)
    ema20    = _ema_2d(close, 20)
    ema20_flag = (close > ema20).astype(np.float32) # 1 if above EMA20
    hv30     = _hist_vol(close, 30)
    iv_est   = np.clip(hv30 * iv_scaling, 0.10, 5.0)  # floor 10%, cap 500%

    # ── Simplified GO Score (0–5) ──────────────────────────────────────────
    go = np.zeros_like(close)

    with np.errstate(invalid="ignore"):
        # Rel strength (0–2 pts)
        go = np.where(rel_str >= 2.0, go + 2.0,
             np.where(rel_str >= 0.5, go + 1.0, go))
        # Volume surge (0–2 pts)
        go = np.where(vol_ratio >= 2.0, go + 2.0,
             np.where(vol_ratio >= 1.3, go + 1.0, go))
        # RSI sweet spot (0–1 pt)
        go = np.where((rsi14 >= 45.0) & (rsi14 <= 65.0), go + 1.0, go)

    # Zero out rows where data is incomplete
    bad = np.isnan(rsi14) | np.isnan(vol_ratio) | np.isnan(chg_pct)
    go[bad] = np.nan

    # 5-day % return for momentum confirmation
    mom5 = np.full_like(close, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mom5[5:] = (close[5:] - close[:-5]) / close[:-5] * 100.0

    # SPY bull flag: spy_close > its own 20-day EMA, broadcast to (n, n_syms)
    spy_ema20     = _ema_2d(spy_close.reshape(-1, 1), 20)
    spy_bull_raw  = (spy_close.reshape(-1, 1) > spy_ema20).astype(np.float32)
    spy_bull      = np.broadcast_to(spy_bull_raw, close.shape).copy()

    return {
        "chg_pct":    chg_pct,
        "spy_chg":    np.broadcast_to(spy_chg, close.shape).copy(),
        "rel_str":    rel_str,
        "vol_ratio":  vol_ratio,
        "rsi14":      rsi14,
        "ema20_flag": ema20_flag,
        "iv_est":     iv_est,
        "go_score":   go,
        "mom5":       mom5,
        "spy_bull":   spy_bull,
    }
