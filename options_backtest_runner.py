"""
Options 45-60 DTE Backtest Runner
===================================
Dashboard wrapper around strategies/swing_options_45_60d/src/backtest.py.

Loads OHLCV, computes indicators + signals, runs event-driven backtest,
returns a result dict suitable for the dashboard render function.
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
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ── Strategy path ──────────────────────────────────────────────────────────────
_STRATEGY_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "swing_options_45_60d")
)
if _STRATEGY_DIR not in sys.path:
    sys.path.insert(0, _STRATEGY_DIR)

from src.indicators  import compute_all_indicators
from src.signals     import generate_signals
from src.backtest    import run_backtest
from src.options_sim import calc_option_entry

# ── Default params (mirrors config.yaml) ──────────────────────────────────────
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
    "delta_target": 0.55,
    "delta_min": 0.40, "delta_max": 0.70,
    "theta_max_daily_pct": 0.015,
    "gamma_min": 0.005, "gamma_max": 0.050,
    "theta_vega_ratio_max": 0.40,
    "max_entry_sigma": 0.50,
    "min_signals_required": 4,
    "atr_stop_multiplier": 2.0,
    "atr_target_multiplier": 4.0,
    "iv_rank_min": 5,
    "iv_rank_max_call": 35,
    "iv_rank_max_put": 65,
    "percent_per_trade": 5.0,
}

_yf_lock = Lock()


# ── Data helpers ───────────────────────────────────────────────────────────────

def _fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        with _yf_lock:
            df = yf.download(ticker, start=start, end=end,
                             interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index.date)
        return df
    except Exception:
        return pd.DataFrame()


def _fetch_vix(start: str, end: str) -> pd.Series:
    df = _fetch_ohlcv("^VIX", start, end)
    return df["Close"] if not df.empty else pd.Series(dtype=float)


# ── Main public API ────────────────────────────────────────────────────────────

def run_options_backtest(
    tickers:          List[str],
    start_date:       str  = None,
    end_date:         str  = None,
    initial_capital:  float = 25_000.0,
    tp_pct:           float = 0.50,
    sl_pct:           float = 0.25,
    dte_entry:        float = 50.0,
    max_hold_days:    int   = 21,
    max_positions:    int   = 4,
    iv_premium:       float = 1.10,
    use_regime_filter: bool = True,
    min_score_override: float = None,
    params:           dict = None,
    max_workers:      int   = 8,
    progress_cb:      Optional[Callable] = None,
) -> Dict:
    """
    Run walk-forward options backtest across a list of tickers.

    progress_cb(pct: float, msg: str) — optional progress callback.

    Returns dict:
        trades_df     — DataFrame of all closed trades
        equity_df     — Date-indexed equity curve DataFrame
        metrics       — summary metrics dict
        per_ticker    — {symbol: {trades, win_rate, avg_pnl, total_pnl}}
        initial_capital
    """
    if end_date is None:
        end_date   = datetime.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

    p = {**DEFAULT_PARAMS, **(params or {})}
    warmup_start = (
        datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=220)
    ).strftime("%Y-%m-%d")

    n_tickers = len(tickers)
    _cb = progress_cb or (lambda pct, msg: None)

    _cb(0.02, "Fetching VIX…")
    vix_series = _fetch_vix(warmup_start, end_date)

    # Compute IV rank from VIX
    if not vix_series.empty:
        from src.data_loader import compute_iv_rank
        iv_rank_series = compute_iv_rank(vix_series)
    else:
        iv_rank_series = None

    _cb(0.05, f"Fetching {n_tickers} tickers…")

    # ── Fetch + compute indicators in parallel ─────────────────────────────────
    all_data: Dict[str, pd.DataFrame] = {}
    done_count = 0

    def _process_ticker(sym: str):
        df = _fetch_ohlcv(sym, warmup_start, end_date)
        if df.empty or len(df) < 220:
            return sym, None
        try:
            df = compute_all_indicators(df, p)
        except Exception:
            return sym, None

        # Align IV rank to df index
        if iv_rank_series is not None:
            aligned_iv = iv_rank_series.reindex(df.index, method="ffill").fillna(35.0)
        else:
            aligned_iv = None

        try:
            df = generate_signals(df, p, iv_rank_series=aligned_iv)
        except Exception:
            return sym, None

        return sym, df

    tickers_with_spy = list(dict.fromkeys(["SPY"] + tickers))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_process_ticker, sym): sym for sym in tickers_with_spy}
        for fut in as_completed(futs):
            done_count += 1
            sym, df = fut.result()
            if df is not None:
                all_data[sym] = df
            _cb(0.05 + 0.55 * done_count / len(tickers_with_spy),
                f"Processed {done_count}/{len(tickers_with_spy)}")

    if not all_data:
        return {
            "trades_df": pd.DataFrame(),
            "equity_df": pd.DataFrame(),
            "metrics":   {},
            "per_ticker": {},
            "initial_capital": initial_capital,
        }

    _cb(0.62, "Running backtest simulation…")

    trades_df, equity_df = run_backtest(
        all_data          = all_data,
        params            = p,
        initial_capital   = initial_capital,
        tp_pct            = tp_pct,
        sl_pct            = sl_pct,
        dte_entry         = dte_entry,
        max_hold_days     = max_hold_days,
        max_positions     = max_positions,
        iv_premium        = iv_premium,
        use_regime_filter = use_regime_filter,
        min_score_override = min_score_override,
        vix_series        = vix_series if not vix_series.empty else None,
    )

    _cb(0.92, "Computing metrics…")

    metrics = {}
    per_ticker: Dict = {}

    if not trades_df.empty:
        wins      = trades_df[trades_df["pnl"] > 0]
        losses    = trades_df[trades_df["pnl"] <= 0]
        total_pnl = float(trades_df["pnl"].sum())
        win_rate  = len(wins) / len(trades_df) * 100 if len(trades_df) else 0
        avg_win   = float(wins["pnl"].mean())   if len(wins)   else 0
        avg_loss  = float(losses["pnl"].mean()) if len(losses) else 0
        pf        = abs(wins["pnl"].sum() / losses["pnl"].sum()) \
                    if losses["pnl"].sum() != 0 else float("inf")

        final_equity  = float(equity_df["total_equity"].iloc[-1]) \
                        if not equity_df.empty else initial_capital
        total_return  = (final_equity - initial_capital) / initial_capital * 100

        # Max drawdown from equity curve
        eq = equity_df["total_equity"].values if not equity_df.empty else [initial_capital]
        peak = np.maximum.accumulate(eq)
        dd   = (eq - peak) / (peak + 1e-9) * 100
        max_dd = float(dd.min())

        metrics = {
            "total_trades":  len(trades_df),
            "win_rate":      round(win_rate, 1),
            "total_pnl":     round(total_pnl, 2),
            "total_return":  round(total_return, 2),
            "avg_win":       round(avg_win, 2),
            "avg_loss":      round(avg_loss, 2),
            "profit_factor": round(pf, 2) if np.isfinite(pf) else None,
            "max_drawdown":  round(max_dd, 2),
            "final_equity":  round(final_equity, 2),
            "avg_hold_days": round(float(trades_df["days_held"].mean()), 1),
        }

        for sym, grp in trades_df.groupby("symbol"):
            g_wins = grp[grp["pnl"] > 0]
            per_ticker[sym] = {
                "trades":    len(grp),
                "win_rate":  round(len(g_wins) / len(grp) * 100, 1),
                "total_pnl": round(float(grp["pnl"].sum()), 2),
                "avg_pnl":   round(float(grp["pnl"].mean()), 2),
            }

    _cb(1.0, "Done")

    return {
        "trades_df":       trades_df,
        "equity_df":       equity_df,
        "metrics":         metrics,
        "per_ticker":      per_ticker,
        "initial_capital": initial_capital,
    }
