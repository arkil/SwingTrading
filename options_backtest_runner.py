"""
Options 45-60 DTE Backtest Runner
===================================
Dashboard wrapper around strategies/swing_options_45_60d/src/backtest_engine.py.

Fetches OHLCV, builds a CBT-style config dict, runs BacktestEngine, and
reshapes the results into the trades_df/equity_df/metrics/per_ticker shape
render_options_backtest() (dashboard.py) expects.
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

from src.backtest_engine import BacktestEngine

# ── Default params (mirrors strategies/swing_options_45_60d/config.yaml) ───────
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
    "delta_target": 0.55,
    "delta_min": 0.40, "delta_max": 0.70,
    "theta_max_daily_pct": 0.015,
    "gamma_min": 0.005, "gamma_max": 0.050,
    "theta_vega_ratio_max": 0.40,
    "max_entry_sigma": 0.50,
    "iv_rank_max": 30,
    "dte_entry": 52,
    "dte_exit_remaining": 21,
    "min_score": 7.5,
    "percent_per_trade": 2.0,
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


# ── Main public API ────────────────────────────────────────────────────────────

def run_options_backtest(
    tickers:          List[str],
    start_date:       str  = None,
    end_date:         str  = None,
    initial_capital:  float = 25_000.0,
    tp_pct:           float = 1.00,
    sl_pct:           float = 0.50,
    dte_entry:        float = 52.0,
    max_hold_days:    int   = 21,
    max_positions:    int   = 3,
    iv_premium:       float = 1.10,
    use_regime_filter: bool = True,
    min_score_override: float = None,
    percent_per_trade: float = None,
    params:           dict = None,
    max_workers:      int   = 8,
    progress_cb:      Optional[Callable] = None,
) -> Dict:
    """
    Run a walk-forward options backtest across a list of tickers, driving the
    same BacktestEngine that strategies/swing_options_45_60d/run_backtest.py
    drives from config.yaml.

    Note: iv_premium and use_regime_filter are accepted for call-site
    compatibility, but BacktestEngine hardcodes its own entry IV premium and
    always applies its own SPY-regime gate (src/screener.py::_direction_for)
    — neither is separately toggle-able from here.

    progress_cb(pct: float, msg: str) — optional progress callback.

    Returns dict:
        trades_df     — DataFrame of all closed trades
        equity_df     — Date-indexed equity curve DataFrame (date, total_equity)
        metrics       — summary metrics dict
        per_ticker    — {symbol: {trades, win_rate, avg_pnl, total_pnl}}
        initial_capital
    """
    if end_date is None:
        end_date   = datetime.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

    p = {**DEFAULT_PARAMS, **(params or {})}

    n_tickers = len(tickers)
    _cb = progress_cb or (lambda pct, msg: None)

    _cb(0.05, f"Fetching {n_tickers} tickers + SPY + VIX…")

    # ── Fetch raw OHLCV in parallel — BacktestEngine computes indicators itself ──
    all_data: Dict[str, pd.DataFrame] = {}
    done_count = 0
    tickers_with_spy = list(dict.fromkeys(["SPY", "^VIX"] + tickers))

    def _fetch(sym: str):
        df = _fetch_ohlcv(sym, start_date, end_date)
        return sym, (df if not df.empty else None)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch, sym): sym for sym in tickers_with_spy}
        for fut in as_completed(futs):
            done_count += 1
            sym, df = fut.result()
            if df is not None:
                all_data[sym] = df
            _cb(0.05 + 0.55 * done_count / len(tickers_with_spy),
                f"Fetched {done_count}/{len(tickers_with_spy)}")

    vix_df = all_data.pop("^VIX", pd.DataFrame())

    if "SPY" not in all_data or vix_df.empty:
        return {
            "trades_df": pd.DataFrame(),
            "equity_df": pd.DataFrame(),
            "metrics":   {},
            "per_ticker": {},
            "initial_capital": initial_capital,
        }

    _cb(0.62, "Running backtest simulation…")

    # max_hold_days is the dashboard's exposed knob for the engine's
    # "exit once DTE remaining drops to X" threshold.
    dte_exit_remaining = int(max_hold_days) if max_hold_days else int(p["dte_exit_remaining"])

    config = {
        "strategy_params": {
            **p,
            "dte_entry": dte_entry,
            "dte_exit_remaining": dte_exit_remaining,
            "min_score": min_score_override if min_score_override is not None else p["min_score"],
        },
        "account": {"initial_capital": initial_capital},
        "sizing": {
            "max_positions": max_positions,
            "percent_per_trade": percent_per_trade if percent_per_trade is not None else p["percent_per_trade"],
        },
        "risk": {
            "stop_loss":     {"percent": sl_pct * 100},
            "take_profit":   {"percent": tp_pct * 100},
            "trailing_stop": {"enabled": True, "activation": 50.0, "distance": 25.0},
        },
    }

    engine = BacktestEngine(config)
    engine.run(all_data, vix_df)

    _cb(0.92, "Computing metrics…")

    trades_df = pd.DataFrame([{
        "symbol":             t.symbol,
        "direction":          t.direction,
        "entry_date":         t.entry_date,
        "exit_date":          t.exit_date,
        "option_entry_price": t.entry_premium,
        "option_exit_price":  t.exit_premium,
        "contracts":          t.contracts,
        "pnl":                t.pnl,
        "pnl_pct":            t.pnl_pct / 100.0,
        "days_held":          t.dte_at_entry - t.dte_at_exit,
        "exit_reason":        t.exit_reason,
    } for t in engine.trades])

    equity_df = pd.DataFrame(engine.equity_curve)
    if not equity_df.empty:
        equity_df = equity_df.rename(columns={"equity": "total_equity"})

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
        eq = equity_df["total_equity"].values if not equity_df.empty else np.array([initial_capital])
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
