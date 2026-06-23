"""
Strategy Backtester — CBT Framework
=====================================
Walk-forward backtest of the Daily Alerts strategy (same signals used
by alerts_engine._build_alert) against historical OHLCV data.

Entry:  signal fires with score >= min_score → buy/short next bar open
Exit:   first of [Stop, T1, T2, T3] hit intraday, else close after max_hold days

Position sizing: 1% portfolio risk per trade
    shares = (capital * 0.01) / |entry - stop|
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.expanduser("~/.claude/cbt-framework"))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from engine.metrics import Trade, calculate_all_metrics, calculate_trade_stats

# Import signal helpers from alerts_engine (reuse, don't duplicate)
from alerts_engine import _build_alert


def _normalize_idx(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index.date)
    return df

_yf_lock = Lock()

# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        with _yf_lock:
            raw = yf.download(
                ticker, start=start, end=end,
                interval="1d", progress=False, auto_adjust=True,
            )
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return _normalize_idx(raw)
    except Exception:
        return pd.DataFrame()


def _fetch_spy(start: str, end: str) -> pd.DataFrame:
    return _fetch_history("SPY", start, end)


# ── Walk-forward single ticker backtest ───────────────────────────────────────

def backtest_ticker(
    ticker: str,
    df_full: pd.DataFrame,
    spy_full: pd.DataFrame,
    min_score: int = 5,
    max_hold: int = 20,
    step: int = 3,
    initial_capital: float = 10_000.0,
    risk_pct: float = 0.01,
) -> Tuple[List[Trade], List[float]]:
    """
    Walk-forward backtest on a single ticker.

    Returns (trades, equity_curve).
    equity_curve is bar-by-bar portfolio value.
    """
    trades: List[Trade] = []
    capital = initial_capital
    equity_curve: List[float] = [capital]

    n = len(df_full)
    i = 100  # warmup — need enough bars for all indicators

    while i < n - 2:
        df_slice  = df_full.iloc[:i + 1]
        spy_slice = spy_full.iloc[:min(i + 1, len(spy_full))]

        # Re-use the exact same signal logic as the live alerts engine
        signal = _build_alert(ticker, df_slice, spy_slice, info={})

        if signal and signal["Score"] >= min_score:
            direction = signal["Direction"]
            entry_idx = i + 1
            if entry_idx >= n:
                break

            entry_price = float(df_full["Open"].iloc[entry_idx])
            if entry_price <= 0:
                i += step
                continue

            stop  = signal["Stop"]
            t1    = signal["T1"]
            t2    = signal["T2"]
            t3    = signal["T3"]

            # Position size: risk 1% of current capital
            risk_per_share = abs(entry_price - stop)
            if risk_per_share <= 0:
                i += step
                continue
            shares = max(1, int((capital * risk_pct) / risk_per_share))

            # Simulate forward bars
            exit_price:  Optional[float] = None
            exit_reason: str = "MAX_HOLD"
            exit_idx:    int = min(entry_idx + max_hold, n - 1)

            for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
                bar = df_full.iloc[j]

                if direction == "BUY":
                    # Stop hit (low pierces stop)
                    if bar["Low"] <= stop:
                        exit_price  = stop
                        exit_reason = "STOP"
                        exit_idx    = j
                        break
                    # Targets (high reaches level)
                    elif bar["High"] >= t3:
                        exit_price  = t3
                        exit_reason = "T3"
                        exit_idx    = j
                        break
                    elif bar["High"] >= t2:
                        exit_price  = t2
                        exit_reason = "T2"
                        exit_idx    = j
                        break
                    elif bar["High"] >= t1:
                        exit_price  = t1
                        exit_reason = "T1"
                        exit_idx    = j
                        break
                else:  # SHORT
                    if bar["High"] >= stop:
                        exit_price  = stop
                        exit_reason = "STOP"
                        exit_idx    = j
                        break
                    elif bar["Low"] <= t3:
                        exit_price  = t3
                        exit_reason = "T3"
                        exit_idx    = j
                        break
                    elif bar["Low"] <= t2:
                        exit_price  = t2
                        exit_reason = "T2"
                        exit_idx    = j
                        break
                    elif bar["Low"] <= t1:
                        exit_price  = t1
                        exit_reason = "T1"
                        exit_idx    = j
                        break

            if exit_price is None:
                exit_price = float(df_full["Close"].iloc[exit_idx])

            # P&L
            raw_pnl = (
                (exit_price - entry_price) * shares if direction == "BUY"
                else (entry_price - exit_price) * shares
            )
            pnl_pct = (
                (exit_price - entry_price) / entry_price * 100 if direction == "BUY"
                else (entry_price - exit_price) / entry_price * 100
            )

            capital += raw_pnl
            equity_curve.append(capital)

            trades.append(Trade(
                entry_time       = str(df_full.index[entry_idx]),
                exit_time        = str(df_full.index[exit_idx]),
                direction        = 1 if direction == "BUY" else -1,
                entry_price      = round(entry_price, 2),
                exit_price       = round(exit_price, 2),
                size             = float(shares),
                pnl              = round(raw_pnl, 2),
                pnl_percent      = round(pnl_pct, 2),
                exit_reason      = exit_reason,
                duration_seconds = (exit_idx - entry_idx) * 86400,
                fees             = 0.0,
            ))

            i = exit_idx + 1  # skip to after exit
        else:
            i += step

    return trades, equity_curve


# ── Multi-ticker backtest ─────────────────────────────────────────────────────

def run_strategy_backtest(
    tickers:         List[str],
    start_date:      str  = None,
    end_date:        str  = None,
    min_score:       int  = 5,
    max_hold:        int  = 20,
    initial_capital: float = 10_000.0,
    risk_pct:        float = 0.01,
    max_workers:     int  = 8,
    progress_cb      = None,
) -> Dict:
    """
    Run walk-forward backtest across multiple tickers.

    Returns a dict with:
        metrics      — CBT calculate_all_metrics result
        trades       — flat list of all Trade objects
        trade_df     — DataFrame of trade log
        equity_curve — combined equity curve
        per_ticker   — {ticker: {trades, win_rate, total_return}}
    """
    if end_date is None:
        end_date   = datetime.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

    # Extra history for indicator warmup (100 bars ≈ 140 calendar days)
    warmup_start = (
        datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=160)
    ).strftime("%Y-%m-%d")

    spy = _fetch_spy(warmup_start, end_date)

    all_trades: List[Trade]   = []
    combined_equity           = [initial_capital]
    per_ticker: Dict          = {}
    total = len(tickers)

    def _run_one(ticker: str):
        df = _fetch_history(ticker, warmup_start, end_date)
        if df.empty or len(df) < 110:
            return ticker, [], [initial_capital]
        trades, eq = backtest_ticker(
            ticker, df, spy,
            min_score       = min_score,
            max_hold        = max_hold,
            initial_capital = initial_capital,
            risk_pct        = risk_pct,
        )
        return ticker, trades, eq

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_run_one, tk): tk for tk in tickers}
        for fut in as_completed(futs):
            done += 1
            if progress_cb:
                progress_cb(done, total, futs[fut])
            try:
                ticker, trades, eq = fut.result()
                if trades:
                    all_trades.extend(trades)
                    ts = calculate_trade_stats(trades)
                    per_ticker[ticker] = {
                        "trades":       len(trades),
                        "win_rate":     round(ts["win_rate"], 1),
                        "profit_factor":round(ts["profit_factor"], 2),
                        "total_return": round(
                            sum(t.pnl for t in trades) / initial_capital * 100, 1
                        ),
                        "avg_win":      round(ts["avg_winner"], 2),
                        "avg_loss":     round(ts["avg_loser"], 2),
                    }
            except Exception:
                pass

    if not all_trades:
        return {
            "metrics":      {},
            "trades":       [],
            "trade_df":     pd.DataFrame(),
            "equity_curve": combined_equity,
            "per_ticker":   {},
        }

    # Sort all trades by entry time for equity curve
    all_trades.sort(key=lambda t: t.entry_time)

    # Rebuild combined equity curve
    cap = initial_capital
    combined_equity = [cap]
    for t in all_trades:
        cap += t.pnl
        combined_equity.append(cap)

    metrics = calculate_all_metrics(
        equity_curve    = combined_equity,
        trades          = all_trades,
        initial_capital = initial_capital,
        periods_per_year= 252,
    )

    # Build trade log DataFrame
    trade_df = pd.DataFrame([{
        "Ticker":      futs.get(None, ""),  # resolved below
        "Entry Date":  t.entry_time[:10],
        "Exit Date":   t.exit_time[:10],
        "Direction":   "BUY" if t.direction == 1 else "SHORT",
        "Entry $":     t.entry_price,
        "Exit $":      t.exit_price,
        "Shares":      int(t.size),
        "P&L $":       t.pnl,
        "P&L %":       t.pnl_percent,
        "Exit Reason": t.exit_reason,
        "Days Held":   max(1, t.duration_seconds // 86400),
    } for t in all_trades])

    return {
        "metrics":      metrics,
        "trades":       all_trades,
        "trade_df":     trade_df,
        "equity_curve": combined_equity,
        "per_ticker":   per_ticker,
        "start_date":   start_date,
        "end_date":     end_date,
        "tickers":      tickers,
    }
