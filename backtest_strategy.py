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
# Prefer the copy vendored into this repo (works on Streamlit Cloud); fall
# back to the local CBT Framework install for local dev.
for _CBT_DIR in (
    os.path.join(os.path.dirname(__file__), "vendor", "cbt_framework"),
    os.path.expanduser("~/.claude/cbt-framework"),
):
    if os.path.isdir(os.path.join(_CBT_DIR, "engine")):
        sys.path.insert(0, _CBT_DIR)
        break

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
    buy_pyramid: bool = False,
    dynamic_stop_pct: float = 2.0,
    sell_mode: str = "runner",
) -> Tuple[List[Trade], List[float]]:
    """
    Walk-forward backtest on a single ticker.

    buy_pyramid:       BUY-only. When True, replaces the single-shot
                       exit-at-first-level model with scale-in pyramiding on
                       the way to T1 (not at T1/T2 themselves):
                         - price reaches 1/3 of the way from entry to T1
                           -> add another leg, same $ capital as entry
                         - price reaches 2/3 of the way from entry to T1
                           -> add another leg, same $ capital again
                       The buy/add side is identical across all sell_mode
                       values below. Only what happens at T1/T2/T3 changes.
                       False (default) preserves the original single-exit
                       behavior: whichever of Stop/T3/T2/T1 is touched first
                       closes the entire trade.
    dynamic_stop_pct:  % below the T1 price the stop moves to once T1 is
                       reached (only used when buy_pyramid=True).
    sell_mode:         Only used when buy_pyramid=True. Controls what the
                       sell side does at T1/T2/T3:
                         "runner"          - no selling at T1/T2. T1 only
                                             arms a dynamic stop
                                             (dynamic_stop_pct % below T1)
                                             protecting the whole position.
                                             T2 has no effect. Sell 100% at T3.
                         "all_at_t1"       - sell the entire accumulated
                                             position the moment T1 is
                                             touched. T2/T3 never seen.
                         "scale_50_25_25"  - sell 50% of the position at T1
                                             (and arm the dynamic stop for
                                             the rest), 25% at T2, final 25%
                                             at T3.

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

            pyramid = direction == "BUY" and buy_pyramid

            if pyramid:
                # Scale-in on the way to T1 (not at T1/T2 themselves) — same
                # for every sell_mode:
                #   1/3 of the way from entry to T1 -> add a leg, same $ capital
                #   2/3 of the way from entry to T1 -> add a leg, same $ capital
                add_level_1  = entry_price + (1 / 3) * (t1 - entry_price)
                add_level_2  = entry_price + (2 / 3) * (t1 - entry_price)

                total_shares = float(shares)   # grows as legs are added
                total_cost   = entry_price * shares
                leg_notional = entry_price * shares  # "same capital" per add
                current_stop = stop
                added_leg1   = False
                added_leg2   = False
                stop_armed   = False

                remaining_shares    = total_shares  # shares still held (post partial sells)
                realized_proceeds   = 0.0           # $ banked from partial sells so far
                sold_t1             = False
                sold_t2             = False

                def _add_leg_1():
                    nonlocal total_shares, total_cost, added_leg1, remaining_shares
                    if not added_leg1:
                        add_shares      = leg_notional / add_level_1
                        total_shares   += add_shares
                        remaining_shares += add_shares
                        total_cost     += add_shares * add_level_1
                        added_leg1      = True

                def _add_leg_2():
                    nonlocal total_shares, total_cost, added_leg2, remaining_shares
                    if not added_leg2:
                        add_shares      = leg_notional / add_level_2
                        total_shares   += add_shares
                        remaining_shares += add_shares
                        total_cost     += add_shares * add_level_2
                        added_leg2      = True

                for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
                    bar = df_full.iloc[j]

                    # Stop check first, using whatever stop currently applies
                    if bar["Low"] <= current_stop:
                        realized_proceeds += remaining_shares * current_stop
                        remaining_shares   = 0.0
                        exit_price  = current_stop
                        exit_reason = "TRAIL_STOP" if stop_armed else "STOP"
                        exit_idx    = j
                        break

                    # Buy-side adds (identical for every sell_mode)
                    if bar["High"] >= add_level_2 and not added_leg2:
                        _add_leg_1()
                        _add_leg_2()
                    elif bar["High"] >= add_level_1 and not added_leg1:
                        _add_leg_1()

                    if sell_mode == "all_at_t1":
                        if bar["High"] >= t1:
                            realized_proceeds += remaining_shares * t1
                            remaining_shares   = 0.0
                            exit_price  = t1
                            exit_reason = "T1_ALL"
                            exit_idx    = j
                            break

                    elif sell_mode == "scale_50_25_25":
                        if bar["High"] >= t3:
                            if not sold_t1:
                                sell_qty = 0.5 * total_shares
                                realized_proceeds += sell_qty * t1
                                remaining_shares  -= sell_qty
                                sold_t1 = True
                            if not sold_t2:
                                sell_qty = 0.25 * total_shares
                                realized_proceeds += sell_qty * t2
                                remaining_shares  -= sell_qty
                                sold_t2 = True
                            realized_proceeds += remaining_shares * t3
                            remaining_shares   = 0.0
                            exit_price  = t3
                            exit_reason = "T3_FINAL"
                            exit_idx    = j
                            break
                        elif bar["High"] >= t2 and not sold_t2:
                            if not sold_t1:
                                sell_qty = 0.5 * total_shares
                                realized_proceeds += sell_qty * t1
                                remaining_shares  -= sell_qty
                                sold_t1 = True
                                stop_armed   = True
                                current_stop = t1 * (1 - dynamic_stop_pct / 100)
                            sell_qty = 0.25 * total_shares
                            realized_proceeds += sell_qty * t2
                            remaining_shares  -= sell_qty
                            sold_t2 = True
                        elif bar["High"] >= t1 and not sold_t1:
                            sell_qty = 0.5 * total_shares
                            realized_proceeds += sell_qty * t1
                            remaining_shares  -= sell_qty
                            sold_t1 = True
                            stop_armed   = True
                            current_stop = t1 * (1 - dynamic_stop_pct / 100)

                    else:  # "runner" (default): no selling at T1/T2, dynamic stop arms at T1
                        if bar["High"] >= t3:
                            exit_price  = t3
                            exit_reason = "T3_ALL"
                            exit_idx    = j
                            realized_proceeds += remaining_shares * t3
                            remaining_shares   = 0.0
                            break
                        elif bar["High"] >= t1 and not stop_armed:
                            stop_armed   = True
                            current_stop = t1 * (1 - dynamic_stop_pct / 100)

                if exit_price is None:
                    # MAX_HOLD: liquidate whatever remains at the final close
                    close_price = float(df_full["Close"].iloc[exit_idx])
                    realized_proceeds += remaining_shares * close_price
                    exit_price = close_price

                shares      = total_shares
                avg_entry   = total_cost / total_shares
                raw_pnl     = realized_proceeds - total_cost
                pnl_pct     = raw_pnl / total_cost * 100
                entry_price = avg_entry  # for reporting: weighted avg cost basis

            else:
                current_stop = stop
                for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
                    bar = df_full.iloc[j]

                    if direction == "BUY":
                        # Stop hit (low pierces stop)
                        if bar["Low"] <= current_stop:
                            exit_price  = current_stop
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
    buy_pyramid:      bool  = False,
    dynamic_stop_pct: float = 2.0,
    sell_mode:        str   = "runner",
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
            min_score        = min_score,
            max_hold         = max_hold,
            initial_capital  = initial_capital,
            risk_pct         = risk_pct,
            buy_pyramid      = buy_pyramid,
            dynamic_stop_pct = dynamic_stop_pct,
            sell_mode        = sell_mode,
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
