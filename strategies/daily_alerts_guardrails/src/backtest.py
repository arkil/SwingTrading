"""
Trade simulation engine (Numba-accelerated inner loop).

Entry:  signal on bar t → enter at Open of bar t+1
Stop:   entry - 1.5 × ATR[t]
Target: entry + 2.0 × ATR[t]
Exit:   first of stop hit, target hit, or max_hold_days expired (close at market)

Returns a list of Trade namedtuples and a summary dict.
"""

import numpy as np
import pandas as pd
import numba
from dataclasses import dataclass
from typing import List, Tuple


@numba.njit(cache=True)
def _simulate_numba(
    opens:     np.ndarray,   # shape (N,)
    highs:     np.ndarray,
    lows:      np.ndarray,
    closes:    np.ndarray,
    signals:   np.ndarray,   # bool, shape (N,)
    stops:     np.ndarray,   # stop price at signal bar
    targets:   np.ndarray,   # target price at signal bar
    directions: np.ndarray,  # 1=BUY, -1=SELL
    max_hold:  int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays: entry_idx, exit_idx, entry_price, exit_price,
                    pnl_r, days_held, direction, exit_reason
    exit_reason: 0=stop, 1=target, 2=time
    """
    N = len(closes)
    # Pre-allocate worst case (every bar a signal)
    cap = N
    ei    = np.empty(cap, dtype=np.int64)
    xi    = np.empty(cap, dtype=np.int64)
    ep    = np.empty(cap, dtype=np.float64)
    xp    = np.empty(cap, dtype=np.float64)
    rr    = np.empty(cap, dtype=np.float64)
    dh    = np.empty(cap, dtype=np.int64)
    dr    = np.empty(cap, dtype=np.int64)   # direction
    xr    = np.empty(cap, dtype=np.int64)   # exit reason
    count = 0

    in_trade = False
    entry_idx  = 0
    entry_price = 0.0
    stop_price  = 0.0
    target_price = 0.0
    direction   = 1

    for i in range(N - 1):
        if in_trade:
            days_held = i - entry_idx
            risk = abs(entry_price - stop_price)
            if risk == 0.0:
                risk = entry_price * 0.01

            if direction == 1:   # LONG
                hit_stop   = lows[i] <= stop_price
                hit_target = highs[i] >= target_price
            else:                # SHORT
                hit_stop   = highs[i] >= stop_price
                hit_target = lows[i] <= target_price

            exited = False
            if hit_stop and hit_target:
                # Ambiguous — assume stop (conservative)
                exit_p = stop_price
                pnl    = direction * (stop_price - entry_price) / risk
                ei[count] = entry_idx; xi[count] = i
                ep[count] = entry_price; xp[count] = stop_price
                rr[count] = pnl; dh[count] = days_held
                dr[count] = direction; xr[count] = 0
                count += 1; in_trade = False; exited = True
            elif hit_stop:
                exit_p = stop_price
                pnl    = direction * (stop_price - entry_price) / risk
                ei[count] = entry_idx; xi[count] = i
                ep[count] = entry_price; xp[count] = stop_price
                rr[count] = pnl; dh[count] = days_held
                dr[count] = direction; xr[count] = 0
                count += 1; in_trade = False; exited = True
            elif hit_target:
                exit_p = target_price
                pnl    = direction * (target_price - entry_price) / risk
                ei[count] = entry_idx; xi[count] = i
                ep[count] = entry_price; xp[count] = target_price
                rr[count] = pnl; dh[count] = days_held
                dr[count] = direction; xr[count] = 1
                count += 1; in_trade = False; exited = True
            elif days_held >= max_hold:
                exit_p = closes[i]
                pnl    = direction * (closes[i] - entry_price) / risk
                ei[count] = entry_idx; xi[count] = i
                ep[count] = entry_price; xp[count] = closes[i]
                rr[count] = pnl; dh[count] = days_held
                dr[count] = direction; xr[count] = 2
                count += 1; in_trade = False; exited = True

        if not in_trade and signals[i]:
            # Enter at next bar's open
            entry_price  = opens[i + 1]
            stop_price   = stops[i]
            target_price = targets[i]
            direction    = directions[i]
            entry_idx    = i + 1
            in_trade     = True

    return (ei[:count], xi[:count], ep[:count], xp[:count],
            rr[:count], dh[:count], dr[:count], xr[:count])


def simulate(
    ohlcv: pd.DataFrame,
    scored: pd.DataFrame,
    max_hold_days: int = 20,
    ticker: str = "UNKN",
) -> pd.DataFrame:
    """
    Run trade simulation for one ticker.

    Args:
        ohlcv:   raw OHLCV DataFrame
        scored:  output of signals.score_frame()
        max_hold_days: force-close after this many bars

    Returns:
        DataFrame of trades (empty if no signals).
    """
    # Align on common index
    idx = ohlcv.index.intersection(scored.index)
    if len(idx) < 60:
        return pd.DataFrame()

    o  = ohlcv.loc[idx, "Open"].values.astype(np.float64)
    h  = ohlcv.loc[idx, "High"].values.astype(np.float64)
    l  = ohlcv.loc[idx, "Low"].values.astype(np.float64)
    c  = ohlcv.loc[idx, "Close"].values.astype(np.float64)
    sg = scored.loc[idx, "Signal"].values.astype(bool)
    st = scored.loc[idx, "Stop"].values.astype(np.float64)
    tg = scored.loc[idx, "Target"].values.astype(np.float64)
    dv = np.where(scored.loc[idx, "Direction"] == "BUY", 1, -1).astype(np.int64)

    ei, xi, ep, xp, rr, dh, dr, xr = _simulate_numba(o, h, l, c, sg, st, tg, dv, max_hold_days)

    if len(ei) == 0:
        return pd.DataFrame()

    dates = idx.values
    reason_map = {0: "STOP", 1: "TARGET", 2: "TIME"}
    dir_map     = {1: "BUY", -1: "SELL"}

    trades = pd.DataFrame({
        "Ticker":       ticker,
        "EntryDate":    dates[ei],
        "ExitDate":     dates[xi],
        "Direction":    [dir_map[d] for d in dr],
        "EntryPrice":   ep.round(4),
        "ExitPrice":    xp.round(4),
        "PnL_R":        rr.round(4),      # R-multiples (>0 = win)
        "DaysHeld":     dh,
        "ExitReason":   [reason_map[r] for r in xr],
        "Win":          rr > 0,
    })

    return trades


def summarise(trades: pd.DataFrame) -> dict:
    """Aggregate trade-level stats into a summary dict."""
    if trades.empty:
        return {"n_trades": 0}

    pnl = trades["PnL_R"]
    wins = trades["Win"]
    return {
        "n_trades":      len(trades),
        "win_rate":      round(wins.mean() * 100, 1),
        "avg_r":         round(pnl.mean(), 3),
        "median_r":      round(pnl.median(), 3),
        "avg_win_r":     round(pnl[wins].mean(), 3) if wins.any() else 0,
        "avg_loss_r":    round(pnl[~wins].mean(), 3) if (~wins).any() else 0,
        "profit_factor": round(pnl[wins].sum() / abs(pnl[~wins].sum()), 2)
                         if (~wins).any() and pnl[~wins].sum() != 0 else np.inf,
        "expectancy":    round(
            wins.mean() * pnl[wins].mean() +
            (~wins).mean() * pnl[~wins].mean(), 3
        ) if wins.any() and (~wins).any() else round(pnl.mean(), 3),
        "max_consec_loss": _max_consec(wins),
        "avg_days_held": round(trades["DaysHeld"].mean(), 1),
        "pct_stop":      round((trades["ExitReason"] == "STOP").mean() * 100, 1),
        "pct_target":    round((trades["ExitReason"] == "TARGET").mean() * 100, 1),
        "pct_time":      round((trades["ExitReason"] == "TIME").mean() * 100, 1),
    }


def _max_consec(wins: pd.Series) -> int:
    max_l = cur = 0
    for w in wins:
        if not w:
            cur += 1
            max_l = max(max_l, cur)
        else:
            cur = 0
    return max_l


def equity_curve(trades: pd.DataFrame, initial: float = 100_000) -> pd.Series:
    """
    Build a daily equity curve from trade-level R-multiples.
    Assumes fixed 1% risk per trade.
    """
    if trades.empty:
        return pd.Series(dtype=float)

    df = trades.sort_values("ExitDate").copy()
    equity = initial
    curve  = []
    dates  = []
    for _, row in df.iterrows():
        risk_dollars = equity * 0.01
        equity += row["PnL_R"] * risk_dollars
        curve.append(equity)
        dates.append(row["ExitDate"])

    return pd.Series(curve, index=pd.to_datetime(dates), name="Equity")


def sharpe(equity: pd.Series) -> float:
    if len(equity) < 3:
        return 0.0
    ret = equity.pct_change().dropna()
    if ret.std() == 0:
        return 0.0
    return round(float(ret.mean() / ret.std() * (252 ** 0.5)), 2)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd   = (equity - peak) / peak * 100
    return round(float(dd.min()), 2)
