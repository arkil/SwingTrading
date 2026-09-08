"""
Signal History — reusable "how has this signal performed before?" utility
=============================================================================
Generic forward-return aggregation over any OHLCV dataframe that already has
a per-bar boolean "signal fired here" mask. Deliberately does not know
anything about *which* strategy produced the signal — Livermore, Minervini,
EMA crossovers, etc. can all plug in the same way, so this stays one shared
utility instead of a bespoke historical-replay loop per screener.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def compute_signal_forward_returns(
    df: pd.DataFrame,
    signal_mask: pd.Series,
    forward_days: Iterable[int] = (5, 10, 20),
    price_col: str = "Close",
    direction: str = "bullish",
) -> dict:
    """
    For every bar where signal_mask is True, look forward N trading days and
    measure the price return. Aggregates into win-rate / avg-return per
    horizon — i.e. "of the last N times this exact signal fired on this
    ticker, how often did it work, and by how much."

    Args:
        df: OHLCV dataframe (any per-ticker frame with a price_col column).
        signal_mask: boolean Series aligned to df.index — True where the
            signal fired.
        forward_days: horizons to measure, in trading days.
        price_col: column to measure returns from.
        direction: "bullish" — a win is a positive forward return (for
            long/UPWARD-style signals). "bearish" — a win is a negative
            forward return (for short/DOWNWARD-style signals).

    Returns:
        {n: {"n": count, "win_rate": pct or None, "avg_return_pct": pct or None}}
        for each horizon in forward_days. n=0 / None values mean not enough
        history to measure yet — not zero edge.
    """
    close = df[price_col]
    signal_positions = np.flatnonzero(signal_mask.reindex(df.index, fill_value=False).to_numpy())

    results = {}
    for n in forward_days:
        rets = []
        for pos in signal_positions:
            fwd_pos = pos + n
            if fwd_pos >= len(df):
                continue
            entry = float(close.iloc[pos])
            fwd = float(close.iloc[fwd_pos])
            if entry <= 0:
                continue
            rets.append((fwd - entry) / entry * 100)

        if not rets:
            results[n] = {"n": 0, "win_rate": None, "avg_return_pct": None}
            continue

        arr = np.array(rets)
        wins = (arr > 0) if direction == "bullish" else (arr < 0)
        results[n] = {
            "n": len(arr),
            "win_rate": round(float(wins.mean() * 100), 1),
            "avg_return_pct": round(float(arr.mean()), 2),
        }
    return results
