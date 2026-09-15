"""
Screener for the swing_options_45_60d strategy.

Public API
----------
    screen_universe(data, params, iv_rank, min_score) -> pd.DataFrame

Composite score = weighted sum of 7 boolean signals (0-10 scale):
    ema_aligned, trend_200 (bull only), macd_ok, adx>adx_min, vol_surge,
    bb_expansion, stoch_ok, obv_ok
"""

from __future__ import annotations

import pandas as pd

# Weight table sums to 10.0 across the 7 "core" signals used in the memory
# note ("7 indicator signals"); ADX/EMA trend confirmers weighted slightly
# higher as primary trend gates.
_WEIGHTS = {
    "ema_aligned": 1.5,
    "macd_ok": 1.5,
    "adx_ok": 1.5,
    "vol_surge": 1.25,
    "bb_expansion": 1.25,
    "stoch_ok": 1.5,
    "obv_ok": 1.5,
}


def _score_row(row: pd.Series, params: dict) -> float:
    score = 0.0
    if bool(row.get("ema_aligned", False)):
        score += _WEIGHTS["ema_aligned"]
    if bool(row.get("macd_ok", False)):
        score += _WEIGHTS["macd_ok"]
    if float(row.get("adx", 0.0)) > params.get("adx_min", 20):
        score += _WEIGHTS["adx_ok"]
    if bool(row.get("vol_surge", False)):
        score += _WEIGHTS["vol_surge"]
    if bool(row.get("bb_expansion", False)):
        score += _WEIGHTS["bb_expansion"]
    if bool(row.get("stoch_ok", False)):
        score += _WEIGHTS["stoch_ok"]
    if bool(row.get("obv_ok", False)):
        score += _WEIGHTS["obv_ok"]
    return round(score, 2)


def _direction_for(row: pd.Series, spy_bull: bool) -> str | None:
    """
    Regime filter: calls only in bull regime (SPY > SMA200), puts only when
    the symbol itself looks overbought/bearish in a non-bull regime.
    """
    rsi = float(row.get("rsi", 50.0))
    ema_bull = bool(row.get("ema_bull", False))
    ema_bear = bool(row.get("ema_bear", False))
    trend_200 = bool(row.get("trend_200", False))

    if spy_bull and trend_200 and ema_bull:
        return "call"
    if not spy_bull and (not trend_200) and (ema_bear or rsi > 65):
        return "put"
    return None


def screen_universe(
    data: dict[str, pd.DataFrame],
    params: dict,
    iv_rank: float,
    min_score: float,
) -> pd.DataFrame:
    """
    Screen the latest bar of each symbol's indicator-enriched DataFrame for
    qualifying swing-options candidates.

    Args:
        data: {symbol: indicator-enriched DataFrame (see src.indicators)}
        params: strategy params dict
        iv_rank: current IV rank (0-100); candidates suppressed if too high
        min_score: minimum composite score (0-10) required to include a symbol

    Returns:
        DataFrame with columns: symbol, direction, close, score, hist_vol,
        atr, adx, rsi, ema_aligned, trend_200, macd_ok, vol_surge,
        supertrend_ok, bb_expansion, stoch_ok, obv_ok
    """
    iv_rank_max = params.get("iv_rank_max", 30)
    if iv_rank is not None and iv_rank > iv_rank_max:
        return pd.DataFrame()

    spy_df = data.get("SPY")
    spy_bull = True
    if spy_df is not None and len(spy_df) > 0:
        spy_last = spy_df.iloc[-1]
        spy_bull = bool(spy_last.get("trend_200", True))

    rows = []
    for symbol, df in data.items():
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        if pd.isna(last.get("adx")) or pd.isna(last.get("ema_fast")):
            continue  # still in indicator warmup

        direction = _direction_for(last, spy_bull)
        if direction is None:
            continue

        score = _score_row(last, params)
        if score < min_score:
            continue

        rows.append({
            "symbol": symbol,
            "direction": direction,
            "close": float(last["close"]),
            "score": score,
            "hist_vol": float(last.get("hist_vol", 0.20)) if pd.notna(last.get("hist_vol")) else 0.20,
            "atr": float(last.get("atr", 0.0)) if pd.notna(last.get("atr")) else 0.0,
            "adx": float(last.get("adx", 0.0)) if pd.notna(last.get("adx")) else 0.0,
            "rsi": float(last.get("rsi", 50.0)) if pd.notna(last.get("rsi")) else 50.0,
            "ema_aligned": bool(last.get("ema_aligned", False)),
            "trend_200": bool(last.get("trend_200", False)),
            "macd_ok": bool(last.get("macd_ok", False)),
            "vol_surge": bool(last.get("vol_surge", False)),
            "supertrend_ok": bool(last.get("supertrend_ok", False)),
            "bb_expansion": bool(last.get("bb_expansion", False)),
            "stoch_ok": bool(last.get("stoch_ok", False)),
            "obv_ok": bool(last.get("obv_ok", False)),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
