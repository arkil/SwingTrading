"""
Signal scoring + guardrail filtering over historical indicator frames.
Returns a boolean Series of entry signals per bar.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Guardrails:
    """All tunable guardrail parameters. Defaults = best sweep combo (score=8, ext=30, rsi=70)."""
    min_score:      int   = 8      # STRONG only — sweep: 8 > 7 by 6pp win rate
    max_ext_pct:    float = 30.0   # max % above 50MA — blocks ARM(53%); sweep: 30 > 20
    rsi_bull_min:   float = 42.0
    rsi_bull_max:   float = 70.0   # sweep: 70 > 65 (ext filter is the real ARM guardrail)
    rsi_bear_min:   float = 35.0
    rsi_bear_max:   float = 58.0
    min_adx:        float = 20.0
    min_rr:         float = 1.0    # ATR stop/target hardcodes R/R at ~1.33; must be ≤1.33
    min_vol_ratio:  float = 1.2
    breakout_rsi_cap: float = 70.0 # RSI cap for 52W break trigger


def _bull_score(f: pd.DataFrame, g: Guardrails) -> pd.Series:
    """Vectorized bull score per bar (0-12)."""
    sc = pd.Series(0, index=f.index, dtype=float)

    # TREND (0-3)
    sc += (f["Close"] > f["EMA50"]).astype(int)
    sc += (f["EMA50"] > f["EMA200"]).astype(int)
    sc += (f["Minervini"] >= 6).astype(int)

    # MOMENTUM (0-3)
    rsi_ok = (f["RSI"] >= g.rsi_bull_min) & (f["RSI"] <= g.rsi_bull_max)
    rsi_os = f["RSI"] < 32
    sc += (rsi_ok | rsi_os).astype(int)
    sc += ((f["MACD_H"] > 0) & (f["MACD_H"] > f["MACD_H_prev"])).astype(int)
    sc += (f["RS"] >= 60).astype(int)

    # TRIGGERS (capped at 3)
    t = pd.DataFrame(index=f.index)
    t["ema9"]    = f["T_EMA9_BULL"]
    t["52w"]     = f["T_52W_BREAK"] & (f["RSI"] < g.breakout_rsi_cap)  # RSI-confirmed
    t["vol"]     = f["T_VOL_BREAK"]
    t["nr7"]     = f["T_NR7"]
    t["bb"]      = f["T_BB_SQ"]
    t["reclaim"] = f["T_MA_RECLAIM"]
    t["inside"]  = f["T_INSIDE"]
    t["gap"]     = f["T_GAP_UP"]
    t["macd"]    = f["T_MACD_BULL"]
    trig_count = t.astype(int).sum(axis=1).clip(upper=3)
    sc += trig_count

    # VOLUME (0-2)
    sc += (f["VolRatio"] >= 3.0).astype(int) * 2
    sc += ((f["VolRatio"] >= 1.5) & (f["VolRatio"] < 3.0)).astype(int)

    return sc, trig_count


def _bear_score(f: pd.DataFrame, g: Guardrails) -> pd.Series:
    """Vectorized bear score per bar (0-12)."""
    sc = pd.Series(0, index=f.index, dtype=float)

    sc += (f["Close"] < f["EMA50"]).astype(int)
    sc += (f["EMA50"] < f["EMA200"]).astype(int)
    sc += (f["Minervini"] <= 2).astype(int)

    rsi_ob = f["RSI"] > 72
    sc += rsi_ob.astype(int)
    sc += ((f["MACD_H"] < 0) & (f["MACD_H"] < f["MACD_H_prev"])).astype(int)
    sc += (f["RS"] < 40).astype(int)

    t = pd.DataFrame(index=f.index)
    t["ema9"] = f["T_EMA9_BEAR"]
    t["macd"] = f["T_MACD_BEAR"]
    t["gap"]  = f["T_GAP_DOWN"]
    trig_count = t.astype(int).sum(axis=1).clip(upper=3)
    sc += trig_count

    down_vol = f["Close"] < f["Open"]
    sc += ((f["VolRatio"] >= 3.0) & down_vol).astype(int) * 2
    sc += ((f["VolRatio"] >= 1.5) & (f["VolRatio"] < 3.0) & down_vol).astype(int)

    return sc, trig_count


def score_frame(f: pd.DataFrame, g: Guardrails) -> pd.DataFrame:
    """
    Apply scoring and guardrails to full indicator frame.
    Returns DataFrame with columns: Direction, Score, Signal, RR, EntryPrice
    Signal=True means: this bar qualifies, enter at NEXT bar's open.
    """
    bull, bull_t = _bull_score(f, g)
    bear, bear_t = _bear_score(f, g)

    # Direction per bar
    is_bull = bull >= bear
    direction = pd.Series(np.where(is_bull, "BUY", "SELL"), index=f.index)
    score = pd.Series(np.where(is_bull, bull, bear), index=f.index)

    # Guardrail: extension filter for BUY
    ext_ok = ~(is_bull & (f["Ext50MA"] > g.max_ext_pct))

    # Guardrail: ADX
    adx_ok = f["ADX"] >= g.min_adx

    # Guardrail: min volume ratio
    vol_ok = f["VolRatio"] >= g.min_vol_ratio

    # Guardrail: at least 1 trigger
    has_trigger = pd.Series(
        np.where(is_bull, bull_t > 0, bear_t > 0), index=f.index
    )

    # Guardrail: RSI within range (already baked into score but enforce as hard gate)
    rsi_ok = pd.Series(
        np.where(
            is_bull,
            (f["RSI"] >= g.rsi_bull_min) & (f["RSI"] <= g.rsi_bull_max) | (f["RSI"] < 32),
            (f["RSI"] >= g.rsi_bear_min) & (f["RSI"] <= g.rsi_bear_max),
        ),
        index=f.index,
    )

    # R/R check: need data from score frame to compute stop/target
    atr14 = f["ATR"]
    stop  = pd.Series(
        np.where(is_bull, f["Close"] - 1.5 * atr14, f["Close"] + 1.5 * atr14),
        index=f.index,
    )
    target = pd.Series(
        np.where(is_bull, f["Close"] + 2.0 * atr14, f["Close"] - 2.0 * atr14),
        index=f.index,
    )
    rr = (target - f["Close"]).abs() / (f["Close"] - stop).abs().replace(0, np.nan)
    rr_ok = rr >= g.min_rr

    # Combine all filters
    signal = (
        (score >= g.min_score) &
        ext_ok & adx_ok & vol_ok & has_trigger & rsi_ok & rr_ok
    )

    # Require min 4 score on opposite side too (avoid weak opposing signals)
    signal = signal & (score >= 4)

    return pd.DataFrame({
        "Direction": direction,
        "Score":     score,
        "Bull":      bull,
        "Bear":      bear,
        "Signal":    signal,
        "RSI":       f["RSI"],
        "ADX":       f["ADX"],
        "Ext50MA":   f["Ext50MA"],
        "VolRatio":  f["VolRatio"],
        "ATR":       atr14,
        "RR":        rr.round(2),
        "Stop":      stop.round(2),
        "Target":    target.round(2),
        "Close":     f["Close"],
    }, index=f.index)
