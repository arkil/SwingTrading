from __future__ import annotations

from datetime import time

import pandas as pd

from daytrade_backtester.strategies.base import BaseStrategy, Signal
from daytrade_backtester.utils.indicators import adx, atr, bollinger_bands, ema, rsi


def _parse_hhmm(value: str, default: str) -> time:
    txt = (value or default).strip()
    try:
        hh, mm = txt.split(":", 1)
        return time(hour=int(hh), minute=int(mm))
    except Exception:
        hh, mm = default.split(":", 1)
        return time(hour=int(hh), minute=int(mm))


def _to_minutes(t: time) -> int:
    return (t.hour * 60) + t.minute


class BollingerRsiReversalStrategy(BaseStrategy):
    """
    Bollinger Band + RSI mean-reversion strategy (research-improved).

    Signal logic (all gates must pass):
      1. BB band touch  : close outside lower/upper band (or re-entry confirmation).
      2. RSI extreme    : RSI ≤ oversold (long) or RSI ≥ overbought (short).
      3. Session guard  : avoid first / last N minutes of the session.
      4. Volume filter  : relative volume ≥ min_rel_volume (optional).
      5. Long-only gate : if long_only=true, short signals are suppressed.

    Re-entry confirmation (require_reentry_confirmation=true):
      - Instead of entering on the breakout bar itself, wait for the first bar
        that closes *back inside* the band.  This eliminates "band-walk" entries
        where price continues trending along the outer band.
    """

    name = "bollinger_rsi_reversal"

    def prepare(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        out = df.copy()
        bb_length = int(params.get("bb_length", 20))
        bb_std = float(params.get("bb_std", 2.0))
        rsi_length = int(params.get("rsi_length", 14))
        atr_length = int(params.get("atr_length", 14))
        vol_ma_len = int(params.get("volume_ma_length", 20))

        bb = bollinger_bands(out["close"], length=bb_length, stdev=bb_std)
        out = pd.concat([out, bb], axis=1)
        out["rsi"] = rsi(out["close"], length=rsi_length)
        out["atr"] = atr(out["high"], out["low"], out["close"], length=atr_length)

        # Volume context (used only when min_rel_volume > 0 in config).
        out["volume_sma"] = out["volume"].rolling(vol_ma_len).mean()
        out["rel_volume"] = out["volume"] / out["volume_sma"].replace(0, pd.NA)

        # ADX — trend-strength filter.  ADX < 20 = ranging; ADX > 25 = trending.
        adx_length = int(params.get("adx_length", 14))
        out["adx"] = adx(out["high"], out["low"], out["close"], length=adx_length)

        # Medium-term SMA — only computed when the regime filter is enabled.
        # Skipping when disabled avoids sma_length (up to 390) extra NaN rows
        # that dropna() would otherwise strip, reducing effective bar count.
        if params.get("require_long_above_sma", False):
            sma_len = int(params.get("sma_length", 390))
            out["sma_long"] = out["close"].rolling(sma_len).mean()

        # EMA trend context — for summary split only, not used in signal logic.
        out["ema10"] = ema(out["close"], 10)
        out["ema20"] = ema(out["close"], 20)
        return out

    def signal(self, df: pd.DataFrame, idx: int, params: dict) -> Signal | None:
        row = df.iloc[idx]
        prev = df.iloc[idx - 1] if idx > 0 else None

        needed = ["bb_lower", "bb_upper", "rsi"]
        if any(pd.isna(row[col]) for col in needed):
            return None

        # ── Session time guard ────────────────────────────────────────────────
        ts = df.index[idx]
        # Bar index is UTC — convert to ET before checking session minutes.
        try:
            from zoneinfo import ZoneInfo
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                ts = ts.astimezone(ZoneInfo("America/New_York"))
        except Exception:
            pass
        ts_minutes = (ts.hour * 60) + ts.minute
        session_start = _parse_hhmm(str(params.get("session_start", "09:30")), "09:30")
        session_end   = _parse_hhmm(str(params.get("session_end",   "16:00")), "16:00")
        open_cut  = int(params.get("avoid_open_minutes",  0))
        close_cut = int(params.get("avoid_close_minutes", 0))
        start_m = _to_minutes(session_start)
        end_m   = _to_minutes(session_end)
        if open_cut  > 0 and ts_minutes < (start_m + open_cut):
            return None
        if close_cut > 0 and ts_minutes >= (end_m - close_cut):
            return None

        # ── Volume filter (optional — set min_rel_volume: 0 to disable) ──────
        min_rel_vol = float(params.get("min_rel_volume", 0.0))
        if min_rel_vol > 0:
            rv = row.get("rel_volume")
            if rv is None or pd.isna(rv) or float(rv) < min_rel_vol:
                return None

        # ── ADX trend-strength filter (optional — set max_adx: 0 to disable) ─
        # Mean reversion fails on strongly trending days.  ADX > 25 = strong trend.
        # Only enter when ADX is below threshold (ranging / weakly trending market).
        max_adx_val = float(params.get("max_adx", 0.0))
        if max_adx_val > 0:
            adx_val = row.get("adx")
            if adx_val is None or pd.isna(adx_val) or float(adx_val) > max_adx_val:
                return None

        # ── SMA regime filter (optional — set require_long_above_sma: false) ──
        # Only take long (call) trades when close > medium-term SMA.
        # Prevents buying dips during sustained downtrends where calls fight the trend.
        if bool(params.get("require_long_above_sma", False)):
            sma_val = row.get("sma_long")
            close_val = row.get("close")
            if sma_val is not None and not pd.isna(sma_val) and not pd.isna(close_val):
                if float(close_val) < float(sma_val):
                    return None

        rsi_oversold  = float(params.get("rsi_oversold",  30.0))
        rsi_overbought = float(params.get("rsi_overbought", 70.0))
        require_reentry = bool(params.get("require_reentry_confirmation", False))
        long_only = bool(params.get("long_only", False))

        # ── Entry signal ──────────────────────────────────────────────────────
        if require_reentry and prev is not None:
            # Re-entry confirmation: previous bar broke out, current bar closes back inside.
            bullish = (
                float(prev["close"]) < float(prev["bb_lower"])
                and float(row["close"]) >= float(row["bb_lower"])
                and float(prev["rsi"]) <= rsi_oversold
            )
            bearish = (
                float(prev["close"]) > float(prev["bb_upper"])
                and float(row["close"]) <= float(row["bb_upper"])
                and float(prev["rsi"]) >= rsi_overbought
            )
        else:
            # Breakout-bar entry: enter on the bar that first closes outside the band.
            bullish = float(row["close"]) < float(row["bb_lower"]) and float(row["rsi"]) <= rsi_oversold
            bearish = float(row["close"]) > float(row["bb_upper"]) and float(row["rsi"]) >= rsi_overbought

        if long_only:
            bearish = False

        if bullish:
            return Signal(side="long", reason="bb_lower_rsi_oversold")
        if bearish:
            return Signal(side="short", reason="bb_upper_rsi_overbought")
        return None
