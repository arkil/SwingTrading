from __future__ import annotations

import pandas as pd

from daytrade_backtester.strategies.base import BaseStrategy, Signal
from daytrade_backtester.utils.indicators import atr, bollinger_bands, ema, rsi


class BollingerRsiReversalStrategy(BaseStrategy):
    name = "bollinger_rsi_reversal"

    def prepare(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        out = df.copy()
        bb_length = int(params.get("bb_length", 20))
        bb_std = float(params.get("bb_std", 2.0))
        rsi_length = int(params.get("rsi_length", 14))
        atr_length = int(params.get("atr_length", 14))
        volume_ma_length = int(params.get("volume_ma_length", 20))

        bb = bollinger_bands(out["close"], length=bb_length, stdev=bb_std)
        out = pd.concat([out, bb], axis=1)
        out["rsi"] = rsi(out["close"], length=rsi_length)
        out["atr"] = atr(out["high"], out["low"], out["close"], length=atr_length)

        out["volume_sma"] = out["volume"].rolling(volume_ma_length).mean()
        out["rel_volume"] = out["volume"] / out["volume_sma"].replace(0, pd.NA)

        # Trend context columns for summary splits.
        out["ema10"] = ema(out["close"], 10)
        out["ema20"] = ema(out["close"], 20)
        return out

    def signal(self, df: pd.DataFrame, idx: int, params: dict) -> Signal | None:
        row = df.iloc[idx]
        needed = ["bb_lower", "bb_upper", "rsi", "volume_sma", "rel_volume"]
        if any(pd.isna(row[col]) for col in needed):
            return None

        rsi_oversold = float(params.get("rsi_oversold", 35))
        rsi_overbought = float(params.get("rsi_overbought", 65))

        min_rel_volume = float(params.get("min_rel_volume", 1.2))
        min_volume_sma = float(params.get("min_volume_sma", 1_000_000))

        volume_ok = row["rel_volume"] >= min_rel_volume and row["volume_sma"] >= min_volume_sma
        if not volume_ok:
            return None

        bullish = row["close"] < row["bb_lower"] and row["rsi"] <= rsi_oversold
        bearish = row["close"] > row["bb_upper"] and row["rsi"] >= rsi_overbought

        if bullish:
            return Signal(side="long", reason="close_below_lower_bb_rsi_oversold_volume_confirmed")
        if bearish:
            return Signal(side="short", reason="close_above_upper_bb_rsi_overbought_volume_confirmed")
        return None
