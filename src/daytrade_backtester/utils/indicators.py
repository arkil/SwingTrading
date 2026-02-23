from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    value = 100 - (100 / (1 + rs))
    return value.fillna(50)


def bollinger_bands(series: pd.Series, length: int = 20, stdev: float = 2.0) -> pd.DataFrame:
    basis = series.rolling(length).mean()
    sigma = series.rolling(length).std(ddof=0)
    upper = basis + stdev * sigma
    lower = basis - stdev * sigma
    return pd.DataFrame({"bb_basis": basis, "bb_upper": upper, "bb_lower": lower})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()
