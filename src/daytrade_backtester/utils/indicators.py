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


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, index: pd.Index) -> pd.Series:
    tp = (high + low + close) / 3.0
    tpv = tp * volume
    day_key = pd.Series(index).dt.date
    cum_tpv = tpv.groupby(day_key).cumsum()
    cum_vol = volume.groupby(day_key).cumsum().replace(0, pd.NA)
    return (cum_tpv / cum_vol).astype(float)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    mf = tp * volume
    tp_prev = tp.shift(1)
    pos_mf = mf.where(tp > tp_prev, 0.0)
    neg_mf = mf.where(tp < tp_prev, 0.0)
    pos_sum = pos_mf.rolling(length).sum()
    neg_sum = neg_mf.rolling(length).sum().replace(0, pd.NA)
    mfr = pos_sum / neg_sum
    value = 100 - (100 / (1 + mfr))
    return value.fillna(50)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().fillna(0.0)
    signed_vol = volume.where(direction > 0, -volume.where(direction < 0, 0.0))
    return signed_vol.cumsum().astype(float)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average Directional Index — measures trend *strength*, not direction.

    ADX < 20 : weak / ranging market  → mean-reversion edge is strong
    ADX 20–25: developing trend
    ADX > 25 : strong trend           → mean-reversion edge is weak
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    both_positive = (plus_dm > 0) & (minus_dm > 0)
    plus_dm = plus_dm.where(~both_positive | (plus_dm >= minus_dm), 0.0)
    minus_dm = minus_dm.where(~both_positive | (minus_dm > plus_dm), 0.0)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / length
    smooth_tr = tr.ewm(alpha=alpha, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    _nan = float("nan")
    plus_di = 100.0 * smooth_plus / smooth_tr.replace(0, _nan)
    minus_di = 100.0 * smooth_minus / smooth_tr.replace(0, _nan)

    di_sum = (plus_di + minus_di).replace(0, _nan)
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum).astype(float)
    return dx.ewm(alpha=alpha, adjust=False).mean().fillna(0.0)
