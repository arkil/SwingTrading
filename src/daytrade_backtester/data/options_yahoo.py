from __future__ import annotations

from datetime import timedelta

import pandas as pd
import yfinance as yf

from daytrade_backtester.data.cache import load_df_cache, save_df_cache


def _nearest_expiry(expiries: list[str], target_date: pd.Timestamp) -> str | None:
    valid = [pd.Timestamp(e).date() for e in expiries]
    want = target_date.date()
    future = [d for d in valid if d >= want]
    if not future:
        return None
    chosen = min(future)
    return chosen.strftime("%Y-%m-%d")


def _select_contract_symbol(chain: pd.DataFrame, side: str, spot: float, otm_steps: int) -> str | None:
    if chain.empty or "strike" not in chain.columns or "contractSymbol" not in chain.columns:
        return None

    strikes = sorted(float(s) for s in chain["strike"].dropna().unique())
    if not strikes:
        return None

    if side == "long":
        candidates = [s for s in strikes if s >= spot]
    else:
        candidates = [s for s in strikes if s <= spot]

    if not candidates:
        return None

    idx = min(max(otm_steps - 1, 0), len(candidates) - 1)
    strike = candidates[idx] if side == "long" else candidates[-(idx + 1)]

    row = chain.loc[chain["strike"] == strike].head(1)
    if row.empty:
        return None
    return str(row.iloc[0]["contractSymbol"])


def _fetch_option_bars(contract_symbol: str, start: pd.Timestamp, end: pd.Timestamp, interval: str, timezone: str) -> pd.DataFrame:
    payload = {
        "contract": contract_symbol,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "interval": interval,
        "timezone": timezone,
        "source": "yahoo_options",
    }
    cached = load_df_cache("options_yahoo_bars", payload)
    if cached is not None and not cached.empty:
        return cached

    df = yf.download(
        tickers=contract_symbol,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=False,
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    if "close" not in df.columns:
        return pd.DataFrame()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(timezone)

    save_df_cache("options_yahoo_bars", payload, df)
    return df


def _price_near_time(df: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if df.empty:
        return None

    right = df.loc[df.index >= ts]
    if not right.empty:
        return float(right.iloc[0]["close"])

    left = df.loc[df.index <= ts]
    if not left.empty:
        return float(left.iloc[-1]["close"])

    return None


def lookup_option_entry_exit(
    symbol: str,
    side: str,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    spot_entry: float,
    interval: str,
    timezone: str,
    dte_target_days: int,
    otm_steps: int,
) -> tuple[str | None, float | None, float | None, str]:
    try:
        base = yf.Ticker(symbol)
        expiries = list(base.options)
        if not expiries:
            return None, None, None, "no_chain_from_yahoo"

        expiry = _nearest_expiry(expiries, entry_time + pd.Timedelta(days=dte_target_days))
        if not expiry:
            return None, None, None, "no_matching_expiry"

        expiry_dt = pd.Timestamp(expiry, tz=entry_time.tz)
        if abs((expiry_dt - entry_time).days) > 10:
            return None, None, None, "historical_chain_unavailable"

        chain = base.option_chain(expiry)
        table = chain.calls if side == "long" else chain.puts
        contract = _select_contract_symbol(table, side, spot_entry, otm_steps)
        if not contract:
            return None, None, None, "contract_not_found"

        bars = _fetch_option_bars(contract, entry_time, exit_time, interval, timezone)
        if bars.empty:
            return contract, None, None, "no_price_bars"

        entry_opt = _price_near_time(bars, entry_time)
        exit_opt = _price_near_time(bars, exit_time)
        if entry_opt is None or exit_opt is None:
            return contract, entry_opt, exit_opt, "missing_entry_or_exit_price"

        return contract, entry_opt, exit_opt, "ok"
    except Exception:
        return None, None, None, "api_error"
