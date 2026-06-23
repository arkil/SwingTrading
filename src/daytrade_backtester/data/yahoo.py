from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from daytrade_backtester.config.models import DataConfig
from daytrade_backtester.data.cache import load_df_cache, save_df_cache


def _cache_payload(cfg: DataConfig) -> dict:
    return {
        "symbol": cfg.symbol,
        "interval": cfg.interval,
        "period": cfg.period,
        "date_start": cfg.date_start,
        "date_end": cfg.date_end,
        "timezone": cfg.timezone,
        "session_start": cfg.session_start,
        "session_end": cfg.session_end,
        "source": "yahoo_underlying",
    }


def load_intraday_bars(cfg: DataConfig) -> pd.DataFrame:
    payload = _cache_payload(cfg)
    cached = load_df_cache("underlying_yahoo", payload)
    if cached is not None and not cached.empty:
        return cached

    # Use explicit date range when provided; fall back to period.
    # yfinance end is exclusive, but config.date_end is documented as inclusive —
    # add one day so the last requested date is included.
    if cfg.date_start:
        yf_end = str(date.fromisoformat(cfg.date_end) + timedelta(days=1)) if cfg.date_end else None
        df = yf.download(
            tickers=cfg.symbol,
            start=cfg.date_start,
            end=yf_end,
            interval=cfg.interval,
            auto_adjust=True,
            progress=False,
            prepost=False,
            threads=False,
        )
    else:
        df = yf.download(
            tickers=cfg.symbol,
            period=cfg.period,
            interval=cfg.interval,
            auto_adjust=True,
            progress=False,
            prepost=False,
            threads=False,
        )

    if df.empty:
        raise ValueError("No data returned from Yahoo Finance")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(cfg.timezone)

    session = df.between_time(cfg.session_start, cfg.session_end, inclusive="left")
    session = session.dropna(subset=["open", "high", "low", "close"])

    save_df_cache("underlying_yahoo", payload, session)
    return session
