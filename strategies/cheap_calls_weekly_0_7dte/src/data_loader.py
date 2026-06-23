"""
Data loader for cheap_calls_weekly_0_7dte backtest.
Fetches daily OHLCV for the scanner universe + SPY via yfinance,
caches to Polars parquet in Data/. Re-uses cache if files exist.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional
import numpy as np

import pandas as pd
import polars as pl
import yfinance as yf

# Default universe — mirrors "Focused" scanner universe
FOCUSED_UNIVERSE = [
    "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","AVGO","PLTR",
    "APP","ARM","CRWD","PANW","AXON","FICO","DDOG","NET","ZS","MRVL",
    "SNOW","TTD","HUBS","DUOL","CAVA","CELH","ONON","SHOP","COIN","HOOD",
    # Watchlist additions
    "ANET","CSCO","SMCI","MSTR","RKLB","RXRX","IONQ","BABA",
    "NIO","GRAB","BROS","MNDY","GTLB","BILL","DOCN","ASTS",
    "LUNR","KULR","SOUN","BBAI","ARQQ","QBTS",
]
BENCHMARK = "SPY"
DATA_DIR = Path(__file__).parent.parent / "Data"


def _fetch_ohlcv(tickers: list[str], start: str, end: Optional[str] = None) -> pl.DataFrame:
    """Download daily OHLCV via yfinance, return long-format Polars DataFrame."""
    print(f"  Downloading {len(tickers)} tickers from yfinance...")
    raw = yf.download(
        tickers=" ".join(tickers),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned empty data")

    frames: list[pl.DataFrame] = []
    for sym in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                sub = raw[sym][["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                sub = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            sub.index.name = "date"
            sub = sub.reset_index()
            sub.columns = [c.lower() for c in sub.columns]
            # Cast all numeric columns to float64 for consistent Polars schema
            for col in ["open", "high", "low", "close", "volume"]:
                if col in sub.columns:
                    sub[col] = sub[col].astype(float)
            sub["symbol"] = sym
            sub["date"] = sub["date"].dt.normalize()
            frames.append(pl.from_pandas(sub))
        except Exception as e:
            print(f"  WARNING: {sym} failed — {e}")

    if not frames:
        raise RuntimeError("No data fetched for any ticker")

    return pl.concat(frames).sort(["symbol", "date"])


def load_universe(
    tickers: Optional[list[str]] = None,
    start: str = "2023-01-01",
    end: Optional[str] = None,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """
    Load OHLCV for universe. Fetches from yfinance and caches to parquet.
    Subsequent calls load from cache unless force_refresh=True.
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "ohlcv_universe.parquet"
    tickers = tickers or FOCUSED_UNIVERSE

    if cache.exists() and not force_refresh:
        print(f"Loading universe from cache: {cache}")
        df = pl.read_parquet(cache)
        cached_symbols = set(df["symbol"].unique().to_list())
        missing = [t for t in tickers if t not in cached_symbols]
        if missing:
            print(f"  Cache missing {len(missing)} symbols, fetching...")
            new = _fetch_ohlcv(missing, start=start, end=end)
            df = pl.concat([df, new]).sort(["symbol", "date"])
            df.write_parquet(cache)
        return df

    df = _fetch_ohlcv(tickers, start=start, end=end)
    df.write_parquet(cache)
    print(f"Saved universe to {cache}  ({df.shape[0]:,} rows, {df['symbol'].n_unique()} symbols)")
    return df


def load_spy(
    start: str = "2023-01-01",
    end: Optional[str] = None,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Load SPY daily OHLCV, cached to Data/spy_daily.parquet."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "spy_daily.parquet"

    if cache.exists() and not force_refresh:
        print(f"Loading SPY from cache: {cache}")
        return pl.read_parquet(cache)

    df = _fetch_ohlcv([BENCHMARK], start=start, end=end)
    df = df.drop("symbol")
    df.write_parquet(cache)
    print(f"Saved SPY to {cache}  ({df.shape[0]} rows)")
    return df


def pivot_to_wide(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """
    Convert long-format DataFrame to dict of 2-D NumPy arrays:
    shape (n_dates, n_symbols), aligned on date.
    Returns: {field: array, ...} plus 'dates' (sorted unique dates) and 'symbols'.
    """
    fields = ["open", "high", "low", "close", "volume"]
    dates = sorted(df["date"].unique().to_list())
    symbols = sorted(df["symbol"].unique().to_list())

    date_idx = {d: i for i, d in enumerate(dates)}
    sym_idx  = {s: i for i, s in enumerate(symbols)}
    n, m = len(dates), len(symbols)

    arrays = {f: np.full((n, m), np.nan) for f in fields}
    for row in df.iter_rows(named=True):
        i = date_idx[row["date"]]
        j = sym_idx[row["symbol"]]
        for f in fields:
            arrays[f][i, j] = row[f]

    arrays["dates"]   = np.array(dates)
    arrays["symbols"] = np.array(symbols)
    return arrays
