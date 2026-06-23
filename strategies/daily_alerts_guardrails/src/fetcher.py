"""
Data fetcher with parquet cache.
Downloads OHLCV from yfinance, caches per-ticker in Data/cache/.
"""

import os
import sys
import time
import warnings
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

_CACHE_DIR = Path(__file__).parent.parent / "Data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_TTL_HOURS = 20       # refresh if older than this
_YF_LOCK = threading.Lock()


def _cache_path(ticker: str) -> Path:
    return _CACHE_DIR / f"{ticker.upper()}.parquet"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=_CACHE_TTL_HOURS)


def fetch_ticker(ticker: str, days: int = 800, force: bool = False) -> pd.DataFrame:
    """
    Return daily OHLCV DataFrame for `ticker` covering `days` of history.
    Uses parquet cache; re-downloads if stale.

    Returns empty DataFrame on failure.
    """
    path = _cache_path(ticker)

    if not force and _is_fresh(path):
        try:
            df = pd.read_parquet(path)
            cutoff = datetime.today() - timedelta(days=days + 30)
            df = df[df.index >= pd.Timestamp(cutoff.date())]
            return df
        except Exception:
            pass

    end   = datetime.today()
    start = end - timedelta(days=days + 60)
    for attempt in range(3):
        try:
            with _YF_LOCK:
                raw = yf.download(
                    ticker, start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="1d", auto_adjust=True, progress=False,
                )
            if raw is None or raw.empty:
                return pd.DataFrame()
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if getattr(df.index, "tz", None):
                df.index = df.index.tz_localize(None)
            df.index = pd.to_datetime(df.index.date)
            df.index.name = "Date"
            df.to_parquet(path)
            return df
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return pd.DataFrame()


def fetch_spy(days: int = 800) -> pd.DataFrame:
    return fetch_ticker("SPY", days=days)


def fetch_batch(
    tickers: List[str],
    days: int = 800,
    max_workers: int = 12,
    progress: bool = True,
) -> dict:
    """
    Fetch multiple tickers in parallel. Returns {ticker: DataFrame}.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    result = {}
    total = len(tickers)
    done  = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_ticker, tk, days): tk for tk in tickers}
        for fut in as_completed(futs):
            tk = futs[fut]
            done += 1
            if progress and done % 50 == 0:
                print(f"  fetched {done}/{total} …", flush=True)
            try:
                df = fut.result()
                if not df.empty:
                    result[tk] = df
            except Exception:
                pass

    return result
