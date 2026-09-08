"""
Polygon.io Data Cache Manager
==============================

Stores real option prices locally for backtesting reuse.
Reduces API calls, speeds up backtests, provides reproducible data.

Features:
- Cache to CSV (fast, simple)
- Automatic cache lookup before API calls
- Cache invalidation (update if older than N days)
- Fallback to API if cache miss
- Debug mode to see cache hits/misses
"""

from typing import Optional, Dict, List
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ── Cache directory setup ──────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "Data" / "polygon_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_META_FILE = CACHE_DIR / "cache_metadata.json"
CACHE_MAX_AGE_DAYS = 30  # Refresh cache older than this


# ── Cache metadata (tracks what we have) ───────────────────────────────────────

def _load_cache_metadata() -> dict:
    """Load cache index of what's stored locally."""
    if CACHE_META_FILE.exists():
        try:
            return json.loads(CACHE_META_FILE.read_text())
        except Exception as e:
            logger.warning(f"Cache metadata corrupted: {e}")
            return {}
    return {}


def _save_cache_metadata(metadata: dict):
    """Save cache index."""
    try:
        CACHE_META_FILE.write_text(json.dumps(metadata, indent=2, default=str))
    except Exception as e:
        logger.warning(f"Failed to save cache metadata: {e}")


def _get_cache_key(symbol: str, strike: float, expiry_str: str, direction: str) -> str:
    """Generate cache filename: SPY_500_20240119_CALL.csv"""
    return f"{symbol}_{int(strike)}_{expiry_str}_{direction.upper()}.csv"


# ── Cache lookup ───────────────────────────────────────────────────────────────

def get_cached_option_price(
    symbol: str,
    strike: float,
    expiry_str: str,
    direction: str,
    target_date: str = None,
) -> Optional[Dict]:
    """
    Lookup option price in local cache.

    Returns dict with cached price data or None if not found/expired.
    """
    cache_key = _get_cache_key(symbol, strike, expiry_str, direction)
    cache_file = CACHE_DIR / cache_key

    # Check if file exists
    if not cache_file.exists():
        return None

    # Check age
    file_age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
    if file_age_days > CACHE_MAX_AGE_DAYS:
        logger.debug(f"Cache expired for {cache_key} (age: {file_age_days} days)")
        return None

    # Load and parse
    try:
        df = pd.read_csv(cache_file)

        # Find row for target date (if specified)
        if target_date:
            df['date'] = pd.to_datetime(df['date'])
            target_dt = pd.to_datetime(target_date)

            # Find closest date
            df['date_diff'] = abs((df['date'] - target_dt).dt.days)
            row = df.loc[df['date_diff'].idxmin()]

            if row['date_diff'] > 30:  # Too far from target date
                return None

            return {
                'price': row['close'],
                'open': row.get('open', row['close']),
                'high': row.get('high', row['close']),
                'low': row.get('low', row['close']),
                'volume': int(row.get('volume', 0)),
                'timestamp': row['date'],
                'source': 'cache_local',
                'cache_age_days': file_age_days,
            }
        else:
            # Return most recent
            row = df.iloc[-1]
            return {
                'price': row['close'],
                'open': row.get('open', row['close']),
                'high': row.get('high', row['close']),
                'low': row.get('low', row['close']),
                'volume': int(row.get('volume', 0)),
                'timestamp': row['date'],
                'source': 'cache_local',
                'cache_age_days': file_age_days,
            }
    except Exception as e:
        logger.warning(f"Failed to read cache {cache_key}: {e}")
        return None


# ── Cache storage ──────────────────────────────────────────────────────────────

def cache_polygon_data(
    symbol: str,
    strike: float,
    expiry_str: str,
    direction: str,
    data_list: list,
    verbose: bool = False,
) -> bool:
    """
    Store Polygon.io data in local cache.

    Args:
        symbol: "SPY"
        strike: 500.0
        expiry_str: "20240119"
        direction: "call" or "put"
        data_list: List of dicts with {date, open, high, low, close, volume}
        verbose: Print cache operations

    Returns:
        True if cached successfully
    """
    if not data_list:
        return False

    try:
        # Convert to DataFrame
        df = pd.DataFrame(data_list)

        # Ensure date column
        if 'date' not in df.columns and 'timestamp' in df.columns:
            df['date'] = df['timestamp']

        # Save to CSV
        cache_key = _get_cache_key(symbol, strike, expiry_str, direction)
        cache_file = CACHE_DIR / cache_key
        df.to_csv(cache_file, index=False)

        # Update metadata
        metadata = _load_cache_metadata()
        metadata[cache_key] = {
            'symbol': symbol,
            'strike': float(strike),
            'expiry': expiry_str,
            'direction': direction.upper(),
            'rows': len(df),
            'cached_at': datetime.now().isoformat(),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
        }
        _save_cache_metadata(metadata)

        if verbose:
            logger.info(f"✅ Cached {len(df)} bars: {cache_key}")

        return True
    except Exception as e:
        logger.warning(f"Failed to cache {symbol} {strike}: {e}")
        return False


# ── Bulk cache operations ──────────────────────────────────────────────────────

def clear_cache(symbol: str = None, older_than_days: int = None):
    """
    Clear local cache.

    Args:
        symbol: Clear only this symbol (None = all)
        older_than_days: Clear only cache older than N days (None = all)
    """
    cleared = 0

    for cache_file in CACHE_DIR.glob("*.csv"):
        # Filter by symbol
        if symbol and not cache_file.name.startswith(symbol.upper()):
            continue

        # Filter by age
        if older_than_days is not None:
            file_age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
            if file_age_days < older_than_days:
                continue

        cache_file.unlink()
        cleared += 1

    logger.info(f"Cleared {cleared} cache files")


def cache_stats() -> dict:
    """Get cache statistics."""
    cache_files = list(CACHE_DIR.glob("*.csv"))
    total_rows = 0

    for cache_file in cache_files:
        try:
            df = pd.read_csv(cache_file)
            total_rows += len(df)
        except Exception:
            pass

    return {
        'cache_dir': str(CACHE_DIR),
        'cached_options': len(cache_files),
        'total_price_bars': total_rows,
        'cache_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
    }


def print_cache_stats():
    """Print cache statistics."""
    stats = cache_stats()
    print(f"\n📊 Polygon.io Cache Statistics:")
    print(f"   Location: {stats['cache_dir']}")
    print(f"   Cached Options: {stats['cached_options']}")
    print(f"   Total Price Bars: {stats['total_price_bars']}")
    print(f"   Cache Size: {stats['cache_size_mb']:.1f} MB")


# ── Integration with Polygon API ───────────────────────────────────────────────

def fetch_and_cache_polygon_data(
    polygon_client,
    symbol: str,
    strike: float,
    expiry_str: str,
    direction: str,
    start_date: str = "2023-01-01",
    end_date: str = None,
    verbose: bool = False,
) -> Optional[Dict]:
    """
    Fetch from Polygon.io API and store in local cache.

    Returns latest price data with source='polygon_api'.
    """
    try:
        # Build option ticker
        direction_char = "C" if direction.lower() == "call" else "P"
        option_ticker = f"O:{symbol}{expiry_str}{direction_char}{int(strike):08d}"

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Fetch from Polygon.io
        bars = list(polygon_client.get_aggs(
            ticker=option_ticker,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=10000
        ))

        if not bars:
            return None

        # Convert to cache format
        data_list = []
        for bar in bars:
            data_list.append({
                'date': bar.timestamp.strftime("%Y-%m-%d") if hasattr(bar.timestamp, 'strftime') else str(bar.timestamp),
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
            })

        # Cache it
        cache_success = cache_polygon_data(
            symbol, strike, expiry_str, direction, data_list, verbose=verbose
        )

        if cache_success and verbose:
            logger.info(f"✅ Fetched & cached {len(data_list)} bars for {symbol} {strike}")

        # Return latest
        latest = data_list[-1]
        return {
            'price': latest['close'],
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'],
            'volume': latest['volume'],
            'timestamp': latest['date'],
            'source': 'polygon_api_fresh',
            'bars_cached': len(data_list),
        }

    except Exception as e:
        logger.debug(f"Polygon.io fetch failed for {symbol} {strike}: {e}")
        return None


# ── Example usage ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Show cache stats
    print_cache_stats()

    # Example: Clear old cache
    # clear_cache(older_than_days=30)

    # Example: Get cached price
    # price = get_cached_option_price("SPY", 500.0, "20240119", "call")
    # print(price)
