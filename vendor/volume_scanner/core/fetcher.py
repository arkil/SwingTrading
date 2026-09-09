"""
core/fetcher.py — Finnhub API client

Handles all HTTP communication with Finnhub:
  - Real-time quotes  (/quote)
  - OHLCV candles     (/stock/candle)
  - Earnings calendar (/stock/earnings)
  - Insider trades    (/stock/insider-transactions)
  - Company news      (/company-news)

Rate limiting:
  Finnhub free tier allows 60 calls/minute.
  This module enforces a token-bucket limiter so you never hit 429 errors.

To swap in a different data provider:
  1. Create a new class that inherits from BaseFetcher
  2. Implement get_quote() and get_candles() with the same signatures
  3. Point config.DATA_PROVIDER to your new class name
"""

import time
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional
from threading import Lock
import datetime

logger = logging.getLogger(__name__)


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class QuoteData:
    """
    Real-time quote snapshot from Finnhub /quote endpoint.

    Fields:
        symbol      : Ticker symbol (e.g. "AAPL")
        price       : Current price (c)
        change      : Price change from prior close (d)
        change_pct  : Percentage change from prior close (dp)
        high        : Day high (h)
        low         : Day low (l)
        open        : Day open (o)
        prev_close  : Previous close price (pc)
        volume      : Today's volume — NOTE: Finnhub /quote does not always
                      return volume; use candle data for reliable volume.
        timestamp   : Unix timestamp of the quote
        ok          : False if the fetch failed or returned no data
    """
    symbol:     str
    price:      float = 0.0
    change:     float = 0.0
    change_pct: float = 0.0
    high:       float = 0.0
    low:        float = 0.0
    open:       float = 0.0
    prev_close: float = 0.0
    volume:     int   = 0
    timestamp:  int   = 0
    ok:         bool  = False


@dataclass
class CandleData:
    """
    OHLCV candle array from Finnhub /stock/candle endpoint.

    Fields:
        symbol      : Ticker symbol
        opens       : List of open prices
        highs       : List of high prices
        lows        : List of low prices
        closes      : List of close prices
        volumes     : List of volumes (this is what we use for ratio calculation)
        timestamps  : List of Unix timestamps for each bar
        ok          : False if fetch failed or no data returned
        resolution  : The candle resolution that was fetched ("1", "15", "60", "D", etc.)

    Indexing:
        Index 0 = oldest bar
        Index -1 = most recent bar (may be in-progress for intraday)
        Index -2 = last COMPLETED bar (use this for ratio calculations)
    """
    symbol:     str
    opens:      list = field(default_factory=list)
    highs:      list = field(default_factory=list)
    lows:       list = field(default_factory=list)
    closes:     list = field(default_factory=list)
    volumes:    list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    ok:         bool = False
    resolution: str  = ""

    def __len__(self):
        return len(self.closes)

    @property
    def latest_close(self) -> float:
        return self.closes[-1] if self.closes else 0.0

    @property
    def latest_volume(self) -> int:
        return self.volumes[-1] if self.volumes else 0


# ── Rate limiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Token-bucket rate limiter.

    Ensures we never exceed `max_calls` requests within any `period_seconds` window.
    Thread-safe for use with concurrent.futures.

    Usage:
        limiter = RateLimiter(max_calls=55, period_seconds=60)
        limiter.wait()   # blocks until a call slot is available
        response = requests.get(...)
    """

    def __init__(self, max_calls: int = 55, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self._calls: list[float] = []
        self._lock = Lock()

    def wait(self):
        """Block until a rate-limit slot is available."""
        with self._lock:
            now = time.time()
            # Remove calls older than the period window
            self._calls = [t for t in self._calls if now - t < self.period]
            if len(self._calls) >= self.max_calls:
                sleep_for = self.period - (now - self._calls[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
            self._calls.append(time.time())


# ── Base fetcher interface ─────────────────────────────────────────────────────

class BaseFetcher:
    """
    Abstract interface for a market data fetcher.

    To use a different data provider (e.g. Alpha Vantage, Polygon, IEX):
      1. Subclass this
      2. Implement get_quote() and get_candles()
      3. Match the return types exactly (QuoteData, CandleData)
    """

    def get_quote(self, symbol: str) -> QuoteData:
        raise NotImplementedError

    def get_candles(self, symbol: str, resolution: str,
                    from_ts: int, to_ts: int) -> CandleData:
        raise NotImplementedError


# ── Finnhub fetcher ────────────────────────────────────────────────────────────

class FinnhubFetcher(BaseFetcher):
    """
    Finnhub REST API client.

    Free tier limits:
      - 60 API calls / minute
      - Real-time US stock quotes
      - 1-minute candles (US stocks only)
      - Daily, weekly candles
      - 1 year of intraday history on free plan

    API documentation: https://finnhub.io/docs/api

    Parameters:
        api_key         : Your Finnhub API key
        max_calls_min   : Rate limit cap (default 55, safely under the 60 limit)
        timeout_seconds : HTTP request timeout
        retries         : Number of retry attempts on transient failures
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str, max_calls_min: int = 55,
                 timeout_seconds: int = 10, retries: int = 2):
        if not api_key or api_key == "your_finnhub_api_key_here":
            raise ValueError(
                "Invalid Finnhub API key. Get a free key at https://finnhub.io"
            )
        self.api_key = api_key
        self.timeout = timeout_seconds
        self.retries = retries
        self._limiter = RateLimiter(max_calls=max_calls_min, period_seconds=60.0)
        self._session = requests.Session()
        self._session.params = {"token": self.api_key}  # type: ignore

    def _get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """
        Internal GET helper with rate limiting and retry.

        Returns parsed JSON dict or None on failure.
        """
        url = f"{self.BASE_URL}/{endpoint}"
        for attempt in range(self.retries + 1):
            try:
                self._limiter.wait()
                resp = self._session.get(url, params=params or {}, timeout=self.timeout)
                if resp.status_code == 429:
                    logger.warning("Rate limited by Finnhub — waiting 10s")
                    time.sleep(10)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt < self.retries:
                    wait = 2 ** attempt
                    logger.debug(f"Retry {attempt+1} for {endpoint}: {e} (waiting {wait}s)")
                    time.sleep(wait)
                else:
                    logger.warning(f"Failed to fetch {endpoint} after {self.retries+1} attempts: {e}")
                    return None
        return None

    def get_quote(self, symbol: str) -> QuoteData:
        """
        Fetch real-time quote for a symbol.

        Finnhub /quote fields:
            c  = current price
            d  = change
            dp = percent change
            h  = day high
            l  = day low
            o  = open
            pc = previous close
            t  = timestamp

        Note: Finnhub /quote does NOT reliably return volume.
              Use get_candles() with resolution="D" for volume data.
        """
        data = self._get("quote", {"symbol": symbol})
        if not data or data.get("c", 0) == 0:
            logger.debug(f"No quote data for {symbol}")
            return QuoteData(symbol=symbol, ok=False)

        return QuoteData(
            symbol=symbol,
            price=data.get("c", 0.0),
            change=data.get("d", 0.0),
            change_pct=data.get("dp", 0.0),
            high=data.get("h", 0.0),
            low=data.get("l", 0.0),
            open=data.get("o", 0.0),
            prev_close=data.get("pc", 0.0),
            timestamp=data.get("t", 0),
            ok=True,
        )

    def get_candles(self, symbol: str, resolution: str,
                    from_ts: int, to_ts: int) -> CandleData:
        """
        Fetch OHLCV candles for a symbol.

        Parameters:
            symbol     : Ticker (e.g. "AAPL")
            resolution : Bar size — "1", "5", "15", "30", "60", "D", "W", "M"
            from_ts    : Start time as Unix timestamp (seconds)
            to_ts      : End time as Unix timestamp (seconds)

        Returns CandleData with parallel arrays: opens, highs, lows, closes, volumes, timestamps.

        Important: The last bar (index -1) may be in-progress (current bar).
                   Use index -2 for the last COMPLETED bar when computing ratios.
        """
        data = self._get("stock/candle", {
            "symbol":     symbol,
            "resolution": resolution,
            "from":       from_ts,
            "to":         to_ts,
        })

        if not data or data.get("s") != "ok":
            logger.debug(f"No candle data for {symbol} @ {resolution}")
            return CandleData(symbol=symbol, ok=False, resolution=resolution)

        return CandleData(
            symbol=symbol,
            opens=data.get("o", []),
            highs=data.get("h", []),
            lows=data.get("l", []),
            closes=data.get("c", []),
            volumes=data.get("v", []),
            timestamps=data.get("t", []),
            ok=True,
            resolution=resolution,
        )

    def get_earnings_calendar(self, symbol: str) -> Optional[list]:
        """
        Fetch upcoming earnings dates for a symbol.

        Returns list of earnings events sorted by date, or None on failure.
        Each event: {"date": "YYYY-MM-DD", "epsEstimate": float, ...}
        """
        import datetime
        today = datetime.date.today()
        future = today + datetime.timedelta(days=90)
        data = self._get("calendar/earnings", {
            "symbol": symbol,
            "from":   today.isoformat(),
            "to":     future.isoformat(),
        })
        if not data:
            return None
        return data.get("earningsCalendar", [])

    def get_insider_trades(self, symbol: str) -> Optional[list]:
        """
        Fetch recent insider transactions for a symbol.

        Returns list of transactions, or None on failure.
        Each transaction: {"name": str, "share": int, "change": int,
                           "transactionDate": "YYYY-MM-DD", "transactionCode": str}
        transactionCode "P" = purchase, "S" = sale
        """
        data = self._get("stock/insider-transactions", {"symbol": symbol})
        if not data:
            return None
        return data.get("data", [])

    def get_company_news(self, symbol: str, days_back: int = 3) -> Optional[list]:
        """
        Fetch recent news for a symbol.

        Returns list of news items or None.
        Each item: {"headline": str, "summary": str, "datetime": int, "sentiment": float}
        """
        import datetime
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=days_back)
        data = self._get("company-news", {
            "symbol": symbol,
            "from":   from_date.isoformat(),
            "to":     today.isoformat(),
        })
        return data if isinstance(data, list) else None


# ── yfinance fetcher ───────────────────────────────────────────────────────────

class YFinanceFetcher(BaseFetcher):
    """
    Yahoo Finance fetcher via yfinance — free, no API key required.

    Implements the same get_quote() / get_candles() interface as FinnhubFetcher
    so it is a drop-in replacement.

    Resolution mapping (Finnhub → yfinance interval):
        "1"  → "1m"
        "5"  → "5m"
        "15" → "15m"
        "30" → "30m"
        "60" → "60m"
        "D"  → "1d"
        "W"  → "1wk"
    """

    _RESOLUTION_MAP = {
        "1":  "1m",
        "5":  "5m",
        "15": "15m",
        "30": "30m",
        "60": "60m",
        "D":  "1d",
        "W":  "1wk",
    }

    def __init__(self):
        try:
            import yfinance  # noqa: F401
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

    def get_quote(self, symbol: str) -> QuoteData:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            price = getattr(info, "last_price", None) or 0.0
            prev_close = getattr(info, "previous_close", None) or 0.0
            change = price - prev_close if price and prev_close else 0.0
            change_pct = (change / prev_close * 100) if prev_close else 0.0
            volume = getattr(info, "last_volume", None) or 0
            return QuoteData(
                symbol=symbol,
                price=float(price),
                change=float(change),
                change_pct=float(change_pct),
                high=float(getattr(info, "day_high", 0) or 0),
                low=float(getattr(info, "day_low", 0) or 0),
                open=float(getattr(info, "open", 0) or 0),
                prev_close=float(prev_close),
                volume=int(volume),
                timestamp=int(time.time()),
                ok=bool(price),
            )
        except Exception as e:
            logger.debug(f"yfinance quote error {symbol}: {e}")
            return QuoteData(symbol=symbol, ok=False)

    def get_candles(self, symbol: str, resolution: str,
                    from_ts: int, to_ts: int) -> CandleData:
        try:
            import yfinance as yf
            interval = self._RESOLUTION_MAP.get(resolution, "1d")
            start = datetime.datetime.utcfromtimestamp(from_ts).strftime("%Y-%m-%d")
            end = datetime.datetime.utcfromtimestamp(to_ts + 86400).strftime("%Y-%m-%d")

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)

            if df is None or df.empty:
                return CandleData(symbol=symbol, ok=False, resolution=resolution)

            return CandleData(
                symbol=symbol,
                opens=df["Open"].tolist(),
                highs=df["High"].tolist(),
                lows=df["Low"].tolist(),
                closes=df["Close"].tolist(),
                volumes=[int(v) for v in df["Volume"].tolist()],
                timestamps=[int(t.timestamp()) for t in df.index],
                ok=True,
                resolution=resolution,
            )
        except Exception as e:
            logger.debug(f"yfinance candle error {symbol} @ {resolution}: {e}")
            return CandleData(symbol=symbol, ok=False, resolution=resolution)
