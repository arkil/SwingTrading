"""
Shared upcoming-earnings source
===============================
ONE place that answers "which tickers report in the next N days" — disk-cached
so yfinance is hit for earnings dates at most once a day.  Reused by:
  • the Home page today-earnings strip
  • the Macro Calendar "Earnings Watch" tab
  • the Earnings IV-Crush scanner + its nightly prefetch job

It wraps `economic_calendar.get_earnings_calendar` (the exact function the Home
page already uses) and adds a JSON disk cache keyed by (universe, days).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "Data", "earnings_iv")

# universe keys understood everywhere in the app
UNIVERSES = {
    "watchlist": "Watchlist (~112)",
    "ndx": "Nasdaq 100",
    "sp500": "S&P 500",
    "both": "S&P 500 + Nasdaq",
}


def universe_tickers(key: str) -> list[str]:
    from livermore_pivotal_screener import (
        WATCHLIST_TICKERS, get_sp500_tickers, get_nasdaq100_tickers, _dedup,
    )
    if key == "watchlist":
        return list(WATCHLIST_TICKERS)
    if key == "ndx":
        return get_nasdaq100_tickers()
    if key == "sp500":
        return get_sp500_tickers()
    if key == "both":
        return _dedup(get_sp500_tickers() + get_nasdaq100_tickers())
    return list(WATCHLIST_TICKERS)


def _cache_path(universe: str, days: int) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"upcoming_{universe}_{days}d.json")


def get_upcoming_earnings(
    days: int = 14,
    universe: str = "both",
    max_age_hours: float = 20.0,
    force: bool = False,
) -> list[dict]:
    """[{symbol, date (ISO str), time}] for names in `universe` reporting within
    `days` days. Served from the JSON disk cache unless it's older than
    `max_age_hours` (or force=True), in which case it's rebuilt from yfinance."""
    path = _cache_path(universe, days)

    if not force and os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < max_age_hours * 3600:
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass

    from economic_calendar import get_earnings_calendar
    tickers = universe_tickers(universe)
    rows = get_earnings_calendar(tickers, days=days)

    out, seen = [], set()
    for e in rows:
        sym = e["event"].split(" ")[0]
        if sym in seen:
            continue
        seen.add(sym)
        d = e["date"]
        out.append({
            "symbol": sym,
            "date": d.isoformat() if isinstance(d, date) else str(d),
            "time": e.get("time", ""),
        })
    out.sort(key=lambda x: (x["date"], x["symbol"]))

    try:
        with open(path, "w") as f:
            json.dump(out, f)
    except Exception:
        pass
    return out


def cache_age_hours(universe: str = "both", days: int = 14) -> float | None:
    path = _cache_path(universe, days)
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 3600.0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="both", choices=list(UNIVERSES))
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rows = get_upcoming_earnings(a.days, a.universe, force=a.force)
    print(f"{len(rows)} names reporting in next {a.days}d ({a.universe}):")
    for r in rows:
        print(f"  {r['date']}  {r['symbol']:6s}  {r['time']}")
