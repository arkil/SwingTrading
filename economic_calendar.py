"""
economic_calendar.py — Macro events calendar, news feed, and strategy impact context

Public API
----------
get_upcoming_events(days_ahead=45)       → list[dict]  sorted upcoming macro events
get_news_feed(max_items=20)              → list[dict]  recent financial headlines
get_event_context()                      → dict        countdown / warning summary
get_earnings_calendar(tickers, days=14)  → list[dict]  upcoming earnings dates
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, date
from typing import Optional
import logging
import re

import requests
import yfinance as yf
import pytz

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; compatible)"})


# ── Static high-impact macro event schedule ───────────────────────────────────
# Dates sourced from federalreserve.gov, BLS.gov, and CME.
# Decision day is the SECOND day of each two-day FOMC meeting.

_STATIC_EVENTS: list[dict] = [
    # ── FOMC rate decisions (2025) ─────────────────────────────────────────────
    {"date": "2025-01-29", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2025-03-19", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2025-05-07", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2025-06-18", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2025-07-30", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2025-09-17", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2025-10-29", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2025-12-10", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    # ── FOMC rate decisions (2026, estimated from typical Fed pattern) ─────────
    {"date": "2026-01-28", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2026-03-18", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2026-04-29", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2026-06-10", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2026-07-29", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2026-09-16", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2026-10-28", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},
    {"date": "2026-12-09", "event": "FOMC Rate Decision",       "category": "FOMC",   "impact": "HIGH"},

    # ── CPI releases (2025) ────────────────────────────────────────────────────
    {"date": "2025-01-15", "event": "CPI (Dec 2024)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2025-02-12", "event": "CPI (Jan 2025)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2025-03-12", "event": "CPI (Feb 2025)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2025-04-10", "event": "CPI (Mar 2025)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2025-05-13", "event": "CPI (Apr 2025)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2025-06-11", "event": "CPI (May 2025)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2025-07-11", "event": "CPI (Jun 2025)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2025-08-12", "event": "CPI (Jul 2025)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2025-09-10", "event": "CPI (Aug 2025)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2025-10-15", "event": "CPI (Sep 2025)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2025-11-13", "event": "CPI (Oct 2025)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2025-12-10", "event": "CPI (Nov 2025)",           "category": "CPI",    "impact": "HIGH"},
    # ── CPI releases (2026, estimated ~2nd Wed each month) ────────────────────
    {"date": "2026-01-14", "event": "CPI (Dec 2025)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2026-02-11", "event": "CPI (Jan 2026)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2026-03-11", "event": "CPI (Feb 2026)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2026-04-09", "event": "CPI (Mar 2026)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2026-05-13", "event": "CPI (Apr 2026)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2026-06-10", "event": "CPI (May 2026)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2026-07-09", "event": "CPI (Jun 2026)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2026-08-12", "event": "CPI (Jul 2026)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2026-09-09", "event": "CPI (Aug 2026)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2026-10-14", "event": "CPI (Sep 2026)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2026-11-11", "event": "CPI (Oct 2026)",           "category": "CPI",    "impact": "HIGH"},
    {"date": "2026-12-09", "event": "CPI (Nov 2026)",           "category": "CPI",    "impact": "HIGH"},

    # ── NFP / Jobs Reports (2025) ──────────────────────────────────────────────
    {"date": "2025-01-10", "event": "Non-Farm Payrolls (Dec)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2025-02-07", "event": "Non-Farm Payrolls (Jan)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2025-03-07", "event": "Non-Farm Payrolls (Feb)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2025-04-04", "event": "Non-Farm Payrolls (Mar)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2025-05-02", "event": "Non-Farm Payrolls (Apr)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2025-06-06", "event": "Non-Farm Payrolls (May)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2025-07-03", "event": "Non-Farm Payrolls (Jun)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2025-08-01", "event": "Non-Farm Payrolls (Jul)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2025-09-05", "event": "Non-Farm Payrolls (Aug)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2025-10-03", "event": "Non-Farm Payrolls (Sep)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2025-11-07", "event": "Non-Farm Payrolls (Oct)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2025-12-05", "event": "Non-Farm Payrolls (Nov)",  "category": "NFP",    "impact": "HIGH"},
    # ── NFP (2026, first Friday each month) ───────────────────────────────────
    {"date": "2026-01-09", "event": "Non-Farm Payrolls (Dec 2025)", "category": "NFP", "impact": "HIGH"},
    {"date": "2026-02-06", "event": "Non-Farm Payrolls (Jan)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2026-03-06", "event": "Non-Farm Payrolls (Feb)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2026-04-03", "event": "Non-Farm Payrolls (Mar)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2026-05-01", "event": "Non-Farm Payrolls (Apr)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2026-06-05", "event": "Non-Farm Payrolls (May)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2026-07-02", "event": "Non-Farm Payrolls (Jun)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2026-08-07", "event": "Non-Farm Payrolls (Jul)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2026-09-04", "event": "Non-Farm Payrolls (Aug)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2026-10-02", "event": "Non-Farm Payrolls (Sep)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2026-11-06", "event": "Non-Farm Payrolls (Oct)",  "category": "NFP",    "impact": "HIGH"},
    {"date": "2026-12-04", "event": "Non-Farm Payrolls (Nov)",  "category": "NFP",    "impact": "HIGH"},

    # ── GDP Advances (quarterly) ───────────────────────────────────────────────
    {"date": "2025-01-30", "event": "GDP Advance (Q4 2024)",    "category": "GDP",    "impact": "HIGH"},
    {"date": "2025-04-30", "event": "GDP Advance (Q1 2025)",    "category": "GDP",    "impact": "HIGH"},
    {"date": "2025-07-30", "event": "GDP Advance (Q2 2025)",    "category": "GDP",    "impact": "HIGH"},
    {"date": "2025-10-29", "event": "GDP Advance (Q3 2025)",    "category": "GDP",    "impact": "HIGH"},
    {"date": "2026-01-28", "event": "GDP Advance (Q4 2025)",    "category": "GDP",    "impact": "HIGH"},
    {"date": "2026-04-29", "event": "GDP Advance (Q1 2026)",    "category": "GDP",    "impact": "HIGH"},
    {"date": "2026-07-29", "event": "GDP Advance (Q2 2026)",    "category": "GDP",    "impact": "HIGH"},
    {"date": "2026-10-28", "event": "GDP Advance (Q3 2026)",    "category": "GDP",    "impact": "HIGH"},

    # ── Quadruple / Triple Witching (3rd Fri of Mar/Jun/Sep/Dec) ──────────────
    {"date": "2025-03-21", "event": "Quadruple Witching",       "category": "OpEx",   "impact": "MEDIUM"},
    {"date": "2025-06-20", "event": "Quadruple Witching",       "category": "OpEx",   "impact": "MEDIUM"},
    {"date": "2025-09-19", "event": "Quadruple Witching",       "category": "OpEx",   "impact": "MEDIUM"},
    {"date": "2025-12-19", "event": "Quadruple Witching",       "category": "OpEx",   "impact": "MEDIUM"},
    {"date": "2026-03-20", "event": "Quadruple Witching",       "category": "OpEx",   "impact": "MEDIUM"},
    {"date": "2026-06-19", "event": "Quadruple Witching",       "category": "OpEx",   "impact": "MEDIUM"},
    {"date": "2026-09-18", "event": "Quadruple Witching",       "category": "OpEx",   "impact": "MEDIUM"},
    {"date": "2026-12-18", "event": "Quadruple Witching",       "category": "OpEx",   "impact": "MEDIUM"},

    # ── Monthly OpEx (3rd Friday of non-witching months) ──────────────────────
    {"date": "2026-01-16", "event": "Monthly Options Expiration", "category": "OpEx", "impact": "LOW"},
    {"date": "2026-02-20", "event": "Monthly Options Expiration", "category": "OpEx", "impact": "LOW"},
    {"date": "2026-04-17", "event": "Monthly Options Expiration", "category": "OpEx", "impact": "LOW"},
    {"date": "2026-05-15", "event": "Monthly Options Expiration", "category": "OpEx", "impact": "LOW"},
    {"date": "2026-07-17", "event": "Monthly Options Expiration", "category": "OpEx", "impact": "LOW"},
    {"date": "2026-08-21", "event": "Monthly Options Expiration", "category": "OpEx", "impact": "LOW"},
    {"date": "2026-10-16", "event": "Monthly Options Expiration", "category": "OpEx", "impact": "LOW"},
    {"date": "2026-11-20", "event": "Monthly Options Expiration", "category": "OpEx", "impact": "LOW"},

    # ── Jackson Hole (late Aug) ────────────────────────────────────────────────
    {"date": "2025-08-21", "event": "Jackson Hole Symposium",   "category": "Fed",    "impact": "HIGH"},
    {"date": "2026-08-27", "event": "Jackson Hole Symposium",   "category": "Fed",    "impact": "HIGH"},

    # ── PCE (Fed's preferred inflation measure, last Fri of month) ────────────
    {"date": "2026-01-30", "event": "PCE Price Index (Dec 2025)", "category": "PCE",  "impact": "HIGH"},
    {"date": "2026-02-27", "event": "PCE Price Index (Jan 2026)", "category": "PCE",  "impact": "HIGH"},
    {"date": "2026-03-27", "event": "PCE Price Index (Feb 2026)", "category": "PCE",  "impact": "HIGH"},
    {"date": "2026-04-30", "event": "PCE Price Index (Mar 2026)", "category": "PCE",  "impact": "HIGH"},
    {"date": "2026-05-29", "event": "PCE Price Index (Apr 2026)", "category": "PCE",  "impact": "HIGH"},
    {"date": "2026-06-26", "event": "PCE Price Index (May 2026)", "category": "PCE",  "impact": "HIGH"},

    # ── Earnings seasons (approximate start dates) ─────────────────────────────
    {"date": "2026-04-13", "event": "Q1 2026 Earnings Season Begins", "category": "Earnings", "impact": "MEDIUM"},
    {"date": "2026-07-13", "event": "Q2 2026 Earnings Season Begins", "category": "Earnings", "impact": "MEDIUM"},
    {"date": "2026-10-12", "event": "Q3 2026 Earnings Season Begins", "category": "Earnings", "impact": "MEDIUM"},
]

# ── Category metadata ─────────────────────────────────────────────────────────
_CATEGORY_META = {
    "FOMC":     {"icon": "🏦", "color": "#f59e0b", "strategy_note": "Rate decisions move all sectors. Reduce size 1–2 days before. Avoid new breakout entries on decision day."},
    "CPI":      {"icon": "📊", "color": "#ef4444", "strategy_note": "High-impact inflation print. Growth/tech stocks most sensitive. Wait for print before entering momentum longs."},
    "NFP":      {"icon": "👷", "color": "#ef4444", "strategy_note": "Jobs data moves USD and rate expectations. Watch financials and consumer discretionary most."},
    "GDP":      {"icon": "📈", "color": "#f97316", "strategy_note": "Quarterly growth print. Lower market-moving than CPI/NFP but sets macro tone."},
    "PCE":      {"icon": "💹", "color": "#f59e0b", "strategy_note": "Fed's preferred inflation gauge. Critical for rate-path expectations; treat like CPI."},
    "OpEx":     {"icon": "📅", "color": "#8b5cf6", "strategy_note": "Options expiration increases intraday volatility and can cause pin risk near strike prices."},
    "Fed":      {"icon": "🎙️", "color": "#f59e0b", "strategy_note": "Major Fed speeches (Jackson Hole) can reprice rate expectations significantly."},
    "Earnings": {"icon": "💰", "color": "#22c55e", "strategy_note": "Individual stock volatility. Avoid holding overnight through a stock's own earnings."},
}

_IMPACT_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

# ── RSS feed sources (ordered by preference) ─────────────────────────────────
_RSS_FEEDS = [
    ("CNBC Markets",   "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
    ("CNBC Top News",  "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("MarketWatch",    "http://feeds.marketwatch.com/marketwatch/topstories/"),
    ("Reuters Biz",    "https://feeds.reuters.com/reuters/businessNews"),
    ("Yahoo Finance",  "https://finance.yahoo.com/rss/topfinstories"),
]

# ── Forex Factory live calendar (free, no API key) ────────────────────────────
_FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _today() -> date:
    return datetime.now(_ET).date()


def _days_away(event_date: date) -> int:
    return (event_date - _today()).days


def _parse_date(s: str) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Forex Factory live feed
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_ff_events(days_ahead: int = 14) -> list[dict]:
    """Pull this-week economic events from Forex Factory XML feed."""
    try:
        resp = _SESSION.get(_FF_URL, timeout=8)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        logger.debug(f"FF calendar fetch failed: {e}")
        return []

    events = []
    today = _today()
    cutoff = today + timedelta(days=days_ahead)

    for ev in root.findall("event"):
        try:
            title    = (ev.findtext("title") or "").strip()
            country  = (ev.findtext("country") or "").strip().upper()
            impact   = (ev.findtext("impact") or "").strip().capitalize()
            date_str = (ev.findtext("date") or "").strip()
            time_str = (ev.findtext("time") or "").strip()
            forecast = (ev.findtext("forecast") or "").strip()
            prev     = (ev.findtext("previous") or "").strip()

            if country != "USD":
                continue  # US-only events for now

            ev_date = _parse_date(date_str)
            if ev_date is None or ev_date < today or ev_date > cutoff:
                continue

            impact_map = {"High": "HIGH", "Medium": "MEDIUM", "Low": "LOW"}
            impact_norm = impact_map.get(impact, "LOW")

            events.append({
                "date":     ev_date,
                "event":    title,
                "category": "Economic",
                "impact":   impact_norm,
                "time":     time_str,
                "forecast": forecast,
                "previous": prev,
                "source":   "ForexFactory",
            })
        except Exception:
            continue

    return events


# ═════════════════════════════════════════════════════════════════════════════
# Static events
# ═════════════════════════════════════════════════════════════════════════════

def _get_static_events(days_ahead: int = 45) -> list[dict]:
    today  = _today()
    cutoff = today + timedelta(days=days_ahead)
    result = []
    for ev in _STATIC_EVENTS:
        ev_date = _parse_date(ev["date"])
        if ev_date and today <= ev_date <= cutoff:
            result.append({
                "date":     ev_date,
                "event":    ev["event"],
                "category": ev["category"],
                "impact":   ev["impact"],
                "time":     "08:30 ET" if ev["category"] in ("CPI", "NFP", "GDP", "PCE") else "14:00 ET" if ev["category"] == "FOMC" else "",
                "forecast": "",
                "previous": "",
                "source":   "Scheduled",
            })
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Earnings calendar via yfinance
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_one_earnings(sym: str, today: date, cutoff: date) -> list[dict]:
    try:
        tk = yf.Ticker(sym)
        cal = tk.calendar
        if cal is None:
            return []
        if isinstance(cal, dict):
            earn_dates = cal.get("Earnings Date", [])
        else:
            return []
        if not isinstance(earn_dates, (list, tuple)):
            earn_dates = [earn_dates]
        results = []
        for ed in earn_dates:
            if ed is None:
                continue
            if hasattr(ed, "date"):
                ed = ed.date()
            elif isinstance(ed, str):
                ed = _parse_date(ed)
            if ed and today <= ed <= cutoff:
                results.append({
                    "date":     ed,
                    "event":    f"{sym} Earnings",
                    "category": "Earnings",
                    "impact":   "MEDIUM",
                    "time":     "Pre/After Market",
                    "forecast": "",
                    "previous": "",
                    "source":   "yfinance",
                })
        return results
    except Exception:
        return []


def get_earnings_calendar(tickers: list[str], days: int = 14) -> list[dict]:
    """Return upcoming earnings for the given tickers within `days` days (parallel fetch)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    today  = _today()
    cutoff = today + timedelta(days=days)
    results = []

    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(_fetch_one_earnings, sym, today, cutoff): sym for sym in tickers}
        for future in as_completed(futures):
            results.extend(future.result())

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main public function — combined events
# ═════════════════════════════════════════════════════════════════════════════

def get_upcoming_events(
    days_ahead: int = 45,
    include_earnings: bool = False,
    tickers: Optional[list[str]] = None,
) -> list[dict]:
    """
    Return upcoming macro events sorted by date (ascending).
    Merges static schedule + live Forex Factory feed.
    Deduplicates by (date, category).
    """
    static = _get_static_events(days_ahead)
    live   = _fetch_ff_events(min(days_ahead, 14))

    # Build dedup key set from static events
    seen = {(e["date"], e["category"]) for e in static}

    merged = list(static)
    for ev in live:
        # Only add FF events that aren't already covered by static schedule
        key = (ev["date"], ev["category"])
        if key not in seen:
            merged.append(ev)
            seen.add(key)

    if include_earnings and tickers:
        earn = get_earnings_calendar(tickers, min(days_ahead, 21))
        # Deduplicate by (date, event)
        earn_seen = {(e["date"], e["event"]) for e in merged}
        for ev in earn:
            if (ev["date"], ev["event"]) not in earn_seen:
                merged.append(ev)

    merged.sort(key=lambda e: (_days_away(e["date"]), _IMPACT_ORDER.get(e["impact"], 9)))
    return merged


# ═════════════════════════════════════════════════════════════════════════════
# Strategy context (banner / warnings for scanner pages)
# ═════════════════════════════════════════════════════════════════════════════

def get_event_context(days_ahead: int = 7) -> dict:
    """
    Returns a compact context dict used to inject warnings into scanner pages.

    Keys:
        warnings    list[str]   human-readable warning strings
        next_events list[dict]  upcoming HIGH-impact events within days_ahead
        banner_html str         pre-built HTML banner (empty if no warnings)
    """
    events    = get_upcoming_events(days_ahead)
    high_only = [e for e in events if e["impact"] == "HIGH"]
    warnings  = []

    for ev in high_only:
        d = _days_away(ev["date"])
        meta = _CATEGORY_META.get(ev["category"], {})
        icon = meta.get("icon", "⚡")
        if d == 0:
            warnings.append(f"{icon} **TODAY** — {ev['event']} ({ev.get('time','')})")
        elif d == 1:
            warnings.append(f"{icon} **TOMORROW** — {ev['event']}")
        elif d <= 3:
            warnings.append(f"{icon} In {d} days — {ev['event']} ({ev['date'].strftime('%a %b %d')})")
        elif d <= days_ahead:
            warnings.append(f"{icon} In {d} days — {ev['event']} ({ev['date'].strftime('%b %d')})")

    if not warnings:
        banner_html = ""
    else:
        items_html = "".join(f"<li style='margin:2px 0'>{w}</li>" for w in warnings)
        banner_html = (
            "<div style='background:rgba(245,158,11,0.12);border-left:3px solid #f59e0b;"
            "border-radius:6px;padding:10px 14px;margin-bottom:12px;font-size:0.85rem'>"
            "<strong style='color:#f59e0b'>⚠️ Upcoming Market-Moving Events</strong>"
            f"<ul style='margin:6px 0 0 0;padding-left:18px;color:#cbd5e1'>{items_html}</ul>"
            "</div>"
        )

    return {
        "warnings":    warnings,
        "next_events": high_only,
        "banner_html": banner_html,
    }


# ═════════════════════════════════════════════════════════════════════════════
# News feed via RSS
# ═════════════════════════════════════════════════════════════════════════════

def _parse_rss(url: str, source_name: str, max_items: int = 8) -> list[dict]:
    try:
        resp = _SESSION.get(url, timeout=8)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        logger.debug(f"RSS fetch failed ({source_name}): {e}")
        return []

    items = []
    # Standard RSS 2.0
    for item in root.iter("item"):
        title   = (item.findtext("title")       or "").strip()
        link    = (item.findtext("link")        or "").strip()
        pub     = (item.findtext("pubDate")     or "").strip()
        summary = (item.findtext("description") or "").strip()

        # Strip HTML tags from summary
        summary = re.sub(r"<[^>]+>", "", summary)[:200]

        items.append({
            "title":   title,
            "link":    link,
            "pub":     pub,
            "summary": summary,
            "source":  source_name,
        })
        if len(items) >= max_items:
            break

    return items


def get_news_feed(max_items: int = 24) -> list[dict]:
    """
    Fetch recent financial headlines from multiple RSS sources.
    Returns merged list up to max_items, newest-first.
    """
    all_items: list[dict] = []
    per_feed = max(4, max_items // len(_RSS_FEEDS))

    for name, url in _RSS_FEEDS:
        items = _parse_rss(url, name, per_feed)
        all_items.extend(items)
        if len(all_items) >= max_items * 2:
            break

    # Deduplicate by title
    seen_titles: set[str] = set()
    unique = []
    for item in all_items:
        t = item["title"].lower().strip()
        if t and t not in seen_titles:
            seen_titles.add(t)
            unique.append(item)

    return unique[:max_items]
