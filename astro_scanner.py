"""
Financial Astrology Scanner
============================
Techniques from W.D. Gann, Larry Pesavento, Bill Meridian & Rajeev Prakash.

Signals implemented
  • Moon phases     — New Moon (bullish reversal), Full Moon (bearish reversal)
  • Lunar quarters  — First/Last Quarter = caution zones
  • Mercury Rx      — Volatility, false breakouts (3× per year)
  • Venus/Mars Rx   — Sentiment extremes, commodity reversal
  • Saturn–Uranus   — Systemic / crash risk (2008, 2020 confirmed)
  • Jupiter–Saturn  — 20-year economic cycle aspects
  • Mars–Saturn     — Sharp sell-off signal
  • Jupiter–Uranus  — Surprise bull run / tech breakout
  • Solar ingress   — Cardinal signs = Gann seasonal turns
  • Gann cycles     — 45/90/144/180/270/360 days from pivot H/L

Sources
  Pesavento & Smoleny — "A Trader's Guide to Financial Astrology" (Wiley)
  Bill Meridian — "Planetary Stock Trading"
  Rajeev Prakash — rajeevprakash.com
  Bank of Scotland study — New Moon buy / Full Moon sell strategy
  Federal Reserve Bank of Atlanta — geomagnetic storm / market correlation
"""

import ephem
import math
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# ─── Ephemeris helpers ────────────────────────────────────────────────────────

def _ed(dt) -> ephem.Date:
    if isinstance(dt, ephem.Date):
        return dt
    return ephem.Date(dt)

def _dt(d: ephem.Date) -> datetime:
    return d.datetime()


# ─── Moon Phase ───────────────────────────────────────────────────────────────

MOON_SIGNAL_MAP = {
    "New Moon":       ("BULLISH",  3, "Buy signal — Bank of Scotland: New Moon initiates up-move"),
    "Full Moon":      ("BEARISH",  3, "Sell/short signal — Bank of Scotland: Short on Full Moon"),
    "First Quarter":  ("NEUTRAL",  1, "Caution zone — momentum check, possible pause in trend"),
    "Last Quarter":   ("NEUTRAL",  1, "Caution zone — market often retests lows near Last Quarter"),
}

MOON_EMOJI = {
    "New Moon":         "🌑",
    "Waxing Crescent":  "🌒",
    "First Quarter":    "🌓",
    "Waxing Gibbous":   "🌔",
    "Full Moon":        "🌕",
    "Waning Gibbous":   "🌖",
    "Last Quarter":     "🌗",
    "Waning Crescent":  "🌘",
}


def _moon_phase_name(illumination: float, phase_pct: float) -> str:
    if illumination < 3 or phase_pct < 1.5 or phase_pct > 98.5:
        return "New Moon"
    elif phase_pct < 24:
        return "Waxing Crescent"
    elif phase_pct < 28:
        return "First Quarter"
    elif phase_pct < 48:
        return "Waxing Gibbous"
    elif illumination > 97:
        return "Full Moon"
    elif phase_pct < 74:
        return "Waning Gibbous"
    elif phase_pct < 78:
        return "Last Quarter"
    else:
        return "Waning Crescent"


def get_current_moon(dt: datetime = None) -> dict:
    dt = dt or datetime.utcnow()
    ed = _ed(dt)
    moon = ephem.Moon(ed)
    illumination = float(moon.phase)

    prev_new  = ephem.previous_new_moon(ed)
    next_new  = ephem.next_new_moon(ed)
    next_full = ephem.next_full_moon(ed)
    next_fq   = ephem.next_first_quarter_moon(ed)
    next_lq   = ephem.next_last_quarter_moon(ed)

    lunar_cycle = 29.53059
    days_since_new = float(ed - prev_new)
    phase_pct = (days_since_new / lunar_cycle) * 100

    phase = _moon_phase_name(illumination, phase_pct)
    emoji = MOON_EMOJI.get(phase, "🌙")

    next_events = sorted([
        ("New Moon",      float(next_new  - ed)),
        ("Full Moon",     float(next_full - ed)),
        ("First Quarter", float(next_fq   - ed)),
        ("Last Quarter",  float(next_lq   - ed)),
    ], key=lambda x: x[1])

    return {
        "phase":          phase,
        "emoji":          emoji,
        "illumination":   round(illumination, 1),
        "days_since_new": round(days_since_new, 1),
        "phase_pct":      round(phase_pct, 1),
        "next_events":    next_events,   # [(name, days_away), ...]
        "next_new_dt":    _dt(next_new).date(),
        "next_full_dt":   _dt(next_full).date(),
    }


def get_moon_event_dates(months_back: int = 12) -> pd.DataFrame:
    """All new & full moon dates over the past N months."""
    dt = datetime.utcnow()
    end = _ed(dt)
    start = _ed(dt - timedelta(days=months_back * 30.5))

    events = []
    cursor = start
    while cursor < end:
        for func, name, bias in [
            (ephem.next_new_moon,  "New Moon",  "BULLISH"),
            (ephem.next_full_moon, "Full Moon", "BEARISH"),
        ]:
            evt = func(cursor)
            if evt < end:
                events.append({
                    "Date":  _dt(evt).date(),
                    "Event": name,
                    "Bias":  bias,
                })
        cursor = ephem.next_new_moon(cursor) + 1

    df = (pd.DataFrame(events)
          .drop_duplicates("Date")
          .sort_values("Date")
          .reset_index(drop=True))
    return df


def analyze_moon_returns(price_df: pd.DataFrame,
                         moon_events: pd.DataFrame,
                         fwd_days: int = 5) -> pd.DataFrame:
    """N-day forward returns after each new/full moon event."""
    if price_df is None or price_df.empty or moon_events.empty:
        return pd.DataFrame()

    pdf = price_df.copy()
    pdf.index = pd.to_datetime(pdf.index)
    if hasattr(pdf.index, "tz") and pdf.index.tz is not None:
        pdf.index = pdf.index.tz_localize(None)

    rows = []
    for _, row in moon_events.iterrows():
        evt_date = pd.Timestamp(row["Date"])
        future = pdf.index[pdf.index >= evt_date]
        if len(future) < fwd_days + 1:
            continue
        entry_idx   = future[0]
        exit_idx    = future[min(fwd_days, len(future) - 1)]
        entry_price = float(pdf.loc[entry_idx,  "Close"])
        exit_price  = float(pdf.loc[exit_idx,   "Close"])
        ret = (exit_price / entry_price - 1) * 100
        bias = row["Bias"]
        correct = ("✅" if (bias == "BULLISH" and ret > 0) or
                           (bias == "BEARISH" and ret < 0) else "❌")
        rows.append({
            "Event Date":      row["Date"],
            "Event":           row["Event"],
            "Bias Expected":   bias,
            "Entry":           round(entry_price, 2),
            "Exit":            round(exit_price,  2),
            f"{fwd_days}d Ret%": round(ret, 2),
            "Correct?":        correct,
        })

    return pd.DataFrame(rows)


# ─── Planetary Retrograde ─────────────────────────────────────────────────────

_PLANET_CLS = {
    "Mercury": ephem.Mercury,
    "Venus":   ephem.Venus,
    "Mars":    ephem.Mars,
    "Jupiter": ephem.Jupiter,
    "Saturn":  ephem.Saturn,
    "Uranus":  ephem.Uranus,
    "Neptune": ephem.Neptune,
}

_RETRO_META = {
    "Mercury": ("HIGH",   "☿",  "Communication chaos, data errors, false breakouts — reduce position size"),
    "Venus":   ("MEDIUM", "♀",  "Commodity/sentiment reversal — Gold, luxury & consumer sectors at risk"),
    "Mars":    ("MEDIUM", "♂",  "Aggressive moves stall — energy & defense pullback likely"),
    "Jupiter": ("LOW",    "♃",  "Expansion themes pause — growth/tech may lag"),
    "Saturn":  ("LOW",    "♄",  "Structural caution — financials & real assets under pressure"),
    "Uranus":  ("LOW",    "♅",  "Tech disruption narrative reverses — crypto/innovation caution"),
    "Neptune": ("LOW",    "♆",  "Speculative excess unwinds — biotech/crypto sentiment extreme"),
}


def _geocentric_ecl_lon(cls, date_ephem) -> float:
    """Geocentric ecliptic longitude (tropical) — correct for retrograde detection."""
    obj = cls()
    obj.compute(date_ephem)
    ecl = ephem.Ecliptic(obj, epoch=date_ephem)
    return math.degrees(float(ecl.lon)) % 360


def _is_retrograde(cls, dt: datetime, delta: int = 3) -> bool:
    """True if planet is moving backward in geocentric ecliptic longitude."""
    ed1 = _ed(dt - timedelta(days=delta))
    ed2 = _ed(dt + timedelta(days=delta))
    lon1 = _geocentric_ecl_lon(cls, ed1)
    lon2 = _geocentric_ecl_lon(cls, ed2)
    diff = lon2 - lon1
    if diff > 180:  diff -= 360
    if diff < -180: diff += 360
    return diff < 0


def get_retrograde_status(dt: datetime = None) -> List[dict]:
    dt = dt or datetime.utcnow()
    out = []
    for name, cls in _PLANET_CLS.items():
        retro = _is_retrograde(cls, dt)
        impact, sym, desc = _RETRO_META[name]
        out.append({
            "Planet":      f"{sym} {name}",
            "Status":      "☿ Retrograde" if retro else "Direct ✓",
            "Is_Retro":    retro,
            "Impact":      impact if retro else "—",
            "Description": desc  if retro else "",
        })
    return out


# ─── Planetary Aspects ────────────────────────────────────────────────────────

ASPECT_DEFS = {              # (exact_angle, max_orb_degrees)
    "Conjunction": (0,   8),
    "Sextile":     (60,  6),
    "Square":      (90,  8),
    "Trine":       (120, 8),
    "Opposition":  (180, 8),
}
ASPECT_SYMBOLS = {
    "Conjunction": "☌", "Sextile": "⚹", "Square": "□",
    "Trine": "△", "Opposition": "☍",
}
ASPECT_BIAS = {
    "Conjunction": "NEUTRAL",
    "Sextile":     "BULLISH",
    "Square":      "BEARISH",
    "Trine":       "BULLISH",
    "Opposition":  "BEARISH",
}

# Key pairs with market significance (Meridian / Pesavento research)
KEY_PAIRS = [
    ("Jupiter", "Saturn"),   # Great Conjunction — major economic cycle
    ("Saturn",  "Uranus"),   # Systemic disruption (2008, 2020)
    ("Jupiter", "Uranus"),   # Surprise bull runs
    ("Mars",    "Saturn"),   # Sharp sell-off signal
    ("Venus",   "Mars"),     # Sentiment extreme
    ("Sun",     "Saturn"),   # Risk-off
    ("Jupiter", "Neptune"),  # Speculative bubble risk
    ("Mars",    "Uranus"),   # Sudden violent moves
    ("Sun",     "Jupiter"),  # Optimism / rally day
    ("Venus",   "Jupiter"),  # Expansion, M&A enthusiasm
]

_ALL_BODIES = {"Sun": ephem.Sun, **_PLANET_CLS}

_ASPECT_DESCS = {
    ("Jupiter", "Saturn", "Conjunction"):  "Great Conjunction — new economic era, major reset",
    ("Jupiter", "Saturn", "Square"):       "Jupiter□Saturn — growth vs contraction, sector rotation",
    ("Jupiter", "Saturn", "Opposition"):   "Jupiter☍Saturn — expansion peaks, correction risk",
    ("Jupiter", "Saturn", "Trine"):        "Jupiter△Saturn — stable growth, bull market confirmation",
    ("Saturn",  "Uranus", "Square"):       "Saturn□Uranus — systemic disruption (corr. 2008, 2020 crashes)",
    ("Saturn",  "Uranus", "Opposition"):   "Saturn☍Uranus — old structure vs disruption, extreme volatility",
    ("Saturn",  "Uranus", "Conjunction"):  "Saturn☌Uranus — paradigm shift, tech/finance restructuring",
    ("Jupiter", "Uranus", "Conjunction"):  "Jupiter☌Uranus — surprise bull run, technology breakout",
    ("Jupiter", "Uranus", "Trine"):        "Jupiter△Uranus — innovation-led rally",
    ("Jupiter", "Uranus", "Opposition"):   "Jupiter☍Uranus — speculative excess, bubble risk",
    ("Mars",    "Saturn", "Square"):       "Mars□Saturn — sharp sell-off; energy sector weakness",
    ("Mars",    "Saturn", "Opposition"):   "Mars☍Saturn — sudden drops; conflict between action & restriction",
    ("Mars",    "Saturn", "Conjunction"):  "Mars☌Saturn — frustrated energy; bearish momentum builds",
    ("Sun",     "Jupiter","Conjunction"):  "Sun☌Jupiter — optimism spike; rally day, watch for reversal after",
    ("Venus",   "Jupiter","Conjunction"):  "Venus☌Jupiter — euphoria peak; M&A/expansion enthusiasm",
}


def get_planetary_aspects(dt: datetime = None) -> List[dict]:
    dt = dt or datetime.utcnow()
    ed = _ed(dt)

    lons = {}
    for name, cls in _ALL_BODIES.items():
        p = cls(ed)
        lons[name] = math.degrees(float(p.hlong)) % 360

    found = []
    for p1, p2 in KEY_PAIRS:
        if p1 not in lons or p2 not in lons:
            continue
        angle = abs(lons[p1] - lons[p2]) % 360
        if angle > 180:
            angle = 360 - angle
        for asp_name, (exact, orb) in ASPECT_DEFS.items():
            if abs(angle - exact) <= orb:
                sym  = ASPECT_SYMBOLS[asp_name]
                bias = ASPECT_BIAS[asp_name]
                desc = _ASPECT_DESCS.get((p1, p2, asp_name),
                       f"{p1} {sym} {p2}: watch for market inflection")
                found.append({
                    "Aspect":      f"{p1} {sym} {p2}",
                    "Type":        asp_name,
                    "Orb (°)":     round(abs(angle - exact), 1),
                    "Bias":        bias,
                    "Description": desc,
                })
                break
    return found


# ─── Upcoming Events Calendar ─────────────────────────────────────────────────

def get_upcoming_events(days: int = 45, dt: datetime = None) -> pd.DataFrame:
    """Forward calendar of astrological market events."""
    dt = dt or datetime.utcnow()
    ed = _ed(dt)
    events = []

    # ── Moon phases ──────────────────────────────────────────────────────────
    cursor = ed
    for _ in range(8):
        for func, name in [
            (ephem.next_new_moon,            "New Moon"),
            (ephem.next_full_moon,           "Full Moon"),
            (ephem.next_first_quarter_moon,  "First Quarter"),
            (ephem.next_last_quarter_moon,   "Last Quarter"),
        ]:
            evt = func(cursor)
            days_away = float(evt - ed)
            if 0 < days_away <= days:
                bias, strength, desc = MOON_SIGNAL_MAP.get(name, ("NEUTRAL", 1, ""))
                emoji = MOON_EMOJI.get(name, "🌙")
                events.append({
                    "Date":        _dt(evt).date(),
                    "Days Away":   round(days_away, 1),
                    "Event":       f"{emoji} {name}",
                    "Type":        "Lunar",
                    "Strength":    "★" * strength,
                    "Bias":        bias,
                    "Description": desc,
                })
        cursor = ephem.next_new_moon(cursor) + 1
        if float(cursor - ed) > days:
            break

    # ── Retrograde stations (Mercury, Venus, Mars) ────────────────────────────
    for name, cls in [("Mercury", ephem.Mercury), ("Venus", ephem.Venus), ("Mars", ephem.Mars)]:
        impact, sym, desc = _RETRO_META[name]
        prev_retro = _is_retrograde(cls, dt - timedelta(days=1))
        for offset in range(1, days + 1):
            cur_retro = _is_retrograde(cls, dt + timedelta(days=offset))
            if cur_retro != prev_retro:
                station = "Retrograde Begins" if cur_retro else "Goes Direct"
                bias     = "BEARISH" if cur_retro else "BULLISH"
                strength = 3 if name == "Mercury" else 2
                events.append({
                    "Date":        (dt + timedelta(days=offset)).date(),
                    "Days Away":   offset,
                    "Event":       f"{sym} {name} {station}",
                    "Type":        "Retrograde",
                    "Strength":    "★" * strength,
                    "Bias":        bias,
                    "Description": desc,
                })
                break
            prev_retro = cur_retro

    # ── Solar ingress into cardinal signs (Gann seasonal turns) ──────────────
    CARDINAL = {0: "Aries ♈", 90: "Cancer ♋", 180: "Libra ♎", 270: "Capricorn ♑"}
    sun_obj = ephem.Sun()
    for offset in range(1, days + 1):
        cur  = dt + timedelta(days=offset)
        prev = cur - timedelta(days=1)
        sun_obj.compute(_ed(cur))
        lon_now  = math.degrees(float(sun_obj.hlong)) % 360
        sun_obj.compute(_ed(prev))
        lon_prev = math.degrees(float(sun_obj.hlong)) % 360
        for deg, sign_name in CARDINAL.items():
            crossed = (lon_prev % 360 < deg <= lon_now % 360) or \
                      (deg == 0 and lon_prev > 355 and lon_now < 5)
            if crossed:
                events.append({
                    "Date":        cur.date(),
                    "Days Away":   offset,
                    "Event":       f"☀️ Sun → {sign_name}",
                    "Type":        "Solar Ingress",
                    "Strength":    "★★",
                    "Bias":        "NEUTRAL",
                    "Description": f"Gann cardinal turn: trend change signal within 1-3 trading days of ingress",
                })

    df = pd.DataFrame(events)
    if df.empty:
        return df
    df = (df.sort_values("Days Away")
            .drop_duplicates(subset=["Date", "Event"])
            .reset_index(drop=True))
    return df


# ─── Market Bias Score ────────────────────────────────────────────────────────

def compute_market_bias(dt: datetime = None) -> dict:
    """Aggregate all astrological signals into a single market bias."""
    dt = dt or datetime.utcnow()
    score = 0.0
    signals = []

    # Moon phase
    moon = get_current_moon(dt)
    phase = moon["phase"]
    if "New Moon" in phase:
        score += 3
        signals.append(("🌑 New Moon", "+3", "BULLISH", "Bank of Scotland buy signal"))
    elif "Full Moon" in phase:
        score -= 3
        signals.append(("🌕 Full Moon", "−3", "BEARISH", "Bank of Scotland sell signal"))
    elif "Waxing" in phase:
        score += 1
        signals.append((f"{moon['emoji']} {phase}", "+1", "BULLISH", "Waxing moon — gentle tailwind"))
    elif "Waning" in phase:
        score -= 1
        signals.append((f"{moon['emoji']} {phase}", "−1", "BEARISH", "Waning moon — gentle headwind"))

    # Approaching key phase within 2 days
    for evt_name, days_away in moon["next_events"][:2]:
        if 0 < days_away <= 2:
            if evt_name == "New Moon":
                score += 2
                signals.append((f"🌑 New Moon in {days_away:.1f}d", "+2", "BULLISH", "Pre-New Moon accumulation"))
            elif evt_name == "Full Moon":
                score -= 2
                signals.append((f"🌕 Full Moon in {days_away:.1f}d", "−2", "BEARISH", "Pre-Full Moon distribution"))

    # Retrograde planets
    retros = get_retrograde_status(dt)
    for r in retros:
        if not r["Is_Retro"]:
            continue
        planet = r["Planet"].split()[-1]
        if planet == "Mercury":
            score -= 2
            signals.append(("☿ Mercury Rx", "−2", "BEARISH", "Volatility / false breakouts"))
        elif planet in ("Venus", "Mars"):
            score -= 1
            signals.append((r["Planet"] + " Rx", "−1", "BEARISH", "Sentiment caution"))
        else:
            score -= 0.5
            signals.append((r["Planet"] + " Rx", "−0.5", "BEARISH", "Outer planet caution"))

    # Planetary aspects
    aspects = get_planetary_aspects(dt)
    for asp in aspects:
        if asp["Bias"] == "BEARISH":
            score -= 2
            signals.append((asp["Aspect"], "−2", "BEARISH", asp["Description"]))
        elif asp["Bias"] == "BULLISH":
            score += 2
            signals.append((asp["Aspect"], "+2", "BULLISH", asp["Description"]))

    score = max(-10.0, min(10.0, score))

    if score >= 3:
        overall, color, emoji = "BULLISH", "#26a69a", "📈"
    elif score <= -3:
        overall, color, emoji = "BEARISH", "#ef5350", "📉"
    else:
        overall, color, emoji = "NEUTRAL",  "#ffb74d", "➡️"

    return {
        "score":       round(score, 1),
        "overall":     overall,
        "color":       color,
        "emoji":       emoji,
        "signals":     signals,   # [(label, delta, bias, desc), ...]
        "moon":        moon,
        "retrogrades": retros,
        "aspects":     aspects,
    }


# ─── Gann Time Cycles ─────────────────────────────────────────────────────────

GANN_CYCLES = [45, 90, 144, 180, 270, 360]


def find_gann_cycle_dates(price_df: pd.DataFrame, lookahead_days: int = 90) -> pd.DataFrame:
    """Project Gann time cycles from recent 252-bar pivot high and low."""
    if price_df is None or len(price_df) < 20:
        return pd.DataFrame()

    recent = price_df.tail(252)
    hi_idx = recent["High"].idxmax()
    lo_idx = recent["Low"].idxmin()

    hi_date  = pd.Timestamp(hi_idx).to_pydatetime().date()
    lo_date  = pd.Timestamp(lo_idx).to_pydatetime().date()
    hi_price = float(recent.loc[hi_idx, "High"])
    lo_price = float(recent.loc[lo_idx, "Low"])

    today = date.today()
    rows  = []
    for ptype, pdate, pprice in [("52W High", hi_date, hi_price),
                                  ("52W Low",  lo_date, lo_price)]:
        for cycle in GANN_CYCLES:
            target = pdate + timedelta(days=cycle)
            away   = (target - today).days
            if -5 <= away <= lookahead_days:
                rows.append({
                    "Gann Date":   target,
                    "Days Away":   away,
                    "Pivot Type":  ptype,
                    "Pivot Date":  pdate,
                    "Pivot Price": round(pprice, 2),
                    "Cycle":       f"{cycle}d",
                    "Status":      ("⚡ TODAY" if away == 0
                                   else f"Past {abs(away)}d" if away < 0
                                   else f"In {away}d"),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Days Away").reset_index(drop=True)
    return df


# ─── Top-level scanner (returns upcoming events DataFrame) ───────────────────

def run_astro_scanner(days_forward: int = 45) -> pd.DataFrame:
    """Main entry point — returns upcoming astrological events for next N days."""
    return get_upcoming_events(days=days_forward)


# ═══════════════════════════════════════════════════════════════════════════════
# VEDIC ASTROLOGY — BHADRA / VISHTI KARANA
# ═══════════════════════════════════════════════════════════════════════════════
#
# Karana = half a Tithi = 6° of Moon–Sun angular separation
# The 60 Karanas of the lunar month:
#   0          → Kimstughna (fixed)
#   1–56       → 8 cycles of 7 movable Karanas:
#                Bava, Balava, Kaulava, Taitila, Garija, Vanija, Vishti/Bhadra
#   57,58,59   → Shakuni, Chatushpada, Naga (fixed)
#
# Vishti/Bhadra: index 6 in each 7-karana cycle → karana_seq {7,14,21,28,35,42,49,56}
# Duration: ~10.9 hours (6° ÷ 0.549°/hr relative Moon motion)
# Nature:    Inauspicious — obstacles, delays, reversals
# Vedic traders avoid starting new positions during Vishti
# ═══════════════════════════════════════════════════════════════════════════════

import pytz as _pytz

_ET = _pytz.timezone("America/New_York")

_TITHI_NAMES = [
    "", "Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami",
    "Shashthi", "Saptami", "Ashtami", "Navami", "Dashami",
    "Ekadashi", "Dwadashi", "Trayodashi", "Chaturdashi", "Purnima/Amavasya",
]
_MOVABLE_KARANAS = ["Bava", "Balava", "Kaulava", "Taitila", "Garija", "Vanija", "Vishti (Bhadra)"]
_FIXED_KARANAS   = {0: "Kimstughna", 57: "Shakuni", 58: "Chatushpada", 59: "Naga"}

# Market sentiment by Karana (Vedic tradition)
_KARANA_SENTIMENT = {
    "Bava":            "BULLISH",
    "Balava":          "BULLISH",
    "Kaulava":         "BULLISH",
    "Taitila":         "NEUTRAL",
    "Garija":          "NEUTRAL",
    "Vanija":          "BULLISH",   # Vanija = merchant → trade favorable
    "Vishti (Bhadra)": "BEARISH",   # Inauspicious — the focus of our backtest
    "Kimstughna":      "NEUTRAL",
    "Shakuni":         "BEARISH",   # Also inauspicious
    "Chatushpada":     "NEUTRAL",
    "Naga":            "BEARISH",
}


def _moon_sun_diff(dt: datetime) -> float:
    """Moon − Sun ecliptic longitude (0°–360°)."""
    ed = _ed(dt)
    sun_lon  = math.degrees(float(ephem.Sun(ed).hlong))  % 360
    moon_lon = math.degrees(float(ephem.Moon(ed).hlong)) % 360
    return (moon_lon - sun_lon) % 360


def get_karana(dt: datetime = None) -> dict:
    """Full Karana (Panchang element) for the given UTC datetime."""
    dt   = dt or datetime.utcnow()
    diff = _moon_sun_diff(dt)
    k    = int(diff / 6)          # 0–59

    if k in _FIXED_KARANAS:
        name      = _FIXED_KARANAS[k]
        is_vishti = False
    elif 1 <= k <= 56:
        idx       = (k - 1) % 7
        name      = _MOVABLE_KARANAS[idx]
        is_vishti = (idx == 6)
    else:
        name, is_vishti = "Unknown", False

    # Degrees remaining in this karana → hours until next karana
    deg_remain     = 6.0 - (diff % 6.0)
    hours_remain   = deg_remain / 0.5490          # Moon moves ~0.549°/hr vs Sun

    # Tithi info
    tithi_seq      = int(diff / 12) + 1           # 1–30
    paksha         = "Shukla" if tithi_seq <= 15 else "Krishna"
    tithi_idx      = tithi_seq if tithi_seq <= 15 else tithi_seq - 15
    tithi_name     = _TITHI_NAMES[min(tithi_idx, 15)]

    sentiment      = _KARANA_SENTIMENT.get(name, "NEUTRAL")

    return {
        "karana":          name,
        "karana_seq":      k,
        "is_vishti":       is_vishti,
        "sentiment":       sentiment,
        "hours_remaining": round(hours_remain, 1),
        "tithi":           f"{paksha} {tithi_name}",
        "moon_sun_diff":   round(diff, 2),
    }


def is_vishti(dt: datetime) -> bool:
    """True if Vishti/Bhadra karana is active at the given UTC datetime."""
    diff = _moon_sun_diff(dt)
    k    = int(diff / 6)
    if k == 0 or k >= 57:
        return False
    return (k - 1) % 7 == 6


def _market_open_utc(d: date) -> datetime:
    naive = datetime(d.year, d.month, d.day, 9, 30)
    return _ET.localize(naive).astimezone(_pytz.utc).replace(tzinfo=None)

def _market_mid_utc(d: date) -> datetime:
    naive = datetime(d.year, d.month, d.day, 12, 30)
    return _ET.localize(naive).astimezone(_pytz.utc).replace(tzinfo=None)

def _market_close_utc(d: date) -> datetime:
    naive = datetime(d.year, d.month, d.day, 15, 45)
    return _ET.localize(naive).astimezone(_pytz.utc).replace(tzinfo=None)


def backtest_vishti(price_df: pd.DataFrame,
                    check_mode: str = "any") -> dict:
    """
    Backtest the Vishti/Bhadra effect on daily US market returns.

    check_mode
      "open"  — Bhadra must be active at market OPEN (9:30 AM ET)
      "any"   — Bhadra active at OPEN or MIDDAY or CLOSE (broader definition)
      "all"   — Bhadra must cover the ENTIRE session (strictest)

    Returns a dict with summary stats + per-day DataFrame.
    """
    from scipy import stats as _stats

    if price_df is None or len(price_df) < 20:
        return {}

    pdf = price_df.copy()
    pdf.index = pd.to_datetime(pdf.index)
    if hasattr(pdf.index, "tz") and pdf.index.tz is not None:
        pdf.index = pdf.index.tz_localize(None)

    pdf["ret"] = pdf["Close"].pct_change() * 100
    pdf = pdf.dropna(subset=["ret"])

    flags = []
    for idx in pdf.index:
        d = idx.date()
        try:
            t_open  = _market_open_utc(d)
            t_mid   = _market_mid_utc(d)
            t_close = _market_close_utc(d)
        except Exception:
            flags.append(False)
            continue

        o = is_vishti(t_open)
        m = is_vishti(t_mid)
        c = is_vishti(t_close)

        if check_mode == "open":
            active = o
        elif check_mode == "all":
            active = o and m and c
        else:   # "any" — default
            active = o or m or c

        flags.append(active)

    pdf["bhadra"] = flags

    b_df  = pdf[pdf["bhadra"]]
    nb_df = pdf[~pdf["bhadra"]]

    if len(b_df) < 5 or len(nb_df) < 5:
        return {}

    t_stat, p_val = _stats.ttest_ind(b_df["ret"].dropna(),
                                     nb_df["ret"].dropna(),
                                     equal_var=False)

    # Median test too
    _, p_mann = _stats.mannwhitneyu(b_df["ret"].dropna(),
                                    nb_df["ret"].dropna(),
                                    alternative="two-sided")

    # Cumulative performance
    pdf["cum_all"]      = (1 + pdf["ret"] / 100).cumprod()
    pdf["cum_nobhadra"] = np.where(pdf["bhadra"], np.nan,
                                   (1 + pdf["ret"] / 100))
    pdf["cum_nobhadra"] = pdf["cum_nobhadra"].fillna(1.0).cumprod()

    # Monthly aggregation
    pdf["ym"] = pdf.index.to_period("M")
    monthly = (pdf.groupby(["ym", "bhadra"])["ret"]
               .mean()
               .unstack(fill_value=np.nan)
               .rename(columns={True: "Bhadra Avg %", False: "Normal Avg %"}))

    return {
        "total_days":        len(pdf),
        "bhadra_days":       len(b_df),
        "normal_days":       len(nb_df),
        "bhadra_pct_time":   round(len(b_df) / len(pdf) * 100, 1),
        "bhadra_mean":       round(b_df["ret"].mean(), 4),
        "normal_mean":       round(nb_df["ret"].mean(), 4),
        "bhadra_median":     round(b_df["ret"].median(), 4),
        "normal_median":     round(nb_df["ret"].median(), 4),
        "bhadra_std":        round(b_df["ret"].std(), 4),
        "normal_std":        round(nb_df["ret"].std(), 4),
        "bhadra_win_rate":   round((b_df["ret"] > 0).mean() * 100, 1),
        "normal_win_rate":   round((nb_df["ret"] > 0).mean() * 100, 1),
        "bhadra_max_loss":   round(b_df["ret"].min(), 2),
        "normal_max_loss":   round(nb_df["ret"].min(), 2),
        "t_stat":            round(t_stat, 3),
        "p_val":             round(p_val, 4),
        "p_mann":            round(p_mann, 4),
        "significant":       p_val < 0.10,
        "detail_df":         pdf[["Close", "ret", "bhadra"]].copy(),
        "monthly_df":        monthly,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SULABH JAIN / VEDIC PREDICTION ENGINE
# Sources: Chariot Palmistry, Rajeev Prakash, Bill Meridian
# ═══════════════════════════════════════════════════════════════════════════════

# Lahiri ayanamsa (degrees to subtract from tropical to get Vedic sidereal)
# Accurate to ±0.1° for 2020-2030
_LAHIRI_AYANAMSA = 23.85

_NAKSHATRA_NAMES = [
    "Ashwini","Bharani","Krittika","Rohini","Mrigashira","Ardra",
    "Punarvasu","Pushya","Ashlesha","Magha","Purva Phalguni","Uttara Phalguni",
    "Hasta","Chitra","Swati","Vishakha","Anuradha","Jyeshtha",
    "Mula","Purva Ashadha","Uttara Ashadha","Shravana","Dhanishtha",
    "Shatabhisha","Purva Bhadrapada","Uttara Bhadrapada","Revati",
]

# Market sentiment per Nakshatra (Vedic financial astrology tradition)
_NAKSHATRA_MARKET = {
    "Ashwini":           ("BULLISH",  "New beginnings, fast moves — initiating energy"),
    "Bharani":           ("BEARISH",  "Holding/restriction energy — caution near highs"),
    "Krittika":          ("BEARISH",  "Sharp cutting energy — sudden reversals possible"),
    "Rohini":            ("BULLISH",  "Growth, abundance — favorable for commodity longs"),
    "Mrigashira":        ("NEUTRAL",  "Searching energy — choppy, range-bound likely"),
    "Ardra":             ("BEARISH",  "Rahu-ruled storm nakshatra — high volatility, sell-offs"),
    "Punarvasu":         ("BULLISH",  "Return/renewal — recoveries and rebounds"),
    "Pushya":            ("BULLISH",  "Most auspicious — strong trend days, Saturn's best"),
    "Ashlesha":          ("BEARISH",  "Mercury's serpent — deception, false breakouts"),
    "Magha":             ("NEUTRAL",  "Ketu-ruled ancestors — flat/institutional churning"),
    "Purva Phalguni":    ("BEARISH",  "Venus excess — euphoria top signal for growth stocks"),
    "Uttara Phalguni":   ("BULLISH",  "Sun-ruled order — institutional accumulation"),
    "Hasta":             ("BULLISH",  "Moon-ruled skill — precision moves, crafted rallies"),
    "Chitra":            ("NEUTRAL",  "Mars-ruled artistry — sector rotation, mixed signals"),
    "Swati":             ("NEUTRAL",  "Rahu-ruled independence — spreads widen, options active"),
    "Vishakha":          ("BULLISH",  "Jupiter goal-oriented — breakout confirmation days"),
    "Anuradha":          ("BULLISH",  "Saturn friendship — steady accumulation, support holds"),
    "Jyeshtha":          ("BEARISH",  "Mercury-ruled elder — pride/hubris reversals, sell arrogance"),
    "Mula":              ("BEARISH",  "Ketu roots destroyed — crashes, margin calls, restructuring"),
    "Purva Ashadha":     ("BEARISH",  "Venus excess again — commodity bubble warning"),
    "Uttara Ashadha":    ("BULLISH",  "Sun victory — trend continuation, breakout confirmation"),
    "Shravana":          ("BULLISH",  "Moon listening — accumulation on news, buy-the-dip"),
    "Dhanishtha":        ("NEUTRAL",  "Mars-Saturn richness — commodities mixed, silver volatile"),
    "Shatabhisha":       ("BEARISH",  "Rahu healing — hidden risks emerge, gap-downs"),
    "Purva Bhadrapada":  ("BEARISH",  "Jupiter-Mars fire — panic selling, impulsive drops"),
    "Uttara Bhadrapada": ("NEUTRAL",  "Saturn depth — slow consolidation, patient accumulation"),
    "Revati":            ("BULLISH",  "Mercury journey complete — trend exhaustion turns bullish"),
}

# Vedic sign names and financial character
_SIGN_NAMES = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces",
]
_SIGN_MARKET = {
    "Aries":       "BULLISH",  # Mars-ruled, initiative
    "Taurus":      "BULLISH",  # Venus-ruled, stable growth
    "Gemini":      "NEUTRAL",  # Mercury-ruled, volatile
    "Cancer":      "BEARISH",  # Moon-ruled, sentiment swings
    "Leo":         "BULLISH",  # Sun-ruled, confidence
    "Virgo":       "BEARISH",  # Mercury-ruled, critical/corrective
    "Libra":       "BULLISH",  # Venus-ruled, balance/recovery
    "Scorpio":     "BEARISH",  # Mars/Ketu-ruled, hidden risks
    "Sagittarius": "BULLISH",  # Jupiter-ruled, optimism
    "Capricorn":   "BEARISH",  # Saturn-ruled, contraction
    "Aquarius":    "NEUTRAL",  # Saturn/Rahu-ruled, disruption
    "Pisces":      "NEUTRAL",  # Jupiter-ruled, speculation
}

# Rahu transit through signs — market themes (Sulabh Jain / Rajeev Prakash)
_RAHU_SIGN_THEME = {
    "Aries":       ("BULLISH",  "Rahu in Aries — speculation in defense/commodities, aggressive momentum"),
    "Taurus":      ("BULLISH",  "Rahu in Taurus — gold/silver/crypto bubble building, asset inflation"),
    "Gemini":      ("NEUTRAL",  "Rahu in Gemini — tech/communication volatility, AI speculation"),
    "Cancer":      ("BEARISH",  "Rahu in Cancer — real estate anxiety, USD instability"),
    "Leo":         ("BEARISH",  "Rahu in Leo — political risk premium, gold safe-haven bid"),
    "Virgo":       ("NEUTRAL",  "Rahu in Virgo — healthcare/pharma speculation, debt concerns"),
    "Libra":       ("BULLISH",  "Rahu in Libra — finance/luxury speculation, M&A frenzy"),
    "Scorpio":     ("BEARISH",  "Rahu in Scorpio — hidden leverage exposed, derivatives risk"),
    "Sagittarius": ("BULLISH",  "Rahu in Sagittarius — global markets boom, emerging market flows"),
    "Capricorn":   ("BEARISH",  "Rahu in Capricorn — systemic risk (2019-2022 era), govt/credit crises"),
    "Aquarius":    ("NEUTRAL",  "Rahu in Aquarius — crypto/tech disruption wave, decentralisation"),
    "Pisces":      ("BEARISH",  "Rahu in Pisces — speculative excess in oil/energy, delusion peaks"),
}


def _tropical_to_sidereal(lon: float) -> float:
    return (lon - _LAHIRI_AYANAMSA) % 360


def _lon_to_sign(lon: float) -> str:
    return _SIGN_NAMES[int(lon / 30) % 12]


def _lon_to_nakshatra(lon: float) -> str:
    return _NAKSHATRA_NAMES[int(lon / (360 / 27)) % 27]


def _mean_lunar_node(dt: datetime) -> float:
    """
    Mean longitude of Moon's ascending node (Rahu) in degrees, tropical.
    Formula: Jean Meeus, Astronomical Algorithms 2nd ed., Ch 47.
    """
    jd = ephem.julian_date(_ed(dt))
    T  = (jd - 2451545.0) / 36525.0   # Julian centuries from J2000.0
    omega = (125.0445479
             - 1934.1362608 * T
             + 0.0020754    * T * T
             + T**3 / 467441.0
             - T**4 / 60616000.0)
    return omega % 360


def get_rahu_ketu(dt: datetime = None) -> dict:
    """
    Compute Rahu (North Node) and Ketu (South Node) tropical and sidereal positions.
    Rahu = Moon's mean ascending node (Meeus formula). Ketu = Rahu + 180°.
    """
    dt = dt or datetime.utcnow()
    rahu_tropical = _mean_lunar_node(dt)
    ketu_tropical = (rahu_tropical + 180) % 360

    rahu_sid = _tropical_to_sidereal(rahu_tropical)
    ketu_sid = _tropical_to_sidereal(ketu_tropical)

    rahu_sign = _lon_to_sign(rahu_sid)
    ketu_sign = _lon_to_sign(ketu_sid)

    rahu_bias, rahu_theme = _RAHU_SIGN_THEME.get(rahu_sign, ("NEUTRAL", "Rahu transiting " + rahu_sign))

    return {
        "rahu_sign":    rahu_sign,
        "ketu_sign":    ketu_sign,
        "rahu_lon_sid": round(rahu_sid, 2),
        "ketu_lon_sid": round(ketu_sid, 2),
        "rahu_bias":    rahu_bias,
        "rahu_theme":   rahu_theme,
    }


def get_moon_nakshatra(dt: datetime = None) -> dict:
    """Return Moon's current Nakshatra (Vedic) and its market sentiment."""
    dt = dt or datetime.utcnow()
    moon = ephem.Moon(_ed(dt))
    moon_tropical = math.degrees(float(moon.hlong)) % 360
    moon_sid = _tropical_to_sidereal(moon_tropical)
    nk = _lon_to_nakshatra(moon_sid)
    bias, desc = _NAKSHATRA_MARKET.get(nk, ("NEUTRAL", ""))
    return {
        "nakshatra": nk,
        "bias":      bias,
        "desc":      desc,
        "moon_sid":  round(moon_sid, 2),
    }


def get_vedic_planets(dt: datetime = None) -> List[dict]:
    """Return Vedic (sidereal) sign positions for all planets."""
    dt = dt or datetime.utcnow()
    ed = _ed(dt)
    planets = [
        ("Sun",     ephem.Sun),
        ("Moon",    ephem.Moon),
        ("Mercury", ephem.Mercury),
        ("Venus",   ephem.Venus),
        ("Mars",    ephem.Mars),
        ("Jupiter", ephem.Jupiter),
        ("Saturn",  ephem.Saturn),
    ]
    rows = []
    for name, cls in planets:
        p   = cls(ed)
        lon = math.degrees(float(p.hlong)) % 360
        sid = _tropical_to_sidereal(lon)
        sign = _lon_to_sign(sid)
        nk   = _lon_to_nakshatra(sid)
        rows.append({
            "Planet":     name,
            "Sign":       sign,
            "Nakshatra":  nk,
            "Lon (sid°)": round(sid, 1),
            "Bias":       _SIGN_MARKET.get(sign, "NEUTRAL"),
        })
    return rows


def compute_vedic_daily_score(dt: datetime = None) -> dict:
    """
    Aggregate Vedic signals (Rahu-Ketu, Moon Nakshatra, Karana, Bhadra)
    into a single daily market score.  Range: -10 to +10.
    """
    dt = dt or datetime.utcnow()

    score   = 0.0
    signals = []

    # 1. Rahu sign bias (slow-moving, background regime)
    rk = get_rahu_ketu(dt)
    if rk["rahu_bias"] == "BULLISH":
        score += 1.5
        signals.append(("☊ Rahu", "+1.5", "BULLISH", rk["rahu_theme"]))
    elif rk["rahu_bias"] == "BEARISH":
        score -= 1.5
        signals.append(("☊ Rahu", "−1.5", "BEARISH", rk["rahu_theme"]))

    # 2. Moon Nakshatra (changes every ~1 day — primary daily signal)
    nk = get_moon_nakshatra(dt)
    if nk["bias"] == "BULLISH":
        score += 2.5
        signals.append((f"☽ {nk['nakshatra']}", "+2.5", "BULLISH", nk["desc"]))
    elif nk["bias"] == "BEARISH":
        score -= 2.5
        signals.append((f"☽ {nk['nakshatra']}", "−2.5", "BEARISH", nk["desc"]))

    # 3. Karana (changes every ~11 hours — intraday Vedic filter)
    kar = get_karana(dt)
    if kar["karana"] == "Vishti (Bhadra)":
        score -= 3.0
        signals.append(("⚡ Vishti/Bhadra", "−3.0", "BEARISH",
                         "Inauspicious Karana — avoid new entries, reversal risk"))
    elif kar["karana"] == "Vanija":
        score += 1.0
        signals.append(("🛒 Vanija Karana", "+1.0", "BULLISH", "Merchant Karana — trade favorable"))
    elif kar["karana"] in ("Shakuni", "Naga"):
        score -= 1.5
        signals.append((f"⚠️ {kar['karana']}", "−1.5", "BEARISH", "Inauspicious fixed Karana"))

    # 4. Jupiter sign — macro expansion/contraction
    ed = _ed(dt)
    jup = ephem.Jupiter(ed)
    jup_sid_lon = _tropical_to_sidereal(math.degrees(float(jup.hlong)) % 360)
    jup_sign = _lon_to_sign(jup_sid_lon)
    if jup_sign in ("Aries", "Leo", "Sagittarius", "Cancer", "Pisces"):
        score += 1.0
        signals.append(("♃ Jupiter", "+1.0", "BULLISH", f"Jupiter in {jup_sign} — expansion/optimism"))
    elif jup_sign in ("Capricorn", "Gemini", "Virgo"):
        score -= 1.0
        signals.append(("♃ Jupiter", "−1.0", "BEARISH", f"Jupiter in {jup_sign} — contraction"))

    # 5. Saturn sign — structural caution
    sat = ephem.Saturn(ed)
    sat_sid_lon = _tropical_to_sidereal(math.degrees(float(sat.hlong)) % 360)
    sat_sign = _lon_to_sign(sat_sid_lon)
    if sat_sign in ("Capricorn", "Aquarius", "Libra"):
        score -= 1.0
        signals.append(("♄ Saturn", "−1.0", "BEARISH", f"Saturn in {sat_sign} — structural pressure"))
    elif sat_sign in ("Aries", "Leo", "Cancer"):
        score += 0.5
        signals.append(("♄ Saturn", "+0.5", "BULLISH", f"Saturn in {sat_sign} — discipline enables growth"))

    score = max(-10.0, min(10.0, score))

    if score >= 2:
        overall, color, emoji = "BULLISH", "#26a69a", "📈"
    elif score <= -2:
        overall, color, emoji = "BEARISH", "#ef5350", "📉"
    else:
        overall, color, emoji = "NEUTRAL",  "#ffb74d", "➡️"

    return {
        "score":    round(score, 1),
        "overall":  overall,
        "color":    color,
        "emoji":    emoji,
        "signals":  signals,
        "rahu_ketu": rk,
        "nakshatra": nk,
        "karana":    kar,
    }


def build_prediction_calendar(days: int = 30, dt: datetime = None) -> pd.DataFrame:
    """
    Build a forward N-day market prediction calendar combining Western + Vedic signals.
    Returns one row per day with: Date, Western Score, Vedic Score, Combined Score, Bias, Key Events.
    """
    dt = dt or datetime.utcnow()
    rows = []
    for offset in range(days):
        target_dt = dt + timedelta(days=offset)
        target_date = target_dt.date()

        # Skip weekends
        if target_date.weekday() >= 5:
            continue

        # Western bias
        try:
            west = compute_market_bias(target_dt)
            w_score = float(west["score"])
        except Exception:
            w_score = 0.0

        # Vedic bias
        try:
            ved = compute_vedic_daily_score(target_dt)
            v_score = float(ved["score"])
        except Exception:
            v_score = 0.0

        combined = round((w_score + v_score) / 2, 1)

        if combined >= 2:
            bias, emoji = "BULLISH", "📈"
        elif combined <= -2:
            bias, emoji = "BEARISH", "📉"
        else:
            bias, emoji = "NEUTRAL", "➡️"

        # Key events that day
        try:
            nk_name = get_moon_nakshatra(target_dt)["nakshatra"]
        except Exception:
            nk_name = "—"
        try:
            kar_name = get_karana(target_dt)["karana"]
        except Exception:
            kar_name = "—"

        events = []
        if kar_name == "Vishti (Bhadra)":
            events.append("⚡ Bhadra")
        if "Mercury" in kar_name or "Retrograde" in kar_name:
            events.append("☿ Rx")

        rows.append({
            "Date":          target_date,
            "Day":           target_date.strftime("%a"),
            "Western Score": round(w_score, 1),
            "Vedic Score":   round(v_score, 1),
            "Combined":      combined,
            "Bias":          f"{emoji} {bias}",
            "Moon Nakshatra": nk_name,
            "Karana":         kar_name,
            "Key Flags":      " · ".join(events) if events else "",
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# SULABH JAIN MACRO CYCLE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Derived from Sulabh Jain's (Chariot Palmistry) Vedic astrology framework:
#   - Jupiter-Saturn 20-year synodic cycle  → decade-level market regimes
#   - Jupiter sign transits (~1 yr/sign)    → annual bull/bear themes
#   - Saturn sign transits (~2.5 yr/sign)   → structural economic pressure
#   - Rahu-Ketu 18-month transits           → thematic sector rotation
#   - Eclipse windows (±15 days)            → volatility / turning points
#   - Mars-Saturn hard aspects              → conflict / correction triggers
#
# Validated predictions (cheatsheet image):
#   2020-2022 Pessimism  → Jupiter debilitated Capricorn + Saturn Capricorn
#   2023-2026 Optimism   → Jupiter Aries→Gemini bull run
#   End 2026-2027 Disaster → Saturn enters Aries + Jupiter-Saturn opposition
#   2028-2030 Global War → Jupiter debilitated Virgo + Saturn Aries conflict
# ═══════════════════════════════════════════════════════════════════════════════

from datetime import date as _date2


# ── Decade-level macro regimes (Jupiter-Saturn cycle basis) ───────────────────

_DECADE_REGIMES: list[dict] = [
    {
        "period": "2020 – 2022",
        "label": "Pessimism / Crisis",
        "short": "PESSIMISM",
        "regime": "BEARISH",
        "start": _date2(2020, 1, 1),
        "end":   _date2(2022, 12, 31),
        "description": (
            "Jupiter + Saturn both in Capricorn (debilitation) → institutions fail, "
            "hardship, systemic shocks. COVID crash, inflation surge, NASDAQ –33% (2022). "
            "Jupiter-Saturn Great Conjunction Dec 21 2020 resets the 20-yr cycle."
        ),
        "basis": "Jupiter debilitated Capricorn (Dec 2019 – Nov 2020). Saturn Capricorn (Jan 2020 – Jan 2023). "
                 "Rahu in Taurus + Ketu in Scorpio amplified financial system stress.",
        "key_events": ["COVID crash Mar 2020", "Inflation surge 2021-22", "NASDAQ –33% 2022", "Crypto crash 2022"],
        "sectors_buy":   ["Gold (safe haven)", "Healthcare", "Defence", "Commodities"],
        "sectors_avoid": ["Tech growth", "Real Estate (peak)", "Crypto (peak 2021)"],
    },
    {
        "period": "2023 – Mid 2026",
        "label": "Optimism / Boom",
        "short": "OPTIMISM",
        "regime": "BULLISH",
        "start": _date2(2023, 1, 1),
        "end":   _date2(2026, 9, 30),
        "description": (
            "Jupiter moves Aries → Taurus → Gemini → Cancer (exaltation). "
            "AI boom, SPX all-time highs, crypto supercycle, real estate boom. "
            "Rahu in Aquarius (May 2025) amplifies tech/AI/crypto themes. "
            "Money printing keeps markets elevated despite underlying tensions."
        ),
        "basis": "Jupiter Aries (Apr 2023), Taurus (May 2024), Gemini (May 2025), Cancer exaltation (May 2026). "
                 "Saturn in Aquarius then Pisces (liquidity + stimulus environment).",
        "key_events": ["ChatGPT / AI boom 2023", "SPX all-time highs 2024-25",
                       "Crypto supercycle 2024-25", "Real Estate boom 2021-26",
                       "Stock Market Bubble (Apr 2022 – 2026)"],
        "sectors_buy":   ["AI / Tech", "Crypto", "Real Estate", "Gold / Silver", "Commodities", "India equities"],
        "sectors_avoid": ["Cash (debasement)", "Bonds (inflation)"],
    },
    {
        "period": "End 2026 – 2027",
        "label": "Disaster / The Big Crash",
        "short": "DISASTER",
        "regime": "BEARISH",
        "start": _date2(2026, 9, 1),
        "end":   _date2(2027, 12, 31),
        "description": (
            "Saturn enters Aries (April 2026) — the sign of aggression, conflict, market panic. "
            "Three market corrections cascade into The Big One (Oct 2026). "
            "Real estate crash, political chaos, new virus variant. "
            "Crypto bottoms Oct-Nov 2026. Major money printing begins response."
        ),
        "basis": "Saturn in Aries (2026-2028) = war/conflict archetype. "
                 "Jupiter-Saturn opposition 2026 = peak stress. "
                 "Rahu enters Capricorn (Nov 2026) = financial reckoning axis. "
                 "Multiple eclipse windows Aug-Oct 2026 trigger turning points.",
        "key_events": ["3 stock market corrections (2026)", "Real estate crash Oct 2026",
                       "Crypto bottom Oct-Nov 2026", "Political chaos / riots Aug-Sept 2026",
                       "New virus/disease Aug-Sept 2026", "Major money printing Dec 2026"],
        "sectors_buy":   ["Defence (boom)", "Cash", "Short tech/RE"],
        "sectors_avoid": ["Real Estate (crash)", "Banks", "Consumer discretionary"],
    },
    {
        "period": "2028 – 2030",
        "label": "New Global War / Conflicts",
        "short": "GLOBAL WAR",
        "regime": "BEARISH",
        "start": _date2(2028, 1, 1),
        "end":   _date2(2030, 12, 31),
        "description": (
            "Jupiter in Virgo (debilitation 2028-29) + Saturn in Aries = peak military conflict era. "
            "Historical precedent: every Jupiter debilitation + Saturn hard aspect aligns with major wars "
            "(WWI, WWII, Gulf War, 9/11 era). New geopolitical order forms. "
            "Supertech and FinTech begin recovery from early 2027."
        ),
        "basis": "Jupiter debilitated Virgo (Jul 2028 – Aug 2029). Saturn in Aries (2026-2028). "
                 "Mars-Saturn conjunctions in fire signs amplify conflict. "
                 "Rahu in Sagittarius (2028) = religion/ideology-driven conflicts.",
        "key_events": ["New global conflicts early 2028", "Defence sector boom",
                       "Mining boom Apr 2028", "2nd Real Estate boom Apr 2028",
                       "FinTech / Stocks / Crypto boom Jan 2027",
                       "Health Tech boom May 2027"],
        "sectors_buy":   ["Defence", "Mining", "Commodities", "Health Tech", "FinTech (from 2027)"],
        "sectors_avoid": ["Growth tech (early 2028)", "Consumer", "Banks"],
    },
    {
        "period": "2031 – 2032",
        "label": "Recovery / New Supercycle",
        "short": "NEW CYCLE",
        "regime": "BULLISH",
        "start": _date2(2031, 1, 1),
        "end":   _date2(2032, 12, 31),
        "description": (
            "Jupiter exits debilitation, Saturn moves to Gemini. New 20-year bull cycle ignites. "
            "Supertechnology leads — quantum computing, AGI, biotech convergence. "
            "A new monetary system emerges from the 2026-2030 chaos."
        ),
        "basis": "Jupiter in Libra/Scorpio (2030-31). Saturn in Gemini (2031). "
                 "Jupiter-Saturn conjunction in air/earth sign begins new growth era.",
        "key_events": ["Supertechnology boom 2029-31", "New monetary system", "AGI commercialisation"],
        "sectors_buy":   ["Supertechnology", "Biotech", "Space", "New Energy", "Crypto 2.0"],
        "sectors_avoid": [],
    },
]

# ── Sector Rotation Timeline ──────────────────────────────────────────────────

_SECTOR_TIMELINE: list[dict] = [
    {"sector": "Housing Bubble",          "period": "2021 – 2026",      "regime": "PEAK/BUST",  "emoji": "🏠"},
    {"sector": "Crypto Bubble",           "period": "2021 – 2026",      "regime": "VOLATILE",   "emoji": "₿"},
    {"sector": "Gold / Silver Bubble",    "period": "Apr 2022 – 2026",  "regime": "PEAK",       "emoji": "🥇"},
    {"sector": "Stock Market Bubble",     "period": "Apr 2022 – 2026",  "regime": "PEAK",       "emoji": "📈"},
    {"sector": "AI Bubble",               "period": "Apr 2022 – Feb/Mar 2028", "regime": "PEAK","emoji": "🤖"},
    {"sector": "India Equities",          "period": "Mid 2021 – 2028+", "regime": "BULLISH",    "emoji": "🇮🇳"},
    {"sector": "Commodities",             "period": "2021 – 2026",      "regime": "BULLISH",    "emoji": "⛏️"},
    {"sector": "FinTech / Stocks / Crypto Boom", "period": "Jan 2027",  "regime": "NEW BULL",  "emoji": "💹"},
    {"sector": "Health Tech Boom",        "period": "May 2027",         "regime": "NEW BULL",   "emoji": "🏥"},
    {"sector": "Defence Sector",          "period": "Early 2028",       "regime": "BOOM",       "emoji": "🛡️"},
    {"sector": "Mining Boom",             "period": "Apr 2028",         "regime": "BOOM",       "emoji": "⛏️"},
    {"sector": "2nd Real Estate Boom",    "period": "Apr 2028",         "regime": "NEW BULL",   "emoji": "🏗️"},
    {"sector": "Supertechnology",         "period": "2029+",            "regime": "BOOM",       "emoji": "🚀"},
]

# ── Annual Roadmaps (month-by-month) ─────────────────────────────────────────
# Based on planetary transits + eclipse windows for each year.
# "verified": True = already occurred and matched prediction.

_ANNUAL_ROADMAPS: dict[int, list[dict]] = {
    2026: [
        {
            "period": "Jan – April",
            "months": [1, 2, 3, 4],
            "theme": "Multiple New Wars / Geopolitical Instability",
            "regime": "VOLATILE",
            "confidence": 4,
            "verified": True,
            "basis": "Mars transiting Capricorn-Aquarius. Rahu in Aquarius = geopolitical chaos. "
                     "Saturn in Pisces = porous borders, confusion.",
            "sectors": ["Defence", "Oil", "Gold"],
            "action": "Hold defence stocks. Buy gold dips. Reduce cyclical exposure.",
        },
        {
            "period": "Feb – March",
            "months": [2, 3],
            "theme": "1st Stock Market Correction",
            "regime": "BEARISH",
            "confidence": 5,
            "verified": True,
            "basis": "Solar eclipse Feb 17 (Aquarius) creates 30-day volatility window. "
                     "Saturn-Jupiter stress aspect. Mercury retrograde confusion.",
            "sectors": ["Cash", "Puts / Hedges"],
            "action": "Reduce long exposure before Feb 10. Buy puts on indices.",
        },
        {
            "period": "April – June",
            "months": [4, 5, 6],
            "theme": "Markets Rally / More Money Printing",
            "regime": "BULLISH",
            "confidence": 4,
            "verified": False,
            "basis": "Jupiter enters Cancer (exaltation) May 2026 = peak optimism signal. "
                     "Central banks respond with QE. Rahu in Aquarius = tech recovery.",
            "sectors": ["Tech", "Real Estate", "Crypto"],
            "action": "Re-enter long positions. Buy RE stocks and crypto dips.",
        },
        {
            "period": "June – July",
            "months": [6, 7],
            "theme": "Gold / Silver Correction",
            "regime": "NEUTRAL",
            "confidence": 3,
            "verified": False,
            "basis": "Rahu-Ketu axis mid-transit creates commodity confusion. "
                     "Dollar strength episode as Mars enters Aries.",
            "sectors": ["Gold", "Silver"],
            "action": "Take profits on gold/silver positions. Re-enter at support.",
        },
        {
            "period": "July",
            "months": [7],
            "theme": "2nd Market Correction",
            "regime": "BEARISH",
            "confidence": 4,
            "verified": False,
            "basis": "Mars enters Aries (own sign) = aggressive selling. "
                     "Mars opposing Saturn in Libra = sharp conflict/correction trigger.",
            "sectors": ["Cash", "Volatility plays"],
            "action": "Reduce exposure mid-June. Wait for July bottom to reload.",
        },
        {
            "period": "July – Sept",
            "months": [7, 8, 9],
            "theme": "Real Estate Boom Peaks (top Aug/Sept)",
            "regime": "BULLISH",
            "confidence": 4,
            "verified": False,
            "basis": "Jupiter in Cancer = ruler of Real Estate in exaltation. "
                     "Peak RE prices expected Aug-Sept before Jupiter weakens.",
            "sectors": ["Real Estate", "REITs", "Construction"],
            "action": "Take profits on RE exposure by end of August.",
        },
        {
            "period": "Aug – Sept",
            "months": [8, 9],
            "theme": "Extreme Political Chaos, Riots, Geopolitical Crisis",
            "regime": "VOLATILE",
            "confidence": 4,
            "verified": False,
            "basis": "Mars in Aries (own/aggressive sign) + Saturn stress. "
                     "Rahu-Ketu shifting axis. Solar eclipse Aug 13 amplifies instability.",
            "sectors": ["Defence", "Gold", "Safe havens"],
            "action": "Hold defence / gold. Avoid all risk assets.",
        },
        {
            "period": "August",
            "months": [8],
            "theme": "3rd Market Correction",
            "regime": "BEARISH",
            "confidence": 5,
            "verified": False,
            "basis": "Solar eclipse Aug 13 (Leo) = 30-day volatility window. "
                     "Mars-Saturn hard aspect. Venus retrograde confusion. "
                     "Triple eclipse season creates maximum uncertainty.",
            "sectors": ["Cash", "Short equity"],
            "action": "Go to heavy cash or short before Aug 1.",
        },
        {
            "period": "Aug – Sept",
            "months": [8, 9],
            "theme": "New Virus / Disease Threat",
            "regime": "VOLATILE",
            "confidence": 3,
            "verified": False,
            "basis": "Saturn in Pisces (12th sign) = hidden diseases, hospitals under stress. "
                     "Ketu in Leo = immunity issues. Historical: Saturn-Pisces + Ketu = health crises.",
            "sectors": ["Healthcare", "Pharma", "Biotech"],
            "action": "Buy healthcare/pharma as defensive play.",
        },
        {
            "period": "October",
            "months": [10],
            "theme": "Real Estate Crash + THE BIG ONE (Market Crash Begins)",
            "regime": "BEARISH",
            "confidence": 5,
            "verified": False,
            "basis": "Jupiter leaving Cancer exaltation peak. Saturn-Jupiter opposition exact. "
                     "Lunar eclipse Oct 2026 = 30-day window. Rahu enters Capricorn (Nov) = financial axis. "
                     "Historical: Jupiter-Saturn oppositions always mark major market turning points.",
            "sectors": ["Short equity", "Cash", "Bonds"],
            "action": "Maximum defensive position. This is the generational crash setup.",
        },
        {
            "period": "Oct – Nov",
            "months": [10, 11],
            "theme": "Crypto Bottom / Capitulation",
            "regime": "NEUTRAL",
            "confidence": 4,
            "verified": False,
            "basis": "Rahu enters Capricorn (financial reckoning sign) Nov 2026. "
                     "Crypto oversold + Rahu-Capricorn historically = institutional accumulation bottom.",
            "sectors": ["Crypto (accumulate at bottom)"],
            "action": "Begin DCA into BTC/ETH at capitulation lows.",
        },
        {
            "period": "December",
            "months": [12],
            "theme": "Major Money Printing + 1st Disease Variant",
            "regime": "NEUTRAL",
            "confidence": 3,
            "verified": False,
            "basis": "Government/central bank panic response to crash. "
                     "Saturn-Pisces health axis: 1st variant of new disease reported. "
                     "QE announcement begins recovery setup for Jan 2027.",
            "sectors": ["Crypto (early recovery)", "Gold", "FinTech"],
            "action": "Begin accumulating risk assets on QE announcement.",
        },
    ],
    2025: [
        {
            "period": "Jan – April",
            "months": [1, 2, 3, 4],
            "theme": "AI / Tech Boom Continues, Crypto Rally",
            "regime": "BULLISH",
            "confidence": 4,
            "verified": True,
            "basis": "Jupiter in Taurus (wealth, stability). Rahu in Pisces = speculation/crypto boom. "
                     "Saturn transitioning Aquarius→Pisces.",
            "sectors": ["AI/Tech", "Crypto", "Gold"],
            "action": "Stay long tech and crypto. Trail stops up.",
        },
        {
            "period": "April – May",
            "months": [4, 5],
            "theme": "Market Volatility / Correction Episode",
            "regime": "VOLATILE",
            "confidence": 3,
            "verified": True,
            "basis": "Saturn enters Pisces (Mar 29). Jupiter ingress Gemini (May 14). "
                     "Transition periods between major planetary sign changes = volatility.",
            "sectors": ["Cash", "Gold"],
            "action": "Reduce exposure around ingress dates.",
        },
        {
            "period": "May – Dec",
            "months": [5, 6, 7, 8, 9, 10, 11, 12],
            "theme": "Jupiter in Gemini — AI/Tech/Communication Superboom",
            "regime": "BULLISH",
            "confidence": 4,
            "verified": True,
            "basis": "Jupiter in Gemini (communication, tech, trade) + Rahu in Aquarius (from May 18) "
                     "= maximum AI/tech/crypto optimism. Stock market at extreme valuations.",
            "sectors": ["AI/Tech", "Crypto", "Communications", "Fintech"],
            "action": "Ride the bubble but trail stops. Set alerts for Oct-Nov pullback.",
        },
    ],
    2027: [
        {
            "period": "Jan 2027",
            "months": [1],
            "theme": "FinTech / Stocks / Crypto Boom Begins Recovery",
            "regime": "BULLISH",
            "confidence": 4,
            "verified": False,
            "basis": "Central bank QE fully deployed. Jupiter in Leo (confidence, risk appetite). "
                     "Rahu in Capricorn = institutional crypto accumulation complete.",
            "sectors": ["Crypto", "FinTech", "Growth stocks"],
            "action": "Scale into quality growth positions bought at 2026 crash lows.",
        },
        {
            "period": "May 2027",
            "months": [5],
            "theme": "Health Tech Boom",
            "regime": "BULLISH",
            "confidence": 3,
            "verified": False,
            "basis": "Post-disease response drives massive health tech investment. "
                     "Jupiter in Leo = large-scale investments in innovation.",
            "sectors": ["Health Tech", "Biotech", "Pharma"],
            "action": "Accumulate health tech positions from late 2026.",
        },
        {
            "period": "June – Dec 2027",
            "months": [6, 7, 8, 9, 10, 11, 12],
            "theme": "Saturn Enters Aries — Conflict Escalation",
            "regime": "VOLATILE",
            "confidence": 4,
            "verified": False,
            "basis": "Saturn enters Aries (Jun 2027) = military aggression, political confrontations. "
                     "Markets choppy but supported by QE.",
            "sectors": ["Defence", "Commodities", "Gold"],
            "action": "Overweight defence. Keep commodity exposure.",
        },
    ],
    2028: [
        {
            "period": "Early 2028",
            "months": [1, 2, 3],
            "theme": "New Global Conflicts / Defence Sector Boom",
            "regime": "VOLATILE",
            "confidence": 4,
            "verified": False,
            "basis": "Jupiter enters Virgo (debilitation Jul 2028). Saturn in Aries. "
                     "Mars-Saturn conjunctions in fire signs = military escalation.",
            "sectors": ["Defence", "Oil", "Gold"],
            "action": "Heavy defence weighting. Gold as reserve.",
        },
        {
            "period": "April 2028",
            "months": [4],
            "theme": "Mining Boom + 2nd Real Estate Boom",
            "regime": "BULLISH",
            "confidence": 3,
            "verified": False,
            "basis": "Rahu in Sagittarius (resource-rich sign). Jupiter aspect on 2nd house resources. "
                     "War-time demand drives commodity supercycle.",
            "sectors": ["Mining", "Real Estate", "Commodities"],
            "action": "Accumulate mining and RE stocks in early 2028.",
        },
    ],
}


def get_decade_cheatsheet() -> dict:
    """Return macro regime cheatsheet + sector timeline.
    
    Returns dict with:
        'regimes': list of decade regime dicts
        'sectors': list of sector timeline dicts
        'current_regime': the dict for the current period
        'next_regime': the dict for the upcoming period
    """
    today = _date2.today()
    current = None
    next_r  = None
    for i, r in enumerate(_DECADE_REGIMES):
        if r["start"] <= today <= r["end"]:
            current = r
            if i + 1 < len(_DECADE_REGIMES):
                next_r = _DECADE_REGIMES[i + 1]
            break
    if current is None:
        # Fall back to last entry
        current = _DECADE_REGIMES[-1]

    return {
        "regimes":        _DECADE_REGIMES,
        "sectors":        _SECTOR_TIMELINE,
        "current_regime": current,
        "next_regime":    next_r,
    }


def get_annual_roadmap(year=None) -> dict:
    """Return month-by-month market roadmap for the given year.
    
    Returns dict with:
        'year': int
        'events': list of roadmap event dicts (may include eclipse events)
        'eclipses': list of eclipse dicts computed via ephem
    """
    import datetime as _datetime_mod
    year = year or _datetime_mod.datetime.utcnow().year

    events = _ANNUAL_ROADMAPS.get(year, [])

    # Compute eclipse windows for the year via ephem
    eclipses = _compute_year_eclipses(year)

    return {
        "year":     year,
        "events":   events,
        "eclipses": eclipses,
    }


def _compute_year_eclipses(year: int) -> list[dict]:
    """Compute solar and lunar eclipses for a year using ephem."""
    results = []
    try:
        start = ephem.Date(f"{year}/1/1")
        end   = ephem.Date(f"{year+1}/1/1")
        cursor = start

        while cursor < end:
            # Solar eclipse check — new moon near node
            nm = ephem.next_new_moon(cursor)
            if nm >= end:
                break
            moon = ephem.Moon()
            moon.compute(nm)
            moon_lat = abs(math.degrees(float(moon.hlat)))
            nm_dt = _dt(nm)
            if moon_lat < 1.6 and nm_dt.year == year:
                results.append({
                    "date":        nm_dt.date(),
                    "type":        "Solar Eclipse",
                    "emoji":       "🌑",
                    "window_days": 30,
                    "bias":        "VOLATILE",
                    "description": f"Solar eclipse — 30-day high-volatility window around {nm_dt.strftime('%b %d')}",
                })

            # Lunar eclipse check — full moon near node
            fm = ephem.next_full_moon(cursor)
            if fm >= end:
                break
            moon.compute(fm)
            moon_lat = abs(math.degrees(float(moon.hlat)))
            fm_dt = _dt(fm)
            if moon_lat < 1.1 and fm_dt.year == year:
                results.append({
                    "date":        fm_dt.date(),
                    "type":        "Lunar Eclipse",
                    "emoji":       "🌕",
                    "window_days": 14,
                    "bias":        "BEARISH",
                    "description": f"Lunar eclipse — 14-day bearish reversal window around {fm_dt.strftime('%b %d')}",
                })

            cursor = nm + 25  # Jump past current cycle
    except Exception:
        pass

    return sorted(results, key=lambda e: e["date"])


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATIONAL FORECAST ENGINE
# Derives predictions algorithmically from planetary positions.
# No hardcoded year-specific data — works for any date range.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Interpretation tables ─────────────────────────────────────────────────────

# (score -3..+3, regime label, description, sectors_bullish, sectors_bearish)
_JUPITER_SIGN_TABLE = {
    "Aries":       (+1, "BULLISH",  "Innovation & fresh energy — moderate bull",
                    ["Defence", "Energy", "Startups"], []),
    "Taurus":      (+2, "BULLISH",  "Wealth & stability — real estate and commodity boom",
                    ["Real Estate", "Gold", "Banks"], []),
    "Gemini":      (+2, "BULLISH",  "Tech, AI, trade expansion — communication stocks soar",
                    ["Tech", "AI", "Media", "Fintech"], []),
    "Cancer":      (+3, "BULLISH",  "EXALTATION — peak economic boom, real estate at highs",
                    ["Real Estate", "Food", "Commodities", "Consumer"], []),
    "Leo":         (+1, "BULLISH",  "Confidence & luxury — stock speculation and pride",
                    ["Luxury", "Entertainment", "Speculation"], []),
    "Virgo":       (-2, "BEARISH",  "DEBILITATION — institutional stress, corrections likely",
                    ["Healthcare", "Pharma"], ["Banks", "Tech", "Growth"]),
    "Libra":       ( 0, "NEUTRAL",  "Balance & diplomacy — moderate, range-bound markets",
                    ["Legal", "Finance"], []),
    "Scorpio":     (-1, "BEARISH",  "Hidden dangers & debt transformation — volatile",
                    ["Crypto", "Insurance"], ["Banks", "Consumer"]),
    "Sagittarius": (+2, "BULLISH",  "International expansion — global markets & commodities boom",
                    ["International", "Commodities", "Education"], []),
    "Capricorn":   (-3, "BEARISH",  "DEBILITATION — severe hardship, systemic collapses",
                    ["Gold", "Defence"], ["Everything else"]),
    "Aquarius":    (+1, "BULLISH",  "Tech & social innovation — moderate gains, crypto themes",
                    ["Tech", "Crypto", "Solar/EV"], []),
    "Pisces":      ( 0, "NEUTRAL",  "Speculation & spirituality — mixed signals, oil/healthcare",
                    ["Oil", "Healthcare", "Pharma"], []),
}

_SATURN_SIGN_TABLE = {
    "Aries":       (-3, "BEARISH",  "DEBILITATED — aggression, war, market panic & crashes",
                    ["Defence", "Gold"], ["Consumer", "Banks", "RE"]),
    "Taurus":      ( 0, "NEUTRAL",  "Slow wealth accumulation — stable but sluggish",
                    ["Real Estate", "Agriculture"], []),
    "Gemini":      ( 0, "NEUTRAL",  "Communication & regulatory delays — choppy tech",
                    [], ["Media", "Tech"]),
    "Cancer":      (-1, "BEARISH",  "Housing & family stress — real estate headwinds",
                    [], ["Real Estate", "Consumer"]),
    "Leo":         (-1, "BEARISH",  "Government control — political risk premium in markets",
                    ["Gold"], ["Entertainment", "Luxury"]),
    "Virgo":       ( 0, "NEUTRAL",  "Analytical & healthcare focus — cautious market",
                    ["Healthcare"], []),
    "Libra":       (+2, "BULLISH",  "EXALTATION — law, order, stability — structural bull",
                    ["Finance", "Legal", "Banks"], []),
    "Scorpio":     (-1, "BEARISH",  "Hidden debt crisis, leverage exposed",
                    [], ["Banks", "Fintech"]),
    "Sagittarius": ( 0, "NEUTRAL",  "International tensions — commodities mixed",
                    ["Oil", "Defence"], []),
    "Capricorn":   (-2, "BEARISH",  "Own sign — severe hardship, systemic collapse risk",
                    ["Gold"], ["Growth", "Tech", "Consumer"]),
    "Aquarius":    ( 0, "NEUTRAL",  "Tech regulation & social reform — disruption period",
                    ["Crypto", "Solar"], ["Big Tech"]),
    "Pisces":      (-1, "BEARISH",  "Confusion, health crises, debt — liquidity concerns",
                    ["Pharma", "Oil"], ["Banks"]),
}

_RAHU_SIGN_SCORE = {
    "Aries":       +1, "Taurus":      +2, "Gemini":   +1, "Cancer":      -1,
    "Leo":         -1, "Virgo":        0, "Libra":    +1, "Scorpio":     -2,
    "Sagittarius": +1, "Capricorn":   -2, "Aquarius":  0, "Pisces":      -1,
}

_RAHU_SIGN_SECTORS = {
    "Aries":       ["Defence", "Energy"],
    "Taurus":      ["Gold", "Silver", "Crypto", "RE"],
    "Gemini":      ["AI", "Media", "Comms"],
    "Cancer":      ["RE (peak)", "Food"],
    "Leo":         ["Gold", "Political risk plays"],
    "Virgo":       ["Healthcare", "Pharma"],
    "Libra":       ["Finance", "Luxury", "M&A"],
    "Scorpio":     ["Crypto", "Derivatives", "Insurance"],
    "Sagittarius": ["EM stocks", "Commodities", "International"],
    "Capricorn":   ["Govt bonds", "Gold"],
    "Aquarius":    ["Crypto", "Tech", "AI"],
    "Pisces":      ["Oil", "Pharma"],
}

# Aspect patterns: angle → (name, score_impact, description)
_HARD_ASPECTS = {
    0:   ("Conjunction", -1, "Planets merged — amplifies whichever is stronger"),
    90:  ("Square",      -2, "Hard tension — conflict, sharp corrections"),
    180: ("Opposition",  -3, "Maximum stress — market turning point, reversal risk"),
}
_SOFT_ASPECTS = {
    60:  ("Sextile",  +1, "Opportunity aspect — mild tailwind"),
    120: ("Trine",    +2, "Harmony — smooth bull trend"),
}


_INNER_PLANETS = {ephem.Sun, ephem.Moon, ephem.Mercury, ephem.Venus}

def _planet_tropical_lon(planet_cls, dt: datetime) -> float:
    """Geocentric ecliptic (tropical) longitude for all planets.
    Uses ephem.Ecliptic with matching epoch for correct geocentric position."""
    ed = _ed(dt)
    obj = planet_cls()
    obj.compute(ed)
    return math.degrees(float(ephem.Ecliptic(obj, epoch=ed).lon)) % 360


def _angular_separation(lon1: float, lon2: float) -> float:
    diff = abs(lon1 - lon2) % 360
    return diff if diff <= 180 else 360 - diff


def _check_aspect(lon1: float, lon2: float, orb: float = 8.0):
    sep = _angular_separation(lon1, lon2)
    for target, (name, score, desc) in {**_HARD_ASPECTS, **_SOFT_ASPECTS}.items():
        if abs(sep - target) <= orb:
            return {"aspect": name, "orb": round(abs(sep - target), 1),
                    "score": score, "description": desc, "separation": round(sep, 1)}
    return None


def _get_planet_aspects(dt: datetime) -> list[dict]:
    """Active aspects between Jupiter, Saturn, Mars (orb 8°)."""
    pairs = [
        ("Jupiter", ephem.Jupiter, "Saturn", ephem.Saturn),
        ("Mars",    ephem.Mars,    "Saturn", ephem.Saturn),
        ("Jupiter", ephem.Jupiter, "Mars",   ephem.Mars),
    ]
    results = []
    for n1, c1, n2, c2 in pairs:
        lon1 = _planet_tropical_lon(c1, dt)
        lon2 = _planet_tropical_lon(c2, dt)
        asp  = _check_aspect(lon1, lon2)
        if asp:
            asp["planet1"] = n1
            asp["planet2"] = n2
            results.append(asp)
    return results


def _is_mercury_rx(dt: datetime) -> bool:
    try:
        return _is_retrograde(ephem.Mercury, dt)
    except Exception:
        return False


def _eclipse_window_score(dt: datetime, eclipses: list[dict]) -> float:
    """Return -1 if within 15 days of an eclipse, -0.5 if within 30 days."""
    from datetime import timedelta
    d = dt.date() if hasattr(dt, "date") else dt
    for e in eclipses:
        delta = abs((e["date"] - d).days)
        if delta <= 15:
            return -1.0
        if delta <= 30:
            return -0.5
    return 0.0


def compute_monthly_forecast(year: int, month: int,
                              eclipses=None) -> dict:
    """
    Full market forecast for one calendar month.
    Samples positions on 1st, 8th, 15th, 22nd.

    Trigger set:
      Base layer : Jupiter/Saturn sign (exaltation/debilitation), Rahu sign
      Aspect     : Mars-Saturn, Jupiter-Saturn, Jupiter-Mars hard/soft
      Modifiers  : Jupiter Atichara (neutralises exaltation)
                   Jupiter combustion, Venus Rx, Mercury Rx
      Yogas      : Guru Chandal, Saturn-Rahu, Mars-Rahu (Angarak)
                   Saturn-Neptune hard aspect (historical crash signal)
      Sandhi     : Planet at sign boundary (last/first 1°)
                   Rahu at sign boundary (maximum malefic)
      Eclipse    : Window score, Sarpa Dosha, eclipse point activation
      Transition : Rahu axis sign-change month
    """
    import calendar as _cal
    from collections import Counter as _Counter

    if eclipses is None:
        eclipses = _compute_year_eclipses(year)

    sample_days = [1, 8, 15, 22]
    jup_scores, sat_scores, rahu_scores = [], [], []
    asp_scores, ecl_scores = [], []
    jup_signs_seen, sat_signs_seen, rahu_signs_seen = [], [], []
    all_aspects = []
    rx_days = 0

    # Per-sample extra-trigger collections
    combustion_scores, venus_rx_scores = [], []
    guru_chandal_scores, sat_rahu_scores, mars_rahu_scores = [], [], []
    sat_nep_scores, sandhi_scores, rahu_sandhi_scores = [], [], []
    eclipse_trig_scores = []
    jup_atichara_flags = []
    jup_retro_scores, sat_retro_scores, mars_retro_scores = [], [], []

    for day in sample_days:
        try:
            day = min(day, _cal.monthrange(year, month)[1])
            dt  = datetime(year, month, day, 12, 0, 0)

            # ── Slow planet signs ──────────────────────────────────────────
            jup_lon  = _planet_tropical_lon(ephem.Jupiter, dt)
            jup_sid  = _tropical_to_sidereal(jup_lon)
            jup_sign = _lon_to_sign(jup_sid)
            jup_signs_seen.append(jup_sign)
            jup_data = _JUPITER_SIGN_TABLE.get(jup_sign, (0, "NEUTRAL", "", [], []))

            # Atichara: dampen Jupiter score when it's moving too fast
            ati_mult = _jupiter_atichara_multiplier(dt)
            jup_atichara_flags.append(ati_mult < 1.0)
            jup_scores.append(jup_data[0] * ati_mult)

            sat_lon  = _planet_tropical_lon(ephem.Saturn, dt)
            sat_sid  = _tropical_to_sidereal(sat_lon)
            sat_sign = _lon_to_sign(sat_sid)
            sat_signs_seen.append(sat_sign)
            sat_data = _SATURN_SIGN_TABLE.get(sat_sign, (0, "NEUTRAL", "", [], []))
            sat_scores.append(sat_data[0])

            rahu_tropical = _mean_lunar_node(dt)
            rahu_sid  = _tropical_to_sidereal(rahu_tropical)
            rahu_sign = _lon_to_sign(rahu_sid)
            rahu_signs_seen.append(rahu_sign)
            rahu_scores.append(_RAHU_SIGN_SCORE.get(rahu_sign, 0))

            # ── Aspects (hard only for bearish signal; soft aspects neutral) ──
            for a in _get_planet_aspects(dt):
                # Only count hard aspects (conjunction, square, opposition) as score contributors
                hard_score = a["score"] if a["score"] < 0 else 0
                asp_scores.append(hard_score)
                all_aspects.append(a)

            # ── Eclipse window ─────────────────────────────────────────────
            ecl_scores.append(_eclipse_window_score(dt, eclipses))

            # ── Mercury Rx ─────────────────────────────────────────────────
            if _is_mercury_rx(dt):
                rx_days += 1

            # ── Additional triggers ────────────────────────────────────────
            combustion_scores.append(_jupiter_combustion_trigger(dt)[0])
            venus_rx_scores.append(_venus_rx_trigger(dt)[0])
            guru_chandal_scores.append(_guru_chandal_trigger(dt)[0])
            sat_rahu_scores.append(_saturn_rahu_trigger(dt)[0])
            mars_rahu_scores.append(_mars_rahu_trigger(dt)[0])
            sat_nep_scores.append(_saturn_neptune_trigger(dt)[0])
            sandhi_scores.append(_graha_sandhi_trigger(dt)[0])
            rahu_sandhi_scores.append(_rahu_sandhi_trigger(dt)[0])
            eclipse_trig_scores.append(_eclipse_trigger_score(dt, eclipses)[0])
            jup_retro_scores.append(_planet_retrograde_trigger(ephem.Jupiter, "Jupiter", dt)[0])
            sat_retro_scores.append(_planet_retrograde_trigger(ephem.Saturn, "Saturn", dt)[0])
            mars_retro_scores.append(_planet_retrograde_trigger(ephem.Mars, "Mars", dt)[0])

        except Exception:
            pass

    if not jup_scores:
        return {}

    def _dominant(lst):
        return _Counter(lst).most_common(1)[0][0] if lst else "Unknown"

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    jup_sign  = _dominant(jup_signs_seen)
    sat_sign  = _dominant(sat_signs_seen)
    rahu_sign = _dominant(rahu_signs_seen)
    jup_info  = _JUPITER_SIGN_TABLE.get(jup_sign, (0, "NEUTRAL", "", [], []))
    sat_info  = _SATURN_SIGN_TABLE.get(sat_sign,  (0, "NEUTRAL", "", [], []))

    # ── Composite score ────────────────────────────────────────────────────
    # Base layer (Atichara already baked into jup_scores)
    base_score = (
        3 * _avg(jup_scores) +
        2 * _avg(sat_scores) +
        1 * _avg(rahu_scores)
    ) / 6  # normalised -3..+3

    # Classic layers
    asp_adj  = _avg(asp_scores) * 0.45 if asp_scores else 0
    ecl_adj  = _avg(ecl_scores)
    rx_adj   = -0.3 if rx_days >= 2 else 0.0

    # Additional trigger layers (each weighted at 0.7 of raw score)
    extra_triggers = [
        _avg(combustion_scores),
        _avg(venus_rx_scores),
        _avg(guru_chandal_scores),
        _avg(sat_rahu_scores),
        _avg(mars_rahu_scores),
        _avg(sat_nep_scores),
        _avg(sandhi_scores),
        _avg(rahu_sandhi_scores),
        _avg(eclipse_trig_scores),
        _avg(jup_retro_scores),
        _avg(sat_retro_scores),
        _avg(mars_retro_scores),
    ]
    extra_adj = sum(extra_triggers) * 0.7

    # One-off month-level triggers
    sarpa_score,  sarpa_flag   = _sarpa_dosha_trigger(eclipses, year, month)
    axis_score,   axis_flag    = _rahu_axis_shift_trigger(year, month)
    ingress_score, ingress_flag = _jupiter_sign_change_trigger(year, month)
    month_extra = (sarpa_score + axis_score + ingress_score) * 0.7

    total = round(base_score + asp_adj + ecl_adj + rx_adj + extra_adj + month_extra, 2)
    total = max(-4.0, min(3.5, total))

    # ── Regime ────────────────────────────────────────────────────────────
    if total >= 1.5:
        regime, emoji = "BULLISH",  "▲"
    elif total >= 0.3:
        regime, emoji = "NEUTRAL",  "→"
    elif total >= -0.5:
        regime, emoji = "VOLATILE", "⚡"
    else:
        regime, emoji = "BEARISH",  "▼"

    # ── Sectors ───────────────────────────────────────────────────────────
    bull_sectors = list(dict.fromkeys(
        jup_info[3] + sat_info[3] + _RAHU_SIGN_SECTORS.get(rahu_sign, [])
    ))
    bear_sectors = list(dict.fromkeys(jup_info[4] + sat_info[4]))

    # ── Unique aspects ────────────────────────────────────────────────────
    seen_asp, unique_aspects = set(), []
    for a in all_aspects:
        k = (a["planet1"], a["planet2"], a["aspect"])
        if k not in seen_asp:
            seen_asp.add(k); unique_aspects.append(a)

    # ── Key events (narrative) ────────────────────────────────────────────
    key_events = []
    atichara_active = any(jup_atichara_flags)

    if jup_info[0] == 3:
        label = "chaotic exaltation (Atichara)" if atichara_active else "peak boom signal"
        key_events.append(f"Jupiter exalted in {jup_sign} — {label}")
    if jup_info[0] == -2:
        key_events.append(f"Jupiter debilitated in {jup_sign} — correction risk")
    if sat_info[0] <= -2:
        key_events.append(f"Saturn debilitated in {sat_sign} — war/crash environment")
    if sat_info[0] == 2:
        key_events.append(f"Saturn exalted in {sat_sign} — structural stability")
    if atichara_active:
        key_events.append("Jupiter Atichara (fast motion) — exaltation power halved")

    # Collect non-empty trigger descriptions for mid-month sample
    mid_dt = datetime(year, month, 15, 12, 0, 0)
    for fn in [_jupiter_combustion_trigger, _venus_rx_trigger,
               _guru_chandal_trigger, _saturn_rahu_trigger,
               _mars_rahu_trigger, _saturn_neptune_trigger,
               _rahu_sandhi_trigger]:
        try:
            s, desc = fn(mid_dt)
            if s < 0 and desc:
                key_events.append(desc)
        except Exception:
            pass
    try:
        s, desc = _graha_sandhi_trigger(mid_dt)
        if s < 0 and desc:
            key_events.append(desc)
    except Exception:
        pass
    try:
        s, desc = _eclipse_trigger_score(mid_dt, eclipses)
        if s < 0 and desc:
            key_events.append(desc)
    except Exception:
        pass

    for a in unique_aspects:
        if a["score"] <= -2:
            key_events.append(f"{a['planet1']}-{a['planet2']} {a['aspect']} — {a['description']}")
    if ecl_adj < -0.4:
        key_events.append("Eclipse window active — heightened volatility")
    if rx_days >= 2:
        key_events.append("Mercury retrograde — false breakouts, confusion")
    if sarpa_flag:
        key_events.append(sarpa_flag)
    if axis_flag:
        key_events.append(axis_flag)
    if ingress_flag:
        key_events.append(ingress_flag)

    month_name = datetime(year, month, 1).strftime("%B %Y")

    return {
        "month":          month,
        "month_name":     month_name,
        "score":          total,
        "regime":         regime,
        "emoji":          emoji,
        "jupiter_sign":   jup_sign,
        "jupiter_score":  round(_avg(jup_scores), 2),
        "jupiter_atichara": atichara_active,
        "jupiter_desc":   jup_info[2],
        "saturn_sign":    sat_sign,
        "saturn_score":   sat_info[0],
        "saturn_desc":    sat_info[2],
        "rahu_sign":      rahu_sign,
        "rahu_desc":      _RAHU_SIGN_THEME.get(rahu_sign, ("NEUTRAL", ""))[1],
        "aspects":        unique_aspects,
        "eclipse_active":  ecl_adj < -0.3,
        "mercury_rx":      rx_days >= 2,
        "venus_rx":        _avg(venus_rx_scores) < -0.3,
        "jupiter_retro":   _avg(jup_retro_scores) < -0.3,
        "atichara":        atichara_active,
        "sectors_bull":    bull_sectors[:5],
        "sectors_bear":    bear_sectors[:4],
        "key_events":      key_events[:8],
        "score_breakdown": {
            "base":           round(base_score, 2),
            "aspects":        round(asp_adj, 2),
            "eclipse_win":    round(ecl_adj, 2),
            "mercury_rx":     round(rx_adj, 2),
            "extra_triggers": round(extra_adj, 2),
            "month_events":   round(month_extra, 2),
        },
        "basis": (
            f"Jupiter {jup_sign}{' [Atichara]' if atichara_active else ''} ({jup_info[2]}). "
            f"Saturn {sat_sign} ({sat_info[2]}). Rahu {rahu_sign}."
        ),
    }


def generate_annual_forecast(year: int) -> dict:
    """
    Compute month-by-month forecast for an entire year.

    Returns:
        'year': int
        'months': list of monthly forecast dicts (12 entries)
        'periods': grouped list — consecutive months with same regime merged
        'eclipses': eclipse list for the year
        'annual_score': average composite score
        'annual_regime': overall year regime
    """
    eclipses = _compute_year_eclipses(year)
    months   = []
    for m in range(1, 13):
        mf = compute_monthly_forecast(year, m, eclipses)
        if mf:
            months.append(mf)

    if not months:
        return {"year": year, "months": [], "periods": [], "eclipses": eclipses,
                "annual_score": 0, "annual_regime": "UNKNOWN"}

    annual_score = round(sum(m["score"] for m in months) / len(months), 2)

    if annual_score >= 1.2:
        annual_regime = "BULLISH"
    elif annual_score >= 0.2:
        annual_regime = "NEUTRAL"
    elif annual_score >= -0.5:
        annual_regime = "VOLATILE"
    else:
        annual_regime = "BEARISH"

    # Group consecutive months with same regime into periods
    periods = []
    i = 0
    _MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    while i < len(months):
        j = i
        while j < len(months) and months[j]["regime"] == months[i]["regime"]:
            j += 1
        group = months[i:j]
        start_abbr = _MONTH_ABBR[group[0]["month"] - 1]
        end_abbr   = _MONTH_ABBR[group[-1]["month"] - 1]
        period_label = start_abbr if len(group) == 1 else f"{start_abbr} – {end_abbr}"

        # Collect all unique key events and sectors across grouped months
        all_events   = []
        all_bull_sec = []
        all_bear_sec = []
        all_aspects  = []
        for mf in group:
            all_events   += mf["key_events"]
            all_bull_sec += mf["sectors_bull"]
            all_bear_sec += mf["sectors_bear"]
            all_aspects  += mf["aspects"]
        # Deduplicate
        seen_ev = set()
        unique_events = [e for e in all_events if not (e in seen_ev or seen_ev.add(e))]
        bull_sec = list(dict.fromkeys(all_bull_sec))[:5]
        bear_sec = list(dict.fromkeys(all_bear_sec))[:4]
        seen_asp = set()
        unique_asp = []
        for a in all_aspects:
            k = (a["planet1"], a["planet2"], a["aspect"])
            if k not in seen_asp:
                seen_asp.add(k)
                unique_asp.append(a)

        avg_score = round(sum(m["score"] for m in group) / len(group), 2)
        periods.append({
            "period":        period_label,
            "months_list":   [m["month"] for m in group],
            "regime":        group[0]["regime"],
            "emoji":         group[0]["emoji"],
            "avg_score":     avg_score,
            "jupiter_sign":  group[0]["jupiter_sign"],
            "saturn_sign":   group[0]["saturn_sign"],
            "rahu_sign":     group[0]["rahu_sign"],
            "key_events":    unique_events[:6],
            "sectors_bull":  bull_sec,
            "sectors_bear":  bear_sec,
            "aspects":       unique_asp,
            "eclipse_active": any(m["eclipse_active"] for m in group),
            "mercury_rx":    any(m["mercury_rx"] for m in group),
            "basis":         group[0]["basis"],
        })
        i = j

    return {
        "year":          year,
        "months":        months,
        "periods":       periods,
        "eclipses":      eclipses,
        "annual_score":  annual_score,
        "annual_regime": annual_regime,
    }


def generate_multi_year_outlook(start_year: int, end_year: int) -> list[dict]:
    """Annual summary for each year in range — for decade-level view."""
    results = []
    for y in range(start_year, end_year + 1):
        fc = generate_annual_forecast(y)
        # Just the annual summary, not all 12 months
        results.append({
            "year":          y,
            "annual_score":  fc["annual_score"],
            "annual_regime": fc["annual_regime"],
            "eclipses":      len(fc["eclipses"]),
            "periods":       fc["periods"],
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL TRIGGER FUNCTIONS (appended — called by compute_monthly_forecast)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_atichara(planet_cls, dt: datetime, normal_deg_per_day: float) -> bool:
    """True if planet moving faster than its normal daily speed (Atichara)."""
    try:
        lon1 = _planet_tropical_lon(planet_cls, dt)
        lon2 = _planet_tropical_lon(planet_cls, dt + timedelta(days=1))
        motion = abs((lon2 - lon1 + 180) % 360 - 180)
        return motion > normal_deg_per_day
    except Exception:
        return False


def _jupiter_atichara_multiplier(dt: datetime) -> float:
    """
    When Jupiter is in Atichara (fast motion > 0.22°/day), its exaltation
    becomes chaotic — reduce its base score contribution by 50%.
    Normal Jupiter = ~0.083°/day; Atichara threshold = 0.22°/day.
    """
    return 0.5 if _is_atichara(ephem.Jupiter, dt, 0.22) else 1.0


def _jupiter_combustion_trigger(dt: datetime):
    """Jupiter within 11° of Sun = combust. Exaltation neutralised."""
    try:
        sun_lon = _planet_tropical_lon(ephem.Sun, dt)
        jup_lon = _planet_tropical_lon(ephem.Jupiter, dt)
        sep = _angular_separation(sun_lon, jup_lon)
        if sep <= 5:
            return -2.5, f"Jupiter cazimi/combust (Sun-Jup {sep:.1f}°) — exaltation erased"
        if sep <= 8:
            return -1.5, f"Jupiter combusting (Sun-Jup {sep:.1f}°)"
        if sep <= 11:
            return -0.7, f"Jupiter approaching combust ({sep:.1f}°)"
    except Exception:
        pass
    return 0.0, ""


def _venus_rx_trigger(dt: datetime):
    """Venus retrograde — valuation repricing, financial sentiment reversal."""
    try:
        if _is_retrograde(ephem.Venus, dt):
            return -0.8, "Venus retrograde — valuation repricing"
    except Exception:
        pass
    return 0.0, ""


def _guru_chandal_trigger(dt: datetime):
    """Jupiter-Rahu conjunction ≤10° = Guru Chandal Yoga."""
    try:
        jup_lon  = _planet_tropical_lon(ephem.Jupiter, dt)
        rahu_lon = _mean_lunar_node(dt)
        sep = _angular_separation(jup_lon, rahu_lon)
        if sep <= 5:
            return -2.0, f"Guru Chandal Yoga (Jup-Rahu {sep:.1f}°) — exaltation overwhelmed"
        if sep <= 10:
            return -1.0, f"Guru Chandal Yoga forming ({sep:.1f}°)"
        if sep <= 15:
            return -0.4, f"Guru Chandal approaching ({sep:.1f}°)"
    except Exception:
        pass
    return 0.0, ""


def _saturn_rahu_trigger(dt: datetime):
    """Saturn-Rahu conjunction ≤10° = Durbhiksha (severe economic hardship)."""
    try:
        sat_lon  = _planet_tropical_lon(ephem.Saturn, dt)
        rahu_lon = _mean_lunar_node(dt)
        sep = _angular_separation(sat_lon, rahu_lon)
        if sep <= 5:
            return -2.0, f"Saturn-Rahu {sep:.1f}° — Durbhiksha (economic crisis)"
        if sep <= 10:
            return -1.2, f"Saturn-Rahu {sep:.1f}° — hardship signal"
        if sep <= 15:
            return -0.5, f"Saturn-Rahu approaching ({sep:.1f}°)"
    except Exception:
        pass
    return 0.0, ""


def _mars_rahu_trigger(dt: datetime):
    """Mars-Rahu conjunction ≤10° = Angarak Yoga — violent shock, crash spike."""
    try:
        mars_lon = _planet_tropical_lon(ephem.Mars, dt)
        rahu_lon = _mean_lunar_node(dt)
        sep = _angular_separation(mars_lon, rahu_lon)
        if sep <= 5:
            return -2.0, f"Angarak Yoga (Mars-Rahu {sep:.1f}°) — violent correction"
        if sep <= 10:
            return -1.0, f"Mars-Rahu {sep:.1f}° — spike risk"
        if sep <= 15:
            return -0.4, f"Mars-Rahu approaching ({sep:.1f}°)"
    except Exception:
        pass
    return 0.0, ""


def _saturn_neptune_trigger(dt: datetime):
    """Saturn-Neptune hard aspects (0°/90°/180°, orb 8°) — historical crash signature.
    Particularly potent when aligned with NYSE natal chart degrees."""
    try:
        sat_lon = _planet_tropical_lon(ephem.Saturn, dt)
        nep = ephem.Neptune()
        nep.compute(_ed(dt))
        nep_lon = math.degrees(float(nep.hlong)) % 360
        sep = _angular_separation(sat_lon, nep_lon)
        for target, label in [(0, "conjunction"), (90, "square"), (180, "opposition")]:
            orb = abs(sep - target)
            if orb <= 4:
                return -1.5, f"Saturn-Neptune {label} ({sep:.0f}°) — historical crash pattern"
            if orb <= 8:
                return -0.8, f"Saturn-Neptune {label} forming ({sep:.0f}°)"
    except Exception:
        pass
    return 0.0, ""


def _graha_sandhi_trigger(dt: datetime):
    """Jupiter/Saturn/Mars within last or first 1° of a sidereal sign = weakened."""
    score, flags = 0.0, []
    try:
        for name, cls in [("Jupiter", ephem.Jupiter), ("Saturn", ephem.Saturn), ("Mars", ephem.Mars)]:
            lon = _planet_tropical_lon(cls, dt)
            sid = _tropical_to_sidereal(lon)
            deg = sid % 30
            if deg >= 29.0 or deg <= 1.0:
                score -= 0.8; flags.append(f"{name} Sandhi")
            elif deg >= 28.0 or deg <= 2.0:
                score -= 0.4; flags.append(f"{name} near Sandhi")
    except Exception:
        pass
    return score, " · ".join(flags)


def _rahu_sandhi_trigger(dt: datetime):
    """Rahu in last 2° of a sign = maximum deception, worst Sandhi."""
    try:
        sid = _tropical_to_sidereal(_mean_lunar_node(dt))
        deg = sid % 30
        if deg >= 29.0 or deg <= 1.0:
            return -1.5, f"Rahu Sandhi ({deg:.1f}°) — maximum confusion"
        if deg >= 28.0 or deg <= 2.0:
            return -0.8, f"Rahu near Sandhi ({deg:.1f}°)"
    except Exception:
        pass
    return 0.0, ""


def _rahu_axis_shift_trigger(year: int, month: int):
    """Rahu changes signs during this month = karmic thematic reset."""
    import calendar as _c
    try:
        d1 = datetime(year, month, 1)
        d2 = datetime(year, month, _c.monthrange(year, month)[1])
        s1 = _lon_to_sign(_tropical_to_sidereal(_mean_lunar_node(d1)))
        s2 = _lon_to_sign(_tropical_to_sidereal(_mean_lunar_node(d2)))
        if s1 != s2:
            return -1.5, f"Rahu axis shift {s1}→{s2} — karmic sector reset"
    except Exception:
        pass
    return 0.0, ""


def _eclipse_trigger_score(dt: datetime, eclipses: list):
    """Mars or Sun within 3° of a recent eclipse degree activates dormant eclipse."""
    score, flags = 0.0, []
    try:
        sun_lon  = _planet_tropical_lon(ephem.Sun, dt)
        mars_lon = _planet_tropical_lon(ephem.Mars, dt)
        for ecl in eclipses:
            ecl_dt  = datetime(ecl["date"].year, ecl["date"].month, ecl["date"].day)
            ecl_lon = _planet_tropical_lon(ephem.Sun, ecl_dt)
            for pname, plon in [("Sun", sun_lon), ("Mars", mars_lon)]:
                sep = _angular_separation(plon, ecl_lon)
                if sep <= 2:
                    score -= 1.2; flags.append(f"{pname} on eclipse point")
                elif sep <= 4:
                    score -= 0.5; flags.append(f"{pname} near eclipse point")
    except Exception:
        pass
    return score, " · ".join(flags)


def _sarpa_dosha_trigger(eclipses: list, year: int, month: int):
    """Solar + lunar eclipse within 30 days of this month = Sarpa Dosha."""
    from datetime import date as _d2
    center = _d2(year, month, 15)
    nearby   = [e for e in eclipses if abs((e["date"] - center).days) <= 30]
    nearby60 = [e for e in eclipses if abs((e["date"] - center).days) <= 60]
    t30 = {e["type"] for e in nearby}
    t60 = {e["type"] for e in nearby60}
    if "Solar Eclipse" in t30 and "Lunar Eclipse" in t30:
        return -1.5, "Sarpa Dosha — dual eclipse window (solar + lunar ≤30 days)"
    if "Solar Eclipse" in t60 and "Lunar Eclipse" in t60:
        return -0.6, "Sarpa Dosha fading (eclipse season ≤60 days)"
    return 0.0, ""


def _jupiter_sign_change_trigger(year: int, month: int):
    """
    Jupiter changing signs this month = transitional volatility.
    Extra bearish if moving FROM a better sign TO a worse one
    (e.g., Cancer exaltation → Leo neutral = peak-reversal crash signal).
    """
    import calendar as _c
    try:
        dt1 = datetime(year, month, 1)
        dt2 = datetime(year, month, _c.monthrange(year, month)[1])
        sign1 = _lon_to_sign(_tropical_to_sidereal(_planet_tropical_lon(ephem.Jupiter, dt1)))
        sign2 = _lon_to_sign(_tropical_to_sidereal(_planet_tropical_lon(ephem.Jupiter, dt2)))
        if sign1 == sign2:
            return 0.0, ""
        # Score the direction of change
        from_score = _JUPITER_SIGN_TABLE.get(sign1, (0,))[0]
        to_score   = _JUPITER_SIGN_TABLE.get(sign2, (0,))[0]
        delta = to_score - from_score
        if delta < 0:
            return -2.0 * abs(delta) / 5, f"Jupiter {sign1}→{sign2} ingress — leaving {'exaltation' if from_score==3 else 'positive sign'} (peak reversal)"
        else:
            return +0.5, f"Jupiter {sign1}→{sign2} ingress — improving position"
    except Exception:
        pass
    return 0.0, ""


def _planet_retrograde_trigger(planet_cls, planet_name: str, dt: datetime):
    """Outer planet retrograde (Jupiter/Saturn/Mars) = caution signal."""
    try:
        if _is_retrograde(planet_cls, dt):
            scores = {"Jupiter": -0.6, "Saturn": -0.3, "Mars": -0.4}
            score = scores.get(planet_name, -0.3)
            return score, f"{planet_name} retrograde — {planet_name.lower()} themes inverted"
    except Exception:
        pass
    return 0.0, ""
