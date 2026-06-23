# Financial Astrology Scanner

**File:** `astro_scanner.py`  
**Dashboard:** `?scanner=astro`  
**Library:** `ephem` (astronomical ephemeris)  
**Sources:** W.D. Gann, Larry Pesavento & Sharon Smoleny, Bill Meridian, Rajeev Prakash

---

## What It Does

Calculates current and upcoming planetary events — moon phases, Mercury retrograde, planetary aspects, solar ingresses, and Gann time cycles — and maps them to historical market tendencies. Use as a timing overlay alongside technical signals, not as a standalone entry system.

---

## Signal Categories

### Moon Phase Signals
Based on the Bank of Scotland study and lunar cycle trading research:

| Phase | Signal | Strength |
|-------|--------|----------|
| New Moon | BULLISH | 3 — Buy signal; markets tend to rally into Full Moon |
| Full Moon | BEARISH | 3 — Sell/short signal; markets often peak near Full Moon |
| First Quarter | NEUTRAL | 1 — Caution zone; possible pause or consolidation |
| Last Quarter | NEUTRAL | 1 — Caution zone; markets often retest lows |

Moon phase signal fires within ±1 day of the exact phase.

### Mercury Retrograde
- Occurs ~3× per year, lasts ~3 weeks each time
- Associated with: miscommunication, false breakouts, increased volatility, reversals
- **Signal:** CAUTION during retrograde periods
- Source: Federal Reserve Bank of Atlanta geomagnetic/planetary research

### Planetary Aspects (Hard Aspects = Market Stress)

| Aspect | Planets | Historical Tendency |
|--------|---------|---------------------|
| Saturn–Uranus | Conjunction / Opposition / Square | Systemic stress, crash risk (confirmed: 2008, 2020) |
| Jupiter–Saturn | Conjunction | 20-year economic cycle turn |
| Mars–Saturn | Conjunction / Square | Sharp sell-off signal; short-term bearish |
| Jupiter–Uranus | Conjunction | Surprise bull run, tech breakout (positive) |
| Venus Retrograde | — | Sentiment extreme, commodity reversal |
| Mars Retrograde | — | Energy/military sector volatility |

Aspects fire when two planets are within `orb` degrees of exact alignment (default orb: 2°).

### Solar Ingress — Gann Seasonal Turns
Gann identified the four cardinal sign ingresses as major seasonal turning points:

| Ingress | Date (approx) | Gann's View |
|---------|---------------|-------------|
| Aries (0°) | ~March 20 | Spring equinox — major turn |
| Cancer (0°) | ~June 21 | Summer solstice — seasonal high/low |
| Libra (0°) | ~September 22 | Autumn equinox — major turn |
| Capricorn (0°) | ~December 21 | Winter solstice — seasonal turn |

### Gann Time Cycles
Counts days from significant pivot highs/lows (user-specified or auto-detected from SPY). Flags when a stock or the market is at a Gann time-cycle anniversary:

| Cycle | Days |
|-------|------|
| Gann Square | 45, 90, 144, 180, 270, 360 |
| Gann Annual | 365 |

A cluster of multiple Gann cycles arriving simultaneously amplifies the signal.

---

## Output

The scanner produces a **calendar view** showing:
- Today's active signals and their strength
- Upcoming events in the next 30/60/90 days
- Combined signal strength score (sum of all active signal strengths)
- Recommended bias (BULLISH / BEARISH / NEUTRAL / CAUTION)

---

## How to Run

```bash
# Show today's signals and upcoming 30-day calendar
python astro_scanner.py

# Look ahead 60 days
python astro_scanner.py --days 60

# Show all aspects regardless of orb
python astro_scanner.py --orb 5

# Focus on moon phases only
python astro_scanner.py --moon-only

# Include Gann cycles from a specific pivot date
python astro_scanner.py --pivot 2024-10-15

# Save calendar to CSV
python astro_scanner.py --output astro_calendar.csv
```

---

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--days` | 30 | Days ahead to compute calendar |
| `--orb` | 2.0 | Orb in degrees for aspect detection |
| `--moon-only` | false | Only show moon phase signals |
| `--pivot` | auto | Pivot date for Gann cycle calculation (YYYY-MM-DD) |
| `--output` | — | Save calendar to CSV |

---

## Important Caveat

Financial astrology is a supplementary timing tool — not a primary entry system. Use these signals to:
- Confirm or question a technical entry near a lunar phase or planetary aspect
- Avoid entering positions during Mercury Retrograde when false breakouts are more common
- Size down around Saturn–Uranus stress windows
- Look for reversals at solar ingress dates when price is extended

Never trade on astro signals alone without technical confirmation.
