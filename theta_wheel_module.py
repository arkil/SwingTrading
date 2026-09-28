"""
Theta Wheel — weekly cash-secured puts on trending, fundamentally strong names
==============================================================================
Implements the approach "Bob Dog" describes in the KINFO interview
"Selling Options For Weekly Income While Working Full-Time (The Strategy)"
(youtube.com/watch?v=-dB0LK6Lykc):

  * Read the WEEKLY chart (less noise than daily, faster than monthly).
  * Bullish regime + bullish momentum "stars"  -> sell cash-secured puts
    Mon/Tue that expire that Friday.  (He targets ~1%/week; see the strike
    rule below, changed after testing.)
  * Scale in: first star = a starter (~20% of the name's allocation),
    add more puts as more stars fire.  Never full size on the first trade.
  * Bearish stars / falling knife -> no new puts.  (He wheels assigned shares
    with covered calls at cost basis; the tested playbook closes ITM puts instead.)
  * Let winners expire or buy back for a penny.
  * Fundamentals pick WHAT (growth 20%+, profitable, cash); technicals pick WHEN.
  * The 200-week MA is the long-term "bottoming" zone to start accumulating.
  * Keep a short focus list (10–20 names) and learn how they move.
  * Check the macro tape first (SPY / QQQ / IWM weekly + fear).

His star indicators are private (invite-only), so the stars here are rebuilt
from the public tools he says anyone can use: weekly MACD cross, weekly RSI vs
50, and price above/below all the moving averages.  The gold/gray regime is
price vs a rising/falling 20-week EMA.

Changes after a real-option-print backtest (~/workplace/strategies/theta_wheel,
RESULTS.md): strike >= 5% OTM paying >= 0.5% (not "whatever pays 1%"), skip thin
weeks, skip earnings weeks, take ITM puts off instead of wheeling, no SPY size cap,
signals from completed weeks only.

Data: yfinance (free, delayed).  Cash-secured only — collateral comes out of
cash, never margin.  Educational — not advice.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

_DIR = os.path.join(os.path.dirname(__file__), "Data", "theta_wheel")
_HIST_DIR = os.path.join(_DIR, "weekly")
_FUND_DIR = os.path.join(_DIR, "fund")
_JSON = os.path.join(_DIR, "latest.json")
_CSV = os.path.join(_DIR, "latest.csv")
_WATCHLIST = os.path.join(_DIR, "watchlist.json")

# Names from the interview (his "core" list is ~10–20 stocks he trades over
# and over). Edit on the page — saved to Data/theta_wheel/watchlist.json.
DEFAULT_WATCHLIST = [
    "IGV", "CRWV", "IREN", "CIFR", "GLXY", "EOSE", "ASTS", "RDW", "HIMS",
    "MSTR", "COIN", "MARA", "HOOD", "OPEN", "TSLA", "BE", "SNDK", "TE", "ADBE",
]
MACRO_TICKERS = ["SPY", "QQQ", "IWM"]
_ETFS = {"SPY", "QQQ", "IWM", "DIA", "IGV", "SMH", "SOXX", "XLK", "XLF", "XLE", "ARKK",
         "IBIT", "GLD", "SLV", "TLT", "XBI", "KRE"}

# Strike rule from the real-print backtest (strategies/theta_wheel/RESULTS.md): puts
# closer than 5% OTM lost money net on average; the most-OTM strike paying >= 0.5% of
# collateral, but never closer than 5% OTM, was the one setting that held up out of sample.
MIN_OTM = 0.05              # strike at least 5% below spot
MIN_YIELD = 0.005           # premium >= 0.5% of collateral for the week
SCALE_TIERS = {0: 0.0, 1: 0.20, 2: 0.50, 3: 1.00}   # bull stars -> fraction of the name's allocation


# --------------------------------------------------------------------------- #
# Watchlist
# --------------------------------------------------------------------------- #
def load_watchlist() -> list:
    try:
        with open(_WATCHLIST) as f:
            wl = [str(s).strip().upper() for s in json.load(f) if str(s).strip()]
            if wl:
                return wl
    except Exception:
        pass
    return list(DEFAULT_WATCHLIST)


def save_watchlist(tickers) -> list:
    os.makedirs(_DIR, exist_ok=True)
    wl, seen = [], set()
    for t in tickers:
        t = str(t).strip().upper()
        if t and t not in seen:
            seen.add(t)
            wl.append(t)
    with open(_WATCHLIST, "w") as f:
        json.dump(wl, f)
    return wl


# --------------------------------------------------------------------------- #
# Data (disk-cached so the dashboard never hits yfinance on a page view)
# --------------------------------------------------------------------------- #
def weekly_history(symbol: str, max_age_h: float = 12.0) -> pd.DataFrame:
    """~6y of weekly OHLCV (enough for the 200-week MA). Parquet-cached."""
    import yfinance as yf
    from earnings_iv_module import _retry
    os.makedirs(_HIST_DIR, exist_ok=True)
    p = os.path.join(_HIST_DIR, f"{symbol.upper()}.parquet")
    if os.path.exists(p) and (time.time() - os.path.getmtime(p)) < max_age_h * 3600:
        try:
            df = pd.read_parquet(p)
            if not df.empty:
                return df
        except Exception:
            pass
    df = _retry(lambda: yf.Ticker(symbol).history(period="6y", interval="1wk", auto_adjust=False),
                label=f"{symbol} weekly")
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep].dropna(subset=["Close"]).copy()
    try:
        df.to_parquet(p)
    except Exception:
        pass
    return df


def fundamentals(symbol: str, max_age_h: float = 24 * 7) -> dict:
    """Growth / profitability / cash snapshot from yfinance .info. JSON-cached ~1 week."""
    import yfinance as yf
    from earnings_iv_module import _retry
    os.makedirs(_FUND_DIR, exist_ok=True)
    p = os.path.join(_FUND_DIR, f"{symbol.upper()}.json")
    if os.path.exists(p) and (time.time() - os.path.getmtime(p)) < max_age_h * 3600:
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    try:
        info = _retry(lambda: yf.Ticker(symbol).info, tries=3, label=f"{symbol} info") or {}
    except Exception:
        info = {}

    def _g(k):
        v = info.get(k)
        try:
            v = float(v)
            return v if math.isfinite(v) else None
        except Exception:
            return None

    out = {
        "name": info.get("shortName") or symbol,
        "quote_type": info.get("quoteType"),
        "sector": info.get("sector"),
        "market_cap": _g("marketCap"),
        "rev_growth": _g("revenueGrowth"),          # yoy, latest quarter
        "profit_margin": _g("profitMargins"),
        "trailing_eps": _g("trailingEps"),
        "total_cash": _g("totalCash"),
        "trailing_pe": _g("trailingPE"),
    }
    if info:
        with open(p, "w") as f:
            json.dump(out, f)
    return out


def fundamentals_grade(symbol: str, f: dict) -> tuple:
    """(grade, notes). PASS = growing 20%+ AND profitable; PARTIAL = one of them."""
    if symbol in _ETFS or (f.get("quote_type") or "").upper() == "ETF":
        return "ETF", ["sector/index ETF — no single-company fundamentals"]
    g, pm, eps = f.get("rev_growth"), f.get("profit_margin"), f.get("trailing_eps")
    growing = g is not None and g >= 0.20
    profitable = (pm is not None and pm > 0) or (eps is not None and eps > 0)
    notes = []
    notes.append(f"revenue growth {g*100:+.0f}% yoy" if g is not None else "revenue growth n/a")
    notes.append("profitable" if profitable else "not profitable")
    cash, mc = f.get("total_cash"), f.get("market_cap")
    if cash and mc:
        notes.append(f"cash {cash/1e9:.1f}B = {cash/mc*100:.0f}% of mkt cap")
    if g is None and pm is None and eps is None:
        return "N/A", notes
    grade = "PASS" if growing and profitable else "PARTIAL" if (growing or profitable) else "FAIL"
    return grade, notes


# --------------------------------------------------------------------------- #
# Weekly signal engine (pure, no I/O)
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def weekly_signals(w: pd.DataFrame) -> pd.DataFrame:
    """Adds regime + star columns to a weekly OHLC frame.

    regime   BULL (gold)  close > 20w EMA and the EMA is rising
             BEAR (gray)  close < 20w EMA and the EMA is falling
             NEUTRAL      otherwise
    bull stars (0–3): weekly MACD > signal · RSI(14) > 50 · close above 10w, 20w AND 40w MAs
    bear stars (0–3): the mirror image
    """
    c = w["Close"]
    out = w.copy()
    out["ema10"] = c.ewm(span=10, adjust=False).mean()
    out["ema20"] = c.ewm(span=20, adjust=False).mean()
    out["sma40"] = c.rolling(40).mean()                  # ≈ 200-day
    out["sma200w"] = c.rolling(200).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    out["macd"], out["macd_sig"] = macd, sig
    out["rsi"] = _rsi(c)

    slope = out["ema20"].diff(3)
    out["regime"] = np.where((c > out["ema20"]) & (slope > 0), "BULL",
                             np.where((c < out["ema20"]) & (slope < 0), "BEAR", "NEUTRAL"))
    ma40 = out["sma40"].fillna(out["ema20"])
    above_all = (c > out["ema10"]) & (c > out["ema20"]) & (c > ma40)
    below_all = (c < out["ema10"]) & (c < out["ema20"]) & (c < ma40)
    out["bull_stars"] = ((macd > sig).astype(int) + (out["rsi"] > 50).astype(int)
                         + above_all.astype(int))
    out["bear_stars"] = ((macd < sig).astype(int) + (out["rsi"] < 50).astype(int)
                         + below_all.astype(int))
    # A "star" prints on the week a signal count steps UP (a new momentum point).
    out["blue_star"] = out["bull_stars"] > out["bull_stars"].shift(1).fillna(0)
    out["purple_star"] = out["bear_stars"] > out["bear_stars"].shift(1).fillna(0)
    return out


def _weeks_since(mask: pd.Series) -> int:
    idx = np.flatnonzero(mask.to_numpy())
    return int(len(mask) - 1 - idx[-1]) if idx.size else 999


def completed_weeks(w: pd.DataFrame, today: date = None) -> pd.DataFrame:
    """Drop the in-progress week so signals use finished weekly bars only (his Sunday read,
    and what the backtest used). A Mon–Fri run drops this week's partial bar; a weekend
    run keeps it because the week is done."""
    today = today or date.today()
    if w is None or w.empty or today.weekday() >= 5:
        return w
    idx = pd.DatetimeIndex(w.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    last = idx[-1].normalize()
    # yfinance labels a weekly bar by its Monday (or the preceding Sunday once tz-stripped)
    last_monday = last + pd.Timedelta(days=(-last.weekday()) if last.weekday() <= 4 else (7 - last.weekday()))
    this_monday = pd.Timestamp(today) - pd.Timedelta(days=today.weekday())
    return w.iloc[:-1] if last_monday >= this_monday else w


def technical_read(w: pd.DataFrame) -> dict:
    s = weekly_signals(w)
    last = s.iloc[-1]
    close = float(last["Close"])
    sma200w = float(last["sma200w"]) if pd.notna(last["sma200w"]) else None
    dist200 = (close / sma200w - 1) * 100 if sma200w else None
    hi52 = float(s["High"].tail(52).max())
    # when did the current bullish run start (first week bull_stars >= 1 after a 0)?
    bull_run = 0
    for v in reversed(s["bull_stars"].tolist()):
        if v >= 1:
            bull_run += 1
        else:
            break
    return {
        "close": close,
        "regime": str(last["regime"]),
        "bull_stars": int(last["bull_stars"]),
        "bear_stars": int(last["bear_stars"]),
        "rsi": float(last["rsi"]),
        "macd_up": bool(last["macd"] > last["macd_sig"]),
        "sma200w": sma200w,
        "dist_200w_pct": dist200,
        "near_200w": dist200 is not None and -5 <= dist200 <= 12,
        "off_52w_high_pct": (close / hi52 - 1) * 100 if hi52 else None,
        "weeks_since_blue": _weeks_since(s["blue_star"]),
        "weeks_since_purple": _weeks_since(s["purple_star"]),
        "bull_run_weeks": bull_run,
        "ret_4w_pct": (close / float(s["Close"].iloc[-5]) - 1) * 100 if len(s) > 5 else None,
    }


def chart_series(w: pd.DataFrame, n: int = 156) -> dict:
    """Compact JSON-safe weekly series for the page's chart."""
    s = weekly_signals(w).tail(n)
    def _col(name, nd=2):
        return [None if pd.isna(v) else round(float(v), nd) for v in s[name]]
    return {
        "t": [d.strftime("%Y-%m-%d") for d in s.index],
        "o": _col("Open"), "h": _col("High"), "l": _col("Low"), "c": _col("Close"),
        "ema20": _col("ema20"), "sma40": _col("sma40"), "sma200w": _col("sma200w"),
        "regime": s["regime"].tolist(),
        "blue": [bool(x) for x in s["blue_star"]],
        "purple": [bool(x) for x in s["purple_star"]],
    }


# --------------------------------------------------------------------------- #
# Options: this Friday's cash-secured put (and a covered call at ~spot)
# --------------------------------------------------------------------------- #
def target_friday_expiry(expiries: list, today: date = None) -> tuple:
    """The Friday he'd sell into: this week's if it's Mon–Wed, else next week's
    (Thu/Fri entries have almost no time value left). Returns (expiry, dte)."""
    today = today or date.today()
    best = None
    for e in expiries:
        d = pd.Timestamp(e).date()
        dte = (d - today).days
        if d.weekday() != 4 or dte < 0:
            continue
        min_dte = 2 if today.weekday() <= 2 else 5
        if dte >= min_dte and (best is None or dte < best[1]):
            best = (e, dte)
    if best is None:                             # no Friday listed — nearest expiry 2–12 DTE
        for e in expiries:
            dte = (pd.Timestamp(e).date() - today).days
            if 2 <= dte <= 12 and (best is None or dte < best[1]):
                best = (e, dte)
    return best or (None, None)


def pick_put(puts: pd.DataFrame, spot: float, iv: float, dte: int,
             min_otm: float = MIN_OTM, min_yield: float = MIN_YIELD) -> dict:
    """Most-OTM put that still pays >= `min_yield` of collateral (strike x 100), never
    closer than `min_otm` below spot. Returns (put, why): put is {} and `why` says why when nothing qualifies
    when nothing qualifies — that week is a skip, not a forced trade."""
    from earnings_iv_module import _quote, bs_delta
    T = max(dte, 1) / 365.0
    cands, best_near = [], None
    for K in sorted(puts["strike"].unique()):
        K = float(K)
        if K > spot or K < spot * 0.60:
            continue
        mid, src, liquid, info = _quote(puts, K, spot, iv, T, is_call=False)
        if src != "quote" or mid <= 0:
            continue
        row = puts[puts["strike"] == K].iloc[0]
        kiv = float(row.get("impliedVolatility") or 0)
        kiv = kiv if kiv > 0.05 else iv
        dlt = abs(bs_delta(spot, K, T, kiv, is_call=False))
        c = {
            "strike": K, "mid": mid, "bid": float(row.get("bid") or 0),
            "yield": mid / K, "delta": dlt if math.isfinite(dlt) else None,
            "otm_pct": (1 - K / spot) * 100, "breakeven": K - mid, "collateral": K * 100,
            "liquid": liquid, "oi": info["oi"], "vol": info["vol"],
        }
        if K <= spot * (1 - min_otm):
            cands.append(c)
        elif best_near is None or c["yield"] > best_near["yield"]:
            best_near = c
    hit = [c for c in cands if c["yield"] >= min_yield]
    hit = [c for c in hit if c["liquid"]] or hit
    if hit:
        return min(hit, key=lambda c: c["strike"]), None
    if cands:
        top = max(cands, key=lambda c: c["yield"])
        why = (f"premium too thin: best put ≥{min_otm*100:.0f}% OTM pays "
               f"{top['yield']*100:.2f}%/wk (< {min_yield*100:.1f}%)")
    elif best_near:
        why = (f"premium too thin: nothing ≥{min_otm*100:.0f}% OTM is quoted "
               f"(nearest pays {best_near['yield']*100:.2f}% at {best_near['otm_pct']:.1f}% OTM)")
    else:
        why = "no live bid/ask on the puts (pre-market or feed outage) — re-run after 10:00 ET"
    return {}, why


def atm_call(calls: pd.DataFrame, spot: float, iv: float, dte: int) -> dict:
    """Covered call at ~spot — his exit: 'buy at 100, sell the 100 call, take the 3%'."""
    from earnings_iv_module import _quote, _nearest
    if calls is None or calls.empty:
        return {}
    K = _nearest(calls, spot)
    mid, src, liquid, _ = _quote(calls, K, spot, iv, max(dte, 1) / 365.0, is_call=True)
    if src != "quote" or mid <= 0:
        return {}
    return {"strike": K, "mid": mid, "yield": mid / spot, "liquid": liquid}


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #
def decide(tech: dict, fund_grade: str, earnings_in_window: bool) -> tuple:
    """(action, tier_fraction, reasons)."""
    reasons = []
    bs, br, reg = tech["bull_stars"], tech["bear_stars"], tech["regime"]
    if earnings_in_window:
        return "SKIP — EARNINGS", 0.0, ["earnings before Friday's expiry — skip this week's put"]
    if reg == "BEAR" or br >= 2:
        why = f"weekly regime {reg}, {br}/3 bearish stars"
        if tech.get("near_200w"):
            return ("WATCH — 200W ZONE", 0.0,
                    [why + " — but sitting on the 200-week MA; wait for the first blue star"])
        return "AVOID — FALLING KNIFE", 0.0, [why + " — no new puts on this name"]
    if fund_grade == "FAIL":
        return "WATCH — WEAK FUNDAMENTALS", 0.0, ["technicals OK but not growing and not profitable"]
    if bs == 0 or (reg == "NEUTRAL" and bs < 2):
        return "WATCH", 0.0, [f"no bullish momentum yet ({bs}/3 stars, regime {reg})"]
    tier = SCALE_TIERS[bs]
    if fund_grade in ("PARTIAL", "N/A"):
        tier = min(tier, SCALE_TIERS[2])
        reasons.append(f"fundamentals {fund_grade} — capped at half size")
    label = ("SELL PUT — STARTER" if tier <= SCALE_TIERS[1]
             else "SELL PUT — ADD" if tier <= SCALE_TIERS[2] else "SELL PUT — FULL")
    reasons.insert(0, f"weekly regime {reg}, {bs}/3 bullish stars")
    return label, tier, reasons


def macro_read() -> dict:
    """Weekly regime for SPY / QQQ / IWM + VIX from the shared regime source."""
    out = {"indexes": {}, "vix": None, "vix_rank": None, "label": None}
    for sym in MACRO_TICKERS:
        try:
            t = technical_read(completed_weeks(weekly_history(sym)))
            out["indexes"][sym] = {k: t[k] for k in ("regime", "bull_stars", "bear_stars", "close", "rsi")}
        except Exception as e:  # noqa: BLE001
            out["indexes"][sym] = {"error": str(e)}
    try:
        from market_context import get_regime_summary
        s = get_regime_summary()
        v = s.get("vix", {})
        out["vix"], out["vix_rank"], out["label"] = v.get("level"), v.get("rank_1y"), s.get("label")
    except Exception:
        pass
    spy = out["indexes"].get("SPY", {})
    out["bull"] = spy.get("regime") == "BULL" or (spy.get("bull_stars", 0) >= 2 and spy.get("regime") != "BEAR")
    return out


def analyze(symbol: str, today: date = None) -> dict:
    """Full read for one ticker. Pure (no st.*); never raises."""
    import yfinance as yf
    from earnings_iv_module import _retry, _chain_impl, _atm_iv
    from earnings_cache import cached_earnings_dates
    symbol = symbol.strip().upper()
    today = today or date.today()
    try:
        w = weekly_history(symbol)
    except Exception as e:  # noqa: BLE001
        return {"symbol": symbol, "error": f"history: {e}"}
    if w is None or len(w) < 30:
        return {"symbol": symbol, "error": "not enough weekly history"}
    w = completed_weeks(w, today)

    tech = technical_read(w)
    f = fundamentals(symbol)
    grade, fnotes = fundamentals_grade(symbol, f)
    # live price for strike selection — tech["close"] is last completed week's close
    from earnings_iv_module import _fresh_spot
    spot = _fresh_spot(symbol) or tech["close"]

    put, call, exp, dte, iv, opt_err = {}, {}, None, None, None, None
    try:
        expiries = list(_retry(lambda: yf.Ticker(symbol).options, label=f"{symbol} options"))
        exp, dte = target_friday_expiry(expiries, today)
        if exp is None:
            opt_err = "no weekly expiry in the next 2 weeks"
        else:
            calls, puts = _chain_impl(symbol, exp)
            iv, _ = _atm_iv(calls, puts, spot)
            if iv is None:
                opt_err = "IV feed empty (pre-market?)"
            else:
                put, why = pick_put(puts, spot, iv, dte)
                call = atm_call(calls, spot, iv, dte)
                if not put:
                    opt_err = why
    except Exception as e:  # noqa: BLE001
        opt_err = f"options: {type(e).__name__}: {e}"

    earn_next, earn_in = None, False
    if symbol not in _ETFS:
        try:
            now = pd.Timestamp.now(tz="UTC")
            fut = [t for t in cached_earnings_dates(symbol) if t >= now - pd.Timedelta(days=1)]
            if fut:
                earn_next = min(fut)
                if exp is not None:
                    earn_in = earn_next.date() <= pd.Timestamp(exp).date()
        except Exception:
            pass

    action, tier, reasons = decide(tech, grade, earn_in)
    if action.startswith("SELL") and not put:
        no_quotes = (opt_err or "").startswith("no live bid/ask") or "IV feed" in (opt_err or "")
        action, tier = ("WATCH — NO QUOTES" if no_quotes else "WATCH — PREMIUM THIN"), 0.0
        reasons.append(opt_err or "no qualifying put")
    if tech["near_200w"] and action.startswith("SELL"):
        reasons.append("within reach of the 200-week MA — his favourite accumulation zone")
    if put and not put.get("liquid"):
        reasons.append("thin chain — work a limit at mid, don't hit the bid")

    return {
        "symbol": symbol, "error": None, "name": f.get("name"),
        "spot": spot, "tech": tech, "fund": f, "fund_grade": grade, "fund_notes": fnotes,
        "exp": exp, "dte": dte, "iv": iv, "put": put, "call": call, "opt_err": opt_err,
        "earn_next": earn_next.date().isoformat() if earn_next is not None else None,
        "earn_in_window": earn_in, "action": action, "tier": tier, "reasons": reasons,
        "chart": chart_series(w),
    }


def scan_row(r: dict) -> dict:
    t, p, c = r["tech"], r.get("put") or {}, r.get("call") or {}
    def _n(x, nd=1):
        return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)
    return {
        "Ticker": r["symbol"],
        "Action": r["action"],
        "Regime": t["regime"],
        "Bull★": t["bull_stars"], "Bear★": t["bear_stars"],
        "Wks since ★": t["weeks_since_blue"] if t["weeks_since_blue"] < 999 else None,
        "Fund": r["fund_grade"],
        "Rev gr %": _n((r["fund"].get("rev_growth") or 0) * 100, 0) if r["fund"].get("rev_growth") is not None else None,
        "Spot": _n(r["spot"], 2),
        "vs 200W %": _n(t["dist_200w_pct"]),
        "Expiry": r.get("exp"),
        "Put K": p.get("strike"),
        "Put mid": _n(p.get("mid"), 2),
        "Yield %/wk": _n(p.get("yield", 0) * 100, 2) if p else None,
        "OTM %": _n(p.get("otm_pct")),
        "Δ": _n(p.get("delta"), 2),

        "CC yield %": _n(c.get("yield", 0) * 100, 2) if c else None,
        "Earnings": r.get("earn_next"),
        "Tier": int(round(r["tier"] * 100)),
    }


_ACTION_ORDER = ["SELL PUT — FULL", "SELL PUT — ADD", "SELL PUT — STARTER", "WATCH — 200W ZONE",
                 "WATCH — NO QUOTES", "WATCH", "WATCH — PREMIUM THIN", "WATCH — WEAK FUNDAMENTALS", "SKIP — EARNINGS", "AVOID — FALLING KNIFE"]


def build_scan(tickers=None, progress=None, today: date = None) -> dict:
    """Scan the watchlist and persist to Data/theta_wheel/latest.json."""
    from earnings_prefetch import _run_pool, _clean, _safe_write
    os.makedirs(_DIR, exist_ok=True)
    t0 = time.time()
    names = tickers or load_watchlist()
    macro = macro_read()
    print(f"[theta] macro: {({k: v.get('regime') for k, v in macro['indexes'].items()})} "
          f"VIX {macro.get('vix')} → {'bull' if macro['bull'] else 'not bull'}")
    rows, details, skipped = [], {}, []
    for sym, r in _run_pool(lambda s: analyze(s, today=today),
                            names, progress=progress):
        if not isinstance(r, dict) or r.get("error"):
            skipped.append(f"{sym} ({(r or {}).get('error', 'failed')})")
            print(f"  [skip] {sym}: {(r or {}).get('error')}")
            continue
        rows.append(scan_row(r))
        details[sym] = _clean(r)
    rank = {a: i for i, a in enumerate(_ACTION_ORDER)}
    rows.sort(key=lambda x: (rank.get(x["Action"], 99), -(x["Yield %/wk"] or 0)))
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_names": len(names), "n_ok": len(rows),
        "n_sell": sum(1 for x in rows if x["Action"].startswith("SELL")),
        "skipped": skipped, "macro": _clean(macro),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    meta = _safe_write(_JSON, _CSV, rows, details, meta, min_keep=3, max_age_h=24 * 7)
    print(f"[theta] done in {meta.get('elapsed_sec')}s — {meta.get('n_ok')} ok, "
          f"{meta.get('n_sell')} sell-put, {len(skipped)} skipped → {_JSON}")
    return meta


def load_latest():
    try:
        with open(_JSON) as f:
            blob = json.load(f)
    except Exception:
        return None, None, None
    return pd.DataFrame(blob.get("rows", [])), blob.get("details", {}), blob.get("meta", {})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Theta Wheel weekly CSP scan")
    ap.add_argument("--tickers", nargs="*", help="override the saved watchlist")
    a = ap.parse_args()
    build_scan(a.tickers or None)


# ═════════════════════════════════════════════════════════════════════════════
# Streamlit page (dashboard reads the cache; never scans on a page view)
# ═════════════════════════════════════════════════════════════════════════════
_ACTION_ICON = {"SELL": "🟢", "WATCH": "🟡", "SKIP": "🟠", "AVOID": "🔴"}
_REGIME_ICON = {"BULL": "🟨", "BEAR": "⬜", "NEUTRAL": "▫️"}

_PLAYBOOK = """
**Weekly routine**
1. **Weekend** — chart the focus list on the **weekly** (finished weeks only). Glance at SPY / QQQ / IWM and VIX for context.
2. **Monday / Tuesday** — sell a cash-secured put that expires **this Friday** on names with a bullish weekly regime and blue stars.
3. **Strike** — the furthest-OTM put that still pays **≥ 0.5% of collateral**, and never closer than **5% below the stock price**.
   If nothing qualifies, **skip the name this week** — don't reach for a closer strike to hit a number.
4. **Wed–Fri** — let theta work. Let winners expire, or buy them back for **$0.01–0.05**.
5. **Friday, put in the money** — buy it back (or sell assigned shares Monday) and take the loss. Don't bag-hold and sell calls at cost basis.

**Scaling in (never full size on the first trade)**
- 1 bullish star → starter, **~20%** of that name's allocation
- 2 stars → add, up to **~50%**
- 3 stars → full allocation
- Fundamentals only PARTIAL → half size at most.

**Skip**
- Bearish regime or 2+ purple stars → no new puts (don't sell puts into a falling knife).
- Earnings before Friday's expiry.
- Premium too thin at ≥ 5% OTM.

**What to trade**
- A short focus list (10–20 names). Volatile names are the ones that pay at 5% OTM — and can drop 30%+ in a week, so size so one bad week is survivable.
- Fundamentals say WHAT (revenue growth 20%+, profitable, real cash). Technicals say WHEN.
- **Cash-secured only**: collateral = strike × 100 per contract, from cash, never margin.

**Changed from the video** — he targets ~1%/week on whatever strike pays it, wheels assigned shares with covered calls at cost basis,
and caps size when SPY isn't bullish. Testing on real option prints (2024–26) didn't support those three, so the rules above replace them.
"""


def _chart(r: dict, dark: bool):
    import plotly.graph_objects as go
    ch = r.get("chart") or {}
    if not ch.get("t"):
        return None
    t = ch["t"]
    fig = go.Figure()
    for reg, color in (("BULL", "#d4a017"), ("NEUTRAL", "#8c8c8c"), ("BEAR", "#5a5a5a")):
        idx = [i for i, g in enumerate(ch["regime"]) if g == reg]
        if not idx:
            continue
        fig.add_trace(go.Candlestick(
            x=[t[i] for i in idx], open=[ch["o"][i] for i in idx], high=[ch["h"][i] for i in idx],
            low=[ch["l"][i] for i in idx], close=[ch["c"][i] for i in idx],
            name={"BULL": "bull regime", "NEUTRAL": "neutral", "BEAR": "bear regime"}[reg],
            increasing=dict(line=dict(color=color), fillcolor=color),
            decreasing=dict(line=dict(color=color), fillcolor="rgba(0,0,0,0)")))
    for key, name, color, width in (("ema20", "20w EMA", "#4c9be8", 1.2), ("sma40", "40w (≈200d)", "#a970ff", 1.2),
                                    ("sma200w", "200w MA", "#e0b000", 3.5)):
        if any(v is not None for v in ch.get(key, [])):
            fig.add_trace(go.Scatter(x=t, y=ch[key], name=name, mode="lines",
                                     line=dict(color=color, width=width)))
    bx = [t[i] for i, b in enumerate(ch["blue"]) if b]
    by = [ch["l"][i] * 0.94 for i, b in enumerate(ch["blue"]) if b]
    px = [t[i] for i, b in enumerate(ch["purple"]) if b]
    py = [ch["h"][i] * 1.06 for i, b in enumerate(ch["purple"]) if b]
    fig.add_trace(go.Scatter(x=bx, y=by, mode="markers", name="blue ★ (bullish step)",
                             marker=dict(symbol="star", size=11, color="#1f77ff")))
    fig.add_trace(go.Scatter(x=px, y=py, mode="markers", name="purple ★ (bearish step)",
                             marker=dict(symbol="star", size=11, color="#b04cff")))
    put = r.get("put") or {}
    if put.get("strike"):
        fig.add_hline(y=put["strike"], line=dict(color="#2ca02c", dash="dot"),
                      annotation_text=f"put {put['strike']:g}")
    fig.update_layout(template="plotly_dark" if dark else "plotly_white", height=380,
                      margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.08, font=dict(size=10)),
                      title=dict(text=f"{r['symbol']} — weekly", font=dict(size=13)))
    return fig


def _ticket(r: dict, cash: float, per_name_pct: float):
    import streamlit as st
    from earnings_iv_module import _md, _cap
    t, p, c, fnd = r["tech"], r.get("put") or {}, r.get("call") or {}, r.get("fund") or {}
    for reason in r.get("reasons", []):
        _md("- " + reason)
    cL, cR = st.columns([1, 1.5])
    with cL:
        if p and r["action"].startswith("SELL"):
            alloc = cash * per_name_pct / 100 * r["tier"]
            n = int(alloc // p["collateral"]) if p["collateral"] > 0 else 0
            _md(f"**Sell to open** {n if n else '—'}× `{r['symbol']} {r['exp']} {p['strike']:g} P` "
                f"— limit ~${p['mid']:.2f} (bid ${p['bid']:.2f})")
            _md(f"Premium **${p['mid']*100:,.0f}/contract** = {p['yield']*100:.2f}% of "
                f"${p['collateral']:,.0f} collateral · Δ {p['delta'] or 0:.2f} · {p['otm_pct']:.1f}% OTM · "
                f"breakeven ${p['breakeven']:.2f}")
            _md(f"Size: {r['tier']*100:.0f}% tier of a {per_name_pct:.0f}%-of-cash allocation = "
                f"${alloc:,.0f} → **{n} contract(s)**, ${n*p['collateral']:,.0f} cash reserved")
            if n == 0:
                _cap(f"⚠ one contract needs ${p['collateral']:,.0f} cash — more than this tier allows. "
                     f"Skip it, or raise the per-name %, knowing you'd be over-allocated.")
            _cap(f"OI {p['oi']:,} · vol {p['vol']:,}" + ("" if p.get("liquid") else " · ⚠ thin — use a limit at mid"))
        elif p:
            _md(f"Reference put (not a signal): `{p['strike']:g} P` ~${p['mid']:.2f} → "
                f"{p['yield']*100:.2f}%/wk, {p['otm_pct']:.1f}% OTM")
        elif r.get("opt_err"):
            _cap(f"options: {r['opt_err']}")
        if c:
            _md(f"**Holding shares you want to exit?** ~ATM covered call `{c['strike']:g} C` ~${c['mid']:.2f} "
                f"= {c['yield']*100:.2f}% this week")
        st.markdown("---")
        _md(f"**Fundamentals: {r['fund_grade']}** — " + "; ".join(r.get("fund_notes", [])))
        _md(f"Weekly: regime **{t['regime']}**, {t['bull_stars']}/3 bull · {t['bear_stars']}/3 bear, "
            f"RSI {t['rsi']:.0f}, MACD {'↑' if t['macd_up'] else '↓'}"
            + (f", {t['dist_200w_pct']:+.0f}% vs 200-week MA" if t.get("dist_200w_pct") is not None else "")
            + (f", {t['off_52w_high_pct']:+.0f}% from 52w high" if t.get("off_52w_high_pct") is not None else ""))
        if r.get("earn_next"):
            _cap(f"next earnings {r['earn_next']}" + (" — INSIDE this expiry" if r.get("earn_in_window") else ""))
    with cR:
        fig = _chart(r, st.session_state.get("dark_mode", False))
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"tw_{r['symbol']}")


_PAPER_LIVE = ("open", "pending", "assigned", "selling_shares")


def _render_paper():
    """Forward test on the Alpaca paper account (theta_wheel_paper.py) — real fills, read from its ledger."""
    import streamlit as st
    p = os.path.join(_DIR, "paper_ledger.json")
    try:
        with open(p) as f:
            led = json.load(f)
    except Exception:
        led = []
    st.caption("Real paper-account fills from `theta_wheel_paper.py` — Mon/Tue 10:20 ET entries, "
               "15:40 ET management. Cash-secured, capped at the bot's cash sleeve.")
    if not led:
        st.info("No paper trades yet. Dry run: `.venv/bin/python theta_wheel_paper.py open --dry-run`. "
                "The schedule is off until you install `disabled_launch_agents/com.swingtrading.theta-paper.plist`.")
        return
    df = pd.DataFrame(led)
    done = df[df["status"].isin(["closed", "expired"])]
    live = df[df["status"].isin(_PAPER_LIVE)]
    c = st.columns(4)
    c[0].metric("Closed trades", len(done))
    c[1].metric("Realized P&L", f"${done.get('pnl', pd.Series(dtype=float)).fillna(0).sum():,.0f}")
    c[2].metric("Wins", int((done.get("pnl", pd.Series(dtype=float)).fillna(0) > 0).sum()))
    qty = [r.get("qty") or r.get("qty_req") or 0 for r in led if r["status"] in _PAPER_LIVE]
    coll = float((live["strike"].to_numpy() * 100 * qty).sum()) if len(live) else 0.0
    c[3].metric("Open collateral", f"${coll:,.0f}")
    cols = [k for k in ("opened_on", "occ", "status", "qty", "qty_req", "limit", "credit", "debit", "pnl",
                        "otm", "yield", "spot_at_open", "closed_on") if k in df.columns]
    st.dataframe(df[cols].iloc[::-1], use_container_width=True, hide_index=True)


def render_theta_wheel():
    import streamlit as st
    from earnings_iv_module import _built_str

    st.markdown(
        "<div class='scanner-header'><span style='font-size:2rem'>🎡</span>"
        "<span class='scanner-title'>Theta Wheel — Weekly CSPs</span></div>"
        "<div class='scanner-desc'>Sell cash-secured puts Mon/Tue that expire Friday, only on "
        "fundamentally strong names in a <b>bullish weekly trend</b>, scaling in as bullish stars "
        "stack up. Strike ≥ 5% OTM paying ≥ 0.5% of collateral, else skip the week. "
        "After the KINFO interview with “Bob Dog”. Free delayed data. Educational, not advice.</div>",
        unsafe_allow_html=True)

    tab_scan, tab_paper, tab_rules, tab_list = st.tabs(["📡 This week", "📒 Paper trades", "📘 Playbook", "📝 Focus list"])

    with tab_paper:
        _render_paper()

    with tab_list:
        wl = load_watchlist()
        st.caption("Keep it short — 10–20 names you know well. One ticker per line or comma-separated.")
        txt = st.text_area("Focus list", "\n".join(wl), height=260, key="tw_wl")
        c1, c2 = st.columns(2)
        if c1.button("💾 Save list", use_container_width=True):
            new = save_watchlist(x for x in txt.replace(",", "\n").split())
            st.success(f"Saved {len(new)} tickers — hit Refresh on the 'This week' tab.")
        if c2.button("↺ Reset to interview defaults", use_container_width=True):
            save_watchlist(DEFAULT_WATCHLIST)
            st.rerun()

    with tab_rules:
        st.markdown(_PLAYBOOK.replace("$", "\\$"))
        st.caption("Stars here are rebuilt from public indicators (his own are private): "
                   "bull ★ = weekly MACD > signal · weekly RSI > 50 · close above the 10w, 20w and 40w MAs. "
                   "Gold/gray regime = close vs a rising/falling 20-week EMA.")

    with tab_scan:
        df, details, meta = load_latest()
        top = st.columns([2.6, 0.9, 1.0, 0.9])
        if meta:
            top[0].caption(f"Last scan {_built_str(meta.get('built_at'))} · {meta.get('n_ok', 0)}/"
                           f"{meta.get('n_names', '?')} names · {meta.get('n_sell', 0)} put-sell signals")
        else:
            top[0].warning("No scan yet — click **Refresh** (during market hours; the IV feed is flat pre-market).")
        refresh = top[1].button("🔄 Refresh", use_container_width=True, help="Live scan of the focus list (~15–60s).")
        cash = top[2].number_input("Cash ($)", 1000, 100_000_000, 25_000, step=1000, key="tw_cash",
                                   help="Cash only — CSP collateral never uses margin.")
        per_name = top[3].slider("Max % / name", 5, 40, 20, 5, key="tw_pct",
                                  help="Full allocation for one name at 3 stars. He concentrates — most gains from ~5 names.")
        if refresh:
            prog = st.progress(0.0, text="Starting…")
            build_scan(progress=lambda d, n, s: prog.progress(d / max(n, 1), text=f"{d}/{n} · {s}"))
            prog.empty()
            st.rerun()
        if df is None or df.empty:
            return

        m = meta.get("macro") or {}
        chips = []
        for sym, v in (m.get("indexes") or {}).items():
            if v.get("regime"):
                chips.append(f"{_REGIME_ICON.get(v['regime'], '')} **{sym}** {v['regime']} "
                             f"({v.get('bull_stars', 0)}★ / {v.get('bear_stars', 0)}☆)")
        if m.get("vix") is not None:
            chips.append(f"VIX **{m['vix']:.1f}**" + (f" ({m['vix_rank']:.0f}%ile 1y)" if m.get("vix_rank") is not None else ""))
        (st.success if m.get("bull") else st.warning)(
            "Macro (context only, doesn't change sizing): " + " · ".join(chips))

        groups = ["SELL", "WATCH", "SKIP", "AVOID"]
        counts = {g: int(df["Action"].str.startswith(g).sum()) for g in groups}
        show = st.multiselect("Show", groups, default=["SELL", "WATCH"],
                              format_func=lambda g: f"{_ACTION_ICON[g]} {g} ({counts[g]})")
        view = df[df["Action"].apply(lambda a: a.split()[0] in show)] if show else df
        view = view.copy()
        view["Action"] = view["Action"].apply(lambda a: f"{_ACTION_ICON.get(a.split()[0], '')} {a}")
        st.dataframe(view, use_container_width=True, hide_index=True,
                     column_config={"Tier": st.column_config.NumberColumn("Tier %", help="% of the name's allocation to deploy this week")})
        st.download_button("⬇ CSV", view.to_csv(index=False).encode(), "theta_wheel.csv", "text/csv")

        st.subheader("Tickets")
        for sym in view["Ticker"]:
            r = details.get(sym)
            if not r:
                continue
            a = r["action"]
            with st.expander(f"{_ACTION_ICON.get(a.split()[0], '')} {sym} · {a}"
                             + (f" · {r['put']['yield']*100:.2f}%/wk" if r.get("put") else ""),
                             expanded=a.startswith("SELL") and sym == view["Ticker"].iloc[0]):
                _ticket(r, cash, per_name)
        if meta.get("skipped"):
            with st.expander(f"{len(meta['skipped'])} skipped"):
                st.caption(", ".join(meta["skipped"]))
