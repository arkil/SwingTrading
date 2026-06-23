"""
Livermore Market Method - Pivotal Point Screener
================================================
Identifies Jesse Livermore's Pivotal Points across a stock universe.

Livermore's Pivotal Point Rules:
  - UPWARD PIVOTAL POINT (Bullish): Price rallies to a new swing high,
    pulls back (natural reaction), then rallies through the previous high.
    That prior high was the "pivotal point" — the breakout signals a big move up.

  - DOWNWARD PIVOTAL POINT (Bearish): Price drops to a new swing low,
    bounces (natural rally), then falls through the previous low.
    That prior low was the pivotal point — the breakdown signals a big move down.

  - CONTINUATION PIVOTAL POINT: After breaking a pivotal point, price
    consolidates near the breakout level, then thrusts again in the same direction.

Practical implementation:
  1. Detect swing highs/lows over a configurable lookback window.
  2. Confirm a pivotal point when a subsequent bar closes through it
     after a counter-move of at least `min_reaction_pct`.
  3. Check volume expansion on the breakout bar (Livermore emphasised this).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
from typing import Optional, List
import requests
import io
import warnings
warnings.filterwarnings("ignore")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}


# ── Index universe fetchers ───────────────────────────────────────────────────

def _wiki_tables(url: str) -> list:
    """Fetch Wikipedia page tables using a browser User-Agent to avoid 403s."""
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def get_sp500_tickers() -> List[str]:
    """Fetch current S&P 500 constituents from Wikipedia."""
    try:
        tables = _wiki_tables(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        # First table has the constituents; column is 'Symbol'
        tickers = tables[0]["Symbol"].tolist()
        # Wikipedia uses dots (BRK.B); Yahoo Finance uses dashes (BRK-B)
        return [str(t).replace(".", "-") for t in tickers]
    except Exception as e:
        print(f"  [WARN] Could not fetch S&P 500 tickers: {e}")
        return []


def get_nasdaq100_tickers() -> List[str]:
    """Fetch current Nasdaq-100 constituents from Wikipedia."""
    try:
        tables = _wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100")
        # Find the table that has a 'Ticker' column
        for tbl in tables:
            if "Ticker" in tbl.columns:
                tickers = tbl["Ticker"].dropna().tolist()
                return [str(t).replace(".", "-") for t in tickers]
        raise ValueError("No table with 'Ticker' column found on page.")
    except Exception as e:
        print(f"  [WARN] Could not fetch Nasdaq-100 tickers: {e}")
        return []


def get_universe(name: str) -> List[str]:
    """
    Resolve a named universe to a deduplicated list of tickers.
    WATCHLIST_TICKERS is appended to every universe so custom names
    are always screened regardless of which index is chosen.

    name: 'default' | 'sp500' | 'nasdaq100' | 'both' | 'watchlist'
    """
    if name == "watchlist":
        return WATCHLIST_TICKERS

    if name == "sp500":
        tickers = get_sp500_tickers()
        if not tickers:
            print("  [WARN] Falling back to default universe.")
            return DEFAULT_TICKERS
        return _dedup(tickers + WATCHLIST_TICKERS)

    if name == "nasdaq100":
        tickers = get_nasdaq100_tickers()
        if not tickers:
            print("  [WARN] Falling back to default universe.")
            return DEFAULT_TICKERS
        return _dedup(tickers + WATCHLIST_TICKERS)

    if name == "both":
        sp  = get_sp500_tickers()
        ndq = get_nasdaq100_tickers()
        combined = sp + ndq
        if not combined:
            print("  [WARN] Falling back to default universe.")
            return DEFAULT_TICKERS
        return _dedup(combined + WATCHLIST_TICKERS)

    # default
    return DEFAULT_TICKERS


# ── Custom watchlist (always included in every universe) ──────────────────────
WATCHLIST_TICKERS = [
    # Optical networking & fiber
    "LITE",   # Lumentum Holdings
    "AAOI",   # Applied Optoelectronics
    "CIEN",   # Ciena Corporation
    "COHR",   # Coherent Corp
    "VIAV",   # Viavi Solutions
    "CLFD",   # Clearfield
    "NPKI",   # Photronics (photomasks / optical)
    "IPGP",   # IPG Photonics
    "ANGO",   # AngioDynamics
    "GLW",    # Corning
    # Networking / telecom infrastructure
    "ANET",   # Arista Networks
    "CSCO",   # Cisco
    "NTGR",   # NETGEAR
    "CALX",   # Calix Networks
    "RBBN",   # Ribbon Communications
    "SHEN",   # Shenandoah Telecom
    "LUMN",   # Lumen Technologies
    "TMUS",   # T-Mobile
    "NET",    # Cloudflare
    # Semiconductors for optical / photonics
    "MTSI",   # MACOM Technology
    "POET",   # POET Technologies
    "OLED",   # Universal Display
    "IMOS",   # ChipMOS Technologies
    "AMBA",   # Ambarella
    "SLAB",   # Silicon Laboratories
    "SWKS",   # Skyworks Solutions
    "QRVO",   # Qorvo
    "VCNX",   # Vaccinex
    "AVGO",   # Broadcom
    "NVDA",   # NVIDIA
    "MU",     # Micron Technology
    "TER",    # Teradyne
    "TSM",    # Taiwan Semiconductor
    # Storage
    "WDC",    # Western Digital
    "STX",    # Seagate Technology
    "SNDK",   # SanDisk (Western Digital spin-off)
    "DELL",   # Dell Technologies
    # Data center / cloud infrastructure
    "VRT",    # Vertiv Holdings
    "GEV",    # GE Vernova
    "FSLY",   # Fastly
    "CRCL",   # Circle Internet (crypto/fintech infra)
    "GCT",    # GigaCloud Technology
    # Space & satellite
    "RKLB",   # Rocket Lab
    "ASTS",   # AST SpaceMobile
    "PL",     # Planet Labs
    "SPHR",   # Sphere Entertainment
    # Clean energy / power
    "BE",     # Bloom Energy
    "MOD",    # Modine Manufacturing
    "POWL",   # Powell Industries
    # EV / autonomy
    "TSLA",   # Tesla
    # Biotech / healthcare
    "LLY",    # Eli Lilly
    "XBI",    # SPDR S&P Biotech ETF
    # Consumer / food
    "CAVA",   # CAVA Group
    "COST",   # Costco
    # Mega-cap tech
    "GOOGL",  # Alphabet
    "NBIS",   # Nebius Group (AI infra)

    # ── AI Infrastructure ─────────────────────────────────────────────────
    "CRWV",   # CoreWeave — pure-play AI cloud ($10B+ revenue projected 2026)
    "SMCI",   # Super Micro Computer — AI servers & rack systems
    "LRCX",   # Lam Research — semiconductor etch/deposition equipment
    "CEL",    # Celestica — AI server deployment & integration
    "ALAB",   # Astera Labs — PCIe/CXL connectivity for AI data centers
    "MRVL",   # Marvell Technology — custom AI silicon & networking
    "ARM",    # Arm Holdings — CPU architecture for AI edge & data center
    "PLTR",   # Palantir — AI software platform (defense + enterprise)

    # ── Nuclear & Power Generation ────────────────────────────────────────
    "SMR",    # NuScale Power — small modular reactor pure-play
    "OKLO",   # Oklo — next-gen microreactor pure-play
    "CEG",    # Constellation Energy — largest US nuclear operator
    "CCJ",    # Cameco — uranium producer
    "LEU",    # Centrus Energy — uranium enrichment
    "BWXT",   # BWX Technologies — nuclear components & naval reactors
    "VST",    # Vistra — nuclear + natural gas power generation
    "NRG",    # NRG Energy — power generation, AI data center contracts
    "UUUU",   # Energy Fuels — uranium & rare earth producer

    # ── Defense Tech ──────────────────────────────────────────────────────
    "KTOS",   # Kratos Defense — tactical drones (XQ-58A Valkyrie)
    "AVAV",   # AeroVironment — military small UAS
    "AXON",   # Axon Enterprise — digital weapons & software for defense
    "HII",    # Huntington Ingalls — military shipbuilding
    "PSN",    # Parsons Corporation — defense & cyber solutions
    "CACI",   # CACI International — defense intelligence & cyber
    "LDOS",   # Leidos — defense IT & AI systems
    "RCAT",   # Red Cat Holdings — military drone pure-play

    # ── Robotics & Automation ─────────────────────────────────────────────
    "ISRG",   # Intuitive Surgical — surgical robotics leader
    "PATH",   # UiPath — robotic process automation software
    "SYM",    # Symbotic — warehouse automation (Walmart, Target)
    "SERV",   # Serve Robotics — autonomous sidewalk delivery robots
    "ACMR",   # ACM Research — semiconductor cleaning equipment

    # ── GLP-1 / Obesity & Diabetes ───────────────────────────────────────
    "NVO",    # Novo Nordisk — Wegovy/Ozempic pioneer
    "VKTX",   # Viking Therapeutics — VK2735 dual agonist pure-play
    "AMGN",   # Amgen — maritide obesity drug candidate
    "REGN",   # Regeneron — cardiometabolic pipeline

    # ── Cybersecurity ─────────────────────────────────────────────────────
    "CRWD",   # CrowdStrike — endpoint & cloud security leader
    "FTNT",   # Fortinet — network security, 30% margins
    "ZS",     # Zscaler — cloud-native zero-trust security
    "OKTA",   # Okta — identity & access management
    "S",      # SentinelOne — AI-native endpoint security
    "CYBR",   # CyberArk — privileged access management
    "QLYS",   # Qualys — vulnerability management

    # ── Quantum Computing ─────────────────────────────────────────────────
    "IONQ",   # IonQ — trapped-ion quantum computing pure-play
    "QBTS",   # D-Wave Quantum — quantum annealing systems
    "RGTI",   # Rigetti Computing — superconducting quantum processors
    "QUBT",   # Quantum Computing Inc. — quantum software & hardware

    # ── Grid Modernization & Power Infrastructure ─────────────────────────
    "ETN",    # Eaton — power management & grid equipment
    "PWR",    # Quanta Services — grid construction & hardening
    "ITRI",   # Itron — smart meters & grid analytics
    "HUBB",   # Hubbell — electrical grid components
    "FIX",    # Comfort Systems — electrical & mechanical grid buildout
    "EME",    # EMCOR Group — electrical infrastructure contractor

    # ── Onshoring / US Manufacturing ──────────────────────────────────────
    "NUE",    # Nucor — largest US steel producer
    "STLD",   # Steel Dynamics — US steel manufacturing
    "URI",    # United Rentals — equipment rental for construction boom
    "VMC",    # Vulcan Materials — construction aggregates
    "MLM",    # Martin Marietta — aggregates & heavy building materials
    "GRC",    # Gorman-Rupp — pumps for manufacturing & infrastructure
]


def _dedup(tickers: List[str]) -> List[str]:
    """Deduplicate while preserving order."""
    seen: set = set()
    out: List[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ── Default universe (fallback / --universe default) ─────────────────────────
DEFAULT_TICKERS = _dedup([
    # Large-cap US stocks
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO",
    "JPM", "V", "UNH", "XOM", "LLY", "JNJ", "MA", "HD", "PG", "MRK",
    "COST", "ABBV", "BAC", "NFLX", "CRM", "AMD", "ORCL", "WMT", "CVX",
    "TMO", "ADBE", "QCOM", "TXN", "PEP", "ACN", "DHR", "NEE", "MCD",
    "PM", "LIN", "AMGN", "LOW", "INTU", "SPGI", "GE", "NOW", "IBM",
    "GS", "CAT", "RTX", "BKNG", "DE", "UPS", "AXP", "SYK", "ISRG",
    "T", "VRTX", "PANW", "LRCX", "ADI", "GILD", "REGN", "PLD", "CB",
    "MU", "SO", "KLAC", "CI", "CME", "MDLZ", "EOG", "SLB", "BDX",
    "SCHW", "ZTS", "WM", "ICE", "PYPL", "CEG", "NOC", "HUM", "SNPS",
    "APH", "SHW", "CDNS", "USB", "TJX", "DUK", "MAR", "PNC", "MCO",
] + WATCHLIST_TICKERS)


# ── Core pivot detection ──────────────────────────────────────────────────────

def find_swing_highs_lows(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Mark swing highs and swing lows using a centred rolling window of width 2n+1.
    A swing high: bar whose High equals the rolling max over [i-n .. i+n].
    A swing low : bar whose Low  equals the rolling min over [i-n .. i+n].
    Fully vectorised — no Python loops, no numpy scalar comparisons.
    """
    df    = df.copy()
    # Select first column in case yfinance returned duplicate columns
    _h = df["High"]
    highs = _h.iloc[:, 0] if isinstance(_h, pd.DataFrame) else _h
    _l = df["Low"]
    lows  = _l.iloc[:, 0] if isinstance(_l, pd.DataFrame) else _l

    win          = 2 * n + 1
    roll_hi_max  = highs.rolling(win, center=True, min_periods=win).max()
    roll_lo_min  = lows.rolling(win, center=True, min_periods=win).min()

    df["swing_high"] = (highs == roll_hi_max)
    df["swing_low"]  = (lows  == roll_lo_min)
    return df


def classify_pivotal_points(
    df: pd.DataFrame,
    swing_window: int = 5,
    min_reaction_pct: float = 2.0,
    volume_expansion: bool = True,
    avg_vol_window: int = 20,
) -> pd.DataFrame:
    """
    Identify Livermore Pivotal Point signals.

    Returns df with added columns:
        pivot_high      – price level of the pivotal resistance broken
        pivot_low       – price level of the pivotal support broken
        signal          – 'UPWARD_PIVOT' | 'DOWNWARD_PIVOT' | ''
        breakout_bar    – True on the bar that confirms the signal
        vol_expansion   – True if volume > avg_vol_window average on that bar
    """
    df = find_swing_highs_lows(df, n=swing_window)

    if volume_expansion:
        df["avg_vol"] = df["Volume"].rolling(avg_vol_window).mean()

    df["pivot_high"]    = np.nan
    df["pivot_low"]     = np.nan
    df["signal"]        = ""
    df["breakout_bar"]  = False
    df["vol_expansion"] = False

    last_swing_high_price = None
    last_swing_high_idx   = None
    last_swing_low_price  = None
    last_swing_low_idx    = None

    for i in range(len(df)):
        row = df.iloc[i]

        # Record swing highs/lows as we encounter them
        if row["swing_high"]:
            last_swing_high_price = row["High"]
            last_swing_high_idx   = i

        if row["swing_low"]:
            last_swing_low_price = row["Low"]
            last_swing_low_idx   = i

        # ── UPWARD PIVOT: close breaks above a prior swing high ──────────────
        if (
            last_swing_high_price is not None
            and last_swing_high_idx is not None
            and i > last_swing_high_idx + swing_window          # enough bars after pivot
            and row["Close"] > last_swing_high_price            # close through pivotal high
        ):
            # Verify there was a meaningful reaction between the swing high and now
            reaction_low = df.iloc[last_swing_high_idx:i]["Low"].min()
            reaction_pct = (last_swing_high_price - reaction_low) / last_swing_high_price * 100

            if reaction_pct >= min_reaction_pct:
                df.at[df.index[i], "signal"]       = "UPWARD_PIVOT"
                df.at[df.index[i], "pivot_high"]   = last_swing_high_price
                df.at[df.index[i], "breakout_bar"] = True

                if volume_expansion and not np.isnan(row["avg_vol"]):
                    df.at[df.index[i], "vol_expansion"] = row["Volume"] > row["avg_vol"]

                # Reset so we don't re-fire on consecutive bars
                last_swing_high_price = None
                last_swing_high_idx   = None

        # ── DOWNWARD PIVOT: close breaks below a prior swing low ─────────────
        if (
            last_swing_low_price is not None
            and last_swing_low_idx is not None
            and i > last_swing_low_idx + swing_window
            and row["Close"] < last_swing_low_price
        ):
            reaction_high = df.iloc[last_swing_low_idx:i]["High"].max()
            reaction_pct  = (reaction_high - last_swing_low_price) / last_swing_low_price * 100

            if reaction_pct >= min_reaction_pct:
                df.at[df.index[i], "signal"]      = "DOWNWARD_PIVOT"
                df.at[df.index[i], "pivot_low"]   = last_swing_low_price
                df.at[df.index[i], "breakout_bar"] = True

                if volume_expansion and not np.isnan(row["avg_vol"]):
                    df.at[df.index[i], "vol_expansion"] = row["Volume"] > row["avg_vol"]

                last_swing_low_price = None
                last_swing_low_idx   = None

    return df


def continuation_pivot(df: pd.DataFrame, lookback_bars: int = 5) -> pd.DataFrame:
    """
    After an initial pivotal breakout, flag a continuation if price
    consolidates within `lookback_bars` of the breakout high/low and
    then makes a new extreme in the direction of the original signal.
    """
    df = df.copy()
    df["continuation"] = False

    last_signal      = None
    last_signal_close = None
    bars_since_signal = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        if row["signal"] in ("UPWARD_PIVOT", "DOWNWARD_PIVOT"):
            last_signal       = row["signal"]
            last_signal_close = row["Close"]
            bars_since_signal = 0
        elif last_signal is not None:
            bars_since_signal += 1

            if bars_since_signal <= lookback_bars * 3:
                if (last_signal == "UPWARD_PIVOT"
                        and row["Close"] > last_signal_close):
                    df.at[idx, "continuation"] = True
                    last_signal_close = row["Close"]
                elif (last_signal == "DOWNWARD_PIVOT"
                        and row["Close"] < last_signal_close):
                    df.at[idx, "continuation"] = True
                    last_signal_close = row["Close"]
            else:
                last_signal = None

    return df


# ── Screener ─────────────────────────────────────────────────────────────────

def screen_ticker(
    ticker: str,
    period_days: int = 252,
    swing_window: int = 5,
    min_reaction_pct: float = 2.0,
    recent_bars: int = 10,
) -> Optional[dict]:
    """
    Download data for one ticker and return its most recent pivotal signal,
    or None if no signal in the last `recent_bars` bars.
    """
    end   = datetime.today()
    start = end - timedelta(days=period_days + 60)   # extra buffer for indicators

    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception:
        return None

    if raw is None or len(raw) < 60:
        return None

    # Flatten multi-level columns if present, then drop any duplicates
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.loc[:, ~raw.columns.duplicated()]

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = classify_pivotal_points(
        df,
        swing_window=swing_window,
        min_reaction_pct=min_reaction_pct,
    )
    df = continuation_pivot(df)

    # Look for any signal in the most recent `recent_bars` bars
    recent = df.tail(recent_bars)
    signals = recent[recent["signal"] != ""]

    if signals.empty:
        return None

    last_sig = signals.iloc[-1]
    current_price = df["Close"].iloc[-1]

    # Distance from pivot level
    pivot_level = (
        last_sig["pivot_high"]
        if last_sig["signal"] == "UPWARD_PIVOT"
        else last_sig["pivot_low"]
    )

    # 52-week high/low context
    year_data    = df.tail(252)
    high_52w     = year_data["High"].max()
    low_52w      = year_data["Low"].min()
    pct_from_52h = (current_price / high_52w - 1) * 100
    pct_from_52l = (current_price / low_52w  - 1) * 100

    # Trend filter: 50-day EMA vs 200-day EMA
    df["ema50"]  = df["Close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["Close"].ewm(span=200, adjust=False).mean()
    trend = "UPTREND" if df["ema50"].iloc[-1] > df["ema200"].iloc[-1] else "DOWNTREND"

    bars_ago = len(df) - 1 - df.index.get_loc(last_sig.name)

    return {
        "Ticker":          ticker,
        "Signal":          last_sig["signal"],
        "Signal Date":     last_sig.name.strftime("%Y-%m-%d"),
        "Bars Ago":        bars_ago,
        "Pivot Level":     round(pivot_level, 2),
        "Close":           round(current_price, 2),
        "% from Pivot":    round((current_price / pivot_level - 1) * 100, 2),
        "Vol Expansion":   "YES" if last_sig["vol_expansion"] else "NO",
        "Continuation":    "YES" if last_sig["continuation"] else "NO",
        "52W High":        round(high_52w, 2),
        "% from 52W High": round(pct_from_52h, 2),
        "% from 52W Low":  round(pct_from_52l, 2),
        "Trend (EMA)":     trend,
    }


def run_screener(
    tickers: List[str],
    period_days: int = 252,
    swing_window: int = 5,
    min_reaction_pct: float = 2.0,
    recent_bars: int = 10,
    signal_filter: str = "ALL",      # ALL | UPWARD_PIVOT | DOWNWARD_PIVOT
    trend_aligned: bool = False,
    vol_required: bool = False,
) -> pd.DataFrame:

    results = []
    total = len(tickers)

    print(f"\n{'='*60}")
    print(f"  Livermore Pivotal Point Screener")
    print(f"  Scanning {total} tickers  |  {datetime.today().strftime('%Y-%m-%d')}")
    print(f"  Swing window: {swing_window} bars  |  Min reaction: {min_reaction_pct}%")
    print(f"  Signal window: last {recent_bars} bars")
    print(f"{'='*60}\n")

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:>3}/{total}] {ticker:<8}", end="\r")
        result = screen_ticker(
            ticker,
            period_days=period_days,
            swing_window=swing_window,
            min_reaction_pct=min_reaction_pct,
            recent_bars=recent_bars,
        )
        if result:
            results.append(result)

    print(" " * 40, end="\r")  # clear progress line

    if not results:
        print("  No pivotal point signals found.\n")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)

    # Optional filters
    if signal_filter != "ALL":
        df_results = df_results[df_results["Signal"] == signal_filter]

    if vol_required:
        df_results = df_results[df_results["Vol Expansion"] == "YES"]

    if trend_aligned:
        df_results = df_results[
            ((df_results["Signal"] == "UPWARD_PIVOT")   & (df_results["Trend (EMA)"] == "UPTREND"))  |
            ((df_results["Signal"] == "DOWNWARD_PIVOT") & (df_results["Trend (EMA)"] == "DOWNTREND"))
        ]

    # Sort: upward pivots first by % from 52W High (closer to highs = stronger), then downward
    df_results["_sort_key"] = df_results["Signal"].map(
        {"UPWARD_PIVOT": 0, "DOWNWARD_PIVOT": 1}
    )
    df_results = df_results.sort_values(
        ["_sort_key", "% from 52W High"], ascending=[True, False]
    ).drop(columns=["_sort_key"]).reset_index(drop=True)

    return df_results


def print_results(df: pd.DataFrame):
    if df.empty:
        return

    up   = df[df["Signal"] == "UPWARD_PIVOT"]
    down = df[df["Signal"] == "DOWNWARD_PIVOT"]

    def _print_section(title: str, section: pd.DataFrame):
        if section.empty:
            return
        print(f"\n{'─'*80}")
        print(f"  {title}  ({len(section)} signals)")
        print(f"{'─'*80}")
        cols = ["Ticker", "Signal Date", "Bars Ago", "Pivot Level",
                "Close", "% from Pivot", "Vol Expansion",
                "% from 52W High", "Trend (EMA)"]
        print(section[cols].to_string(index=False))

    _print_section("UPWARD PIVOTAL POINTS  ▲  (Bullish Breakouts)", up)
    _print_section("DOWNWARD PIVOTAL POINTS  ▼  (Bearish Breakdowns)", down)

    print(f"\n{'='*80}")
    print(f"  Total signals: {len(df)}  |  Bullish: {len(up)}  |  Bearish: {len(down)}")
    print(f"{'='*80}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Livermore Market Method - Pivotal Point Screener"
    )
    p.add_argument(
        "--tickers", nargs="+", default=None,
        help="Space-separated list of tickers. Defaults to built-in universe.",
    )
    p.add_argument(
        "--file", type=str, default=None,
        help="Path to a text/CSV file with one ticker per line.",
    )
    p.add_argument(
        "--period", type=int, default=252,
        help="Lookback period in trading days (default: 252 = 1 year).",
    )
    p.add_argument(
        "--swing-window", type=int, default=5,
        help="Bars on each side to confirm a swing high/low (default: 5).",
    )
    p.add_argument(
        "--min-reaction", type=float, default=2.0,
        help="Min %% counter-move required between pivot and breakout (default: 2.0).",
    )
    p.add_argument(
        "--recent-bars", type=int, default=10,
        help="Only return signals fired in the last N bars (default: 10).",
    )
    p.add_argument(
        "--signal", choices=["ALL", "UPWARD_PIVOT", "DOWNWARD_PIVOT"], default="ALL",
        help="Filter by signal type (default: ALL).",
    )
    p.add_argument(
        "--trend-aligned", action="store_true",
        help="Only show signals aligned with the 50/200 EMA trend.",
    )
    p.add_argument(
        "--vol-required", action="store_true",
        help="Only show signals with above-average volume on the breakout bar.",
    )
    p.add_argument(
        "--universe",
        choices=["default", "sp500", "nasdaq100", "both", "watchlist"],
        default="default",
        help=(
            "Named stock universe to scan. "
            "'sp500' fetches S&P 500 (~503 tickers) + watchlist, "
            "'nasdaq100' fetches Nasdaq-100 (~101 tickers) + watchlist, "
            "'both' merges both indices + watchlist (~550 tickers), "
            "'watchlist' scans only the custom watchlist (optical/networking names). "
            "Ignored when --tickers or --file is provided. (default: default)"
        ),
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Save results to a CSV file at this path.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Build ticker list (priority: --tickers > --file > --universe)
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.file:
        with open(args.file) as fh:
            tickers = [
                line.strip().upper()
                for line in fh
                if line.strip() and not line.startswith("#")
            ]
    else:
        tickers = get_universe(args.universe)

    results = run_screener(
        tickers=tickers,
        period_days=args.period,
        swing_window=args.swing_window,
        min_reaction_pct=args.min_reaction,
        recent_bars=args.recent_bars,
        signal_filter=args.signal,
        trend_aligned=args.trend_aligned,
        vol_required=args.vol_required,
    )

    print_results(results)

    if args.output and not results.empty:
        results.to_csv(args.output, index=False)
        print(f"  Results saved to: {args.output}\n")
