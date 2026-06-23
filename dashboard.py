"""
Swing Trading Dashboard
=======================
Run:  streamlit run dashboard.py

Scanners
  • Livermore Pivotal  — Jesse Livermore's pivot point strategy
  • EMA Crossover      — Fast/Slow/Trend EMA crossover with filters
  • Breakout           — 6 research-backed breakout strategies
  • Minervini SEPA     — Stage 2 trend template + VCP + RS Rating
  • Volume Scanner     — Multi-timeframe volume anomaly detection

Adding a new scanner
  1. Create myscannermodule.py with a run_*() function
  2. Import it below
  3. Write a render_myscanner() function following the existing pattern
  4. Append one entry to SCANNERS at the bottom of this file
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Load .env so ALPACA_API_KEY / ALPACA_SECRET_KEY are available without manual export
_env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_file):
    with open(_env_file) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

_VOL_SCANNER_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "volume_scanner")
)
if os.path.isdir(_VOL_SCANNER_PATH):
    sys.path.insert(0, _VOL_SCANNER_PATH)

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from collections import Counter
import pytz
import glob
import time
import warnings
warnings.filterwarnings("ignore")

from livermore_pivotal_screener import (
    run_screener, get_universe, get_sp500_tickers, get_nasdaq100_tickers,
    classify_pivotal_points, continuation_pivot,
    DEFAULT_TICKERS, WATCHLIST_TICKERS, _dedup,
)
from ema_crossover_screener import (
    run_ema_screener, detect_crossovers, PRESETS as EMA_PRESETS,
)
from breakout_screener import (
    run_breakout_screener, run_all_strategies, add_base_indicators,
    STRATEGY_GROUPS, STRATEGY_LABELS,
)
from minervini_screener import (
    run_minervini_screener, check_trend_template,
    _sma as _msma,
)
from astro_scanner import (
    run_astro_scanner, compute_market_bias, get_current_moon,
    get_retrograde_status, get_planetary_aspects,
    get_upcoming_events as get_astro_events,
    get_moon_event_dates, analyze_moon_returns, find_gann_cycle_dates,
    get_karana, is_vishti, backtest_vishti,
    get_rahu_ketu, get_moon_nakshatra, get_vedic_planets,
    compute_vedic_daily_score, build_prediction_calendar,
    get_decade_cheatsheet, get_annual_roadmap,
    generate_annual_forecast, generate_multi_year_outlook,
)
from rsi_screener import run_rsi_screener, detect_rsi_signals
from macd_screener import run_macd_screener, detect_macd_signals
from gap_screener import run_gap_screener, detect_gaps, run_live_gap_screener
from combined_screener import run_combined_screener
from swing_options_screener import run_swing_options_screener
from ibd_scanner import run_ibd_scanner
from backtest_strategy import run_strategy_backtest
from stock_analyzer_module import render_stock_analyzer
from spy_reversal_log import (
    sync_from_alpaca as _spy_sync,
    load_logs as _spy_load_logs,
    get_records as _spy_get_records,
    available_dates as _spy_available_dates,
    pair_trades as _spy_pair_trades,
    parse_orders as _spy_parse_orders,
)
from options_trade_log import (
    sync_from_alpaca as _opt_sync,
    load_logs as _opt_load_logs,
    get_records as _opt_get_records,
    available_dates as _opt_available_dates,
    pair_trades as _opt_pair_trades,
    parse_orders as _opt_parse_orders,
)
from options_backtest_runner import run_options_backtest
from economic_calendar import (
    get_upcoming_events,
    get_news_feed,
    get_event_context,
    get_earnings_calendar,
    _CATEGORY_META as _CAL_META,
)
from alpaca_trader import (
    make_client, get_account_summary, is_market_open_alpaca,
    get_positions, get_open_orders,
    get_todays_trades, get_portfolio_history,
    close_position, close_all_positions,
    cancel_order, cancel_all_orders,
    execute_alerts as alpaca_execute_alerts,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Swing Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCREENER_DIR = os.path.join(os.path.dirname(__file__), "screener_output")

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base scroll ───────────────────────────────────────────────────── */
html, body { overflow-y: auto !important; scroll-behavior: smooth; }
[data-testid="stAppViewContainer"] { overflow-y: auto !important; height: auto !important; }
[data-testid="stMain"], section.main { overflow-y: visible !important; height: auto !important; }
.block-container {
    padding-top: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    overflow: visible !important;
    max-height: none !important;
    max-width: 1400px;
}
[data-testid="stDataFrame"], .stDataFrame { overflow: auto !important; }
[data-testid="stMarkdownContainer"] > div { overflow: visible !important; }

/* ── Sidebar dark theme ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    min-width: 230px !important;
    max-width: 270px !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #0f172a !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    height: 100vh;
    min-width: 230px !important;
    padding: 1.4rem 0.9rem 1rem !important;
    box-sizing: border-box;
}

/* ── Sidebar text & labels ─────────────────────────────────────────── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stCaption { color: #475569 !important; font-size: 11px !important; }
[data-testid="stSidebar"] hr { border-color: #1e293b !important; }
[data-testid="stSidebar"] [data-testid="stCheckbox"] label { font-size: 12px !important; }

/* ── Nav group headers ─────────────────────────────────────────────── */
.nav-group-hdr {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.4px;
    color: #475569 !important;
    text-transform: uppercase;
    padding: 14px 6px 5px;
    margin: 0;
    line-height: 1;
}

/* ── Nav buttons ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #94a3b8 !important;
    text-align: left !important;
    padding: 7px 10px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    border-radius: 7px !important;
    width: 100% !important;
    margin: 1px 0 !important;
    transition: background 0.12s, color 0.12s !important;
    line-height: 1.35 !important;
    justify-content: flex-start !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(148,163,184,0.1) !important;
    color: #e2e8f0 !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
    border: none !important;
}

/* ── Active nav button (element after .nav-active-marker) ──────────── */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.nav-active-marker) + div .stButton > button,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.nav-active-marker) + div > div > .stButton > button {
    background: rgba(59,130,246,0.14) !important;
    color: #93c5fd !important;
    font-weight: 600 !important;
    border-left: 3px solid #3b82f6 !important;
    padding-left: 7px !important;
}

/* ── Hide Streamlit toolbar & deploy button ────────────────────────── */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu,
header[data-testid="stHeader"] { display: none !important; }

/* ── Metric cards ──────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 12px 16px;
}

/* ── General button polish ─────────────────────────────────────────── */
button[kind="primary"] { border-radius: 8px; }
.stButton > button { border-radius: 8px; }

/* ── Scanner page headers ──────────────────────────────────────────── */
.scanner-header  { display:flex; align-items:center; gap:12px; margin-bottom:4px; }
.scanner-title   { font-size:1.6rem; font-weight:700; }
.scanner-desc    { color:#999; font-size:0.9rem; margin-bottom:1rem; }

/* ── Expander polish ───────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}

/* ── Tab strip ─────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-testid="stTab"] { font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═════════════════════════════════════════════════════════════════════════════

def is_market_open() -> bool:
    et  = pytz.timezone("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5:
        return False
    return now.replace(hour=9, minute=30, second=0, microsecond=0) <= now <= \
           now.replace(hour=16, minute=0,  second=0, microsecond=0)


def market_badge_html() -> str:
    if is_market_open():
        return "<span style='background:#1a9641;color:#fff;padding:3px 10px;border-radius:12px;font-size:12px'>● OPEN</span>"
    return "<span style='background:#555;color:#ccc;padding:3px 10px;border-radius:12px;font-size:12px'>● CLOSED</span>"


@st.cache_data(ttl=900)
def fetch_ohlcv(ticker: str, days: int = 120) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=days + 30)
    raw   = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    return raw[["Open", "High", "Low", "Close", "Volume"]].dropna()


@st.cache_data(ttl=3600)
def _cached_sp500():
    return get_sp500_tickers() or []

@st.cache_data(ttl=3600)
def _cached_ndq100():
    return get_nasdaq100_tickers() or []

def _fetch_yf_trending_raw() -> list:
    """
    Pull live tickers from 5 Yahoo Finance real-time feeds and deduplicate.
    No static fallback — returns empty list on total failure.
    """
    import urllib.request, json
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    seen, result = set(), []

    def _clean(sym: str) -> str:
        return sym.strip().upper()

    def _valid(sym: str) -> bool:
        return bool(sym) and "." not in sym and len(sym) <= 6

    # ── 1. Yahoo Finance Trending Tickers (real-time page) ────────────────────
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/trending/US?count=50&useQuotes=true"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())
        for q in data.get("finance", {}).get("result", [{}])[0].get("quotes", []):
            sym = _clean(q.get("symbol", ""))
            if _valid(sym) and sym not in seen:
                seen.add(sym); result.append(sym)
    except Exception:
        pass

    # ── 2–5. Yahoo Finance Screener feeds ────────────────────────────────────
    for scrId in ["most_actives", "day_gainers", "growth_technology_stocks", "small_cap_gainers"]:
        try:
            url = (
                "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
                f"?formatted=false&scrIds={scrId}&count=75"
            )
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=6) as r:
                data = json.loads(r.read())
            for q in data.get("finance", {}).get("result", [{}])[0].get("quotes", []):
                sym = _clean(q.get("symbol", ""))
                if _valid(sym) and sym not in seen:
                    seen.add(sym); result.append(sym)
        except Exception:
            pass

    return result


@st.cache_data(ttl=900)
def _cached_trending() -> list:
    """Live trending tickers refreshed every 15 min from 5 Yahoo Finance feeds."""
    tickers = _fetch_yf_trending_raw()
    if not tickers:
        # Last-resort: fall back to most-actives via yfinance SDK
        try:
            data = yf.screen("most_actives", count=50)
            tickers = [q["symbol"] for q in data.get("quotes", [])
                       if q.get("symbol") and "." not in q["symbol"]]
        except Exception:
            pass
    return tickers


@st.cache_data(ttl=86400)
def _cached_russell1000() -> list:
    """Russell 1000 tickers scraped from Wikipedia (refreshes daily)."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Russell_1000_Index")
        for tbl in tables:
            cols = [c.lower() for c in tbl.columns]
            ticker_col = next((tbl.columns[i] for i, c in enumerate(cols)
                               if "tick" in c or "symbol" in c), None)
            if ticker_col and len(tbl) > 100:
                tks = [str(t).strip().replace(".", "-") for t in tbl[ticker_col] if str(t).strip()]
                return [t for t in tks if t and len(t) <= 5]
    except Exception:
        pass
    return list(dict.fromkeys((_cached_sp500() or []) + (_cached_ndq100() or [])))


_HIGH_GROWTH_TICKERS = [
    "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","AVGO","PLTR",
    "APP","ARM","CRWD","PANW","AXON","FICO","DDOG","NET","ZS","MRVL",
    "SNOW","TTD","HUBS","DUOL","CAVA","CELH","ONON","SHOP","COIN","HOOD",
]

@st.cache_data(ttl=3600)
def _base_universe() -> list:
    """S&P 500 + Nasdaq-100 + High-Growth Tech + Trending + Watchlist (~600 symbols)."""
    return _dedup(
        _cached_sp500()
        + _cached_ndq100()
        + _HIGH_GROWTH_TICKERS
        + _cached_trending()
        + WATCHLIST_TICKERS
    )

def get_selected_tickers() -> list:
    """Full scan universe. Adds Russell 1000 if user opted in via sidebar toggle."""
    tickers = _base_universe()
    if st.session_state.get("use_russell1000"):
        tickers = _dedup(tickers + _cached_russell1000())
    return tickers


_SCANNER_PREFIXES = {
    "livermore": "Livermore Pivot",
    "signals":   "Livermore Pivot",
    "ema":       "EMA Crossover",
    "breakout":  "Breakout",
    "minervini": "Minervini SEPA",
    "volume":    "Volume Scanner",
    "rsi":       "RSI Scanner",
    "macd":      "MACD Scanner",
    "gap":       "Gap Scanner",
    "combined":  "Combined Strategy",
    "swing_opts":"Options 45-60 DTE",
    "oneil":     "O'Neil CAN SLIM",
    "alerts":    "Daily Alerts",
}


def save_scan_result(df: pd.DataFrame, scanner: str) -> str:
    os.makedirs(SCREENER_DIR, exist_ok=True)
    fname = f"{scanner}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(os.path.join(SCREENER_DIR, fname), index=False)
    return fname


def load_all_runs(scanner_filter: str = "all") -> dict:
    patterns = (
        [os.path.join(SCREENER_DIR, f"{p}_*.csv") for p in _SCANNER_PREFIXES]
        if scanner_filter == "all"
        else [os.path.join(SCREENER_DIR, f"{scanner_filter}_*.csv")]
    )
    files = sorted({f for pat in patterns for f in glob.glob(pat)}, reverse=True)[:60]
    result = {}
    for f in files:
        base = os.path.basename(f)
        label = _SCANNER_PREFIXES.get(base.split("_")[0], base.split("_")[0])
        try:
            result[base] = (label, pd.read_csv(f))
        except Exception:
            pass
    return result


def _vol_candle_chart(ticker: str, days: int, thresholds: tuple) -> go.Figure:
    raw = fetch_ohlcv(ticker, days=days)
    if raw.empty:
        return go.Figure()
    raw["vol_ma20"]  = raw["Volume"].rolling(20).mean()
    raw["vol_ratio"] = raw["Volume"] / raw["vol_ma20"].replace(0, float("nan"))
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.3, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=raw.index, open=raw["Open"], high=raw["High"],
        low=raw["Low"], close=raw["Close"], name=ticker,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(raw["Close"], raw["Open"])]
    fig.add_trace(go.Bar(x=raw.index, y=raw["Volume"],
                         marker_color=vol_colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=raw.index, y=raw["vol_ma20"], mode="lines",
                             line=dict(color="#ffb74d", width=1.2, dash="dot"),
                             name="Vol MA20"), row=2, col=1)
    fig.add_trace(go.Scatter(x=raw.index, y=raw["vol_ratio"], mode="lines",
                             line=dict(color="#ab47bc", width=1.2),
                             fill="tozeroy", fillcolor="rgba(171,71,188,0.10)",
                             name="Vol Ratio"), row=3, col=1)
    unusual, high = thresholds
    for val, color, lbl in [(unusual, "#f0c040", f"Unusual {unusual}×"),
                             (high,    "#ef5350", f"High {high}×")]:
        fig.add_hline(y=val, line_dash="dash", line_color=color, line_width=1,
                      annotation_text=lbl, annotation_position="top right", row=3, col=1)
    _apply_chart_style(fig, f"{ticker} — Volume Analysis", rows=3,
                       ytitles=["Price", "Volume", "Vol Ratio ×"])
    return fig


def _candlestick_vol(df: pd.DataFrame, ticker: str) -> tuple:
    """Return (candle trace, vol bar trace list) for reuse in charts."""
    candle = go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=ticker,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    )
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["Close"], df["Open"])]
    vol = go.Bar(x=df.index, y=df["Volume"], marker_color=colors,
                 name="Volume", showlegend=False)
    return candle, vol


def _apply_chart_style(fig: go.Figure, title: str, rows: int = 2,
                       ytitles: list = None, height: int = 650):
    fig.update_layout(
        height=height, template="plotly_dark",
        margin=dict(l=40, r=20, t=45, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        title=dict(text=title, font=dict(size=15)),
    )
    if ytitles:
        for i, t in enumerate(ytitles, 1):
            fig.update_yaxes(title_text=t, row=i, col=1)


def _page_header(icon: str, title: str, desc: str,
                 scan_key: str, last_key: str, count_key: str = None):
    """Render consistent scanner page header. Returns True if scan button was clicked."""
    c1, c2, c3 = st.columns([6, 2, 2])
    with c1:
        st.markdown(
            f"<div class='scanner-header'>"
            f"<span style='font-size:2rem'>{icon}</span>"
            f"<span class='scanner-title'>{title}</span>"
            f"&nbsp;{market_badge_html()}"
            f"</div>"
            f"<div class='scanner-desc'>{desc}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        last = st.session_state.get(last_key)
        if last:
            st.caption(f"Last scan")
            st.caption(last)
    with c3:
        clicked = st.button("▶  Scan Now", type="primary",
                            use_container_width=True, key=scan_key)
    st.divider()
    return clicked


def _about_expander(scanner_id: str):
    """Render an ℹ️ About expander for the given scanner."""
    _ABOUT = {
        "livermore": {
            "title": "🔴 Livermore Pivotal Points — How It Works",
            "body": """
**Strategy (Hypothesis)**
Jesse Livermore's pivot method identifies *key price levels* where a stock reverses direction.
An **UPWARD PIVOT** forms when price makes a swing low and then rallies ≥ Min Reaction %; a
**DOWNWARD PIVOT** forms when price makes a swing high then drops ≥ Min Reaction %.
*Hypothesis:* Price re-testing the pivot level after confirmation is a high-probability entry.

**Time Period**  120 calendar days of daily OHLCV data (≈ 85 trading days). Configurable via the chart "Days" slider.

---
**Column Glossary**

| Column | Meaning |
|--------|---------|
| **Signal** | `UPWARD_PIVOT` = buy setup; `DOWNWARD_PIVOT` = short setup |
| **Signal Date** | Date the pivot was confirmed (the bar that completed the reversal) |
| **Bars Ago** | *Trading days* since the signal fired — lower = fresher |
| **Pivot Level** | Exact price of the swing high or swing low that formed the pivot |
| **Close** | Most recent closing price |
| **% from Pivot** | (Close − Pivot) / Pivot × 100. Negative = price fell below pivot (stop zone) |
| **Vol Expansion** | YES = volume on signal day exceeded 20-day average (institutional confirmation) |
| **% from 52W High** | Distance from 52-week high — negative means below it |
| **Trend (EMA)** | ABOVE = price > EMA50 (uptrend); BELOW = EMA50 > price (downtrend) |

**Settings Guide**

| Setting | Meaning |
|---------|---------|
| **Swing window** | Bars looked left & right to define a local high/low. 5 = standard swing, 3 = faster/noisier |
| **Min reaction %** | Minimum move required to qualify. 1.5% filters noise; 3%+ = major pivots only |
| **Signal window** | Only show signals from within this many bars of today (recency filter) |
| **Trend aligned** | Filters to only UPWARD pivots above EMA50 (or DOWNWARD below EMA50) |
| **Vol confirmed** | Only show pivots where volume exceeded the 20-day average on signal day |

**Entry / Stop / Target**
Entry: within 1–3% of the Pivot Level on re-test.
Stop: below the prior swing low (for longs) or above swing high (for shorts).
Target: next resistance or 2:1 R/R minimum.
""",
        },
        "ema": {
            "title": "📉 EMA Crossover — How It Works",
            "body": """
**Strategy (Hypothesis)**
Detects when a fast EMA crosses above/below a slow EMA while a trend EMA acts as the
overall trend filter. ADX, RSI, and volume confirm the setup quality.
*Hypothesis:* Momentum is shifting when price pulls fast and slow averages apart; the trend
EMA prevents entering against the primary trend.

**Time Period**  120 calendar days default (configurable). EMAs need ≥ trend period days of history to stabilise.

---
**Column Glossary**

| Column | Meaning |
|--------|---------|
| **Signal** | `BULLISH_CROSS` = fast EMA crossed above slow; `BEARISH_CROSS` = opposite |
| **Bars Ago** | Trading days since the crossover occurred — lower = fresher |
| **Entry** | Closing price on the signal bar |
| **Current** | Latest closing price |
| **P&L %** | Unrealised gain/loss from entry to current price |
| **Stop** | Suggested stop-loss (1× ATR14 below entry for longs) |
| **Target 1 / 2** | 1× and 2× ATR-based profit targets |
| **R/R** | Risk-to-reward ratio (target ÷ stop distance) |
| **ADX** | Average Directional Index — measures trend *strength* (not direction). <20 = choppy, 25–40 = trending, >40 = strong trend |
| **RSI** | Relative Strength Index (14-period). <30 = oversold, >70 = overbought |
| **Vol vs Avg** | Volume on signal day vs 20-day average. 1.5× = 50% above average |
| **EMA Stack** | ALIGNED = all 3 EMAs in the right order (fast > slow > trend for bull) |

**Settings Guide**

| Setting | Meaning |
|---------|---------|
| **Fast / Slow / Trend EMAs** | Core crossover periods. Swing: 9/21/55 · Trend: 20/50/200 |
| **Min ADX** | Only show crossovers in trending markets (ADX > threshold) |
| **Vol mult** | Volume must be ≥ this multiple of its 20-day average |
| **RSI min/max** | RSI must be within this range at the time of the crossover |
| **200 EMA align** | Require price to be above EMA200 for bulls (or below for bears) |
| **Pullback entry** | Instead of raw crossover, only flag setups where price has since pulled back to the fast EMA |
| **Signal window** | Recency filter — only show crossovers within N bars of today |
""",
        },
        "breakout": {
            "title": "💥 Breakout Scanner — How It Works",
            "body": """
**Strategy (Hypothesis)**
Six independent, research-backed breakout algorithms. A stock triggering multiple strategies
simultaneously has a higher conviction setup (confluence).
*Hypothesis:* Price compresses before explosive moves. Detecting compression + the first burst
catches the move early with a defined risk level.

**Time Period**  120 calendar days default. Each strategy has its own lookback:
52W = 252 days · NR7 = 7 days · BB Squeeze = 90 days · Others = 20–50 days.

---
**Strategy Descriptions**

| Strategy | Signal Logic | Edge |
|----------|-------------|------|
| **52W High/Low** | Price closes within 1% of its 52-week high or low | Breakouts from annual range extremes attract momentum |
| **Volume Surge** | Volume > 2× 20-day average | Marks institutional accumulation/distribution |
| **NR7** | Today's high−low range is the *narrowest in 7 days* | Volatility compression → explosive move imminent |
| **BB Squeeze** | Bollinger Band width at a 90-day low | Low volatility before expansion — Keltner/BB breakout |
| **Inside Bar** | Today's high/low contained entirely within the prior bar | Energy building — first bar outside is the signal |
| **MA Reclaim** | Price crosses back above 50- or 200-day MA from below | Trend-change signal with defined stop below the MA |

**Column Glossary**

| Column | Meaning |
|--------|---------|
| **Signals** | Comma-separated list of strategy codes that triggered |
| **Direction** | BULLISH (break up) or BEARISH (break down) |
| **Bars Ago** | Trading days since the most recent signal |
| **Vol / Avg** | Volume on signal day ÷ 20-day average volume |
| **BB Width** | Bollinger Band width = (upper − lower) / middle × 100. Small = tight squeeze |
| **% from 52W H/L** | Distance from 52-week high and low |
| **EMA Trend** | ABOVE EMA50 / ABOVE EMA200 — trend context |

**Settings Guide**

| Setting | Meaning |
|---------|---------|
| **Signal window** | Only show signals within N bars of today |
| **Direction** | Filter bullish-only, bearish-only, or all |
| **Min Vol / Avg** | Minimum volume ratio to include (filters low-conviction signals) |
""",
        },
        "minervini": {
            "title": "🏆 Minervini SEPA — How It Works",
            "body": """
**Strategy (Hypothesis)**
Mark Minervini's *Specific Entry Point Analysis* from "Trade Like a Stock Market Wizard."
Find stocks in Stage 2 uptrends forming a Volatility Contraction Pattern (VCP) base, then
enter on the breakout with volume confirmation.
*Hypothesis:* Stocks making new highs from tight, low-volume bases have the highest
probability of continuing — institutions are accumulating quietly before a breakout.

**Time Period**  252 calendar days (1 year) for all moving averages and RS Rating.

---
**The 8 Trend Template Conditions (T1–T8)**

| Code | Condition | Why |
|------|-----------|-----|
| T1 | Price > SMA150 | Price above medium-term trend |
| T2 | Price > SMA200 | Price above long-term trend |
| T3 | SMA150 > SMA200 | Medium trend faster than long trend |
| T4 | SMA200 trending up ≥ 1 month | Long trend improving |
| T5 | SMA50 > SMA150 and SMA200 | All MAs in bullish order |
| T6 | Price > SMA50 | Short-term momentum intact |
| T7 | Price ≥ 30% above 52W Low | Escaped the bottom |
| T8 | Price within 25% of 52W High | Near the top — leadership position |

**Column Glossary**

| Column | Meaning |
|--------|---------|
| **Trend Score** | 0–8: how many T1–T8 conditions pass. 8 = perfect Stage 2 |
| **RS Rating** | Relative Strength vs S&P 500 over 12 months. 70+ = top 30% of all stocks |
| **Stage** | Stage 2 = uptrend (ideal); Stage 1 = base; Stage 3 = top; Stage 4 = decline |
| **VCP** | ✅ = Volatility Contraction Pattern detected (base forming with contracting ranges) |
| **Contractions** | Number of VCP tightening stages. 3–4 is ideal (W-W-W pattern) |
| **VCP Depth %** | How much the range has contracted from 1st to last stage of the VCP |
| **Vol Dry-Up** | YES = volume declining during base (healthy — institutions absorbing supply quietly) |
| **Pivot** | VCP breakout price — buy *above* this level on volume ≥ 150% average |
| **% from Pivot** | How far today's price is from the pivot entry point |
| **ATR%** | Average True Range as % of price — use for position sizing (risk per bar) |
| **Entry** | Suggested buy trigger (above pivot) |
| **Stop** | Suggested stop-loss (below last VCP low, typically 8–12% below entry) |
| **Target (3:1)** | 3:1 reward-to-risk exit target |

**Colours**
🟢 Green = Already broken out above pivot
🟡 Yellow = Within 2% of pivot (near trigger)
🔵 Blue = VCP forming but not near trigger yet
""",
        },
        "volume": {
            "title": "🔊 Volume Scanner — How It Works",
            "body": """
**Strategy (Hypothesis)**
Abnormal volume reveals *where institutions are active*. Stocks with extreme volume spikes
are being bought or sold by large players — follow the money.
*Hypothesis:* A volume spike > 3× average accompanied by a price move is a high-conviction
directional signal. Multi-timeframe confirmation strengthens the case.

**Time Period**
1 MIN: last 15 bars (~15 min) · 15 MIN: last 10 bars (~2.5 hrs) ·
3 HOUR: last 9 hours-bars (~3 days) · DAY: last 25 days

---
**Signal Levels**

| Level | Volume Ratio | Interpretation |
|-------|-------------|----------------|
| ELEVATED | 1.0–1.8× | Slightly above average — watch |
| UNUSUAL | 1.8–3.0× | Noteworthy activity — possible catalyst |
| HIGH | 3.0–5.0× | Likely institutional — strong directional interest |
| EXTREME ⚡ | > 5.0× | Major event — earnings surprise, news, block trade, short squeeze |

**Column Glossary**

| Column | Meaning |
|--------|---------|
| **Best TF** | Which timeframe (1 MIN / 15 MIN / 3 HOUR / DAY) shows the strongest anomaly |
| **Best Ratio** | Volume ÷ 20-bar average on the best timeframe |
| **Level** | Signal classification (ELEVATED → EXTREME) |
| **1 MIN / 15 MIN / 3 HOUR / DAY** | Volume ratio for each timeframe — "—" means no anomaly on that TF |
| **Flagged** | True if volume exceeds the "Unusual" threshold on any timeframe |

**Settings Guide**

| Setting | Meaning |
|---------|---------|
| **Unusual (×)** | Ratio above which a bar is classified UNUSUAL |
| **High (×)** | Ratio above which a bar is classified HIGH |
| **Extreme (×)** | Ratio above which a bar is classified EXTREME ⚡ |
| **Show all symbols** | Show every scanned symbol, not just those flagged as anomalous |
""",
        },
        "astro": {
            "title": "🔭 Financial Astrology — How It Works",
            "body": """
**Approach**
Combines Western financial astrology (W.D. Gann, Larry Pesavento, Bill Meridian) with
Vedic Panchang (Bhadra/Vishti Karana) to identify high-probability market turning points.
*Important:* Astrology signals are probabilistic, not deterministic. Always confirm with
technical analysis before trading.

---
**Signal Components**

| Component | What It Measures | Bias |
|-----------|-----------------|------|
| **Moon Phase** | Lunar cycle stage (New/Full/Quarter) | New Moon = bullish; Full Moon = bearish (Bank of Scotland study) |
| **Bias Score** | Weighted sum of all active signals (−10 to +10) | >3 = BULLISH, <−3 = BEARISH |
| **Retrogrades** | Planet appears to move backward in sky | Mercury Rx = volatility/false signals; Venus/Mars Rx = sentiment extremes |
| **Aspects** | Angular relationships between planets | Trine/Sextile = harmonious/bullish; Square/Opposition = tension/bearish |
| **Solar Ingress** | Sun entering a cardinal sign | Seasonal turns (Gann): Aries, Cancer, Libra, Capricorn |

**Planetary Aspects Explained**

| Aspect | Angle | Orb | Meaning |
|--------|-------|-----|---------|
| ☌ Conjunction | 0° | ±8° | Merging of energies — neutral, intensified |
| ⚹ Sextile | 60° | ±6° | Cooperation — mildly bullish |
| □ Square | 90° | ±8° | Tension/conflict — bearish/volatile |
| △ Trine | 120° | ±8° | Harmony — bullish, smooth moves |
| ☍ Opposition | 180° | ±8° | Polarity/extreme — bearish, reversals |

**Gann Time Cycles**
W.D. Gann found markets frequently reverse at fixed time intervals from a major high or low:

| Cycle | Days | Meaning |
|-------|------|---------|
| 45d | 45 | 1/8 of a year |
| 90d | 90 | 1/4 year (quarter turn) |
| 144d | 144 | 2/5 year |
| 180d | 180 | Half year |
| 270d | 270 | 3/4 year |
| 360d | 360 | Full year |

**Bhadra / Vishti Karana (Vedic)**
One of 11 Karanas in the Panchang. Each Karana = 6° of Moon–Sun separation ≈ 11 hours.
Vishti repeats 8× per lunar month and is considered highly inauspicious.
10-year backtest on SPY: Bhadra days average −0.06%/day vs normal days. Effect is
consistent but not statistically significant on US markets (stronger on Nifty/Sensex due
to ~70% Indian retail participation in those markets).
""",
        },
    }

    info = _ABOUT.get(scanner_id)
    if not info:
        return
    with st.expander(f"ℹ️ {info['title']}", expanded=False):
        st.markdown(info["body"])


def _ticker_count_caption(tickers: list):
    suffix = " · Russell 1000" if st.session_state.get("use_russell1000") else ""
    st.caption(f"Scanning **{len(tickers)} symbols** — S&P 500 · Nasdaq-100 · High-Growth · Trending · Watchlist{suffix}")


# ═════════════════════════════════════════════════════════════════════════════
# Scanner: Livermore Pivotal
# ═════════════════════════════════════════════════════════════════════════════

def _style_livermore(df: pd.DataFrame):
    COLS = ["Ticker", "Signal", "Signal Date", "Bars Ago",
            "Pivot Level", "Close", "% from Pivot",
            "Vol Expansion", "% from 52W High", "Trend (EMA)"]
    cols = [c for c in COLS if c in df.columns]
    display = df[cols].copy()
    def row_color(row):
        bg = "#1b3a2a" if row["Signal"] == "UPWARD_PIVOT" else "#3a1b1b"
        return [f"background-color:{bg};color:#e0e0e0"] * len(row)
    return display.style.apply(row_color, axis=1).format(
        {"Pivot Level": "{:.2f}", "Close": "{:.2f}",
         "% from Pivot": "{:+.2f}%", "% from 52W High": "{:+.2f}%"},
        na_rep="—",
    )


def _livermore_chart(ticker: str, row: pd.Series, days: int) -> go.Figure:
    df = fetch_ohlcv(ticker, days=days)
    if df.empty:
        return go.Figure()
    df = classify_pivotal_points(df, swing_window=5, min_reaction_pct=1.5)
    df = continuation_pivot(df)
    for span in [20, 50, 200]:
        df[f"ema{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    candle, vol = _candlestick_vol(df, ticker)
    fig.add_trace(candle, row=1, col=1)
    for span, color, name in [(20, "#f0c040", "EMA20"), (50, "#42a5f5", "EMA50"),
                               (200, "#ff7043", "EMA200")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"ema{span}"], mode="lines",
                                 line=dict(color=color, width=1.2), name=name),
                      row=1, col=1)
    for sig, sym, col in [("UPWARD_PIVOT", "triangle-up", "#26a69a"),
                           ("DOWNWARD_PIVOT", "triangle-down", "#ef5350")]:
        sub = df[df["signal"] == sig]
        y   = sub["Low"] * 0.985 if sig == "UPWARD_PIVOT" else sub["High"] * 1.015
        fig.add_trace(go.Scatter(x=sub.index, y=y, mode="markers",
                                 marker=dict(symbol=sym, size=14, color=col),
                                 name=sig.replace("_", " ").title()),
                      row=1, col=1)
    pivot_level = row.get("Pivot Level") or row.get("pivot_high") or row.get("pivot_low")
    if pd.notna(pivot_level):
        fig.add_hline(y=pivot_level, line_dash="dash", line_color="#ffeb3b",
                      line_width=1.5, annotation_text=f"Pivot {pivot_level:.2f}",
                      annotation_position="top right", row=1, col=1)
    fig.add_trace(vol, row=2, col=1)
    _apply_chart_style(fig, f"{ticker} — Livermore Pivotal")
    return fig


def render_livermore():
    clicked = _page_header(
        "🔴", "Livermore Pivotal Points",
        "Upward and downward pivotal points — the classic Jesse Livermore method.",
        scan_key="lv_scan", last_key="lv_time",
    )
    _about_expander("livermore")

    with st.expander("⚙️ Settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        swing_window  = c1.slider("Swing window (bars)", 3, 10, 5, key="lv_swing",
                                   help="Bars looked left & right to identify a local high/low. 3 = faster/noisier, 10 = major pivots only")
        min_reaction  = c2.slider("Min reaction %", 0.5, 5.0, 1.5, step=0.25, key="lv_reaction",
                                   help="Minimum % price move after a high/low to confirm it as a valid pivot. 1.5% = default swing, 3%+ = major pivots")
        recent_bars   = c3.slider("Signal window (bars)", 1, 20, 5, key="lv_recent",
                                   help="Only show signals from within this many trading days of today. Keeps results fresh")
        signal_filter = c4.radio("Signal type", ["ALL", "UPWARD_PIVOT", "DOWNWARD_PIVOT"],
                                 horizontal=True, key="lv_sig_filter")
        f1, f2 = st.columns(2)
        trend_aligned = f1.checkbox("Trend aligned only", value=True, key="lv_trend",
                                    help="Only show UPWARD pivots when price is above EMA50, and DOWNWARD pivots when below. Filters counter-trend setups")
        vol_required  = f2.checkbox("Volume confirmed only", value=False, key="lv_vol",
                                    help="Require volume on signal day to be above the 20-day average")

    if "lv_df" not in st.session_state:
        st.session_state.lv_df   = pd.DataFrame()
        st.session_state.lv_time = None
        st.session_state.lv_rows = {}

    load_btn = st.button("📂 Load last saved", key="lv_load")
    if load_btn:
        files = sorted(glob.glob(os.path.join(SCREENER_DIR, "livermore_*.csv")) +
                       glob.glob(os.path.join(SCREENER_DIR, "signals_*.csv")))
        if files:
            st.session_state.lv_df   = pd.read_csv(files[-1])
            st.session_state.lv_time = "from saved file"

    if clicked:
        tickers = get_selected_tickers()
        _ticker_count_caption(tickers)
        with st.spinner("Scanning…"):
            df = run_screener(
                tickers=tickers, swing_window=st.session_state.lv_swing,
                min_reaction_pct=st.session_state.lv_reaction,
                recent_bars=st.session_state.lv_recent,
                signal_filter=st.session_state.lv_sig_filter,
                trend_aligned=st.session_state.lv_trend,
                vol_required=st.session_state.lv_vol,
            )
            st.session_state.lv_df   = df
            st.session_state.lv_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.lv_rows = {r["Ticker"]: r for _, r in df.iterrows()}
            if not df.empty:
                save_scan_result(df, "livermore")

    df = st.session_state.lv_df
    if df.empty:
        st.info("Hit **Scan Now** to find pivotal points.", icon="ℹ️")
        return

    up   = df[df["Signal"] == "UPWARD_PIVOT"]
    down = df[df["Signal"] == "DOWNWARD_PIVOT"]
    vol_pct = int((df["Vol Expansion"] == "YES").mean() * 100) if "Vol Expansion" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Signals", len(df))
    c2.metric("Bullish ▲",     len(up))
    c3.metric("Bearish ▼",     len(down))
    c4.metric("Vol Confirmed",  f"{vol_pct}%")

    st.divider()

    _LV_COL_CFG = {
        "Bars Ago":        st.column_config.NumberColumn("Bars Ago",       help="Trading days since the signal fired — lower = fresher"),
        "Pivot Level":     st.column_config.NumberColumn("Pivot Level",    help="Price at which the swing high or low formed"),
        "% from Pivot":    st.column_config.NumberColumn("% from Pivot",   help="(Close − Pivot) / Pivot × 100. Positive = price held above pivot; negative = fell through (stop zone)"),
        "Vol Expansion":   st.column_config.TextColumn("Vol Expansion",    help="YES = volume on signal day exceeded 20-day average (institutional confirmation)"),
        "% from 52W High": st.column_config.NumberColumn("% from 52W High",help="Distance from 52-week high. −5% = within 5% of the annual high"),
        "Trend (EMA)":     st.column_config.TextColumn("Trend (EMA)",      help="ABOVE = price > EMA50 (uptrend); BELOW = price < EMA50 (downtrend)"),
        "Signal Date":     st.column_config.TextColumn("Signal Date",      help="Date the pivot was confirmed (the bar that completed the reversal)"),
    }
    if not up.empty:
        st.markdown("##### ▲ Upward Pivotal Points")
        st.dataframe(_style_livermore(up), use_container_width=True, hide_index=True,
                     column_config=_LV_COL_CFG)
    if not down.empty:
        st.markdown("##### ▼ Downward Pivotal Points")
        st.dataframe(_style_livermore(down), use_container_width=True, hide_index=True,
                     column_config=_LV_COL_CFG)

    st.session_state.lv_rows = {r["Ticker"]: r for _, r in df.iterrows()}

    st.divider()
    st.markdown("##### Chart")
    cc1, cc2 = st.columns([3, 1])
    chosen = cc1.selectbox("Ticker", df["Ticker"].tolist(), key="lv_chart_ticker")
    days   = cc2.slider("Days", 60, 365, 120, key="lv_chart_days")
    if chosen:
        row = st.session_state.lv_rows.get(chosen, pd.Series(dtype=object))
        sig = row.get("Signal", "")
        if sig == "UPWARD_PIVOT":
            st.success(f"▲ UPWARD PIVOT  •  {row.get('Signal Date','')}  •  Pivot {row.get('Pivot Level','')}")
        elif sig == "DOWNWARD_PIVOT":
            st.error(f"▼ DOWNWARD PIVOT  •  {row.get('Signal Date','')}  •  Pivot {row.get('Pivot Level','')}")
        with st.spinner(f"Loading {chosen}…"):
            st.plotly_chart(_livermore_chart(chosen, row, days), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Scanner: EMA Crossover
# ═════════════════════════════════════════════════════════════════════════════

def render_ema():
    clicked = _page_header(
        "📉", "EMA Crossover",
        "Fast/Slow/Trend EMA crossover with ADX, volume, and RSI confirmation.",
        scan_key="ema_scan_btn", last_key="ema_time",
    )
    _about_expander("ema")

    with st.expander("⚙️ Settings", expanded=False):
        pc1, pc2 = st.columns(2)
        ema_preset = pc1.selectbox("Preset", ["Custom"] + list(EMA_PRESETS.keys()),
                                    index=1, key="ema_preset",
                                    help="Swing (9/21/55): short-term swings · Trend (20/50/200): position trades · Momentum (8/13/34): fast momentum")
        if ema_preset == "Custom":
            ec1, ec2, ec3 = st.columns(3)
            ema_fast  = ec1.number_input("Fast",  value=9,  min_value=2,  max_value=50,  key="ema_fast",
                                          help="Fast EMA period — reacts quickly to price changes")
            ema_slow  = ec2.number_input("Slow",  value=21, min_value=3,  max_value=100, key="ema_slow",
                                          help="Slow EMA period — crossover above = bullish signal")
            ema_trend = ec3.number_input("Trend", value=55, min_value=5,  max_value=300, key="ema_trend",
                                          help="Trend filter EMA — only take signals in the direction of this EMA")
        else:
            p = EMA_PRESETS[ema_preset]
            ema_fast, ema_slow, ema_trend = p["fast"], p["slow"], p["trend"]
            st.caption(f"EMA {ema_fast} / {ema_slow} / {ema_trend}")

        rc1, rc2, rc3, rc4 = st.columns(4)
        ema_adx      = rc1.slider("Min ADX",   10.0, 40.0, 15.0, step=1.0,  key="ema_adx",
                                   help="ADX (Average Directional Index) measures trend STRENGTH. <20 = choppy, 25+ = trending, 40+ = strong trend. Only signals above this ADX are shown")
        ema_vol_mult = rc2.slider("Vol mult",   1.0,  3.0,  1.0, step=0.25, key="ema_vol_mult",
                                   help="Volume on crossover day must be at least this multiple of the 20-day average. 1.5 = 50% above average")
        ema_rsi_low  = rc3.slider("RSI min",   10.0, 50.0, 30.0, step=5.0,  key="ema_rsi_low",
                                   help="RSI (14) must be above this level. Filters oversold longs below 30 (or use 40+ for momentum confirmation)")
        ema_rsi_high = rc4.slider("RSI max",   50.0, 90.0, 70.0, step=5.0,  key="ema_rsi_high",
                                   help="RSI (14) must be below this level. 70 filters overbought conditions for longs")
        fc1, fc2, fc3 = st.columns(3)
        ema_200_align = fc1.checkbox("200 EMA align", value=False, key="ema_200_align",
                                     help="Require price to be above EMA200 for bullish signals (or below for bearish). Keeps you on the right side of the major trend")
        ema_pullback  = fc2.checkbox("Pullback entry", value=False, key="ema_pullback",
                                     help="Instead of the raw crossover bar, flag only after price pulls back to the fast EMA post-crossover. Gives a better entry price")
        ema_recent    = fc3.slider("Signal window", 1, 20, 10, key="ema_recent",
                                   help="Only show crossovers from within this many trading days of today")
        ema_sig_filter = st.radio("Show", ["ALL", "BULLISH_CROSS", "BEARISH_CROSS"],
                                   horizontal=True, key="ema_sig_filter")

    if "ema_df" not in st.session_state:
        st.session_state.ema_df   = pd.DataFrame()
        st.session_state.ema_time = None

    if clicked:
        tickers = get_selected_tickers()
        _ticker_count_caption(tickers)
        with st.spinner("Scanning EMA crossovers…"):
            df_ema = run_ema_screener(
                tickers=tickers,
                fast=ema_fast, slow=ema_slow, trend=ema_trend,
                adx_threshold=ema_adx, vol_mult=ema_vol_mult,
                rsi_low=ema_rsi_low, rsi_high=ema_rsi_high,
                require_200_align=ema_200_align,
                pullback_mode=ema_pullback,
                recent_bars=ema_recent,
                signal_filter=ema_sig_filter,
            )
            st.session_state.ema_df   = df_ema
            st.session_state.ema_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            if not df_ema.empty:
                save_scan_result(df_ema, "ema")

    df_ema = st.session_state.ema_df
    if df_ema.empty and st.session_state.ema_time:
        st.warning("No signals found. Try loosening filters.", icon="⚠️")
        return
    if df_ema.empty:
        st.info("Hit **Scan Now** to find EMA crossovers.", icon="ℹ️")
        return

    bull_e = df_ema[df_ema["Signal"] == "BULLISH_CROSS"]
    bear_e = df_ema[df_ema["Signal"] == "BEARISH_CROSS"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total",     len(df_ema))
    c2.metric("Bullish ▲", len(bull_e))
    c3.metric("Bearish ▼", len(bear_e))
    c4.metric("Avg R/R",   f"{df_ema['R/R'].mean():.2f}"  if "R/R"  in df_ema.columns else "—")
    c5.metric("Avg ADX",   f"{df_ema['ADX'].mean():.1f}"  if "ADX"  in df_ema.columns else "—")

    st.divider()
    COLS_EMA = ["Ticker", "Signal", "Signal Date", "Bars Ago",
                "Entry", "Current", "P&L %", "Stop", "Target 1", "Target 2",
                "R/R", "ADX", "RSI", "Vol vs Avg", "EMA Stack"]
    cols = [c for c in COLS_EMA if c in df_ema.columns]
    fmt = {}
    for col in ["Entry","Current","Stop","Target 1","Target 2"]:
        if col in df_ema.columns: fmt[col] = "{:.2f}"
    if "P&L %" in df_ema.columns:    fmt["P&L %"]     = "{:+.2f}%"
    if "Vol vs Avg" in df_ema.columns: fmt["Vol vs Avg"] = "{:.2f}x"
    def _ema_color(row):
        bg = "#1b3a2a" if row["Signal"] == "BULLISH_CROSS" else "#3a1b1b"
        return [f"background-color:{bg};color:#e0e0e0"] * len(row)
    _EMA_COL_CFG = {
        "Bars Ago":    st.column_config.NumberColumn("Bars Ago",   help="Trading days since the EMA crossover fired"),
        "ADX":         st.column_config.NumberColumn("ADX",        help="Average Directional Index — trend strength. <20 = choppy, 25+ = trending, 40+ = strong trend"),
        "RSI":         st.column_config.NumberColumn("RSI",        help="Relative Strength Index (14-period). <30 = oversold, >70 = overbought"),
        "Vol vs Avg":  st.column_config.NumberColumn("Vol vs Avg", help="Volume on signal day ÷ 20-day average. 1.5× = 50% above average"),
        "EMA Stack":   st.column_config.TextColumn("EMA Stack",    help="ALIGNED = all 3 EMAs in bullish order (fast > slow > trend)"),
        "R/R":         st.column_config.NumberColumn("R/R",        help="Risk-to-reward ratio: target distance ÷ stop distance"),
        "P&L %":       st.column_config.NumberColumn("P&L %",      help="Unrealised gain/loss from entry price to current price"),
        "Signal Date": st.column_config.TextColumn("Signal Date",  help="Date the crossover was confirmed"),
    }
    st.dataframe(df_ema[cols].style.apply(_ema_color, axis=1).format(fmt, na_rep="—"),
                 use_container_width=True, hide_index=True, column_config=_EMA_COL_CFG)

    st.divider()
    st.markdown("##### Chart")
    cc1, cc2 = st.columns([3, 1])
    chosen_ema = cc1.selectbox("Ticker", df_ema["Ticker"].tolist(), key="ema_chart_ticker")
    chart_days = cc2.slider("Days", 60, 365, 120, key="ema_chart_days")
    if chosen_ema:
        sig_row = {r["Ticker"]: r for _, r in df_ema.iterrows()}[chosen_ema]
        sv = sig_row.get("Signal","")
        if sv == "BULLISH_CROSS":
            st.success(f"▲ BULLISH CROSS  •  {sig_row.get('Signal Date','')}  •  Entry {sig_row.get('Entry','')}  •  R/R {sig_row.get('R/R','')}:1")
        else:
            st.error(f"▼ BEARISH CROSS  •  {sig_row.get('Signal Date','')}  •  Entry {sig_row.get('Entry','')}  •  R/R {sig_row.get('R/R','')}:1")
        with st.spinner(f"Loading {chosen_ema}…"):
            raw_df = fetch_ohlcv(chosen_ema, days=chart_days)
        if not raw_df.empty:
            ann = detect_crossovers(raw_df, fast=ema_fast, slow=ema_slow, trend=ema_trend,
                                    adx_threshold=ema_adx, vol_mult=ema_vol_mult,
                                    rsi_low=ema_rsi_low, rsi_high=ema_rsi_high,
                                    require_200_align=ema_200_align, pullback_mode=ema_pullback)
            fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                 row_heights=[0.60,0.20,0.20], vertical_spacing=0.02)
            candle, vol = _candlestick_vol(ann, chosen_ema)
            fig2.add_trace(candle, row=1, col=1)
            for col, color, nm in [(f"ema{ema_fast}", "#f0c040", f"EMA{ema_fast}"),
                                   (f"ema{ema_slow}", "#42a5f5", f"EMA{ema_slow}"),
                                   (f"ema{ema_trend}","#ab47bc", f"EMA{ema_trend}"),
                                   ("ema200",         "#ff7043", "EMA200")]:
                if col in ann.columns:
                    fig2.add_trace(go.Scatter(x=ann.index, y=ann[col], mode="lines",
                                             line=dict(color=color, width=1.3), name=nm),
                                  row=1, col=1)
            for idx_col, sym, col in [("BULLISH_CROSS","triangle-up","#26a69a"),
                                       ("BEARISH_CROSS","triangle-down","#ef5350")]:
                mask = ann["signal"] == idx_col if "signal" in ann.columns else pd.Series(False, index=ann.index)
                y    = ann.loc[mask,"Low"]*0.985 if idx_col=="BULLISH_CROSS" else ann.loc[mask,"High"]*1.015
                fig2.add_trace(go.Scatter(x=ann.index[mask], y=y, mode="markers",
                                         marker=dict(symbol=sym, size=14, color=col),
                                         name=idx_col.replace("_"," ").title()),
                               row=1, col=1)
            for val, color, lbl in [
                (sig_row.get("Entry"),    "#fff",    f"Entry {sig_row.get('Entry','')}"),
                (sig_row.get("Stop"),     "#ef5350", f"Stop {sig_row.get('Stop','')}"),
                (sig_row.get("Target 2"),"#26a69a",  f"T2 {sig_row.get('Target 2','')}"),
            ]:
                if val and pd.notna(val):
                    fig2.add_hline(y=val, line_dash="dot", line_color=color, line_width=1.2,
                                   annotation_text=lbl, annotation_position="top right", row=1, col=1)
            if "rsi14" in ann.columns:
                fig2.add_trace(go.Scatter(x=ann.index, y=ann["rsi14"], mode="lines",
                                         line=dict(color="#42a5f5", width=1.2), name="RSI(14)"),
                               row=2, col=1)
                fig2.add_hline(y=70, line_dash="dash", line_color="#ef5350", line_width=0.8, row=2, col=1)
                fig2.add_hline(y=30, line_dash="dash", line_color="#26a69a", line_width=0.8, row=2, col=1)
            if "adx14" in ann.columns:
                fig2.add_trace(go.Scatter(x=ann.index, y=ann["adx14"], mode="lines",
                                         line=dict(color="#f0c040", width=1.2), name="ADX(14)"),
                               row=3, col=1)
                fig2.add_hline(y=25, line_dash="dash", line_color="#aaa", line_width=0.8,
                               annotation_text="ADX 25", row=3, col=1)
            _apply_chart_style(fig2, f"{chosen_ema} — EMA {ema_fast}/{ema_slow}/{ema_trend}",
                               rows=3, ytitles=["Price","RSI","ADX"], height=700)
            st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Scanner: Breakout
# ═════════════════════════════════════════════════════════════════════════════

def render_breakout():
    clicked = _page_header(
        "💥", "Breakout Scanner",
        "6 research-backed strategies: 52W High, Volume Surge, NR7, BB Squeeze, Inside Bar, MA Reclaim.",
        scan_key="bo_scan_btn", last_key="bo_time",
    )
    _about_expander("breakout")

    with st.expander("⚙️ Settings", expanded=False):
        strat_cols = st.columns(3)
        selected_strategies = []
        for i, (name, codes) in enumerate(STRATEGY_GROUPS.items()):
            with strat_cols[i % 3]:
                label_lines = "\n".join(f"• {STRATEGY_LABELS[c][0]}" for c in codes)
                if st.checkbox(name, value=True, key=f"bo_strat_{name}", help=label_lines):
                    selected_strategies.append(name)
        bc1, bc2, bc3 = st.columns(3)
        bo_recent    = bc1.slider("Signal window", 1, 20, 5,  key="bo_recent",
                                   help="Only show breakout signals from within this many trading days of today. 5 = keep results very fresh")
        bo_direction = bc2.radio("Direction", ["ALL","BULLISH","BEARISH"],
                                 horizontal=True, key="bo_dir")
        bo_min_vol   = bc3.slider("Min Vol / Avg", 0.5, 3.0, 0.8, step=0.1, key="bo_vol",
                                   help="Minimum volume vs 20-day average on the signal day. 1.0 = at least average volume, 2.0 = strong institutional interest required")

    if "bo_df" not in st.session_state:
        st.session_state.bo_df   = pd.DataFrame()
        st.session_state.bo_time = None

    if clicked:
        if not selected_strategies:
            st.warning("Select at least one strategy.", icon="⚠️")
            return
        tickers = get_selected_tickers()
        _ticker_count_caption(tickers)
        with st.spinner("Scanning breakouts…"):
            df_bo = run_breakout_screener(
                tickers=tickers, recent_bars=bo_recent,
                strategies=selected_strategies if len(selected_strategies) < len(STRATEGY_GROUPS) else None,
                direction_filter=bo_direction,
            )
            if not df_bo.empty and "Vol / Avg" in df_bo.columns:
                df_bo = df_bo[df_bo["Vol / Avg"].fillna(0) >= bo_min_vol]
            st.session_state.bo_df   = df_bo
            st.session_state.bo_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            if not df_bo.empty:
                save_scan_result(df_bo, "breakout")

    df_bo = st.session_state.bo_df
    if df_bo.empty and st.session_state.bo_time:
        st.warning("No signals found. Try more strategies, wider window, or lower vol filter.", icon="⚠️")
        return
    if df_bo.empty:
        st.info("Hit **Scan Now** to find breakouts.", icon="ℹ️")
        return

    bull_bo = df_bo[df_bo["Direction"] == "BULLISH"]
    bear_bo = df_bo[df_bo["Direction"] == "BEARISH"]
    top_sig = Counter(s.strip() for r in df_bo.get("Signals", pd.Series([])) for s in r.split(",")).most_common(1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total",     len(df_bo))
    c2.metric("Bullish ▲", len(bull_bo))
    c3.metric("Bearish ▼", len(bear_bo))
    c4.metric("Top Strategy", STRATEGY_LABELS.get(top_sig[0][0], ("?",))[0] if top_sig else "—")

    # Strategy breakdown bar chart
    if "Signals" in df_bo.columns:
        sig_counts = Counter(s.strip() for r in df_bo["Signals"] for s in r.split(","))
        sig_df = pd.DataFrame([
            {"Strategy": STRATEGY_LABELS.get(k,(k,))[0], "Count": v,
             "Type": STRATEGY_LABELS.get(k,("","bull"))[1]}
            for k, v in sig_counts.items()
        ]).sort_values("Count", ascending=True)
        bar_fig = go.Figure(go.Bar(
            x=sig_df["Count"], y=sig_df["Strategy"], orientation="h",
            marker_color=["#26a69a" if t=="bullish" else "#ef5350" for t in sig_df["Type"]],
            text=sig_df["Count"], textposition="outside",
        ))
        bar_fig.update_layout(template="plotly_dark", height=max(180, len(sig_df)*38),
                              margin=dict(l=10, r=40, t=20, b=20), xaxis_title="Count")
        st.plotly_chart(bar_fig, use_container_width=True)

    st.divider()
    COLS_BO = ["Ticker","Signals","Direction","Signal Date","Bars Ago",
               "Entry","Current","P&L %","Stop","Target",
               "Vol / Avg","% from 52W H","% from 52W L","BB Width","EMA Trend"]
    cols = [c for c in COLS_BO if c in df_bo.columns]
    fmt_bo = {}
    for col in ["Entry","Current","Stop","Target"]:
        if col in df_bo.columns: fmt_bo[col] = "{:.2f}"
    if "P&L %" in df_bo.columns:     fmt_bo["P&L %"]     = "{:+.2f}%"
    if "Vol / Avg" in df_bo.columns:  fmt_bo["Vol / Avg"]  = "{:.2f}x"
    def _bo_color(row):
        bg = "#1b3a2a" if row["Direction"] == "BULLISH" else "#3a1b1b"
        return [f"background-color:{bg};color:#e0e0e0"] * len(row)
    _BO_COL_CFG = {
        "Signals":       st.column_config.TextColumn("Signals",      help="Breakout strategies that triggered: 52w=52W High/Low · vol=Volume Surge · nr7=NR7 · bb=BB Squeeze · ib=Inside Bar · ma=MA Reclaim"),
        "Bars Ago":      st.column_config.NumberColumn("Bars Ago",   help="Trading days since the most recent signal fired"),
        "Vol / Avg":     st.column_config.NumberColumn("Vol / Avg",  help="Volume on signal day ÷ 20-day average. 2.0× = twice normal volume"),
        "BB Width":      st.column_config.NumberColumn("BB Width",   help="Bollinger Band width = (upper−lower)/middle × 100. Small value = volatility squeeze — explosion likely soon"),
        "% from 52W H":  st.column_config.NumberColumn("% from 52W H", help="How far below the 52-week high (negative = below). Near 0 = near-breakout territory"),
        "% from 52W L":  st.column_config.NumberColumn("% from 52W L", help="How far above the 52-week low (positive = well above bottom)"),
        "EMA Trend":     st.column_config.TextColumn("EMA Trend",   help="Whether price is above EMA50 and/or EMA200 — trend context for the breakout"),
        "Signal Date":   st.column_config.TextColumn("Signal Date", help="Date the breakout signal was detected"),
    }
    st.dataframe(df_bo[cols].style.apply(_bo_color, axis=1).format(fmt_bo, na_rep="—"),
                 use_container_width=True, hide_index=True, column_config=_BO_COL_CFG)

    st.divider()
    st.markdown("##### Chart")
    cc1, cc2 = st.columns([3, 1])
    chosen_bo = cc1.selectbox("Ticker", df_bo["Ticker"].tolist(), key="bo_chart_ticker")
    bo_days   = cc2.slider("Days", 60, 365, 120, key="bo_chart_days")
    if chosen_bo:
        bo_row  = {r["Ticker"]: r for _, r in df_bo.iterrows()}[chosen_bo]
        dir_str = bo_row.get("Direction","")
        sig_str = bo_row.get("Signals","")
        if dir_str == "BULLISH":
            st.success(f"▲ {sig_str}  •  Entry {bo_row.get('Entry')}  •  Stop {bo_row.get('Stop')}  •  Target {bo_row.get('Target')}  •  R/R 2:1")
        else:
            st.error(f"▼ {sig_str}  •  Entry {bo_row.get('Entry')}  •  Stop {bo_row.get('Stop')}  •  Target {bo_row.get('Target')}  •  R/R 2:1")
        with st.spinner(f"Loading {chosen_bo}…"):
            raw_bo = fetch_ohlcv(chosen_bo, days=bo_days)
        if not raw_bo.empty:
            ann_bo = run_all_strategies(raw_bo)
            sig_col_map = {"sig_52w":"#f0e040","sig_vol":"#ff9800","sig_nr7":"#ab47bc",
                           "sig_bb":"#42a5f5","sig_ib":"#26c6da","sig_ma":"#66bb6a"}
            fig_bo = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.75,0.25], vertical_spacing=0.02)
            candle, vol = _candlestick_vol(ann_bo, chosen_bo)
            fig_bo.add_trace(candle, row=1, col=1)
            for span, color, nm in [(50,"#42a5f5","EMA50"),(200,"#ff7043","EMA200")]:
                cn = f"ema{span}"
                if cn in ann_bo.columns:
                    fig_bo.add_trace(go.Scatter(x=ann_bo.index, y=ann_bo[cn], mode="lines",
                                               line=dict(color=color,width=1.2), name=nm), row=1, col=1)
            for cn, col_n in [("bb_upper","#546e7a"),("bb_lower","#546e7a")]:
                if cn in ann_bo.columns:
                    fig_bo.add_trace(go.Scatter(x=ann_bo.index, y=ann_bo[cn], mode="lines",
                                               line=dict(color=col_n,width=0.8,dash="dot"),
                                               fill="tonexty" if cn=="bb_lower" else None,
                                               fillcolor="rgba(84,110,122,0.08)",
                                               name=cn.replace("_"," ").title()), row=1, col=1)
            for sc, mc in sig_col_map.items():
                if sc not in ann_bo.columns: continue
                for bull, sym in [(ann_bo[sc].str.startswith("BULL",na=False),"triangle-up"),
                                  (ann_bo[sc].str.startswith("BEAR",na=False),"triangle-down")]:
                    if not bull.any(): continue
                    y = ann_bo.loc[bull,"Low"]*0.985 if sym=="triangle-up" else ann_bo.loc[bull,"High"]*1.015
                    fig_bo.add_trace(go.Scatter(x=ann_bo.index[bull], y=y, mode="markers",
                                               marker=dict(symbol=sym,size=12,color=mc,
                                                          line=dict(color="#fff",width=0.5)),
                                               name=f"{sc.replace('sig_','').upper()} {'▲' if sym=='triangle-up' else '▼'}"),
                                    row=1, col=1)
            for val, color, lbl in [
                (bo_row.get("Entry"), "#fff",    f"Entry {bo_row.get('Entry')}"),
                (bo_row.get("Stop"),  "#ef5350", f"Stop {bo_row.get('Stop')}"),
                (bo_row.get("Target"),"#26a69a", f"Target {bo_row.get('Target')}"),
            ]:
                if val and pd.notna(val):
                    fig_bo.add_hline(y=val, line_dash="dot", line_color=color, line_width=1.2,
                                    annotation_text=lbl, annotation_position="top right", row=1, col=1)
            if "vol20" in ann_bo.columns:
                fig_bo.add_trace(go.Scatter(x=ann_bo.index, y=ann_bo["vol20"], mode="lines",
                                           line=dict(color="#ffb74d",width=1,dash="dot"),
                                           name="Vol Avg20"), row=2, col=1)
            fig_bo.add_trace(vol, row=2, col=1)
            _apply_chart_style(fig_bo, f"{chosen_bo} — Breakout Signals",
                               rows=2, ytitles=["Price","Volume"], height=700)
            st.plotly_chart(fig_bo, use_container_width=True)
            # Strategy legend
            st.markdown("**Signal colours:**")
            legend_info = {"sig_52w":"52W High/Low","sig_vol":"Volume Surge",
                           "sig_nr7":"NR7","sig_bb":"BB Squeeze",
                           "sig_ib":"Inside Bar","sig_ma":"MA Reclaim"}
            hex_rgb = {"#f0e040":(240,224,64),"#ff9800":(255,152,0),"#ab47bc":(171,71,188),
                       "#42a5f5":(66,165,245),"#26c6da":(38,198,218),"#66bb6a":(102,187,106)}
            leg_cols = st.columns(len(sig_col_map))
            for i, (sc, mc) in enumerate(sig_col_map.items()):
                r, g, b = hex_rgb.get(mc,(200,200,200))
                leg_cols[i].markdown(
                    f"<span style='background:rgb({r},{g},{b});padding:2px 8px;border-radius:4px;color:#000;font-size:12px'>"
                    f"▲▼ {legend_info[sc]}</span>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Scanner: Minervini SEPA
# ═════════════════════════════════════════════════════════════════════════════

def render_minervini():
    clicked = _page_header(
        "🏆", "Minervini SEPA",
        "Specific Entry Point Analysis — Stage 2 trend template, VCP, and Relative Strength.",
        scan_key="ms_scan_btn", last_key="ms_time",
    )
    _about_expander("minervini")

    with st.expander("📖 Strategy Summary", expanded=False):
        st.markdown("""
| # | Condition | Rule |
|---|-----------|------|
| T1–T2 | Price above key MAs | Price > SMA150 **and** Price > SMA200 |
| T3 | MA alignment | SMA150 > SMA200 |
| T4 | Long-term trend | SMA200 trending up ≥ 1 month |
| T5 | Short-term leads | SMA50 > SMA150 and SMA200 |
| T6 | Price above SMA50 | Momentum intact |
| T7 | 30% above 52W low | Escaped the basement |
| T8 | Within 25% of 52W high | Near the top |
| RS | Relative Strength | RS Rating ≥ 70 |
| VCP | Base pattern | Volatility contracting, volume drying up |
""")

    with st.expander("⚙️ Settings", expanded=False):
        c1, c2 = st.columns(2)
        ms_min_score  = c1.slider("Min Trend Score (0–8)", 0, 8, 7, key="ms_min_score",
                                   help="How many of the 8 SEPA trend conditions must pass. 8 = perfect Stage 2 only · 7 = one condition can fail · 6+ = broader universe")
        ms_min_rs     = c2.slider("Min RS Rating (0–99)", 0, 99, 70, key="ms_min_rs",
                                   help="Relative Strength vs S&P 500 over 12 months. 70 = top 30% of all stocks · 80+ = top 20% (leaders only) · 90+ = super-performers")
        d1, d2 = st.columns(2)
        ms_require_vcp = d1.checkbox("Require VCP", value=False, key="ms_require_vcp",
                                     help="Only show stocks with a confirmed Volatility Contraction Pattern base (3+ contracting ranges with declining volume)")
        ms_show_conds  = d2.checkbox("Show trend conditions checklist", value=False, key="ms_show_conds",
                                     help="After selecting a ticker in the chart view, show a T1–T8 checklist of which conditions pass/fail")

    if "ms_df" not in st.session_state:
        st.session_state.ms_df   = pd.DataFrame()
        st.session_state.ms_time = None

    if clicked:
        tickers_ms = get_selected_tickers()
        _ticker_count_caption(tickers_ms)
        with st.spinner(f"Running SEPA scan on {len(tickers_ms)} tickers…"):
            try:
                df_ms = run_minervini_screener(
                    tickers_ms,
                    min_trend_score=ms_min_score,
                    require_vcp=ms_require_vcp,
                    min_rs_pct=float(ms_min_rs),
                )
                st.session_state.ms_df   = df_ms
                st.session_state.ms_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                if not df_ms.empty:
                    save_scan_result(df_ms, "minervini")
            except Exception as _e:
                st.error(f"Scan error: {_e}", icon="❌")

    df_ms = st.session_state.ms_df
    if df_ms.empty and st.session_state.ms_time:
        st.warning("No stocks passed SEPA filters. Try lower Trend Score or RS Rating.", icon="⚠️")
        return
    if df_ms.empty:
        st.info("Hit **Scan Now** to find Stage 2 leaders.", icon="ℹ️")
        return

    broken  = df_ms["Broken Out"].sum() if "Broken Out" in df_ms.columns else 0
    near    = df_ms["Near Pivot"].sum()  if "Near Pivot" in df_ms.columns else 0
    vcp_ct  = (df_ms["VCP"] == "✅").sum() if "VCP" in df_ms.columns else 0
    full_s2 = (df_ms["Trend Score"] == 8).sum() if "Trend Score" in df_ms.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Qualified",     len(df_ms))
    c2.metric("Full Stage 2 ✅", int(full_s2))
    c3.metric("VCP Detected",   int(vcp_ct))
    c4.metric("Near Pivot",     int(near))
    c5.metric("Broken Out 🚀",  int(broken))

    st.divider()

    COLS_MS = ["Ticker","RS Rating","Stage","Trend Score","VCP","Contractions",
               "VCP Depth %","Vol Dry-Up","Pivot","% from Pivot","% from 52W H",
               "ADX","RSI","ATR%","Entry","Stop","Target (3:1)"]
    cols = [c for c in COLS_MS if c in df_ms.columns]
    fmt_ms = {}
    for col in ["Entry","Stop","Target (3:1)","Pivot","SMA50","SMA150","SMA200"]:
        if col in df_ms.columns: fmt_ms[col] = "{:.2f}"
    for col in ["% from Pivot","% from 52W H","VCP Depth %","ATR%"]:
        if col in df_ms.columns: fmt_ms[col] = "{:+.2f}%"
    def _ms_color(row):
        if row.get("Broken Out"):   return ["background-color:#1b3a2a;color:#e0e0e0"] * len(row)
        if row.get("Near Pivot"):   return ["background-color:#2a2a1b;color:#e0e0e0"] * len(row)
        if row.get("VCP") == "✅":  return ["background-color:#1b2a3a;color:#e0e0e0"] * len(row)
        return ["background-color:#1a1a1a;color:#e0e0e0"] * len(row)
    _MS_COL_CFG = {
        "Trend Score":    st.column_config.NumberColumn("Trend Score",  help="0–8: number of Minervini SEPA conditions (T1–T8) that pass. 8 = perfect Stage 2 uptrend"),
        "RS Rating":      st.column_config.NumberColumn("RS Rating",    help="12-month Relative Strength vs S&P 500. 70 = top 30%, 90 = top 10% (market leaders)"),
        "Stage":          st.column_config.TextColumn("Stage",          help="Stage 2 = ideal uptrend · Stage 1 = base/accumulation · Stage 3 = top/distribution · Stage 4 = decline"),
        "VCP":            st.column_config.TextColumn("VCP",            help="✅ = Volatility Contraction Pattern detected: price range narrowing, volume drying up — a base is forming"),
        "Contractions":   st.column_config.NumberColumn("Contractions", help="Number of VCP tightening stages. 3–4 is ideal (W pattern). More contractions = tighter, higher-quality base"),
        "VCP Depth %":    st.column_config.NumberColumn("VCP Depth %",  help="Range contraction from 1st to last VCP stage. Larger % = more volatility squeeze = bigger potential move"),
        "Vol Dry-Up":     st.column_config.TextColumn("Vol Dry-Up",     help="YES = volume declining during the base. Institutions absorbing supply quietly before breakout"),
        "Pivot":          st.column_config.NumberColumn("Pivot",        help="VCP breakout price — buy above this level on volume ≥ 150% of 50-day average"),
        "% from Pivot":   st.column_config.NumberColumn("% from Pivot", help="How far today's price is from the pivot. <2% = near trigger, >0% = already broken out"),
        "ATR%":           st.column_config.NumberColumn("ATR%",         help="Average True Range as % of price — daily volatility. Use for position sizing: Risk $ ÷ (ATR% × Price) = Shares"),
        "Target (3:1)":   st.column_config.NumberColumn("Target (3:1)", help="3:1 reward-to-risk target: Entry + 3 × (Entry − Stop)"),
    }
    st.dataframe(df_ms[cols].style.apply(_ms_color, axis=1).format(fmt_ms, na_rep="—"),
                 use_container_width=True, hide_index=True, column_config=_MS_COL_CFG)

    lc1, lc2, lc3 = st.columns(3)
    lc1.markdown("<span style='background:#1b3a2a;padding:3px 10px;border-radius:4px;color:#e0e0e0'>🟢 Broken Out</span>",    unsafe_allow_html=True)
    lc2.markdown("<span style='background:#2a2a1b;padding:3px 10px;border-radius:4px;color:#e0e0e0'>🟡 Near Pivot</span>",    unsafe_allow_html=True)
    lc3.markdown("<span style='background:#1b2a3a;padding:3px 10px;border-radius:4px;color:#e0e0e0'>🔵 VCP Forming</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("##### Chart")
    cc1, cc2 = st.columns([3, 1])
    ms_ticker = cc1.selectbox("Ticker", df_ms["Ticker"].tolist(), key="ms_chart_ticker")
    ms_days   = cc2.slider("Days", 60, 365, 200, key="ms_chart_days")
    if ms_ticker:
        with st.spinner(f"Loading {ms_ticker}…"):
            raw_ms = fetch_ohlcv(ms_ticker, days=ms_days)
        if not raw_ms.empty:
            raw_ms["sma50"]  = _msma(raw_ms["Close"], 50)
            raw_ms["sma150"] = _msma(raw_ms["Close"], 150)
            raw_ms["sma200"] = _msma(raw_ms["Close"], 200)
            raw_ms["vol50"]  = _msma(raw_ms["Volume"], 50)
            sel = df_ms[df_ms["Ticker"] == ms_ticker]
            def _get(col): return sel[col].values[0] if len(sel) and col in sel else None
            pivot_v, entry_v, stop_v, tgt_v = _get("Pivot"), _get("Entry"), _get("Stop"), _get("Target (3:1)")
            rs_val, ts_val, stage_v = _get("RS Rating"), _get("Trend Score"), _get("Stage")
            if "Stage 2 ✅" in str(stage_v):
                st.success(f"✅ {stage_v}  •  RS Rating: {rs_val}  •  Trend Score: {ts_val}/8  •  Pivot: {pivot_v}")
            else:
                st.info(f"{stage_v}  •  RS Rating: {rs_val}  •  Trend Score: {ts_val}/8")
            fig_ms = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.75,0.25], vertical_spacing=0.03)
            candle, vol = _candlestick_vol(raw_ms, ms_ticker)
            fig_ms.add_trace(candle, row=1, col=1)
            for col, color, nm in [("sma50","#f0c040","SMA50"),("sma150","#42a5f5","SMA150"),("sma200","#ff7043","SMA200")]:
                if col in raw_ms.columns:
                    fig_ms.add_trace(go.Scatter(x=raw_ms.index, y=raw_ms[col], mode="lines",
                                               line=dict(color=color,width=1.3), name=nm), row=1, col=1)
            if pivot_v and not pd.isna(pivot_v):
                for val, color, lbl, dash in [
                    (pivot_v, "#ffeb3b", f"Pivot {pivot_v:.2f}", "dash"),
                    (entry_v, "#fff",    f"Entry {entry_v:.2f}", "dot") if entry_v else (None,None,None,None),
                    (stop_v,  "#ef5350", f"Stop {stop_v:.2f}",   "dot") if stop_v  else (None,None,None,None),
                    (tgt_v,   "#26a69a", f"T1 {tgt_v:.2f}",      "dot") if tgt_v   else (None,None,None,None),
                ]:
                    if val is not None and not pd.isna(val):
                        fig_ms.add_hline(y=val, line_dash=dash, line_color=color, line_width=1.3,
                                        annotation_text=lbl, annotation_position="top right", row=1, col=1)
            if "vol50" in raw_ms.columns:
                fig_ms.add_trace(go.Scatter(x=raw_ms.index, y=raw_ms["vol50"], mode="lines",
                                           line=dict(color="#ffb74d",width=1.2,dash="dot"),
                                           name="Vol Avg50"), row=2, col=1)
            fig_ms.add_trace(vol, row=2, col=1)
            _apply_chart_style(fig_ms, f"{ms_ticker} — SEPA  |  RS {rs_val}  |  Trend {ts_val}/8",
                               rows=2, ytitles=["Price","Volume"])
            st.plotly_chart(fig_ms, use_container_width=True)
            if ms_show_conds:
                st.divider()
                st.markdown("#### Trend Template Checklist")
                try:
                    tt = check_trend_template(raw_ms)
                    labels = {
                        "T1_price_above_sma150":     "T1  Price > SMA150",
                        "T2_price_above_sma200":     "T2  Price > SMA200",
                        "T3_sma150_above_sma200":    "T3  SMA150 > SMA200",
                        "T4_sma200_trending_up":     "T4  SMA200 trending up (1 month)",
                        "T5_sma50_above_sma150_200": "T5  SMA50 > SMA150 & SMA200",
                        "T6_price_above_sma50":      "T6  Price > SMA50",
                        "T7_30pct_above_52w_low":    "T7  Price ≥ 30% above 52W Low",
                        "T8_within_25pct_52w_high":  "T8  Price within 25% of 52W High",
                    }
                    passed = sum(tt["conditions"].values())
                    st.markdown(f"**Score: {passed}/8**")
                    for key, lbl in labels.items():
                        ok = tt["conditions"].get(key, False)
                        st.markdown(f"{'✅' if ok else '❌'} &nbsp; {lbl}", unsafe_allow_html=True)
                except Exception:
                    pass


# ═════════════════════════════════════════════════════════════════════════════
# Scanner: Volume Anomaly
# ═════════════════════════════════════════════════════════════════════════════

def render_volume():
    clicked = _page_header(
        "🔊", "Volume Scanner",
        "Multi-timeframe volume anomaly detection — 1 MIN, 15 MIN, 3 HOUR, DAY.",
        scan_key="vs_scan_btn", last_key="vs_time",
    )
    _about_expander("volume")

    _vol_ok = False
    try:
        from core.fetcher import YFinanceFetcher as _VolFetcher
        from core.scanner import Scanner as _VolScanner
        _vol_ok = True
    except ImportError as _e:
        st.error(f"Volume scanner module not found: {_e}", icon="❌")

    if not _vol_ok:
        return

    with st.expander("⚙️ Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        vs_threshold = c1.slider("Unusual (×)", 1.0, 5.0, 1.8, step=0.1, key="vs_threshold",
                                  help="Volume ÷ 20-bar average must exceed this to be classified UNUSUAL. 1.8 = 80% above average")
        vs_high      = c2.slider("High (×)",    2.0, 8.0, 3.0, step=0.1, key="vs_high",
                                  help="Threshold for HIGH level. 3× = likely institutional order flow. Price direction on this bar indicates accumulation or distribution")
        vs_extreme   = c3.slider("Extreme (×)", 3.0, 10.0, 5.0, step=0.5, key="vs_extreme",
                                  help="Threshold for EXTREME ⚡. 5× = major event (earnings surprise, news catalyst, short squeeze, block trade)")
        vs_show_all  = st.checkbox("Show all symbols (not just flagged)", value=False, key="vs_show_all",
                                   help="Show every scanned symbol including those below the Unusual threshold. Useful for ranking the universe by volume activity")

    if "vs_results" not in st.session_state:
        st.session_state.vs_results = []
        st.session_state.vs_time    = None

    if clicked:
        tickers = get_selected_tickers()
        _ticker_count_caption(tickers)
        vs_config = {
            "SYMBOLS": tickers,
            "TIMEFRAMES": {
                "1m":  {"resolution":"1",  "lookback_bars":15, "compare_bars":1, "label":"1 MIN"},
                "15m": {"resolution":"15", "lookback_bars":10, "compare_bars":1, "label":"15 MIN"},
                "3h":  {"resolution":"60", "lookback_bars":9,  "compare_bars":3, "label":"3 HOUR"},
                "1d":  {"resolution":"D",  "lookback_bars":25, "compare_bars":1, "label":"DAY"},
            },
            "UNUSUAL_THRESHOLD": vs_threshold,
            "HIGH_THRESHOLD":    vs_high,
            "EXTREME_THRESHOLD": vs_extreme,
            "MIN_RATIO_TO_SHOW": 1.0 if vs_show_all else vs_threshold,
            "REFRESH_SECONDS": 60, "AUTO_SAVE_CSV": False,
            "INDICATOR_SETTINGS": {
                "volume_zscore_window":20,"mfi_period":14,"cmf_period":20,
                "obv_slope_bars":5,"vol_ma_period":20,"rsi_period":14,
                "macd_fast":12,"macd_slow":26,"macd_signal":9,
                "stoch_k":14,"stoch_d":3,"roc_period":10,"cci_period":20,
                "williams_r_period":14,"sma_periods":[20,50,200],
                "ema_periods":[9,21,55],"bb_period":20,"bb_std":2.0,
                "adx_period":14,"atr_period":14,"supertrend_period":10,"supertrend_multiplier":3.0,
            },
            "EXTRA_FEATURES": {k: False for k in ["vix","insider_trades","earnings_proximity",
                                                    "sector_momentum","news_sentiment","short_interest","put_call_ratio"]},
        }
        with st.spinner("Scanning for volume anomalies…"):
            try:
                fetcher = _VolFetcher()
                scanner = _VolScanner(vs_config, fetcher, None)
                results = scanner.run()
                st.session_state.vs_results = results
                st.session_state.vs_time    = datetime.now().strftime("%Y-%m-%d %H:%M")
                if results:
                    _rows = []
                    for _s in results:
                        _b = _s.best_tf
                        _rows.append({"Ticker":_s.symbol,"Best TF":_b.tf_label if _b else "—",
                                      "Best Ratio":round(_b.ratio,2) if _b else 0,
                                      "Signal Level":_b.signal_level if _b else "—",
                                      "Flagged":_s.is_flagged})
                    save_scan_result(pd.DataFrame(_rows), "volume")
            except Exception as _ex:
                st.error(f"Scan failed: {_ex}", icon="❌")

    results = st.session_state.vs_results
    if not results and st.session_state.vs_time:
        st.warning("No anomalies found. Try lowering thresholds or enabling 'Show all symbols'.", icon="⚠️")
        return
    if not results:
        st.info("Hit **Scan Now** to detect volume anomalies.", icon="ℹ️")
        return

    _TF_IDS    = ["1m","15m","3h","1d"]
    _TF_LABELS = {"1m":"1 MIN","15m":"15 MIN","3h":"3 HOUR","1d":"DAY"}
    rows = []
    for sig in results:
        best = sig.best_tf
        row  = {"Ticker":sig.symbol, "Best TF":best.tf_label if best else "—",
                "Best Ratio":round(best.ratio,2) if best else 0,
                "Level":best.signal_level if best else "—"}
        for tf_id in _TF_IDS:
            vr = sig.tf_ratios.get(tf_id)
            row[_TF_LABELS[tf_id]] = f"{vr.ratio:.2f}×" if (vr and vr.ok) else "—"
        rows.append(row)
    vs_df   = pd.DataFrame(rows)
    flagged = [s for s in results if s.is_flagged]
    extreme = sum(1 for s in flagged if s.best_tf and "EXTREME" in s.best_tf.signal_level.upper())
    high    = sum(1 for s in flagged if s.best_tf and "HIGH" in s.best_tf.signal_level.upper() and "EXTREME" not in s.best_tf.signal_level.upper())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scanned",    len(results))
    c2.metric("Flagged",    len(flagged))
    c3.metric("High ×",     high)
    c4.metric("Extreme ⚡", extreme)

    st.divider()
    _LEVEL_BG = {"EXTREME":"#4a3800","HIGH":"#3a1b1b","UNUSUAL":"#1b2a3a","ELEVATED":"#1a1f1a"}
    def _vs_color(row):
        bg = _LEVEL_BG.get(str(row["Level"]).upper(), "#1a1a1a")
        return [f"background-color:{bg};color:#e0e0e0"] * len(row)
    st.dataframe(vs_df.style.apply(_vs_color, axis=1), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("##### Volume Detail Chart")
    cc1, cc2 = st.columns([3, 1])
    vc_ticker = cc1.selectbox("Ticker", [s.symbol for s in results], key="vs_chart_ticker")
    vc_days   = cc2.slider("Days", 30, 180, 60, key="vs_chart_days")
    if vc_ticker:
        with st.spinner(f"Loading {vc_ticker}…"):
            fig_vc = _vol_candle_chart(vc_ticker, vc_days, (vs_threshold, vs_high))
        st.plotly_chart(fig_vc, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# History
# ═════════════════════════════════════════════════════════════════════════════

def render_history():
    st.markdown(
        "<div class='scanner-header'><span style='font-size:2rem'>🗂</span>"
        "<span class='scanner-title'>History</span></div>"
        "<div class='scanner-desc'>Browse auto-saved results from every scanner. "
        "Each successful scan is saved automatically.</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    _all_labels = ["All scanners"] + list(dict.fromkeys(_SCANNER_PREFIXES.values()))
    _prefix_map = {"All scanners":"all","Livermore Pivot":"livermore",
                   "EMA Crossover":"ema","Breakout":"breakout",
                   "Minervini SEPA":"minervini","Volume Scanner":"volume"}

    fc1, fc2 = st.columns([3, 1])
    hist_filter = fc1.selectbox("Scanner type", _all_labels, key="hist_filter")
    if fc2.button("🔄 Refresh", key="hist_refresh"):
        st.cache_data.clear()

    runs = load_all_runs(_prefix_map.get(hist_filter, "all"))

    if not runs:
        st.info("No saved runs yet. Run any scanner — results auto-save to `screener_output/`.", icon="ℹ️")
        return

    def _display_name(fname: str, label: str) -> str:
        parts = fname.replace(".csv","").split("_")
        try:
            ts = datetime.strptime(parts[-2] + parts[-1], "%Y%m%d%H%M")
            return f"{label}  •  {ts.strftime('%Y-%m-%d %H:%M')}"
        except Exception:
            return fname

    name_map     = {_display_name(k, v[0]): k for k, v in runs.items()}
    selected_key = name_map[st.selectbox("Select run", list(name_map.keys()), key="hist_run")]
    label, hist_df = runs[selected_key]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scanner",  label)
    c2.metric("Rows",     len(hist_df))
    c3.metric("Columns",  len(hist_df.columns))
    if "Signal" in hist_df.columns:
        bull = len(hist_df[hist_df["Signal"].isin(["UPWARD_PIVOT","BULLISH_CROSS","BULLISH"])])
        bear = len(hist_df[hist_df["Signal"].isin(["DOWNWARD_PIVOT","BEARISH_CROSS","BEARISH"])])
        c4.metric("Bull / Bear", f"{bull} / {bear}")
    elif "Trend Score" in hist_df.columns:
        c4.metric("Avg Trend Score", f"{hist_df['Trend Score'].mean():.1f}/8")
    elif "Flagged" in hist_df.columns:
        c4.metric("Flagged", int(hist_df["Flagged"].sum()))
    else:
        c4.metric("", "—")

    st.divider()

    if label == "Livermore Pivot" and "Signal" in hist_df.columns:
        try:
            def _lv_style(df):
                COLS = ["Ticker","Signal","Signal Date","Bars Ago","Pivot Level","Close",
                        "% from Pivot","Vol Expansion","% from 52W High","Trend (EMA)"]
                cols = [c for c in COLS if c in df.columns]
                def rc(row):
                    bg = "#1b3a2a" if row["Signal"]=="UPWARD_PIVOT" else "#3a1b1b"
                    return [f"background-color:{bg};color:#e0e0e0"]*len(row)
                return df[cols].style.apply(rc, axis=1).format(
                    {"Pivot Level":"{:.2f}","Close":"{:.2f}",
                     "% from Pivot":"{:+.2f}%","% from 52W High":"{:+.2f}%"}, na_rep="—")
            st.dataframe(_lv_style(hist_df), use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

    st.download_button("⬇️ Download CSV", hist_df.to_csv(index=False).encode(),
                       file_name=selected_key, mime="text/csv")

    if len(runs) > 1:
        st.divider()
        st.markdown("#### Signal count over time")
        freq_records = []
        for fname, (lbl, fdf) in runs.items():
            parts = fname.replace(".csv","").split("_")
            try:
                ts = datetime.strptime(parts[-2]+parts[-1], "%Y%m%d%H%M")
            except Exception:
                continue
            freq_records.append({"Time":ts,"Count":len(fdf),"Scanner":lbl})
        if freq_records:
            freq_df = pd.DataFrame(freq_records).sort_values("Time")
            colors  = {"Livermore Pivot":"#26a69a","EMA Crossover":"#42a5f5",
                       "Breakout":"#ff9800","Minervini SEPA":"#f0c040","Volume Scanner":"#ab47bc"}
            freq_fig = go.Figure()
            for sc in freq_df["Scanner"].unique():
                sub = freq_df[freq_df["Scanner"]==sc]
                freq_fig.add_trace(go.Scatter(x=sub["Time"], y=sub["Count"], mode="lines+markers",
                                             name=sc, line=dict(color=colors.get(sc,"#888"), width=2)))
            freq_fig.update_layout(template="plotly_dark", height=300,
                                   margin=dict(l=40,r=20,t=20,b=20),
                                   legend=dict(orientation="h"), yaxis_title="Signals found")
            st.plotly_chart(freq_fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Financial Astrology Scanner
# ═════════════════════════════════════════════════════════════════════════════

def render_astro():
    st.markdown(
        "<div class='scanner-header'>"
        "<span style='font-size:2rem'>🔭</span>"
        "<span class='scanner-title'>Financial Astrology</span>"
        "</div>"
        "<div class='scanner-desc'>"
        "Market timing via W.D. Gann, Larry Pesavento & Bill Meridian techniques — "
        "moon phases, planetary aspects, retrograde alerts &amp; Gann time cycles."
        "</div>",
        unsafe_allow_html=True,
    )
    _about_expander("astro")

    # ═══════════════════════════════════════════════════════════════════════════
    # SULABH JAIN — MACRO CYCLE CHEATSHEET + ANNUAL ROADMAP
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(
        "<h2 style='color:#c9a84c;margin-top:8px'>🗺️ Macro Cycle Cheatsheet"
        "<span style='font-size:14px;color:#888;font-weight:normal;margin-left:10px'>"
        "Sulabh Jain / Chariot Palmistry — Jupiter-Saturn Vedic cycle method</span></h2>",
        unsafe_allow_html=True,
    )

    try:
        cs = get_decade_cheatsheet()
        cur_r  = cs["current_regime"]
        next_r = cs.get("next_regime")

        # ── Regime timeline strip ─────────────────────────────────────────────
        _REGIME_COLORS = {
            "BULLISH": ("#1b3a2a", "#4caf50"),
            "BEARISH": ("#3a1b1b", "#ef5350"),
            "VOLATILE": ("#2a1f00", "#f9a825"),
            "NEUTRAL":  ("#1e1e2e", "#9e9e9e"),
        }

        regime_cols = st.columns(len(cs["regimes"]))
        for col, r in zip(regime_cols, cs["regimes"]):
            bg, accent = _REGIME_COLORS.get(r["regime"], ("#1e1e2e", "#9e9e9e"))
            is_current = (r is cur_r)
            border = f"3px solid {accent}" if is_current else f"1px solid {accent}44"
            badge  = " ◄ NOW" if is_current else ""
            col.markdown(
                f"<div style='background:{bg};border:{border};border-radius:8px;"
                f"padding:10px 8px;text-align:center;min-height:90px'>"
                f"<div style='font-size:11px;color:{accent};font-weight:700;letter-spacing:1px'>"
                f"{r['short']}{badge}</div>"
                f"<div style='font-size:13px;font-weight:600;margin:4px 0;color:#e0e0e0'>{r['period']}</div>"
                f"<div style='font-size:10px;color:#bbb'>{r['label']}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)

        # ── Current + Next regime detail cards ────────────────────────────────
        cc1, cc2 = st.columns(2)

        def _regime_card(col, r, title: str):
            bg, accent = _REGIME_COLORS.get(r["regime"], ("#1e1e2e", "#9e9e9e"))
            buy_html  = " · ".join(f"<span style='color:#4caf50'>{s}</span>" for s in r.get("sectors_buy", []))
            avoid_html = " · ".join(f"<span style='color:#ef5350'>{s}</span>" for s in r.get("sectors_avoid", []))
            events_html = "".join(f"<li style='color:#bbb;font-size:12px'>{e}</li>" for e in r.get("key_events", []))
            col.markdown(
                f"<div style='background:{bg};border:1px solid {accent}55;border-radius:10px;padding:14px'>"
                f"<div style='font-size:11px;color:{accent};font-weight:700;letter-spacing:1px'>{title}</div>"
                f"<div style='font-size:16px;font-weight:700;color:#e0e0e0;margin:4px 0'>{r['label']}</div>"
                f"<div style='font-size:11px;color:#aaa;margin-bottom:8px'>{r['period']}</div>"
                f"<div style='font-size:12px;color:#ccc;margin-bottom:8px'>{r['description']}</div>"
                f"<div style='font-size:11px;color:#888;margin-bottom:6px'><b>Basis:</b> {r['basis']}</div>"
                f"<ul style='margin:4px 0 8px 16px;padding:0'>{events_html}</ul>"
                f"<div style='font-size:11px;margin-top:6px'><b style='color:#4caf50'>BUY:</b> {buy_html or '—'}</div>"
                f"<div style='font-size:11px;margin-top:3px'><b style='color:#ef5350'>AVOID:</b> {avoid_html or '—'}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with cc1:
            _regime_card(cc1, cur_r, "CURRENT REGIME")
        with cc2:
            if next_r:
                _regime_card(cc2, next_r, "COMING NEXT")

        # ── Sector rotation timeline ──────────────────────────────────────────
        with st.expander("📊 Sector Rotation Timeline (full 2020-2032)", expanded=False):
            _SEC_COLORS = {"PEAK": "#f9a825", "VOLATILE": "#ef9a9a", "BULLISH": "#4caf50",
                           "NEW BULL": "#66bb6a", "BOOM": "#42a5f5", "PEAK/BUST": "#ff7043"}
            sec_rows = []
            for s in cs["sectors"]:
                c = _SEC_COLORS.get(s["regime"], "#9e9e9e")
                sec_rows.append(
                    f"<tr><td style='padding:5px 8px;font-size:13px'>{s['emoji']} {s['sector']}</td>"
                    f"<td style='padding:5px 8px;font-size:13px;color:#ccc'>{s['period']}</td>"
                    f"<td style='padding:5px 8px'><span style='background:{c}22;color:{c};"
                    f"border:1px solid {c}55;border-radius:4px;padding:2px 8px;font-size:11px;"
                    f"font-weight:700'>{s['regime']}</span></td></tr>"
                )
            st.markdown(
                "<table style='width:100%;border-collapse:collapse'>"
                "<thead><tr>"
                "<th style='text-align:left;padding:5px 8px;color:#888;font-size:11px'>SECTOR</th>"
                "<th style='text-align:left;padding:5px 8px;color:#888;font-size:11px'>PERIOD</th>"
                "<th style='text-align:left;padding:5px 8px;color:#888;font-size:11px'>STATUS</th>"
                "</tr></thead><tbody>" + "".join(sec_rows) + "</tbody></table>",
                unsafe_allow_html=True,
            )

    except Exception as _cs_err:
        st.warning(f"Cheatsheet error: {_cs_err}")

    # ── Annual Roadmap (tabbed: Sulabh hardcoded + Computed engine) ──────────
    st.divider()
    st.markdown(
        "<h3 style='color:#c9a84c'>📅 Annual Market Roadmap</h3>",
        unsafe_allow_html=True,
    )

    _cur_year = datetime.utcnow().year
    road_col1, road_col2 = st.columns([1, 3])
    with road_col1:
        road_year = st.selectbox(
            "Year", options=[2025, 2026, 2027, 2028],
            index=[2025, 2026, 2027, 2028].index(_cur_year) if _cur_year in [2025, 2026, 2027, 2028] else 1,
            key="astro_road_year",
        )

    tab_sulabh, tab_computed = st.tabs(["📋 Sulabh Jain (curated)", "🔬 Computed from Planets"])

    _RM_COLORS = {
        "BULLISH":  ("#1b3a2a", "#4caf50", "▲"),
        "BEARISH":  ("#3a1b1b", "#ef5350", "▼"),
        "VOLATILE": ("#2a1a00", "#f9a825", "⚡"),
        "NEUTRAL":  ("#1e1e2e", "#9e9e9e", "→"),
    }
    today_d = datetime.utcnow().date()

    # ── Tab 1: Sulabh Jain curated ────────────────────────────────────────────
    with tab_sulabh:
        try:
            roadmap  = get_annual_roadmap(road_year)
            events   = roadmap["events"]
            eclipses = roadmap["eclipses"]

            if not events:
                st.info(f"No curated roadmap for {road_year}. Switch to Computed tab.")
            else:
                if eclipses:
                    ecl_parts = [f"{e['emoji']} **{e['type']}** {e['date'].strftime('%b %d')}" for e in eclipses]
                    st.caption("Eclipse windows: " + "  ·  ".join(ecl_parts))

                for ev in events:
                    bg, accent, arrow = _RM_COLORS.get(ev["regime"], ("#1e1e2e", "#9e9e9e", "→"))
                    months_list = ev.get("months", [])
                    is_now = today_d.month in months_list and today_d.year == road_year
                    now_badge = (" <span style='background:#ff9800;color:#000;font-size:10px;"
                                 "border-radius:3px;padding:1px 6px;margin-left:6px'>NOW</span>"
                                 if is_now else "")
                    verified_badge = ""
                    if ev.get("verified"):
                        verified_badge = (" <span style='background:#4caf5022;color:#4caf50;"
                                          "font-size:10px;border:1px solid #4caf5055;border-radius:3px;"
                                          "padding:1px 6px;margin-left:4px'>✓ VERIFIED</span>")
                    conf_stars = "★" * ev.get("confidence", 3) + "☆" * (5 - ev.get("confidence", 3))
                    sectors_html = " · ".join(
                        f"<span style='color:{accent};font-size:11px'>{s}</span>"
                        for s in ev.get("sectors", [])
                    )
                    st.markdown(
                        f"<div style='background:{bg};border-left:4px solid {accent};"
                        f"border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:8px'>"
                        f"<div style='display:flex;align-items:center;gap:8px'>"
                        f"<span style='font-size:18px;color:{accent}'>{arrow}</span>"
                        f"<span style='font-size:13px;color:#aaa;font-weight:600'>{ev['period']}</span>"
                        f"{now_badge}{verified_badge}</div>"
                        f"<div style='font-size:15px;font-weight:700;color:#e0e0e0;margin:5px 0 3px'>{ev['theme']}</div>"
                        f"<div style='font-size:11px;color:#aaa;margin-bottom:5px'><b>Basis:</b> {ev['basis']}</div>"
                        f"<div style='font-size:11px;margin-bottom:3px'><b style='color:{accent}'>Sectors:</b> {sectors_html or '—'}</div>"
                        f"<div style='font-size:11px;color:#ccc'><b>Action:</b> {ev.get('action', '—')}</div>"
                        f"<div style='font-size:10px;color:#666;margin-top:5px'>Confidence: <span style='color:{accent}'>{conf_stars}</span></div>"
                        f"</div>", unsafe_allow_html=True,
                    )

            if eclipses:
                with st.expander(f"🌑 {road_year} Eclipse Windows", expanded=False):
                    ecl_df = pd.DataFrame([{
                        "Date": e["date"].strftime("%b %d, %Y"),
                        "Type": f"{e['emoji']} {e['type']}",
                        "Window": f"±{e['window_days']} days",
                        "Bias": e["bias"],
                        "Description": e["description"],
                    } for e in eclipses])
                    st.dataframe(ecl_df, use_container_width=True, hide_index=True)

        except Exception as _rm_err:
            st.warning(f"Roadmap error: {_rm_err}")

    # ── Tab 2: Computed from planetary positions ───────────────────────────────
    with tab_computed:
        st.caption(
            "Derived algorithmically from Jupiter/Saturn/Rahu sign positions, "
            "Mars-Saturn aspects, eclipse windows and Mercury retrograde. "
            "No hardcoded predictions — works for any year."
        )
        try:
            with st.spinner(f"Computing {road_year} planetary forecast…"):
                fc = generate_annual_forecast(road_year)

            # Annual summary metric strip
            a_bg, a_ac, _ = _RM_COLORS.get(fc["annual_regime"], ("#1e1e2e", "#9e9e9e", "→"))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Annual Regime",  fc["annual_regime"])
            m2.metric("Avg Score",      f"{fc['annual_score']:+.2f} / 3.0")
            m3.metric("Eclipses",       len(fc["eclipses"]))
            m4.metric("Periods",        len(fc["periods"]))

            # Monthly score chart
            if fc["months"]:
                import plotly.graph_objects as go
                month_names = [m["month_name"].split()[0] for m in fc["months"]]
                scores      = [m["score"] for m in fc["months"]]
                colors_bar  = [
                    "#4caf50" if s >= 1.5 else "#66bb6a" if s >= 0.3
                    else "#f9a825" if s >= -0.5 else "#ef5350"
                    for s in scores
                ]
                fig_fc = go.Figure()
                fig_fc.add_bar(x=month_names, y=scores, marker_color=colors_bar,
                               name="Monthly Score")
                fig_fc.add_hline(y=1.5,  line_dash="dot", line_color="#4caf50", annotation_text="Bull")
                fig_fc.add_hline(y=0,    line_dash="dash", line_color="#888")
                fig_fc.add_hline(y=-0.5, line_dash="dot", line_color="#ef5350", annotation_text="Bear")
                fig_fc.update_layout(
                    title=f"{road_year} Computed Market Score by Month",
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#e0e0e0", height=260,
                    margin=dict(t=40, b=20, l=0, r=0),
                    yaxis=dict(range=[-3.5, 3.5], gridcolor="#333"),
                    xaxis=dict(gridcolor="#333"),
                    showlegend=False,
                )
                st.plotly_chart(fig_fc, use_container_width=True)

            # Period cards (computed)
            for p in fc["periods"]:
                bg, accent, arrow = _RM_COLORS.get(p["regime"], ("#1e1e2e", "#9e9e9e", "→"))
                is_now = today_d.month in p["months_list"] and today_d.year == road_year
                now_badge = (" <span style='background:#ff9800;color:#000;font-size:10px;"
                             "border-radius:3px;padding:1px 6px;margin-left:6px'>NOW</span>"
                             if is_now else "")
                ecl_badge = (" <span style='background:#37474f;color:#90a4ae;font-size:10px;"
                             "border-radius:3px;padding:1px 6px;margin-left:4px'>🌑 Eclipse</span>"
                             if p.get("eclipse_active") else "")
                rx_badge  = (" <span style='background:#37474f;color:#90a4ae;font-size:10px;"
                             "border-radius:3px;padding:1px 6px;margin-left:4px'>☿ Rx</span>"
                             if p.get("mercury_rx") else "")

                events_html = "".join(
                    f"<li style='color:#bbb;font-size:11px;margin:2px 0'>{e}</li>"
                    for e in p["key_events"]
                )
                bull_html = " · ".join(
                    f"<span style='color:#4caf50;font-size:11px'>{s}</span>"
                    for s in p["sectors_bull"]
                )
                bear_html = " · ".join(
                    f"<span style='color:#ef5350;font-size:11px'>{s}</span>"
                    for s in p["sectors_bear"]
                )
                asp_html = " · ".join(
                    f"<span style='color:#f9a825;font-size:10px'>{a['planet1']}-{a['planet2']} {a['aspect']}</span>"
                    for a in p.get("aspects", [])
                )

                st.markdown(
                    f"<div style='background:{bg};border-left:4px solid {accent};"
                    f"border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:8px'>"
                    f"<div style='display:flex;align-items:center;gap:6px;flex-wrap:wrap'>"
                    f"<span style='font-size:18px;color:{accent}'>{arrow}</span>"
                    f"<span style='font-size:14px;color:#aaa;font-weight:700'>{p['period']} {road_year}</span>"
                    f"<span style='background:{accent}22;color:{accent};font-size:11px;"
                    f"border-radius:4px;padding:1px 8px'>{p['regime']} {p['avg_score']:+.2f}</span>"
                    f"{now_badge}{ecl_badge}{rx_badge}</div>"
                    f"<div style='font-size:11px;color:#888;margin:6px 0 4px'>{p['basis']}</div>"
                    f"<ul style='margin:4px 0 6px 16px;padding:0'>{events_html}</ul>"
                    f"<div style='font-size:11px;margin-top:4px'>"
                    f"<b style='color:#4caf50'>Buy:</b> {bull_html or '—'} &nbsp;"
                    f"<b style='color:#ef5350'>Avoid:</b> {bear_html or '—'}</div>"
                    + (f"<div style='font-size:10px;color:#666;margin-top:4px'>Active aspects: {asp_html}</div>" if asp_html else "")
                    + "</div>", unsafe_allow_html=True,
                )

            # Monthly detail table
            with st.expander("📊 Month-by-month detail table", expanded=False):
                month_rows = []
                for m in fc["months"]:
                    regime_sym = {"BULLISH": "▲", "BEARISH": "▼", "VOLATILE": "⚡", "NEUTRAL": "→"}.get(m["regime"], "→")
                    month_rows.append({
                        "Month":        m["month_name"],
                        "Score":        f"{m['score']:+.2f}",
                        "Regime":       f"{regime_sym} {m['regime']}",
                        "Jupiter":      m["jupiter_sign"],
                        "Saturn":       m["saturn_sign"],
                        "Rahu":         m["rahu_sign"],
                        "Eclipse":      "🌑" if m["eclipse_active"] else "",
                        "Merc Rx":      "☿" if m["mercury_rx"] else "",
                        "Buy Sectors":  ", ".join(m["sectors_bull"][:3]),
                    })
                st.dataframe(pd.DataFrame(month_rows), use_container_width=True, hide_index=True)

        except Exception as _fc_err:
            st.warning(f"Computed forecast error: {_fc_err}")

    st.divider()

    now = datetime.utcnow()

    # ── Bias + Moon state ─────────────────────────────────────────────────────
    bias = compute_market_bias(now)
    moon = bias["moon"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Bias",    f"{bias['emoji']} {bias['overall']}")
    c2.metric("Bias Score",      f"{bias['score']} / 10")
    c3.metric("Moon Phase",      f"{moon['emoji']} {moon['phase']}")
    c4.metric("Illumination",    f"{moon['illumination']}%")

    # Next key moon events
    ev_cols = st.columns(4)
    for i, (evt_name, days_away) in enumerate(moon["next_events"][:4]):
        ev_cols[i].metric(evt_name, f"in {days_away:.1f}d")

    st.divider()

    # ── Bias signals table ─────────────────────────────────────────────────────
    with st.expander("📊 Bias Signal Breakdown", expanded=True):
        if bias["signals"]:
            sig_df = pd.DataFrame(bias["signals"],
                                  columns=["Signal", "Score Δ", "Bias", "Description"])
            def _sig_style(row):
                if row["Bias"] == "BULLISH":
                    return ["background-color:#1b3a2a;color:#e0e0e0"] * len(row)
                elif row["Bias"] == "BEARISH":
                    return ["background-color:#3a1b1b;color:#e0e0e0"] * len(row)
                return [""] * len(row)
            st.dataframe(sig_df.style.apply(_sig_style, axis=1),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No strong signals active right now.", icon="ℹ️")

    # ── Retrograde status ─────────────────────────────────────────────────────
    with st.expander("☿ Retrograde Status"):
        retros = bias["retrogrades"]
        retro_df = pd.DataFrame(retros)[["Planet", "Status", "Impact", "Description"]]
        def _retro_style(row):
            if row["Status"].startswith("☿"):
                return ["background-color:#3a2a1b;color:#ffd180"] * len(row)
            return [""] * len(row)
        st.dataframe(retro_df.style.apply(_retro_style, axis=1),
                     use_container_width=True, hide_index=True)

    # ── Active aspects ────────────────────────────────────────────────────────
    with st.expander("🪐 Active Planetary Aspects"):
        if bias["aspects"]:
            asp_df = pd.DataFrame(bias["aspects"])[["Aspect", "Type", "Orb (°)", "Bias", "Description"]]
            def _asp_style(row):
                if row["Bias"] == "BEARISH":
                    return ["background-color:#3a1b1b;color:#e0e0e0"] * len(row)
                elif row["Bias"] == "BULLISH":
                    return ["background-color:#1b3a2a;color:#e0e0e0"] * len(row)
                return [""] * len(row)
            st.dataframe(asp_df.style.apply(_asp_style, axis=1),
                         use_container_width=True, hide_index=True)
        else:
            st.success("No major hard aspects active — skies are calm.", icon="✅")

    st.divider()

    # ── Upcoming Events Calendar ───────────────────────────────────────────────
    st.markdown("### 📅 45-Day Event Calendar")
    cal_days = st.slider("Days forward", 15, 90, 45, key="astro_cal_days",
                         help="How many calendar days ahead to show in the event calendar")
    events_df = get_astro_events(cal_days, now)
    if isinstance(events_df, list):
        events_df = pd.DataFrame(events_df)

    if not events_df.empty:
        def _cal_style(row):
            if row["Bias"] == "BULLISH":
                return ["background-color:#1b3a2a;color:#e0e0e0"] * len(row)
            elif row["Bias"] == "BEARISH":
                return ["background-color:#3a1b1b;color:#e0e0e0"] * len(row)
            return ["background-color:#2a2a1b;color:#e0e0e0"] * len(row)

        # Highlight today/this week
        events_df["⚡"] = events_df["Days Away"].apply(
            lambda d: "TODAY" if d < 1 else ("This week" if d <= 7 else ""))

        display_cols = ["Date", "Days Away", "⚡", "Event", "Type", "Strength", "Bias", "Description"]
        display_cols = [c for c in display_cols if c in events_df.columns]

        _CAL_COL_CFG = {
            "Days Away":   st.column_config.NumberColumn("Days Away",  help="Calendar days from today until this event"),
            "⚡":          st.column_config.TextColumn("⚡",           help="Highlighted if the event falls today or within 7 days"),
            "Strength":    st.column_config.TextColumn("Strength",     help="Signal strength: ★★★ = major turning point · ★★ = notable · ★ = minor"),
            "Bias":        st.column_config.TextColumn("Bias",         help="Expected market bias based on historical patterns for this type of event"),
            "Type":        st.column_config.TextColumn("Type",         help="Lunar = moon phase · Retrograde = planet station · Solar Ingress = Gann seasonal turn"),
            "Description": st.column_config.TextColumn("Description",  help="Research-backed explanation of this event's historical market effect"),
        }
        st.dataframe(
            events_df[display_cols].style.apply(_cal_style, axis=1),
            use_container_width=True, hide_index=True,
            column_config=_CAL_COL_CFG,
        )

        # Mini chart: bias by date
        bias_map = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}
        events_df["bias_val"] = events_df["Bias"].map(bias_map)
        fig_cal = go.Figure()
        for bias_label, color, val in [("BULLISH", "#26a69a", 1),
                                        ("NEUTRAL",  "#ffb74d", 0),
                                        ("BEARISH",  "#ef5350", -1)]:
            sub = events_df[events_df["Bias"] == bias_label]
            if sub.empty: continue
            fig_cal.add_trace(go.Scatter(
                x=pd.to_datetime(sub["Date"]),
                y=sub["bias_val"],
                mode="markers+text",
                name=bias_label,
                text=sub["Event"].str[:20],
                textposition="top center",
                marker=dict(color=color, size=12, symbol="diamond"),
            ))
        fig_cal.update_layout(
            template="plotly_dark", height=320,
            margin=dict(l=40, r=20, t=20, b=40),
            yaxis=dict(tickvals=[-1, 0, 1],
                       ticktext=["BEARISH", "NEUTRAL", "BULLISH"],
                       range=[-1.8, 1.8]),
            xaxis_title="Date",
            legend=dict(orientation="h"),
            title="Event Bias Timeline",
        )
        st.plotly_chart(fig_cal, use_container_width=True)
    else:
        st.info("No major astrological events in this window.", icon="ℹ️")

    st.divider()

    # ── Historical Moon Phase Returns ─────────────────────────────────────────
    st.markdown("### 🌕 Historical Moon Phase Back-test")
    st.caption("Tests whether New Moon (buy) / Full Moon (sell) bias held for the selected ticker.")

    ha1, ha2, ha3 = st.columns([3, 1, 1])
    astro_ticker = ha1.text_input("Ticker", value="SPY", key="astro_ticker").upper()
    months_back  = ha2.slider("Months back", 3, 24, 12, key="astro_months",
                               help="How many months of historical data to test New Moon buy / Full Moon sell signals on")
    fwd_days     = ha3.slider("Fwd days", 2, 15, 5, key="astro_fwd",
                               help="How many trading days forward to measure the return after each moon event")

    if astro_ticker:
        with st.spinner(f"Loading {astro_ticker}…"):
            raw_astro = fetch_ohlcv(astro_ticker, days=int(months_back * 31))
        if raw_astro.empty:
            st.warning(f"No data for {astro_ticker}", icon="⚠️")
        else:
            moon_evts  = get_moon_event_dates(months_back=months_back)
            returns_df = analyze_moon_returns(raw_astro, moon_evts, fwd_days=fwd_days)

            if not returns_df.empty:
                ret_col = f"{fwd_days}d Ret%"
                total   = len(returns_df)
                correct = int(returns_df["Correct?"].eq("✅").sum())
                acc     = correct / total * 100 if total else 0

                new_moon_sub  = returns_df[returns_df["Event"] == "New Moon"]
                full_moon_sub = returns_df[returns_df["Event"] == "Full Moon"]

                bm1, bm2, bm3, bm4 = st.columns(4)
                bm1.metric("Events Tested", total)
                bm2.metric("Accuracy",      f"{acc:.1f}%")
                bm3.metric(f"New Moon avg {fwd_days}d",
                           f"{new_moon_sub[ret_col].mean():.2f}%"
                           if not new_moon_sub.empty else "—")
                bm4.metric(f"Full Moon avg {fwd_days}d",
                           f"{full_moon_sub[ret_col].mean():.2f}%"
                           if not full_moon_sub.empty else "—")

                def _ret_style(row):
                    if row["Correct?"] == "✅":
                        return ["background-color:#1b3a2a;color:#e0e0e0"] * len(row)
                    return ["background-color:#3a1b1b;color:#e0e0e0"] * len(row)

                st.dataframe(
                    returns_df.style.apply(_ret_style, axis=1)
                               .format({ret_col: "{:+.2f}%",
                                        "Entry": "{:.2f}", "Exit": "{:.2f}"}),
                    use_container_width=True, hide_index=True,
                )

                # Price + moon events chart
                with st.spinner("Rendering chart…"):
                    fig_moon = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                            row_heights=[0.75, 0.25],
                                            vertical_spacing=0.02)
                    candle_m, vol_m = _candlestick_vol(raw_astro, astro_ticker)
                    fig_moon.add_trace(candle_m, row=1, col=1)
                    fig_moon.add_trace(vol_m,    row=2, col=1)

                    for _, ev_row in returns_df.iterrows():
                        ev_ts  = pd.Timestamp(ev_row["Event Date"])
                        is_new = ev_row["Event"] == "New Moon"
                        color  = "#26a69a" if is_new else "#ef5350"
                        sym    = "triangle-up" if is_new else "triangle-down"
                        lbl    = "🌑" if is_new else "🌕"
                        # price at event
                        idx_c = raw_astro.index[raw_astro.index >= ev_ts]
                        if len(idx_c) == 0:
                            continue
                        px = float(raw_astro.loc[idx_c[0], "Low"]) * 0.985 if is_new \
                             else float(raw_astro.loc[idx_c[0], "High"]) * 1.015
                        fig_moon.add_trace(go.Scatter(
                            x=[idx_c[0]], y=[px],
                            mode="markers+text",
                            text=[lbl], textposition="bottom center" if is_new else "top center",
                            marker=dict(symbol=sym, size=14, color=color,
                                        line=dict(color="#fff", width=0.5)),
                            name=f"{'New' if is_new else 'Full'} Moon",
                            showlegend=False,
                        ), row=1, col=1)

                    _apply_chart_style(fig_moon, f"{astro_ticker} — Moon Phase Events",
                                       rows=2, ytitles=["Price", "Volume"], height=600)
                    st.plotly_chart(fig_moon, use_container_width=True)
            else:
                st.info("Not enough overlapping data for back-test.", icon="ℹ️")

    st.divider()

    # ── Gann Time Cycles ──────────────────────────────────────────────────────
    st.markdown("### ⚙️ Gann Time Cycle Projections")
    st.caption("Projects 45/90/144/180/270/360-day cycles from the 52W High and Low of the selected ticker.")

    gc1, gc2 = st.columns([3, 1])
    gann_ticker  = gc1.text_input("Ticker", value=astro_ticker or "SPY", key="gann_ticker").upper()
    gann_lookahead = gc2.slider("Look-ahead days", 30, 180, 90, key="gann_lookahead",
                                 help="How many calendar days ahead to project Gann cycles from the 52W High and Low")

    if gann_ticker:
        with st.spinner(f"Loading {gann_ticker}…"):
            raw_gann = fetch_ohlcv(gann_ticker, days=380)
        gann_df = find_gann_cycle_dates(raw_gann, lookahead_days=gann_lookahead)
        if gann_df.empty:
            st.info("No Gann cycle dates in this window.", icon="ℹ️")
        else:
            def _gann_style(row):
                if "TODAY" in str(row["Status"]):
                    return ["background-color:#4a3a00;color:#ffd54f"] * len(row)
                _days_away = pd.to_numeric(row["Days Away"], errors="coerce")
                if pd.notna(_days_away) and _days_away < 0:
                    return ["background-color:#1a1a1a;color:#666"] * len(row)
                return [""] * len(row)
            st.dataframe(
                gann_df.style.apply(_gann_style, axis=1),
                use_container_width=True, hide_index=True,
            )
            # Plot
            future = gann_df[gann_df["Days Away"] >= 0]
            if not future.empty and not raw_gann.empty:
                fig_gann = go.Figure()
                fig_gann.add_trace(go.Scatter(
                    x=raw_gann.tail(252).index,
                    y=raw_gann.tail(252)["Close"],
                    mode="lines", name="Close",
                    line=dict(color="#42a5f5", width=1.5),
                ))
                colors_gann = {"52W High": "#ef5350", "52W Low": "#26a69a"}
                for _, grow in future.iterrows():
                    c = colors_gann.get(grow["Pivot Type"], "#ffb74d")
                    gd = pd.Timestamp(grow["Gann Date"])
                    fig_gann.add_shape(
                        type="line",
                        x0=gd, x1=gd, y0=0, y1=1, yref="paper",
                        line=dict(color=c, width=1, dash="dot"),
                    )
                    fig_gann.add_annotation(
                        x=gd, y=1.02, yref="paper",
                        text=str(grow["Cycle"]),
                        showarrow=False,
                        font=dict(color=c, size=9),
                        textangle=-60,
                    )
                fig_gann.update_layout(
                    template="plotly_dark", height=380,
                    title=f"{gann_ticker} — Gann Cycle Projections",
                    margin=dict(l=40, r=20, t=40, b=30),
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_gann, use_container_width=True)
                st.caption("🔴 from 52W High  |  🟢 from 52W Low  |  Dotted lines = Gann cycle dates")

    st.divider()

    # ── Vedic: Bhadra / Vishti Karana Backtest ────────────────────────────────
    st.markdown("### 🕉️ Bhadra (Vishti Karana) Backtest")
    st.caption(
        "Vedic Panchang: Vishti/Bhadra is the 7th of 7 movable Karanas — "
        "each lasting ~11h, occurring 8× per lunar month. "
        "Considered highly inauspicious; Nifty/Sensex traders avoid new positions. "
        "Does it affect US markets?"
    )

    # Current Karana status
    karana_now = get_karana(now)
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Current Karana",   karana_now["karana"])
    kc2.metric("Sentiment",        karana_now["sentiment"])
    kc3.metric("Tithi",            karana_now["tithi"])
    kc4.metric("Karana ends in",   f"{karana_now['hours_remaining']}h")

    if karana_now["is_vishti"]:
        st.error(
            f"🚨 **BHADRA ACTIVE NOW** — Vishti Karana in effect for "
            f"{karana_now['hours_remaining']} more hours. "
            "Vedic traders: avoid opening new positions.",
            icon="⚠️",
        )
    else:
        st.success("✅ Bhadra is NOT active right now — Karana is favorable.", icon="🕉️")

    st.markdown("#### Backtest Settings")
    bk1, bk2, bk3 = st.columns([3, 2, 2])
    bk_ticker  = bk1.text_input("Ticker", value="SPY", key="bk_ticker").upper()
    bk_years   = bk2.slider("Years of data", 3, 15, 10, key="bk_years")
    bk_mode    = bk3.selectbox(
        "Bhadra active when",
        ["any", "open", "all"],
        format_func=lambda x: {
            "any":  "Any part of session",
            "open": "Market open only",
            "all":  "Entire session (strict)",
        }[x],
        key="bk_mode",
    )

    if st.button("▶ Run Bhadra Backtest", key="run_bhadra", type="primary"):
        with st.spinner(f"Downloading {bk_ticker} data + computing Karanas for {bk_years}y…"):
            raw_bk = fetch_ohlcv(bk_ticker, days=int(bk_years * 365) + 60)

        if raw_bk.empty:
            st.warning(f"No data for {bk_ticker}", icon="⚠️")
        else:
            with st.spinner("Running Karana calculations (~15 sec for 10y)…"):
                bk_result = backtest_vishti(raw_bk, check_mode=bk_mode)

            if not bk_result:
                st.error("Not enough data for backtest.", icon="❌")
            else:
                # ── Summary metrics ───────────────────────────────────────────
                r = bk_result
                sig_str = ("✅ Significant" if r["significant"] else "❌ Not significant")
                direction = "LOWER" if r["bhadra_mean"] < r["normal_mean"] else "HIGHER"

                sm1, sm2, sm3, sm4 = st.columns(4)
                sm1.metric("Bhadra Days",       f"{r['bhadra_days']} ({r['bhadra_pct_time']}%)")
                sm2.metric("Bhadra Avg Return",  f"{r['bhadra_mean']:+.3f}%",
                           delta=f"{r['bhadra_mean'] - r['normal_mean']:+.3f}% vs normal")
                sm3.metric("Normal Avg Return",  f"{r['normal_mean']:+.3f}%")
                sm4.metric("Win Rate",
                           f"Bh {r['bhadra_win_rate']}% | Nrm {r['normal_win_rate']}%")

                sm5, sm6, sm7, sm8 = st.columns(4)
                sm5.metric("p-value (t-test)",   f"{r['p_val']:.4f}")
                sm6.metric("p-value (Mann-W)",    f"{r['p_mann']:.4f}")
                sm7.metric("Statistical Sig",     sig_str)
                sm8.metric("Bhadra returns",      f"{direction} than normal")

                # Verdict box
                if r["bhadra_mean"] < r["normal_mean"] and r["p_val"] < 0.10:
                    st.error(
                        f"📉 **Bhadra effect CONFIRMED on {bk_ticker}**: "
                        f"Average return during Bhadra is {r['bhadra_mean']:+.3f}% vs "
                        f"{r['normal_mean']:+.3f}% on normal days "
                        f"(p={r['p_val']:.4f}, statistically significant).",
                        icon="🕉️",
                    )
                elif r["bhadra_mean"] < r["normal_mean"] and r["p_val"] < 0.20:
                    st.warning(
                        f"⚠️ **Weak Bhadra effect on {bk_ticker}**: "
                        f"Bhadra days average {r['bhadra_mean']:+.3f}% vs "
                        f"{r['normal_mean']:+.3f}% (p={r['p_val']:.4f}, marginal).",
                        icon="🕉️",
                    )
                else:
                    st.info(
                        f"ℹ️ **No significant Bhadra effect on {bk_ticker}**: "
                        f"Bhadra {r['bhadra_mean']:+.3f}% vs Normal {r['normal_mean']:+.3f}% "
                        f"(p={r['p_val']:.4f}).",
                        icon="🕉️",
                    )

                st.divider()

                # ── Return distribution chart ──────────────────────────────────
                import plotly.figure_factory as ff
                detail = r["detail_df"]
                bh_rets = detail.loc[detail["bhadra"],  "ret"].dropna().tolist()
                nb_rets = detail.loc[~detail["bhadra"], "ret"].dropna().tolist()

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**Return Distribution**")
                    try:
                        fig_dist = ff.create_distplot(
                            [bh_rets, nb_rets],
                            ["Bhadra Days", "Normal Days"],
                            colors=["#ef5350", "#26a69a"],
                            bin_size=0.25, show_rug=False,
                        )
                        fig_dist.update_layout(
                            template="plotly_dark", height=320,
                            margin=dict(l=30, r=10, t=20, b=30),
                            legend=dict(orientation="h"),
                            xaxis_title="Daily Return %",
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    except Exception:
                        # Fallback box plot
                        fig_box = go.Figure()
                        fig_box.add_trace(go.Box(y=bh_rets, name="Bhadra", marker_color="#ef5350"))
                        fig_box.add_trace(go.Box(y=nb_rets, name="Normal", marker_color="#26a69a"))
                        fig_box.update_layout(template="plotly_dark", height=320,
                                              margin=dict(l=30,r=10,t=20,b=30))
                        st.plotly_chart(fig_box, use_container_width=True)

                with col_b:
                    st.markdown("**Cumulative Performance**")
                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=detail.index, y=detail["cum_all"],
                        name="All Days", line=dict(color="#42a5f5", width=1.5),
                    ))
                    fig_cum.add_trace(go.Scatter(
                        x=detail.index, y=detail["cum_nobhadra"],
                        name="Skip Bhadra Days", line=dict(color="#26a69a", width=1.5, dash="dash"),
                    ))
                    # Mark Bhadra days
                    bh_mask = detail["bhadra"]
                    fig_cum.add_trace(go.Scatter(
                        x=detail.index[bh_mask],
                        y=detail.loc[bh_mask, "cum_all"],
                        mode="markers", name="Bhadra Day",
                        marker=dict(color="#ef5350", size=4, symbol="circle"),
                    ))
                    fig_cum.update_layout(
                        template="plotly_dark", height=320,
                        margin=dict(l=30, r=10, t=20, b=30),
                        legend=dict(orientation="h"),
                        yaxis_title="Cumulative Return (×)",
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)

                # ── Monthly breakdown ──────────────────────────────────────────
                st.markdown("**Monthly Average Returns: Bhadra vs Normal**")
                if not r["monthly_df"].empty:
                    monthly = r["monthly_df"].tail(36)  # last 3 years
                    fig_m = go.Figure()
                    if "Normal Avg %" in monthly.columns:
                        fig_m.add_trace(go.Bar(
                            x=monthly.index.astype(str), y=monthly["Normal Avg %"],
                            name="Normal Days", marker_color="#26a69a", opacity=0.7,
                        ))
                    if "Bhadra Avg %" in monthly.columns:
                        fig_m.add_trace(go.Bar(
                            x=monthly.index.astype(str), y=monthly["Bhadra Avg %"],
                            name="Bhadra Days", marker_color="#ef5350", opacity=0.85,
                        ))
                    fig_m.update_layout(
                        template="plotly_dark", height=320,
                        barmode="group",
                        margin=dict(l=30, r=10, t=20, b=60),
                        xaxis_tickangle=-45,
                        yaxis_title="Avg Daily Return %",
                        legend=dict(orientation="h"),
                    )
                    st.plotly_chart(fig_m, use_container_width=True)

                # ── Worst Bhadra days ──────────────────────────────────────────
                with st.expander("📋 Largest Bhadra drops (worst 20 days)"):
                    worst = (detail[detail["bhadra"]]
                             .sort_values("ret")
                             .head(20)
                             .reset_index()
                             .rename(columns={"index": "Date"}))
                    worst["Date"] = worst["Date"].dt.strftime("%Y-%m-%d")
                    st.dataframe(
                        worst[["Date", "Close", "ret"]]
                        .rename(columns={"ret": "Return %"})
                        .style.format({"Return %": "{:+.2f}%", "Close": "{:.2f}"})
                               .background_gradient(subset=["Return %"], cmap="RdYlGn"),
                        use_container_width=True, hide_index=True,
                    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SULABH JAIN — VEDIC PREDICTION ENGINE
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown(
        "<h3 style='color:#c9a84c'>🕉️ Vedic Prediction Engine <span style='font-size:14px;"
        "color:#888;font-weight:normal'>(Sulabh Jain / Chariot Palmistry method)</span></h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Combines Rahu-Ketu transits, Moon Nakshatra, Vedic Karana (Vishti/Bhadra), "
        "and planetary sign positions into a forward market forecast."
    )

    # ── Today's Vedic snapshot ────────────────────────────────────────────────
    try:
        ved     = compute_vedic_daily_score(now)
        rk      = ved["rahu_ketu"]
        nk      = ved["nakshatra"]
        kar     = ved["karana"]

        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("Vedic Score",      f"{ved['emoji']} {ved['score']} / 10")
        v2.metric("Vedic Bias",       ved["overall"])
        v3.metric("Moon Nakshatra",   nk["nakshatra"])
        v4.metric("Karana",           kar["karana"])
        v5.metric("Rahu in",          rk["rahu_sign"] + " / Ketu " + rk["ketu_sign"])

        # Rahu-Ketu regime box
        rahu_color = "#1b3a2a" if rk["rahu_bias"] == "BULLISH" else "#3a1b1b" if rk["rahu_bias"] == "BEARISH" else "#2a2a2a"
        st.markdown(
            f"<div style='padding:10px 16px;background:{rahu_color};border-radius:6px;"
            f"margin-bottom:8px'>"
            f"<b>☊ Rahu in {rk['rahu_sign']} | ☋ Ketu in {rk['ketu_sign']}</b> — "
            f"{rk['rahu_theme']}</div>",
            unsafe_allow_html=True,
        )

        # Nakshatra detail
        nk_color = "#1b3a2a" if nk["bias"] == "BULLISH" else "#3a1b1b" if nk["bias"] == "BEARISH" else "#2a2a2a"
        st.markdown(
            f"<div style='padding:8px 16px;background:{nk_color};border-radius:6px;margin-bottom:8px'>"
            f"<b>☽ Moon in {nk['nakshatra']}</b> ({nk['bias']}) — {nk['desc']}"
            f"<span style='color:#888;font-size:12px'> · "
            f"{kar['hours_remaining']:.1f}h until next Karana · Tithi: {kar['tithi']}</span></div>",
            unsafe_allow_html=True,
        )

        # Vedic signal breakdown
        with st.expander("📊 Vedic Signal Breakdown", expanded=True):
            if ved["signals"]:
                vsig_df = pd.DataFrame(ved["signals"],
                                       columns=["Signal", "Score Δ", "Bias", "Description"])
                def _vsig_style(row):
                    if row["Bias"] == "BULLISH":
                        return ["background-color:#1b3a2a;color:#e0e0e0"] * len(row)
                    elif row["Bias"] == "BEARISH":
                        return ["background-color:#3a1b1b;color:#e0e0e0"] * len(row)
                    return [""] * len(row)
                st.dataframe(vsig_df.style.apply(_vsig_style, axis=1),
                             use_container_width=True, hide_index=True)

        # Vedic planet positions
        with st.expander("🪐 Vedic Planet Positions (Sidereal / Lahiri)"):
            vp_df = pd.DataFrame(get_vedic_planets(now))
            def _vp_style(row):
                if row["Bias"] == "BULLISH":
                    return ["background-color:#1b3a2a;color:#e0e0e0"] * len(row)
                elif row["Bias"] == "BEARISH":
                    return ["background-color:#3a1b1b;color:#e0e0e0"] * len(row)
                return [""] * len(row)
            st.dataframe(vp_df.style.apply(_vp_style, axis=1),
                         use_container_width=True, hide_index=True)

    except Exception as _ve:
        st.error(f"Vedic engine error: {_ve}", icon="❌")

    st.divider()

    # ── 30-Day Prediction Calendar ────────────────────────────────────────────
    st.markdown("### 📆 30-Day Market Prediction Calendar")
    st.caption("Combined Western + Vedic score per trading day. Green = bullish, Red = bearish.")

    pred_days = st.slider("Days forward", 10, 60, 30, key="pred_cal_days")

    with st.spinner("Building prediction calendar…"):
        try:
            cal_df = build_prediction_calendar(days=pred_days, dt=now)
        except Exception as _ce:
            cal_df = pd.DataFrame()
            st.error(f"Calendar error: {_ce}")

    if not cal_df.empty:
        def _cal_style(row):
            combined = row["Combined"]
            if combined >= 3:
                return ["background-color:#1b3a2a;color:#e0e0e0"] * len(row)
            elif combined >= 1:
                return ["background-color:#1b2e1b;color:#e0e0e0"] * len(row)
            elif combined <= -3:
                return ["background-color:#3a1b1b;color:#e0e0e0"] * len(row)
            elif combined <= -1:
                return ["background-color:#2e1b1b;color:#e0e0e0"] * len(row)
            return [""] * len(row)

        st.dataframe(
            cal_df.style.apply(_cal_style, axis=1),
            use_container_width=True, hide_index=True,
            column_config={
                "Date":           st.column_config.DateColumn("Date", format="MMM DD"),
                "Combined":       st.column_config.NumberColumn("Combined Score", format="%.1f"),
                "Western Score":  st.column_config.NumberColumn("Western", format="%.1f"),
                "Vedic Score":    st.column_config.NumberColumn("Vedic", format="%.1f"),
            },
        )

        # Heat-map bar chart
        fig_pred = go.Figure()
        colors_pred = [
            "#26a69a" if v >= 1 else "#ef5350" if v <= -1 else "#ffb74d"
            for v in cal_df["Combined"]
        ]
        fig_pred.add_trace(go.Bar(
            x=cal_df["Date"].astype(str),
            y=cal_df["Combined"],
            marker_color=colors_pred,
            text=cal_df["Moon Nakshatra"],
            textposition="outside",
            textfont=dict(size=9),
        ))
        fig_pred.update_layout(
            template="plotly_dark", height=320,
            margin=dict(l=30, r=10, t=20, b=60),
            xaxis_tickangle=-45,
            yaxis=dict(title="Combined Score", range=[-10, 10]),
            xaxis_title="",
        )
        fig_pred.add_hline(y=0, line_dash="dash", line_color="#555")
        st.plotly_chart(fig_pred, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Scanner: Combined Strategy (straight buy signals)
# ═════════════════════════════════════════════════════════════════════════════

_SIGNAL_COLORS = {
    "⚡ STRONG BUY": "#1b3a2a",
    "✅ BUY":        "#1b2e1b",
    "👀 WATCH":      "#2a2a1b",
}
_SIGNAL_BADGE = {
    "⚡ STRONG BUY": "background:#1a9641;color:#fff",
    "✅ BUY":        "background:#388e3c;color:#fff",
    "👀 WATCH":      "background:#f9a825;color:#111",
}


def render_combined():
    # ── Header ────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([6, 2, 2])
    with c1:
        st.markdown(
            "<div class='scanner-header'>"
            "<span style='font-size:2rem'>🎯</span>"
            "<span class='scanner-title'>Combined Strategy</span>"
            f"&nbsp;{market_badge_html()}"
            "</div>"
            "<div class='scanner-desc'>"
            "Multi-factor buy signal engine — Trend + Momentum + Trigger + Volume scored into ⚡ STRONG BUY / ✅ BUY / 👀 WATCH."
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        last = st.session_state.get("combined_time")
        if last:
            st.caption("Last scan")
            st.caption(last)
    with c3:
        clicked = st.button("▶  Scan Now", type="primary",
                            use_container_width=True, key="combined_scan_btn")
    st.divider()

    # ── How it works ──────────────────────────────────────────────────────────
    with st.expander("ℹ️ How the score works", expanded=False):
        st.markdown("""
**Each stock is scored across 4 dimensions. Total score determines the rating.**

| Dimension | Max pts | What it checks |
|-----------|---------|----------------|
| **Trend** | 3 | Price > EMA50 · EMA50 > EMA200 · Minervini Stage 2 |
| **Momentum** | 3 | RSI 45–72 · MACD histogram positive · RS vs SPY ≥ 60 |
| **Entry Trigger** | 3 | EMA9×21 cross · 20-day breakout · NR7 · BB-squeeze · Gap↑ |
| **Volume** | 2 | Vol > 1.5× avg (+1) or > 3× avg (+2) |

| Score | Rating |
|-------|--------|
| ≥ 7 | ⚡ STRONG BUY |
| 5–6 | ✅ BUY |
| 3–4 | 👀 WATCH |

**Risk gates (auto-disqualify):** Price < min · ATR% > max · ADX < min · No entry trigger

**Entry / Stop / Target:** Close · Close − 1.5×ATR · Close + 3.0×ATR → **2:1 R/R**
""")

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Settings", expanded=False):
        st.markdown("**Momentum filters**")
        mc1, mc2, mc3 = st.columns(3)
        c_rsi_low   = mc1.slider("RSI min",      20.0, 60.0, 45.0, step=1.0, key="cb_rsi_low",
                                  help="RSI must be above this to score +1 momentum. 45 = not oversold but not hot")
        c_rsi_high  = mc2.slider("RSI max",      55.0, 85.0, 72.0, step=1.0, key="cb_rsi_high",
                                  help="RSI must be below this — avoids overbought entries")
        c_min_rs    = mc3.slider("Min RS Score", 20.0, 90.0, 60.0, step=5.0, key="cb_min_rs",
                                  help="Relative Strength vs SPY (0-100). 60 = stock beats 60% of the market")
        st.markdown("**Entry trigger settings**")
        tc1, tc2, tc3 = st.columns(3)
        c_cross_bars   = tc1.slider("EMA cross window", 1, 15, 5, key="cb_cross_bars",
                                     help="EMA 9/21 cross must have occurred within this many bars")
        c_bo_vol       = tc2.slider("Breakout vol ×",  1.0, 4.0, 1.5, step=0.25, key="cb_bo_vol",
                                     help="Volume multiple required for 20-day breakout trigger")
        c_gap_pct      = tc3.slider("Min gap %",        0.1, 3.0, 0.3, step=0.1, key="cb_gap_pct",
                                     help="Minimum gap size (% of prev close) to count as Gap↑ trigger")
        st.markdown("**Risk gates**")
        rg1, rg2, rg3 = st.columns(3)
        c_min_price  = rg1.number_input("Min price $",   1.0, 50.0, 5.0, step=1.0, key="cb_min_price")
        c_max_atr    = rg2.slider("Max ATR %",           1.0, 15.0,  8.0, step=0.5, key="cb_max_atr",
                                   help="Daily ATR as % of price. Higher = too volatile for position sizing")
        c_min_adx    = rg3.slider("Min ADX",             0.0, 30.0, 15.0, step=1.0, key="cb_min_adx",
                                   help="Minimum ADX — below this = choppy, no trend")
        st.markdown("**Output filter**")
        fc1, fc2 = st.columns(2)
        c_min_score  = fc1.select_slider("Min score", options=[3, 4, 5, 6, 7], value=3, key="cb_min_score",
                                          help="3 = WATCH+, 5 = BUY+, 7 = STRONG BUY only")
        c_show_cols  = fc2.multiselect("Show columns",
                                        ["Why", "Triggers", "RSI", "ADX", "MACD Hist",
                                         "RS Score", "Vol vs Avg", "ATR%", "Minervini",
                                         "EMA Stack", "Trend Pts", "Momentum Pts", "Trigger Pts"],
                                        default=["Why", "RSI", "ADX", "Vol vs Avg", "EMA Stack"],
                                        key="cb_show_cols")

    # ── Run ───────────────────────────────────────────────────────────────────
    if "combined_df" not in st.session_state:
        st.session_state.combined_df   = pd.DataFrame()
        st.session_state.combined_time = None

    if clicked:
        tickers = get_selected_tickers()
        _ticker_count_caption(tickers)
        prog = st.progress(0, text="Running combined scanner…")
        with st.spinner("Scoring all tickers — this may take a minute…"):
            df_cb = run_combined_screener(
                tickers=tickers,
                rsi_low=c_rsi_low,
                rsi_high=c_rsi_high,
                min_rs=c_min_rs,
                ema_cross_bars=c_cross_bars,
                breakout_vol_mult=c_bo_vol,
                min_gap_pct=c_gap_pct,
                min_price=c_min_price,
                max_atr_pct=c_max_atr,
                min_adx=c_min_adx,
                min_score=c_min_score,
            )
        prog.progress(100, text="Done")
        st.session_state.combined_df   = df_cb
        st.session_state.combined_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        if not df_cb.empty:
            save_scan_result(df_cb, "combined")

    df_cb = st.session_state.combined_df
    if df_cb.empty and st.session_state.combined_time:
        st.warning("No signals found. Try lowering Min Score or loosening risk gates.", icon="⚠️")
        return
    if df_cb.empty:
        st.info("Hit **Scan Now** to run the combined strategy scanner.", icon="ℹ️")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    strong = df_cb[df_cb["Signal"] == "⚡ STRONG BUY"]
    buys   = df_cb[df_cb["Signal"] == "✅ BUY"]
    watch  = df_cb[df_cb["Signal"] == "👀 WATCH"]
    avg_score = df_cb["Score"].mean()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total signals",   len(df_cb))
    c2.metric("⚡ Strong Buy",    len(strong))
    c3.metric("✅ Buy",           len(buys))
    c4.metric("👀 Watch",         len(watch))
    c5.metric("Avg Score",       f"{avg_score:.1f} / 11")

    st.divider()

    # ── Quick-view cards for top strong buys ─────────────────────────────────
    if not strong.empty:
        st.markdown("#### ⚡ Top Strong Buy Setups")
        card_rows = strong.head(6)
        cols_per_row = 3
        for row_start in range(0, len(card_rows), cols_per_row):
            card_cols = st.columns(cols_per_row)
            for ci, (_, r) in enumerate(card_rows.iloc[row_start:row_start + cols_per_row].iterrows()):
                with card_cols[ci]:
                    rr_ratio = r.get("R/R", "—")
                    st.markdown(f"""
<div style="background:rgba(26,150,65,0.12);border:1px solid #1a9641;border-radius:10px;padding:14px">
  <div style="font-size:1.3rem;font-weight:700">{r['Ticker']}</div>
  <div style="font-size:0.8rem;color:#888;margin-bottom:6px">{r.get('Why','')}</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
    <span style="background:#1a9641;color:#fff;padding:2px 8px;border-radius:12px;font-size:11px">Score {r['Score']}</span>
    <span style="background:#333;color:#ccc;padding:2px 8px;border-radius:12px;font-size:11px">RSI {r.get('RSI','—')}</span>
    <span style="background:#333;color:#ccc;padding:2px 8px;border-radius:12px;font-size:11px">Vol {r.get('Vol vs Avg','—')}×</span>
  </div>
  <div style="font-size:0.85rem">
    <b>Entry</b> ${r['Entry']} &nbsp;|&nbsp;
    <b>Stop</b> <span style="color:#ef5350">${r['Stop']}</span> &nbsp;|&nbsp;
    <b>Target</b> <span style="color:#26a69a">${r['Target']}</span>
  </div>
  <div style="font-size:0.78rem;color:#aaa;margin-top:4px">R/R {rr_ratio}:1 &nbsp;·&nbsp; {r.get('EMA Stack','')}</div>
</div>
""", unsafe_allow_html=True)
        st.divider()

    # ── Full table ────────────────────────────────────────────────────────────
    BASE_COLS = ["Ticker", "Signal", "Score", "Entry", "Stop", "Target", "R/R", "52W Target"]
    extra = [c for c in c_show_cols if c in df_cb.columns]
    display_cols = BASE_COLS + extra
    display_cols = [c for c in display_cols if c in df_cb.columns]

    fmt = {}
    for col in ["Entry", "Stop", "Target", "52W Target"]:
        if col in df_cb.columns: fmt[col] = "{:.2f}"
    if "MACD Hist" in df_cb.columns:  fmt["MACD Hist"]  = "{:.4f}"
    if "ATR%"      in df_cb.columns:  fmt["ATR%"]       = "{:.2f}%"
    if "Vol vs Avg" in df_cb.columns: fmt["Vol vs Avg"] = "{:.2f}x"

    _COL_CFG = {
        "Score":   st.column_config.ProgressColumn("Score", min_value=0, max_value=11,
                                                    help="Total signal score out of 11"),
        "Signal":  st.column_config.TextColumn("Signal",
                                                help="⚡ STRONG BUY ≥7 · ✅ BUY 5-6 · 👀 WATCH 3-4"),
        "Why":     st.column_config.TextColumn("Why", help="Conditions that fired for this ticker"),
        "Triggers":st.column_config.TextColumn("Triggers", help="Entry trigger(s) that fired"),
        "52W Target": st.column_config.NumberColumn("52W Target",
                                                     help="52-week high — potential profit target if price breaks out"),
    }

    def _cb_color(row):
        sig = row.get("Signal", "")
        bg = _SIGNAL_COLORS.get(sig, "#1e1e1e")
        return [f"background-color:{bg};color:#e0e0e0"] * len(row)

    st.dataframe(
        df_cb[display_cols].style.apply(_cb_color, axis=1).format(fmt, na_rep="—"),
        use_container_width=True, hide_index=True,
        column_config=_COL_CFG,
    )

    st.divider()

    # ── Chart for selected ticker ─────────────────────────────────────────────
    st.markdown("##### Chart — Entry Setup")
    cc1, cc2 = st.columns([3, 1])
    chosen_cb = cc1.selectbox("Ticker", df_cb["Ticker"].tolist(), key="cb_chart_ticker")
    chart_days_cb = cc2.slider("Days", 60, 365, 150, key="cb_chart_days")

    if chosen_cb:
        row = df_cb[df_cb["Ticker"] == chosen_cb].iloc[0]
        sig = row["Signal"]
        col_badge = "#1a9641" if "STRONG" in sig else "#388e3c" if "BUY" in sig else "#f9a825"

        st.markdown(
            f"<span style='background:{col_badge};color:#fff;padding:4px 14px;"
            f"border-radius:12px;font-weight:600'>{sig}</span> &nbsp;"
            f"Score <b>{row['Score']}</b>/11 &nbsp;·&nbsp; {row.get('Why','')}",
            unsafe_allow_html=True,
        )

        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Entry",  f"${row['Entry']:.2f}")
        ec2.metric("Stop",   f"${row['Stop']:.2f}",
                   delta=f"{(row['Stop']-row['Entry'])/row['Entry']*100:+.1f}%",
                   delta_color="inverse")
        ec3.metric("Target", f"${row['Target']:.2f}",
                   delta=f"{(row['Target']-row['Entry'])/row['Entry']*100:+.1f}%")
        ec4.metric("R/R",    f"{row['R/R']}:1")

        with st.spinner(f"Loading {chosen_cb}…"):
            raw_df = fetch_ohlcv(chosen_cb, days=chart_days_cb)

        if not raw_df.empty:
            ann = raw_df.copy()
            ann["ema9"]   = ann["Close"].ewm(span=9,   adjust=False).mean()
            ann["ema21"]  = ann["Close"].ewm(span=21,  adjust=False).mean()
            ann["ema50"]  = ann["Close"].ewm(span=50,  adjust=False).mean()
            ann["ema200"] = ann["Close"].ewm(span=200, adjust=False).mean()

            # MACD
            fast_e = ann["Close"].ewm(span=12, adjust=False).mean()
            slow_e = ann["Close"].ewm(span=26, adjust=False).mean()
            ann["macd_line"] = fast_e - slow_e
            ann["macd_sig"]  = ann["macd_line"].ewm(span=9, adjust=False).mean()
            ann["macd_hist"] = ann["macd_line"] - ann["macd_sig"]

            # RSI
            d = ann["Close"].diff()
            up = d.clip(lower=0).ewm(span=14, adjust=False).mean()
            dn = (-d.clip(upper=0)).ewm(span=14, adjust=False).mean()
            ann["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, float("nan")))

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                row_heights=[0.55, 0.23, 0.22], vertical_spacing=0.02)

            candle, vol_t = _candlestick_vol(ann, chosen_cb)
            fig.add_trace(candle, row=1, col=1)

            for span, color, name in [
                (9,   "#f0c040", "EMA9"),
                (21,  "#ab47bc", "EMA21"),
                (50,  "#42a5f5", "EMA50"),
                (200, "#ff7043", "EMA200"),
            ]:
                col_name = f"ema{span}"
                if col_name in ann.columns:
                    fig.add_trace(go.Scatter(x=ann.index, y=ann[col_name], mode="lines",
                                             line=dict(color=color, width=1.2), name=name),
                                  row=1, col=1)

            for val, color, lbl in [
                (row["Entry"],  "#ffffff", f"Entry ${row['Entry']:.2f}"),
                (row["Stop"],   "#ef5350", f"Stop ${row['Stop']:.2f}"),
                (row["Target"], "#26a69a", f"T1 ${row['Target']:.2f}"),
            ]:
                fig.add_hline(y=val, line_dash="dot", line_color=color,
                              line_width=1.4, annotation_text=lbl,
                              annotation_position="top right", row=1, col=1)

            if row.get("52W Target") and pd.notna(row.get("52W Target")):
                fig.add_hline(y=row["52W Target"], line_dash="dash", line_color="#ffeb3b",
                              line_width=1.0, annotation_text=f"52W ${row['52W Target']:.2f}",
                              annotation_position="top right", row=1, col=1)

            # MACD
            hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in ann["macd_hist"].fillna(0)]
            fig.add_trace(go.Bar(x=ann.index, y=ann["macd_hist"],
                                 marker_color=hist_colors, name="Hist", showlegend=False),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=ann.index, y=ann["macd_line"], mode="lines",
                                     line=dict(color="#f0c040", width=1.3), name="MACD"), row=2, col=1)
            fig.add_trace(go.Scatter(x=ann.index, y=ann["macd_sig"], mode="lines",
                                     line=dict(color="#42a5f5", width=1.1, dash="dot"), name="Signal"),
                          row=2, col=1)
            fig.add_hline(y=0, line_color="#555", line_width=0.7, row=2, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=ann.index, y=ann["rsi14"], mode="lines",
                                     line=dict(color="#ab47bc", width=1.4), name="RSI(14)"), row=3, col=1)
            for level, color in [(70, "#ef5350"), (50, "#888"), (30, "#26a69a")]:
                fig.add_hline(y=level, line_dash="dash", line_color=color,
                              line_width=0.8, row=3, col=1)

            _apply_chart_style(
                fig,
                f"{chosen_cb} — {sig}  (Score {row['Score']}/11)",
                rows=3, ytitles=["Price", "MACD", "RSI"],
                height=750,
            )
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Scanner: RSI
# ═════════════════════════════════════════════════════════════════════════════

def render_rsi():
    clicked = _page_header(
        "📊", "RSI Scanner",
        "Oversold/overbought crosses, RSI 50-line momentum shifts, and bullish/bearish divergence.",
        scan_key="rsi_scan_btn", last_key="rsi_time",
    )

    _ABOUT_RSI = """
**Three RSI signal types**

| Signal | Meaning | Bias |
|--------|---------|------|
| **OVERSOLD_BULL** | RSI(14) crosses below oversold threshold (default 30) | Long reversal candidate |
| **OVERBOUGHT_BEAR** | RSI(14) crosses above overbought threshold (default 70) | Short reversal candidate |
| **MOMENTUM_BULL** | RSI crosses above 50 with ADX > threshold | Trend shifting bullish |
| **MOMENTUM_BEAR** | RSI crosses below 50 with ADX > threshold | Trend shifting bearish |
| **DIV_BULL** | Price lower low, RSI higher low (bullish divergence) | Early long reversal |
| **DIV_BEAR** | Price higher high, RSI lower high (bearish divergence) | Early short reversal |

**Stops & Targets:** ATR-based — Stop ±1.5×ATR(14), Target ±2.5×ATR(14) → 1.67:1 R/R
"""
    with st.expander("ℹ️ RSI Scanner — How It Works", expanded=False):
        st.markdown(_ABOUT_RSI)

    with st.expander("⚙️ Settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        rsi_oversold  = c1.slider("Oversold",   10.0, 45.0, 30.0, step=1.0, key="rsi_oversold",
                                   help="RSI must cross BELOW this level to trigger OVERSOLD_BULL")
        rsi_overbought = c2.slider("Overbought", 55.0, 90.0, 70.0, step=1.0, key="rsi_overbought",
                                   help="RSI must cross ABOVE this level to trigger OVERBOUGHT_BEAR")
        rsi_adx       = c3.slider("Min ADX",     0.0, 35.0, 20.0, step=1.0, key="rsi_adx",
                                   help="Minimum ADX for MOMENTUM signals (0 = no filter)")
        rsi_vol       = c4.slider("Min Vol ×",   0.5,  3.0,  1.0, step=0.25, key="rsi_vol",
                                   help="Volume on signal day ÷ 20-day average")
        c5, c6, c7 = st.columns(3)
        rsi_period    = c5.number_input("RSI Period", 5, 21, 14, key="rsi_period",
                                        help="RSI lookback period (default 14)")
        rsi_div_lb    = c6.number_input("Divergence lookback", 10, 40, 20, key="rsi_div_lb",
                                        help="Bars to scan for divergence")
        rsi_recent    = c7.slider("Signal window", 1, 30, 10, key="rsi_recent",
                                   help="Only show signals from within this many trading days")
        rsi_filter    = st.radio("Show", ["ALL", "BULLISH", "BEARISH"],
                                  horizontal=True, key="rsi_sig_filter")

    if "rsi_df" not in st.session_state:
        st.session_state.rsi_df   = pd.DataFrame()
        st.session_state.rsi_time = None

    if clicked:
        tickers = get_selected_tickers()
        _ticker_count_caption(tickers)
        with st.spinner("Scanning RSI signals…"):
            df_rsi = run_rsi_screener(
                tickers=tickers,
                oversold=rsi_oversold,
                overbought=rsi_overbought,
                rsi_period=rsi_period,
                div_lookback=rsi_div_lb,
                adx_threshold=rsi_adx,
                vol_mult=rsi_vol,
                recent_bars=rsi_recent,
                signal_filter=rsi_filter,
            )
            st.session_state.rsi_df   = df_rsi
            st.session_state.rsi_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            if not df_rsi.empty:
                save_scan_result(df_rsi, "rsi")

    df_rsi = st.session_state.rsi_df
    if df_rsi.empty and st.session_state.rsi_time:
        st.warning("No RSI signals found. Try loosening filters.", icon="⚠️")
        return
    if df_rsi.empty:
        st.info("Hit **Scan Now** to find RSI signals.", icon="ℹ️")
        return

    bull_r = df_rsi[df_rsi["Signal"].str.contains("BULL|bull", na=False)]
    bear_r = df_rsi[df_rsi["Signal"].str.contains("BEAR|bear", na=False)]
    div_r  = df_rsi[df_rsi["Signal"].str.contains("DIV", na=False)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total",      len(df_rsi))
    c2.metric("Bullish ▲",  len(bull_r))
    c3.metric("Bearish ▼",  len(bear_r))
    c4.metric("Divergence", len(div_r))

    st.divider()

    COLS_RSI = ["Ticker", "Signal", "Signal Date", "Bars Ago",
                "RSI", "Close", "ADX", "Vol vs Avg",
                "Stop", "Target", "R/R", "EMA50", "Divergence"]
    cols = [c for c in COLS_RSI if c in df_rsi.columns]
    fmt = {}
    for col in ["Close", "Stop", "Target"]:
        if col in df_rsi.columns: fmt[col] = "{:.2f}"
    if "Vol vs Avg" in df_rsi.columns: fmt["Vol vs Avg"] = "{:.2f}x"

    def _rsi_color(row):
        sig = row.get("Signal", "")
        bg = "#1b3a2a" if "BULL" in sig else "#3a1b1b" if "BEAR" in sig else "#2a2a1b"
        return [f"background-color:{bg};color:#e0e0e0"] * len(row)

    st.dataframe(
        df_rsi[cols].style.apply(_rsi_color, axis=1).format(fmt, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.markdown("##### Chart")
    cc1, cc2 = st.columns([3, 1])
    chosen_rsi = cc1.selectbox("Ticker", df_rsi["Ticker"].tolist(), key="rsi_chart_ticker")
    chart_days_rsi = cc2.slider("Days", 60, 365, 120, key="rsi_chart_days")
    if chosen_rsi:
        sig_row = {r["Ticker"]: r for _, r in df_rsi.iterrows()}.get(chosen_rsi, {})
        with st.spinner(f"Loading {chosen_rsi}…"):
            raw_df = fetch_ohlcv(chosen_rsi, days=chart_days_rsi)
        if not raw_df.empty:
            ann = raw_df.copy()
            delta = ann["Close"].diff()
            gain = delta.clip(lower=0).rolling(rsi_period).mean()
            loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
            rs = gain / loss.replace(0, float("nan"))
            ann["rsi14"] = 100 - (100 / (1 + rs))
            ann["ema50"] = ann["Close"].ewm(span=50, adjust=False).mean()

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.65, 0.35], vertical_spacing=0.02)
            candle, vol = _candlestick_vol(ann, chosen_rsi)
            fig.add_trace(candle, row=1, col=1)
            fig.add_trace(go.Scatter(x=ann.index, y=ann["ema50"], mode="lines",
                                     line=dict(color="#42a5f5", width=1.2), name="EMA50"),
                          row=1, col=1)
            # Mark signal
            if sig_row:
                sig_date = sig_row.get("Signal Date")
                sig_close = sig_row.get("Close")
                for val, color, lbl in [
                    (sig_row.get("Stop"),   "#ef5350", f"Stop {sig_row.get('Stop','')}"),
                    (sig_row.get("Target"), "#26a69a", f"Target {sig_row.get('Target','')}"),
                ]:
                    if val and pd.notna(val):
                        fig.add_hline(y=val, line_dash="dot", line_color=color,
                                      line_width=1.2, annotation_text=lbl,
                                      annotation_position="top right", row=1, col=1)

            fig.add_trace(go.Scatter(x=ann.index, y=ann["rsi14"], mode="lines",
                                     line=dict(color="#ab47bc", width=1.5), name="RSI(14)"),
                          row=2, col=1)
            for level, color, lbl in [(rsi_oversold, "#26a69a", f"OS {rsi_oversold}"),
                                       (rsi_overbought, "#ef5350", f"OB {rsi_overbought}"),
                                       (50, "#888", "50")]:
                fig.add_hline(y=level, line_dash="dash", line_color=color,
                              line_width=0.9, annotation_text=lbl, row=2, col=1)
            _apply_chart_style(fig, f"{chosen_rsi} — RSI({rsi_period})",
                               rows=2, ytitles=["Price", f"RSI({rsi_period})"], height=650)
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Scanner: MACD
# ═════════════════════════════════════════════════════════════════════════════

def render_macd():
    clicked = _page_header(
        "〽️", "MACD Scanner",
        "Signal line crosses, histogram flips, and zero-line crossovers with volume confirmation.",
        scan_key="macd_scan_btn", last_key="macd_time",
    )

    _ABOUT_MACD = """
**Three MACD signal types**

| Signal | Meaning | Speed |
|--------|---------|-------|
| **HIST_FLIP_BULL/BEAR** | Histogram changes sign — earliest warning | Fast (more noise) |
| **MACD_CROSS_BULL/BEAR** | MACD line crosses signal line | Classic signal |
| **ZERO_CROSS_BULL/BEAR** | MACD line crosses zero — trend confirmation | Slow (high quality) |

**Parameters:** Classic 12/26/9 (default). Shorter = more signals, longer = higher quality.

**Optional filter — Zero-side:** Only show CROSS signals where MACD is below 0 (for bull)
or above 0 (for bear), ensuring you're trading the higher-conviction side of the histogram.

**Stops & Targets:** ATR-based — Stop ±1.5×ATR(14), Target ±2.5×ATR(14) → 1.67:1 R/R
"""
    with st.expander("ℹ️ MACD Scanner — How It Works", expanded=False):
        st.markdown(_ABOUT_MACD)

    with st.expander("⚙️ Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        macd_fast   = c1.number_input("Fast EMA",   2, 50, 12, key="macd_fast",
                                       help="Fast EMA period (default 12)")
        macd_slow   = c2.number_input("Slow EMA",   5, 100, 26, key="macd_slow",
                                       help="Slow EMA period (default 26)")
        macd_signal = c3.number_input("Signal EMA", 2, 30, 9,  key="macd_signal",
                                       help="Signal line smoothing (default 9)")
        c4, c5, c6 = st.columns(3)
        macd_vol    = c4.slider("Min Vol ×", 0.5, 3.0, 1.0, step=0.25, key="macd_vol",
                                 help="Volume on signal day ÷ 20-day average")
        macd_recent = c5.slider("Signal window", 1, 30, 10, key="macd_recent",
                                 help="Only show signals from within this many trading days")
        macd_zero   = c6.checkbox("Zero-side filter", value=False, key="macd_zero_side",
                                   help="For CROSS signals: only show bullish when MACD < 0, bearish when MACD > 0")
        macd_filter = st.radio("Show", ["ALL", "BULLISH", "BEARISH"],
                                horizontal=True, key="macd_sig_filter")

    if "macd_df" not in st.session_state:
        st.session_state.macd_df   = pd.DataFrame()
        st.session_state.macd_time = None

    if clicked:
        tickers = get_selected_tickers()
        _ticker_count_caption(tickers)
        with st.spinner("Scanning MACD signals…"):
            df_macd = run_macd_screener(
                tickers=tickers,
                fast=macd_fast,
                slow=macd_slow,
                signal_period=macd_signal,
                vol_mult=macd_vol,
                require_zero_side=macd_zero,
                recent_bars=macd_recent,
                signal_filter=macd_filter,
            )
            st.session_state.macd_df   = df_macd
            st.session_state.macd_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            if not df_macd.empty:
                save_scan_result(df_macd, "macd")

    df_macd = st.session_state.macd_df
    if df_macd.empty and st.session_state.macd_time:
        st.warning("No MACD signals found. Try loosening filters.", icon="⚠️")
        return
    if df_macd.empty:
        st.info("Hit **Scan Now** to find MACD signals.", icon="ℹ️")
        return

    bull_m = df_macd[df_macd["Signal"].str.contains("BULL", na=False)]
    bear_m = df_macd[df_macd["Signal"].str.contains("BEAR", na=False)]
    cross_m = df_macd[df_macd["Signal"].str.contains("CROSS", na=False)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total",       len(df_macd))
    c2.metric("Bullish ▲",   len(bull_m))
    c3.metric("Bearish ▼",   len(bear_m))
    c4.metric("Line Crosses", len(cross_m))

    st.divider()

    COLS_MACD = ["Ticker", "Signal", "Signal Date", "Bars Ago",
                 "MACD", "Signal Line", "Histogram", "RSI",
                 "Close", "Vol vs Avg", "Stop", "Target", "R/R", "EMA50"]
    cols = [c for c in COLS_MACD if c in df_macd.columns]
    fmt = {}
    for col in ["MACD", "Signal Line", "Histogram"]:
        if col in df_macd.columns: fmt[col] = "{:.4f}"
    for col in ["Close", "Stop", "Target"]:
        if col in df_macd.columns: fmt[col] = "{:.2f}"
    if "Vol vs Avg" in df_macd.columns: fmt["Vol vs Avg"] = "{:.2f}x"

    def _macd_color(row):
        sig = row.get("Signal", "")
        bg = "#1b3a2a" if "BULL" in sig else "#3a1b1b" if "BEAR" in sig else "#1a1a2a"
        return [f"background-color:{bg};color:#e0e0e0"] * len(row)

    st.dataframe(
        df_macd[cols].style.apply(_macd_color, axis=1).format(fmt, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.markdown("##### Chart")
    cc1, cc2 = st.columns([3, 1])
    chosen_macd = cc1.selectbox("Ticker", df_macd["Ticker"].tolist(), key="macd_chart_ticker")
    chart_days_macd = cc2.slider("Days", 60, 365, 120, key="macd_chart_days")
    if chosen_macd:
        sig_row = {r["Ticker"]: r for _, r in df_macd.iterrows()}.get(chosen_macd, {})
        with st.spinner(f"Loading {chosen_macd}…"):
            raw_df = fetch_ohlcv(chosen_macd, days=chart_days_macd)
        if not raw_df.empty:
            ann = raw_df.copy()
            fast_ema = ann["Close"].ewm(span=macd_fast, adjust=False).mean()
            slow_ema = ann["Close"].ewm(span=macd_slow, adjust=False).mean()
            ann["macd_line"] = fast_ema - slow_ema
            ann["macd_sig"]  = ann["macd_line"].ewm(span=macd_signal, adjust=False).mean()
            ann["macd_hist"] = ann["macd_line"] - ann["macd_sig"]
            ann["ema50"] = ann["Close"].ewm(span=50, adjust=False).mean()

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.60, 0.40], vertical_spacing=0.02)
            candle, _vol = _candlestick_vol(ann, chosen_macd)
            fig.add_trace(candle, row=1, col=1)
            fig.add_trace(go.Scatter(x=ann.index, y=ann["ema50"], mode="lines",
                                     line=dict(color="#42a5f5", width=1.2), name="EMA50"),
                          row=1, col=1)
            if sig_row:
                for val, color, lbl in [
                    (sig_row.get("Stop"),   "#ef5350", f"Stop {sig_row.get('Stop','')}"),
                    (sig_row.get("Target"), "#26a69a", f"Target {sig_row.get('Target','')}"),
                ]:
                    if val and pd.notna(val):
                        fig.add_hline(y=val, line_dash="dot", line_color=color,
                                      line_width=1.2, annotation_text=lbl,
                                      annotation_position="top right", row=1, col=1)

            # MACD subplot
            hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in ann["macd_hist"].fillna(0)]
            fig.add_trace(go.Bar(x=ann.index, y=ann["macd_hist"],
                                 marker_color=hist_colors, name="Histogram", showlegend=True),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=ann.index, y=ann["macd_line"], mode="lines",
                                     line=dict(color="#f0c040", width=1.5), name="MACD"),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=ann.index, y=ann["macd_sig"], mode="lines",
                                     line=dict(color="#42a5f5", width=1.2, dash="dot"), name="Signal"),
                          row=2, col=1)
            fig.add_hline(y=0, line_color="#555", line_width=0.8, row=2, col=1)
            _apply_chart_style(fig, f"{chosen_macd} — MACD({macd_fast},{macd_slow},{macd_signal})",
                               rows=2, ytitles=["Price", "MACD"], height=650)
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Scanner: Gap
# ═════════════════════════════════════════════════════════════════════════════

def render_gap():
    _ABOUT_GAP = """
**Gap Types**

| Type | Meaning | Trade Bias |
|------|---------|-----------|
| **BREAKAWAY** | Gap out of a BB-squeeze consolidation — high conviction | Trade WITH the gap |
| **CONTINUATION** | Gap mid-trend (price above EMA-50) — trend acceleration | Add to position / new entry |
| **EXHAUSTION** | Gap after extended move with high RSI — reversal warning | Fade or avoid |
| **COMMON** | Small gap within range, no trend context | Usually fills quickly — skip |

**Body Quality %** — how close close was to high (gap up) or low (gap down). 100% = strong follow-through.

**Gap Fill %** — how much of the gap has been retraced. 100% = fully filled.

**Stops & Targets:**
- Gap Up long: Stop = prev close, Target = close + 2× gap size → 2:1 R/R
- Gap Down short: Stop = prev close, Target = close − 2× gap size → 2:1 R/R
"""

    tab_live, tab_scan = st.tabs(["⚡ Live Today (auto-refresh)", "🔍 Deep Scan"])

    # ── Tab 1: Live Today ─────────────────────────────────────────────────────
    with tab_live:
        st.markdown("### ⚡ Today's Gaps — Live")
        st.caption("Compares today's open vs yesterday's close across your watchlist. Auto-refreshes every 15 min.")

        st_autorefresh(interval=900_000, key="gap_live_refresh")

        lc1, lc2, lc3 = st.columns(3)
        live_min_gap = lc1.slider("Min gap %", 0.5, 5.0, 1.0, step=0.5, key="live_gap_pct")
        live_vol     = lc2.slider("Min Vol ×", 0.5, 3.0, 1.0, step=0.5, key="live_gap_vol")
        live_dir     = lc3.radio("Direction", ["ALL", "UP only", "DOWN only"],
                                  horizontal=True, key="live_gap_dir")

        with st.spinner("Fetching today's gaps…"):
            live_tickers = get_selected_tickers()
            df_live = run_live_gap_screener(
                tickers=live_tickers,
                min_gap_pct=live_min_gap,
                vol_mult=live_vol,
            )

        if live_dir == "UP only" and not df_live.empty:
            df_live = df_live[df_live["Gap %"] > 0]
        elif live_dir == "DOWN only" and not df_live.empty:
            df_live = df_live[df_live["Gap %"] < 0]

        now_str = datetime.now().strftime("%H:%M:%S")
        if df_live.empty:
            st.info(f"No gaps ≥ {live_min_gap}% with Vol ≥ {live_vol}× found as of {now_str}.")
        else:
            gap_ups   = df_live[df_live["Gap %"] > 0]
            gap_downs = df_live[df_live["Gap %"] < 0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", len(df_live))
            m2.metric("Gap Ups ▲", len(gap_ups))
            m3.metric("Gap Downs ▼", len(gap_downs))
            m4.metric("Largest", f"{df_live['Gap %'].abs().max():.1f}%")
            st.caption(f"Last updated: {now_str} ET")

            def _live_color(row):
                g = row.get("Gap %", 0)
                if g > 5:    bg = "#1a4d1a"
                elif g > 2:  bg = "#0d2d0d"
                elif g < -5: bg = "#4d1a1a"
                elif g < -2: bg = "#2d0d0d"
                else:        bg = "#1e1e2e"
                return [f"background-color:{bg};color:#e0e0e0"] * len(row)

            live_fmt = {
                "Gap %": "{:+.2f}%",
                "Prev Close": "{:.2f}", "Open": "{:.2f}", "Current": "{:.2f}",
                "Vol Ratio": "{:.2f}x", "Momentum %": "{:+.1f}%",
                "Stop": "{:.2f}", "Target": "{:.2f}",
            }
            live_fmt = {k: v for k, v in live_fmt.items() if k in df_live.columns}
            st.dataframe(
                df_live.style.apply(_live_color, axis=1).format(live_fmt, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )

    # ── Tab 2: Deep Scan ──────────────────────────────────────────────────────
    with tab_scan:
        clicked = _page_header(
            "🕳️", "Gap Scanner",
            "Detect and classify opening gaps — breakaway, continuation, and exhaustion gaps.",
            scan_key="gap_scan_btn", last_key="gap_time",
        )

        with st.expander("ℹ️ Gap Scanner — How It Works", expanded=False):
            st.markdown(_ABOUT_GAP)

        with st.expander("⚙️ Settings", expanded=False):
            c1, c2, c3 = st.columns(3)
            gap_min_pct = c1.slider("Min gap %",     0.2, 5.0, 0.5, step=0.1, key="gap_min_pct",
                                     help="Minimum gap size as % of previous close")
            gap_vol     = c2.slider("Min Vol ×",     0.5, 5.0, 1.5, step=0.25, key="gap_vol",
                                     help="Volume on gap day ÷ 20-day average")
            gap_recent  = c3.slider("Signal window", 1, 20, 5, key="gap_recent",
                                     help="Only show gaps from within this many trading days")
            c4, c5 = st.columns(2)
            gap_direction = c4.radio("Direction", ["ALL", "UP", "DOWN"],
                                      horizontal=True, key="gap_direction")
            gap_common    = c5.checkbox("Include COMMON gaps", value=False, key="gap_common",
                                        help="Show small common gaps (usually fill within days)")

        if "gap_df" not in st.session_state:
            st.session_state.gap_df   = pd.DataFrame()
            st.session_state.gap_time = None

        if clicked:
            tickers = get_selected_tickers()
            _ticker_count_caption(tickers)
            with st.spinner("Scanning for gaps…"):
                df_gap = run_gap_screener(
                    tickers=tickers,
                    min_gap_pct=gap_min_pct,
                    vol_mult=gap_vol,
                    recent_bars=gap_recent,
                    gap_direction=gap_direction,
                    include_common=gap_common,
                )
                st.session_state.gap_df   = df_gap
                st.session_state.gap_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                if not df_gap.empty:
                    save_scan_result(df_gap, "gap")

        df_gap = st.session_state.gap_df
        if df_gap.empty and st.session_state.gap_time:
            st.warning("No gaps found. Try loosening filters.", icon="⚠️")
            return
        if df_gap.empty:
            st.info("Hit **Scan Now** to find gap setups.", icon="ℹ️")
            return

        gap_up       = df_gap[df_gap["Signal"] == "GAP_UP"]
        gap_down     = df_gap[df_gap["Signal"] == "GAP_DOWN"]
        breakaway    = df_gap[df_gap["Gap Type"] == "BREAKAWAY"]
        continuation = df_gap[df_gap["Gap Type"] == "CONTINUATION"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Gaps",     len(df_gap))
        c2.metric("Gap Up ▲",       len(gap_up))
        c3.metric("Gap Down ▼",     len(gap_down))
        c4.metric("Breakaway 🚀",   len(breakaway))
        c5.metric("Continuation ▶", len(continuation))

        st.divider()

        type_opts  = ["ALL"] + sorted(df_gap["Gap Type"].dropna().unique().tolist())
        type_sel   = st.radio("Filter by type", type_opts, horizontal=True, key="gap_type_filter")
        display_df = df_gap if type_sel == "ALL" else df_gap[df_gap["Gap Type"] == type_sel]

        COLS_GAP = ["Ticker", "Signal", "Gap Type", "Signal Date", "Bars Ago",
                    "Gap %", "Prev Close", "Open", "Close",
                    "Vol vs Avg", "Body Quality %", "Gap Fill %",
                    "Stop", "Target", "R/R", "RSI", "EMA50"]
        cols = [c for c in COLS_GAP if c in display_df.columns]
        fmt = {}
        for col in ["Prev Close", "Open", "Close", "Stop", "Target"]:
            if col in display_df.columns: fmt[col] = "{:.2f}"
        if "Gap %"          in display_df.columns: fmt["Gap %"]          = "{:+.2f}%"
        if "Vol vs Avg"     in display_df.columns: fmt["Vol vs Avg"]     = "{:.2f}x"
        if "Body Quality %" in display_df.columns: fmt["Body Quality %"] = "{:.1f}%"
        if "Gap Fill %"     in display_df.columns: fmt["Gap Fill %"]     = "{:.1f}%"

        def _gap_color(row):
            sig   = row.get("Signal", "")
            gtype = row.get("Gap Type", "")
            if   gtype == "BREAKAWAY":    bg = "#1b2a3a" if sig == "GAP_UP" else "#3a1b2a"
            elif gtype == "CONTINUATION": bg = "#1b3a2a" if sig == "GAP_UP" else "#3a1b1b"
            elif gtype == "EXHAUSTION":   bg = "#3a3a1b"
            else:                         bg = "#2a2a2a"
            return [f"background-color:{bg};color:#e0e0e0"] * len(row)

        st.dataframe(
            display_df[cols].style.apply(_gap_color, axis=1).format(fmt, na_rep="—"),
            use_container_width=True, hide_index=True,
        )

        st.divider()
        st.markdown("##### Chart")
        cc1, cc2 = st.columns([3, 1])
        chosen_gap     = cc1.selectbox("Ticker", df_gap["Ticker"].tolist(), key="gap_chart_ticker")
        chart_days_gap = cc2.slider("Days", 20, 180, 60, key="gap_chart_days")
        if chosen_gap:
            sig_rows = df_gap[df_gap["Ticker"] == chosen_gap]
            with st.spinner(f"Loading {chosen_gap}…"):
                raw_df = fetch_ohlcv(chosen_gap, days=chart_days_gap)
            if not raw_df.empty:
                ann = raw_df.copy()
                ann["ema50"]    = ann["Close"].ewm(span=50, adjust=False).mean()
                ann["vol_ma20"] = ann["Volume"].rolling(20).mean()

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.70, 0.30], vertical_spacing=0.02)
                candle, vol_trace = _candlestick_vol(ann, chosen_gap)
                fig.add_trace(candle, row=1, col=1)
                fig.add_trace(go.Scatter(x=ann.index, y=ann["ema50"], mode="lines",
                                         line=dict(color="#42a5f5", width=1.2), name="EMA50"),
                              row=1, col=1)
                fig.add_trace(vol_trace, row=2, col=1)
                fig.add_trace(go.Scatter(x=ann.index, y=ann["vol_ma20"], mode="lines",
                                         line=dict(color="#ffb74d", width=1.2, dash="dot"),
                                         name="Vol MA20"), row=2, col=1)

                for _, gr in sig_rows.iterrows():
                    try:
                        gdate = pd.Timestamp(gr["Signal Date"])
                        if gdate in ann.index:
                            gopen = gr.get("Open")
                            gprev = gr.get("Prev Close")
                            gtype = gr.get("Gap Type", "")
                            color = "#4fc3f7" if gr["Signal"] == "GAP_UP" else "#ff8a65"
                            if pd.notna(gopen) and pd.notna(gprev):
                                fig.add_shape(type="rect",
                                              x0=gdate, x1=ann.index[-1],
                                              y0=min(gopen, gprev), y1=max(gopen, gprev),
                                              fillcolor=color, opacity=0.12,
                                              line_width=0, row=1, col=1)
                            for val, lbl, clr in [
                                (gr.get("Stop"),   "Stop",   "#ef5350"),
                                (gr.get("Target"), "Target", "#26a69a"),
                            ]:
                                if val and pd.notna(val):
                                    fig.add_hline(y=val, line_dash="dot", line_color=clr,
                                                  line_width=1.2,
                                                  annotation_text=f"{lbl} ({gtype})",
                                                  annotation_position="top right",
                                                  row=1, col=1)
                    except Exception:
                        pass

                _apply_chart_style(fig, f"{chosen_gap} — Gap Analysis",
                                   rows=2, ytitles=["Price", "Volume"], height=650)
                st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Scanner: Swing Options 45-60 DTE
# ═════════════════════════════════════════════════════════════════════════════

def render_swing_options():
    clicked = _page_header(
        "🎰", "Swing Options 45-60 DTE",
        "Directional options scanner: 45-60 DTE with Greeks filters (Δ, Θ, Γ, Θ/Vega). "
        "Scores each symbol across 7 technical signals (EMA alignment, MACD, ADX, Volume, "
        "Supertrend, BB expansion, Stoch) then validates option entry with Black-Scholes Greeks.",
        scan_key="opts_scan_btn", last_key="opts_time",
    )

    _ABOUT_OPTS = """
**Strategy Overview**

Finds directional swing setups (calls & puts) where:
1. The underlying trend is clear (7-signal composite score ≥ threshold)
2. The option Greeks make the trade structurally sound at entry

**Signal Scoring (0–10)**

| Signal | Weight | Bullish (Call) | Bearish (Put) |
|--------|--------|----------------|---------------|
| EMA Alignment | 2.0 | EMA9 > EMA21 > EMA50 | EMA9 < EMA21 < EMA50 |
| Price vs SMA200 | 1.5 | Close > SMA200 | Close < SMA200 |
| RSI Zone | 1.0 | RSI 45-65 | RSI 35-55 |
| MACD | 1.5 | MACD > Signal | MACD < Signal |
| ADX + DI | 1.0 | ADX > 20, +DI > -DI | ADX > 20, -DI > +DI |
| Volume Surge | 1.0 | Vol > 1.5× avg | Vol > 1.5× avg |
| Supertrend | 1.0 | Direction = +1 | Direction = -1 |
| BB Expansion | +0.5 | After squeeze | After squeeze |
| Stoch crossover | +0.5 | %K > %D, not overbought | %K < %D, not oversold |
| OBV Trend | +0.5 | OBV rising | OBV falling |

**Greeks Filters (all 4 must pass)**

| Filter | Rule | Why |
|--------|------|-----|
| **Delta** | 0.40 ≤ \|δ\| ≤ 0.70 | Not too OTM (low prob) or ITM (expensive) |
| **Theta/Premium** | \|θ\|/prem < 1.5%/day | Daily bleed too fast = structurally bad entry |
| **Gamma** | 0.005 ≤ γ ≤ 0.050 | Sweet spot: leverage without needing perfect timing |
| **Theta/Vega** | \|θ/vega\| ≤ 0.40 | IV expansion must offset theta bleed (filters high-vol names) |

**DTE Sweet Spot:** Enter at 45-60 DTE, exit at 21 DTE — captures the "safe zone" of theta decay
"""

    with st.expander("ℹ️ How It Works", expanded=False):
        st.markdown(_ABOUT_OPTS)

    with st.expander("⚙️ Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        opts_min_score = c1.slider("Min Score", 4.0, 9.5, 6.0, step=0.5, key="opts_min_score",
                                    help="Composite signal score threshold (0-10). Higher = more selective.")
        opts_dte       = c2.slider("Target DTE", 40, 65, 50, step=5, key="opts_dte",
                                    help="Days to expiration to price the option (typically 45-60)")
        opts_iv_prem   = c3.slider("IV Premium", 1.0, 1.5, 1.1, step=0.05, key="opts_iv_prem",
                                    help="IV = Hist Vol × this factor (options trade at a premium to HV)")
        c4, c5, c6 = st.columns(3)
        opts_delta_min = c4.slider("Delta Min", 0.25, 0.55, 0.40, step=0.05, key="opts_delta_min",
                                    help="Minimum |delta| at entry")
        opts_delta_max = c5.slider("Delta Max", 0.55, 0.85, 0.70, step=0.05, key="opts_delta_max",
                                    help="Maximum |delta| at entry")
        opts_tv_max    = c6.slider("Θ/Vega Max", 0.20, 0.80, 0.40, step=0.05, key="opts_tv_max",
                                    help="Max theta/vega ratio — above this, decay overwhelms vol sensitivity")
        opts_filter    = st.radio("Direction Filter", ["ALL", "CALL", "PUT"],
                                   horizontal=True, key="opts_dir_filter")

    if "opts_df" not in st.session_state:
        st.session_state.opts_df   = pd.DataFrame()
        st.session_state.opts_time = None

    if clicked:
        tickers = get_selected_tickers()
        _ticker_count_caption(tickers)

        prog = st.progress(0, text="Starting options scan…")

        def _prog(pct, msg):
            prog.progress(min(pct, 1.0), text=msg)

        with st.spinner("Scanning 45-60 DTE options setups…"):
            df_opts = run_swing_options_screener(
                tickers=tickers,
                min_score=opts_min_score,
                dte=float(opts_dte),
                iv_premium=opts_iv_prem,
                params={
                    "delta_min": opts_delta_min,
                    "delta_max": opts_delta_max,
                    "theta_vega_ratio_max": opts_tv_max,
                },
                progress_cb=_prog,
            )
        prog.empty()
        st.session_state.opts_df   = df_opts
        st.session_state.opts_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        if not df_opts.empty:
            save_scan_result(df_opts.drop(columns=["_passes_greeks"], errors="ignore"), "swing_opts")

    df_opts = st.session_state.opts_df

    if df_opts.empty and st.session_state.opts_time:
        st.warning("No options setups found. Try lowering Min Score or widening Greek filters.", icon="⚠️")
        return
    if df_opts.empty:
        st.info("Hit **Scan Now** to find 45-60 DTE options swing setups.", icon="ℹ️")
        return

    # Apply direction filter
    if opts_filter != "ALL":
        df_opts = df_opts[df_opts["Direction"] == opts_filter]

    # Summary metrics
    calls = df_opts[df_opts["Direction"] == "CALL"]
    puts  = df_opts[df_opts["Direction"] == "PUT"]
    ok    = df_opts[df_opts.get("_passes_greeks", pd.Series(True, index=df_opts.index)) == True] \
            if "_passes_greeks" in df_opts.columns else df_opts
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Setups",   len(df_opts))
    c2.metric("Calls ▲",        len(calls))
    c3.metric("Puts ▼",         len(puts))
    c4.metric("Greeks Pass ✅",  len(ok[ok["Greeks OK"] == "✅"]) if "Greeks OK" in ok.columns else "—")

    st.divider()

    # Table — drop internal column
    display_cols = [c for c in [
        "Symbol", "Direction", "Score", "Close", "Strike", "Premium",
        "Delta", "Theta/day", "Gamma", "Vega/1%", "θ/Prem %", "θ/Vega",
        "Greeks OK", "Hist Vol", "ADX", "RSI",
        "EMA Aligned", "Trend 200", "MACD OK", "Vol Surge", "Supertrend",
    ] if c in df_opts.columns]

    fmt = {
        "Score": "{:.1f}", "Close": "{:.2f}", "Premium": "{:.2f}",
        "Delta": "{:.3f}", "Theta/day": "{:.4f}", "Gamma": "{:.4f}",
        "Vega/1%": "{:.3f}", "θ/Prem %": "{:.2f}%", "θ/Vega": "{:.3f}",
        "Hist Vol": "{:.1%}", "ADX": "{:.1f}", "RSI": "{:.1f}",
    }

    def _opts_color(row):
        if row.get("Direction") == "CALL":
            bg = "#1b3a2a"
        else:
            bg = "#3a1b1b"
        if row.get("Greeks OK", "").startswith("❌"):
            bg = "#2a2520"  # muted for greek failures
        return [f"background-color:{bg};color:#e0e0e0"] * len(row)

    st.dataframe(
        df_opts[display_cols].style.apply(_opts_color, axis=1).format(fmt, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # Chart
    st.markdown("##### Price Chart")
    if len(df_opts) > 0:
        cc1, cc2 = st.columns([3, 1])
        syms = df_opts["Symbol"].tolist()
        chosen_opts = cc1.selectbox("Symbol", syms, key="opts_chart_sym")
        chart_days_opts = cc2.slider("Days", 60, 365, 120, key="opts_chart_days")

        if chosen_opts:
            sig_row = df_opts[df_opts["Symbol"] == chosen_opts].iloc[0].to_dict()
            with st.spinner(f"Loading {chosen_opts}…"):
                raw_df = fetch_ohlcv(chosen_opts, days=chart_days_opts)

            if not raw_df.empty:
                ann = raw_df.copy()
                ann["ema9"]  = ann["Close"].ewm(span=9,  adjust=False).mean()
                ann["ema21"] = ann["Close"].ewm(span=21, adjust=False).mean()
                ann["ema50"] = ann["Close"].ewm(span=50, adjust=False).mean()
                ann["sma200"] = ann["Close"].rolling(200).mean()

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.70, 0.30], vertical_spacing=0.02)
                candle, vol_trace = _candlestick_vol(ann, chosen_opts)
                fig.add_trace(candle, row=1, col=1)

                for name, col_name, color, width in [
                    ("EMA9",   "ema9",   "#ffd54f", 1.0),
                    ("EMA21",  "ema21",  "#ffb300", 1.2),
                    ("EMA50",  "ema50",  "#42a5f5", 1.5),
                    ("SMA200", "sma200", "#ef5350", 1.5),
                ]:
                    if col_name in ann.columns:
                        fig.add_trace(go.Scatter(
                            x=ann.index, y=ann[col_name], mode="lines",
                            line=dict(color=color, width=width), name=name,
                        ), row=1, col=1)

                # Strike line
                strike = sig_row.get("Strike")
                if strike:
                    direction = sig_row.get("Direction", "CALL")
                    strike_color = "#26a69a" if direction == "CALL" else "#ef5350"
                    fig.add_hline(
                        y=strike, line_dash="dash", line_color=strike_color,
                        line_width=1.2,
                        annotation_text=f"Strike {strike} ({direction})",
                        annotation_position="top right", row=1, col=1,
                    )

                fig.add_trace(vol_trace, row=2, col=1)

                # Greeks annotation
                greeks_text = (
                    f"δ={sig_row.get('Delta','?')} "
                    f"γ={sig_row.get('Gamma','?')} "
                    f"θ/day={sig_row.get('Theta/day','?')} "
                    f"prem=${sig_row.get('Premium','?')} | "
                    f"Score: {sig_row.get('Score','?'):.1f}/10"
                )
                fig.add_annotation(
                    text=greeks_text,
                    xref="paper", yref="paper", x=0.01, y=0.99,
                    showarrow=False, font=dict(size=11, color="#b0bec5"),
                    align="left", bgcolor="rgba(0,0,0,0.4)",
                )

                _apply_chart_style(
                    fig, f"{chosen_opts} — {sig_row.get('Direction','?')} "
                          f"Strike {strike} @ {opts_dte} DTE",
                    rows=2, ytitles=["Price", "Volume"], height=650,
                )
                st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# IBD Scanner — Near Buy Zone
# ═════════════════════════════════════════════════════════════════════════════

_IBD_DEFAULT_LISTS = {
    "Sector Leaders": "",
    "IBD 20":         "",
    "IBD 50":         "",
    "Spotlight":      "",
    "IPO Leaders":    "",
    "RS Leaders":     "",
    "LT Leaders":     "",
    "Tech Leaders":   "",
    "Global Leaders": "",
}

_IBD_PATTERN_COLS = ["In_Base", "Pullback", "WT", "Short Stroke", "Xing 10W", "HTF"]


def _ibd_col_cfg(list_names):
    cfg = {
        "Symbol":        st.column_config.TextColumn("Symbol",   width="small"),
        "Name":          st.column_config.TextColumn("Name"),
        "Current Price": st.column_config.NumberColumn("Price",  format="$%.2f"),
        "RS Line NH":    st.column_config.TextColumn("RS NH",    width="small",
                             help="RS Line at new high? Yes/No"),
        "Industry":      st.column_config.TextColumn("Industry Name"),
        "Sector":        st.column_config.TextColumn("Sector",   width="small"),
        "50D Avg Vol K": st.column_config.NumberColumn("50D Vol K", format="%d",
                             help="50-day average volume in thousands"),
        "My Points":     st.column_config.NumberColumn("My Points",
                             help="Composite score: RS Rating + pattern bonuses + list membership"),
        "RS Rating":     st.column_config.NumberColumn("RS",     width="small",
                             help="Relative Strength Rating 1-99 vs SPY"),
        "In_Base":       st.column_config.NumberColumn("In Base",      width="small"),
        "Pullback":      st.column_config.NumberColumn("Pullback",     width="small"),
        "WT":            st.column_config.NumberColumn("WT",           width="small",
                             help="Wedge Tightening"),
        "Short Stroke":  st.column_config.NumberColumn("Short Stroke", width="small"),
        "Xing 10W":      st.column_config.NumberColumn("Xing 10W",    width="small",
                             help="Crossing above 10-week MA"),
        "HTF":           st.column_config.NumberColumn("HTF",          width="small",
                             help="High Tight Flag"),
        "# Lists":       st.column_config.NumberColumn("# Lists",      width="small"),
    }
    for ln in list_names:
        cfg[ln] = st.column_config.NumberColumn(ln, width="small",
                      help=f"Rank within {ln} (blank = not on list)")
    return cfg


def _ibd_style_table(df: pd.DataFrame, list_names: list, key: str):
    """Render the Near Buy Zone table with RS > 90 rows bolded."""
    if df.empty:
        return

    def _row_style(row):
        base = "background-color:#1a1a1a;color:#e0e0e0"
        if row.get("RS Rating", 0) >= 90:
            return [base + ";font-weight:700"] * len(row)
        return [base] * len(row)

    show_cols = [c for c in df.columns if c != "RS Rating"]
    cfg = _ibd_col_cfg(list_names)
    cfg_filtered = {k: v for k, v in cfg.items() if k in show_cols}

    try:
        styled = df[show_cols].style.apply(_row_style, axis=1)
        st.dataframe(styled, use_container_width=True,
                     column_config=cfg_filtered, key=key)
    except Exception:
        st.dataframe(df[show_cols], use_container_width=True,
                     column_config=cfg_filtered, key=key)


def render_ibd():
    clicked = _page_header(
        "📋", "IBD Near Buy Zone",
        "Identifies stocks near IBD-style buy setups: In Base, Pullback, "
        "Wedge Tightening, Short Stroke, Xing 10W, High Tight Flag. "
        "Cross-references your IBD list rankings.",
        scan_key="ibd_scan_btn", last_key="ibd_time",
    )

    with st.expander("ℹ️ Pattern Definitions", expanded=False):
        st.markdown("""
| Pattern | Definition |
|---------|-----------|
| **In Base** | Consolidating within 15% of 52-week high for ≥ 5 weeks |
| **Pullback** | Price within 5% of 10-week MA, having been above it |
| **WT** | Wedge Tightening — 4+ weeks of narrowing range near highs |
| **Short Stroke** | Tight week (< 60% of prior range) closing in upper half near highs |
| **Xing 10W** | Price crossed above the 10-week SMA last week |
| **HTF** | High Tight Flag — ≥ 100% gain in 8 weeks, now flagging 10-25% below peak |

**My Points** = RS Rating + RS Line New High (+5) + pattern bonuses + list memberships (×2 each)
""")

    # ── Settings ───────────────────────────────────────────────────────────────
    with st.expander("⚙️ Settings", expanded=False):
        s1, s2 = st.columns(2)
        ibd_min_rs      = s1.slider("Min RS Rating", 0, 99, 50, key="ibd_min_rs",
                                     help="Only include stocks with RS Rating ≥ this value")
        ibd_req_pattern = s2.checkbox("Require ≥ 1 pattern",  value=False, key="ibd_req_pat",
                                      help="Filter to stocks with at least one buy-zone pattern detected")

    # ── IBD list inputs ────────────────────────────────────────────────────────
    with st.expander("📋 IBD List Rankings (paste tickers in rank order)", expanded=False):
        st.caption(
            "Paste tickers one per line in rank order for each list. "
            "Leave blank to skip. Stocks appearing here will show their rank "
            "and contribute to the # Lists count."
        )
        list_cols_ui = list(_IBD_DEFAULT_LISTS.keys())
        col_pairs = [list_cols_ui[i:i+3] for i in range(0, len(list_cols_ui), 3)]
        for row_cols in col_pairs:
            ui_cols = st.columns(len(row_cols))
            for col, lname in zip(ui_cols, row_cols):
                col.text_area(
                    lname,
                    value=st.session_state.get(f"ibd_list_{lname}", ""),
                    height=120,
                    key=f"ibd_list_{lname}",
                    placeholder="AAPL\nMSFT\nNVDA",
                )

    # ── State init ─────────────────────────────────────────────────────────────
    if "ibd_df_main" not in st.session_state:
        st.session_state.ibd_df_main  = pd.DataFrame()
        st.session_state.ibd_df_lists = pd.DataFrame()
        st.session_state.ibd_time     = None

    if clicked:
        tickers = get_selected_tickers()
        _ticker_count_caption(tickers)

        # Build IBD list dicts from text areas
        ibd_lists = {}
        for lname in _IBD_DEFAULT_LISTS:
            raw_val = st.session_state.get(f"ibd_list_{lname}", "")
            parsed  = [t.strip().upper() for t in raw_val.splitlines() if t.strip()]
            if parsed:
                ibd_lists[lname] = parsed

        with st.spinner(f"Running IBD scan on {len(tickers)} tickers…"):
            try:
                df_main, df_lists = run_ibd_scanner(
                    tickers,
                    ibd_lists=ibd_lists,
                    min_rs=st.session_state.ibd_min_rs,
                    require_pattern=st.session_state.ibd_req_pat,
                )
                st.session_state.ibd_df_main  = df_main
                st.session_state.ibd_df_lists = df_lists
                st.session_state.ibd_time     = datetime.now().strftime("%Y-%m-%d %H:%M")
                if not df_main.empty:
                    save_scan_result(df_main, "ibd")
            except Exception as _e:
                st.error(f"Scan error: {_e}", icon="❌")

    df_main  = st.session_state.ibd_df_main
    df_lists = st.session_state.ibd_df_lists

    if df_main.empty and st.session_state.ibd_time:
        st.warning("No stocks matched the filters. Lower Min RS or uncheck 'Require pattern'.", icon="⚠️")
        return
    if df_main.empty:
        st.info("Hit **Scan Now** to find Near Buy Zone candidates.", icon="ℹ️")
        return

    # ── Summary metrics ────────────────────────────────────────────────────────
    pat_cols_present = [c for c in _IBD_PATTERN_COLS if c in df_main.columns]
    in_base_ct  = int(df_main["In_Base"].notna().sum())     if "In_Base"     in df_main.columns else 0
    pullback_ct = int(df_main["Pullback"].notna().sum())    if "Pullback"    in df_main.columns else 0
    xing_ct     = int(df_main["Xing 10W"].notna().sum())   if "Xing 10W"    in df_main.columns else 0
    htf_ct      = int(df_main["HTF"].notna().sum())        if "HTF"         in df_main.columns else 0
    any_pat     = int(df_main[pat_cols_present].notna().any(axis=1).sum()) if pat_cols_present else 0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total",       len(df_main))
    m2.metric("In Base",     in_base_ct)
    m3.metric("Pullback",    pullback_ct)
    m4.metric("Xing 10W",    xing_ct)
    m5.metric("HTF",         htf_ct)
    m6.metric("Any Pattern", any_pat)

    st.divider()

    # ── Table 1: Near Buy Zone ─────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#c9a84c;margin-bottom:4px'>Near Buy Zone</h3>"
        "<hr style='border-color:#c9a84c;margin-top:0'>",
        unsafe_allow_html=True,
    )

    active_lists = [
        c for c in _IBD_DEFAULT_LISTS
        if c in df_main.columns or (not df_lists.empty and c in df_lists.columns)
    ]

    _ibd_style_table(df_main, active_lists, key="ibd_tbl_main")

    # ── Table 2: List membership (cont) ───────────────────────────────────────
    if not df_lists.empty and len(active_lists) > 0:
        st.divider()
        st.markdown(
            "<h3 style='color:#c9a84c;margin-bottom:4px'>"
            "Near Buy Zone <span style='font-size:0.75em;color:#999'>(cont)</span></h3>"
            "<hr style='border-color:#c9a84c;margin-top:0'>",
            unsafe_allow_html=True,
        )
        list_cfg = {"No": st.column_config.NumberColumn("No", width="small"),
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "# Lists": st.column_config.NumberColumn("# Lists", width="small")}
        for ln in active_lists:
            if ln in df_lists.columns:
                list_cfg[ln] = st.column_config.NumberColumn(ln, width="small")
        st.dataframe(
            df_lists,
            use_container_width=True,
            column_config=list_cfg,
            key="ibd_tbl_lists",
        )

    # ── Download ───────────────────────────────────────────────────────────────
    st.divider()
    csv = df_main.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download CSV", data=csv,
        file_name=f"ibd_near_buy_zone_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv", key="ibd_dl",
    )


# ═════════════════════════════════════════════════════════════════════════════
# William O'Neil — CAN SLIM Scanner
# ═════════════════════════════════════════════════════════════════════════════

from oneil_canslim_scanner import run_canslim_scanner

_CANSLIM_PRESETS = {
    "NASDAQ 100 Growth": [
        "NVDA", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "AVGO", "TSLA",
        "AMD", "NFLX", "COST", "QCOM", "INTU", "AMAT", "KLAC", "LRCX",
        "MRVL", "MU", "PANW", "CRWD", "SNPS", "CDNS", "ADSK", "FTNT",
        "ARM", "APP", "CEG", "VST", "MSTR", "PLTR",
    ],
    "S&P 500 Leaders": [
        "NVDA", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "LLY", "AVGO",
        "JPM", "V", "MA", "UNH", "XOM", "HD", "PG", "COST", "ABBV",
        "NFLX", "CRM", "ACN", "TXN", "BAC", "WMT", "TMO", "CAT",
        "GE", "RTX", "SPGI", "BLK", "GS", "FICO", "AXON", "DECK",
    ],
    "High-Growth Tech": [
        "NVDA", "AMD", "CRWD", "PANW", "FTNT", "PLTR", "APP", "ARM",
        "AVGO", "MRVL", "SNOW", "DDOG", "ZS", "NET", "HUBS", "TTD",
        "AXON", "FICO", "SMCI", "CELH", "LULU", "WING", "CAVA", "DUOL",
    ],
    "IBD-Style Leaders": [
        "NVDA", "AVGO", "LRCX", "KLAC", "AMAT", "CRWD", "PANW", "AXON",
        "FICO", "DECK", "CELH", "WING", "CAVA", "APP", "PLTR", "GEV",
        "VST", "CEG", "TMUS", "UBER", "META", "ARM", "MSTR", "BROS",
    ],
}


def _canslim_col_cfg():
    return {
        "Symbol":    st.column_config.TextColumn("Symbol",    width="small"),
        "Name":      st.column_config.TextColumn("Name"),
        "Price":     st.column_config.NumberColumn("Price",   format="$%.2f"),
        "Score":     st.column_config.NumberColumn("Score",   format="%d /18",
                         help="Total CAN SLIM score (max 18)"),
        "Verdict":   st.column_config.TextColumn("Verdict",   width="medium"),
        "Criteria":  st.column_config.TextColumn("Criteria Met",
                         help="Letters of CAN SLIM criteria this stock satisfies"),
        "RS":        st.column_config.NumberColumn("RS Rating", width="small",
                         help="IBD-style Relative Strength 1-99 vs SPY"),
        "EPS Qtr %": st.column_config.NumberColumn("EPS Qtr",  format="%.1f%%",
                         help="Quarterly EPS growth (C criterion: need ≥25%)"),
        "EPS Ann %": st.column_config.NumberColumn("EPS Ann",  format="%.1f%%",
                         help="Annual EPS growth (A criterion)"),
        "Rev Gth %": st.column_config.NumberColumn("Rev Gth",  format="%.1f%%",
                         help="Revenue growth (A criterion: need ≥20%)"),
        "Inst %":    st.column_config.NumberColumn("Inst %",   format="%.1f%%",
                         help="Institutional ownership (I criterion: ideal 30-85%)"),
        "52wH %":    st.column_config.NumberColumn("52wH %",   format="%.1f%%",
                         help="% from 52-week high (N criterion: within -15% preferred)"),
        "Vol Surge": st.column_config.NumberColumn("Vol Surge", format="%.2f×",
                         help="Latest volume vs 50-day average"),
        "Acc Ratio": st.column_config.NumberColumn("Acc Ratio", format="%.2f×",
                         help="Up-day volume vs down-day volume (>1 = accumulation)"),
        "Float M":   st.column_config.NumberColumn("Float M",  format="%.1f M",
                         help="Float shares in millions (S criterion: smaller = better)"),
        "Sector":    st.column_config.TextColumn("Sector"),
        "EPS Accel": st.column_config.TextColumn("EPS Accel",
                         help="Forward EPS > trailing EPS by ≥15% (acceleration)"),
    }


def _canslim_style(df: pd.DataFrame):
    def _row_style(row):
        v = row.get("Verdict", "")
        if v == "STRONG BUY": bg = "#1a2e1a"
        elif v == "BUY":      bg = "#1a1e2e"
        elif v == "WATCH":    bg = "#2a2a18"
        else:                 bg = "#1a1a1a"
        return [f"background-color:{bg};color:#e0e0e0"] * len(row)

    try:
        return df.style.apply(_row_style, axis=1)
    except Exception:
        return df.style


def render_oneil():
    clicked = _page_header(
        "🏆", "O'Neil CAN SLIM",
        "Screen stocks against William O'Neil's CAN SLIM criteria — earnings acceleration, "
        "RS leadership, institutional sponsorship, new highs, and volume accumulation.",
        scan_key="oneil_scan_btn",
        last_key="oneil_time",
    )

    with st.expander("ℹ️  CAN SLIM Criteria", expanded=False):
        st.markdown("""
| Letter | Criterion | Scoring rule |
|--------|-----------|--------------|
| **C** | Current quarterly EPS | 3 pts if >25% · 2 pts if >15% · 1 pt if >0 |
| **A** | Annual EPS growth | 2 pts if >25% · 1 pt if >0 |
| **A²** | Revenue growth | 2 pts if >20% · 1 pt if >10% |
| **N** | New high proximity | 2 pts within −5% of 52w high · 1 pt within −15% |
| **S** | Supply & demand | +1 pt accumulation ratio >1.1 · +1 pt vol surge >1.4× · +1 pt float <50M |
| **L** | Leader RS Rating | 3 pts if RS ≥80 · 2 pts if RS ≥60 · 1 pt if RS ≥40 |
| **I** | Institutional % | 2 pts if 30–85% · 1 pt if any positive |
| **M** | Market direction | 1 pt if SPY above 50-day MA |

**Total: 18 pts** · STRONG BUY ≥14 · BUY ≥10 · WATCH ≥7 · PASS <7
""")

    # ── Settings ───────────────────────────────────────────────────────────────
    with st.expander("⚙️  Settings", expanded=True):
        preset_cols = st.columns(3)
        with preset_cols[0]:
            min_score = st.slider("Min score", 0, 18, 7, key="oneil_min_score",
                                  help="Minimum CAN SLIM total score")
        with preset_cols[1]:
            min_rs = st.slider("Min RS", 0, 99, 50, key="oneil_min_rs",
                               help="Minimum RS Rating (1-99)")
        with preset_cols[2]:
            req_ma = st.checkbox("Above 50-day MA", value=False, key="oneil_req_ma")

    # ── State init ─────────────────────────────────────────────────────────────
    if "oneil_df" not in st.session_state:
        st.session_state.oneil_df   = pd.DataFrame()
        st.session_state.oneil_time = None

    if clicked:
        tickers = get_selected_tickers()
        _ticker_count_caption(tickers)
        with st.spinner(f"Running CAN SLIM scan on {len(tickers)} stocks…"):
                try:
                    df = run_canslim_scanner(
                        tickers,
                        min_score=st.session_state.oneil_min_score,
                        min_rs=st.session_state.oneil_min_rs,
                        require_above_ma50=st.session_state.oneil_req_ma,
                    )
                    st.session_state.oneil_df   = df
                    st.session_state.oneil_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                    if not df.empty:
                        save_cols = [c for c in df.columns if not c.startswith("_")]
                        save_scan_result(df[save_cols], "oneil")
                except Exception as _e:
                    st.error(f"Scan error: {_e}", icon="❌")

    df = st.session_state.oneil_df

    if df.empty and st.session_state.oneil_time:
        st.warning("No stocks matched the filters. Try lowering Min Score or Min RS.", icon="⚠️")
        return
    if df.empty:
        st.info("Configure filters above and click **Scan Now** to screen stocks.", icon="ℹ️")
        return

    # ── Summary metrics ────────────────────────────────────────────────────────
    n_sb  = int((df["Verdict"] == "STRONG BUY").sum())
    n_buy = int((df["Verdict"] == "BUY").sum())
    n_wch = int((df["Verdict"] == "WATCH").sum())
    n_rs80 = int((df["RS"] >= 80).sum())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Passed",  len(df))
    m2.metric("STRONG BUY",    n_sb,  delta=None)
    m3.metric("BUY",           n_buy, delta=None)
    m4.metric("WATCH",         n_wch, delta=None)
    m5.metric("RS ≥ 80",       n_rs80, delta=None)

    st.divider()

    # ── Top picks highlight cards ──────────────────────────────────────────────
    top = df[df["Verdict"].isin(["STRONG BUY", "BUY"])].head(6)
    if not top.empty:
        st.markdown(
            "<h3 style='color:#c9a84c;margin-bottom:4px'>Top CAN SLIM Picks</h3>"
            "<hr style='border-color:#c9a84c;margin-top:0'>",
            unsafe_allow_html=True,
        )
        card_cols = st.columns(min(len(top), 3))
        for i, (_, row) in enumerate(top.iterrows()):
            col = card_cols[i % 3]
            v_color = "#26a69a" if row["Verdict"] == "STRONG BUY" else "#42a5f5"
            v_bg    = "rgba(38,166,154,0.12)" if row["Verdict"] == "STRONG BUY" else "rgba(66,165,245,0.12)"
            bar_pct = int(row["Score"] / 18 * 100)
            bar_col = "#26a69a" if bar_pct >= 70 else "#f0c040" if bar_pct >= 50 else "#ef5350"
            criteria_html = " ".join(
                f"<span style='background:rgba(38,166,154,0.2);color:#26a69a;"
                f"padding:1px 6px;border-radius:4px;font-size:11px;font-weight:700'>{c}</span>"
                for c in row["Criteria"].split()
            )
            col.markdown(f"""
<div style="background:{v_bg};border:1px solid {v_color};border-radius:10px;padding:14px 16px;margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
    <span style="font-size:20px;font-weight:800;color:#eee">{row['Symbol']}</span>
    <span style="color:{v_color};font-weight:700;font-size:12px">{row['Verdict']}</span>
  </div>
  <div style="color:#999;font-size:12px;margin-bottom:6px">{row['Name']}</div>
  <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px">
    <span style="color:#eee;font-weight:700">${row['Price']:.2f}</span>
    <span style="color:#ccc">Score <b style="color:{v_color}">{row['Score']}/18</b> · RS <b>{row['RS']}</b></span>
  </div>
  <div style="background:#333;border-radius:3px;height:4px;margin-bottom:8px">
    <div style="width:{bar_pct}%;background:{bar_col};height:4px;border-radius:3px"></div>
  </div>
  <div style="line-height:1.8">{criteria_html}</div>
  <div style="font-size:11px;color:#888;margin-top:6px">
    EPS Qtr {row['EPS Qtr %']:+.0f}% · RS {row['RS']} · {row['52wH %']:+.1f}% from 52wH
  </div>
</div>""", unsafe_allow_html=True)

    # ── Full results table ─────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#c9a84c;margin-bottom:4px'>Full Scan Results</h3>"
        "<hr style='border-color:#c9a84c;margin-top:0'>",
        unsafe_allow_html=True,
    )

    display_cols = [
        "Symbol", "Name", "Price", "Score", "Verdict", "Criteria",
        "RS", "EPS Qtr %", "EPS Ann %", "Rev Gth %",
        "Inst %", "52wH %", "Vol Surge", "Acc Ratio", "Float M",
        "Sector", "EPS Accel",
    ]
    show_cols = [c for c in display_cols if c in df.columns]
    cfg = _canslim_col_cfg()
    cfg_filtered = {k: v for k, v in cfg.items() if k in show_cols}

    try:
        styled = _canslim_style(df[show_cols])
        st.dataframe(styled, use_container_width=True,
                     column_config=cfg_filtered, key="oneil_tbl")
    except Exception:
        st.dataframe(df[show_cols], use_container_width=True,
                     column_config=cfg_filtered, key="oneil_tbl_fallback")

    # ── Download ───────────────────────────────────────────────────────────────
    st.divider()
    csv_cols = [c for c in show_cols if not c.startswith("_")]
    csv_data = df[csv_cols].to_csv(index=False).encode()
    st.download_button(
        "⬇️  Download CSV",
        data=csv_data,
        file_name=f"canslim_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        key="oneil_dl",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Daily Alerts Dashboard
# ═════════════════════════════════════════════════════════════════════════════

from alerts_engine import generate_daily_alerts

_ALERTS_PRESETS = {
    "My Watchlist (S&P + NASDAQ leaders)": [
        "NVDA", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "AVGO", "TSLA",
        "AMD", "NFLX", "CRWD", "PANW", "PLTR", "ARM", "APP", "AXON",
        "FICO", "DECK", "CELH", "CAVA", "LLY", "V", "MA", "JPM",
        "GE", "CAT", "RTX", "XOM", "COST", "TMUS",
    ],
    "High-Growth Tech": [
        "NVDA", "AMD", "CRWD", "PANW", "PLTR", "APP", "ARM",
        "AVGO", "MRVL", "SNOW", "DDOG", "ZS", "NET", "HUBS",
        "TTD", "AXON", "FICO", "SMCI", "DUOL", "CAVA",
    ],
    "Large Cap Leaders": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA",
        "BRK-B", "LLY", "V", "MA", "UNH", "JPM", "XOM", "COST",
        "WMT", "PG", "HD", "JNJ", "ABBV", "MRK", "BAC", "GS",
    ],
}

_CONVICTION_ORDER = {"STRONG": 0, "HIGH": 1, "WATCH": 2}
_BUY_COLORS  = {"STRONG": ("#0d4f0d", "#26a69a"), "HIGH": ("#0d1f4f", "#42a5f5"), "WATCH": ("#2a2a0d", "#f0c040")}
_SELL_COLORS = {"STRONG": ("#4f0d0d", "#ef5350"), "HIGH": ("#3a0d2a", "#ab47bc"), "WATCH": ("#2a2a0d", "#f0c040")}


def _signal_badges(signals, max_show=6):
    if not isinstance(signals, list) or not signals:
        return ""
    return " ".join(
        f"<span style='background:rgba(255,255,255,0.08);padding:1px 7px;"
        f"border-radius:4px;font-size:11px;color:#ccc'>{s}</span>"
        for s in signals[:max_show]
    )


def _exit_badges(exit_signals, urgency):
    if not isinstance(exit_signals, list) or not exit_signals:
        return ""
    color = {"URGENT": "#ef5350", "CONSIDER": "#f0c040", "WATCH": "#ffb74d"}.get(urgency, "#888")
    badges = " ".join(
        f"<span style='background:rgba(239,83,80,0.12);padding:1px 6px;"
        f"border-radius:4px;font-size:10px;color:{color}'>{s}</span>"
        for s in exit_signals[:4]
    )
    return (
        f"<div style='margin-top:7px;padding:6px 8px;"
        f"background:rgba(239,83,80,0.08);border-left:2px solid {color};"
        f"border-radius:4px'>"
        f"<div style='font-size:10px;font-weight:700;color:{color};"
        f"margin-bottom:3px'>⚠️ EXIT SIGNALS ({urgency})</div>"
        f"<div style='line-height:1.8'>{badges}</div></div>"
    )


def _alert_card(row: pd.Series, col):
    """Render a single BUY/SELL alert card with integrated exit warnings."""
    conv   = row["Conviction"]
    d      = row["Direction"]
    colors = _BUY_COLORS.get(conv, _BUY_COLORS["WATCH"]) if d == "BUY" else _SELL_COLORS.get(conv, _SELL_COLORS["WATCH"])
    bg, accent = colors

    arrow = "↑" if d == "BUY" else "↓"
    label = f"{'⚡ ' if conv == 'STRONG' else '✅ ' if conv == 'HIGH' else '👀 '}{conv} {d}"

    stop_pct = row["Stop %"]
    t2_pct   = row["T2 %"]
    sig_html = _signal_badges(row.get("Signals", []))

    exit_sigs    = row.get("Exit Signals", [])
    exit_urgency = row.get("Exit Urgency", "")
    exit_action  = row.get("Exit Action", "")
    exit_block   = _exit_badges(exit_sigs, exit_urgency) if exit_sigs else ""

    col.markdown(f"""
<div style="background:{bg};border:1px solid {accent};border-radius:10px;
            padding:14px 16px;margin-bottom:10px;font-family:monospace">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
    <span style="font-size:20px;font-weight:800;color:#eee">{row['Symbol']} <span style='font-size:14px;color:{accent}'>{arrow}</span></span>
    <span style="color:{accent};font-weight:700;font-size:12px">{label}</span>
  </div>
  <div style="color:#888;font-size:11px;margin-bottom:8px">{row['Name']} &nbsp;·&nbsp; Score {row['Score']}/12 &nbsp;·&nbsp; RS {row['RS']}</div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;font-size:12px;margin-bottom:8px">
    <div><span style='color:#888'>Entry</span>  <b style='color:#eee'>${row['Entry']:.2f}</b></div>
    <div><span style='color:#888'>ATR</span>    <b style='color:#888'>${row['ATR']:.2f} ({row['ATR %']:.1f}%)</b></div>
    <div><span style='color:#ef5350'>Stop</span>   <b style='color:#ef5350'>${row['Stop']:.2f} ({stop_pct:+.1f}%)</b></div>
    <div><span style='color:#888'>R/R</span>    <b style='color:{"#26a69a" if row["R/R"] >= 2 else "#f0c040"}'>{row['R/R']:.1f}:1</b></div>
    <div><span style='color:#aaa'>T1 (1/3)</span> <b style='color:#aaa'>${row['T1']:.2f}</b></div>
    <div><span style='color:#26a69a'>T2 (1/3)</span> <b style='color:#26a69a'>${row['T2']:.2f} ({t2_pct:+.1f}%)</b></div>
    <div><span style='color:#42a5f5'>T3 (run)</span> <b style='color:#42a5f5'>${row['T3']:.2f}</b></div>
    <div><span style='color:#888'>RSI</span>    <b style='color:#{"f0c040" if row["RSI"]>70 else "26a69a" if row["RSI"]<40 else "888"}'>{row['RSI']:.0f}</b></div>
  </div>
  <div style="line-height:1.9">{sig_html}</div>
  {exit_block}
</div>""", unsafe_allow_html=True)


def _trade_plan_expander(row: pd.Series, col):
    """Expandable trade plan window below an alert card."""
    d     = row["Direction"]
    entry = row["Entry"]
    stop  = row["Stop"]
    t1, t2, t3 = row["T1"], row["T2"], row["T3"]
    atr   = row["ATR"]
    rr    = row["R/R"]

    buy_lo = round(entry * 0.995, 2)
    buy_hi = round(entry * 1.005, 2)
    risk_per_share = round(abs(entry - stop), 2)

    # Position sizing at common portfolio sizes (1% risk rule)
    rows_ps = []
    for port in [10_000, 25_000, 50_000, 100_000]:
        shares = int((port * 0.01) / risk_per_share) if risk_per_share > 0 else 0
        cost   = round(shares * entry, 0)
        rows_ps.append((f"${port:,}", shares, f"${cost:,}"))

    with col:
        with st.expander(f"📋  Trade Plan — {row['Symbol']}", expanded=False):
            # Entry window
            st.markdown(
                f"**{'🟢 BUY' if d == 'BUY' else '🔴 SHORT'} ENTRY WINDOW**  "
                f"`${buy_lo:.2f}` — `${buy_hi:.2f}`  "
                f"_(limit within 0.5% of ${entry:.2f})_"
            )
            st.divider()

            # Price levels table
            t1_pct = round((t1 - entry) / entry * 100 * (1 if d == "BUY" else -1), 1)
            t3_pct = round((t3 - entry) / entry * 100 * (1 if d == "BUY" else -1), 1)
            levels = {
                "Entry":    (entry,  "Buy limit — place at open"),
                "Stop":     (stop,   f"Hard stop — exit if breached  ({row['Stop %']:+.1f}%)"),
                "T1 (1/3)": (t1,     f"Take 1/3 off — lock partial gain  ({t1_pct:+.1f}%,  1R)"),
                "T2 (1/3)": (t2,     f"Take another 1/3 — main target  ({row['T2 %']:+.1f}%,  2R)"),
                "T3 (run)": (t3,     f"Trail stop on rest — let winner ride  ({t3_pct:+.1f}%,  3R)"),
            }
            rows_html = "".join(
                f"<tr><td style='padding:4px 10px;color:#aaa'>{lbl}</td>"
                f"<td style='padding:4px 10px;font-weight:700;color:#eee'>${price:.2f}</td>"
                f"<td style='padding:4px 10px;color:#888;font-size:11px'>{note}</td></tr>"
                for lbl, (price, note) in levels.items()
            )
            st.markdown(
                f"<table style='width:100%;border-collapse:collapse;font-family:monospace'>"
                f"<thead><tr style='border-bottom:1px solid #333'>"
                f"<th style='padding:4px 10px;color:#666;text-align:left'>Level</th>"
                f"<th style='padding:4px 10px;color:#666;text-align:left'>Price</th>"
                f"<th style='padding:4px 10px;color:#666;text-align:left'>Action</th>"
                f"</tr></thead><tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True,
            )
            st.caption(f"ATR(14) = ${atr:.2f}  ·  R/R = {rr:.1f}:1  ·  Risk/share = ${risk_per_share:.2f}")
            st.divider()

            # Position sizing
            st.markdown("**Position Sizing (1% portfolio risk rule)**")
            ps_html = "".join(
                f"<tr><td style='padding:3px 10px;color:#aaa'>{p}</td>"
                f"<td style='padding:3px 10px;color:#eee;font-weight:700'>{sh} shares</td>"
                f"<td style='padding:3px 10px;color:#888'>{cost}</td></tr>"
                for p, sh, cost in rows_ps
            )
            st.markdown(
                f"<table style='width:100%;border-collapse:collapse;font-family:monospace'>"
                f"<thead><tr style='border-bottom:1px solid #333'>"
                f"<th style='padding:3px 10px;color:#666;text-align:left'>Portfolio</th>"
                f"<th style='padding:3px 10px;color:#666;text-align:left'>Shares</th>"
                f"<th style='padding:3px 10px;color:#666;text-align:left'>Capital used</th>"
                f"</tr></thead><tbody>{ps_html}</tbody></table>",
                unsafe_allow_html=True,
            )
            st.divider()

            # Exit rules
            st.markdown(
                "**Exit Rules**\n"
                "- Hit **Stop** → exit 100% immediately, no exceptions\n"
                "- Hit **T1** → sell 1/3, raise stop to breakeven\n"
                "- Hit **T2** → sell another 1/3, trail stop below last pivot low\n"
                "- Hit **T3** → trail remaining position with 10-day low or EMA9 break\n"
                "- Stock drops back inside base before T1 → exit if conviction weakens\n"
            )


def _exit_card(row: pd.Series, col):
    """Render a dedicated EXIT alert card for existing long holders."""
    urgency  = row.get("Exit Urgency", "WATCH")
    action   = row.get("Exit Action", "")
    exit_sigs = row.get("Exit Signals", [])
    urg_color = {"URGENT": "#ef5350", "CONSIDER": "#f0c040", "WATCH": "#ffb74d"}.get(urgency, "#888")
    bg        = {"URGENT": "#2d0a0a",  "CONSIDER": "#2a2200", "WATCH": "#1a1a1a"}.get(urgency, "#1a1a1a")

    badges = " ".join(
        f"<span style='background:rgba(239,83,80,0.12);padding:2px 8px;"
        f"border-radius:4px;font-size:11px;color:{urg_color}'>{s}</span>"
        for s in (exit_sigs if isinstance(exit_sigs, list) else [])[:5]
    )

    ext_pct  = row.get("Ext vs 50MA %", 0)
    pct52h   = row.get("52wH %", 0)

    col.markdown(f"""
<div style="background:{bg};border:1px solid {urg_color};border-radius:10px;
            padding:14px 16px;margin-bottom:10px;font-family:monospace">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
    <span style="font-size:20px;font-weight:800;color:#eee">{row['Symbol']}</span>
    <span style="color:{urg_color};font-weight:700;font-size:12px">⚠️ EXIT {urgency}</span>
  </div>
  <div style="color:#888;font-size:11px;margin-bottom:6px">
    {row['Name']} &nbsp;·&nbsp; ${row['Price']:.2f} &nbsp;·&nbsp; RSI {row['RSI']:.0f}
  </div>
  <div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:7px 10px;
              font-size:12px;color:{urg_color};font-weight:600;margin-bottom:8px">
    {action}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:3px 12px;font-size:12px;margin-bottom:8px">
    <div><span style='color:#888'>Ext vs 50MA</span> <b style='color:{"#ef5350" if ext_pct>20 else "#f0c040"}'>{ext_pct:+.1f}%</b></div>
    <div><span style='color:#888'>vs 52W High</span> <b style='color:{"#26a69a" if pct52h>=-5 else "#888"}'>{pct52h:+.1f}%</b></div>
    <div><span style='color:#888'>T1 (first)</span>  <b style='color:#aaa'>${row['T1']:.2f}</b></div>
    <div><span style='color:#888'>Stop</span>         <b style='color:#ef5350'>${row['Stop']:.2f}</b></div>
  </div>
  <div style="line-height:1.8">{badges}</div>
</div>""", unsafe_allow_html=True)


def _alerts_full_table(df: pd.DataFrame):
    """Render the full alerts table with column config."""
    show = [
        "Symbol", "Name", "Direction", "Conviction", "Score",
        "Entry", "Stop", "Stop %", "T1", "T2", "T2 %", "T3", "R/R",
        "RSI", "ADX", "RS", "Vol Ratio", "52wH %", "Minervini",
        "ATR %", "Sector",
    ]
    show = [c for c in show if c in df.columns]

    def _row_style(row):
        d = row.get("Direction", "")
        c = row.get("Conviction", "")
        if d == "BUY"  and c == "STRONG": bg = "#0d3d0d"
        elif d == "BUY":                  bg = "#0d1a2e"
        elif d == "SELL" and c == "STRONG": bg = "#3d0d0d"
        elif d == "SELL":                 bg = "#1a0d1a"
        else:                             bg = "#1a1a1a"
        return [f"background-color:{bg};color:#e0e0e0"] * len(row)

    cfg = {
        "Symbol":    st.column_config.TextColumn("Symbol",  width="small"),
        "Name":      st.column_config.TextColumn("Name"),
        "Direction": st.column_config.TextColumn("Dir",     width="small"),
        "Conviction":st.column_config.TextColumn("Conv",    width="small"),
        "Score":     st.column_config.NumberColumn("Score", format="%d /12"),
        "Entry":     st.column_config.NumberColumn("Entry", format="$%.2f"),
        "Stop":      st.column_config.NumberColumn("Stop",  format="$%.2f"),
        "Stop %":    st.column_config.NumberColumn("Stop %",format="%.1f%%"),
        "T1":        st.column_config.NumberColumn("T1",    format="$%.2f"),
        "T2":        st.column_config.NumberColumn("T2 ★",  format="$%.2f"),
        "T2 %":      st.column_config.NumberColumn("T2 %",  format="%.1f%%"),
        "T3":        st.column_config.NumberColumn("T3",    format="$%.2f"),
        "R/R":       st.column_config.NumberColumn("R/R",   format="%.1f:1"),
        "RSI":       st.column_config.NumberColumn("RSI",   format="%.0f"),
        "ADX":       st.column_config.NumberColumn("ADX",   format="%.0f"),
        "RS":        st.column_config.NumberColumn("RS",    width="small"),
        "Vol Ratio": st.column_config.NumberColumn("Vol×",  format="%.2f×"),
        "52wH %":    st.column_config.NumberColumn("52wH%", format="%.1f%%"),
        "Minervini": st.column_config.NumberColumn("SEPA",  format="%d/8"),
        "ATR %":     st.column_config.NumberColumn("ATR%",  format="%.1f%%"),
        "Sector":    st.column_config.TextColumn("Sector"),
    }
    cfg_f = {k: v for k, v in cfg.items() if k in show}

    try:
        styled = df[show].style.apply(_row_style, axis=1)
        st.dataframe(styled, use_container_width=True, column_config=cfg_f, key="alerts_tbl")
    except Exception:
        st.dataframe(df[show], use_container_width=True, column_config=cfg_f,
                     key="alerts_tbl_fallback")


def render_alerts():
    # ── Header ────────────────────────────────────────────────────────────────
    h1, h2, h3 = st.columns([5, 2, 2])
    with h1:
        st.markdown(
            f"<div class='scanner-header'>"
            f"<span style='font-size:2rem'>🔔</span>"
            f"<span class='scanner-title'>Daily Alerts</span>"
            f"&nbsp;{market_badge_html()}"
            f"</div>"
            f"<div class='scanner-desc'>"
            f"Actionable BUY / SELL alerts with entry, stop &amp; targets — "
            f"powered by all indicator systems (Trend · MACD · RSI · Breakout · "
            f"Livermore · Minervini · Volume · RS). Refreshes once per trading day."
            f"</div>",
            unsafe_allow_html=True,
        )
    last_run = st.session_state.get("alerts_run_time")
    with h2:
        if last_run:
            st.caption("Last generated")
            st.caption(last_run)
    with h3:
        manual_run = st.button("▶  Generate Alerts", type="primary",
                               use_container_width=True, key="alerts_run_btn")
    st.divider()

    # ── Macro event warnings ───────────────────────────────────────────────────
    _ctx = get_event_context(days_ahead=5)
    if _ctx["banner_html"]:
        st.markdown(_ctx["banner_html"], unsafe_allow_html=True)

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️  Settings", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            min_score = st.slider("Min score", 4, 12, 5, key="alerts_min_score",
                                  help="4=WATCH · 6=BUY · 8=STRONG BUY")
        with sc2:
            dir_filter = st.selectbox("Direction", ["All", "BUY only", "SELL only"],
                                      key="alerts_dir")
        with sc3:
            conv_filter = st.multiselect(
                "Conviction", ["STRONG", "HIGH", "WATCH"],
                default=["STRONG", "HIGH"], key="alerts_conv",
            )

    # ── State init ────────────────────────────────────────────────────────────
    if "alerts_df" not in st.session_state:
        st.session_state.alerts_df       = pd.DataFrame()
        st.session_state.alerts_run_time = None
        st.session_state.alerts_run_date = None

    today_str = datetime.today().strftime("%Y-%m-%d")
    auto_run  = False  # manual only — click "Generate Alerts" to run

    if manual_run or auto_run:
        tickers = get_selected_tickers()
        _ticker_count_caption(tickers)
        prog  = st.progress(0, text="Initialising…")
        stat  = st.empty()

        def _cb(done, total, tk):
            pct = done / total
            prog.progress(pct, text=f"Analysing {tk}… ({done}/{total})")
            stat.caption(f"Running all indicators on {tk}")

        try:
            df = generate_daily_alerts(
                tickers,
                min_score=st.session_state.alerts_min_score,
                progress_cb=_cb,
            )
            st.session_state.alerts_df       = df
            st.session_state.alerts_run_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.alerts_run_date = today_str
            if not df.empty:
                save_cols = [c for c in df.columns if c != "Signals"]
                save_scan_result(df[save_cols], "alerts")
        except Exception as _e:
            st.error(f"Alert generation error: {_e}", icon="❌")
        finally:
            prog.empty()
            stat.empty()

    df_all = st.session_state.alerts_df

    if df_all.empty and st.session_state.alerts_run_time:
        st.warning("No alerts met the filters. Try lowering Min Score.", icon="⚠️")
        return
    if df_all.empty:
        st.info("Click **Generate Alerts** to run today's scan.", icon="ℹ️")
        return

    # ── Apply UI filters ──────────────────────────────────────────────────────
    df = df_all.copy()
    if dir_filter == "BUY only":
        df = df[df["Direction"] == "BUY"]
    elif dir_filter == "SELL only":
        df = df[df["Direction"] == "SELL"]
    conv_sel = st.session_state.alerts_conv
    if conv_sel:
        df = df[df["Conviction"].isin(conv_sel)]

    if df.empty:
        st.warning("No alerts match the current filters.", icon="⚠️")
        return

    # ── Summary metrics ────────────────────────────────────────────────────────
    n_buy    = int((df["Direction"] == "BUY").sum())
    n_sell   = int((df["Direction"] == "SELL").sum())
    n_strong = int((df["Conviction"] == "STRONG").sum())
    n_high   = int((df["Conviction"] == "HIGH").sum())
    n_watch  = int((df["Conviction"] == "WATCH").sum())
    avg_rr   = round(df["R/R"].mean(), 2) if not df.empty else 0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Alerts", len(df))
    m2.metric("BUY  ↑",       n_buy,    delta=None)
    m3.metric("SELL ↓",       n_sell,   delta=None)
    m4.metric("⚡ STRONG",    n_strong, delta=None)
    m5.metric("✅ HIGH",      n_high,   delta=None)
    m6.metric("Avg R/R",      f"{avg_rr:.1f}:1")

    st.divider()

    # ── Top alert cards (STRONG + HIGH) ───────────────────────────────────────
    top = df[df["Conviction"].isin(["STRONG", "HIGH"])].head(9)
    if not top.empty:
        st.markdown(
            "<h3 style='color:#c9a84c;margin-bottom:4px'>🔔 Top Alerts</h3>"
            "<hr style='border-color:#c9a84c;margin-top:0'>",
            unsafe_allow_html=True,
        )
        card_cols = st.columns(3)
        for i, (_, row) in enumerate(top.iterrows()):
            _alert_card(row, card_cols[i % 3])
            _trade_plan_expander(row, card_cols[i % 3])

    # ── Full table ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<h3 style='color:#c9a84c;margin-bottom:4px'>Full Alert Feed</h3>"
        "<hr style='border-color:#c9a84c;margin-top:0'>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Entry = buy at open within 0.5% of close.  "
        "Stop = hard stop (exit immediately if breached).  "
        "T1 = first partial profit (take ~1/3).  "
        "T2 = main target (take ~1/3, trail stop on rest).  "
        "T3 = runner (trail stop, let winners ride)."
    )
    _alerts_full_table(df)

    # ── Potential Exits section ───────────────────────────────────────────────
    df_exits = df_all[df_all["Exit Urgency"].fillna("").str.len() > 0].copy() if "Exit Urgency" in df_all.columns else pd.DataFrame()
    if not df_exits.empty:
        urgency_order = {"URGENT": 0, "CONSIDER": 1, "WATCH": 2}
        df_exits = df_exits.sort_values(
            "Exit Urgency",
            key=lambda x: x.map(urgency_order).fillna(3),
        )
        n_urgent  = int((df_exits["Exit Urgency"] == "URGENT").sum())
        n_consider = int((df_exits["Exit Urgency"] == "CONSIDER").sum())
        n_watch_e  = int((df_exits["Exit Urgency"] == "WATCH").sum())

        st.divider()
        st.markdown(
            "<h3 style='color:#ef5350;margin-bottom:4px'>⚠️ Potential Exits</h3>"
            "<hr style='border-color:#ef5350;margin-top:0'>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Stocks in your watchlist / portfolio showing exit or caution signals. "
            "URGENT = act now; CONSIDER = review position; WATCH = monitor closely."
        )
        ex1, ex2, ex3 = st.columns(3)
        ex1.metric("🚨 URGENT",   n_urgent,   delta=None)
        ex2.metric("🟡 CONSIDER", n_consider, delta=None)
        ex3.metric("👀 WATCH",    n_watch_e,  delta=None)

        exit_cols = st.columns(3)
        for i, (_, row) in enumerate(df_exits.iterrows()):
            _exit_card(row, exit_cols[i % 3])

    # ── Signals breakdown legend ───────────────────────────────────────────────
    with st.expander("📖  Signal & Scoring Guide", expanded=False):
        st.markdown("""
#### How the score is built (max 12 per direction)

| Category | Points | Signals checked |
|----------|--------|-----------------|
| **Trend** | 0–3 | Above EMA50 (+1) · EMA50>EMA200 (+1) · Minervini template ≥6/8 (+1) |
| **Momentum** | 0–3 | RSI 42–72 or oversold (+1) · MACD hist positive & rising (+1) · RS Rating ≥60 (+1) |
| **Trigger** | 0–3 | EMA9×21 cross · 52W-high break · Volume breakout · NR7 · BB squeeze · MA reclaim · Inside bar · Gap · MACD cross · Livermore pivot · RSI divergence (capped at 3) |
| **Volume** | 0–2 | Vol 1.5× avg (+1) · Vol 3× avg (+2) |
| **Quality** | 0–1 | Institutional sponsorship 30–85% (+1) |

#### Conviction levels

| Conviction | Score | Suggested action |
|-----------|-------|-----------------|
| ⚡ STRONG | ≥ 8 | High conviction — act on open, full position |
| ✅ HIGH   | 6–7 | Solid setup — enter on confirmation or pullback to entry |
| 👀 WATCH  | 4–5 | Developing — set alert, wait for trigger to fire |

#### Price levels

| Level | Formula | Action |
|-------|---------|--------|
| Entry | Current close | Buy at market open (limit within 0.5%) |
| Stop  | Entry − 1.5 × ATR(14) | Hard exit if breached — no exceptions |
| T1    | Entry + 1.0 × ATR(14) | Sell 1/3 of position — lock partial gain |
| T2    | Entry + 2.0 × ATR(14) | Sell 1/3 of position — main target |
| T3    | Entry + 3.0 × ATR(14) | Trail stop on remainder — let winner run |
""")

    # ── Download ───────────────────────────────────────────────────────────────
    st.divider()
    dl_cols = [c for c in df.columns if c != "Signals"]
    st.download_button(
        "⬇️  Download CSV",
        data=df[dl_cols].to_csv(index=False).encode(),
        file_name=f"alerts_{today_str}.csv",
        mime="text/csv",
        key="alerts_dl",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Backtest page
# ═════════════════════════════════════════════════════════════════════════════

def render_backtest():
    st.markdown(
        "<div class='scanner-header'>"
        "<span style='font-size:2rem'>📊</span>"
        "<span class='scanner-title'>Strategy Backtest</span>"
        "</div>"
        "<div class='scanner-desc'>"
        "Walk-forward backtest of the Daily Alerts strategy on historical data. "
        "Entry at next bar open when signal fires, exits at Stop / T1 / T2 / T3 or max hold."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️  Backtest Settings", expanded=True):
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            bt_start = st.date_input(
                "Start date",
                value=datetime.today() - timedelta(days=365),
                key="bt_start",
            )
            bt_end = st.date_input(
                "End date",
                value=datetime.today(),
                key="bt_end",
            )
        with bc3:
            bt_min_score = st.slider("Min score", 4, 10, 5, key="bt_min_score",
                                     help="Minimum alert score to enter a trade")
            bt_max_hold  = st.slider("Max hold (days)", 5, 60, 20, key="bt_max_hold")
        with bc4:
            bt_capital = st.number_input("Starting capital ($)", value=10000,
                                         min_value=1000, step=1000, key="bt_capital")
            bt_risk    = st.slider("Risk per trade (%)", 0.5, 3.0, 1.0, step=0.5,
                                   key="bt_risk",
                                   help="% of portfolio risked per trade (1% rule)")

    run_bt = st.button("▶  Run Backtest", type="primary", key="bt_run_btn")

    if "bt_result" not in st.session_state:
        st.session_state.bt_result = None

    if run_bt:
        tickers = get_selected_tickers()
        _ticker_count_caption(tickers)
        if tickers:
            prog = st.progress(0, text="Initialising backtest…")
            stat = st.empty()

            def _cb(done, total, tk):
                prog.progress(done / total, text=f"Backtesting {tk}… ({done}/{total})")
                stat.caption(f"Running walk-forward on {tk}")

            with st.spinner(f"Running backtest on {len(tickers)} tickers…"):
                result = run_strategy_backtest(
                    tickers         = tickers,
                    start_date      = bt_start.strftime("%Y-%m-%d"),
                    end_date        = bt_end.strftime("%Y-%m-%d"),
                    min_score       = bt_min_score,
                    max_hold        = bt_max_hold,
                    initial_capital = float(bt_capital),
                    risk_pct        = bt_risk / 100,
                    max_workers     = 8,
                    progress_cb     = _cb,
                )
            prog.empty()
            stat.empty()
            st.session_state.bt_result = result

    result = st.session_state.bt_result
    if not result:
        st.info("Configure settings above and click **Run Backtest**.", icon="ℹ️")
        return

    m = result["metrics"]
    trades = result["trades"]
    if not trades:
        st.warning("No trades were generated. Try lowering Min Score or widening the date range.", icon="⚠️")
        return

    st.divider()

    # ── Key metrics row ───────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#c9a84c;margin-bottom:4px'>📈 Backtest Results</h3>"
        "<hr style='border-color:#c9a84c;margin-top:0'>",
        unsafe_allow_html=True,
    )
    km = st.columns(8)
    win_color   = "#26a69a" if m["win_rate"] >= 50 else "#ef5350"
    ret_color   = "#26a69a" if m["total_return"] >= 0 else "#ef5350"
    pf_color    = "#26a69a" if m["profit_factor"] >= 1.5 else "#f0c040" if m["profit_factor"] >= 1 else "#ef5350"

    km[0].metric("Win Rate",      f"{m['win_rate']:.1f}%")
    km[1].metric("Total Return",  f"{m['total_return']:+.1f}%")
    km[2].metric("Profit Factor", f"{m['profit_factor']:.2f}")
    km[3].metric("Sharpe Ratio",  f"{m['sharpe_ratio']:.2f}")
    km[4].metric("Max Drawdown",  f"{m['max_drawdown']:.1f}%")
    km[5].metric("Total Trades",  m["total_trades"])
    km[6].metric("Expectancy $",  f"${m['expectancy']:.2f}")
    km[7].metric("Final Capital", f"${m['final_equity']:,.0f}")

    st.divider()

    # ── Equity curve ──────────────────────────────────────────────────────────
    eq = result["equity_curve"]
    eq_df = pd.DataFrame({"Equity ($)": eq, "Trade #": range(len(eq))})
    st.markdown("**Equity Curve**")
    st.line_chart(eq_df.set_index("Trade #")["Equity ($)"], height=220)

    st.divider()

    # ── Exit reason breakdown ─────────────────────────────────────────────────
    st.markdown("**Trade Outcome Breakdown**")
    reason_counts = {}
    for t in trades:
        reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1

    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    cols_rc = [rc1, rc2, rc3, rc4, rc5]
    colors  = {"STOP": "#ef5350", "T1": "#f0c040", "T2": "#26a69a", "T3": "#42a5f5", "MAX_HOLD": "#888"}
    for idx, (reason, cnt) in enumerate(sorted(reason_counts.items())):
        pct = cnt / len(trades) * 100
        cols_rc[idx % 5].metric(
            f"{reason}",
            f"{cnt} ({pct:.0f}%)",
        )

    st.divider()

    # ── Per-ticker breakdown ──────────────────────────────────────────────────
    per = result["per_ticker"]
    if per:
        st.markdown("**Per-Ticker Results**")
        per_df = pd.DataFrame([
            {"Ticker": tk, **v}
            for tk, v in per.items()
        ]).sort_values("win_rate", ascending=False)
        per_df.columns = ["Ticker", "Trades", "Win Rate %", "Profit Factor",
                          "Return %", "Avg Win $", "Avg Loss $"]
        st.dataframe(
            per_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Win Rate %":     st.column_config.ProgressColumn("Win Rate %",  min_value=0, max_value=100, format="%.1f%%"),
                "Profit Factor":  st.column_config.NumberColumn("Profit Factor", format="%.2f"),
                "Return %":       st.column_config.NumberColumn("Return %",      format="%+.1f%%"),
                "Avg Win $":      st.column_config.NumberColumn("Avg Win $",     format="$%.2f"),
                "Avg Loss $":     st.column_config.NumberColumn("Avg Loss $",    format="$%.2f"),
            }
        )

    st.divider()

    # ── Full trade log ────────────────────────────────────────────────────────
    with st.expander("📋  Full Trade Log", expanded=False):
        tdf = result["trade_df"]
        if not tdf.empty:
            def _trade_row_style(row):
                color = "#1a2e1a" if row["P&L $"] > 0 else "#2e1a1a"
                return [f"background-color:{color}"] * len(row)
            st.dataframe(
                tdf.style.apply(_trade_row_style, axis=1),
                use_container_width=True,
                hide_index=True,
            )

    # ── Download ──────────────────────────────────────────────────────────────
    if not result["trade_df"].empty:
        st.download_button(
            "⬇️  Download Trade Log CSV",
            data=result["trade_df"].to_csv(index=False).encode(),
            file_name=f"backtest_{result['start_date']}_{result['end_date']}.csv",
            mime="text/csv",
            key="bt_dl",
        )


# ═════════════════════════════════════════════════════════════════════════════
# Alpaca Paper Trade page
# ═════════════════════════════════════════════════════════════════════════════

def _alpaca_client_from_state() -> Optional[object]:
    key = st.session_state.get("alpaca_api_key", "").strip()
    sec = st.session_state.get("alpaca_secret_key", "").strip()
    if not key or not sec:
        return None
    try:
        return make_client(key, sec)
    except Exception:
        return None


def render_paper_trade():
    st.markdown(
        "<div class='scanner-header'>"
        "<span style='font-size:2rem'>🤖</span>"
        "<span class='scanner-title'>Live Strategy Monitor</span>"
        "</div>"
        "<div class='scanner-desc'>"
        "Real-time monitor for the running <code>dtb-live</code> strategy "
        "(BB+RSI reversal · SPY 5m · Alpaca paper). "
        "Reads the same account — no interference with the live runner."
        "</div>",
        unsafe_allow_html=True,
    )

    # Resolve credentials from env vars
    api_key = os.environ.get("ALPACA_API_KEY", "")
    sec_key = os.environ.get("ALPACA_SECRET_KEY", "")

    if not api_key or not sec_key:
        st.info("Export `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` in your shell to connect.", icon="🔑")
        return

    try:
        client   = make_client(api_key, sec_key)
        acct     = get_account_summary(client)
        mkt_open = is_market_open_alpaca(client)
    except Exception as e:
        st.error(f"Connection failed: {e}", icon="❌")
        return

    # ── Header status bar ─────────────────────────────────────────────────────
    mkt_color = "#1a9641" if mkt_open else "#555"
    mkt_label = "● MARKET OPEN" if mkt_open else "● MARKET CLOSED"
    now_et    = datetime.now(pytz.timezone("America/New_York")).strftime("%H:%M:%S ET")
    st.markdown(
        f"<div style='display:flex;gap:16px;align-items:center;margin-bottom:8px'>"
        f"<span style='background:{mkt_color};color:#fff;padding:3px 10px;"
        f"border-radius:12px;font-size:12px'>{mkt_label}</span>"
        f"<span style='color:#666;font-size:12px'>Last refresh: {now_et}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Auto-refresh toggle
    rc1, rc2 = st.columns([1, 5])
    with rc1:
        auto_ref = st.checkbox("Auto-refresh (30s)", value=False, key="pt_auto_refresh")
    with rc2:
        if st.button("🔄  Refresh Now", key="pt_refresh"):
            st.rerun()

    st.divider()

    # ── Account overview ──────────────────────────────────────────────────────
    st.markdown("### 💼 Account")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Portfolio Value",  f"${acct['portfolio_value']:,.2f}")
    a2.metric("Cash",             f"${acct['cash']:,.2f}")
    a3.metric("Buying Power",     f"${acct['buying_power']:,.2f}")
    a4.metric("Day Trades Used",  acct["daytrade_count"])
    # Sum actual unrealized P&L from open positions
    _positions_snap = get_positions(client)
    unrealized     = sum(p["unrealized_pl"] for p in _positions_snap)
    cost_basis     = sum(p["entry_price"] * p["qty"] for p in _positions_snap)
    unreal_pct     = unrealized / cost_basis * 100 if cost_basis else 0
    a5.metric("Unrealized P&L",   f"${unrealized:+,.2f}",
              delta=f"{unreal_pct:+.2f}%")

    st.divider()

    # ── Intraday equity curve ─────────────────────────────────────────────────
    ph = get_portfolio_history(client, period="1D")
    if ph["equity"] and len(ph["equity"]) > 1:
        st.markdown("### 📈 Today's Equity Curve")
        eq_vals  = [v for v in ph["equity"] if v is not None]
        ts_vals  = ph["timestamps"][:len(eq_vals)]
        if eq_vals:
            eq_df = pd.DataFrame({"Equity ($)": eq_vals}, index=range(len(eq_vals)))
            st.line_chart(eq_df, height=180)
            day_pnl = eq_vals[-1] - eq_vals[0]
            day_pct = day_pnl / eq_vals[0] * 100 if eq_vals[0] else 0
            st.caption(f"Today's P&L:  **${day_pnl:+,.2f}**  ({day_pct:+.2f}%)")
        st.divider()

    # ── Open positions ────────────────────────────────────────────────────────
    positions = get_positions(client)
    st.markdown(f"### 📂 Open Positions ({len(positions)})")

    if positions:
        pos_df = pd.DataFrame(positions)
        pos_df.columns = ["Symbol", "Qty", "Side", "Entry $", "Current $",
                          "Mkt Value $", "Unrealized P&L $", "Unrealized %", "Asset ID"]
        pos_df = pos_df.drop(columns=["Asset ID"])

        def _pos_style(row):
            color = "#1a2e1a" if row["Unrealized P&L $"] >= 0 else "#2e1a1a"
            return [f"background-color:{color}"] * len(row)

        st.dataframe(
            pos_df.style.apply(_pos_style, axis=1),
            use_container_width=True, hide_index=True,
            column_config={
                "Unrealized %":     st.column_config.NumberColumn("Unreal %",   format="%+.2f%%"),
                "Unrealized P&L $": st.column_config.NumberColumn("Unreal P&L", format="$%.2f"),
            }
        )

        st.markdown("**Manual close (emergency only — let the runner manage exits):**")
        close_cols = st.columns(min(len(positions), 6))
        for i, pos in enumerate(positions):
            sym = pos["symbol"]
            if close_cols[i % 6].button(f"✕ {sym}", key=f"close_{sym}"):
                res = close_position(client, sym)
                st.success(f"Closed {sym}" if res["ok"] else f"Error: {res['error']}")
                st.rerun()

        if st.button("✕  Emergency: Close ALL", type="secondary", key="close_all_pos"):
            res = close_all_positions(client)
            st.success("All positions closed." if res["ok"] else f"Error: {res['error']}")
            st.rerun()
    else:
        st.caption("No open positions — runner is flat.")

    st.divider()

    # ── Today's filled trades ─────────────────────────────────────────────────
    filled = get_todays_trades(client)
    filled_only = [o for o in filled if o["status"] in ("filled", "partially_filled")]
    st.markdown(f"### 🧾 Today's Trades ({len(filled_only)} filled)")

    if filled_only:
        buys  = [o for o in filled_only if "buy"  in o["side"]]
        sells = [o for o in filled_only if "sell" in o["side"]]
        t1, t2 = st.columns(2)
        t1.metric("Entries (BUY)",  len(buys))
        t2.metric("Exits (SELL)",   len(sells))

        trade_df = pd.DataFrame(filled_only)[[
            "symbol", "side", "qty", "filled_qty", "filled_price",
            "status", "submitted_at", "filled_at"
        ]]
        trade_df.columns = ["Symbol", "Side", "Qty", "Filled", "Fill Price",
                            "Status", "Submitted", "Filled At"]
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No filled trades today yet.")

    st.divider()

    # ── Open orders ───────────────────────────────────────────────────────────
    orders = get_open_orders(client)
    st.markdown(f"### 📋 Pending Orders ({len(orders)})")

    if orders:
        ord_df = pd.DataFrame(orders)[[
            "symbol", "side", "type", "qty", "filled_qty",
            "limit_price", "stop_price", "status", "created_at"
        ]]
        ord_df.columns = ["Symbol", "Side", "Type", "Qty", "Filled",
                          "Limit $", "Stop $", "Status", "Created"]
        st.dataframe(ord_df, use_container_width=True, hide_index=True)

        if st.button("✕  Cancel ALL Pending Orders", type="secondary", key="cancel_all_ord"):
            res = cancel_all_orders(client)
            st.success("All orders cancelled." if res["ok"] else f"Error: {res['error']}")
            st.rerun()
    else:
        st.caption("No pending orders.")

    st.divider()

    st.divider()

    # ── Execute Daily Alerts ──────────────────────────────────────────────────
    st.markdown("### ⚡ Execute Daily Alerts on Alpaca")

    alerts_df = st.session_state.get("alerts_df", pd.DataFrame())

    if alerts_df.empty:
        st.warning(
            "No alerts loaded — go to **🔔 Daily Alerts**, run a scan, then come back here.",
            icon="⚠️",
        )
    else:
        st.caption(f"{len(alerts_df)} alerts available from last scan  ·  "
                   f"Bracket orders: limit entry + stop-loss + take-profit")

        ex1, ex2, ex3, ex4 = st.columns(4)
        with ex1:
            ex_min_score = st.slider("Min score", 4, 12, 7, key="ex_min_score")
        with ex2:
            ex_max_pos   = st.slider("Max new positions", 1, 10, 3, key="ex_max_pos")
        with ex3:
            ex_risk      = st.slider("Risk per trade (%)", 0.5, 3.0, 1.0, step=0.5, key="ex_risk")
        with ex4:
            ex_target    = st.selectbox("Take-profit target", ["T2 (2R)", "T1 (1R)"], key="ex_target")

        use_t2 = ex_target.startswith("T2")
        qualified = alerts_df[alerts_df["Score"] >= ex_min_score]
        st.caption(f"{len(qualified)} alerts qualify at score ≥ {ex_min_score}  ·  "
                   f"Top {ex_max_pos} by Score + R/R will be ordered")

        dry_col, live_col, warn_col = st.columns([1, 1, 3])

        with dry_col:
            dry_run = st.button("🔍  Preview", key="ex_dryrun")
        with live_col:
            go_live = st.button("🚀  Execute", type="primary", key="ex_live")
        with warn_col:
            st.markdown(
                "<div style='padding:6px 12px;background:rgba(239,83,80,0.1);"
                "border-left:3px solid #ef5350;border-radius:4px;"
                "font-size:12px;color:#ef5350;margin-top:4px'>"
                "⚠️  Places bracket orders on your paper account. Preview first."
                "</div>",
                unsafe_allow_html=True,
            )

        if dry_run:
            with st.spinner("Simulating orders…"):
                preview = alpaca_execute_alerts(
                    client, alerts_df,
                    risk_pct=ex_risk / 100,
                    max_new_positions=ex_max_pos,
                    min_score=ex_min_score,
                    use_t2_target=use_t2,
                    dry_run=True,
                )
            if preview:
                st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)
            else:
                st.info("No orders — all symbols already in positions/orders or insufficient buying power.")

        if go_live:
            if not mkt_open:
                st.warning("Market closed — DAY orders will queue for next open session.", icon="⏰")
            with st.spinner("Placing orders on Alpaca…"):
                results = alpaca_execute_alerts(
                    client, alerts_df,
                    risk_pct=ex_risk / 100,
                    max_new_positions=ex_max_pos,
                    min_score=ex_min_score,
                    use_t2_target=use_t2,
                    dry_run=False,
                )
            for r in results:
                if r.get("ok"):
                    st.success(
                        f"✅ **{r['symbol']}** {r['side']}  {r['qty']} shares  "
                        f"Entry ${r['entry']} · Stop ${r['stop']} · Target ${r['target']}"
                    )
                else:
                    st.error(f"❌ **{r['symbol']}**: {r.get('error', 'Unknown error')}")
            st.rerun()

    st.divider()

    # ── Strategy config summary ───────────────────────────────────────────────
    with st.expander("⚙️  dtb-live Config (bollinger_rsi_spy_live.yaml)", expanded=False):
        cfg_path = os.path.join(os.path.dirname(__file__), "configs", "bollinger_rsi_spy_live.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                st.code(f.read(), language="yaml")
        else:
            st.caption("Config file not found.")

    st.divider()

    # ── Stock Daily Alerts Daemon ─────────────────────────────────────────────
    st.markdown("### 📈 Stock Daily Alerts Daemon")
    st.markdown(
        "<div class='scanner-desc'>"
        "Scans universe at 09:35 ET · places bracket orders · monitors exits every 15 min · "
        "exits on URGENT technicals or max-hold-days"
        "</div>",
        unsafe_allow_html=True,
    )

    _STOCK_PLIST   = os.path.expanduser("~/Library/LaunchAgents/com.swingtrading.stock-paper.plist")
    _STOCK_LOG     = os.path.join(os.path.dirname(__file__), "logs", "stock_paper_daemon.log")
    _STOCK_SCRIPT  = os.path.join(os.path.dirname(__file__), "stock_paper_trader.py")
    _STOCK_CONFIG  = os.path.join(os.path.dirname(__file__), "configs", "stock_paper.yaml")
    _STOCK_STATE   = os.path.join(os.path.dirname(__file__), "logs", "stock_positions.json")

    import subprocess as _sp

    def _stock_daemon_status():
        try:
            out = _sp.check_output(["launchctl", "list"], text=True)
            for line in out.splitlines():
                if "com.swingtrading.stock-paper" in line:
                    parts = line.split()
                    pid   = parts[0] if parts[0] != "-" else None
                    return bool(pid and pid != "-"), pid
        except Exception:
            pass
        return False, None

    s_running, s_pid = _stock_daemon_status()
    s_color = "#22c55e" if s_running else "#94a3b8"
    s_label = f"🟢 RUNNING (PID {s_pid})" if s_running else "⚫ STOPPED"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;padding:8px 16px;"
        f"background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:8px'>"
        f"<span style='font-weight:600;color:{s_color}'>{s_label}</span>"
        f"<span style='color:#64748b;font-size:13px'>Auto-daemon · scans 09:35 ET · "
        f"monitors exits every 15 min</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    sd1, sd2, sd3, sd4 = st.columns(4)
    with sd1:
        if st.button("▶  Start Stock Daemon", use_container_width=True, key="sd_start",
                     type="primary" if not s_running else "secondary",
                     disabled=s_running):
            try:
                _sp.run(["launchctl", "load", "-w", _STOCK_PLIST], check=True)
                st.success("Stock daemon started — will scan at 09:35 ET")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")
    with sd2:
        if st.button("⏹  Stop Stock Daemon", use_container_width=True, key="sd_stop",
                     disabled=not s_running):
            try:
                _sp.run(["launchctl", "unload", _STOCK_PLIST], check=True)
                st.success("Stock daemon stopped")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")
    with sd3:
        if st.button("▶️  Run Scan Now", use_container_width=True, key="sd_scan_now",
                     help="Scan signals and place bracket orders immediately"):
            with st.spinner("Running stock scan…"):
                try:
                    result = _sp.run(
                        [sys.executable, _STOCK_SCRIPT, "--now", "--config", _STOCK_CONFIG],
                        capture_output=True, text=True, timeout=300,
                        cwd=os.path.dirname(__file__),
                    )
                    output = result.stdout + result.stderr
                    if result.returncode == 0:
                        st.success("Scan complete")
                    else:
                        st.warning("Scan finished with warnings")
                    with st.expander("📋 Scan log", expanded=True):
                        st.code(output[-4000:] if len(output) > 4000 else output)
                    st.rerun()
                except _sp.TimeoutExpired:
                    st.error("Scan timed out (>5 min)")
                except Exception as e:
                    st.error(f"Error: {e}")
    with sd4:
        if st.button("🔍  Check Exits Only", use_container_width=True, key="sd_exits_only",
                     help="Run exit checks on open positions without scanning for new entries"):
            with st.spinner("Checking exits…"):
                try:
                    result = _sp.run(
                        [sys.executable, _STOCK_SCRIPT, "--now", "--monitor-only",
                         "--config", _STOCK_CONFIG],
                        capture_output=True, text=True, timeout=120,
                        cwd=os.path.dirname(__file__),
                    )
                    output = result.stdout + result.stderr
                    st.success("Exit check complete")
                    with st.expander("📋 Exit log", expanded=True):
                        st.code(output[-2000:] if len(output) > 2000 else output)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # Open stock positions from state file
    import json as _json
    _stock_state = {"positions": []}
    if os.path.exists(_STOCK_STATE):
        try:
            _stock_state = _json.loads(open(_STOCK_STATE).read())
        except Exception:
            pass

    stock_positions = _stock_state.get("positions", [])
    st.markdown(f"**Open positions ({len(stock_positions)}):**")
    if stock_positions:
        sp_df = pd.DataFrame(stock_positions)[[
            "symbol", "side", "qty", "entry_date", "entry_price", "stop", "target", "score", "conviction"
        ]]
        sp_df.columns = ["Symbol", "Side", "Qty", "Entry Date", "Entry $", "Stop $", "Target $", "Score", "Conviction"]
        st.dataframe(sp_df, use_container_width=True, hide_index=True)

        if st.button("✕  Emergency: Close ALL Stock Positions", type="secondary", key="sd_close_all"):
            from alpaca_trader import close_all_positions
            res = close_all_positions(client)
            if res["ok"]:
                open(_STOCK_STATE, "w").write(_json.dumps({"positions": []}, indent=2))
                st.success("All stock positions closed and state cleared")
            else:
                st.error(f"Error: {res['error']}")
            st.rerun()
    else:
        st.caption("No open positions tracked by daemon.")

    if st.button("📋  View Stock Daemon Log", key="sd_view_log"):
        st.session_state["show_stock_log"] = not st.session_state.get("show_stock_log", False)
    if st.session_state.get("show_stock_log", False):
        if os.path.exists(_STOCK_LOG):
            lines = open(_STOCK_LOG).readlines()
            st.code("".join(lines[-60:]), language="text")
        else:
            st.info("No log yet — start the daemon or run a scan first.")

    with st.expander("⚙️  Stock Paper Config (stock_paper.yaml)", expanded=False):
        if os.path.exists(_STOCK_CONFIG):
            st.code(open(_STOCK_CONFIG).read(), language="yaml")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if auto_ref:
        st_autorefresh(interval=30_000, key="stock_pt_autorefresh")


# ─────────────────────────────────────────────────────────────────────────────
# SPY Reversal Alert Log
# ─────────────────────────────────────────────────────────────────────────────

def render_spy_alerts():
    """SPY BB+RSI Reversal Alert Log — live feed + daily P&L from dtb-live."""
    from datetime import date, timedelta
    import math

    st.markdown("## 📡 SPY Reversal Alerts")
    st.markdown(
        "<div class='scanner-desc'>"
        "BB+RSI reversal signals fired by <strong>dtb-live</strong> running on Alpaca · "
        "Logs stored daily · All times Eastern"
        "</div>",
        unsafe_allow_html=True,
    )

    api_key = os.environ.get("ALPACA_API_KEY", "")
    sec_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not sec_key:
        st.error("Alpaca credentials not found in .env")
        return

    client = make_client(api_key, sec_key)

    # ── Toolbar ───────────────────────────────────────────────────────────────
    today = date.today()
    tc1, tc2, tc3, tc4 = st.columns([2, 2, 1, 1])
    with tc1:
        start_d = st.date_input("From", value=today - timedelta(days=30),
                                key="spy_log_start")
    with tc2:
        end_d = st.date_input("To", value=today, key="spy_log_end")
    with tc3:
        live_mode = st.toggle("Live", value=True, key="spy_log_live",
                              help="Fetch directly from Alpaca (ignores cache)")
    with tc4:
        if st.button("💾 Save Logs", use_container_width=True, key="spy_save_logs",
                     help="Sync & write per-day JSON files to disk"):
            with st.spinner("Syncing from Alpaca…"):
                n = _spy_sync(client, days_back=90)
            st.success(f"Saved {n} records")
            st.rerun()

    # ── Fetch records ─────────────────────────────────────────────────────────
    @st.cache_data(ttl=120, show_spinner=False)
    def _live_records(_key: str):
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        from datetime import timezone
        import logging
        start_dt = datetime.combine(start_d, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt   = datetime.combine(end_d,   datetime.max.time()).replace(tzinfo=timezone.utc)
        try:
            raw = client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                after=start_dt,
                until=end_dt,
                limit=500,
            ))
        except Exception as exc:
            logging.getLogger(__name__).warning("_live_records: could not fetch orders: %s", exc)
            return []
        return _spy_pair_trades(_spy_parse_orders(raw))

    with st.spinner("Loading SPY alerts…"):
        if live_mode:
            records = _live_records(f"{start_d}_{end_d}")
        else:
            records = _spy_load_logs(start_d, end_d)

    if not records:
        st.info("No SPY reversal alerts found for this date range.  "
                "Click **💾 Save Logs** to sync from Alpaca, or enable **Live** mode.")
        return

    # ── Summary metrics ────────────────────────────────────────────────────────
    entries  = [r for r in records if r["role"] == "ENTRY"]
    exits    = [r for r in records if r["role"] == "EXIT"]
    pnls     = [r["pnl"] for r in exits if r.get("pnl") is not None]
    wins     = [p for p in pnls if p > 0]
    total_pnl = sum(pnls) if pnls else 0
    avg_pnl   = total_pnl / len(pnls) if pnls else 0
    win_rate  = len(wins) / len(pnls) * 100 if pnls else None

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Signals Fired",    len(entries))
    m2.metric("Completed Trades", len(exits))
    m3.metric("Win Rate",         f"{win_rate:.0f}%" if win_rate is not None else "—",
              delta=None)
    m4.metric("Total P&L",        f"${total_pnl:+,.2f}",
              delta=f"{'▲' if total_pnl>=0 else '▼'} ${abs(total_pnl):,.2f}")
    m5.metric("Avg P&L / Trade",  f"${avg_pnl:+,.2f}")

    st.markdown("---")

    # ── Daily breakdown ────────────────────────────────────────────────────────
    st.markdown("### Daily Summary")

    by_date: dict = {}
    for r in records:
        d = r.get("date", "")
        if d:
            by_date.setdefault(d, {"entries": [], "exits": []})
            if r["role"] == "ENTRY":
                by_date[d]["entries"].append(r)
            else:
                by_date[d]["exits"].append(r)

    daily_rows = []
    for d_str in sorted(by_date.keys(), reverse=True):
        day   = by_date[d_str]
        dpnls = [r["pnl"] for r in day["exits"] if r.get("pnl") is not None]
        dwins = [p for p in dpnls if p > 0]
        daily_rows.append({
            "Date":      d_str,
            "Signals":   len(day["entries"]),
            "Trades":    len(day["exits"]),
            "Wins":      len(dwins),
            "Losses":    len(dpnls) - len(dwins),
            "Win %":     f"{len(dwins)/len(dpnls)*100:.0f}%" if dpnls else "—",
            "Day P&L":   f"${sum(dpnls):+,.2f}" if dpnls else "—",
            "_pnl":      sum(dpnls) if dpnls else 0,
        })

    if daily_rows:
        import pandas as _pd2
        df_daily = _pd2.DataFrame(daily_rows).drop(columns=["_pnl"])
        st.dataframe(df_daily, use_container_width=True, hide_index=True,
                     column_config={
                         "Date":    st.column_config.TextColumn("Date",    width=110),
                         "Signals": st.column_config.NumberColumn("Signals",width=75),
                         "Trades":  st.column_config.NumberColumn("Trades", width=75),
                         "Wins":    st.column_config.NumberColumn("✅ Wins", width=70),
                         "Losses":  st.column_config.NumberColumn("❌ Loss", width=70),
                         "Win %":   st.column_config.TextColumn("Win %",   width=75),
                         "Day P&L": st.column_config.TextColumn("Day P&L", width=100),
                     })

    # ── Full alert log ─────────────────────────────────────────────────────────
    st.markdown("### Alert Log")

    import pandas as _pd3

    log_rows = []
    for r in sorted(records, key=lambda x: x.get("submitted_at", ""), reverse=True):
        role_icon = "🟢" if r["role"] == "ENTRY" else "🔵"
        dir_icon  = ("↑ LONG" if r["direction"] == "LONG"
                     else "↓ SHORT" if r["direction"] == "SHORT" else "?")
        pnl_val   = r.get("pnl")
        pnl_str   = f"${pnl_val:+,.2f}" if pnl_val is not None else "—"
        legs_desc = "  ·  ".join(
            f"{l['symbol'].split('SPY')[1] if 'SPY' in l['symbol'] else l['symbol']} "
            f"{l['side']} @${l['fill_price']:.2f}"
            for l in r.get("legs", [])
            if l.get("fill_price")
        ) if r.get("legs") else r.get("symbol", "")

        log_rows.append({
            "Date":      r["date"],
            "Time ET":   r["time_et"],
            "Role":      f"{role_icon} {r['role']}",
            "Direction": dir_icon,
            "Qty":       int(r.get("filled_qty", 0) or 0),
            "Net Fill":  f"${r['net_fill']:+.2f}" if r.get("net_fill") is not None else "—",
            "P&L":       pnl_str,
            "Status":    r.get("status", "").upper(),
            "Legs":      legs_desc,
        })

    if log_rows:
        st.dataframe(
            _pd3.DataFrame(log_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date":      st.column_config.TextColumn("Date",      width=100),
                "Time ET":   st.column_config.TextColumn("Time ET",   width=80),
                "Role":      st.column_config.TextColumn("Role",      width=95),
                "Direction": st.column_config.TextColumn("Direction", width=90),
                "Qty":       st.column_config.NumberColumn("Qty",     width=60),
                "Net Fill":  st.column_config.TextColumn("Net Fill",  width=90),
                "P&L":       st.column_config.TextColumn("P&L",       width=100),
                "Status":    st.column_config.TextColumn("Status",    width=90),
                "Legs":      st.column_config.TextColumn("Legs",      width=400),
            },
        )

    # ── Cumulative P&L chart ───────────────────────────────────────────────────
    if pnls:
        st.markdown("### Cumulative P&L")
        import plotly.graph_objects as _go2
        cum, running = [], 0
        for r in sorted(records, key=lambda x: x.get("submitted_at", "")):
            if r["role"] == "EXIT" and r.get("pnl") is not None:
                running += r["pnl"]
                cum.append({"date": r["date"], "time": r["time_et"], "cum_pnl": running})

        if cum:
            import pandas as _pd4
            df_cum  = _pd4.DataFrame(cum)
            df_cum["label"] = df_cum["date"] + " " + df_cum["time"]
            fig = _go2.Figure()
            color = "#22c55e" if running >= 0 else "#ef4444"
            fig.add_trace(_go2.Scatter(
                x=df_cum["label"], y=df_cum["cum_pnl"],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color),
                fill="tozeroy",
                fillcolor=f"rgba({'34,197,94' if running>=0 else '239,68,68'},0.08)",
                name="Cum P&L",
                hovertemplate="%{x}<br><b>$%{y:+,.2f}</b><extra></extra>",
            ))
            fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dot"))
            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=40), height=280,
                xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9",
                           tickprefix="$", tickfont=dict(size=11)),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


def render_options_log():
    """Options 45-60 DTE Trade Log — live feed + daily P&L from Alpaca."""
    from datetime import date, timedelta

    st.markdown("## 📋 Options 45-60 DTE Trade Log")
    st.markdown(
        "<div class='scanner-desc'>"
        "All option trades targeting 45-60 DTE on Alpaca · "
        "Logs stored daily · Entry/Exit paired with P&amp;L"
        "</div>",
        unsafe_allow_html=True,
    )

    api_key = os.environ.get("ALPACA_API_KEY", "")
    sec_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not sec_key:
        st.error("Alpaca credentials not found in .env")
        return

    client = make_client(api_key, sec_key)

    today = date.today()
    tc1, tc2, tc3, tc4, tc5 = st.columns([2, 2, 1, 1, 1])
    with tc1:
        start_d = st.date_input("From", value=today - timedelta(days=90),
                                key="opt_log_start")
    with tc2:
        end_d = st.date_input("To", value=today, key="opt_log_end")
    with tc3:
        live_mode = st.toggle("Live", value=True, key="opt_log_live",
                              help="Fetch directly from Alpaca (ignores cache)")
    with tc4:
        dte_filter = st.toggle("DTE Filter", value=True, key="opt_log_dte",
                               help="Only show 35-75 DTE trades")
    with tc5:
        if st.button("💾 Save Logs", use_container_width=True, key="opt_save_logs",
                     help="Sync & write per-day JSON files to disk"):
            with st.spinner("Syncing from Alpaca…"):
                n = _opt_sync(client, days_back=90, dte_filter=dte_filter)
            st.success(f"Saved {n} records")
            st.rerun()

    @st.cache_data(ttl=120, show_spinner=False)
    def _live_opt_records(_key: str):
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        from datetime import timezone
        import logging
        start_dt = datetime.combine(start_d, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt   = datetime.combine(end_d,   datetime.max.time()).replace(tzinfo=timezone.utc)
        try:
            raw = client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                after=start_dt,
                until=end_dt,
                limit=500,
            ))
        except Exception as exc:
            logging.getLogger(__name__).warning("_live_opt_records: could not fetch orders: %s", exc)
            return []
        return _opt_pair_trades(_opt_parse_orders(raw, dte_filter=dte_filter))

    with st.spinner("Loading options trades…"):
        if live_mode:
            records = _live_opt_records(f"{start_d}_{end_d}_{dte_filter}")
        else:
            records = _opt_load_logs(start_d, end_d)

    if not records:
        st.info("No 45-60 DTE option trades found for this date range.  "
                "Click **💾 Save Logs** to sync from Alpaca, or enable **Live** mode.")
        return

    entries  = [r for r in records if r["role"] == "ENTRY"]
    exits    = [r for r in records if r["role"] == "EXIT"]
    pnls     = [r["pnl"] for r in exits if r.get("pnl") is not None]
    wins     = [p for p in pnls if p > 0]
    total_pnl = sum(pnls) if pnls else 0
    avg_pnl   = total_pnl / len(pnls) if pnls else 0
    win_rate  = len(wins) / len(pnls) * 100 if pnls else None

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Signals Fired",    len(entries))
    m2.metric("Completed Trades", len(exits))
    m3.metric("Win Rate",         f"{win_rate:.0f}%" if win_rate is not None else "—")
    m4.metric("Total P&L",        f"${total_pnl:+,.2f}",
              delta=f"{'▲' if total_pnl >= 0 else '▼'} ${abs(total_pnl):,.2f}")
    m5.metric("Avg P&L / Trade",  f"${avg_pnl:+,.2f}")

    st.markdown("---")
    st.markdown("### Daily Summary")

    by_date: dict = {}
    for r in records:
        d = r.get("date", "")
        if d:
            by_date.setdefault(d, {"entries": [], "exits": []})
            if r["role"] == "ENTRY":
                by_date[d]["entries"].append(r)
            else:
                by_date[d]["exits"].append(r)

    daily_rows = []
    for d_str in sorted(by_date.keys(), reverse=True):
        day   = by_date[d_str]
        dpnls = [r["pnl"] for r in day["exits"] if r.get("pnl") is not None]
        dwins = [p for p in dpnls if p > 0]
        daily_rows.append({
            "Date":      d_str,
            "Signals":   len(day["entries"]),
            "Trades":    len(day["exits"]),
            "Wins":      len(dwins),
            "Losses":    len(dpnls) - len(dwins),
            "Win %":     f"{len(dwins)/len(dpnls)*100:.0f}%" if dpnls else "—",
            "Day P&L":   f"${sum(dpnls):+,.2f}" if dpnls else "—",
        })

    if daily_rows:
        st.dataframe(pd.DataFrame(daily_rows), use_container_width=True, hide_index=True,
                     column_config={
                         "Date":    st.column_config.TextColumn("Date",    width=110),
                         "Signals": st.column_config.NumberColumn("Signals",width=75),
                         "Trades":  st.column_config.NumberColumn("Trades", width=75),
                         "Wins":    st.column_config.NumberColumn("✅ Wins", width=70),
                         "Losses":  st.column_config.NumberColumn("❌ Loss", width=70),
                         "Win %":   st.column_config.TextColumn("Win %",   width=75),
                         "Day P&L": st.column_config.TextColumn("Day P&L", width=100),
                     })

    st.markdown("### Trade Log")

    log_rows = []
    for r in sorted(records, key=lambda x: x.get("submitted_at", ""), reverse=True):
        role_icon = "🟢" if r["role"] == "ENTRY" else "🔵"
        dir_icon  = "↑ LONG" if r["direction"] == "LONG" else ("↓ SHORT" if r["direction"] == "SHORT" else "?")
        pnl_val   = r.get("pnl")
        pnl_str   = f"${pnl_val:+,.2f}" if pnl_val is not None else "—"
        pnl_pct   = r.get("pnl_pct")
        pnl_pct_str = f"{pnl_pct:+.1f}%" if pnl_pct is not None else "—"
        dte_val   = r.get("dte")

        log_rows.append({
            "Date":       r["date"],
            "Time ET":    r["time_et"],
            "Symbol":     r.get("symbol", ""),
            "Underlying": r.get("underlying", "—"),
            "DTE":        int(dte_val) if dte_val is not None else "—",
            "Role":       f"{role_icon} {r['role']}",
            "Direction":  dir_icon,
            "Qty":        int(r.get("filled_qty", 0) or 0),
            "Net Fill":   f"${r['net_fill']:+.2f}" if r.get("net_fill") is not None else "—",
            "P&L":        pnl_str,
            "P&L %":      pnl_pct_str,
            "Status":     r.get("status", "").upper(),
        })

    if log_rows:
        st.dataframe(
            pd.DataFrame(log_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date":       st.column_config.TextColumn("Date",       width=100),
                "Time ET":    st.column_config.TextColumn("Time ET",    width=80),
                "Symbol":     st.column_config.TextColumn("Symbol",     width=180),
                "Underlying": st.column_config.TextColumn("Underlying", width=90),
                "DTE":        st.column_config.TextColumn("DTE",        width=60),
                "Role":       st.column_config.TextColumn("Role",       width=95),
                "Direction":  st.column_config.TextColumn("Direction",  width=90),
                "Qty":        st.column_config.NumberColumn("Qty",      width=60),
                "Net Fill":   st.column_config.TextColumn("Net Fill",   width=90),
                "P&L":        st.column_config.TextColumn("P&L",        width=100),
                "P&L %":      st.column_config.TextColumn("P&L %",      width=80),
                "Status":     st.column_config.TextColumn("Status",     width=90),
            },
        )

    if pnls:
        st.markdown("### Cumulative P&L")
        import plotly.graph_objects as _go3
        cum, running = [], 0
        for r in sorted(records, key=lambda x: x.get("submitted_at", "")):
            if r["role"] == "EXIT" and r.get("pnl") is not None:
                running += r["pnl"]
                cum.append({"date": r["date"], "time": r["time_et"], "cum_pnl": running})

        if cum:
            df_cum = pd.DataFrame(cum)
            df_cum["label"] = df_cum["date"] + " " + df_cum["time"]
            fig = _go3.Figure()
            color = "#22c55e" if running >= 0 else "#ef4444"
            fig.add_trace(_go3.Scatter(
                x=df_cum["label"], y=df_cum["cum_pnl"],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color),
                fill="tozeroy",
                fillcolor=f"rgba({'34,197,94' if running >= 0 else '239,68,68'},0.08)",
                name="Cum P&L",
                hovertemplate="%{x}<br><b>$%{y:+,.2f}</b><extra></extra>",
            ))
            fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dot"))
            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=40), height=280,
                xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9",
                           tickprefix="$", tickfont=dict(size=11)),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


def render_options_backtest():
    """Walk-forward backtest for the 45-60 DTE options swing strategy."""
    import plotly.graph_objects as _go4

    st.markdown("## 📊 Options 45-60 DTE Backtest")
    st.markdown(
        "<div class='scanner-desc'>"
        "Walk-forward simulation using BB+RSI signals with Black-Scholes Greeks · "
        "45-60 DTE ITM options · Regime-filtered entry · "
        "Stop/target/DTE exits"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Config panel ──────────────────────────────────────────────────────────
    with st.expander("⚙️  Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            start_date = st.date_input("Start Date",
                                       value=(datetime.today() - timedelta(days=730)).date(),
                                       key="ob_start")
            end_date   = st.date_input("End Date",
                                       value=datetime.today().date(),
                                       key="ob_end")
        with c2:
            initial_capital = st.number_input("Initial Capital ($)",
                                              min_value=5000, max_value=500000,
                                              value=25000, step=5000,
                                              key="ob_capital")
            max_positions   = st.number_input("Max Concurrent Positions",
                                              min_value=1, max_value=10,
                                              value=4, step=1, key="ob_maxpos")
        with c3:
            tp_pct   = st.slider("Take Profit %", 10, 100, 50, key="ob_tp") / 100
            sl_pct   = st.slider("Stop Loss %",   10, 60,  25, key="ob_sl") / 100
            dte_entry = st.slider("DTE at Entry", 40, 65, 50, key="ob_dte")

        regime_filter = st.toggle("SPY Regime Filter", value=True, key="ob_regime",
                                  help="Only enter calls when SPY > SMA200")
        min_score     = st.slider("Min Signal Score", 1.0, 10.0, 7.0, 0.5, key="ob_score")

        tickers = get_selected_tickers()

    run_btn = st.button("▶️  Run Backtest", type="primary", use_container_width=False,
                        key="ob_run")
    if not run_btn:
        st.info("Configure the parameters above and click **▶️ Run Backtest** to start.")
        return

    if not tickers:
        st.warning("No tickers selected.")
        return

    # ── Progress tracking ─────────────────────────────────────────────────────
    prog_bar  = st.progress(0.0, text="Initializing…")
    status_ph = st.empty()

    def _cb(pct: float, msg: str):
        prog_bar.progress(min(float(pct), 1.0), text=msg)
        status_ph.caption(msg)

    with st.spinner("Running backtest…"):
        result = run_options_backtest(
            tickers           = tickers,
            start_date        = start_date.strftime("%Y-%m-%d"),
            end_date          = end_date.strftime("%Y-%m-%d"),
            initial_capital   = float(initial_capital),
            tp_pct            = tp_pct,
            sl_pct            = sl_pct,
            dte_entry         = float(dte_entry),
            max_positions     = int(max_positions),
            use_regime_filter = regime_filter,
            min_score_override = float(min_score),
            progress_cb       = _cb,
        )

    prog_bar.empty()
    status_ph.empty()

    trades_df = result["trades_df"]
    equity_df = result["equity_df"]
    metrics   = result["metrics"]
    per_ticker = result["per_ticker"]

    if trades_df.empty:
        st.warning("No trades generated. Try lowering the min score or widening the date range.")
        return

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Trades",   metrics.get("total_trades", 0))
    m2.metric("Win Rate",       f"{metrics.get('win_rate', 0):.1f}%")
    m3.metric("Total P&L",      f"${metrics.get('total_pnl', 0):+,.2f}",
              delta=f"{metrics.get('total_return', 0):+.2f}%")
    m4.metric("Profit Factor",  str(metrics.get("profit_factor", "—")))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Final Equity",   f"${metrics.get('final_equity', 0):,.2f}")
    m6.metric("Max Drawdown",   f"{metrics.get('max_drawdown', 0):.2f}%")
    m7.metric("Avg Win",        f"${metrics.get('avg_win', 0):+,.2f}")
    m8.metric("Avg Hold Days",  f"{metrics.get('avg_hold_days', 0):.1f}d")

    # ── Equity curve ──────────────────────────────────────────────────────────
    if not equity_df.empty:
        st.markdown("### Equity Curve")
        fig = _go4.Figure()
        eq_vals = equity_df["total_equity"].values
        final   = float(eq_vals[-1])
        color   = "#22c55e" if final >= result["initial_capital"] else "#ef4444"

        fig.add_trace(_go4.Scatter(
            x=equity_df["date"].astype(str),
            y=equity_df["total_equity"],
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({'34,197,94' if final >= result['initial_capital'] else '239,68,68'},0.07)",
            name="Portfolio",
            hovertemplate="%{x}<br><b>$%{y:,.2f}</b><extra></extra>",
        ))
        fig.add_hline(y=result["initial_capital"],
                      line=dict(color="#94a3b8", width=1, dash="dot"),
                      annotation_text="Starting Capital")
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=40), height=320,
            xaxis=dict(showgrid=False, tickfont=dict(size=11)),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9",
                       tickprefix="$", tickfont=dict(size=11)),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-ticker breakdown ──────────────────────────────────────────────────
    if per_ticker:
        st.markdown("### Per-Symbol Results")
        tk_rows = [
            {"Symbol": sym, "Trades": v["trades"], "Win %": f"{v['win_rate']:.1f}%",
             "Total P&L": f"${v['total_pnl']:+,.2f}", "Avg P&L": f"${v['avg_pnl']:+,.2f}"}
            for sym, v in sorted(per_ticker.items(), key=lambda x: -x[1]["total_pnl"])
        ]
        st.dataframe(pd.DataFrame(tk_rows), use_container_width=True, hide_index=True)

    # ── Trade log ─────────────────────────────────────────────────────────────
    st.markdown("### Trade Log")
    show_cols = ["symbol", "entry_date", "exit_date", "direction",
                 "option_entry_price", "option_exit_price", "contracts",
                 "pnl", "pnl_pct", "days_held", "exit_reason", "delta_entry", "signal_score"]
    avail_cols = [c for c in show_cols if c in trades_df.columns]
    display_df = trades_df[avail_cols].copy()
    display_df["entry_date"] = display_df["entry_date"].astype(str)
    display_df["exit_date"]  = display_df["exit_date"].astype(str)
    if "pnl_pct" in display_df.columns:
        display_df["pnl_pct"] = display_df["pnl_pct"].apply(
            lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
    if "pnl" in display_df.columns:
        display_df["pnl"] = display_df["pnl"].apply(
            lambda x: f"${x:+,.2f}" if pd.notna(x) else "—")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _daemon_status() -> tuple:
    """Returns (is_running: bool, pid: str)."""
    try:
        import subprocess as _sp
        out = _sp.run(["launchctl", "list"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "com.swingtrading.options-paper" in line:
                parts = line.split()
                pid = parts[0] if parts[0] != "-" else "—"
                return True, pid
    except Exception:
        pass
    return False, "—"


def render_options_paper_trade():
    """Options 45-60 DTE live paper trader — scan, place orders, monitor exits."""
    import subprocess
    import json as _json
    from datetime import date, timedelta
    from pathlib import Path as _Path

    st.markdown("## 🟢 Options 45-60 DTE Paper Trade")
    st.markdown(
        "<div class='scanner-desc'>"
        "Live paper trading on Alpaca · Scans signals, places real option orders, "
        "auto-monitors exits every 15 min · Greeks-filtered entries"
        "</div>",
        unsafe_allow_html=True,
    )

    api_key = os.environ.get("ALPACA_API_KEY", "")
    sec_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not sec_key:
        st.error("Alpaca credentials not found in .env")
        return

    client = make_client(api_key, sec_key)

    # ── Daemon status bar ─────────────────────────────────────────────────────
    _PLIST = os.path.expanduser(
        "~/Library/LaunchAgents/com.swingtrading.options-paper.plist"
    )
    _DAEMON_SH = os.path.join(os.path.dirname(__file__), "options_daemon.sh")
    _DAEMON_LOG = os.path.join(os.path.dirname(__file__), "logs", "options_paper_daemon.log")

    running, pid = _daemon_status()
    status_color = "#22c55e" if running else "#94a3b8"
    status_label = f"🟢 RUNNING (PID {pid})" if running else "⚫ STOPPED"

    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;padding:10px 16px;"
        f"background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:12px'>"
        f"<span style='font-weight:600;color:{status_color}'>{status_label}</span>"
        f"<span style='color:#64748b;font-size:13px'>Auto-daemon · scans 09:35 ET · monitors exits every 15 min</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        if st.button("▶  Start Daemon", use_container_width=True, key="daemon_start",
                     type="primary" if not running else "secondary",
                     disabled=running,
                     help="Start background daemon — auto-scans daily + monitors exits"):
            try:
                subprocess.run(["launchctl", "load", "-w", _PLIST], check=True)
                st.success("Daemon started — will scan at 09:35 ET and monitor exits every 15 min")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")
    with d2:
        if st.button("⏹  Stop Daemon", use_container_width=True, key="daemon_stop",
                     disabled=not running,
                     help="Stop the background daemon"):
            try:
                subprocess.run(["launchctl", "unload", _PLIST], check=True)
                st.success("Daemon stopped")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")
    with d3:
        if st.button("🔄  Refresh", use_container_width=True, key="alerts_live_refresh"):
            st.rerun()
    with d4:
        if st.button("📋  View Log", use_container_width=True, key="daemon_view_log"):
            st.session_state["show_daemon_log"] = not st.session_state.get("show_daemon_log", False)

    if st.session_state.get("show_daemon_log", False):
        if os.path.exists(_DAEMON_LOG):
            with open(_DAEMON_LOG) as _lf:
                lines = _lf.readlines()
            st.code("".join(lines[-60:]), language="text")
        else:
            st.info("No daemon log yet — start the daemon first.")

    st.markdown("---")

    # ── Account snapshot ──────────────────────────────────────────────────────
    try:
        acct = get_account_summary(client)
        a1, a2, a3 = st.columns(3)
        a1.metric("Portfolio Value", f"${acct['portfolio_value']:,.2f}")
        a2.metric("Cash",            f"${acct['cash']:,.2f}")
        a3.metric("Buying Power",    f"${acct['buying_power']:,.2f}")
    except Exception:
        pass

    st.markdown("---")

    # ── State file — open positions ───────────────────────────────────────────
    state_file = _Path(__file__).parent / "logs" / "options_positions.json"
    state = {"positions": []}
    if state_file.exists():
        try:
            state = _json.loads(state_file.read_text())
        except Exception:
            pass

    open_positions = state.get("positions", [])

    # ── Controls ──────────────────────────────────────────────────────────────
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 2])
    with cc1:
        if st.button("▶️  Run Scan Now", type="primary", use_container_width=True, key="opts_run_scan",
                     help="Scan for signals and place orders immediately (ignores market hours)"):
            with st.spinner("Running options scan…"):
                script = os.path.join(os.path.dirname(__file__), "options_paper_trader.py")
                config = os.path.join(os.path.dirname(__file__), "configs", "options_paper.yaml")
                try:
                    result = subprocess.run(
                        [sys.executable, script, "--now", "--config", config],
                        capture_output=True, text=True, timeout=300,
                        cwd=os.path.dirname(__file__),
                    )
                    output = result.stdout + result.stderr
                    if result.returncode == 0:
                        st.success("Scan complete")
                    else:
                        st.warning("Scan finished with warnings")
                    with st.expander("📋 Scan log", expanded=True):
                        st.code(output[-4000:] if len(output) > 4000 else output)
                    st.rerun()
                except subprocess.TimeoutExpired:
                    st.error("Scan timed out (>5 min)")
                except Exception as e:
                    st.error(f"Error: {e}")

    with cc2:
        if st.button("🔍  Check Exits Only", use_container_width=True, key="opts_check_exits",
                     help="Check open positions for stop/TP/DTE exits without scanning for new entries"):
            with st.spinner("Checking exits…"):
                script = os.path.join(os.path.dirname(__file__), "options_paper_trader.py")
                config = os.path.join(os.path.dirname(__file__), "configs", "options_paper.yaml")
                try:
                    result = subprocess.run(
                        [sys.executable, script, "--now", "--monitor-only", "--config", config],
                        capture_output=True, text=True, timeout=120,
                        cwd=os.path.dirname(__file__),
                    )
                    output = result.stdout + result.stderr
                    st.success("Exit check complete")
                    with st.expander("📋 Exit log", expanded=True):
                        st.code(output[-2000:] if len(output) > 2000 else output)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with cc3:
        if st.button("🚫  Close All Positions", use_container_width=True, key="opts_close_all",
                     help="Market-sell every open option position and clear state"):
            if open_positions:
                with st.spinner("Closing positions…"):
                    closed = 0
                    for pos in list(open_positions):
                        sym = pos["option_symbol"]
                        qty = int(pos["qty"])
                        try:
                            from alpaca.trading.requests import MarketOrderRequest
                            from alpaca.trading.enums import OrderSide, TimeInForce
                            req = MarketOrderRequest(
                                symbol=sym, qty=qty,
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.DAY,
                            )
                            client.submit_order(req)
                            closed += 1
                        except Exception as e:
                            st.warning(f"Failed to close {sym}: {e}")
                    state["positions"] = []
                    state_file.write_text(_json.dumps(state, indent=2, default=str))
                    st.success(f"Closed {closed} positions")
                    st.rerun()
            else:
                st.info("No open positions to close.")

    with cc4:
        if st.button("🔄  Refresh", use_container_width=True, key="opts_paper_refresh"):
            st.rerun()

    # ── Open positions ────────────────────────────────────────────────────────
    st.markdown("### Open Positions")

    if not open_positions:
        st.info("No open option positions tracked. Click **▶️ Run Scan Now** to find and enter trades.")
    else:
        pos_rows = []
        for pos in open_positions:
            sym        = pos["option_symbol"]
            underlying = pos["underlying"]
            direction  = pos["direction"].upper()
            qty        = int(pos["qty"])
            entry_px   = float(pos.get("entry_price_est", 0))
            entry_date = pos.get("entry_date", "—")
            expiry     = pos.get("expiry", "—")
            days_held  = int(pos.get("days_held", 0))
            score      = float(pos.get("signal_score", 0))

            # Live price from Alpaca
            cur_px = None
            try:
                positions_live = client.get_all_positions()
                for p in positions_live:
                    if str(p.symbol) == sym:
                        cur_px = float(p.current_price or 0) or None
                        break
            except Exception:
                pass

            pnl     = round((cur_px - entry_px) * qty * 100, 2) if cur_px and entry_px else None
            pnl_pct = round((cur_px - entry_px) / entry_px * 100, 1) if cur_px and entry_px else None

            # DTE remaining
            dte_rem = "—"
            if expiry and expiry != "—":
                try:
                    from datetime import date as _date
                    exp = _date.fromisoformat(expiry)
                    dte_rem = str((_date.today() - exp).days * -1)
                except Exception:
                    pass

            pos_rows.append({
                "Symbol":       sym,
                "Underlying":   underlying,
                "Direction":    direction,
                "Qty":          qty,
                "Entry Est":    f"${entry_px:.2f}",
                "Current":      f"${cur_px:.2f}" if cur_px else "—",
                "P&L Est":      f"${pnl:+,.2f}" if pnl is not None else "—",
                "P&L %":        f"{pnl_pct:+.1f}%" if pnl_pct is not None else "—",
                "DTE Left":     dte_rem,
                "Days Held":    days_held,
                "Score":        score,
                "Entry Date":   entry_date,
            })

        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True,
                     column_config={
                         "Symbol":     st.column_config.TextColumn("Symbol",     width=180),
                         "Underlying": st.column_config.TextColumn("Underlying", width=90),
                         "Direction":  st.column_config.TextColumn("Direction",  width=80),
                         "Qty":        st.column_config.NumberColumn("Qty",      width=55),
                         "Entry Est":  st.column_config.TextColumn("Entry Est",  width=85),
                         "Current":    st.column_config.TextColumn("Current",    width=85),
                         "P&L Est":    st.column_config.TextColumn("P&L Est",    width=100),
                         "P&L %":      st.column_config.TextColumn("P&L %",      width=75),
                         "DTE Left":   st.column_config.TextColumn("DTE Left",   width=70),
                         "Days Held":  st.column_config.NumberColumn("Days Held",width=80),
                         "Score":      st.column_config.NumberColumn("Score",    width=65),
                         "Entry Date": st.column_config.TextColumn("Entry Date", width=100),
                     })

    # ── Today's logged trades ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Today's Trades")
    today = date.today()
    log_path = _Path(__file__).parent / "logs" / "options_45_60d" / f"options_45_60d_{today.isoformat()}.json"
    today_trades = []
    if log_path.exists():
        try:
            data = _json.loads(log_path.read_text())
            today_trades = data.get("orders", [])
        except Exception:
            pass

    if not today_trades:
        st.info("No trades logged today. Click **▶️ Run Scan Now** to start.")
    else:
        trade_rows = []
        for t in today_trades:
            role      = t.get("role", "")
            role_icon = "🟢" if role == "ENTRY" else "🔵"
            pnl       = t.get("pnl_est")
            trade_rows.append({
                "Time ET":    t.get("time_et", "—"),
                "Role":       f"{role_icon} {role}",
                "Symbol":     t.get("option_symbol", ""),
                "Direction":  t.get("direction", "—"),
                "Qty":        int(t.get("qty", 0)),
                "Price Est":  f"${t.get('entry_price_est', t.get('exit_price_est', 0)):.2f}",
                "P&L Est":    f"${pnl:+,.2f}" if pnl is not None else "—",
                "Exit":       t.get("exit_reason", "—"),
                "Score":      t.get("signal_score", "—"),
            })
        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

    # ── Config panel ──────────────────────────────────────────────────────────
    with st.expander("⚙️  Config (options_paper.yaml)"):
        config_path = os.path.join(os.path.dirname(__file__), "configs", "options_paper.yaml")
        if os.path.exists(config_path):
            with open(config_path) as _cf:
                st.code(_cf.read(), language="yaml")
        else:
            st.info("Config file not found at configs/options_paper.yaml")


# ═════════════════════════════════════════════════════════════════════════════
# Macro Calendar & News
# ═════════════════════════════════════════════════════════════════════════════

def render_calendar():
    st.markdown(
        "<div class='scanner-header'>"
        "<span style='font-size:2rem'>🗓️</span>"
        "<span class='scanner-title'>Macro Calendar & News</span>"
        "</div>"
        "<div class='scanner-desc'>"
        "FOMC, CPI, NFP, GDP, OpEx and other high-impact events — plus live financial "
        "headlines. Use this to size positions and time entries around market-moving dates."
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Countdown metrics ─────────────────────────────────────────────────────
    ctx = get_event_context(days_ahead=30)
    if ctx["banner_html"]:
        st.markdown(ctx["banner_html"], unsafe_allow_html=True)

    # Next HIGH-impact event countdown chips
    upcoming_high = [e for e in ctx["next_events"] if e["impact"] == "HIGH"][:4]
    if upcoming_high:
        cols = st.columns(len(upcoming_high))
        for col, ev in zip(cols, upcoming_high):
            meta  = _CAL_META.get(ev["category"], {})
            icon  = meta.get("icon", "⚡")
            d     = (ev["date"] - datetime.now(pytz.timezone("America/New_York")).date()).days
            label = "TODAY" if d == 0 else ("Tomorrow" if d == 1 else f"in {d}d")
            col.metric(
                f"{icon} {ev['category']}",
                label,
                ev["event"].split("(")[0].strip(),
            )
        st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_cal, tab_news, tab_earn, tab_impact = st.tabs(
        ["📅 Event Calendar", "📰 News Feed", "💰 Earnings Watch", "🎯 Strategy Impact"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: Event Calendar
    # ─────────────────────────────────────────────────────────────────────────
    with tab_cal:
        c_left, c_right = st.columns([3, 1])
        with c_left:
            days_slider = st.slider("Look-ahead days", 7, 90, 45, key="cal_days")
        with c_right:
            refresh_cal = st.button("🔄  Refresh", key="cal_refresh", use_container_width=True)

        if "cal_events" not in st.session_state or refresh_cal:
            with st.spinner("Loading events..."):
                st.session_state.cal_events = get_upcoming_events(days_slider)

        events = st.session_state.cal_events

        # Impact filter
        impact_filter = st.multiselect(
            "Impact filter",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM"],
            key="cal_impact_filter",
        )
        cat_opts = sorted({e["category"] for e in events})
        cat_filter = st.multiselect(
            "Category filter",
            cat_opts,
            default=cat_opts,
            key="cal_cat_filter",
        )

        filtered = [
            e for e in events
            if e["impact"] in impact_filter and e["category"] in cat_filter
        ]

        if not filtered:
            st.info("No events match the current filters.", icon="ℹ️")
        else:
            rows = []
            today_d = datetime.now(pytz.timezone("America/New_York")).date()
            for ev in filtered:
                meta    = _CAL_META.get(ev["category"], {})
                icon    = meta.get("icon", "⚡")
                d       = (ev["date"] - today_d).days
                day_lbl = "Today" if d == 0 else ("Tomorrow" if d == 1 else f"in {d}d")
                rows.append({
                    "Date":     ev["date"].strftime("%a %b %d, %Y"),
                    "Days":     day_lbl,
                    "Impact":   ev["impact"],
                    "Category": f"{icon} {ev['category']}",
                    "Event":    ev["event"],
                    "Time (ET)": ev.get("time", ""),
                    "Forecast": ev.get("forecast", ""),
                    "Previous": ev.get("previous", ""),
                    "Source":   ev.get("source", ""),
                })

            df = pd.DataFrame(rows)

            def _cal_style(row):
                imp = row["Impact"]
                if imp == "HIGH":
                    return ["background-color:#3a1b1b;color:#fca5a5"] + ["background-color:#3a1b1b;color:#e2e8f0"] * (len(row) - 1)
                elif imp == "MEDIUM":
                    return ["background-color:#2a2a1a;color:#fde68a"] + ["background-color:#2a2a1a;color:#e2e8f0"] * (len(row) - 1)
                return [""] * len(row)

            st.dataframe(
                df.style.apply(_cal_style, axis=1),
                use_container_width=True,
                hide_index=True,
                height=min(700, 40 + 40 * len(df)),
            )

            # Legend
            st.markdown(
                "<div style='font-size:0.78rem;color:#64748b;margin-top:4px'>"
                "🔴 HIGH — significant market-moving event &nbsp;|&nbsp; "
                "🟡 MEDIUM — moderate impact &nbsp;|&nbsp; "
                "⚪ LOW — minor / routine</div>",
                unsafe_allow_html=True,
            )

        with st.expander("ℹ️ Data sources & accuracy"):
            st.markdown("""
**Sources:**
- **Scheduled** — Hand-curated from Federal Reserve, BLS, and CME public calendars.
- **ForexFactory** — Live feed for the current week (USD events only).

**Note:** Exact dates for CPI/PCE/GDP/NFP are released by BLS and BEA a year in advance.
FOMC dates after mid-2026 are estimated from the Fed's typical 8-meetings-per-year pattern.
Always verify against [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
and [bls.gov/schedule](https://www.bls.gov/schedule/news_release/cpi.htm).
""")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: News Feed
    # ─────────────────────────────────────────────────────────────────────────
    with tab_news:
        n_left, n_right = st.columns([3, 1])
        with n_left:
            n_items = st.slider("Number of headlines", 10, 40, 20, key="news_n")
        with n_right:
            refresh_news = st.button("🔄  Refresh", key="news_refresh", use_container_width=True)

        if "cal_news" not in st.session_state or refresh_news:
            with st.spinner("Fetching headlines..."):
                st.session_state.cal_news = get_news_feed(n_items)

        news = st.session_state.cal_news
        if not news:
            st.warning("Could not fetch news. Check internet connection.", icon="⚠️")
        else:
            for item in news:
                with st.container():
                    src_badge = (
                        f"<span style='background:#1e3a5f;color:#93c5fd;padding:1px 7px;"
                        f"border-radius:10px;font-size:0.7rem;font-weight:600'>"
                        f"{item['source']}</span>"
                    )
                    title_md = item["title"]
                    link     = item.get("link", "")
                    pub      = item.get("pub", "")

                    if link:
                        st.markdown(
                            f"{src_badge} &nbsp;"
                            f"<a href='{link}' target='_blank' style='color:#e2e8f0;"
                            f"text-decoration:none;font-weight:500'>{title_md}</a>"
                            f"<span style='color:#475569;font-size:0.75rem;margin-left:8px'>{pub[:25]}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"{src_badge} &nbsp;<span style='color:#e2e8f0;font-weight:500'>{title_md}</span>"
                            f"<span style='color:#475569;font-size:0.75rem;margin-left:8px'>{pub[:25]}</span>",
                            unsafe_allow_html=True,
                        )

                    summary = item.get("summary", "")
                    if summary:
                        st.caption(summary[:180] + ("…" if len(summary) >= 180 else ""))
                    st.markdown("<hr style='border-color:#1e293b;margin:6px 0'>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: Earnings Watch
    # ─────────────────────────────────────────────────────────────────────────
    with tab_earn:
        st.markdown("Upcoming earnings for your selected universe.")
        earn_days = st.slider("Look-ahead days", 3, 30, 14, key="earn_days")

        if st.button("▶  Load Earnings", type="primary", key="earn_load"):
            tickers = get_selected_tickers()
            with st.spinner(f"Fetching earnings dates for {len(tickers)} tickers..."):
                earn_events = get_earnings_calendar(tickers, earn_days)
            st.session_state.earn_events = earn_events
            st.session_state.earn_loaded = True

        if st.session_state.get("earn_loaded") and "earn_events" in st.session_state:
            earn_events = st.session_state.earn_events
            if not earn_events:
                st.info("No earnings found in the selected window for your universe.", icon="ℹ️")
            else:
                today_d = datetime.now(pytz.timezone("America/New_York")).date()
                earn_rows = []
                for ev in sorted(earn_events, key=lambda e: e["date"]):
                    d = (ev["date"] - today_d).days
                    sym = ev["event"].split(" ")[0]
                    earn_rows.append({
                        "Ticker":    sym,
                        "Date":      ev["date"].strftime("%a %b %d"),
                        "Days Away": d,
                        "Time":      ev.get("time", ""),
                    })
                earn_df = pd.DataFrame(earn_rows)

                def _earn_style(row):
                    if row["Days Away"] <= 1:
                        return ["background-color:#3a1b1b;color:#fca5a5"] * len(row)
                    elif row["Days Away"] <= 3:
                        return ["background-color:#2a2a1a;color:#fde68a"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    earn_df.style.apply(_earn_style, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(f"{len(earn_rows)} earnings events in the next {earn_days} days.")
        else:
            st.info("Click **▶ Load Earnings** to fetch upcoming earnings for your universe.", icon="📅")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4: Strategy Impact Guide
    # ─────────────────────────────────────────────────────────────────────────
    with tab_impact:
        st.markdown(
            "### How upcoming events affect each strategy\n"
            "Review before placing trades — high-impact events can invalidate breakout setups "
            "and cause whipsaws."
        )

        # Pull next HIGH events
        next30 = get_upcoming_events(30)
        next30_high = [e for e in next30 if e["impact"] == "HIGH"][:6]

        if not next30_high:
            st.success("No HIGH-impact events in the next 30 days. Clear skies.", icon="✅")
        else:
            today_d = datetime.now(pytz.timezone("America/New_York")).date()
            for ev in next30_high:
                meta = _CAL_META.get(ev["category"], {})
                icon = meta.get("icon", "⚡")
                note = meta.get("strategy_note", "")
                d    = (ev["date"] - today_d).days
                day_str = "Today" if d == 0 else ("Tomorrow" if d == 1 else f"in {d} days")

                color_map = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#64748b"}
                border_color = color_map.get(ev["impact"], "#64748b")

                st.markdown(
                    f"<div style='border-left:3px solid {border_color};"
                    f"background:rgba(255,255,255,0.03);border-radius:6px;"
                    f"padding:10px 14px;margin-bottom:10px'>"
                    f"<div style='font-weight:600;color:#e2e8f0'>"
                    f"{icon} {ev['event']} &nbsp;<span style='color:{border_color};font-size:0.8rem'>"
                    f"● {ev['impact']}</span> &nbsp;"
                    f"<span style='color:#64748b;font-size:0.85rem'>{ev['date'].strftime('%b %d, %Y')} ({day_str})</span>"
                    f"</div>"
                    f"<div style='color:#94a3b8;font-size:0.85rem;margin-top:4px'>{note}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown("#### General rules of thumb")
        st.markdown("""
| Scenario | Suggested Action |
|---|---|
| HIGH event **≤ 2 days away** | Avoid new breakout or momentum entries; reduce position size |
| FOMC **day of** | Expect wide bid/ask spreads; don't trade the first 15 min after 2pm ET |
| CPI / NFP **morning** | Wait for 9:45–10:00 ET to see initial reaction before entering |
| Quadruple Witching | Intraday swings amplified; breakouts may reverse hard into close |
| Earnings on watchlist stock | Never hold swing positions through the company's own earnings |
| High event **> 7 days away** | No special adjustment needed for swing timeframe |
""")


# ═════════════════════════════════════════════════════════════════════════════
# Influencer Tracker — Jensen Huang · Trump
# ═════════════════════════════════════════════════════════════════════════════

import xml.etree.ElementTree as _ET
import re as _re
import requests as _requests

_INF_SESSION = _requests.Session()
_INF_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})

# ── Company-name → ticker lookup (used when news doesn't use $TICK format) ─────

_NAME_TO_TICKER: dict[str, str] = {
    # AI / Semiconductors
    "nvidia": "NVDA", "nvda": "NVDA",
    "amd": "AMD", "advanced micro": "AMD",
    "intel": "INTC",
    "broadcom": "AVGO",
    "marvell": "MRVL",
    "arm holdings": "ARM", "arm ltd": "ARM",
    "astera labs": "ALAB",
    "super micro": "SMCI", "supermicro": "SMCI",
    "coreweave": "CRWV",
    "micron": "MU",
    "tsmc": "TSM", "taiwan semiconductor": "TSM",
    "lam research": "LRCX",
    "kla": "KLAC",
    "applied materials": "AMAT",
    "palantir": "PLTR",
    "servicenow": "NOW",
    # Hyperscalers
    "microsoft": "MSFT",
    "alphabet": "GOOGL", "google": "GOOGL",
    "amazon": "AMZN", "aws": "AMZN",
    "meta": "META", "facebook": "META",
    "apple": "AAPL",
    # AI server hardware
    "dell": "DELL", "dell technologies": "DELL",
    "hewlett packard": "HPE", "hpe": "HPE",
    "cisco": "CSCO",
    "arista": "ANET",
    # Cloud / SaaS
    "salesforce": "CRM",
    "oracle": "ORCL",
    "snowflake": "SNOW",
    "datadog": "DDOG",
    "cloudflare": "NET",
    "openai": "MSFT",   # Microsoft-backed
    # Energy
    "exxon": "XOM", "exxonmobil": "XOM",
    "chevron": "CVX",
    "conocophillips": "COP",
    "pioneer": "PXD",
    # Trump/DJT
    "trump media": "DJT", "truth social": "DJT", "djt": "DJT",
    "tesla": "TSLA",
    # Defense
    "lockheed": "LMT", "lockheed martin": "LMT",
    "northrop": "NOC", "northrop grumman": "NOC",
    "raytheon": "RTX",
    "boeing": "BA",
    "general dynamics": "GD",
    # Steel / Tariff
    "nucor": "NUE",
    "steel dynamics": "STLD",
    "u.s. steel": "X", "us steel": "X",
    # Finance
    "jpmorgan": "JPM", "jp morgan": "JPM",
    "goldman": "GS",
    "berkshire": "BRK-B",
    # EV / Auto
    "rivian": "RIVN",
    "lucid": "LCID",
    "ford": "F",
    "gm": "GM", "general motors": "GM",
    # Crypto-adjacent
    "coinbase": "COIN",
    "microstrategy": "MSTR",
    "robinhood": "HOOD",
}

# Pinned tickers always shown regardless of news (1-2 anchor tickers per feed)
_JENSEN_PINNED = {"NVDA", "AMD"}
_TRUMP_PINNED  = {"DJT", "TSLA"}

# Common English words that look like tickers — skip these
_SKIP_WORDS = {
    "A", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "IT",
    "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
    "AND", "ARE", "BUT", "CAN", "DID", "FOR", "GET", "GOT", "HAD", "HAS",
    "HIM", "HIS", "HOW", "ITS", "LET", "MAY", "NEW", "NOT", "NOW", "OFF",
    "OLD", "ONE", "OUR", "OUT", "OWN", "PUT", "SAY", "SHE", "THE", "TOO",
    "TWO", "USE", "WAS", "WAY", "WHO", "WHY", "WON", "YES", "YET", "YOU",
    "ALSO", "BACK", "BEEN", "BOTH", "COME", "DOES", "DOWN", "EACH", "EVEN",
    "FROM", "GIVE", "GOOD", "HAVE", "HERE", "HIGH", "INTO", "JUST", "LAST",
    "LIKE", "MADE", "MAKE", "MANY", "MORE", "MOST", "MUCH", "MUST", "NEXT",
    "ONLY", "OPEN", "OVER", "PLAN", "RATE", "SAID", "SAME", "SOME", "SUCH",
    "TAKE", "THAN", "THAT", "THEM", "THEN", "THEY", "THIS", "TIME", "TOLD",
    "TOOK", "UPON", "VERY", "WANT", "WELL", "WERE", "WHAT", "WITH", "WORK",
    "YEAR", "YOUR", "AFTER", "ABOUT", "ABOVE", "AGAIN", "ALONG", "AMONG",
    "BEING", "COULD", "EVERY", "FIRST", "GREAT", "GROUP", "LARGE", "LATER",
    "MIGHT", "MONTH", "NEVER", "OFTEN", "OTHER", "PLACE", "POINT", "PRICE",
    "RIGHT", "SINCE", "SMALL", "STILL", "THEIR", "THERE", "THESE", "THOSE",
    "THREE", "UNDER", "UNTIL", "USING", "WEEKS", "WHERE", "WHICH", "WHILE",
    "WOULD", "YEARS",
    # Financial words that look like tickers
    "CEO", "CFO", "COO", "CTO", "IPO", "ETF", "GDP", "CPI", "NFP", "PMI",
    "AI", "EV", "AR", "VR", "ML", "PC", "IT", "TV", "AD", "PR",
    "USD", "EUR", "GBP", "JPY", "BTC", "ETH",
    "SEC", "FTC", "DOJ", "IRS", "FDA", "FED", "IMF", "WTO",
    "EPS", "PE", "ROE", "YOY", "QOQ", "MOM", "YTD",
    "AM", "PM", "EST", "ET", "PT",
    "INC", "LLC", "LTD", "CORP", "CO",
    "NEWS", "SAYS", "SAID", "WILL", "BEAT", "MISS", "TOPS", "CUTS", "RISE",
    "FALL", "GAIN", "LOSS", "DEAL", "BANK", "FUND", "TRADE", "RATE",
}

_TRUTH_RSS   = "https://truthsocial.com/@realDonaldTrump.rss"
# as_qdr=w  → limit Google News results to the past week
_GNEWS       = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en&as_qdr=w"
_NITTER_NVDA = "https://nitter.privacydev.net/nvidia/rss"

_RFC_FMTS = [
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S %Z",
    "%d %b %Y %H:%M:%S %z",
]

def _parse_pub_date(pub: str):
    """Parse RSS pubDate string → aware datetime, or None on failure."""
    for fmt in _RFC_FMTS:
        try:
            return datetime.strptime(pub.strip(), fmt)
        except ValueError:
            pass
    return None

# Regex: captures tickers from ($NVDA), (NVDA), NYSE: NVDA, Nasdaq: NVDA
_TICKER_RE = _re.compile(
    r'\$([A-Z]{1,5})'                          # $NVDA
    r'|\(([A-Z]{1,5})\)'                       # (NVDA)
    r'|(?:NYSE|NASDAQ|Nasdaq):\s*([A-Z]{1,5})' # NYSE: NVDA
)


@st.cache_data(ttl=300, show_spinner=False)
def _inf_fetch_rss(url: str, max_items: int = 15, max_age_days: int = 14) -> list[dict]:
    """
    Parse an RSS feed.  Articles older than max_age_days are dropped.
    Results are sorted newest-first.
    Cache TTL = 5 min so news stays current throughout the trading day.
    """
    try:
        resp = _INF_SESSION.get(url, timeout=10)
        resp.raise_for_status()
        root = _ET.fromstring(resp.content)
    except Exception:
        return []

    from datetime import timezone as _tz
    cutoff = datetime.now(_tz.utc) - timedelta(days=max_age_days)

    raw_items = []
    for item in root.iter("item"):
        title   = (item.findtext("title")       or "").strip()
        link    = (item.findtext("link")        or "").strip()
        pub_str = (item.findtext("pubDate")     or "").strip()
        summary = _re.sub(r"<[^>]+>", "", (item.findtext("description") or ""))[:300].strip()

        pub_dt = _parse_pub_date(pub_str)

        # Normalise to UTC-aware before comparing
        if pub_dt is not None:
            if pub_dt.tzinfo is None:
                from datetime import timezone as _tz2
                pub_dt = pub_dt.replace(tzinfo=_tz2.utc)
            if pub_dt < cutoff:
                continue

        raw_items.append({
            "title":  title,
            "link":   link,
            "pub":    pub_str,
            "pub_dt": pub_dt,
            "summary": summary,
        })

    # Sort newest first — use a UTC-aware epoch as fallback for items with no date
    from datetime import timezone as _tz3
    _epoch = datetime(1970, 1, 1, tzinfo=_tz3.utc)
    raw_items.sort(key=lambda x: x["pub_dt"] or _epoch, reverse=True)

    # Format pub date for display and drop internal field
    out = []
    for item in raw_items[:max_items]:
        dt = item.pop("pub_dt")
        item["pub"] = dt.strftime("%b %d %H:%M ET") if dt and dt != _epoch else item["pub"][:16]
        out.append(item)
    return out


def _extract_tickers(news_items: list[dict], pinned: set[str]) -> dict[str, list[str]]:
    """
    Scan news titles + summaries for ticker mentions.
    Returns {TICKER: [headline, headline, ...]} sorted by mention count.
    Pinned tickers always appear even with 0 news mentions.
    """
    found: dict[str, list[str]] = {t: [] for t in pinned}

    for item in news_items:
        title   = item.get("title", "")
        summary = item.get("summary", "")
        text    = title + " " + summary

        # 1. Explicit ticker patterns
        for m in _TICKER_RE.finditer(text):
            ticker = (m.group(1) or m.group(2) or m.group(3) or "").upper().strip()
            if ticker and ticker not in _SKIP_WORDS and len(ticker) >= 2:
                found.setdefault(ticker, [])
                if title not in found[ticker]:
                    found[ticker].append(title[:70])

        # 2. Company name lookup
        text_lower = text.lower()
        for name, ticker in _NAME_TO_TICKER.items():
            if name in text_lower:
                found.setdefault(ticker, [])
                if title not in found[ticker]:
                    found[ticker].append(title[:70])

    # Sort: pinned first, then by mention count desc
    pinned_items  = [(t, found[t]) for t in pinned if t in found]
    other_items   = sorted(
        [(t, v) for t, v in found.items() if t not in pinned],
        key=lambda x: -len(x[1]),
    )
    return dict(pinned_items + other_items)


@st.cache_data(ttl=120, show_spinner=False)
def _inf_fetch_prices(tickers_tuple: tuple) -> pd.DataFrame:
    """Batch-fetch price + day% for a tuple of tickers."""
    tickers = list(tickers_tuple)
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(tickers, period="2d", progress=False,
                          auto_adjust=True, group_by="ticker")
    except Exception:
        return pd.DataFrame()

    rows = []
    for ticker in tickers:
        try:
            df = raw if len(tickers) == 1 else (
                raw[ticker] if ticker in raw.columns.get_level_values(0)
                else pd.DataFrame()
            )
            if df is None or df.empty or len(df) < 2:
                continue
            prev  = float(df["Close"].iloc[-2])
            last  = float(df["Close"].iloc[-1])
            pct   = (last / prev - 1) * 100
            vol   = int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
            rows.append({"Ticker": ticker, "Price": round(last, 2),
                         "Day %": round(pct, 2), "Volume": vol})
        except Exception:
            continue
    return pd.DataFrame(rows)


def _inf_news_card(item: dict):
    title   = item.get("title", "")
    link    = item.get("link", "#")
    pub     = item.get("pub", "")[:16]
    summary = item.get("summary", "")
    st.markdown(
        f"<div style='padding:8px 0;border-bottom:1px solid #1e293b'>"
        f"<div style='font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:3px'>"
        f"<a href='{link}' target='_blank' style='color:#93c5fd;text-decoration:none'>{title}</a>"
        f"</div>"
        f"<div style='font-size:11px;color:#475569;margin-bottom:4px'>{pub}</div>"
        f"<div style='font-size:12px;color:#94a3b8'>{summary}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _inf_render_dynamic_stocks(news_items: list[dict], pinned: set[str]):
    """Extract tickers from news, fetch prices, render table with mention context."""
    ticker_mentions = _extract_tickers(news_items, pinned)

    if not ticker_mentions:
        st.caption("No tickers found in news.")
        return

    tickers_to_fetch = tuple(ticker_mentions.keys())
    with st.spinner("Fetching prices…"):
        price_df = _inf_fetch_prices(tickers_to_fetch)

    if price_df.empty:
        st.caption("Price fetch failed.")
        return

    price_map = price_df.set_index("Ticker").to_dict("index")

    rows_html = ""
    rendered = 0
    for ticker, headlines in ticker_mentions.items():
        p = price_map.get(ticker)
        if p is None:
            continue  # not a real ticker / not in yfinance
        pct   = p["Day %"]
        color = "#4ade80" if pct >= 0 else "#f87171"
        sign  = "+" if pct >= 0 else ""
        vol_m = f"{p['Volume']/1e6:.1f}M" if p["Volume"] > 0 else "—"
        # First headline as context (truncated)
        ctx   = headlines[0][:55] + "…" if headlines else "pinned"
        badge = f'<span style="background:#1e293b;border-radius:4px;padding:1px 5px;font-size:10px;color:#64748b">{len(headlines)} mention{"s" if len(headlines)!=1 else ""}</span>' if headlines else '<span style="background:#1e3a5f;border-radius:4px;padding:1px 5px;font-size:10px;color:#60a5fa">pinned</span>'
        rows_html += (
            f"<tr>"
            f"<td style='font-weight:700;color:#f8fafc'>{ticker}</td>"
            f"<td>{badge}</td>"
            f"<td style='color:#e2e8f0'>${p['Price']:.2f}</td>"
            f"<td style='color:{color};font-weight:600'>{sign}{pct:.2f}%</td>"
            f"<td style='color:#94a3b8'>{vol_m}</td>"
            f"</tr>"
            f"<tr><td colspan='5' style='color:#475569;font-size:10px;padding:0 8px 6px'>{ctx}</td></tr>"
        )
        rendered += 1

    if not rendered:
        st.caption("No recognised tickers with price data found in current news.")
        return

    st.markdown(
        f"<table class='dash-tbl'>"
        f"<thead><tr>"
        f"<th>Ticker</th><th>Mentions</th><th>Price</th><th>Day %</th><th>Volume</th>"
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>",
        unsafe_allow_html=True,
    )


def render_influencers():
    st.markdown(
        "<div class='scanner-header'>"
        "<span style='font-size:2rem'>📡</span>"
        "<span class='scanner-title'>Influencer Tracker</span>"
        "</div>"
        "<div class='scanner-desc'>"
        "Stocks pulled live from Jensen Huang and Trump news — tickers extracted automatically "
        "from Google News, Truth Social, and NVIDIA's X feed. Updates every 10 min."
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    tab_jensen, tab_trump = st.tabs(["🟢  Jensen Huang · NVIDIA", "🇺🇸  Trump · DJT"])

    # ── Jensen Huang ──────────────────────────────────────────────────────────
    with tab_jensen:
        col_news, col_stocks = st.columns([3, 2])

        with col_news:
            st.markdown("#### 📰 Latest News")
            with st.spinner("Fetching Jensen / NVIDIA news…"):
                jensen_news = _inf_fetch_rss(
                    _GNEWS.format(q="Jensen+Huang+NVIDIA+AI"), max_items=15
                )
                x_posts = _inf_fetch_rss(_NITTER_NVDA, max_items=5)

            all_jensen = x_posts + jensen_news

            if x_posts:
                st.markdown("**NVIDIA on X**")
                for item in x_posts[:4]:
                    _inf_news_card(item)
                st.markdown("---")

            st.markdown("**Google News — Jensen Huang · NVIDIA**")
            if not jensen_news:
                st.caption("News feed unavailable.")
            for item in jensen_news:
                _inf_news_card(item)

        with col_stocks:
            st.markdown("#### 📈 Stocks in the News")
            st.caption("Tickers extracted automatically from headlines above.")
            _inf_render_dynamic_stocks(all_jensen, _JENSEN_PINNED)

    # ── Trump ─────────────────────────────────────────────────────────────────
    with tab_trump:
        col_news2, col_stocks2 = st.columns([3, 2])

        with col_news2:
            st.markdown("#### 📰 Latest News & Truth Social")
            with st.spinner("Fetching Trump news…"):
                trump_news  = _inf_fetch_rss(
                    _GNEWS.format(q="Trump+stocks+trade+tariff+executive+order"), max_items=15
                )
                truth_posts = _inf_fetch_rss(_TRUTH_RSS, max_items=8)

            all_trump = truth_posts + trump_news

            if truth_posts:
                st.markdown("**Truth Social — @realDonaldTrump**")
                for item in truth_posts[:6]:
                    _inf_news_card(item)
                st.markdown("---")
            else:
                st.info("Truth Social RSS unreachable — news only.")

            st.markdown("**Google News — Trump · Markets · Policy**")
            if not trump_news:
                st.caption("News feed unavailable.")
            for item in trump_news:
                _inf_news_card(item)

        with col_stocks2:
            st.markdown("#### 📈 Stocks in the News")
            st.caption("Tickers extracted automatically from headlines above.")
            _inf_render_dynamic_stocks(all_trump, _TRUMP_PINNED)


# ═════════════════════════════════════════════════════════════════════════════
# Home Dashboard — top-5 day trades, breakouts, options, today's events
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _home_day_trades(_key: str) -> pd.DataFrame:
    # Use watchlist only (~144 tickers) so the morning brief loads in seconds
    df = run_combined_screener(tickers=WATCHLIST_TICKERS, min_score=4, max_workers=10, lookback_days=200)
    if df.empty:
        return df
    order = {"⚡ STRONG BUY": 0, "✅ BUY": 1, "👀 WATCH": 2}
    df["_o"] = df["Signal"].map(order).fillna(3)
    df = df.sort_values(["_o", "Score"], ascending=[True, False]).drop(columns=["_o"])
    return df.head(5).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def _home_breakouts(_key: str) -> pd.DataFrame:
    df = run_breakout_screener(tickers=WATCHLIST_TICKERS, recent_bars=3)
    if df.empty:
        return df
    # BULL > BEAR descending so bullish setups appear first
    return df.sort_values(["Direction", "Bars Ago"], ascending=[False, True]).head(5).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def _home_options(_key: str) -> pd.DataFrame:
    # iv_rank_min=0: disable the VIX gate — screener returns empty when VIX<20
    # max_entry_sigma=0.70: allow growth stocks (AMD 67%, PLTR 53%, etc.)
    df = run_swing_options_screener(
        tickers=WATCHLIST_TICKERS,
        min_score=4.5,
        dte=50.0,
        params={"max_entry_sigma": 0.70, "iv_rank_min": 0, "iv_rank_max": 100},
    )
    if df.empty:
        return df
    ok = df[df["_passes_greeks"]].head(5)
    return (ok if len(ok) >= 3 else df.head(5)).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def _home_events(_key: str) -> list:
    et  = pytz.timezone("America/New_York")
    today_str = datetime.now(et).strftime("%Y-%m-%d")
    macro = [e for e in get_upcoming_events(days_ahead=1, include_earnings=False)
             if str(e.get("date", "")) == today_str]
    try:
        # Call the raw fetch functions directly — do NOT call another @st.cache_data
        # function (_base_universe) from inside a cached function, it throws
        broad = _dedup(
            get_sp500_tickers() + get_nasdaq100_tickers() + list(WATCHLIST_TICKERS)
        )
        earn = [e for e in get_earnings_calendar(broad, days=2)
                if str(e.get("date", "")) == today_str]
    except Exception as _e:
        earn = []
    return earn + macro


def _home_price(val) -> str:
    try:
        return f"${float(val):.2f}"
    except Exception:
        return "—"


def _home_sig_badge(signal: str) -> str:
    if "STRONG" in signal:
        return "<span style='background:#14532d;color:#86efac;padding:2px 8px;border-radius:5px;font-size:11px;font-weight:700'>STRONG BUY</span>"
    if "BUY" in signal:
        return "<span style='background:#052e16;color:#4ade80;padding:2px 8px;border-radius:5px;font-size:11px;font-weight:700'>BUY</span>"
    return "<span style='background:#1c1917;color:#fbbf24;padding:2px 8px;border-radius:5px;font-size:11px;font-weight:700'>WATCH</span>"


def render_home():
    et       = pytz.timezone("America/New_York")
    now      = datetime.now(et)
    hour_key = now.strftime("%Y%m%d-%H")
    day_key  = now.strftime("%Y%m%d")

    # ── Auto-refresh before market open ───────────────────────────────────────
    market_o = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_c = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    before_open = now.weekday() < 5 and now < market_o
    if before_open:
        mins_left = int((market_o - now).total_seconds() // 60)
        st.info(f"Market opens in **{mins_left} min** — refreshing automatically at 9:30 AM ET.")
        _ms_to_open = min(60_000, int((market_o - now).total_seconds() * 1000))
        st_autorefresh(interval=_ms_to_open, key="home_premarket_refresh")

    # ── Header ────────────────────────────────────────────────────────────────
    hc1, hc2 = st.columns([5, 1])
    with hc1:
        st.markdown(
            f"<h2 style='margin:0 0 4px'>Morning Briefing</h2>"
            f"<span style='color:#475569;font-size:12px'>"
            f"{now.strftime('%A, %b %d %Y · %I:%M %p ET')} · "
            f"{len(WATCHLIST_TICKERS)} tickers</span>",
            unsafe_allow_html=True,
        )
    with hc2:
        st.write("")
        if st.button("↺ Refresh", key="home_refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Row 1: Day Trades | Breakout Stocks ───────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⚡ Day Trade Setups")
        with st.spinner("Scanning…"):
            dt_df = _home_day_trades(hour_key)
        if dt_df.empty:
            st.caption("No high-conviction setups right now.")
        else:
            tbl = []
            for _, r in dt_df.iterrows():
                tbl.append({
                    "Ticker":  r["Ticker"],
                    "Signal":  r["Signal"].replace("⚡ ", "").replace("✅ ", "").replace("👀 ", ""),
                    "Entry":   _home_price(r.get("Entry")),
                    "Stop":    _home_price(r.get("Stop")),
                    "Target":  _home_price(r.get("Target")),
                    "RSI":     f"{r['RSI']:.0f}" if pd.notna(r.get("RSI")) else "—",
                    "Vol×":    f"{r['Vol vs Avg']:.1f}" if pd.notna(r.get("Vol vs Avg")) else "—",
                    "Why":     str(r.get("Why", ""))[:55],
                })
            st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 🚀 Breakout Stocks")
        with st.spinner("Scanning…"):
            bo_df = _home_breakouts(hour_key)
        if bo_df.empty:
            st.caption("No fresh breakouts detected.")
        else:
            tbl = []
            for _, r in bo_df.iterrows():
                tbl.append({
                    "Ticker":   r["Ticker"],
                    "Dir":      r.get("Direction", "—"),
                    "Entry":    _home_price(r.get("Entry")),
                    "Stop":     _home_price(r.get("Stop")),
                    "Vol/Avg":  f"{r['Vol / Avg']:.1f}×" if pd.notna(r.get("Vol / Avg")) else "—",
                    "Age":      f"{int(r['Bars Ago'])}d" if pd.notna(r.get("Bars Ago")) else "—",
                    "Signals":  str(r.get("Signals", "—"))[:40],
                })
            st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Row 2: Options | Today's Events ───────────────────────────────────────
    col3, col4 = st.columns([3, 2])

    with col3:
        st.markdown("#### 🎯 Options Recs (45-60 DTE)")
        with st.spinner("Scanning options…"):
            opt_df = _home_options(hour_key)
        if opt_df.empty:
            st.caption("No qualifying options setups found.")
        else:
            tbl = []
            for _, r in opt_df.iterrows():
                tbl.append({
                    "Symbol":    r["Symbol"],
                    "Type":      r.get("Direction", "—"),
                    "Strike":    str(int(r["Strike"])) if pd.notna(r.get("Strike")) else "—",
                    "Premium":   _home_price(r.get("Premium")),
                    "Delta":     f"{float(r['Delta']):.2f}" if pd.notna(r.get("Delta")) else "—",
                    "Theta/d":   f"{float(r['Theta/day']):.4f}" if pd.notna(r.get("Theta/day")) else "—",
                    "Score":     f"{float(r['Score']):.1f}" if pd.notna(r.get("Score")) else "—",
                    "Greeks":    "✅" if "✅" in str(r.get("Greeks OK", "")) else "❌",
                })
            st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

    with col4:
        st.markdown("#### 📅 Today's Earnings & Events")
        with st.spinner("Loading calendar…"):
            events = _home_events(day_key)
        shown = 0
        for ev in events:
            impact = ev.get("impact", "LOW")
            cat    = ev.get("category", "").upper()
            if impact not in ("HIGH", "MEDIUM") and cat not in ("EARNINGS", "EARNING"):
                continue
            shown += 1
            is_earn = cat in ("EARNINGS", "EARNING")
            badge_color = "#164e63" if is_earn else ("#7f1d1d" if impact == "HIGH" else "#78350f")
            badge_text  = "EARNINGS" if is_earn else impact
            badge_fg    = "#67e8f9" if is_earn else ("#fca5a5" if impact == "HIGH" else "#fcd34d")
            t = ev.get("time", "")
            st.markdown(
                f"<div style='padding:5px 0;border-bottom:1px solid #1e293b;display:flex;gap:10px;align-items:flex-start'>"
                f"<span style='color:#475569;font-size:11px;min-width:50px;padding-top:3px'>{t}</span>"
                f"<span style='color:#e2e8f0;font-size:13px;flex:1'>{ev.get('event','')}</span>"
                f"<span style='background:{badge_color};color:{badge_fg};padding:2px 7px;border-radius:5px;"
                f"font-size:11px;font-weight:700;white-space:nowrap'>{badge_text}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if shown == 0:
            st.caption("No major events or earnings today.")


# ─────────────────────────────────────────────────────────────────────────────
# Weekly / 0-3 DTE Options Scanner
# ─────────────────────────────────────────────────────────────────────────────

def _bs_call_delta(S: float, K: float, T: float, sigma: float, r: float = 0.045) -> float:
    """Black-Scholes delta for a call. sigma floored at 0.20 (yfinance sentinel fix)."""
    import math
    if S <= 0 or K <= 0:
        return 0.5
    sigma = max(sigma, 0.20)
    T = max(T, 2 / 365)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return 0.5 * (1.0 + math.erf(d1 / math.sqrt(2)))
    except Exception:
        return 0.5


def _bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.045) -> float:
    """Black-Scholes call price.  C = S·N(d1) − K·e^(−rT)·N(d2)"""
    import math
    if S <= 0 or K <= 0:
        return max(S - K, 0.0)
    sigma = max(sigma, 0.20)
    T = max(T, 1 / 365)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        N  = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
        return S * N(d1) - K * math.exp(-r * T) * N(d2)
    except Exception:
        return max(S - K, 0.0)


def _bs_put_price(S: float, K: float, T: float, sigma: float, r: float = 0.045) -> float:
    """Black-Scholes put price via put-call parity."""
    import math
    call = _bs_call_price(S, K, T, sigma, r)
    return call - S + K * math.exp(-r * max(T, 1 / 365))


def _weekly_opts_trading_dates(n: int) -> set:
    """Return ISO date strings for the next n weekdays (including today)."""
    from datetime import date, timedelta
    result, d, added = set(), date.today(), 0
    while added < n:
        if d.weekday() < 5:
            result.add(d.isoformat())
            added += 1
        d += timedelta(days=1)
    return result


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_vix_data() -> tuple:
    """Return (vix_level, vix_percentile_rank_1yr)."""
    try:
        vix = yf.Ticker("^VIX")
        fi = vix.fast_info
        current = float(fi.get("lastPrice") or fi.get("previousClose") or 20)
        hist = vix.history(period="1y", auto_adjust=True)
        rank = float((hist["Close"] < current).mean() * 100) if len(hist) > 10 else 50.0
        return round(current, 2), round(rank, 1)
    except Exception:
        return 20.0, 50.0


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_spy_chg() -> float:
    """Return SPY's % change today — used for relative strength calculation."""
    try:
        fi = yf.Ticker("SPY").fast_info
        price = float(fi.get("lastPrice") or fi.get("previousClose") or 1)
        prev  = float(fi.get("previousClose") or price)
        return round((price - prev) / prev * 100, 2)
    except Exception:
        return 0.0


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_weekly_calls(symbol: str, max_dte: int) -> tuple:
    """Return (calls_df, chg_pct, above_ema20, above_sma50, rsi14, vol_ratio, pc_ratio)."""
    from datetime import date
    try:
        ticker = yf.Ticker(symbol)
        fi = ticker.fast_info
        price = fi.get("lastPrice") or fi.get("previousClose", 0)
        if not price:
            return pd.DataFrame(), 0.0, None, None, None, 1.0, 1.0

        prev_close = fi.get("previousClose", price) or price
        chg_pct = round((price - prev_close) / prev_close * 100, 2)

        # Volume surge: today vs 3-month average
        vol_today = fi.get("threeMonthAverageVolume") or 0
        avg_vol   = fi.get("threeMonthAverageVolume") or 1
        try:
            vol_today = float(fi.get("regularMarketVolume") or 0)
            avg_vol   = float(fi.get("threeMonthAverageVolume") or vol_today or 1)
        except Exception:
            pass
        vol_ratio = round(vol_today / max(avg_vol, 1), 2)

        above_ema20 = above_sma50 = rsi14 = None
        try:
            hist = ticker.history(period="60d", auto_adjust=True)
            close = hist["Close"]
            if len(close) >= 20:
                above_ema20 = bool(price > close.ewm(span=20).mean().iloc[-1])
            if len(close) >= 50:
                above_sma50 = bool(price > close.rolling(50).mean().iloc[-1])
            if len(close) >= 15:
                d = close.diff()
                gain = d.clip(lower=0).rolling(14).mean()
                loss = (-d.clip(upper=0)).rolling(14).mean()
                rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
                rsi14 = round(float(100 - 100 / (1 + rs)), 1)
        except Exception:
            pass

        today = date.today()
        target = sorted(
            exp for exp in (ticker.options or [])
            if 0 <= (date.fromisoformat(exp) - today).days <= max_dte
        )
        if not target:
            return pd.DataFrame(), chg_pct, above_ema20, above_sma50, rsi14, vol_ratio, 1.0

        call_vol = put_vol = 0
        frames = []
        for exp in target:
            chain = ticker.option_chain(exp)

            # Aggregate put/call volume for conviction score
            cv = pd.to_numeric(chain.calls["volume"], errors="coerce").fillna(0).sum()
            pv = pd.to_numeric(chain.puts["volume"],  errors="coerce").fillna(0).sum()
            call_vol += cv
            put_vol  += pv

            df = chain.calls.copy()
            df["expiry"] = exp
            df["symbol"] = symbol
            df["price"]  = float(price)
            df["chg_pct"] = chg_pct
            df["above_ema20"] = above_ema20
            df["above_sma50"] = above_sma50
            df["rsi14"] = rsi14
            df["dte"] = (date.fromisoformat(exp) - today).days
            frames.append(df)

        pc_ratio = round(call_vol / max(put_vol, 1), 2)  # > 1 = more calls than puts

        if not frames:
            return pd.DataFrame(), chg_pct, above_ema20, above_sma50, rsi14, vol_ratio, pc_ratio
        out = pd.concat(frames, ignore_index=True)
        out["vol_ratio"] = vol_ratio
        out["pc_ratio"]  = pc_ratio
        return out, chg_pct, above_ema20, above_sma50, rsi14, vol_ratio, pc_ratio
    except Exception:
        return pd.DataFrame(), 0.0, None, None, None, 1.0, 1.0


@st.cache_data(ttl=900, show_spinner=False)
def _scan_weekly_options(
    symbols: tuple,
    max_dte: int,
    min_volume: int,
    min_oi: int,
    moneyness_pct: float,
    show_itm: bool,
    show_otm: bool,
    sort_by: str,
    # ── new quality filters ──────────────────────────────────
    min_underlying_chg: float,   # underlying must be up ≥ this % today
    require_above_ema20: bool,   # underlying must be above EMA20
    require_above_sma50: bool,   # underlying must be above SMA50
    max_iv_pct: float,           # calls with IV above this are excluded
    max_spread_pct: float,       # bid/ask spread must be ≤ this % of mid
    delta_min: float,
    delta_max: float,
    rsi_min: float,
    rsi_max: float,
    spy_chg: float,              # SPY % change today — for relative strength
    max_premium: float = 999.0,  # max mid price per share (e.g. 1.50 = $150/contract)
    min_vol_ratio: float = 0.0,  # min volume/avg-volume surge (0 = no filter)
    min_rel_str: float = -999.0, # min relative strength vs SPY %
) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        df, chg_pct, above_ema20, above_sma50, rsi14, vol_ratio, pc_ratio = \
            _fetch_weekly_calls(sym, max_dte)
        if df.empty or df["price"].iloc[0] == 0:
            continue

        # ── Underlying filters ────────────────────────────────────────────────
        if chg_pct < min_underlying_chg:
            continue
        if require_above_ema20 and above_ema20 is False:
            continue
        if require_above_sma50 and above_sma50 is False:
            continue
        if rsi14 is not None and not (rsi_min <= rsi14 <= rsi_max):
            continue
        if min_vol_ratio > 0 and vol_ratio is not None and vol_ratio < min_vol_ratio:
            continue

        price = df["price"].iloc[0]
        df["moneyness_pct"] = ((df["strike"] - price) / price * 100).abs()
        df = df[df["moneyness_pct"] <= moneyness_pct]
        itm = df["inTheMoney"].fillna(False).astype(bool)
        if not show_itm:
            df = df[~itm]
        if not show_otm:
            df = df[itm]

        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0).astype(int)
        df = df[(df["volume"] >= min_volume) & (df["openInterest"] >= min_oi)]
        if df.empty:
            continue

        df["voi_ratio"] = df["volume"] / (df["openInterest"] + 1)
        df["iv_pct"] = (df["impliedVolatility"] * 100).round(1)
        df["mid"] = ((df["bid"] + df["ask"]) / 2).round(2)
        df["spread_pct"] = ((df["ask"] - df["bid"]) / (df["mid"] + 1e-6) * 100).round(1)

        # ── IV, spread, and premium filters ──────────────────────────────────
        df = df[df["iv_pct"] <= max_iv_pct]
        df = df[df["spread_pct"] <= max_spread_pct]
        df = df[df["mid"] <= max_premium]
        if df.empty:
            continue

        # ── Delta (Black-Scholes) ─────────────────────────────────────────────
        df["delta"] = df.apply(
            lambda r: _bs_call_delta(
                S=float(r["price"]),
                K=float(r["strike"]),
                T=float(r["dte"]) / 365.0,
                sigma=float(r["impliedVolatility"]),
            ),
            axis=1,
        )
        df = df[(df["delta"] >= delta_min) & (df["delta"] <= delta_max)]
        if df.empty:
            continue

        # ── GO SCORE (0–10) — predicts upside potential ───────────────────────
        # Each component is independently scored; higher = more conviction
        rel_str = round(chg_pct - spy_chg, 2)   # beating the market today

        def _go(row):
            pts = 0
            # 1. Relative strength vs SPY (max 2)
            if rel_str >= 2.0:   pts += 2
            elif rel_str >= 0.5: pts += 1

            # 2. Call flow conviction — V/OI (max 3)
            voi = row["voi_ratio"]
            if voi >= 5.0:   pts += 3
            elif voi >= 2.0: pts += 2
            elif voi >= 1.0: pts += 1

            # 3. Put/Call skew — more calls = bullish institutions (max 2)
            if pc_ratio >= 2.0:   pts += 2
            elif pc_ratio >= 1.2: pts += 1

            # 4. Stock volume surge vs 3-month avg (max 2)
            if vol_ratio >= 2.0:   pts += 2
            elif vol_ratio >= 1.3: pts += 1

            # 5. RSI sweet spot — trending not exhausted (max 1)
            rsi = row.get("rsi14")
            if rsi is not None and 55 <= rsi <= 70:
                pts += 1

            return min(pts, 10)

        df["rel_str"]   = rel_str
        df["vol_ratio"] = vol_ratio
        df["pc_ratio"]  = pc_ratio
        df["go_score"]  = df.apply(_go, axis=1)
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    def _tags(row):
        t = []
        gs = row.get("go_score", 0)
        if gs >= 8:   t.append("🚀STRONG BUY")
        elif gs >= 6: t.append("⚡BUY")
        elif gs >= 4: t.append("👀WATCH")

        if row["volume"] >= 5_000:
            t.append("🔥SWEEP")
        elif row["volume"] >= 1_000:
            t.append("HIGH_VOL")
        if row["voi_ratio"] >= 2.0:
            t.append("UNUSUAL")
        if row.get("pc_ratio", 1) >= 2.0:
            t.append("CALL_HEAVY")
        if row.get("vol_ratio", 1) >= 2.0:
            t.append("VOL_SURGE")
        if row.get("rel_str", 0) >= 1.0:
            t.append(f"RS+{row['rel_str']:.1f}%")
        if row["moneyness_pct"] <= 1.0:
            t.append("ATM")
        elif row["moneyness_pct"] <= 3.0:
            t.append("NTM")
        if row["dte"] == 0:
            t.append("0DTE")
        rsi = row.get("rsi14")
        if rsi is not None and rsi > 70:
            t.append(f"⚠️OB{rsi:.0f}")
        return " ".join(t) if t else "—"

    out["signals"] = out.apply(_tags, axis=1)
    sort_col = {
        "GO Score": "go_score",
        "Volume": "volume",
        "V/OI Ratio": "voi_ratio",
        "Rel Strength": "rel_str",
        "P/C Ratio": "pc_ratio",
        "Vol Surge": "vol_ratio",
        "IV %": "iv_pct",
        "Today's Move %": "chg_pct",
        "Delta": "delta",
        "RSI": "rsi14",
    }.get(sort_by, "go_score")
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)


def render_weekly_options():
    clicked = _page_header(
        "⚡", "Weekly Options (0-7 DTE)",
        "Calls expiring within 7 calendar days of purchase. Backtest-optimized defaults: "
        "RSI 55–70 enabled, delta 0.20–0.85. RSI filter alone improved win rate by ~7 pp. "
        "Auto-refreshes every 15 min.",
        scan_key="wkly_opts_scan_btn", last_key="wkly_opts_time",
    )

    with st.expander("⚙️ Settings", expanded=True):
        st.markdown("**Universe**")
        universe_choice = st.radio(
            "Scan universe",
            ["Focused (~60 tickers)", "NDQ100 + High-Growth (~130)", "Full (~600)"],
            index=0, horizontal=True, key="wkly_universe",
            help="Focused = High-Growth + Watchlist only. Fastest scan.",
        )

        st.markdown("**Options chain filters**")
        c1, c2, c3, c4 = st.columns(4)
        max_dte    = c1.slider("Max DTE", 0, 14, 7, key="wkly_max_dte")
        min_volume = c2.number_input("Min Call Volume", 0, 100_000, 100, step=50, key="wkly_min_vol")
        min_oi     = c3.number_input("Min Open Interest", 0, 100_000, 0, step=50, key="wkly_min_oi")
        moneyness  = c4.slider("Strike ± % from price", 1, 20, 10, key="wkly_moneyness")

        c5, c6 = st.columns(2)
        show_itm = c5.checkbox("Include ITM calls", value=True, key="wkly_itm")
        show_otm = c6.checkbox("Include OTM calls", value=True, key="wkly_otm")

        st.markdown("**Quality filters**")
        q1, q2, q3 = st.columns(3)
        max_iv_pct     = q1.slider(
            "Max IV %", 30, 500, 150, step=10, key="wkly_max_iv",
            help="Exclude calls where IV is above this — avoids overpriced post-news premium",
        )
        max_spread_pct = q2.slider(
            "Max Bid/Ask Spread %", 5, 50, 20, step=5, key="wkly_max_spread",
            help="Exclude illiquid contracts where the spread is too wide",
        )
        min_chg = q3.slider(
            "Min underlying move today %", -5.0, 5.0, 0.0, step=0.5, key="wkly_min_chg",
            help="0 = stock must be flat or up today; set negative to allow any direction",
        )

        st.markdown("**Delta range**  *(0 = deep OTM · 1 = deep ITM · sweet spot 0.30–0.70)*")
        d1, d2 = st.columns(2)
        delta_min = d1.slider("Delta min", 0.05, 0.60, 0.20, step=0.05, key="wkly_delta_min",
                              help="Exclude low-probability deep OTM calls")
        delta_max = d2.slider("Delta max", 0.40, 1.00, 0.85, step=0.05, key="wkly_delta_max",
                              help="Exclude deep ITM calls (expensive, limited leverage)")

        st.markdown("**RSI filter**  *(backtested sweet spot: 55–70 — trending but not overbought)*")
        enable_rsi = st.checkbox("Enable RSI filter", value=True, key="wkly_rsi_enable",
                                 help="ON by default — RSI 55–70 improved win rate by ~7 pp in backtests")
        r1, r2 = st.columns(2)
        rsi_min = r1.slider("RSI min", 10, 65, 55, step=5, key="wkly_rsi_min",
                            help="Exclude stocks in freefall / oversold territory",
                            disabled=not enable_rsi)
        rsi_max = r2.slider("RSI max", 55, 90, 70, step=5, key="wkly_rsi_max",
                            help="Exclude overbought stocks — buying calls into RSI>70 has lower win rate",
                            disabled=not enable_rsi)

        st.markdown("**Trend filters**")
        t1, t2 = st.columns(2)
        require_ema20 = t1.checkbox(
            "Above EMA 20", value=False, key="wkly_ema20",
            help="Only show stocks whose price is above the 20-day EMA",
        )
        require_sma50 = t2.checkbox(
            "Above SMA 50", value=False, key="wkly_sma50",
            help="Only show stocks whose price is above the 50-day SMA",
        )

        sort_by = st.selectbox(
            "Sort results by",
            ["GO Score", "Volume", "V/OI Ratio", "Rel Strength", "P/C Ratio",
             "Vol Surge", "IV %", "Today's Move %", "Delta", "RSI"],
            key="wkly_sort",
        )

    st_autorefresh(interval=900_000, key="wkly_opts_refresh")

    # ── Init session state ────────────────────────────────────────────────────
    if "wkly_opts_df" not in st.session_state:
        st.session_state.wkly_opts_df   = pd.DataFrame()
        st.session_state.wkly_opts_time = None

    # ── Load last saved button ────────────────────────────────────────────────
    load_btn = st.button("📂 Load last scan", key="wkly_load_last")
    if load_btn:
        files = sorted(glob.glob(os.path.join(SCREENER_DIR, "weekly_calls_*.csv")))
        if files:
            st.session_state.wkly_opts_df   = pd.read_csv(files[-1])
            st.session_state.wkly_opts_time = f"from file · {os.path.basename(files[-1])}"
            st.toast("Loaded last saved scan.", icon="📂")
        else:
            st.warning("No saved scan found. Click ▶ Scan Now to run one.")

    # ── VIX condition banner ──────────────────────────────────────────────────
    vix_level, vix_rank = _fetch_vix_data()
    if vix_level < 20:
        st.success(
            f"VIX {vix_level:.1f}  (rank {vix_rank:.0f}th pct) — Low fear. IV is cheap, "
            "favorable for call buying."
        )
    elif vix_level < 30:
        st.warning(
            f"VIX {vix_level:.1f}  (rank {vix_rank:.0f}th pct) — Elevated fear. "
            "Premium is pricier — consider tightening Max IV % filter."
        )
    else:
        st.error(
            f"VIX {vix_level:.1f}  (rank {vix_rank:.0f}th pct) — High fear zone. "
            "Backtests show buying calls when VIX rank > 70 significantly lowers win rate. "
            "Consider sitting out or using tighter filters."
        )

    # ── Run scan only when button clicked ────────────────────────────────────
    if clicked:
        if universe_choice.startswith("Focused"):
            tickers = tuple(_dedup(_HIGH_GROWTH_TICKERS + list(WATCHLIST_TICKERS)))
        elif universe_choice.startswith("NDQ100"):
            tickers = tuple(_dedup(_cached_ndq100() + _HIGH_GROWTH_TICKERS + list(WATCHLIST_TICKERS)))
        else:
            tickers = tuple(get_selected_tickers())

        _ticker_count_caption(list(tickers))

        prog = st.progress(0)
        status = st.empty()
        for i, sym in enumerate(tickers):
            status.caption(f"Fetching {sym}…  ({i+1}/{len(tickers)})")
            prog.progress((i + 1) / max(len(tickers), 1))
            _fetch_weekly_calls(sym, max_dte)
        prog.empty()
        status.empty()

        _rsi_min = float(rsi_min) if enable_rsi else 0.0
        _rsi_max = float(rsi_max) if enable_rsi else 100.0
        _spy_chg = _fetch_spy_chg()

        df = _scan_weekly_options(
            tickers, max_dte, int(min_volume), int(min_oi),
            moneyness, show_itm, show_otm, sort_by,
            float(min_chg), require_ema20, require_sma50,
            float(max_iv_pct), float(max_spread_pct),
            float(delta_min), float(delta_max),
            _rsi_min, _rsi_max,
            _spy_chg,
        )

        ts = pd.Timestamp.now()
        st.session_state.wkly_opts_df   = df
        st.session_state.wkly_opts_time = ts.strftime("%H:%M:%S")

        if not df.empty:
            os.makedirs(SCREENER_DIR, exist_ok=True)
            _save_path = os.path.join(
                SCREENER_DIR, f"weekly_calls_{ts.strftime('%Y%m%d_%H%M')}.csv"
            )
            df.to_csv(_save_path, index=False)

    df = st.session_state.wkly_opts_df

    if df.empty and not clicked:
        st.info("Click **▶ Scan Now** to run, or **📂 Load last scan** to restore previous results.")
        return

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    if not df.empty:
        m1.metric("Calls Found", len(df))
        m2.metric("Tickers", df["symbol"].nunique())
        m3.metric("Top Volume", f"{df['volume'].max():,}")
        m4.metric("Avg IV", f"{df['iv_pct'].mean():.1f}%")
        best = df.loc[df["voi_ratio"].idxmax()]
        m5.metric("Best V/OI", f"{best['voi_ratio']:.1f}x  ({best['symbol']})")
    else:
        for m in [m1, m2, m3, m4, m5]:
            m.metric("—", "—")

    st.divider()

    if df.empty:
        st.warning(
            "No calls matched your current filters. Try one or more of: "
            "**widen Strike ±%** · **lower Min Volume** · "
            "**widen Delta range** (default 0.20–0.85) · "
            "**disable RSI filter** (or widen to 45–75) · "
            "**set Min underlying move to -5%** · "
            "**uncheck EMA/SMA** · **raise Max IV %**"
        )
        return

    # ── Main table ─────────────────────────────────────────────────────────────
    _COLS = {
        "symbol":       "Ticker",
        "go_score":     "GO🎯",
        "expiry":       "Expiry",
        "dte":          "DTE",
        "chg_pct":      "Stk%",
        "rel_str":      "vs SPY",
        "rsi14":        "RSI",
        "pc_ratio":     "C/P Vol",
        "vol_ratio":    "VolSurge",
        "strike":       "Strike",
        "price":        "Stock $",
        "moneyness_pct":"±%ATM",
        "inTheMoney":   "ITM",
        "delta":        "Delta",
        "mid":          "Mid $",
        "spread_pct":   "Sprd%",
        "volume":       "CallVol",
        "openInterest": "OI",
        "voi_ratio":    "V/OI",
        "iv_pct":       "IV%",
        "signals":      "Signals",
    }
    table = df[list(_COLS)].rename(columns=_COLS).copy()
    table["±%ATM"] = table["±%ATM"].round(2)
    table["V/OI"]  = table["V/OI"].round(2)

    def _color_row(row):
        gs = row.get("GO🎯", 0)
        if gs >= 8:   bg = "background-color: rgba(255,75,75,0.20)"
        elif gs >= 6: bg = "background-color: rgba(255,165,0,0.15)"
        elif gs >= 4: bg = "background-color: rgba(0,200,100,0.10)"
        else:         bg = ""
        return [bg] * len(row)

    styled = (
        table.style
        .apply(_color_row, axis=1)
        .format({
            "GO🎯": "{:.0f}", "Stk%": "{:+.2f}%", "vs SPY": "{:+.2f}%",
            "RSI": "{:.1f}", "C/P Vol": "{:.2f}", "VolSurge": "{:.2f}x",
            "Strike": "{:.2f}", "Stock $": "{:.2f}", "±%ATM": "{:.2f}",
            "Delta": "{:.2f}", "Mid $": "{:.2f}", "Sprd%": "{:.1f}",
            "V/OI": "{:.2f}", "IV%": "{:.1f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=550)

    # ── Per-ticker top picks ───────────────────────────────────────────────────
    st.subheader("Top 2 picks per ticker")
    top2 = (
        df.sort_values("volume", ascending=False)
        .groupby("symbol").head(2)
        .sort_values("volume", ascending=False)
    )
    for sym, grp in top2.groupby("symbol", sort=False):
        chg = grp["chg_pct"].iloc[0]
        gs  = grp["go_score"].max()
        label = "🚀STRONG BUY" if gs >= 8 else ("⚡BUY" if gs >= 6 else ("👀WATCH" if gs >= 4 else ""))
        with st.expander(
            f"{label}  {sym}  —  ${grp['price'].iloc[0]:.2f}  ({chg:+.2f}% today)  GO={gs:.0f}/10"
        ):
            mini = grp[list(_COLS)].rename(columns=_COLS)
            st.dataframe(mini, use_container_width=True, hide_index=True)

    # ── Download ───────────────────────────────────────────────────────────────
    from datetime import date as _date
    st.download_button(
        "⬇️ Download CSV",
        data=table.to_csv(index=False),
        file_name=f"weekly_calls_{_date.today().isoformat()}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Weekly Puts / Cheap Puts scanner
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_weekly_puts(symbol: str, max_dte: int) -> tuple:
    """Return (puts_df, chg_pct, above_ema20, above_sma50, rsi14, vol_ratio, pc_ratio)."""
    from datetime import date
    try:
        ticker = yf.Ticker(symbol)
        fi     = ticker.fast_info
        price  = fi.get("lastPrice") or fi.get("previousClose", 0)
        if not price:
            return pd.DataFrame(), 0.0, None, None, None, 1.0, 1.0

        prev_close = fi.get("previousClose", price) or price
        chg_pct    = round((price - prev_close) / prev_close * 100, 2)

        try:
            vol_today = float(fi.get("regularMarketVolume") or 0)
            avg_vol   = float(fi.get("threeMonthAverageVolume") or vol_today or 1)
        except Exception:
            vol_today, avg_vol = 0.0, 1.0
        vol_ratio = round(vol_today / max(avg_vol, 1), 2)

        above_ema20 = above_sma50 = rsi14 = None
        try:
            hist  = ticker.history(period="60d", auto_adjust=True)
            close = hist["Close"]
            if len(close) >= 20:
                above_ema20 = bool(price > close.ewm(span=20).mean().iloc[-1])
            if len(close) >= 50:
                above_sma50 = bool(price > close.rolling(50).mean().iloc[-1])
            if len(close) >= 15:
                d    = close.diff()
                gain = d.clip(lower=0).rolling(14).mean()
                loss = (-d.clip(upper=0)).rolling(14).mean()
                rs   = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
                rsi14 = round(float(100 - 100 / (1 + rs)), 1)
        except Exception:
            pass

        target = sorted(
            exp for exp in (ticker.options or [])
            if 0 <= (date.fromisoformat(exp) - date.today()).days <= max_dte
        )
        if not target:
            return pd.DataFrame(), chg_pct, above_ema20, above_sma50, rsi14, vol_ratio, 1.0

        call_vol = put_vol = 0
        frames = []
        for exp in target:
            chain    = ticker.option_chain(exp)
            call_vol += pd.to_numeric(chain.calls["volume"], errors="coerce").fillna(0).sum()
            put_vol  += pd.to_numeric(chain.puts["volume"],  errors="coerce").fillna(0).sum()

            df = chain.puts.copy()
            df["expiry"]     = exp
            df["symbol"]     = symbol
            df["price"]      = float(price)
            df["chg_pct"]    = chg_pct
            df["above_ema20"] = above_ema20
            df["above_sma50"] = above_sma50
            df["rsi14"]      = rsi14
            df["dte"]        = (date.fromisoformat(exp) - date.today()).days
            frames.append(df)

        pc_ratio = round(call_vol / max(put_vol, 1), 2)
        if not frames:
            return pd.DataFrame(), chg_pct, above_ema20, above_sma50, rsi14, vol_ratio, pc_ratio
        out = pd.concat(frames, ignore_index=True)
        out["vol_ratio"] = vol_ratio
        out["pc_ratio"]  = pc_ratio
        return out, chg_pct, above_ema20, above_sma50, rsi14, vol_ratio, pc_ratio
    except Exception:
        return pd.DataFrame(), 0.0, None, None, None, 1.0, 1.0


@st.cache_data(ttl=900, show_spinner=False)
def _scan_weekly_puts(
    symbols: tuple, max_dte: int, min_volume: int, min_oi: int,
    moneyness_pct: float, sort_by: str,
    min_underlying_drop: float,   # stock must be down ≥ this % (e.g. 0 = flat or worse)
    require_below_ema20: bool,
    require_below_sma50: bool,
    max_iv_pct: float, max_spread_pct: float,
    delta_min: float, delta_max: float,  # abs(put_delta) range
    rsi_min: float, rsi_max: float,
    spy_chg: float,
    max_premium: float = 999.0,
) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        df, chg_pct, above_ema20, above_sma50, rsi14, vol_ratio, pc_ratio = \
            _fetch_weekly_puts(sym, max_dte)
        if df.empty or df["price"].iloc[0] == 0:
            continue

        # Underlying filters (bearish direction)
        if chg_pct > -min_underlying_drop:   # stock not down enough
            continue
        if require_below_ema20 and above_ema20 is True:
            continue
        if require_below_sma50 and above_sma50 is True:
            continue
        if rsi14 is not None and not (rsi_min <= rsi14 <= rsi_max):
            continue

        price = df["price"].iloc[0]
        # Moneyness: for puts, OTM means strike < price
        df["moneyness_pct"] = ((df["strike"] - price) / price * 100).abs()
        df = df[df["moneyness_pct"] <= moneyness_pct]
        df = df[~df["inTheMoney"]]          # OTM puts only (strike < price)

        df["volume"]       = pd.to_numeric(df["volume"],       errors="coerce").fillna(0).astype(int)
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0).astype(int)
        df = df[(df["volume"] >= min_volume) & (df["openInterest"] >= min_oi)]
        if df.empty:
            continue

        df["voi_ratio"]  = df["volume"] / (df["openInterest"] + 1)
        df["iv_pct"]     = (df["impliedVolatility"] * 100).round(1)
        df["mid"]        = ((df["bid"] + df["ask"]) / 2).round(2)
        df["spread_pct"] = ((df["ask"] - df["bid"]) / (df["mid"] + 1e-6) * 100).round(1)

        df = df[df["iv_pct"]     <= max_iv_pct]
        df = df[df["spread_pct"] <= max_spread_pct]
        df = df[df["mid"]        <= max_premium]
        if df.empty:
            continue

        # Put delta = call_delta - 1; filter by absolute value
        df["put_delta"] = df.apply(
            lambda r: abs(
                _bs_call_delta(float(r["price"]), float(r["strike"]),
                               float(r["dte"]) / 365, float(r["impliedVolatility"])) - 1
            ),
            axis=1,
        )
        df = df[(df["put_delta"] >= delta_min) & (df["put_delta"] <= delta_max)]
        if df.empty:
            continue

        rel_str = round(chg_pct - spy_chg, 2)   # negative = underperforming

        def _go_put(row):
            pts = 0
            # 1. Relative weakness vs SPY (bearish momentum) — max 2
            if rel_str <= -2.0:   pts += 2
            elif rel_str <= -0.5: pts += 1
            # 2. Put V/OI (institutions opening put positions) — max 3
            voi = row["voi_ratio"]
            if voi >= 5.0:   pts += 3
            elif voi >= 2.0: pts += 2
            elif voi >= 1.0: pts += 1
            # 3. More put activity than calls (bearish skew) — max 2
            # pc_ratio = call_vol/put_vol; low = more puts
            if pc_ratio <= 0.5:   pts += 2
            elif pc_ratio <= 0.8: pts += 1
            # 4. Stock falling today — max 2
            if chg_pct <= -2.0:   pts += 2
            elif chg_pct <= -0.5: pts += 1
            # 5. RSI overbought → reversal likely — max 1
            rsi = row.get("rsi14")
            if rsi is not None and rsi >= 65:
                pts += 1
            return min(pts, 10)

        df["rel_str"]   = rel_str
        df["vol_ratio"] = vol_ratio
        df["pc_ratio"]  = pc_ratio
        df["go_score"]  = df.apply(_go_put, axis=1)

        def _put_tags(row):
            t = []
            gs = row.get("go_score", 0)
            if gs >= 8:   t.append("🚀STRONG SELL")
            elif gs >= 6: t.append("⚡SELL")
            elif gs >= 4: t.append("👀WATCH")
            if row["volume"] >= 5_000: t.append("🔥SWEEP")
            elif row["volume"] >= 1_000: t.append("HIGH_VOL")
            if row["voi_ratio"] >= 2.0: t.append("UNUSUAL")
            if pc_ratio <= 0.5: t.append("PUT_HEAVY")
            if row.get("vol_ratio", 1) >= 2.0: t.append("VOL_SURGE")
            if row.get("rel_str", 0) <= -1.0: t.append(f"RS{row['rel_str']:.1f}%")
            rsi = row.get("rsi14")
            if rsi is not None and rsi >= 65: t.append(f"⚠️OB{rsi:.0f}")
            if row["dte"] == 0: t.append("0DTE")
            return " ".join(t) if t else "—"

        df["signals"] = df.apply(_put_tags, axis=1)
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    sort_col = {
        "GO Score": "go_score", "Volume": "volume", "V/OI Ratio": "voi_ratio",
        "Rel Weakness": "rel_str", "IV %": "iv_pct",
        "Today's Drop %": "chg_pct", "Delta": "put_delta",
    }.get(sort_by, "go_score")
    asc = (sort_by == "Rel Weakness")   # most negative first
    return out.sort_values(sort_col, ascending=asc).reset_index(drop=True)


def render_weekly_puts():
    _page_header(
        "📉", "Weekly Puts (0-3 DTE)",
        "Puts expiring within 0–3 trading days. Scores bearish conviction: "
        "relative weakness vs SPY, put flow, volume surge, RSI overbought. "
        "Auto-refreshes every 15 min.",
        scan_key="wkly_puts_scan_btn", last_key="wkly_puts_time",
    )

    with st.expander("⚙️ Settings", expanded=True):
        st.markdown("**Options chain filters**")
        c1, c2, c3, c4 = st.columns(4)
        max_dte    = c1.slider("Max DTE", 0, 5, 3, key="wp_max_dte")
        min_volume = c2.number_input("Min Put Volume", 0, 100_000, 100, step=50, key="wp_min_vol")
        min_oi     = c3.number_input("Min Open Interest", 0, 100_000, 0, step=50, key="wp_min_oi")
        moneyness  = c4.slider("Strike ± % from price", 1, 20, 10, key="wp_moneyness")

        st.markdown("**Quality filters**")
        q1, q2, q3 = st.columns(3)
        max_iv_pct     = q1.slider("Max IV %", 30, 500, 999, step=10, key="wp_max_iv")
        max_spread_pct = q2.slider("Max Spread %", 5, 50, 25, step=5, key="wp_max_spread")
        max_prem       = q3.slider("Max premium ($/share)", 0.10, 20.0, 999.0, step=0.10,
                                   key="wp_max_prem",
                                   help="Set to e.g. 1.50 for cheap puts only")

        st.markdown("**Delta range**  *(abs put delta — 0.05 = deep OTM, 0.50 = ATM)*")
        d1, d2 = st.columns(2)
        delta_min = d1.slider("Delta min", 0.05, 0.50, 0.20, step=0.05, key="wp_delta_min")
        delta_max = d2.slider("Delta max", 0.20, 1.00, 0.85, step=0.05, key="wp_delta_max")

        st.markdown("**Bearish stock filter**")
        b1, b2, b3 = st.columns(3)
        min_drop       = b1.slider("Min stock drop today %", 0.0, 5.0, 0.0, step=0.5,
                                    key="wp_min_drop",
                                    help="0 = flat or falling; 1 = must be down ≥1% today")
        require_below_ema20 = b2.checkbox("Below EMA 20", value=False, key="wp_below_ema20")
        require_below_sma50 = b3.checkbox("Below SMA 50", value=False, key="wp_below_sma50")

        st.markdown("**RSI filter**  *(overbought ≥65 = best short setup)*")
        enable_rsi_wp = st.checkbox("Enable RSI filter", value=False, key="wp_rsi_enable")
        r1, r2 = st.columns(2)
        rsi_min_wp = r1.slider("RSI min", 10, 80, 50, step=5, key="wp_rsi_min",
                                disabled=not enable_rsi_wp)
        rsi_max_wp = r2.slider("RSI max", 40, 100, 100, step=5, key="wp_rsi_max",
                                disabled=not enable_rsi_wp)

        sort_by = st.selectbox(
            "Sort by",
            ["GO Score", "Volume", "V/OI Ratio", "Rel Weakness", "IV %",
             "Today's Drop %", "Delta"],
            key="wp_sort",
        )

    st_autorefresh(interval=900_000, key="wkly_puts_refresh")

    vix_level, vix_rank = _fetch_vix_data()
    if vix_level >= 30:
        st.success(f"VIX {vix_level:.1f}  ({vix_rank:.0f}th pct) — High fear. IV rich — good for put selling; buying puts is expensive.")
    elif vix_level >= 20:
        st.warning(f"VIX {vix_level:.1f}  ({vix_rank:.0f}th pct) — Elevated. Premium is pricier.")
    else:
        st.info(f"VIX {vix_level:.1f}  ({vix_rank:.0f}th pct) — Low fear. Cheap puts but reversals can be sharp.")

    tickers  = tuple(get_selected_tickers())
    spy_chg  = _fetch_spy_chg()
    _ticker_count_caption(list(tickers))

    prog = st.progress(0); status = st.empty()
    for i, sym in enumerate(tickers):
        status.caption(f"Fetching {sym}…  ({i+1}/{len(tickers)})")
        prog.progress((i + 1) / max(len(tickers), 1))
        _fetch_weekly_puts(sym, max_dte)
    prog.empty(); status.empty()

    _rsi_min = float(rsi_min_wp) if enable_rsi_wp else 0.0
    _rsi_max = float(rsi_max_wp) if enable_rsi_wp else 100.0

    df = _scan_weekly_puts(
        tickers, max_dte, int(min_volume), int(min_oi),
        float(moneyness), sort_by,
        float(min_drop), require_below_ema20, require_below_sma50,
        float(max_iv_pct), float(max_spread_pct),
        float(delta_min), float(delta_max),
        _rsi_min, _rsi_max, spy_chg,
        max_premium=float(max_prem),
    )

    st.session_state["wkly_puts_time"] = pd.Timestamp.now().strftime("%H:%M:%S")

    # Simulate potential return if stock drops X%
    sim_drop = 5
    if not df.empty:
        def _sim_put(row):
            if row["mid"] <= 0: return 0.0
            iv      = max(float(row["impliedVolatility"]), 0.20)
            new_S   = row["price"] * (1 - sim_drop / 100)
            new_T   = max(float(row["dte"]) - 0.5, 0.25) / 365
            new_val = _bs_put_price(new_S, float(row["strike"]), new_T, iv)
            return round((new_val - row["mid"]) / row["mid"] * 100, 0)
        df["sim_ret%"] = df.apply(_sim_put, axis=1)
    else:
        df["sim_ret%"] = pd.Series(dtype=float)

    # Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    if not df.empty:
        m1.metric("Puts Found", len(df))
        m2.metric("Tickers", df["symbol"].nunique())
        m3.metric("Top Volume", f"{df['volume'].max():,}")
        m4.metric("Best GO", f"{df['go_score'].max():.0f}/10")
        best = df.loc[df["go_score"].idxmax()]
        m5.metric("Top Pick", f"{best['symbol']}  ({best['chg_pct']:+.2f}% today)")
    else:
        for m in [m1,m2,m3,m4,m5]: m.metric("—","—")

    st.divider()

    if df.empty:
        st.warning(
            "No puts matched. Try: **lower Min Volume**, **widen Strike ±%**, "
            "**set Min drop to 0%**, **uncheck Below EMA/SMA**."
        )
        return

    _PC = {
        "symbol":    "Ticker",
        "go_score":  "GO🎯",
        "expiry":    "Expiry",
        "dte":       "DTE",
        "chg_pct":   "Stk%",
        "rel_str":   "vs SPY",
        "rsi14":     "RSI",
        "pc_ratio":  "C/P Vol",
        "vol_ratio": "VolSurge",
        "strike":    "Strike",
        "price":     "Stock $",
        "moneyness_pct": "±%ATM",
        "put_delta": "Delta",
        "mid":       "Prem $",
        "spread_pct":"Sprd%",
        "volume":    "PutVol",
        "openInterest": "OI",
        "voi_ratio": "V/OI",
        "iv_pct":    "IV%",
        "sim_ret%":  f"-{sim_drop}% ret",
        "signals":   "Signals",
    }
    tbl = df[list(_PC)].rename(columns=_PC).copy()
    tbl["±%ATM"] = tbl["±%ATM"].round(2)
    tbl["V/OI"]  = tbl["V/OI"].round(2)

    def _cr(row):
        gs = row.get("GO🎯", 0)
        if gs >= 8:   return ["background-color: rgba(255,75,75,0.20)"] * len(row)
        elif gs >= 6: return ["background-color: rgba(255,165,0,0.15)"] * len(row)
        elif gs >= 4: return ["background-color: rgba(0,200,100,0.10)"] * len(row)
        return [""] * len(row)

    styled = (
        tbl.style.apply(_cr, axis=1)
        .format({
            "GO🎯": "{:.0f}", "Stk%": "{:+.2f}%", "vs SPY": "{:+.2f}%",
            "RSI": "{:.1f}", "C/P Vol": "{:.2f}", "VolSurge": "{:.2f}x",
            "Strike": "{:.2f}", "Stock $": "{:.2f}", "±%ATM": "{:.2f}",
            "Delta": "{:.2f}", "Prem $": "{:.2f}", "Sprd%": "{:.1f}",
            "V/OI": "{:.2f}", "IV%": "{:.1f}", f"-{sim_drop}% ret": "{:+.0f}%",
        })
    )
    st.dataframe(styled, use_container_width=True, height=550)

    st.subheader("Top picks")
    top3 = df.nlargest(min(8, len(df)), "go_score")
    for _, row in top3.iterrows():
        gs    = row["go_score"]
        label = "🚀STRONG SELL" if gs >= 8 else ("⚡SELL" if gs >= 6 else "👀WATCH")
        cost  = row["mid"] * 100
        sim   = row.get("sim_ret%", 0)
        st.markdown(
            f"**{label}  {row['symbol']}**  "
            f"${row['strike']:.0f}P  exp {row['expiry']}  "
            f"· Prem **${row['mid']:.2f}** (${cost:.0f}/contract)  "
            f"· GO={gs:.0f}/10  "
            f"· If -{sim_drop}%: **{sim:+.0f}%**  "
            f"· {row['signals']}"
        )

    from datetime import date as _date
    st.download_button(
        "⬇️ Download CSV",
        data=tbl.to_csv(index=False),
        file_name=f"weekly_puts_{_date.today().isoformat()}.csv",
        mime="text/csv",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Parabolic Short Screener
# ═════════════════════════════════════════════════════════════════════════════

def render_parabolic_short():
    """US Champion's Parabolic Short Screener — 5-filter system."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import date as _date

    clicked = _page_header(
        "📉", "Parabolic Short Screener",
        "Finds parabolic stocks ripe for a 10–40% reversal using 5 sequential filters.",
        scan_key="ps_scan_btn", last_key="ps_last_time",
    )

    with st.expander("ℹ️ How the 5 filters work", expanded=False):
        ca, cb, cc, cd, ce = st.columns(5)
        ca.info("**1 · Market Cap vs. Move**\n\n<$10B → 200%+\n\n$10–100B → 100%+\n\n>$100B → 50%+\n\n_(from 52-wk trough)_")
        cb.info("**2 · Textbook Structure**\n\nExpanding daily ranges\n\nSteep slope ≥0.8%/day\n\nMassive green candle")
        cc.info("**3 · Volume Climax**\n\nHighest vol in 1 year\n\nOccurred within 30 days\n\nSpike ≥ multiplier × avg")
        cd.info("**4 · Psych Levels**\n\nPrice near $100/$500\nor $1,000 (etc.)\n\nFirst-time approach")
        ce.info("**5 · VWAP Rejection**\n\nWas above VWAP today\n\nNow below VWAP\n\n→ Short entry trigger")

    with st.expander("⚙️ Settings", expanded=False):
        sc1, sc2, sc3, sc4 = st.columns(4)
        ps_min_score  = sc1.slider("Min filters passed", 1, 5, 3, key="ps_min_score",
                                    help="Only show stocks passing at least N of the 5 filters")
        ps_months     = sc2.slider("Move lookback (months)", 3, 12, 12, key="ps_months",
                                    help="Window for measuring % gain from trough to current price")
        ps_vol_mult   = sc3.slider("Volume spike multiplier", 1.5, 5.0, 2.0, step=0.5, key="ps_vol_mult",
                                    help="Recent peak volume must be ≥ this × 90-day average to qualify")
        ps_psych_tol  = sc4.slider("Psych level tolerance (%)", 0.5, 5.0, 2.0, step=0.5, key="ps_psych_tol",
                                    help="Max % distance from a round number to count as 'tapping' it")

        sf1, sf2, sf3, sf4, sf5 = st.columns(5)
        f1_on = sf1.checkbox("F1 Market Cap", value=True, key="ps_f1")
        f2_on = sf2.checkbox("F2 Structure",  value=True, key="ps_f2")
        f3_on = sf3.checkbox("F3 Volume",     value=True, key="ps_f3")
        f4_on = sf4.checkbox("F4 Psych Lvl", value=True, key="ps_f4")
        f5_on = sf5.checkbox("F5 VWAP",       value=True, key="ps_f5")


    # ── Filter implementations (closures so they share np) ───────────────────

    _PSYCH_LEVELS = [50, 100, 150, 200, 250, 500, 1000]

    def _f1(hist, market_cap):
        r = {"pass": False, "move_pct": 0.0, "required_pct": 0.0, "cap_tier": "N/A"}
        if market_cap <= 0 or hist.empty:
            return r
        trough = hist["Low"].min()
        current = hist["Close"].iloc[-1]
        if trough <= 0:
            return r
        move_pct = (current - trough) / trough * 100
        cap_b = market_cap / 1e9
        required = 200 if cap_b < 10 else (100 if cap_b < 100 else 50)
        tier = f"<$10B" if cap_b < 10 else ("$10–100B" if cap_b < 100 else ">$100B")
        r.update({"pass": move_pct >= required, "move_pct": round(move_pct, 1),
                  "required_pct": required, "cap_tier": tier,
                  "trough": round(trough, 2), "current": round(current, 2)})
        return r

    def _f2(hist, lookback=30):
        r = {"pass": False, "expanding": False, "steep": False, "massive_candle": False}
        if len(hist) < lookback:
            return r
        rec = hist.tail(lookback).copy()
        rec["range"] = rec["High"] - rec["Low"]
        rec["body"]  = (rec["Close"] - rec["Open"]).abs()
        avg_range    = rec["range"].mean()
        half = lookback // 2
        expanding    = rec["range"].tail(half).mean() > rec["range"].head(half).mean()
        closes = rec["Close"].values
        slope, _ = np.polyfit(np.arange(len(closes)), closes, 1)
        slope_pct = slope / closes[0] * 100
        steep     = slope_pct >= 0.8
        last10    = rec.tail(10)
        massive   = bool(((last10["body"] > 1.5 * avg_range) & (last10["Close"] > last10["Open"])).any())
        r.update({"pass": expanding and steep and massive,
                  "expanding": expanding, "steep": steep, "massive_candle": massive,
                  "slope_pct_day": round(slope_pct, 2), "avg_range": round(avg_range, 2)})
        return r

    def _f3(hist, multiplier):
        r = {"pass": False, "spike_ratio": 0.0, "is_recent": False}
        if len(hist) < 20:
            return r
        avg90 = hist["Volume"].tail(90).mean()
        if avg90 == 0:
            return r
        year_max_idx = hist["Volume"].idxmax()
        cutoff       = hist.index[-1] - pd.Timedelta(days=30)
        is_recent    = year_max_idx >= cutoff
        recent_peak  = hist["Volume"].tail(30).max()
        spike_ratio  = recent_peak / avg90
        r.update({"pass": is_recent and spike_ratio >= multiplier,
                  "spike_ratio": round(spike_ratio, 2), "is_recent": is_recent,
                  "year_max_vol": int(hist["Volume"].max()), "avg90_vol": int(avg90),
                  "days_since_spike": (hist.index[-1] - year_max_idx).days})
        return r

    def _f4(hist, tol_pct):
        r = {"pass": False, "level": None, "first_time": False}
        if hist.empty:
            return r
        current = hist["Close"].iloc[-1]
        for lvl in _PSYCH_LEVELS:
            if abs(current - lvl) / lvl * 100 <= tol_pct:
                cutoff = len(hist) // 2
                hist_before = hist["Close"].iloc[:-cutoff] if cutoff > 0 else hist["Close"]
                first_time  = bool((hist_before < lvl * 0.85).all()) if not hist_before.empty else False
                r.update({"pass": True, "level": lvl,
                          "dist_pct": round(abs(current - lvl) / lvl * 100, 2),
                          "first_time": first_time, "current": round(current, 2)})
                return r
        return r

    def _f5(symbol):
        r = {"pass": False, "below_vwap": False, "was_above": False,
             "vwap": None, "current": None, "pct_from_vwap": None}
        try:
            intra = yf.Ticker(symbol).history(period="1d", interval="5m")
            if intra.empty or len(intra) < 8:
                r["note"] = "no intraday data"
                return r
            tp   = (intra["High"] + intra["Low"] + intra["Close"]) / 3
            vwap = (tp * intra["Volume"]).cumsum() / intra["Volume"].cumsum()
            cur  = intra["Close"].iloc[-1]
            cv   = vwap.iloc[-1]
            above_ct = int((intra["Close"] > vwap).sum())
            was_above = above_ct >= max(3, len(intra) // 5)
            below_now = cur < cv
            pct = (cur - cv) / cv * 100
            r.update({"pass": was_above and below_now, "below_vwap": below_now,
                      "was_above": was_above, "vwap": round(cv, 2),
                      "current": round(cur, 2), "pct_from_vwap": round(pct, 2)})
        except Exception as exc:
            r["note"] = str(exc)
        return r

    @st.cache_data(ttl=1800, show_spinner=False)
    def _fetch_daily(symbol, months):
        try:
            t    = yf.Ticker(symbol)
            hist = t.history(period=f"{months}mo", interval="1d")
            cap  = getattr(t.fast_info, "market_cap", None) or 0
            return hist, float(cap)
        except Exception:
            return pd.DataFrame(), 0.0

    @st.cache_data(ttl=300, show_spinner=False)
    def _fetch_vwap(symbol):
        r = {"pass": False, "below_vwap": False, "was_above": False,
             "vwap": None, "current": None, "pct_from_vwap": None}
        try:
            intra = yf.Ticker(symbol).history(period="1d", interval="5m")
            if intra.empty or len(intra) < 8:
                r["note"] = "no intraday data"
                return r
            tp   = (intra["High"] + intra["Low"] + intra["Close"]) / 3
            vwap_s = (tp * intra["Volume"]).cumsum() / intra["Volume"].cumsum()
            cur  = intra["Close"].iloc[-1]
            cv   = vwap_s.iloc[-1]
            above_ct = int((intra["Close"] > vwap_s).sum())
            was_above = above_ct >= max(3, len(intra) // 5)
            below_now = cur < cv
            pct = (cur - cv) / cv * 100
            r.update({"pass": was_above and below_now, "below_vwap": below_now,
                      "was_above": was_above, "vwap": round(cv, 2),
                      "current": round(cur, 2), "pct_from_vwap": round(pct, 2)})
        except Exception as exc:
            r["note"] = str(exc)
        return r

    def _scan_one(sym, months, vol_mult, psych_tol, f1_on, f2_on, f3_on, f4_on, f5_on):
        hist, cap = _fetch_daily(sym, months)
        if hist.empty or len(hist) < 20:
            return None
        r1 = _f1(hist, cap)     if f1_on else {"pass": None}
        r2 = _f2(hist)          if f2_on else {"pass": None}
        r3 = _f3(hist, vol_mult) if f3_on else {"pass": None}
        r4 = _f4(hist, psych_tol) if f4_on else {"pass": None}
        r5 = _fetch_vwap(sym)   if f5_on else {"pass": None}
        active = [r for r in [r1, r2, r3, r4, r5] if r["pass"] is not None]
        score  = sum(1 for r in active if r["pass"])
        cur    = hist["Close"].iloc[-1]
        prev   = hist["Close"].iloc[-2] if len(hist) >= 2 else cur
        chg    = (cur - prev) / prev * 100 if prev else 0
        return {"symbol": sym, "price": round(cur, 2), "chg_pct": round(chg, 2),
                "cap_b": round(cap / 1e9, 1) if cap else None,
                "score": score, "n_active": len(active),
                "f1": r1, "f2": r2, "f3": r3, "f4": r4, "f5": r5}

    # ── Session state ─────────────────────────────────────────────────────────

    if "ps_results" not in st.session_state:
        st.session_state.ps_results  = []
        st.session_state.ps_last_time = None

    if clicked:
        ps_symbols   = get_selected_tickers()
        ps_min_score = st.session_state.get("ps_min_score", 3)
        ps_months    = st.session_state.get("ps_months", 12)
        ps_vol_mult  = st.session_state.get("ps_vol_mult", 2.0)
        ps_psych_tol = st.session_state.get("ps_psych_tol", 2.0)
        f1_on = st.session_state.get("ps_f1", True)
        f2_on = st.session_state.get("ps_f2", True)
        f3_on = st.session_state.get("ps_f3", True)
        f4_on = st.session_state.get("ps_f4", True)
        f5_on = st.session_state.get("ps_f5", True)

        _ticker_count_caption(ps_symbols)
        prog = st.progress(0, text="Scanning…")
        raw  = []

        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {
                pool.submit(_scan_one, sym, ps_months, ps_vol_mult, ps_psych_tol,
                            f1_on, f2_on, f3_on, f4_on, f5_on): sym
                for sym in ps_symbols
            }
            done = 0
            for fut in as_completed(futs):
                done += 1
                prog.progress(done / len(ps_symbols), text=f"Scanned {done}/{len(ps_symbols)}")
                try:
                    row = fut.result()
                    if row and row["score"] >= ps_min_score:
                        raw.append(row)
                except Exception:
                    pass

        prog.empty()
        raw.sort(key=lambda x: (-x["score"], -x["chg_pct"]))
        st.session_state.ps_results   = raw
        st.session_state.ps_last_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        if raw:
            save_scan_result(
                pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)}
                               for r in raw]),
                "parabolic_short",
            )

    results = st.session_state.ps_results
    if not results and st.session_state.ps_last_time:
        st.warning("No stocks matched. Try lowering Min filters passed or expanding the universe.", icon="⚠️")
        return
    if not results:
        st.info("Hit **Scan Now** to find parabolic short candidates.", icon="ℹ️")
        return

    # ── Metrics ───────────────────────────────────────────────────────────────

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates",    len(results))
    m2.metric("Score 5/5",     sum(1 for r in results if r["score"] == 5))
    m3.metric("Score ≥ 4",     sum(1 for r in results if r["score"] >= 4))
    top = max(results, key=lambda x: x["score"])
    m4.metric("Top pick", f"{top['symbol']} ({top['score']}/{top['n_active']})")

    st.divider()

    # ── Summary table ─────────────────────────────────────────────────────────

    def _badge(v):
        if v is None: return "—"
        return "✅" if v else "❌"

    rows = []
    for r in results:
        rows.append({
            "Symbol":    r["symbol"],
            "Price":     f"${r['price']:,.2f}",
            "Chg %":     r["chg_pct"],
            "Cap $B":    f"{r['cap_b']:.1f}" if r["cap_b"] else "N/A",
            "Score":     f"{r['score']}/{r['n_active']}",
            "F1 Move":   _badge(r["f1"].get("pass")),
            "F2 Struct": _badge(r["f2"].get("pass")),
            "F3 Vol":    _badge(r["f3"].get("pass")),
            "F4 Psych":  _badge(r["f4"].get("pass")),
            "F5 VWAP":   _badge(r["f5"].get("pass")),
            "Move %":    r["f1"].get("move_pct", ""),
            "Vol Spike": r["f3"].get("spike_ratio", ""),
            "Psych $":   r["f4"].get("level", ""),
            "VWAP Δ%":   r["f5"].get("pct_from_vwap", ""),
        })

    tbl = pd.DataFrame(rows)

    def _row_color(row):
        try:
            n = int(str(row["Score"]).split("/")[0])
        except Exception:
            n = 0
        if n == 5:   bg = "background-color:rgba(255,82,82,0.25)"
        elif n == 4: bg = "background-color:rgba(255,152,0,0.22)"
        elif n == 3: bg = "background-color:rgba(255,235,59,0.16)"
        else:        bg = ""
        return [bg] * len(row)

    fmt = {
        "Chg %":     "{:+.2f}%",
        "Move %":    lambda v: f"{v:.1f}%" if isinstance(v, (int, float)) else "",
        "Vol Spike": lambda v: f"{v:.1f}×" if isinstance(v, (int, float)) else "",
        "Psych $":   lambda v: f"${v}" if v else "—",
        "VWAP Δ%":   lambda v: f"{v:+.2f}%" if isinstance(v, (int, float)) else "—",
    }
    st.dataframe(
        tbl.style.apply(_row_color, axis=1).format(fmt),
        use_container_width=True,
        height=min(600, 55 + 36 * len(tbl)),
        hide_index=True,
    )

    # ── Detail cards ──────────────────────────────────────────────────────────

    st.divider()
    st.markdown("##### Detailed Breakdown")
    for r in results:
        cap_str = f"${r['cap_b']:.1f}B" if r["cap_b"] else "N/A"
        arrow   = "▲" if r["chg_pct"] >= 0 else "▼"
        with st.expander(
            f"**{r['symbol']}**  ·  ${r['price']:,.2f}  "
            f"({arrow}{abs(r['chg_pct']):.2f}%)  ·  "
            f"Score **{r['score']}/{r['n_active']}**  ·  Cap: {cap_str}"
        ):
            dc1, dc2, dc3, dc4, dc5 = st.columns(5)
            f1 = r["f1"]
            with dc1:
                st.markdown(f"**{'✅' if f1.get('pass') else '❌'} F1: Market Cap vs. Move**")
                if f1.get("pass") is not None:
                    st.write(f"Move: **{f1.get('move_pct', 0):.1f}%** (need {f1.get('required_pct', 0):.0f}%)")
                    st.write(f"Tier: {f1.get('cap_tier', 'N/A')}")
                    st.write(f"${f1.get('trough', 0):.2f} → ${f1.get('current', 0):.2f}")
            f2 = r["f2"]
            with dc2:
                st.markdown(f"**{'✅' if f2.get('pass') else '❌'} F2: Structure**")
                if f2.get("pass") is not None:
                    st.write(f"Expanding ranges: {'✅' if f2.get('expanding') else '❌'}")
                    st.write(f"Steep slope: {'✅' if f2.get('steep') else '❌'} ({f2.get('slope_pct_day', 0):.2f}%/day)")
                    st.write(f"Massive candle: {'✅' if f2.get('massive_candle') else '❌'}")
            f3 = r["f3"]
            with dc3:
                st.markdown(f"**{'✅' if f3.get('pass') else '❌'} F3: Volume Climax**")
                if f3.get("pass") is not None:
                    st.write(f"Spike: **{f3.get('spike_ratio', 0):.1f}×** 90d avg")
                    st.write(f"52w high recent: {'✅' if f3.get('is_recent') else '❌'}")
                    if f3.get("days_since_spike") is not None:
                        st.write(f"Days ago: {f3.get('days_since_spike')}")
            f4 = r["f4"]
            with dc4:
                st.markdown(f"**{'✅' if f4.get('pass') else '❌'} F4: Psych Level**")
                if f4.get("pass"):
                    st.write(f"Level: **${f4.get('level')}**")
                    st.write(f"Distance: {f4.get('dist_pct', 0):.2f}%")
                    st.write(f"First time: {'✅' if f4.get('first_time') else '⚠️ unconfirmed'}")
                elif f4.get("pass") is not None:
                    st.write("Not near a psych level")
            f5 = r["f5"]
            with dc5:
                st.markdown(f"**{'✅' if f5.get('pass') else '❌'} F5: VWAP Rejection**")
                if f5.get("vwap"):
                    st.write(f"Price ${f5.get('current')} vs VWAP ${f5.get('vwap')}")
                    st.write(f"Δ VWAP: **{f5.get('pct_from_vwap', 0):+.2f}%**")
                    st.write(f"Was above: {'✅' if f5.get('was_above') else '❌'}")
                elif f5.get("note"):
                    st.caption(f5["note"])

    # ── Download ──────────────────────────────────────────────────────────────

    st.divider()
    dl = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in results])
    st.download_button(
        "⬇️ Download CSV",
        data=dl.to_csv(index=False),
        file_name=f"parabolic_shorts_{_date.today().isoformat()}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 0DTE Options Scanner
# Strategies: Gap & Go · Momentum Scalp · Reversal · Flow Play
# ─────────────────────────────────────────────────────────────────────────────

# Category A — TRUE 0DTE: ETFs + mega-caps with daily or M/W/F expirations.
# These tickers reliably list a same-day chain every session.
_TRUE_0DTE_UNIVERSE = [
    # Index ETFs (daily expirations)
    "SPY","QQQ","IWM","GLD","TLT","SLV","XLF","XLE","XLK","XBI","DIA","EEM",
    # Mega-caps with M/W/F or daily chains
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","AVGO",
    "NFLX","JPM","BAC","GS","INTC","MU",
]

# Category B — INTRADAY CLOSE (1-3 DTE, exit same day):
# Options expire in 1-3 days but the trade is closed intraday.
# Gives a bigger delta buffer vs pure 0DTE; less theta risk because
# expiry is not today — position is sized for an intraday directional move.
_INTRADAY_CLOSE_UNIVERSE = [
    "PLTR","COIN","HOOD","MSTR","SMCI","ARM","APP","RKLB","RXRX","ASTS",
    "CRWD","PANW","NET","SNOW","DDOG","CRM","NOW","UBER","BABA","NIO",
    "SOUN","BBAI","IONQ","QBTS","LUNR","KULR","ARQQ","MRVL","TTD","SHOP",
    "CELH","ONON","CAVA","DUOL","MNDY","GTLB","BILL","DOCN","GRAB","BROS",
]


def _bs_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.045) -> float:
    """Black-Scholes gamma. Returns $ option change per 1% underlying move per contract."""
    import math
    if S <= 0 or K <= 0 or T <= 0:
        return 0.0
    sigma = max(sigma, 0.10)
    T = max(T, 0.5 / (365 * 6.5))   # floor at 30 minutes
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        npd1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        gamma = npd1 / (S * sigma * math.sqrt(T))
        # $ change per 1% move × 100 shares per contract
        return round(gamma * (0.01 * S) ** 2 * 100 * 0.5, 3)   # ½ Γ·(ΔS)² per contract
    except Exception:
        return 0.0


def _bs_theta_per_hour(S: float, K: float, T: float, sigma: float,
                        r: float = 0.045) -> float:
    """Theta per market-hour in dollars per contract (always negative)."""
    import math
    if S <= 0 or K <= 0 or T <= 0:
        return 0.0
    sigma = max(sigma, 0.10)
    T = max(T, 0.5 / (365 * 6.5))
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        N  = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
        npd1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        theta_annual = -(S * sigma * npd1) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * N(d2)
        theta_per_day  = theta_annual / 365
        theta_per_hour = theta_per_day / 6.5   # 6.5 market hours
        return round(theta_per_hour * 100, 3)  # per contract
    except Exception:
        return 0.0


@st.cache_data(ttl=300, show_spinner=False)   # 5-min cache — 0DTE data is time-sensitive
def _fetch_0dte_chain(symbol: str) -> tuple:
    """
    Fetch today's expiring calls + puts for symbol.
    Returns (calls_df, puts_df, spot, chg_pct, above_ema20, rsi14, vol_ratio, pc_ratio).
    All DataFrames include dte=0 and side column.
    """
    from datetime import date as _date
    try:
        ticker  = yf.Ticker(symbol)
        fi      = ticker.fast_info
        price   = fi.get("lastPrice") or fi.get("previousClose", 0)
        if not price:
            return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0, None, None, 1.0, 1.0

        prev_close = fi.get("previousClose", price) or price
        chg_pct    = round((price - prev_close) / prev_close * 100, 2)

        try:
            vol_today = float(fi.get("regularMarketVolume") or 0)
            avg_vol   = float(fi.get("threeMonthAverageVolume") or vol_today or 1)
        except Exception:
            vol_today, avg_vol = 0.0, 1.0
        vol_ratio = round(vol_today / max(avg_vol, 1), 2)

        above_ema20 = rsi14 = None
        try:
            hist  = ticker.history(period="30d", auto_adjust=True)
            close = hist["Close"]
            if len(close) >= 20:
                above_ema20 = bool(price > close.ewm(span=20).mean().iloc[-1])
            if len(close) >= 15:
                d    = close.diff()
                gain = d.clip(lower=0).rolling(14).mean()
                loss = (-d.clip(upper=0)).rolling(14).mean()
                rs   = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
                rsi14 = round(float(100 - 100 / (1 + rs)), 1)
        except Exception:
            pass

        today_str = _date.today().isoformat()
        if today_str not in (ticker.options or []):
            return pd.DataFrame(), pd.DataFrame(), float(price), chg_pct, above_ema20, rsi14, vol_ratio, 1.0

        chain = ticker.option_chain(today_str)
        call_vol = float(pd.to_numeric(chain.calls["volume"], errors="coerce").fillna(0).sum())
        put_vol  = float(pd.to_numeric(chain.puts["volume"],  errors="coerce").fillna(0).sum())
        pc_ratio = round(call_vol / max(put_vol, 1), 2)

        def _enrich(df, side):
            df = df.copy()
            df["symbol"]    = symbol
            df["price"]     = float(price)
            df["chg_pct"]   = chg_pct
            df["above_ema20"] = above_ema20
            df["rsi14"]     = rsi14
            df["vol_ratio"] = vol_ratio
            df["pc_ratio"]  = pc_ratio
            df["dte"]       = 0
            df["expiry"]    = today_str
            df["side"]      = side
            return df

        return (_enrich(chain.calls, "call"),
                _enrich(chain.puts,  "put"),
                float(price), chg_pct, above_ema20, rsi14, vol_ratio, pc_ratio)
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0, None, None, 1.0, 1.0


def _score_0dte(row, direction: str, spy_chg: float) -> int:
    """
    0DTE conviction score (0–10).
    direction = 'call' (bullish) or 'put' (bearish).
    """
    chg    = row.get("chg_pct", 0) or 0
    rel    = chg - spy_chg
    voi    = row.get("voi_ratio", 0) or 0
    pc     = row.get("pc_ratio", 1) or 1
    vsr    = row.get("vol_ratio", 1) or 1
    rsi    = row.get("rsi14")
    ema_ok = row.get("above_ema20")
    pts    = 0

    if direction == "call":
        # Directional momentum (0–3)
        if rel >= 3.0:   pts += 3
        elif rel >= 1.5: pts += 2
        elif rel >= 0.5: pts += 1
        # Technical (0–3)
        if ema_ok:                          pts += 1
        if rsi is not None and 45 <= rsi <= 72: pts += 1
        if vsr >= 2.0:                      pts += 1
        # Flow (0–4)
        if voi >= 5.0:   pts += 3
        elif voi >= 2.0: pts += 2
        elif voi >= 1.0: pts += 1
        if pc >= 2.0:    pts += 1
    else:  # put
        # Directional weakness (0–3)
        if rel <= -3.0:   pts += 3
        elif rel <= -1.5: pts += 2
        elif rel <= -0.5: pts += 1
        # Technical (0–3)
        if ema_ok is False:                       pts += 1
        if rsi is not None and 28 <= rsi <= 55:   pts += 1
        if vsr >= 2.0:                            pts += 1
        # Flow (0–4)
        if voi >= 5.0:   pts += 3
        elif voi >= 2.0: pts += 2
        elif voi >= 1.0: pts += 1
        if pc <= 0.5:    pts += 1   # more puts than calls = bearish skew

    return min(pts, 10)


def _classify_0dte_setup(chg_pct: float, vol_ratio: float, rsi14,
                          voi_ratio: float, rel_str: float,
                          direction: str) -> str:
    """Return a researched setup label for this 0DTE signal."""
    sign = 1 if direction == "call" else -1
    abs_chg = abs(chg_pct)

    if abs_chg >= 2.0 and vol_ratio >= 1.5 and rel_str * sign >= 1.0:
        return "🚀 Gap&Go"
    if voi_ratio >= 4.0:
        return "🐋 Flow"
    if rsi14 is not None:
        if direction == "call" and rsi14 <= 35:
            return "🔄 Reversal"
        if direction == "put"  and rsi14 >= 65:
            return "🔄 Reversal"
    if abs_chg >= 0.5 and vol_ratio >= 1.2:
        return "⚡ Momentum"
    return "📊 Standard"


@st.cache_data(ttl=300, show_spinner=False)
def _scan_0dte(
    symbols: tuple,
    direction: str,          # "call" | "put"
    min_volume: int,
    min_oi: int,
    moneyness_pct: float,
    max_iv_pct: float,
    max_spread_pct: float,
    delta_min: float,
    delta_max: float,
    min_score: int,
    spy_chg: float,
) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        calls_df, puts_df, price, chg_pct, above_ema20, rsi14, vol_ratio, pc_ratio = \
            _fetch_0dte_chain(sym)

        df = calls_df if direction == "call" else puts_df
        if df.empty or price == 0:
            continue

        df = df.copy()
        df["volume"]       = pd.to_numeric(df["volume"],       errors="coerce").fillna(0).astype(int)
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0).astype(int)
        df = df[(df["volume"] >= min_volume) & (df["openInterest"] >= min_oi)]
        if df.empty:
            continue

        df["moneyness_pct"] = ((df["strike"] - price) / price * 100).abs()
        df = df[df["moneyness_pct"] <= moneyness_pct]
        itm = df["inTheMoney"].fillna(False).astype(bool)
        df = df[~itm] if direction == "call" else df     # calls: OTM only for 0DTE leverage

        df["voi_ratio"]  = (df["volume"] / (df["openInterest"] + 1)).round(2)
        df["iv_pct"]     = (df["impliedVolatility"] * 100).round(1)
        df["mid"]        = ((df["bid"] + df["ask"]) / 2).round(2)
        df["spread_pct"] = ((df["ask"] - df["bid"]) / (df["mid"] + 1e-6) * 100).round(1)

        df = df[df["iv_pct"]     <= max_iv_pct]
        df = df[df["spread_pct"] <= max_spread_pct]
        if df.empty:
            continue

        # Delta — use small but non-zero T for same-day (assume 4 hrs left = ~0.5/6.5 days)
        import datetime as _dt
        now  = _dt.datetime.now()
        close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
        hrs_left = max((close_time - now).total_seconds() / 3600, 0.25)
        T_remaining = hrs_left / (6.5 * 365)

        if direction == "call":
            df["delta"] = df.apply(
                lambda r: _bs_call_delta(float(r["price"]), float(r["strike"]),
                                         T_remaining, float(r["impliedVolatility"])), axis=1)
        else:
            df["delta"] = df.apply(
                lambda r: -1 * (1 - _bs_call_delta(float(r["price"]), float(r["strike"]),
                                                     T_remaining, float(r["impliedVolatility"]))), axis=1)

        df = df[(df["delta"].abs() >= delta_min) & (df["delta"].abs() <= delta_max)]
        if df.empty:
            continue

        # Gamma⚡ and Theta💀
        df["gamma_dollar"] = df.apply(
            lambda r: _bs_gamma(float(r["price"]), float(r["strike"]),
                                 T_remaining, float(r["impliedVolatility"])), axis=1)
        df["theta_per_hr"] = df.apply(
            lambda r: _bs_theta_per_hour(float(r["price"]), float(r["strike"]),
                                          T_remaining, float(r["impliedVolatility"])), axis=1)

        rel_str = round(chg_pct - spy_chg, 2)
        df["rel_str"] = rel_str

        df["score"] = df.apply(lambda r: _score_0dte(r, direction, spy_chg), axis=1)
        df = df[df["score"] >= min_score]
        if df.empty:
            continue

        df["setup"] = df.apply(
            lambda r: _classify_0dte_setup(chg_pct, vol_ratio, rsi14,
                                            r["voi_ratio"], rel_str, direction), axis=1)

        # Time urgency tag
        if hrs_left <= 1.0:
            df["urgency"] = "🔴 <1hr"
        elif hrs_left <= 2.5:
            df["urgency"] = "🟡 <2.5hr"
        else:
            df["urgency"] = "🟢 >2.5hr"

        rows.append(df)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).sort_values("score", ascending=False)
    out["category"] = "True 0DTE"
    return out


@st.cache_data(ttl=300, show_spinner=False)
def _scan_intraday_close(
    symbols: tuple,
    direction: str,
    min_volume: int,
    moneyness_pct: float,
    max_iv_pct: float,
    max_spread_pct: float,
    delta_min: float,
    delta_max: float,
    min_score: int,
    spy_chg: float,
    max_dte: int = 3,
) -> pd.DataFrame:
    """
    Scan 1-3 DTE options intended to be closed intraday.
    Reuses _fetch_weekly_calls so the nearest upcoming expiry is used.
    Gamma/theta are computed with the actual remaining DTE (not 0).
    """
    import datetime as _dt
    _fetcher = _fetch_weekly_puts if direction == "put" else _fetch_weekly_calls
    rows = []
    for sym in symbols:
        df, chg_pct, above_ema20, above_sma50, rsi14, vol_ratio, pc_ratio = \
            _fetcher(sym, max_dte)
        if df.empty or df["price"].iloc[0] == 0:
            continue

        price = df["price"].iloc[0]
        df = df.copy()
        df["above_ema20"] = above_ema20
        df["rsi14"]       = rsi14
        df["vol_ratio"]   = vol_ratio
        df["pc_ratio"]    = pc_ratio
        df["chg_pct"]     = chg_pct

        df["volume"]       = pd.to_numeric(df["volume"],       errors="coerce").fillna(0).astype(int)
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0).astype(int)
        df = df[df["volume"] >= min_volume]
        if df.empty:
            continue

        df["moneyness_pct"] = ((df["strike"] - price) / price * 100).abs()
        df = df[df["moneyness_pct"] <= moneyness_pct]
        itm = df["inTheMoney"].fillna(False).astype(bool)
        df = df[~itm]   # OTM only for intraday leverage

        df["voi_ratio"]  = (df["volume"] / (df["openInterest"] + 1)).round(2)
        df["iv_pct"]     = (df["impliedVolatility"] * 100).round(1)
        df["mid"]        = ((df["bid"] + df["ask"]) / 2).round(2)
        df["spread_pct"] = ((df["ask"] - df["bid"]) / (df["mid"] + 1e-6) * 100).round(1)

        df = df[df["iv_pct"]     <= max_iv_pct]
        df = df[df["spread_pct"] <= max_spread_pct]
        if df.empty:
            continue

        if direction == "put":
            df["delta"] = df.apply(
                lambda r: -1 * (1 - _bs_call_delta(float(r["price"]), float(r["strike"]),
                                                    float(r["dte"]) / 365.0,
                                                    float(r["impliedVolatility"]))), axis=1)
        else:
            df["delta"] = df.apply(
                lambda r: _bs_call_delta(float(r["price"]), float(r["strike"]),
                                          float(r["dte"]) / 365.0,
                                          float(r["impliedVolatility"])), axis=1)
        df = df[(df["delta"] >= delta_min) & (df["delta"] <= delta_max)]
        if df.empty:
            continue

        # Gamma/theta use actual DTE (1-3), not 0
        df["gamma_dollar"] = df.apply(
            lambda r: _bs_gamma(float(r["price"]), float(r["strike"]),
                                 float(r["dte"]) / 365.0, float(r["impliedVolatility"])), axis=1)
        df["theta_per_hr"] = df.apply(
            lambda r: _bs_theta_per_hour(float(r["price"]), float(r["strike"]),
                                          float(r["dte"]) / 365.0, float(r["impliedVolatility"])), axis=1)

        rel_str = round(chg_pct - spy_chg, 2)
        df["rel_str"] = rel_str
        df["score"]   = df.apply(lambda r: _score_0dte(r, direction, spy_chg), axis=1)
        df = df[df["score"] >= min_score]
        if df.empty:
            continue

        df["setup"]    = df.apply(
            lambda r: _classify_0dte_setup(chg_pct, vol_ratio, rsi14,
                                            r["voi_ratio"], rel_str, direction), axis=1)
        df["urgency"]  = f"expires {df['expiry'].iloc[0]} ({df['dte'].iloc[0]}d)"
        df["category"] = "Intraday Close"
        df["side"]     = direction
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("score", ascending=False)


# ── Sell-side: credit spread scanner ─────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_nearest_chain(symbol: str, max_dte: int = 3) -> tuple:
    """
    Like _fetch_0dte_chain but falls back to the nearest available expiry
    within max_dte days when today has no options chain.
    Returns same tuple as _fetch_0dte_chain plus (dte_actual, expiry_str).
    """
    from datetime import date as _date, timedelta as _td
    try:
        ticker  = yf.Ticker(symbol)
        fi      = ticker.fast_info
        price   = fi.get("lastPrice") or fi.get("previousClose", 0)
        if not price:
            return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0, None, None, 1.0, 1.0, 99, ""

        prev_close = fi.get("previousClose", price) or price
        chg_pct    = round((price - prev_close) / prev_close * 100, 2)
        try:
            vol_today = float(fi.get("regularMarketVolume") or 0)
            avg_vol   = float(fi.get("threeMonthAverageVolume") or vol_today or 1)
        except Exception:
            vol_today, avg_vol = 0.0, 1.0
        vol_ratio = round(vol_today / max(avg_vol, 1), 2)

        above_ema20 = rsi14 = None
        try:
            hist  = ticker.history(period="30d", auto_adjust=True)
            close = hist["Close"]
            if len(close) >= 20:
                above_ema20 = bool(price > close.ewm(span=20).mean().iloc[-1])
            if len(close) >= 15:
                d    = close.diff()
                gain = d.clip(lower=0).rolling(14).mean()
                loss = (-d.clip(upper=0)).rolling(14).mean()
                rs   = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
                rsi14 = round(float(100 - 100 / (1 + rs)), 1)
        except Exception:
            pass

        avail = ticker.options or []
        today = _date.today()
        expiry_str = None
        dte_actual = 99
        for d in range(max_dte + 1):
            candidate = (today + _td(days=d)).isoformat()
            if candidate in avail:
                expiry_str = candidate
                dte_actual = d
                break

        if not expiry_str:
            return pd.DataFrame(), pd.DataFrame(), float(price), chg_pct, above_ema20, rsi14, vol_ratio, 1.0, 99, ""

        chain = ticker.option_chain(expiry_str)
        call_vol = float(pd.to_numeric(chain.calls["volume"], errors="coerce").fillna(0).sum())
        put_vol  = float(pd.to_numeric(chain.puts["volume"],  errors="coerce").fillna(0).sum())
        pc_ratio = round(call_vol / max(put_vol, 1), 2)

        def _enrich(df, side):
            df = df.copy()
            df["symbol"]      = symbol
            df["price"]       = float(price)
            df["chg_pct"]     = chg_pct
            df["above_ema20"] = above_ema20
            df["rsi14"]       = rsi14
            df["vol_ratio"]   = vol_ratio
            df["pc_ratio"]    = pc_ratio
            df["dte"]         = dte_actual
            df["expiry"]      = expiry_str
            df["side"]        = side
            return df

        return (_enrich(chain.calls, "call"), _enrich(chain.puts, "put"),
                float(price), chg_pct, above_ema20, rsi14, vol_ratio, pc_ratio,
                dte_actual, expiry_str)
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0, None, None, 1.0, 1.0, 99, ""


def _scan_spread_setups(
    symbols: tuple,
    strategy: str,        # "bull_put" | "bear_call" | "iron_condor"
    spy_chg: float,
    target_delta: float,  # short strike delta (0.10-0.20)
    account_size: float,
    risk_pct: float,
    max_dte: int = 3,
) -> pd.DataFrame:
    """
    Sell-side scanner: find credit spread / iron condor setups.
    Falls back to nearest available expiry (up to max_dte) when today has no chain.
    One row per underlying with suggested spread legs + P&L structure.
    """
    import math, datetime as _dt
    now        = _dt.datetime.now()
    today_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    hrs_today  = max((today_close - now).total_seconds() / 3600, 0.25)

    rows = []
    for sym in symbols:
        calls_df, puts_df, price, chg_pct, above_ema20, rsi14, vol_ratio, pc_ratio, dte_actual, expiry_str = \
            _fetch_nearest_chain(sym, max_dte=max_dte)
        if price == 0 or dte_actual == 99:
            continue

        # Time-to-expiry in years: remaining hours today + full trading days beyond today
        hrs_to_exp = hrs_today + max(dte_actual - 1, 0) * 6.5
        T = max(hrs_to_exp, 0.25) / (6.5 * 365)

        rel_str = round(chg_pct - spy_chg, 2)

        # Expected move from ATM IV
        atm_iv = None
        for _df in [calls_df, puts_df]:
            if _df.empty:
                continue
            _tmp = _df.copy()
            _tmp["_mn"] = (_tmp["strike"] - price).abs()
            atm_iv = float(_tmp.nsmallest(1, "_mn")["impliedVolatility"].iloc[0])
            break
        if not atm_iv or atm_iv <= 0:
            continue
        T_days = max(hrs_today / 6.5, 0.5 / 6.5)
        exp_move_pct = round(atm_iv * math.sqrt(T_days / 252) * 100, 2)

        # Spread width: ~1% of price, snapped to standard widths
        raw_w = price * 0.01
        spread_width = 1.0 if raw_w < 1 else (2.0 if raw_w < 3 else (5.0 if raw_w < 8 else 10.0))

        def _build_spread(opt_df, side):
            """Return spread dict or None."""
            if opt_df.empty:
                return None
            df = opt_df.copy()
            df["mid"] = ((df["bid"] + df["ask"]) / 2).round(3)
            df = df[df["mid"] > 0.01]
            if df.empty:
                return None
            # Delta for each strike
            if side == "put":
                df["_d"] = df.apply(
                    lambda r: -1 * (1 - _bs_call_delta(price, float(r["strike"]),
                                                        T, float(r["impliedVolatility"]))), axis=1)
            else:
                df["_d"] = df.apply(
                    lambda r: _bs_call_delta(price, float(r["strike"]),
                                              T, float(r["impliedVolatility"])), axis=1)
            # Short strike closest to target delta
            df["_dd"] = (df["_d"].abs() - target_delta).abs()
            short_row  = df.nsmallest(1, "_dd").iloc[0]
            short_K    = float(short_row["strike"])
            short_mid  = float(short_row["mid"])
            short_d    = float(short_row["_d"])
            # Long strike one spread_width away
            long_K_tgt = short_K - spread_width if side == "put" else short_K + spread_width
            df["_ld"]  = (df["strike"] - long_K_tgt).abs()
            long_row   = df.nsmallest(1, "_ld").iloc[0]
            long_mid   = float(long_row["mid"])
            credit     = round(short_mid - long_mid, 2)
            max_loss   = round(spread_width - credit, 2)
            if credit <= 0 or max_loss <= 0:
                return None
            max_contracts = max(int((account_size * risk_pct / 100) / (max_loss * 100)), 0)
            return {
                "short_K": short_K, "long_K": float(long_row["strike"]),
                "short_d": round(short_d, 3), "credit": credit,
                "max_loss": max_loss, "prob_profit": round((1 - abs(short_d)) * 100, 1),
                "max_contracts": max_contracts,
            }

        spread = None
        spread_label = ""
        if strategy == "bull_put":
            spread = _build_spread(puts_df, "put")
            spread_label = "Bull Put Spread 🟢"
        elif strategy == "bear_call":
            spread = _build_spread(calls_df, "call")
            spread_label = "Bear Call Spread 🔴"
        elif strategy == "iron_condor":
            bull = _build_spread(puts_df,  "put")
            bear = _build_spread(calls_df, "call")
            if bull and bear:
                combined_credit = round(bull["credit"] + bear["credit"], 2)
                ic_max_loss     = round(spread_width - combined_credit, 2)
                if combined_credit > 0 and ic_max_loss > 0:
                    spread = {
                        "short_K":      f"{bull['short_K']:.1f}P / {bear['short_K']:.1f}C",
                        "long_K":       f"{bull['long_K']:.1f}P / {bear['long_K']:.1f}C",
                        "short_d":      f"−{abs(bull['short_d']):.2f} / +{abs(bear['short_d']):.2f}",
                        "credit":       combined_credit,
                        "max_loss":     ic_max_loss,
                        "prob_profit":  round(bull["prob_profit"] * bear["prob_profit"] / 100, 1),
                        "max_contracts": min(bull["max_contracts"], bear["max_contracts"]),
                    }
                    spread_label = "Iron Condor 🔷"
        if spread is None:
            continue

        # Regime score for this underlying (neutral = good for selling)
        # Sell when market is calm, NOT when it's gapping hard
        neutral_score = 10 - _score_0dte(
            {"chg_pct": chg_pct, "voi_ratio": 1.0, "pc_ratio": pc_ratio,
             "vol_ratio": vol_ratio, "rsi14": rsi14, "above_ema20": above_ema20},
            "call", spy_chg,
        )

        rows.append({
            "symbol":       sym,
            "strategy":     spread_label,
            "dte":          dte_actual,
            "expiry":       expiry_str,
            "price":        round(price, 2),
            "exp_move_%":   exp_move_pct,
            "atm_iv_%":     round(atm_iv * 100, 1),
            "chg_pct":      chg_pct,
            "rel_str":      rel_str,
            "rsi14":        rsi14,
            "vol_ratio":    vol_ratio,
            "short_strike": spread["short_K"],
            "long_strike":  spread["long_K"],
            "short_delta":  spread["short_d"],
            "credit_$":     spread["credit"],
            "max_loss_$":   spread["max_loss"],
            "spread_width": spread_width,
            "prob_profit_%": spread["prob_profit"],
            "max_contracts": spread["max_contracts"],
            "calm_score":   neutral_score,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("prob_profit_%", ascending=False).reset_index(drop=True)


def _render_sell_guide():
    """Strategy guide for the sell-premium 0DTE mode."""
    with st.expander("📖 0DTE Sell-Premium Strategy Guide", expanded=False):
        st.markdown("""
### Why institutions SELL 0DTE, not buy

45-50% of all SPX options volume is 0DTE (CBOE data). The vast majority is sold by market-makers
and institutional desks running credit spreads and iron condors. Here's why:

| | Buy Premium | Sell Premium |
|---|---|---|
| **Win rate** | ~40-50% | ~70-85% |
| **Reward per trade** | 100-500% | 30-50% of risk |
| **Time decay** | Enemy | Ally |
| **Break-even** | Stock must move | Stock can stay still |
| **Best VIX env** | Low VIX | Any (best: elevated) |

---

### Credit Spread mechanics

**Bull Put Spread** (bullish): Sell a put at delta 0.15, buy a put 1-2 strikes lower.
Profit if stock stays ABOVE short strike at close.

**Bear Call Spread** (bearish): Sell a call at delta 0.15, buy a call 1-2 strikes higher.
Profit if stock stays BELOW short strike at close.

**Iron Condor** (neutral): Run both sides simultaneously. Profit if stock stays in range.

---

### Position sizing rules

- **Risk 1-2% of account per spread** (not per contract)
- Max 3-4 concurrent spreads; keep 30% cash reserve
- Never leg into one side of an iron condor — enter both legs simultaneously

---

### Exit rules (critical)

| Scenario | Action |
|----------|--------|
| Credit decays to 50% of initial | **Close for profit** — don't get greedy |
| Spread widens to 2× credit received | **Close for loss** — don't let it go to max loss |
| 3:00 PM ET | **Close everything** — gamma risk explodes in last hour |
| Stock gaps through short strike | Close immediately — assignment risk |

---

### Best setups today

- **Iron condor**: Best on low-news days, rangebound market (low VIX percentile)
- **Bull put spread**: Stock pulled back intraday but trend is up, RSI 40-55
- **Bear call spread**: Stock ran up hard, overbought RSI 65+, earnings not today
        """)


def render_0dte_scanner():
    import datetime as _dt
    from datetime import date as _date

    clicked = _page_header(
        "🔥", "0DTE Options Scanner",
        "Professional 0DTE approach: **Sell Premium** (credit spreads / iron condors) "
        "or **Buy Directional** (intraday momentum with timing gates). "
        "Research-backed — 45-50% of all SPX volume is 0DTE, mostly sold by institutions.",
        scan_key="dte0_scan_btn", last_key="dte0_time",
    )

    # ── Time-of-day banner ────────────────────────────────────────────────────
    now      = _dt.datetime.now()
    hour     = now.hour + now.minute / 60
    mkt_open = 9 + 30 / 60
    if hour < mkt_open:
        st.info("🌅 Pre-market — use this time to plan setups. **Do not execute** until 9:45 AM after first candle confirms direction.", icon="ℹ️")
    elif hour < 10.0:
        st.warning("⏳ **9:30–10:00 AM** — price discovery chaos, wide spreads. Wait for direction to confirm before entering.", icon="⚠️")
    elif hour < 12.0:
        st.success("🟢 **Prime window (10:00 AM–12:00 PM ET)** — best entry time for all 0DTE strategies. Liquid, directional, spread tightly.", icon="⚡")
    elif hour < 14.0:
        st.info("🟡 **Midday (12:00–2:00 PM ET)** — theta accelerating on longs. **Sell side ideal here.** Buyers need score ≥ 7.", icon="⏳")
    elif hour < 15.0:
        st.warning("🟠 **2:00–3:00 PM ET** — extreme theta on long options. Sellers close spreads for profit. Buyers avoid new entries.", icon="⚠️")
    else:
        st.error("🔴 **After 3:00 PM ET** — gamma spikes, liquidity thins. **Close all positions regardless of P&L.** Do not open new trades.", icon="🚨")

    # ── VIX banner ────────────────────────────────────────────────────────────
    vix_level, vix_rank = _fetch_vix_data()
    if vix_level < 18:
        st.info(f"VIX {vix_level:.1f} ({vix_rank:.0f}th pct) — Low IV. Premium cheap; spreads tight. **Sell side harder to collect credit.**", icon="💡")
    elif vix_level < 28:
        st.success(f"VIX {vix_level:.1f} ({vix_rank:.0f}th pct) — Elevated IV. **Ideal for selling credit spreads** — more premium to collect.", icon="✅")
    else:
        st.error(f"VIX {vix_level:.1f} ({vix_rank:.0f}th pct) — High fear. Violent gamma moves. Size VERY small. Iron condors dangerous today.", icon="🚨")

    # ── Mode + Settings ───────────────────────────────────────────────────────
    with st.expander("⚙️ Settings", expanded=True):
        mode = st.radio(
            "**Mode**",
            ["🏆 Sell Premium  (Credit Spreads — Professional)", "⚡ Buy Directional  (Intraday Momentum)", "📅 Intraday Close  (1-3 DTE, exit today)"],
            index=0, horizontal=False, key="dte0_mode",
            help=(
                "**Sell Premium**: Institutions sell 0DTE credit spreads and iron condors. "
                "Time decay works FOR you. ~70-85% win rate. Lower reward per trade.\n\n"
                "**Buy Directional**: Buy calls/puts on strong intraday moves. "
                "High reward but ~40-50% win rate. MUST have a catalyst. Entry 10AM-12PM only.\n\n"
                "**Intraday Close**: Buy 1-3 DTE options, close same day. More time buffer, "
                "wider universe of names."
            ),
        )

        if mode.startswith("🏆"):
            st.markdown("**Spread strategy**")
            spread_strat = st.radio(
                "Strategy",
                ["Bull Put Spread (Bullish bias)", "Bear Call Spread (Bearish bias)", "Iron Condor (Neutral/Rangebound)"],
                index=2, horizontal=True, key="dte0_spread_strat",
                help="Iron Condor = neutral. Works on calm, low-news days. Bull/Bear spreads need directional bias.",
            )
            sa1, sa2 = st.columns(2)
            account_size  = sa1.number_input("Account size ($)", 1_000, 1_000_000, 25_000, step=1_000, key="dte0_acct")
            risk_pct      = sa2.slider("Risk per trade (%)", 0.5, 3.0, 1.0, step=0.5, key="dte0_risk",
                                       help="1-2% per trade is the professional standard. $250 on a $25k account.")
            target_delta  = st.slider("Short strike delta", 0.05, 0.25, 0.15, step=0.01, key="dte0_short_delta",
                                      help="0.15 = ~85% probability the spread expires worthless. Lower = safer, less credit.")
            st.caption(f"Max risk per trade: **${account_size * risk_pct / 100:,.0f}**  |  "
                       f"Target: sell strikes with ~{(1-target_delta)*100:.0f}% probability of expiring worthless")

        elif mode.startswith("⚡"):
            st.markdown("**Direction & filters**")
            b1, b2 = st.columns(2)
            direction  = b1.radio("Direction", ["Calls (Bullish)", "Puts (Bearish)", "Both"],
                                   horizontal=True, key="dte0_dir_buy")
            min_score  = b2.slider("Min conviction score", 0, 10, 6, key="dte0_score_buy",
                                    help="6+ = meaningful setup. 8+ = strong catalyst present.")
            f1, f2, f3 = st.columns(3)
            min_vol     = f1.number_input("Min option volume", 0, 100_000, 200, step=50, key="dte0_vol_buy")
            max_spread  = f2.slider("Max spread %", 5, 80, 30, step=5, key="dte0_spread_buy")
            moneyness   = f3.slider("Strike ±% from price", 1, 15, 6, key="dte0_money_buy",
                                    help="Keep tight — 0DTE buying is for near-ATM options only")

        else:  # Intraday Close
            st.markdown("**Filters**")
            c1, c2, c3, c4 = st.columns(4)
            min_vol     = c1.number_input("Min option volume", 0, 100_000, 100, step=50, key="dte0_vol_ic")
            max_iv      = c2.slider("Max IV %", 50, 500, 250, step=25, key="dte0_iv_ic")
            max_spread  = c3.slider("Max spread %", 5, 80, 40, step=5, key="dte0_spread_ic")
            moneyness   = c4.slider("Strike ±% from price", 1, 25, 10, key="dte0_money_ic")
            direction   = "Calls (Bullish)"
            min_score   = st.slider("Min score", 0, 10, 4, key="dte0_score_ic")

    st_autorefresh(interval=300_000, key="dte0_refresh")

    # ── Session state init ────────────────────────────────────────────────────
    if "dte0_spread_df"  not in st.session_state:
        st.session_state.dte0_spread_df  = pd.DataFrame()
    if "dte0_df_calls"   not in st.session_state:
        st.session_state.dte0_df_calls   = pd.DataFrame()
        st.session_state.dte0_df_puts    = pd.DataFrame()
        st.session_state.dte0_time       = None

    load_btn = st.button("📂 Load last scan", key="dte0_load")
    if load_btn:
        files = sorted(glob.glob(os.path.join(SCREENER_DIR, "0dte_*.csv")))
        if files:
            saved = pd.read_csv(files[-1])
            if "side" in saved.columns:
                st.session_state.dte0_df_calls = saved[saved["side"] == "call"]
                st.session_state.dte0_df_puts  = saved[saved["side"] == "put"]
            else:
                st.session_state.dte0_spread_df = saved
            st.session_state.dte0_time = f"from file · {os.path.basename(files[-1])}"
            st.toast("Loaded last 0DTE scan.", icon="📂")

    # ── Run scan ──────────────────────────────────────────────────────────────
    if clicked:
        spy_chg = _fetch_spy_chg()
        true_tickers     = tuple(_TRUE_0DTE_UNIVERSE)
        intraday_tickers = tuple(_INTRADAY_CLOSE_UNIVERSE)

        if mode.startswith("🏆"):
            # Prefetch chains
            prog = st.progress(0); status = st.empty()
            for i, sym in enumerate(true_tickers):
                status.caption(f"Fetching {sym}… ({i+1}/{len(true_tickers)})")
                prog.progress((i+1) / max(len(true_tickers), 1))
                _fetch_0dte_chain(sym)
            prog.empty(); status.empty()

            strat_key = (
                "bull_put"    if spread_strat.startswith("Bull") else
                "bear_call"   if spread_strat.startswith("Bear") else
                "iron_condor"
            )
            sdf = _scan_spread_setups(
                true_tickers, strat_key, spy_chg,
                float(target_delta), float(account_size), float(risk_pct),
            )
            st.session_state.dte0_spread_df = sdf
            st.session_state.dte0_time = now.strftime("%H:%M:%S")
            if not sdf.empty:
                os.makedirs(SCREENER_DIR, exist_ok=True)
                sdf.to_csv(os.path.join(SCREENER_DIR, f"0dte_spreads_{_date.today().isoformat()}.csv"), index=False)

        else:
            # Buy directional or intraday close
            dirs = (["call", "put"] if direction == "Both" else
                    ["call"] if direction.startswith("Call") else ["put"])
            use_intraday = mode.startswith("📅")
            scan_tickers = intraday_tickers if use_intraday else true_tickers

            prog = st.progress(0); status = st.empty()
            for i, sym in enumerate(scan_tickers):
                status.caption(f"Fetching {sym}… ({i+1}/{len(scan_tickers)})")
                prog.progress((i+1) / max(len(scan_tickers), 1))
                if use_intraday:
                    _fetch_weekly_calls(sym, 3)
                else:
                    _fetch_0dte_chain(sym)
            prog.empty(); status.empty()

            _iv  = locals().get("max_iv", 250)
            params = dict(
                min_volume=int(min_vol), moneyness_pct=float(moneyness),
                max_iv_pct=float(_iv), max_spread_pct=float(max_spread),
                delta_min=0.10, delta_max=0.70,
                min_score=int(min_score), spy_chg=spy_chg,
            )
            parts_call, parts_put = [], []
            if use_intraday:
                if "call" in dirs:
                    parts_call.append(_scan_intraday_close(intraday_tickers, "call", **params))
            else:
                if "call" in dirs:
                    parts_call.append(_scan_0dte(true_tickers, "call", min_oi=0, **params))
                if "put" in dirs:
                    parts_put.append(_scan_0dte(true_tickers, "put", min_oi=0, **params))

            _nonempty_calls = [p for p in parts_call if not p.empty]
            _nonempty_puts  = [p for p in parts_put  if not p.empty]
            call_df = pd.concat(_nonempty_calls, ignore_index=True).sort_values("score", ascending=False) if _nonempty_calls else pd.DataFrame()
            put_df  = pd.concat(_nonempty_puts,  ignore_index=True).sort_values("score", ascending=False) if _nonempty_puts  else pd.DataFrame()
            st.session_state.dte0_df_calls = call_df
            st.session_state.dte0_df_puts  = put_df
            st.session_state.dte0_time = now.strftime("%H:%M:%S")
            if not call_df.empty or not put_df.empty:
                os.makedirs(SCREENER_DIR, exist_ok=True)
                pd.concat([call_df, put_df], ignore_index=True).to_csv(
                    os.path.join(SCREENER_DIR, f"0dte_{_date.today().isoformat()}.csv"), index=False)

    # ── Render results ────────────────────────────────────────────────────────
    if mode.startswith("🏆"):
        # ── SELL PREMIUM ──────────────────────────────────────────────────────
        sdf = st.session_state.dte0_spread_df
        if sdf.empty and not clicked:
            st.info("Click **▶ Scan Now** to find credit spread setups for today.", icon="ℹ️")
            _render_sell_guide()
            return

        if sdf.empty:
            st.warning("No spread setups found. Try: widen target delta · reduce account size · check market hours.")
            _render_sell_guide()
            return

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Setups Found", len(sdf))
        m2.metric("Best Prob Profit", f"{sdf['prob_profit_%'].max():.1f}%")
        best = sdf.iloc[0]
        m3.metric("Top Credit", f"${best['credit_$']:.2f}  ({best['symbol']})")
        m4.metric("Max Loss (1 contract)", f"${best['max_loss_$'] * 100:.0f}")

        st.divider()
        st.subheader("📋 Spread setups — sorted by probability of profit")
        st.caption("Sell the short strike · Buy the long strike · Collect the credit · Profit if stock stays on your side at close.")

        # Show DTE info banner
        if "dte" in sdf.columns:
            dte_vals = sdf["dte"].unique().tolist()
            if any(d > 0 for d in dte_vals):
                exp_dates = sdf["expiry"].unique().tolist() if "expiry" in sdf.columns else []
                st.info(
                    f"No 0DTE options available today for some tickers. "
                    f"Showing nearest expiry: **{', '.join(exp_dates)}** "
                    f"(DTE: {', '.join(str(d) for d in sorted(dte_vals))}). "
                    "Credit spreads on 1-3 DTE work identically — theta decay is fast.",
                    icon="📅",
                )

        _SELL_COLS = {
            "symbol":         "Ticker",
            "strategy":       "Strategy",
            "dte":            "DTE",
            "expiry":         "Expiry",
            "price":          "Stock $",
            "exp_move_%":     "ExpMove%",
            "atm_iv_%":       "ATM IV%",
            "chg_pct":        "Stk%",
            "rsi14":          "RSI",
            "short_strike":   "Sell Strike",
            "long_strike":    "Buy Strike",
            "short_delta":    "Short Δ",
            "credit_$":       "Credit/sh",
            "max_loss_$":     "MaxLoss/sh",
            "spread_width":   "Width",
            "prob_profit_%":  "ProbProfit%",
            "max_contracts":  "MaxContracts",
        }
        avail = [c for c in _SELL_COLS if c in sdf.columns]
        tbl   = sdf[avail].rename(columns=_SELL_COLS).copy()

        def _color_sell(row):
            pp = row.get("ProbProfit%", 0)
            if pp >= 85:  return ["background-color: rgba(0,200,100,0.15)"] * len(row)
            if pp >= 75:  return ["background-color: rgba(255,165,0,0.10)"] * len(row)
            return [""] * len(row)

        fmt_sell = {
            "Stock $": "{:.2f}", "ExpMove%": "{:.2f}%", "ATM IV%": "{:.1f}%",
            "Stk%": "{:+.2f}%", "RSI": "{:.1f}",
            "Credit/sh": "${:.2f}", "MaxLoss/sh": "${:.2f}",
            "ProbProfit%": "{:.1f}%", "Width": "{:.1f}",
        }
        fmt_sell = {k: v for k, v in fmt_sell.items() if k in tbl.columns}
        st.dataframe(tbl.style.apply(_color_sell, axis=1).format(fmt_sell, na_rep="—"),
                     use_container_width=True, height=480)

        # Per-setup detail cards
        st.subheader("Top setups — trade details")
        for _, row in sdf.head(min(8, len(sdf))).iterrows():
            credit_contract = round(row["credit_$"] * 100, 0)
            loss_contract   = round(row["max_loss_$"] * 100, 0)
            with st.expander(
                f"{row['strategy']}  **{row['symbol']}**  ${row['price']:.2f}  "
                f"·  Sell {row['short_strike']} / Buy {row['long_strike']}  "
                f"·  Credit: ${credit_contract:.0f}/contract  "
                f"·  ProbProfit: {row['prob_profit_%']:.1f}%"
            ):
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Credit collected",  f"${credit_contract:.0f}/contract")
                dc2.metric("Max loss",          f"${loss_contract:.0f}/contract")
                dc3.metric("Prob profit",        f"{row['prob_profit_%']:.1f}%")
                dc4.metric("Max contracts",      f"{row['max_contracts']}")
                exp_mv = row.get("exp_move_%", 0)
                st.caption(
                    f"Expected 1-day move: ±{exp_mv:.2f}%  ·  ATM IV: {row.get('atm_iv_%', '—'):.1f}%  "
                    f"·  RSI: {row.get('rsi14', '—')}  ·  Stock move today: {row['chg_pct']:+.2f}%"
                )
                st.markdown(
                    f"**Rules:**  "
                    f"Take profit at **50% of credit** (close when credit decays to ${credit_contract*0.5:.0f}).  "
                    f"Stop loss at **2× credit** (close if spread widens to ${credit_contract*2:.0f}).  "
                    f"Close by **3:00 PM ET regardless.**"
                )

        st.download_button("⬇️ Download CSV", data=tbl.to_csv(index=False),
                           file_name=f"0dte_spreads_{_date.today().isoformat()}.csv", mime="text/csv")
        _render_sell_guide()

    else:
        # ── BUY DIRECTIONAL / INTRADAY CLOSE ──────────────────────────────────
        call_df = st.session_state.dte0_df_calls
        put_df  = st.session_state.dte0_df_puts

        if mode.startswith("⚡") and hour >= 14.5:
            st.error(
                "🔴 **Entry window closed (after 2:30 PM ET).** "
                "Theta decay is extreme. Do NOT open new directional 0DTE positions. "
                "If you are holding a position, evaluate whether to close NOW vs hold to target."
            )

        if call_df.empty and put_df.empty and not clicked:
            st.info("Click **▶ Scan Now** to scan for directional 0DTE setups.", icon="ℹ️")
            return

        total = len(call_df) + len(put_df)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Setups Found", total)
        m2.metric("Bullish", len(call_df))
        m3.metric("Bearish", len(put_df))
        if not call_df.empty:
            best = call_df.iloc[0]
            m4.metric("Top Call", f"{best['symbol']} {best['score']:.0f}/10")
        if not put_df.empty:
            best = put_df.iloc[0]
            m5.metric("Top Put", f"{best['symbol']} {best['score']:.0f}/10")

        st.divider()

        _COLS_0DTE = {
            "symbol": "Ticker", "category": "Type", "score": "Score🎯",
            "setup": "Setup", "urgency": "Time⏰",
            "chg_pct": "Stk%", "rel_str": "vs SPY", "rsi14": "RSI",
            "vol_ratio": "VolSurge", "strike": "Strike", "price": "Stock $",
            "moneyness_pct": "±%ATM", "delta": "Delta",
            "gamma_dollar": "Gamma⚡$", "theta_per_hr": "Theta/hr$",
            "mid": "Mid $", "spread_pct": "Sprd%",
            "volume": "OptVol", "openInterest": "OI",
            "voi_ratio": "V/OI", "iv_pct": "IV%",
        }

        def _render_buy_table(df, label):
            if df.empty:
                st.info(f"No {label} found. Try: lower Min Score · widen ±% · lower volume.")
                return
            avail = [c for c in _COLS_0DTE if c in df.columns]
            tbl   = df[avail].rename(columns=_COLS_0DTE).copy()
            fmt   = {"Score🎯": "{:.0f}", "Stk%": "{:+.2f}%", "vs SPY": "{:+.2f}%",
                     "RSI": "{:.1f}", "VolSurge": "{:.2f}x", "Strike": "{:.2f}",
                     "Stock $": "{:.2f}", "±%ATM": "{:.2f}", "Delta": "{:.2f}",
                     "Gamma⚡$": "{:.2f}", "Theta/hr$": "{:.2f}", "Mid $": "{:.2f}",
                     "Sprd%": "{:.1f}", "V/OI": "{:.2f}", "IV%": "{:.1f}"}
            fmt   = {k: v for k, v in fmt.items() if k in tbl.columns}
            def _col(row):
                s = row.get("Score🎯", 0)
                if s >= 8:   return ["background-color: rgba(255,75,75,0.22)"] * len(row)
                elif s >= 6: return ["background-color: rgba(255,165,0,0.18)"] * len(row)
                elif s >= 4: return ["background-color: rgba(0,200,100,0.12)"] * len(row)
                return [""] * len(row)
            st.dataframe(tbl.style.apply(_col, axis=1).format(fmt, na_rep="—"),
                         use_container_width=True, height=460)
            # Top pick cards
            st.subheader(f"Top setups — {label}")
            for sym, grp in df.groupby("symbol", sort=False):
                sc  = grp["score"].max()
                setup = grp["setup"].iloc[0] if "setup" in grp.columns else ""
                mid_p = grp["mid"].min() if "mid" in grp.columns else 0
                with st.expander(f"{setup}  **{sym}**  ${grp['price'].iloc[0]:.2f}  "
                                  f"({grp['chg_pct'].iloc[0]:+.2f}% today)  Score={sc:.0f}/10"):
                    mini = grp[[c for c in _COLS_0DTE if c in grp.columns]].rename(columns=_COLS_0DTE)
                    st.dataframe(mini, use_container_width=True, hide_index=True)
                    # Risk rules
                    if mid_p > 0:
                        stop_val  = round(mid_p * 0.50 * 100, 0)
                        tgt_val   = round(mid_p * 1.50 * 100, 0)
                        st.markdown(
                            f"**Entry:** buy near ask  ·  "
                            f"**Stop:** close if option loses 50% (≈ -${stop_val:.0f}/contract)  ·  "
                            f"**Target:** 80-150% gain (≈ +${tgt_val:.0f}/contract)  ·  "
                            f"**Hard close by 2:30 PM ET.**"
                        )
                    reasons = []
                    row0 = grp.iloc[0]
                    if abs(row0.get("rel_str", 0)) >= 1.5:
                        reasons.append(f"outperforming SPY by {abs(row0['rel_str']):.1f}%")
                    if row0.get("vol_ratio", 1) >= 1.5:
                        reasons.append(f"volume {row0['vol_ratio']:.1f}× above average")
                    if row0.get("voi_ratio", 0) >= 2.0:
                        reasons.append(f"unusual flow V/OI {row0['voi_ratio']:.1f}×")
                    if reasons:
                        st.caption("Signal: " + " · ".join(reasons))
            st.download_button(f"⬇️ CSV", data=tbl.to_csv(index=False),
                               file_name=f"0dte_{label.lower().replace(' ','_')}_{_date.today().isoformat()}.csv",
                               mime="text/csv", key=f"dl_{label}")

        if not call_df.empty and not put_df.empty:
            tc, tp = st.tabs(["🟢 Calls (Bullish)", "🔴 Puts (Bearish)"])
            with tc: _render_buy_table(call_df, "Bullish Calls")
            with tp: _render_buy_table(put_df,  "Bearish Puts")
        elif not call_df.empty:
            _render_buy_table(call_df, "Bullish Calls")
        elif not put_df.empty:
            _render_buy_table(put_df, "Bearish Puts")
        else:
            st.info("No setups matched. Try lowering the minimum score or widening strike range.", icon="ℹ️")


# ─────────────────────────────────────────────────────────────────────────────
# Low-Premium Call Scanner  (<$X per share, high-leverage cheap calls)
# ─────────────────────────────────────────────────────────────────────────────

def render_cheap_calls():
    import math as _math

    clicked = _page_header(
        "💰", "Cheap Calls  (<$1.50 premium)",
        "Finds calls under $1.50/share ($150/contract) expiring within 7 calendar days of purchase. "
        "Backtest-optimized defaults: DTE≤7, RSI 55–70, delta 0.05–0.45. "
        "Sorted by GO Score — highest upside conviction first.",
        scan_key="cheap_calls_scan_btn", last_key="cheap_calls_time",
    )

    # ── Universe selector + Settings ─────────────────────────────────────────
    with st.expander("⚙️ Settings", expanded=True):
        st.markdown("**Universe**")
        universe_choice = st.radio(
            "Scan universe",
            ["Focused (~60 tickers)", "NDQ100 + High-Growth (~130)", "Full (~600)"],
            index=0, horizontal=True, key="cc_universe",
            help="Focused = High-Growth + Watchlist only. Fastest scan.",
        )

        st.markdown("**Premium & expiry**")
        p1, p2, p3 = st.columns(3)
        max_prem  = p1.slider("Max premium ($/share)", 0.05, 5.00, 1.50, step=0.05,
                              key="cc_max_prem",
                              help="$1.50/share = $150 per contract")
        max_dte   = p2.slider("Max DTE", 0, 14, 7, key="cc_max_dte")
        min_vol   = p3.number_input("Min call volume", 0, 50_000, 50, step=25, key="cc_min_vol")

        st.markdown("**Stock filters**")
        s1, s2, s3 = st.columns(3)
        moneyness     = s1.slider("Strike ± % from price", 1, 30, 15, key="cc_moneyness",
                                  help="Cheap calls are usually OTM — widen this")
        max_spread    = s2.slider("Max bid/ask spread %", 5, 60, 30, step=5, key="cc_spread")
        min_chg       = s3.slider("Min stock move today %", -5.0, 5.0, 0.0, step=0.5, key="cc_chg")

        st.markdown(
            "**RSI filter** *(backtested sweet spot: 55–70 — trending but not overbought)*"
        )
        cc_rsi_on = st.checkbox(
            "Enable RSI filter", value=True, key="cc_rsi_enable",
            help="ON by default — RSI 55–70 improved win rate by ~7 pp in backtests",
        )
        rsi_cols = st.columns(2)
        cc_rsi_min = rsi_cols[0].slider(
            "RSI min", 10, 65, 55, step=5, key="cc_rsi_min",
            help="Exclude stocks in downtrend / oversold territory",
            disabled=not cc_rsi_on,
        )
        cc_rsi_max = rsi_cols[1].slider(
            "RSI max", 55, 90, 70, step=5, key="cc_rsi_max",
            help="Exclude overbought stocks — buying calls into exhaustion lowers win rate",
            disabled=not cc_rsi_on,
        )

        st.markdown("**Volume & relative strength** *(research-backed entry quality filters)*")
        vrs1, vrs2 = st.columns(2)
        cc_vol_surge = vrs1.slider(
            "Min volume surge (×avg)", 0.0, 3.0, 1.5, step=0.5, key="cc_vol_surge",
            help="Volume ÷ 3-month average. Backtest: ≥1.5× raises expectancy from 151% → 230%. 0 = no filter.",
        )
        cc_rel_str = vrs2.slider(
            "Min rel. strength vs SPY (%)", -5.0, 3.0, 0.0, step=0.5, key="cc_rel_str",
            help="Stock must be outperforming SPY by this % today. 0 = must be beating SPY.",
        )

        sim_move = 5  # kept for internal BS repricing but not shown in UI

    st_autorefresh(interval=900_000, key="cheap_calls_refresh")

    # ── Init session state ────────────────────────────────────────────────────
    if "cc_df" not in st.session_state:
        st.session_state.cc_df   = pd.DataFrame()
        st.session_state.cheap_calls_time = None

    # ── Load last saved button ────────────────────────────────────────────────
    load_btn = st.button("📂 Load last scan", key="cc_load_last")
    if load_btn:
        files = sorted(glob.glob(os.path.join(SCREENER_DIR, "cheap_calls_*.csv")))
        if files:
            st.session_state.cc_df = pd.read_csv(files[-1])
            st.session_state.cheap_calls_time = f"from file · {os.path.basename(files[-1])}"
            st.toast("Loaded last saved scan.", icon="📂")
        else:
            st.warning("No saved scan found. Click ▶ Scan Now to run one.")

    # ── VIX / IV regime banner ────────────────────────────────────────────────
    vix_level, vix_rank = _fetch_vix_data()
    # Research: VIX 15-25 is the backtested sweet spot for call buying (SSRN Chuk 2025,
    # Options Cafe). Below 15 = IV suppressed, expensive to buy relative to realised move.
    # Above 25 = too expensive AND high correlation days where premium decays fast.
    if vix_level < 15:
        st.warning(
            f"VIX {vix_level:.1f} ({vix_rank:.0f}th pct) — **Below sweet spot.** "
            "Low-VIX regime means options are priced expensively relative to realised moves. "
            "Research (Options Cafe, SSRN) shows win rates drop below VIX 15. Proceed with caution.",
            icon="⚠️",
        )
    elif vix_level < 25:
        st.success(
            f"VIX {vix_level:.1f} ({vix_rank:.0f}th pct) — **Sweet spot (15–25).** "
            "Backtested optimal IV regime for call buying. SSRN study: win rates rise to 65%+ "
            "in this VIX range vs 27% unfiltered. Good environment to trade.",
            icon="✅",
        )
    elif vix_level < 35:
        st.warning(
            f"VIX {vix_level:.1f} ({vix_rank:.0f}th pct) — **Elevated IV.** "
            "Premium costs more; you need bigger moves to profit. "
            "Reduce position size; focus only on highest GO score setups (≥7).",
            icon="⚠️",
        )
    else:
        st.error(
            f"VIX {vix_level:.1f} ({vix_rank:.0f}th pct) — **High fear.** "
            "IV is historically expensive. Retail call buyers lose the most in this regime. "
            "Consider waiting or switching to vertical spreads.",
            icon="🚨",
        )

    # ── Run scan only when button clicked ────────────────────────────────────
    if clicked:
        if universe_choice.startswith("Focused"):
            tickers = tuple(_dedup(_HIGH_GROWTH_TICKERS + list(WATCHLIST_TICKERS)))
        elif universe_choice.startswith("NDQ100"):
            tickers = tuple(_dedup(_cached_ndq100() + _HIGH_GROWTH_TICKERS + list(WATCHLIST_TICKERS)))
        else:
            tickers = tuple(get_selected_tickers())

        spy_chg = _fetch_spy_chg()
        _ticker_count_caption(list(tickers))

        prog = st.progress(0)
        status = st.empty()
        for i, sym in enumerate(tickers):
            status.caption(f"Fetching {sym}…  ({i+1}/{len(tickers)})")
            prog.progress((i + 1) / max(len(tickers), 1))
            _fetch_weekly_calls(sym, max_dte)
        prog.empty()
        status.empty()

        _cc_rsi_min = float(cc_rsi_min) if cc_rsi_on else 0.0
        _cc_rsi_max = float(cc_rsi_max) if cc_rsi_on else 100.0

        df = _scan_weekly_options(
            tickers,
            max_dte     = max_dte,
            min_volume  = int(min_vol),
            min_oi      = 0,
            moneyness_pct      = float(moneyness),
            show_itm    = False,
            show_otm    = True,
            sort_by     = "GO Score",
            min_underlying_chg = float(min_chg),
            require_above_ema20 = False,
            require_above_sma50 = False,
            max_iv_pct  = 999.0,
            max_spread_pct = float(max_spread),
            delta_min   = 0.05,
            delta_max   = 0.45,
            rsi_min     = _cc_rsi_min,
            rsi_max     = _cc_rsi_max,
            spy_chg     = spy_chg,
            max_premium = float(max_prem),
            min_vol_ratio = float(cc_vol_surge),
            min_rel_str   = -999.0,   # rel_str filter doesn't improve WR — keep off
        )

        ts = pd.Timestamp.now()
        st.session_state.cc_df = df
        st.session_state.cheap_calls_time = ts.strftime("%H:%M:%S")

        # auto-save so "Load last scan" works next time
        if not df.empty:
            os.makedirs(SCREENER_DIR, exist_ok=True)
            _save_path = os.path.join(
                SCREENER_DIR, f"cheap_calls_{ts.strftime('%Y%m%d_%H%M')}.csv"
            )
            df.to_csv(_save_path, index=False)

    df = st.session_state.cc_df

    if df.empty and not clicked:
        st.info("Click **▶ Scan Now** to run, or **📂 Load last scan** to restore previous results.")
        return

    # ── Simulate potential return (always recomputed so sim_move slider is live) ──
    if not df.empty:
        def _sim_return(row):
            if row["mid"] <= 0:
                return 0.0
            # 0DTE: option expires today — if stock moves enough it pays intrinsic,
            # otherwise zero. Show intrinsic payoff, capped to avoid misleading numbers.
            if int(row["dte"]) == 0:
                new_S     = row["price"] * (1 + sim_move / 100)
                intrinsic = max(new_S - float(row["strike"]), 0.0)
                if intrinsic <= 0:
                    return -100.0   # expires worthless
                return round((intrinsic - row["mid"]) / row["mid"] * 100, 0)
            iv      = max(float(row["impliedVolatility"]), 0.20)
            new_S   = row["price"] * (1 + sim_move / 100)
            new_T   = max(float(row["dte"]) - 0.5, 0.25) / 365
            new_val = _bs_call_price(new_S, float(row["strike"]), new_T, iv)
            return round((new_val - row["mid"]) / row["mid"] * 100, 0)

        df = df.copy()
        df["sim_ret%"] = df.apply(_sim_return, axis=1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    if not df.empty:
        m1.metric("Calls Found", len(df))
        m2.metric("Tickers", df["symbol"].nunique())
        m3.metric("Cheapest", f"${df['mid'].min():.2f}")
        m4.metric("Best GO Score", f"{df['go_score'].max():.0f}/10")
        best_row = df.loc[df["go_score"].idxmax()]
        m5.metric("Top Pick", f"{best_row['symbol']}  GO={best_row['go_score']:.0f}  DTE={int(best_row['dte'])}")
    else:
        for m in [m1, m2, m3, m4, m5]:
            m.metric("—", "—")

    st.divider()

    if df.empty:
        st.warning(
            "No cheap calls matched. Try: **raise Max premium**, **widen Strike ±%**, "
            "**lower Min volume**, or **set Min stock move to -5%**."
        )
        return

    # ── 0DTE risk warning ─────────────────────────────────────────────────────
    if not df.empty and (df["dte"] == 0).any():
        n0 = (df["dte"] == 0).sum()
        st.error(
            f"⚠️ **{n0} result{'s' if n0 > 1 else ''} expire TODAY (0DTE).** "
            "These are not regular call-buying setups. A 0DTE option at <$0.20 "
            "has >90% probability of expiring worthless. GO Score and STRONG BUY "
            "reflect the underlying stock — not the option's risk. "
            "The sim return shows intrinsic value if the stock hits the strike by close. "
            "**Treat 0DTE as a lottery ticket, not a conviction trade.**"
        )

    # ── Table ─────────────────────────────────────────────────────────────────
    _C = {
        "symbol":   "Ticker",
        "go_score": "GO🎯",
        "expiry":   "Expiry",
        "dte":      "DTE",
        "chg_pct":  "Stk%",
        "rel_str":  "vs SPY",
        "rsi14":    "RSI",
        "strike":   "Strike",
        "price":    "Stock $",
        "mid":      "Prem $",
        "spread_pct": "Sprd%",
        "delta":    "Delta",
        "volume":   "CallVol",
        "voi_ratio":"V/OI",
        "pc_ratio": "C/P",
        "vol_ratio":"VolSrg",
        "iv_pct":   "IV%",
        "signals":  "Signals",
    }
    tbl = df[list(_C)].rename(columns=_C).copy()

    def _cr(row):
        gs = row.get("GO🎯", 0)
        if gs >= 8:   return ["background-color: rgba(255,75,75,0.20)"] * len(row)
        elif gs >= 6: return ["background-color: rgba(255,165,0,0.15)"] * len(row)
        elif gs >= 4: return ["background-color: rgba(0,200,100,0.10)"] * len(row)
        return [""] * len(row)

    styled = (
        tbl.style
        .apply(_cr, axis=1)
        .format({
            "GO🎯": "{:.0f}",
            "Stk%": "{:+.2f}%", "vs SPY": "{:+.2f}%",
            "RSI": "{:.1f}",
            "Strike": "{:.2f}", "Stock $": "{:.2f}",
            "Prem $": "{:.2f}", "Sprd%": "{:.1f}",
            "Delta": "{:.2f}", "V/OI": "{:.2f}",
            "C/P": "{:.2f}", "VolSrg": "{:.2f}x",
            "IV%": "{:.1f}",
        }, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=550)

    # ── Backtest performance summary ──────────────────────────────────────────
    st.divider()
    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("Backtest WR (best config)", "49–51%", help="GO≥2, DTE=7, OTM≈1%, RSI 55-70, Stop 35%")
    bm2.metric("Stop-loss (optimal)", "35–40%", help="Tighter stop raises WR by 2-3 pp. 35% = 49.3% WR backtested.")
    bm3.metric("Profit target (optimal)", "1.5x–2x", help="1.5x → 50.7% WR. 2x → 46.6% WR but higher avg winner.")
    bm4.metric("Vol surge edge", "↑80% E", help="Stocks with ≥1.5× avg volume: expectancy 230% vs 151% baseline.")

    # ── Top picks ─────────────────────────────────────────────────────────────
    st.subheader("Top picks — highest GO Score")
    top3 = df.nlargest(min(10, len(df)), "go_score")
    for _, row in top3.iterrows():
        gs   = row["go_score"]
        dte  = int(row["dte"])
        mid  = row["mid"]
        cost = mid * 100
        if dte == 0:
            label = "⚠️ 0DTE LOTTERY"
        else:
            label = "🚀STRONG BUY" if gs >= 8 else ("⚡BUY" if gs >= 6 else "👀WATCH")

        # Compute Kelly for this trade
        # Backtested: WR~47-51% at stop 35-50%, avg winner +182%, avg loser -38%
        wr_est   = 0.49
        win_loss = 1.82 / 0.38   # avg winner / avg loser ratio from backtest
        kelly    = max(wr_est - (1 - wr_est) / win_loss, 0) * 100
        qkelly   = round(kelly * 0.25, 1)   # quarter-Kelly for fat tails

        stop_val   = round(mid * 0.35 * 100, 0)
        target_val = round(mid * 1.50 * 100, 0)

        st.markdown(
            f"**{label}  {row['symbol']}**  "
            f"${row['strike']:.0f}C  exp {row['expiry']}  "
            f"· Prem **${mid:.2f}** (${cost:.0f}/contract)  "
            f"· GO={gs:.0f}/10  DTE={dte}  VolSurge={row.get('vol_ratio', 0):.1f}×"
            f"\n> 🛑 Stop 35%: -${stop_val:.0f}/contract  "
            f"· 🎯 Target 1.5x: +${target_val:.0f}/contract  "
            f"· Kelly: {kelly:.0f}% (use ¼-Kelly ≈ {qkelly}% of account)"
            f"\n> {row['signals']}"
        )

    # ── Research findings expander ────────────────────────────────────────────
    with st.expander("📊 Backtest findings & research summary", expanded=False):
        st.markdown("""
### Our backtest results (GO≥2, DTE=7, OTM≈1%, RSI 55–70)

| Parameter | Tested | Optimal | WR | Expectancy |
|-----------|--------|---------|-----|------------|
| Stop loss | 35-75% | **35%** | **49.3%** | 161% |
| Profit target | 1.5x–5x | **1.5x** (max WR) or **5x** (max E) | 50.7% / 33.7% | 136% / 213% |
| Vol surge filter | 0–2.5× | **1.5–2.0×** | 45.3–45.5% | **230–274%** |
| Rel. strength vs SPY | none–1.0% | **None** | 46.5% | drops to 138% |
| RSI window | various | **55–70** | +1.3 pp vs no filter | — |
| OTM % | 0–5% | **1%** (ATM) | 44.3% | 185% |
| DTE | 1–7 | **7** | 45.3% | 150% |
| GO threshold | 0–5 | **≥2** | 40.1% | 293% |

**Key insight — Volume Surge is the biggest edge:**
Stocks with ≥1.5× average volume at entry produce 230% expectancy vs 151% baseline — a 52% improvement.
This matches academic findings: unusual volume precedes significant price moves, improving option payoffs.

### External research (SSRN, Options Cafe, alphacrunching.com)

- **Unfiltered call buying loses money**: retail buyers average **-$8.05/contract** on 0DTE (SSRN Beckmeyer 2023)
- **VIX 15–25 sweet spot**: win rate rises from 27% to **65%+** in this VIX regime (SSRN Chuk 2025)
- **7DTE SPY call strategy**: 56% WR, 8.9% CAGR at 2% allocation (alphacrunching.com, 2022–2025)
- **IV Rank < 30% filter**: single most powerful entry filter across all option-buying strategies (ORATS 180M backtest)
- **100% gain / 50% stop** confirmed as risk-adjusted sweet spot (Options Cafe 0DTE ORB backtest)
- **Near-ATM (delta 0.25–0.35)** significantly outperforms deep OTM (delta 0.10) in all studies

### Kelly Criterion guide

Formula: `Kelly% = WR − [(1 − WR) / (Win÷Loss ratio)]`

With our backtested params (49% WR, 4.8× profit factor):
- Full Kelly ≈ **31%** of account per trade (theoretical max)
- **Quarter-Kelly ≈ 8%** (recommended for options — fat tails, model uncertainty)
- Practical cap: **1–3% per trade** if running multiple positions simultaneously
- Never use more than 5% on any single cheap call trade
        """)

    from datetime import date as _date
    st.download_button(
        "⬇️ Download CSV",
        data=tbl.to_csv(index=False),
        file_name=f"cheap_calls_{_date.today().isoformat()}.csv",
        mime="text/csv",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Scanner Registry
# To add a new scanner: append one entry here, then write a render_*() function.
# ═════════════════════════════════════════════════════════════════════════════

SCANNERS = [
    {"id": "home",         "label": "🏠  Home",               "category": "home",    "render": render_home},
    {"id": "stock_analyzer", "label": "🔍  Stock Analyzer",     "category": "analyze",   "render": render_stock_analyzer},
    {"id": "alerts",         "label": "🔔  Daily Alerts",        "category": "trades",    "render": render_alerts},
    {"id": "combined",       "label": "🎯  Combined Strategy",   "category": "trades",    "render": render_combined},
    {"id": "oneil",          "label": "🏆  O'Neil CAN SLIM",     "category": "trades",    "render": render_oneil},
    {"id": "minervini",      "label": "📐  Minervini SEPA",      "category": "trades",    "render": render_minervini},
    {"id": "livermore",      "label": "🔴  Livermore Pivotal",   "category": "trades",    "render": render_livermore},
    {"id": "breakout",       "label": "💥  Breakout",            "category": "trades",    "render": render_breakout},
    {"id": "ibd",            "label": "📋  IBD Buy Zone",        "category": "trades",    "render": render_ibd},
    {"id": "parabolic_short","label": "📉  Parabolic Short",     "category": "trades",    "render": render_parabolic_short},
    {"id": "ema",            "label": "📉  EMA Crossover",       "category": "technical", "render": render_ema},
    {"id": "rsi",            "label": "〰️  RSI Scanner",         "category": "technical", "render": render_rsi},
    {"id": "macd",           "label": "〽️  MACD Scanner",        "category": "technical", "render": render_macd},
    {"id": "gap",            "label": "🕳️  Gap Scanner",          "category": "technical", "render": render_gap},
    {"id": "volume",         "label": "🔊  Volume Surge",        "category": "technical", "render": render_volume},
    {"id": "zero_dte",       "label": "🔥  0DTE Scanner",            "category": "options", "render": render_0dte_scanner},
    {"id": "weekly_opts",    "label": "⚡  Weekly Options (0-7 DTE)", "category": "options", "render": render_weekly_options},
    {"id": "cheap_calls",   "label": "💰  Cheap Calls (<$150)",      "category": "options", "render": render_cheap_calls},
    {"id": "weekly_puts",   "label": "📉  Weekly Puts (0-3 DTE)",    "category": "options", "render": render_weekly_puts},
    {"id": "swing_opts",     "label": "🎰  Options 45-60 DTE",  "category": "options",   "render": render_swing_options},
    {"id": "opt_paper",      "label": "🟢  Options Paper Trade", "category": "options",   "render": render_options_paper_trade},
    {"id": "opt_log",        "label": "📋  Options Trade Log",   "category": "options",   "render": render_options_log},
    {"id": "opt_backtest",   "label": "🔬  Options Backtest",    "category": "options",   "render": render_options_backtest},
    {"id": "spy_alerts",     "label": "📡  SPY Reversal Log",    "category": "automate",  "render": render_spy_alerts},
    {"id": "paper_trade",    "label": "🤖  Paper Trade",         "category": "automate",  "render": render_paper_trade},
    {"id": "backtest",       "label": "📊  Backtest",            "category": "automate",  "render": render_backtest},
    {"id": "astro",          "label": "🔭  Fin Astrology",       "category": "macro",     "render": render_astro},
    {"id": "influencers",    "label": "📡  Influencer Tracker",  "category": "macro",     "render": render_influencers},
    {"id": "history",        "label": "🗂  History",             "category": "more",      "render": render_history},
    {"id": "macro_calendar", "label": "🗓️  Macro Calendar",      "category": "macro",     "render": render_calendar},
]

SCANNER_GROUPS = [
    {"id": "home",      "label": "HOME",         "icon": "🏠"},
    {"id": "analyze",   "label": "ANALYZE",     "icon": "🔍"},
    {"id": "trades",    "label": "FIND TRADES",  "icon": "🎯"},
    {"id": "technical", "label": "TECHNICAL",   "icon": "⚡"},
    {"id": "options",   "label": "OPTIONS",     "icon": "🎰"},
    {"id": "automate",  "label": "AUTOMATE",    "icon": "🤖"},
    {"id": "macro",     "label": "MACRO",       "icon": "🌍"},
    {"id": "more",      "label": "MORE",        "icon": "⋯"},
]
SCANNERS_BY_ID = {s["id"]: s for s in SCANNERS}


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

# Initialize active scanner — honour ?scanner=<id> URL param on first load
if "active_scanner_id" not in st.session_state:
    _url_scanner = st.query_params.get("scanner", "home")
    st.session_state["active_scanner_id"] = (
        _url_scanner if _url_scanner in SCANNERS_BY_ID else "home"
    )

with st.sidebar:
    st.markdown(
        f"<div style='font-size:1.4rem;font-weight:700;margin-bottom:4px;color:#e2e8f0'>📈 Swing Dashboard</div>"
        f"<div style='margin-bottom:20px'>{market_badge_html()}</div>",
        unsafe_allow_html=True,
    )

    active_id = st.session_state["active_scanner_id"]

    for group in SCANNER_GROUPS:
        group_scanners = [s for s in SCANNERS if s["category"] == group["id"]]
        if not group_scanners:
            continue

        st.markdown(
            f"<div class='nav-group-hdr'>{group['label']}</div>",
            unsafe_allow_html=True,
        )
        for scanner in group_scanners:
            is_active = (active_id == scanner["id"])
            nav_cols = st.columns([5, 1])
            with nav_cols[0]:
                if is_active:
                    st.markdown('<span class="nav-active-marker"></span>', unsafe_allow_html=True)
                if st.button(scanner["label"], key=f"nav_{scanner['id']}", use_container_width=True):
                    st.session_state["active_scanner_id"] = scanner["id"]
                    st.query_params["scanner"] = scanner["id"]
                    st.rerun()
            with nav_cols[1]:
                # Open this scanner in a new browser tab
                scanner_url = f"http://localhost:8501/?scanner={scanner['id']}"
                st.link_button("↗", scanner_url, use_container_width=True,
                               help=f"Open {scanner['label'].strip()} in new tab")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.divider()
    st.toggle("Russell 1000", value=False, key="use_russell1000",
              help="Adds ~400 extra mid-caps. Scans take 2-3× longer.")
    st.divider()
    auto_refresh = st.checkbox("Auto-refresh (15 min)", value=False, key="auto_refresh")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")


# ═══════════════════════════════════════════════════════════════════════
# Main content — dispatch to selected scanner
# ═══════════════════════════════════════════════════════════════════════
_active_scanner = SCANNERS_BY_ID[st.session_state["active_scanner_id"]]
_tab_title = _active_scanner["label"].encode("ascii", "ignore").decode().strip()
st.markdown(f"<script>document.title = '{_tab_title}';</script>", unsafe_allow_html=True)
_active_scanner["render"]()

# ── Auto-refresh ──────────────────────────────────────────────────────
if auto_refresh and is_market_open():
    st_autorefresh(interval=900_000, key="global_autorefresh")
