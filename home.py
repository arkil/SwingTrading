"""
Unified Trading Dashboard — http://localhost:8501/
Run: streamlit run home.py --server.port 8501

Unified dashboard with two modes:

MORNING BRIEFING (Quick Daily View)
  • 5 Day Trade Recommendations (gap + momentum signals, today's freshest setups)
  • 5 Breakout Stocks (NR7, BO-52W, BB-squeeze, MA-reclaim, inside-bar)
  • 5 Options Recommendations (45-60 DTE swing options, best greek profile)
  • Today's Earnings & Major Events

FULL DASHBOARD (All Scanners)
  • All trading scanners (Livermore, EMA, Breakout, Minervini, etc.)
  • Detailed analysis and research tools
  • Interactive scanner interface

Auto-refresh
  • Before market open: page reruns every 60 s until 9:30 AM ET
  • At 9:30 AM ET data is fetched fresh (cache TTL = 5 min during market hours)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

_env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_file):
    with open(_env_file) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

import streamlit as st
import pandas as pd
import pytz
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
html, body { overflow-y: auto !important; }
.block-container {
    padding: 2rem 3rem 3rem !important;
    max-width: 1600px;
}

/* === BREADCRUMB === */
.breadcrumb {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 24px;
    font-size: 13px;
    color: #94a3b8;
}
.breadcrumb a {
    color: #2563eb;
    text-decoration: none;
}
.breadcrumb a:hover {
    text-decoration: underline;
}
.breadcrumb-sep {
    color: #64748b;
}

/* === HEADER SECTION === */
.page-header {
    margin-bottom: 32px;
}
.page-title {
    font-size: 32px;
    font-weight: 600;
    color: #f5f7fa;
    margin: 0 0 8px 0;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 14px;
    color: #cbd5e1;
    margin: 0 0 16px 0;
}
.page-meta {
    font-size: 13px;
    color: #94a3b8;
}

/* === STATUS BAR === */
.status-bar {
    display: flex;
    gap: 24px;
    align-items: center;
    background: #1a2332;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px 20px;
    margin-bottom: 32px;
    font-size: 13px;
    color: #cbd5e1;
}
.status-bar .label {
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}
.status-bar .val   {
    color: #f5f7fa;
    font-weight: 600;
    font-size: 14px;
}
.status-bar .dot-open  {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    display: inline-block;
    margin-right: 6px;
}
.status-bar .dot-closed {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
    display: inline-block;
    margin-right: 6px;
}

/* === SECTION CARD === */
.dash-card {
    background: #1a2332;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 32px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}
.dash-card h3 {
    color: #f5f7fa;
    font-size: 20px;
    font-weight: 600;
    letter-spacing: 0;
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* === SIGNAL BADGES === */
.sig-strong {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.sig-buy    {
    background: rgba(16, 185, 129, 0.1);
    color: #10b981;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.sig-watch  {
    background: rgba(245, 158, 11, 0.1);
    color: #f59e0b;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid rgba(245, 158, 11, 0.2);
}
.sig-call   {
    background: rgba(16, 185, 129, 0.1);
    color: #10b981;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.sig-put    {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid rgba(239, 68, 68, 0.2);
}
.sig-bo     {
    background: rgba(59, 130, 246, 0.1);
    color: #3b82f6;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid rgba(59, 130, 246, 0.2);
}

/* === IMPACT BADGES === */
.imp-high   {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid rgba(239, 68, 68, 0.2);
}
.imp-medium {
    background: rgba(245, 158, 11, 0.1);
    color: #f59e0b;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid rgba(245, 158, 11, 0.2);
}
.imp-earn   {
    background: rgba(6, 182, 212, 0.1);
    color: #06b6d4;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid rgba(6, 182, 212, 0.2);
}

/* === TICKER & PRICE STYLES === */
.ticker {
    font-size: 15px;
    font-weight: 700;
    color: #f5f7fa;
    letter-spacing: 0.5px;
    min-width: 50px;
}
.price  {
    font-size: 14px;
    color: #cbd5e1;
    font-family: "SF Mono", Monaco, monospace;
}
.green  {
    color: #10b981;
    font-weight: 600;
}
.red    {
    color: #ef4444;
    font-weight: 600;
}

/* === DATA TABLE === */
table.dash-tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}
table.dash-tbl th {
    color: #cbd5e1;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 12px 16px;
    border-bottom: 1px solid #334155;
    text-align: left;
    font-weight: 600;
    background: #0f1419;
}
table.dash-tbl td {
    padding: 14px 16px;
    border-bottom: 1px solid #1e293b;
    color: #f5f7fa;
    vertical-align: middle;
    font-variant-numeric: tabular-nums;
}
table.dash-tbl tr:hover td {
    background: #252d47;
}
table.dash-tbl tr:last-child td {
    border-bottom: none;
}

/* === EVENT ROW === */
.event-row {
    padding: 12px 0;
    border-bottom: 1px solid #1e293b;
    display: flex;
    gap: 16px;
    align-items: flex-start;
}
.event-row:last-child {
    border-bottom: none;
}
.event-time {
    color: #94a3b8;
    font-size: 12px;
    min-width: 70px;
    padding-top: 2px;
    font-weight: 500;
}
.event-name {
    color: #f5f7fa;
    font-size: 14px;
    flex: 1;
}

/* === EMPTY STATE === */
.no-data {
    color: #94a3b8;
    font-size: 14px;
    padding: 2rem 1rem;
    text-align: center;
    background: #0f1419;
    border-radius: 6px;
    border: 1px dashed #334155;
}

/* === INFO TEXT === */
.refresh-info {
    color: #94a3b8;
    font-size: 13px;
}

/* === FOCUS STATES === */
button:focus-visible {
    outline: 2px solid #2563eb !important;
    outline-offset: 2px !important;
}

/* === STREAMLIT BUTTON OVERRIDES === */
.stButton > button {
    height: 40px;
    font-size: 14px;
    font-weight: 600;
    border-radius: 6px;
    border: 1px solid #334155 !important;
    background-color: #2563eb !important;
    color: white !important;
    transition: all 150ms ease;
}
.stButton > button:hover {
    background-color: #1e40af !important;
    border-color: #2563eb !important;
}
.stButton > button:focus {
    outline: 2px solid #2563eb !important;
    outline-offset: 2px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Market clock helpers ───────────────────────────────────────────────────────

ET = pytz.timezone("America/New_York")


def _now_et() -> datetime:
    return datetime.now(ET)


def _is_market_open() -> bool:
    now = _now_et()
    if now.weekday() >= 5:
        return False
    o = now.replace(hour=9, minute=30, second=0, microsecond=0)
    c = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return o <= now <= c


def _secs_to_open() -> int:
    now = _now_et()
    o = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now.weekday() >= 5 or now >= o:
        return 0
    return int((o - now).total_seconds())


def _cache_ttl() -> int:
    return 300 if _is_market_open() else 3600


# ── Universe: from the project, no hardcoding ────────────────────────────────

from livermore_pivotal_screener import DEFAULT_TICKERS, WATCHLIST_TICKERS, _dedup

# Use the user's curated watchlist as the home-dashboard scan universe.
# DEFAULT_TICKERS adds large-cap S&P500 on top — both are managed in livermore_pivotal_screener.py.
SCAN_UNIVERSE = _dedup(WATCHLIST_TICKERS)

# ── Data fetchers (cached) ────────────────────────────────────────────────────


def _sidebar():
    with st.sidebar:
        # View mode selector
        st.markdown("### 📊 Dashboard Mode")
        view_mode = st.radio(
            "Select view:",
            ["🌅 Morning Briefing", "📈 Full Dashboard"],
            horizontal=False,
            label_visibility="collapsed"
        )
        st.session_state["view_mode"] = view_mode

        st.markdown("---")
        st.markdown("### Navigation")

        if "Morning Briefing" in view_mode:
            st.markdown("**Quick daily briefing with today's top setups**")
        else:
            st.markdown("**Scanners (All Strategies)**")
            st.markdown("- 🏛️ Livermore Pivotal")
            st.markdown("- 🔄 Combined Strategy")
            st.markdown("- 🚀 Breakouts")
            st.markdown("- 📐 Minervini SEPA")
            st.markdown("- ➡️ EMA Crossover")
            st.markdown("- ⚡ Options 45-60 DTE")
            st.markdown("- 📊 RSI Scanner")
            st.markdown("- 〰️ MACD Scanner")
            st.markdown("- 🔭 Astro / Vedic")
            st.markdown("- 📡 Influencer Tracker")

        st.markdown("---")
        if st.button("↺ Refresh All", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_day_trades(_cache_key: str) -> pd.DataFrame:
    from combined_screener import run_combined_screener
    df = run_combined_screener(
        tickers=SCAN_UNIVERSE,
        min_score=4,
        max_workers=10,
        lookback_days=200,
    )
    if df.empty:
        return df
    # Prefer STRONG BUY > BUY, then by score
    order = {"⚡ STRONG BUY": 0, "✅ BUY": 1, "👀 WATCH": 2}
    df["_ord"] = df["Signal"].map(order).fillna(3)
    df = df.sort_values(["_ord", "Score"], ascending=[True, False])
    return df.drop(columns=["_ord"]).head(5).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_breakouts(_cache_key: str) -> pd.DataFrame:
    from breakout_screener import run_breakout_screener
    df = run_breakout_screener(
        tickers=SCAN_UNIVERSE,
        recent_bars=3,
        direction_filter="ALL",
    )
    if df.empty:
        return df
    # Prefer BULL setups (BULL > BEAR descending), then freshest
    df = df.sort_values(["Direction", "Bars Ago"], ascending=[False, True])
    return df.head(5).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_options(_cache_key: str) -> pd.DataFrame:
    from swing_options_screener import run_swing_options_screener
    df = run_swing_options_screener(
        tickers=SCAN_UNIVERSE,
        min_score=4.5,
        dte=50.0,
        params={"max_entry_sigma": 0.70},
    )
    if df.empty:
        return df
    ok = df[df["_passes_greeks"]].head(5)
    if len(ok) < 5:
        ok = df.head(5)
    return ok.reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_events(_cache_key: str) -> list:
    from economic_calendar import get_upcoming_events, get_earnings_calendar
    from livermore_pivotal_screener import get_sp500_tickers, get_nasdaq100_tickers
    today_str = _now_et().strftime("%Y-%m-%d")
    # Macro events today — date field is a datetime.date object, convert to str
    events = get_upcoming_events(days_ahead=1, include_earnings=False)
    today_macro = [e for e in events if str(e.get("date", "")) == today_str]
    # Earnings: scan SP500 + Nasdaq100 so we catch PANW/DG/GTLB etc.
    try:
        broad = _dedup(get_sp500_tickers() + get_nasdaq100_tickers() + list(SCAN_UNIVERSE))
        earnings = get_earnings_calendar(broad, days=1)
        today_earn = [e for e in earnings if str(e.get("date", "")) == today_str]
    except Exception:
        today_earn = []
    return today_earn + today_macro


# ── Render helpers ────────────────────────────────────────────────────────────


def _sig_badge(signal: str) -> str:
    s = signal.strip()
    if "STRONG" in s:
        return f'<span class="sig-strong">STRONG BUY</span>'
    if "BUY" in s:
        return f'<span class="sig-buy">BUY</span>'
    return f'<span class="sig-watch">WATCH</span>'


def _pct_color(val, positive_is_green: bool = True) -> str:
    try:
        v = float(val)
        cls = "green" if (v >= 0) == positive_is_green else "red"
        sign = "+" if v > 0 else ""
        return f'<span class="{cls}">{sign}{v:.2f}%</span>'
    except Exception:
        return str(val)


def _price_str(val) -> str:
    try:
        return f"${float(val):.2f}"
    except Exception:
        return str(val)


def _render_day_trades():
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.markdown('<h3>⚡ Day Trade Setups</h3>', unsafe_allow_html=True)

    cache_key = _now_et().strftime("%Y%m%d-%H")
    try:
        with st.spinner("Scanning for day trade setups…"):
            df = _fetch_day_trades(cache_key)
    except Exception as e:
        st.error(f"Day trade scan failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if df.empty:
        st.markdown('<div class="no-data">🔍 No high-conviction setups found right now. Check back soon.</div>', unsafe_allow_html=True)
    else:
        rows_html = ""
        for _, r in df.iterrows():
            entry  = _price_str(r.get("Entry", "—"))
            stop   = _price_str(r.get("Stop", "—"))
            target = _price_str(r.get("Target", "—"))
            rsi    = f'{r["RSI"]:.0f}' if pd.notna(r.get("RSI")) else "—"
            vol    = f'{r["Vol vs Avg"]:.1f}×' if pd.notna(r.get("Vol vs Avg")) else "—"
            why    = str(r.get("Why", ""))[:50]
            rows_html += f"""
            <tr>
              <td><span class="ticker">{r['Ticker']}</span></td>
              <td>{_sig_badge(r['Signal'])}</td>
              <td class="price green">{entry}</td>
              <td class="price red">{stop}</td>
              <td class="price green">{target}</td>
              <td class="price">{rsi}</td>
              <td class="price">{vol}</td>
              <td style="color:#94a3b8; font-size:12px;">{why}</td>
            </tr>"""

        st.markdown(f"""
        <table class="dash-tbl">
          <thead><tr>
            <th>Ticker</th><th>Signal</th><th>Entry</th><th>Stop</th>
            <th>Target</th><th>RSI</th><th>Vol%</th><th>Reason</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _render_breakouts():
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.markdown('<h3>🚀 Breakout Stocks</h3>', unsafe_allow_html=True)

    cache_key = _now_et().strftime("%Y%m%d-%H")
    try:
        with st.spinner("Running breakout scan…"):
            df = _fetch_breakouts(cache_key)
    except Exception as e:
        st.error(f"Breakout scan failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if df.empty:
        st.markdown('<div class="no-data">🔍 No fresh breakouts detected. Check back soon.</div>', unsafe_allow_html=True)
    else:
        rows_html = ""
        for _, r in df.iterrows():
            direction = str(r.get("Direction", "BULL"))
            dir_class = "sig-call" if direction == "BULL" else "sig-put"
            strategies = str(r.get("Signals", "—"))[:40]
            bars_ago = int(r.get("Bars Ago", 0))
            entry = _price_str(r.get("Entry", "—"))
            stop = _price_str(r.get("Stop", "—"))
            vol_ratio = f'{float(r["Vol / Avg"]):.1f}×' if "Vol / Avg" in r.index and pd.notna(r.get("Vol / Avg")) else "—"
            rows_html += f"""
            <tr>
              <td><span class="ticker">{r['Ticker']}</span></td>
              <td><span class="{dir_class}">{direction}</span></td>
              <td class="price">{entry}</td>
              <td class="price">{stop}</td>
              <td class="price">{vol_ratio}</td>
              <td class="price green">{bars_ago}d ago</td>
              <td style="color:#94a3b8;font-size:12px">{strategies}</td>
            </tr>"""

        st.markdown(f"""
        <table class="dash-tbl">
          <thead><tr>
            <th>Ticker</th><th>Direction</th><th>Entry</th><th>Stop</th>
            <th>Volume</th><th>Age</th><th>Strategy</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _render_options():
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.markdown('<h3>🎯 Options Recs (45-60 DTE)</h3>', unsafe_allow_html=True)

    cache_key = _now_et().strftime("%Y%m%d-%H")
    with st.spinner("Scanning options setups…"):
        df = _fetch_options(cache_key)

    if df.empty:
        st.markdown('<div class="no-data">🔍 No qualifying options setups found right now.</div>', unsafe_allow_html=True)
    else:
        rows_html = ""
        for _, r in df.iterrows():
            direction = str(r.get("Direction", "CALL"))
            dir_class = "sig-call" if direction == "CALL" else "sig-put"
            premium = _price_str(r.get("Premium", "—"))
            delta = f'{float(r["Delta"]):.2f}' if pd.notna(r.get("Delta")) else "—"
            theta = f'{float(r["Theta/day"]):.4f}' if pd.notna(r.get("Theta/day")) else "—"
            score = f'{float(r["Score"]):.1f}' if pd.notna(r.get("Score")) else "—"
            strike = str(int(r["Strike"])) if pd.notna(r.get("Strike")) else "—"
            greeks_ok = str(r.get("Greeks OK", "—"))
            ok_icon = "✅" if "✅" in greeks_ok else "⚠️"
            rows_html += f"""
            <tr>
              <td><span class="ticker">{r['Symbol']}</span></td>
              <td><span class="{dir_class}">{direction}</span></td>
              <td class="price">{strike}</td>
              <td class="price green">{premium}</td>
              <td class="price">{delta}</td>
              <td class="price red">{theta}</td>
              <td class="price">{score}</td>
              <td style="text-align:center; font-size: 16px;">{ok_icon}</td>
            </tr>"""

        st.markdown(f"""
        <table class="dash-tbl">
          <thead><tr>
            <th>Symbol</th><th>Type</th><th>Strike</th><th>Premium</th>
            <th>Delta</th><th>Theta/Day</th><th>Score</th><th>Greeks</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _render_events():
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.markdown('<h3>📅 Today\'s Events & Earnings</h3>', unsafe_allow_html=True)

    cache_key = _now_et().strftime("%Y%m%d")
    try:
        with st.spinner("Loading today's calendar…"):
            events = _fetch_events(cache_key)
    except Exception as e:
        st.error(f"Events calendar failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    html = ""
    for ev in events:
        impact = ev.get("impact", "LOW")
        cat    = ev.get("category", "").upper()
        name   = ev.get("event", "Unknown")
        is_earnings = cat in ("EARNINGS", "EARNING")

        if is_earnings:
            badge = '<span class="imp-earn">EARNINGS</span>'
        elif impact == "HIGH":
            badge = '<span class="imp-high">⚠️ HIGH</span>'
        elif impact == "MEDIUM":
            badge = '<span class="imp-medium">📊 MEDIUM</span>'
        else:
            continue  # skip LOW impact

        time_str = ev.get("time", "")
        html += f"""
        <div class="event-row">
          <div class="event-time">{time_str}</div>
          <div class="event-name">{name}</div>
          <div>{badge}</div>
        </div>"""

    if html:
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="no-data">📅 No major events or earnings today.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Auto-refresh logic ────────────────────────────────────────────────────────

def _handle_autorefresh():
    """
    Before market open: sleep in 60-second increments and rerun so the data
    loads fresh right when the session starts at 9:30 AM ET.
    During market hours: offer a manual refresh button; data TTL = 5 min.
    """
    secs = _secs_to_open()
    if secs > 0:
        mins = secs // 60
        wait = min(60, secs)
        st.info(f"Market opens in **{mins} min** — page will refresh automatically at 9:30 AM ET.")
        time.sleep(wait)
        st.rerun()


# ── Main layout ───────────────────────────────────────────────────────────────

def main():
    _sidebar()

    # Get view mode from session state
    view_mode = st.session_state.get("view_mode", "🌅 Morning Briefing")

    now = _now_et()
    market_open = _is_market_open()
    dot = '<span class="dot-open"></span>' if market_open else '<span class="dot-closed"></span>'
    status_text = "OPEN" if market_open else "CLOSED"
    time_str = now.strftime("%I:%M %p ET")
    date_str = now.strftime("%A, %b %d %Y")

    # ── Breadcrumb Navigation ──────────────────────────────────────────────────
    breadcrumb_text = "Morning Briefing" if "Morning" in view_mode else "Full Dashboard"
    st.markdown(f"""
    <div class="breadcrumb">
      <span style="color: #f5f7fa; font-weight: 500;">📊 Home</span>
      <span class="breadcrumb-sep">›</span>
      <span style="color: #cbd5e1;">{breadcrumb_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Status Bar ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="status-bar">
      <div style="display: flex; align-items: center; gap: 6px;">
        {dot}<span class="val">{status_text}</span>
      </div>
      <div style="border-left: 1px solid #334155; padding-left: 24px;">
        <span class="label">Time</span>&nbsp;<span class="val">{time_str}</span>
      </div>
      <div>
        <span class="label">Date</span>&nbsp;<span class="val">{date_str}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Show content based on view mode
    if "Morning" in view_mode:
        # ── MORNING BRIEFING VIEW ──────────────────────────────────────────────
        st.markdown("""
        <div class="page-header">
          <h1 class="page-title">📈 Morning Briefing</h1>
          <p class="page-subtitle">Today's trading setups, breakouts, and market events</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Scan info and controls ───────────────────────────────────────────
        n = len(SCAN_UNIVERSE)
        hcol1, hcol2, hcol3 = st.columns([3, 1, 1])
        with hcol1:
            st.markdown(
                f'<p class="page-meta">Scanning **{n}** tickers · Cache: 5 min (market hours) / 1 hr (pre-post) · Auto-refresh enabled</p>',
                unsafe_allow_html=True
            )
        with hcol2:
            st.write("")
        with hcol3:
            if st.button("↺ Refresh Now", use_container_width=True, key="refresh_btn"):
                st.cache_data.clear()
                st.rerun()

        # ── Row 1: Day Trades | Breakout Stocks ────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            _render_day_trades()
        with col2:
            _render_breakouts()

        # ── Row 2: Options | Today's Events ────────────────────────────────
        col3, col4 = st.columns([3, 2])
        with col3:
            _render_options()
        with col4:
            _render_events()

        # ── Auto-refresh before market open ────────────────────────────────
        _handle_autorefresh()

    else:
        # ── FULL DASHBOARD VIEW ────────────────────────────────────────────
        st.markdown("""
        <div class="page-header">
          <h1 class="page-title">📊 Full Dashboard</h1>
          <p class="page-subtitle">All trading scanners and analysis tools</p>
        </div>
        """, unsafe_allow_html=True)

        st.info("""
        ### 🚀 Full Dashboard Features Available:

        **Strategy Scanners:**
        - 🏛️ Livermore Pivotal Points
        - 🎯 Combined Strategy
        - 🚀 Breakout Detection
        - 📐 Minervini SEPA
        - ➡️ EMA Crossover
        - 📊 RSI Scanner
        - 〰️ MACD Scanner

        **Specialized Tools:**
        - ⚡ Options 45-60 DTE
        - 🔭 Financial Astrology
        - 📡 Influencer Tracker
        - 🔔 Daily Alerts Engine
        - 📈 Stock Analyzer
        - 📅 Macro Calendar

        Use the sidebar to select any scanner to dive deep into specific analysis.
        """, icon="📊")

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <p style="color: #94a3b8; font-size: 14px;">
                💡 Tip: Select a scanner from the sidebar to begin detailed analysis
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
