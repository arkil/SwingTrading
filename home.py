"""
Main Trading Dashboard — http://localhost:8501/
Run: streamlit run home.py --server.port 8501

Sections
  • 5 Day Trade Recommendations (gap + momentum signals, today's freshest setups)
  • 5 Breakout Stocks (NR7, BO-52W, BB-squeeze, MA-reclaim, inside-bar)
  • 5 Options Recommendations (45-60 DTE swing options, best greek profile)
  • Today's Earnings & Major Events

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
    padding: 1.2rem 2rem 2rem !important;
    max-width: 1500px;
}
/* Section card */
.dash-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem 1.2rem 1.2rem;
    margin-bottom: 1rem;
}
.dash-card h3 {
    color: #e2e8f0;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: .6px;
    margin: 0 0 .7rem;
    text-transform: uppercase;
}
/* Status bar */
.status-bar {
    display: flex;
    gap: 1.4rem;
    align-items: center;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: .55rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 13px;
    color: #94a3b8;
}
.status-bar .label { color: #475569; font-size: 11px; text-transform: uppercase; letter-spacing: .8px; }
.status-bar .val   { color: #f1f5f9; font-weight: 600; }
.status-bar .dot-open  { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; display:inline-block; margin-right:5px; }
.status-bar .dot-closed { width: 8px; height: 8px; border-radius: 50%; background: #ef4444; display:inline-block; margin-right:5px; }
/* Signal badges */
.sig-strong { background: #14532d; color: #86efac; padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: 700; }
.sig-buy    { background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: 700; }
.sig-watch  { background: #1c1917; color: #fbbf24; padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: 700; }
.sig-call   { background: #052e16; color: #34d399; padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: 700; }
.sig-put    { background: #450a0a; color: #f87171; padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: 700; }
.sig-bo     { background: #1e1b4b; color: #a5b4fc; padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: 700; }
/* Impact badges */
.imp-high   { background: #7f1d1d; color: #fca5a5; padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: 700; }
.imp-medium { background: #78350f; color: #fcd34d; padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: 700; }
.imp-earn   { background: #164e63; color: #67e8f9; padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: 700; }
/* Ticker cells */
.ticker { font-size: 15px; font-weight: 700; color: #f8fafc; letter-spacing: .5px; }
.price  { font-size: 13px; color: #94a3b8; }
.green  { color: #4ade80; font-weight: 600; }
.red    { color: #f87171; font-weight: 600; }
/* Compact table rows */
table.dash-tbl { width: 100%; border-collapse: collapse; font-size: 13px; }
table.dash-tbl th {
    color: #475569; font-size: 10px; text-transform: uppercase; letter-spacing: .8px;
    padding: 4px 8px; border-bottom: 1px solid #1e293b; text-align: left; font-weight: 600;
}
table.dash-tbl td {
    padding: 6px 8px; border-bottom: 1px solid #0f172a; color: #cbd5e1; vertical-align: middle;
}
table.dash-tbl tr:last-child td { border-bottom: none; }
table.dash-tbl tr:hover td { background: #1e293b; }
/* Event row */
.event-row { padding: 6px 0; border-bottom: 1px solid #1e293b; display: flex; gap: 10px; align-items: flex-start; }
.event-row:last-child { border-bottom: none; }
.event-time { color: #475569; font-size: 11px; min-width: 55px; padding-top: 3px; }
.event-name { color: #e2e8f0; font-size: 13px; flex: 1; }
/* No data */
.no-data { color: #475569; font-size: 13px; padding: 1rem 0; text-align: center; }
/* Refresh bar */
.refresh-info { color: #475569; font-size: 11px; }
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
        st.markdown("### Navigation")
        st.page_link("home.py",      label="Morning Briefing",  icon="🌅")
        st.markdown("---")
        st.markdown("**Full Dashboard** → [localhost:8502](http://localhost:8502)")
        st.markdown("---")
        st.markdown("**Scanners (Full Dashboard)**")
        st.markdown("- [Livermore Pivotal](http://localhost:8502/?scanner=livermore)")
        st.markdown("- [Combined Strategy](http://localhost:8502/?scanner=combined)")
        st.markdown("- [Breakouts](http://localhost:8502/?scanner=breakout)")
        st.markdown("- [Minervini SEPA](http://localhost:8502/?scanner=minervini)")
        st.markdown("- [EMA Crossover](http://localhost:8502/?scanner=ema)")
        st.markdown("- [Options 45-60 DTE](http://localhost:8502/?scanner=swing_opts)")
        st.markdown("- [RSI Scanner](http://localhost:8502/?scanner=rsi)")
        st.markdown("- [MACD Scanner](http://localhost:8502/?scanner=macd)")
        st.markdown("- [Astro / Vedic](http://localhost:8501/?scanner=astro)")
        st.markdown("- [📡 Influencer Tracker](http://localhost:8501/?scanner=influencers)")
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
        st.markdown('<div class="no-data">No high-conviction setups right now.</div>', unsafe_allow_html=True)
    else:
        rows_html = ""
        for _, r in df.iterrows():
            entry  = _price_str(r.get("Entry", "—"))
            stop   = _price_str(r.get("Stop", "—"))
            target = _price_str(r.get("Target", "—"))
            rsi    = f'{r["RSI"]:.0f}' if pd.notna(r.get("RSI")) else "—"
            vol    = f'{r["Vol vs Avg"]:.1f}×' if pd.notna(r.get("Vol vs Avg")) else "—"
            why    = str(r.get("Why", ""))[:60]
            rows_html += f"""
            <tr>
              <td><span class="ticker">{r['Ticker']}</span></td>
              <td>{_sig_badge(r['Signal'])}</td>
              <td class="green">{entry}</td>
              <td class="red">{stop}</td>
              <td class="green">{target}</td>
              <td>{rsi}</td>
              <td>{vol}</td>
              <td style="color:#64748b;font-size:11px">{why}</td>
            </tr>"""

        st.markdown(f"""
        <table class="dash-tbl">
          <thead><tr>
            <th>Ticker</th><th>Signal</th><th>Entry</th><th>Stop</th>
            <th>Target</th><th>RSI</th><th>Vol</th><th>Why</th>
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
        st.markdown('<div class="no-data">No fresh breakouts detected.</div>', unsafe_allow_html=True)
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
              <td>{entry}</td>
              <td>{stop}</td>
              <td>{vol_ratio}</td>
              <td style="color:#4ade80">{bars_ago}d ago</td>
              <td style="color:#64748b;font-size:11px">{strategies}</td>
            </tr>"""

        st.markdown(f"""
        <table class="dash-tbl">
          <thead><tr>
            <th>Ticker</th><th>Dir</th><th>Entry</th><th>Stop</th>
            <th>Vol</th><th>Age</th><th>Strategy</th>
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
        st.markdown('<div class="no-data">No qualifying options setups found.</div>', unsafe_allow_html=True)
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
            ok_icon = "✅" if "✅" in greeks_ok else "❌"
            rows_html += f"""
            <tr>
              <td><span class="ticker">{r['Symbol']}</span></td>
              <td><span class="{dir_class}">{direction}</span></td>
              <td>{strike}</td>
              <td class="green">{premium}</td>
              <td>{delta}</td>
              <td style="color:#f87171">{theta}</td>
              <td style="color:#e2e8f0">{score}</td>
              <td style="text-align:center">{ok_icon}</td>
            </tr>"""

        st.markdown(f"""
        <table class="dash-tbl">
          <thead><tr>
            <th>Symbol</th><th>Type</th><th>Strike</th><th>Premium</th>
            <th>Delta</th><th>Theta/d</th><th>Score</th><th>Greeks</th>
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
            badge = '<span class="imp-high">HIGH</span>'
        elif impact == "MEDIUM":
            badge = '<span class="imp-medium">MED</span>'
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
        st.markdown('<div class="no-data">No major events or earnings today.</div>', unsafe_allow_html=True)

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
    now = _now_et()
    market_open = _is_market_open()
    dot = '<span class="dot-open"></span>' if market_open else '<span class="dot-closed"></span>'
    status_text = "OPEN" if market_open else "CLOSED"
    time_str = now.strftime("%I:%M %p ET")
    date_str = now.strftime("%A, %b %d %Y")

    st.markdown(f"""
    <div class="status-bar">
      <div>{dot}<span class="val">{status_text}</span></div>
      <div><span class="label">Time</span>&nbsp;<span class="val">{time_str}</span></div>
      <div><span class="label">Date</span>&nbsp;<span class="val">{date_str}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Header with refresh + link to full scanner dashboard
    hcol1, hcol2, hcol3 = st.columns([4, 1, 1])
    with hcol1:
        st.markdown("## Trading Dashboard")
        n = len(SCAN_UNIVERSE)
        st.markdown(f'<span class="refresh-info">Scanning {n} tickers · cache 5 min market hours · 1 hr pre/post</span>',
                    unsafe_allow_html=True)
    with hcol2:
        st.write("")
        st.write("")
        st.link_button("Full Dashboard →", "http://localhost:8502", use_container_width=True)
    with hcol3:
        st.write("")
        st.write("")
        if st.button("↺ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Row 1: Day Trades | Breakout Stocks ───────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        _render_day_trades()
    with col2:
        _render_breakouts()

    # ── Row 2: Options | Today's Events ───────────────────────────────────────
    col3, col4 = st.columns([3, 2])
    with col3:
        _render_options()
    with col4:
        _render_events()

    # ── Auto-refresh before market open ───────────────────────────────────────
    _handle_autorefresh()


if __name__ == "__main__":
    main()
