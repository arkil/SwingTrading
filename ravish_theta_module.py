"""
Ravish Theta Machine — three option-selling setups
==================================================
After the KINFO "Undiscovered Traders" interview with Ravish (Options with Ravish),
"The Delta-Neutral Setup: How To Profit In Any Market Direction"
(youtube.com/watch?v=mnDFfC1QhZ4):

  1. Overnight earnings iron fly: sell the ATM straddle 15 min before the close into
     the report, wings ~10-20 pts out; take profit in the first 5-10 min next morning.
  2. Theta machine calendar: sell the ~20-30 delta short-dated option, buy the same
     strike a week+ later; book +30-50%; close before the short leg expires.
  3. Post-earnings iron condor: ~30 days out, shorts ~10% OTM both sides, hedged wings;
     exit if price leaves the range.
  4. RSI-oversold bull put spread (youtube.com/watch?v=BCeo6rypt-4): when daily RSI(14)
     is under 35, sell the ATM SPX put / buy 10 points lower, ~14 DTE, TP 50%, no stop.

The strategy + backtester live in the CBT project
  /Users/arkilthakkar/workplace/strategies/ravish_theta/
This page reads that project's live scan (Data/scan_latest.json) — Alpaca option
chain snapshots with real bid/ask + greeks. Rules + PASS / WATCH only.
Defined-risk structures, sized off cash, never margin. Educational — not advice.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd

STRAT_DIR = Path("/Users/arkilthakkar/workplace/strategies/ravish_theta")
VENV_PY = STRAT_DIR / ".venv" / "bin" / "python"
SCAN_JSON = STRAT_DIR / "Data" / "scan_latest.json"
ENV_FILE = Path(__file__).parent / ".env"

_PLAYBOOK = {
    "earnings_fly": """
**Overnight earnings iron fly** — *"the market makers are expecting a 5% move, but historically it moves less"*

1. Only names with a known **after-close (AMC)** or **before-open (BMO)** report.
2. **Enter 15:45 ET** on the last session before the report: sell the ATM call + ATM put,
   buy a call and a put **~9% of spot** out (his NVDA@225 example: 205 / 245 wings).
3. **Exit 09:40 ET** the next morning, whatever the P&L — take what's there in the first 5–10 minutes.
4. **PASS** needs all of: implied move (ATM straddle ÷ spot) **>** the stock's average
   overnight earnings move (last 8 reports) · report time known · short-leg bid/ask ≤ 10% of mid.
5. Max loss is capped by the wings. Size so a full max loss is a small slice of cash.
""",
    "calendar": """
**Theta machine calendar — VIX regime switch** *(his calendar, refined by out-of-sample research on QQQ/SPY)*

1. Check **VIX** at the close.
   * **VIX ≥ 18** → **call calendar**: sell the **25-delta call** ~**4 weeks** out, buy the same
     strike **2 weeks later**. (After a volatility spike the index usually grinds back up.)
   * **VIX < 18** → **put calendar**: same structure on the **25-delta put** below spot.
2. Net debit = max loss. **No stop** — size it so the full debit is an acceptable loss.
3. **Take profit at +30%** of the debit. Otherwise **close the session before the short leg expires**.
4. One open calendar per symbol at a time. Enter weekly (Mondays) while the rule holds.
5. **PASS** needs: regime matches the side · positive debit · long leg ~14 days after the short
   (±4) · leg bid/ask ≤ 10% of mid.
6. It loses when the index keeps trending against the short strike (a selloff that doesn't
   bounce, e.g. Feb–Mar 2025). Keep each one small.

*His own published double calendar (shorts 10–15 days out, long the next session, TP 15–20%,
stop on a strike touch) was also checked and is not used here.*
""",
    "condor": """
**Post-earnings iron condor** — *"after earnings there's a choppy period when the market tries to discover what's next"*

1. Enter **2+ sessions after the earnings reaction day** (window here: 2–8 sessions).
2. Expiry **~30 days** out. **Sell ~10% OTM** put and call, **buy wings ~3% further** out —
   never the naked strangle ("max loss infinite").
3. **Take profit at 50%** of the credit. **Exit if the stock closes outside the short strikes.**
   Otherwise hold to expiry.
4. **PASS** needs: inside the post-earnings window · credit ≥ 15% of the wing width ·
   short-leg bid/ask ≤ 15% of mid.
5. Step aside around macro events (FOMC, war headlines) — *"don't sell flood insurance
   during a flood warning."*
""",
    "rsi_put_spread": """
**RSI-oversold ATM bull put spread (SPX)** — *"enter after a big sell-off when RSI is oversold, dip buyers step in"*

1. Daily chart, **RSI(14)** with the lower band moved from 30 to **35**.
2. **In the morning**, if RSI is **under 35**: sell the **ATM put (~50 delta)**, buy the put **10 SPX
   points lower**, expiry **~2 weeks** out. Credit is about half the width (~$450–500 per SPX spread).
3. **Immediately place a GTC closing order at 50% of the credit.** No stop — set and forget.
   Without a fill, the spread settles at expiry (SPX is cash-settled, European).
4. Max loss = width − credit (~$500–550 per SPX spread). A loss is usually the **full** max loss —
   size so one full loss is small relative to cash.
5. If RSI stays under 35 and price keeps falling you *may* add a second spread at the new ATM strike.
6. **PASS** (SPY row) needs: RSI < 35 · credit ≥ 35% of width · leg bid/ask ≤ 10% of mid.
   The SPX row shows the strikes only — Alpaca has no SPX quotes; price it at your broker.
7. **SPY proxy:** a 1-wide SPY put spread ≈ 1/10 of a 10-wide SPX spread (SPY settles to shares —
   close it before expiry if it's in the money).
""",
}

_COLS = {
    "earnings_fly": ["tier", "symbol", "report", "timing", "enter", "exit", "expiry", "spot", "legs",
                     "credit_mid", "credit_natural", "max_loss", "profit_zone",
                     "implied_move_pct", "hist_avg_move_pct", "short_spread_pct", "fails"],
    "calendar": ["tier", "symbol", "variant", "vix", "spot", "legs", "short_delta", "iv_slope", "debit_mid",
                 "debit_natural", "max_loss", "take_profit_at", "close_by", "fails"],
    "condor": ["tier", "symbol", "last_report", "sessions_since", "expiry", "spot", "legs", "range",
               "credit_mid", "max_loss", "take_profit_at", "pop_est", "fails"],
    "rsi_put_spread": ["tier", "symbol", "rsi", "rsi_asof", "spot", "expiry", "legs", "short_delta",
                       "credit_mid", "credit_natural", "max_loss", "take_profit_at", "note", "fails"],
}

_LABELS = {"credit_mid": "credit $ (mid)", "credit_natural": "credit $ (natural)", "max_loss": "max loss $",
           "debit_mid": "debit $ (mid)", "debit_natural": "debit $ (natural)", "take_profit_at": "TP at $",
           "stop_at": "stop at $", "implied_move_pct": "implied move %", "hist_avg_move_pct": "hist avg move %",
           "short_spread_pct": "short b/a %", "pop_est": "POP est %", "iv_slope": "back/front IV −1", "sessions_since": "sessions since rpt",
           "rsi": "RSI(14)", "rsi_asof": "RSI as of"}


def _env() -> dict:
    env = dict(os.environ)
    try:
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return env


def load_scan() -> dict:
    try:
        return json.loads(SCAN_JSON.read_text())
    except Exception:
        return {}


def run_scan(only: str | None = None):
    cmd = [str(VENV_PY), "scan.py"] + (["--only", only] if only else [])
    try:
        r = subprocess.run(cmd, cwd=STRAT_DIR, capture_output=True, text=True, timeout=900, env=_env())
        return r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _table(rows: list, kind: str):
    import streamlit as st
    if not rows:
        st.info("No candidates in the latest scan.")
        return
    df = pd.DataFrame(rows)
    cols = [c for c in _COLS[kind] if c in df.columns]
    df = df[cols].copy()
    df["fails"] = df["fails"].apply(lambda f: ", ".join(f).replace("_", " ") if f else "—")
    df = df.rename(columns=_LABELS)

    def _tier_style(v):
        return ("background-color:#1b3a2a;font-weight:700" if v == "PASS"
                else "background-color:#3a331b" if v == "WATCH" else "")
    sty = df.style.map(_tier_style, subset=["tier"]).format(precision=2, na_rep="—")
    st.dataframe(sty, use_container_width=True, hide_index=True)


PAPER_DIR = STRAT_DIR / "Data" / "paper"


def _paper_panel():
    """Alpaca PAPER forward-test of the calendar rule (paper_trader.py, launchd
    com.swingtrading.ravish-calendar-paper, weekdays 15:30 ET). Actual paper fills only."""
    import streamlit as st
    st.markdown("##### 🧪 Alpaca paper forward-test")
    try:
        state = json.loads((PAPER_DIR / "state.json").read_text())
    except Exception:
        state = {"open": [], "closed": []}
    st.caption("Runs weekdays ~15:30 ET on the paper account: exits every day, new calendars on the "
               "first session of each week. Only its own `rtcal-` orders are touched.")
    if state["open"]:
        st.markdown("**Open**")
        st.dataframe(pd.DataFrame(state["open"])[["symbol", "side", "legs", "qty", "entry_date", "entry_debit",
                                                  "entry_mid_at_decision", "close_by", "vix_at_entry"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No open paper calendars.")
    if state["closed"]:
        st.markdown("**Closed**")
        st.dataframe(pd.DataFrame(state["closed"])[["symbol", "side", "legs", "closed_qty", "entry_date",
                                                    "entry_debit", "exit_date", "exit_reason", "exit_price", "pnl",
                                                    "ror"]],
                     use_container_width=True, hide_index=True)
    try:
        lines = (PAPER_DIR / "events.jsonl").read_text().splitlines()[-12:]
        with st.expander("Bot log (last 12 events)"):
            st.code("\n".join(lines))
    except Exception:
        pass


def render_ravish_theta():
    import streamlit as st

    st.markdown(
        "<div class='scanner-header'><span style='font-size:2rem'>⏳</span>"
        "<span class='scanner-title'>Ravish Theta Machine</span></div>"
        "<div class='scanner-desc'>Defined-risk option-selling setups from Ravish — "
        "<b>overnight earnings iron fly</b>, <b>theta-machine calendar</b>, "
        "<b>post-earnings iron condor</b> (KINFO interview) and the <b>RSI-oversold SPX bull put spread</b>. No charts: trades come from option Greeks, implied vs "
        "historical move and the calendar. Live Alpaca chain quotes. Rules + PASS / WATCH only. "
        "Cash-sized, no margin. Educational, not advice.</div>",
        unsafe_allow_html=True)

    scan = load_scan()
    top = st.columns([3, 1])
    if scan.get("generated_at"):
        top[0].caption(f"Last scan {scan['generated_at']} · run during market hours for live quotes")
    else:
        top[0].warning("No scan yet — click **Refresh all** (market hours).")
    if top[1].button("🔄 Refresh all", key="rt_refresh_all", use_container_width=True):
        with st.spinner("Pulling live chains for all setups (~1–2 min)…"):
            ok, log = run_scan()
        if ok:
            st.rerun()
        st.error("Scan failed")
        st.code(log)

    tabs = st.tabs(["💥 Earnings iron fly", "⏳ Theta-machine calendar", "🦅 Post-earnings condor",
                    "📉 RSI bull put (SPX)", "📘 Notes"])
    for tab, kind in zip(tabs[:4], ["earnings_fly", "calendar", "condor", "rsi_put_spread"]):
        with tab:
            rows = scan.get(kind, [])
            n_pass = sum(r.get("tier") == "PASS" for r in rows)
            c = st.columns([3, 1])
            c[0].markdown(f"**{n_pass} PASS** · {len(rows) - n_pass} WATCH · per 1 contract")
            if c[1].button("Refresh this", key=f"rt_refresh_{kind}", use_container_width=True):
                with st.spinner("Scanning…"):
                    ok, log = run_scan(kind)
                if ok:
                    st.rerun()
                st.code(log)
            _table(rows, kind)
            with st.expander("Rules", expanded=not rows):
                st.markdown(_PLAYBOOK[kind].replace("$", "\\$"))
            if kind == "calendar":
                _paper_panel()

    with tabs[4]:
        st.markdown(
            "* **Why these work (his thesis):** ~80% of options expire worthless, and option sellers "
            "collect the risk premium baked into implied vol. Be the casino, not the gambler.\n"
            "* **Risk:** always buy the wings. Step aside around FOMC, CPI and geopolitical shocks.\n"
            "* **Learning curve:** trade **1 contract for 3–6 months** (20–50 trades) before sizing up.\n"
            "* **Recency:** judge a setup on the last 1–2 years. A stock that behaved 5 years ago is a different stock now.\n"
            "* **Mid vs natural:** *mid* = midpoint fill, *natural* = crossing the full spread. "
            "Work limit orders near mid.\n"
            f"* Strategy code, config and backtester: `{STRAT_DIR}`")
        errs = scan.get("errors") or []
        if errs:
            with st.expander(f"Scan warnings ({len(errs)})"):
                st.code("\n".join(errs))
