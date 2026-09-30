"""
Ravish expected-move calendar spreads (?scanner=ravish_theta&view=calendar_em)
==============================================================================
From "the ultimate theta machine" video (youtube.com/watch?v=GuI-hH_jhlg):

  * Sell a ~35 DTE option, buy the same strike ~1 week later (SPY: sell Jun 18 / buy Jun 26).
  * Neutral = ATM call calendar. Bullish = call calendar one expected move ABOVE spot.
    Bearish = put calendar one expected move BELOW. His favourite: the double calendar (both).
  * Enter only when implied vol is in the LOW part of its 1-year range (below the median);
    stay away from earnings (IV crush).
  * No stop; start scaling out at +20-30%; close before the short leg expires.

Live suggestions come from the ravish_theta CBT project's scan (scan.py --only calendar_em):
Alpaca chain quotes, EM = 0.85 x ATM straddle, 1-year ATM IV history from Alpaca option prints.
Rules + PASS / WATCH only. Defined risk, sized off cash. Educational, not advice.
"""
from __future__ import annotations

import pandas as pd

from ravish_theta_module import STRAT_DIR, _explicit_legs, load_scan, run_scan

_RULES = """
**Expected-move calendar** — *"the ultimate theta machine"*

1. **Structure:** sell an option **~35 days** out, buy the **same strike ~1 week later**.
   Net debit = max loss. The short decays faster, so the gap between the two widens with time.
2. **Pick the strike by your view** — peak profit is at the strike on the short expiry:
   * **Neutral** → at the money (call or put calendar, same thing).
   * **Bullish** → **call calendar at spot + expected move** (EM of the short expiry).
   * **Bearish** → **put calendar at spot − expected move**. Also a hedge: a drop brings price
     toward the strike *and* lifts IV.
   * **Double calendar** (his preferred) → both EM calendars at once; wider tent, less direction risk.
3. **IV filter:** enter only when **implied vol is below its 1-year median** — calendars are long
   vega, a rising IV widens the profit tent. Don't enter with IV at the top of its range.
4. **No earnings** before the long leg expires — the IV crush hurts the long leg.
5. **Exits:** no stop (size small). **Scale out** as it rises — e.g. 1 of 10 at +25%, 1 at +50%,
   2 at +100%, 3 at +200%. **Close before the short leg expires** (assignment risk).
6. **PASS** needs: IV below 1-yr median · no earnings before the back expiry · positive debit ·
   back leg 7 ± 2 days after the short · each leg's bid/ask ≤ 10% of mid.

*Expected move = 0.85 × ATM front straddle (the broker-style number). IV history = the stock's own
~35-day ATM implied vol per session over the last year, from Alpaca option prints.*
"""

_VIEWS = {"All": None, "Neutral (ATM)": "Neutral", "Bullish (call +EM)": "Bullish",
          "Bearish (put −EM)": "Bearish", "Double": "Double"}

_COLS = ["tier", "symbol", "variant", "legs", "dte_front", "debit_mid", "debit_natural", "max_loss",
         "max_profit_est", "reward_risk", "peak_at", "pnl_at_spot", "breakeven", "scale_out_at", "close_by", "fails"]
_LABELS = {"dte_front": "DTE", "debit_mid": "debit $ (mid)", "debit_natural": "debit $ (natural)",
           "max_loss": "max loss $", "max_profit_est": "max profit $ (model)", "reward_risk": "reward : risk",
           "peak_at": "peak at", "pnl_at_spot": "P&L $ if price unchanged (model)",
           "breakeven": "profit zones (model)", "scale_out_at": "scale out at $ value",
           "close_by": "close by"}


def _tier_style(v):
    return ("background-color:#1b3a2a;font-weight:700" if v == "PASS"
            else "background-color:#3a331b" if v == "WATCH" else "")


def _iv_board(df: pd.DataFrame):
    import streamlit as st
    g = df.groupby("symbol", sort=False).first().reset_index()
    g["IV gate"] = ["✅ low" if (a is not None and b is not None and a < b) else "❌ high"
                    for a, b in zip(g["iv_now"], g["iv_median_1y"])]
    g["earnings before back"] = [("—" if e in ("—", "none") else "⚠️ date unknown" if e == "unknown" else f"⚠️ {e}")
                                 for e in g["earnings"]]
    g["passing setups"] = [int(((df.symbol == s) & (df.tier == "PASS")).sum()) for s in g["symbol"]]
    g = g.sort_values("iv_pctile_1y")
    st.dataframe(
        g[["symbol", "spot", "iv_now", "iv_median_1y", "iv_pctile_1y", "IV gate", "em", "em_pct",
           "earnings before back", "passing setups"]],
        hide_index=True, use_container_width=True,
        column_config={
            "iv_now": st.column_config.NumberColumn("ATM IV %", format="%.1f"),
            "iv_median_1y": st.column_config.NumberColumn("1-yr median %", format="%.1f"),
            "iv_pctile_1y": st.column_config.ProgressColumn("IV percentile (1 yr)", min_value=0, max_value=100,
                                                            format="%d"),
            "em": st.column_config.NumberColumn("expected move ±$", format="%.2f"),
            "em_pct": st.column_config.NumberColumn("EM %", format="%.1f"),
            "spot": st.column_config.NumberColumn("spot", format="%.2f"),
        })


def render_calendar_em():
    import streamlit as st

    st.markdown(
        "<div class='scanner-header'><span style='font-size:2rem'>📅</span>"
        "<span class='scanner-title'>Expected-Move Calendar Spreads</span></div>"
        "<div class='scanner-desc'>Ravish's calendar spreads: sell ~35 DTE, buy the same strike a week later. "
        "<b>Neutral</b> at the money, <b>bullish</b> call calendar one expected move up, <b>bearish</b> put "
        "calendar one expected move down, or the <b>double calendar</b>. Only when IV is below its 1-year "
        "median. Live Alpaca quotes · rules + PASS / WATCH only · defined risk, cash-sized. "
        "Educational, not advice.</div>",
        unsafe_allow_html=True)

    scan = load_scan()
    rows = scan.get("calendar_em") or []
    top = st.columns([3, 1])
    if scan.get("generated_at") and rows:
        top[0].caption(f"Last scan {scan['generated_at']} · per 1 contract (×100) · refresh in market hours "
                       "for tradable quotes")
    else:
        top[0].warning("No calendar scan yet — click **Refresh** (market hours for live quotes).")
    if top[1].button("🔄 Refresh", key="cem_refresh", use_container_width=True):
        with st.spinner("Pulling chains + IV history (~20 s cached, ~2 min first run)…"):
            ok, log = run_scan("calendar_em")
        if ok:
            st.rerun()
        st.error("Scan failed")
        st.code(log)
    if not rows:
        with st.expander("Rules", expanded=True):
            st.markdown(_RULES.replace("$", "\\$"))
        return

    df = pd.DataFrame(rows)
    st.markdown("##### 1 · IV check — is volatility low enough to buy a calendar?")
    _iv_board(df)

    st.markdown("##### 2 · Calendar suggestions")
    c = st.columns([3, 1])
    view = c[0].radio("View", list(_VIEWS), horizontal=True, key="cem_view", label_visibility="collapsed")
    pass_only = c[1].toggle("PASS only", value=True, key="cem_pass_only")
    sub = df
    if _VIEWS[view]:
        sub = sub[sub.variant.str.startswith(_VIEWS[view])]
    if pass_only:
        sub = sub[sub.tier == "PASS"]
    n_pass = int((df.tier == "PASS").sum())
    st.caption(f"**{n_pass} PASS** · {len(df) - n_pass} WATCH across {df.symbol.nunique()} symbols · "
               "ranked by IV percentile (lowest first). *Model* columns = value at the short expiry with "
               "the long leg at today's IV, like the broker's P&L curve — not a forecast. A double calendar has two profit zones with a losing dip between them.")
    if sub.empty:
        st.info("Nothing passes right now in this view — switch off **PASS only** to see the WATCH setups "
                "and which rule each one fails.")
    else:
        t = sub.copy()
        t["legs"] = t["legs"].map(_explicit_legs)
        t["fails"] = t["fails"].apply(lambda f: ", ".join(f).replace("_", " ") if f else "—")
        t = t[[x for x in _COLS if x in t.columns]].rename(columns=_LABELS)
        st.dataframe(t.style.map(_tier_style, subset=["tier"]).format(precision=2, na_rep="—"),
                     use_container_width=True, hide_index=True,
                     column_config={"legs": st.column_config.TextColumn("legs (order)", width="large")})

    with st.expander("Rules", expanded=False):
        st.markdown(_RULES.replace("$", "\\$"))
    with st.expander("Risks (from the video)"):
        st.markdown(
            "* **Range trade:** a single call calendar loses if price falls away, a put calendar if it "
            "rallies away — the double calendar widens the range.\n"
            "* **Volatility crush:** entering while IV is falling, or across earnings, hurts the long leg.\n"
            "* **Assignment:** the short leg expires first; if it's in the money you can be assigned. "
            "Close before the short expiry.\n"
            "* **Sizing:** no stop on single calendars — only risk what you're willing to lose; "
            "any option trade can be wiped out in a black-swan move.\n"
            f"* Code, config and scanner: `{STRAT_DIR}` (`scan.py --only calendar_em`, `src/calendar_em.py`)")
    errs = [e for e in (scan.get("errors") or []) if e.startswith("calendar_em")]
    if errs:
        with st.expander(f"Scan warnings ({len(errs)})"):
            st.code("\n".join(errs))
