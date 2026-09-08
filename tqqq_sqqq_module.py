"""
Dashboard bridge for the tqqq_sqqq_ensemble strategy.

The strategy itself lives in the CBT project at
  /Users/arkilthakkar/workplace/strategies/tqqq_sqqq_ensemble/
This module just surfaces its latest backtest + today's signal inside the
SwingTrading dashboard (scanner id: `tqqq_sqqq`).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

STRAT_DIR = Path("/Users/arkilthakkar/workplace/strategies/tqqq_sqqq_ensemble")
VENV_PY = Path("/Users/arkilthakkar/workplace/Scripts/SwingTrading/.venv/bin/python")


def _load(kind: str) -> dict | None:
    f = STRAT_DIR / f"validated_config_{kind}.json"
    if not f.exists():
        f = STRAT_DIR / "validated_config.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def run_backtest(kind: str = "synth", fill_lag: int = 0) -> tuple[bool, str]:
    cmd = [str(VENV_PY), "backtest.py", "--data", kind, "--fill-lag", str(fill_lag)]
    try:
        r = subprocess.run(cmd, cwd=STRAT_DIR, capture_output=True, text=True, timeout=600)
        return r.returncode == 0, (r.stdout + r.stderr)[-6000:]
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def refresh_signal() -> tuple[bool, str]:
    """Re-fetch data + rerun both backtests (call from a daily cron / button)."""
    out = []
    for step in (["fetch_data.py"], ["backtest.py", "--data", "synth"],
                 ["backtest.py", "--data", "real"]):
        r = subprocess.run([str(VENV_PY), *step], cwd=STRAT_DIR,
                           capture_output=True, text=True, timeout=900)
        out.append(f"$ {' '.join(step)}\n{(r.stdout + r.stderr)[-2000:]}")
        if r.returncode != 0:
            return False, "\n\n".join(out)
    return True, "\n\n".join(out)


def latest() -> dict:
    return {"synth": _load("synth"), "real": _load("real")}


# ---------------------------------------------------------------- streamlit view
def render():
    import streamlit as st

    st.markdown(
        "<div class='scanner-header'><span style='font-size:2rem'>🚀</span>"
        "<span class='scanner-title'>TQQQ / SQQQ Ensemble</span></div>"
        "<div class='scanner-desc'>Reconstruction of Mallik / RealTQQQTrader's WhiteLight system — "
        "a non-correlated ensemble (momentum + mean-reversion) that holds only TQQQ in NDX "
        "uptrends and SQQQ in downtrends. Signals computed on ^NDX, sized by how extended NDX is.</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    data = latest()
    if not data["synth"] and not data["real"]:
        st.warning("No backtest yet. Click **Refresh** to fetch data and run it.", icon="⚠️")
        if st.button("🔄 Refresh (fetch + backtest)", key="tqs_refresh_first"):
            with st.spinner("Fetching ^NDX / TQQQ / SQQQ and running 40-year backtest…"):
                ok, log = refresh_signal()
            st.code(log)
            st.rerun() if ok else st.error("Refresh failed — see log above.")
        return

    ref = data["real"] or data["synth"]
    sig = ref["signal"]
    net = sig.get("net_target", sig.get("target_tqqq", 0) - sig.get("target_sqqq", 0))

    stale = (datetime.now().date() - datetime.fromisoformat(sig["as_of"]).date()).days
    side = "TQQQ" if net > 0.02 else "SQQQ" if net < -0.02 else "CASH"
    pct = abs(net)
    px = sig.get("tqqq_price") if side == "TQQQ" else sig.get("sqqq_price") if side == "SQQQ" else None

    # ═══════════════ SCANNER ═══════════════
    color = {"TQQQ": "#1b3a2a", "SQQQ": "#3a1b1b", "CASH": "#26324a"}[side]
    hdr = (f"BUY / HOLD {side} — {pct:.0%} of account" if side != "CASH"
           else "GO TO CASH — 100%")
    st.markdown(
        f"<div style='background:{color};padding:18px 22px;border-radius:12px'>"
        f"<div style='font-size:0.8rem;opacity:0.65'>SCANNER · signal {sig['as_of']} · "
        f"{sig['regime']} regime · net {net:+.2f}"
        f"{' · ⚠️ DE-RISK ACTIVE' if sig.get('derisk') else ''}</div>"
        f"<div style='font-size:1.7rem;font-weight:800;margin-top:2px'>{hdr}</div></div>",
        unsafe_allow_html=True,
    )
    if stale > 4:
        st.error(f"Signal is {stale} days old — click Refresh before acting.")

    c = st.columns(5)
    c[0].metric("NDX", f"{sig['ndx_close']:,.0f}")
    c[1].metric("vs 250-DMA", f"{sig['ndx_close']/sig['ma_slow']-1:+.1%}")
    c[2].metric("RSI(3)", f"{sig['rsi3']:.0f}")
    c[3].metric("TQQQ", f"${sig.get('tqqq_price','?')}")
    c[4].metric("SQQQ", f"${sig.get('sqqq_price','?')}")

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("🔄 Refresh", key="tqs_refresh"):
        with st.spinner("Fetching + backtesting…"):
            ok, _log = refresh_signal()
        st.rerun() if ok else st.error("Refresh failed.")
    b2.caption("Run once daily ~15:50 ET")

    st.divider()

    # ═══════════════ TRADE TICKET ═══════════════
    lv = sig.get("levels", {})
    m = ref["metrics"]
    acct = st.number_input("Amount you're allocating to this strategy ($)",
                           min_value=100, value=5000, step=500, key="tqs_acct",
                           help="Only what you'd put in THIS strategy — not your whole account.")
    half = st.toggle("Half size (until you've run it live through one correction)",
                     value=True, key="tqs_half")
    scale = 0.5 if half else 1.0
    eff = pct * scale
    tgt_usd = acct * eff
    shares = int(tgt_usd / px) if px else 0
    cash_left = acct - (shares * px if px else 0)

    ndx = sig["ndx_close"]
    above = ndx / lv.get("ndx_250dma", ndx) - 1
    adx = sig.get("adx", 0) or 0

    # ---- thesis bullets ----
    bull = []
    if sig["regime"] == "BULL":
        bull.append(f"NDX **+{above:.0%} above its 250-DMA** ({lv.get('ndx_250dma',0):,.0f}) → trend backbone ON")
    elif sig["regime"] == "BEAR":
        bull.append(f"NDX **{above:.0%} below its 250-DMA** → downtrend, long backbone OFF")
    else:
        bull.append("NDX straddling its 250-DMA → neutral, reduced size")
    bull.append(f"ADX(14) = **{adx:.0f}** " + ("(strong trend, full size)" if adx >= 25
                else f"(choppy → backbone auto-trimmed to {net:+.2f})"))
    r3 = sig["rsi3"]
    bull.append(f"RSI(3) = **{r3:.0f}** " + ("→ oversold, dip-buy tranche active" if r3 < 30
                else "→ no dip-buy add (need < 30)"))
    bull.append(f"NDX **{sig.get('dd_from_high',0):+.0%}** from its 1-yr high · "
                f"5-day drift {sig.get('dist_sma5',0):+.1%}")
    concl = (f"→ **{'BUY / HOLD' if side!='CASH' else 'GO TO'} {side}"
             + (f" at {eff:.0%} of allocation**" if side != "CASH" else "**"))
    bull.append(concl)
    st.markdown("\n".join(f"* {b}" for b in bull))

    # ---- the ticket ----
    if side == "CASH":
        st.error("**Position: FLAT / 100% cash.** Sell any TQQQ or SQQQ today at the close. "
                 "No entry — wait for NDX to reclaim its 250-DMA + 3%.")
    else:
        stop_px = lv.get("tqqq_at_exit")
        stop_pct = lv.get("tqqq_stop_pct", 0)
        risk_usd = shares * (px - stop_px) if (px and stop_px) else 0
        wr = m.get("episode_win_rate")
        D = lambda v: f"\\${v:,.0f}"  # escape $ so Streamlit doesn't read it as LaTeX
        st.markdown(
            f"**{side} position — {sig['regime']} trend follow · signal {sig['as_of']}**\n\n"
            f"* **BUY / HOLD {side}** · ≈ \\${px} → **{shares:,} shares** · {D(tgt_usd)} "
            f"({eff:.0%} of {D(acct)}); keep {D(cash_left)} cash\n"
            f"* **Exit stop:** NDX closes < **{lv.get('exit_longs_below_ndx',0):,.0f}** "
            f"(250-DMA −3%) 3 days running → sell all. {side} ≈ **{D(stop_px)}** there ({stop_pct:+.0%})\n"
            f"* **Add trigger:** RSI(3) < 30 while NDX > {lv.get('ndx_250dma',0):,.0f} → add one more tranche (hold 2–10 days)\n"
            f"* **Crash brake:** any day NDX ≤ −5.5% → cut to cash for 6 days\n\n"
            f"Risk to systematic stop: **≈ {D(risk_usd)}** ({stop_pct:+.0%} on the position)  \n"
            f"Expected hold: **weeks–months** (median {m.get('episode_median_days','~250')} trading days per trend leg)  \n"
            f"Backtested edge: **{(wr or 0):.0%} of directional legs profitable** · "
            f"avg win **{(m.get('episode_avg_win') or 0):+.0%}** vs avg loss "
            f"**{(m.get('episode_avg_loss') or 0):+.0%}** · {(m.get('monthly_win_rate') or 0):.0%} of months green · "
            f"worst month {(m.get('worst_month') or 0):+.0%}"
        )
        if adx < 25:
            st.caption(f"⚠️ ADX {adx:.0f}: this is a chop tape — position auto-sized down from 100% to {net:.0%}.")

    # ---- rules table ----
    with st.expander("Full rule set — every buy / sell / hold trigger"):
        rules = {
            "Trigger (^NDX daily close)": [
                f"NDX > {lv.get('reenter_full_above_ndx',0):,.0f} (250-DMA +3%), 3 days",
                f"RSI(3) < 30 while NDX > ~{lv.get('ndx_250dma',0):,.0f}",
                f"NDX < {lv.get('exit_longs_below_ndx',0):,.0f} (250-DMA −3%), 3 days",
                "Any day NDX ≤ −5.5% (≈ −16% TQQQ)",
                f"NDX < {lv.get('exit_longs_below_ndx',0):,.0f} for 25+ days + 250-DMA falling",
            ],
            "Action": [
                "BUY / hold full TQQQ", "ADD a TQQQ dip tranche", "SELL all TQQQ → cash",
                "Cut to cash 6 days (then dips re-enter)", "Add SQQQ hedge up to 60%",
            ],
            "Hold": ["weeks–months", "2–10 days", "—", "6 days", "weeks–months"],
        }
        st.table(pd.DataFrame(rules))
        st.caption("Signals on ^NDX close; orders in TQQQ/SQQQ at/near the US close. "
                   "Position strategy, not intraday. Detail: `strategies/tqqq_sqqq_ensemble/USAGE.md`.")

    with st.expander("The 7 systems — today's contribution"):
        DESC = {
            "mom_long":    "1 · Momentum long (TQQQ) — trend backbone, hold above the 250-DMA",
            "mr_long_1":   "2 · Mean-reversion long (TQQQ) — shallow dip, RSI(3)<30",
            "mr_long_2":   "3 · Mean-reversion long (TQQQ) — deeper dip, RSI(3)<20",
            "mr_long_3":   "4 · Mean-reversion long (TQQQ) — capitulation, RSI(3)<14",
            "mr_short_1":  "5 · Mean-reversion short (SQQQ) — fade an overbought bounce in a downtrend",
            "mom_short_1": "6 · Momentum short (SQQQ) — confirmed downtrend continuation",
            "mom_short_2": "7 · Momentum short (SQQQ) — deeper breakdown (100-day low)",
        }
        ps = sig.get("per_system", {})
        srows = [{"System": DESC[k], "Today's weight": f"{ps.get(k, 0):+.2f}"} for k in DESC]
        st.table(pd.DataFrame(srows))
        st.caption(
            f"Sum → net {net:+.2f} → {'TQQQ' if net>0 else 'SQQQ' if net<0 else 'cash'}. "
            "Momentum weights are then ×ADX-quality ×extension-taper ×drawdown-ramp "
            "×Bollinger-momentum-boost (the video's '20 & 250 MA + Bollinger for sizing'). "
            "Full spec: `strategies/tqqq_sqqq_ensemble/SYSTEMS.md`."
        )

    # ═══════════════ backtest (collapsed) ═══════════════
    with st.expander("Backtest performance"):
        rows = []
        for kind in ("real", "synth"):
            d = data.get(kind)
            if not d:
                continue
            m, bh = d["metrics"], d["benchmark_bh_tqqq"]
            rows.append({
                "Window": f"{kind} {m['start'][:4]}–{m['end'][:4]}",
                "CAGR": f"{m['cagr']:.0%}", "MaxDD": f"{m['max_drawdown']:.0%}",
                "Sharpe": m["sharpe"], "Calmar": m["calmar"],
                "Trades/wk": m.get("rebalances_per_week"),
                "B&H TQQQ (CAGR/DD)": f"{bh['cagr']:.0%} / {bh['max_drawdown']:.0%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        for kind in ("real", "synth"):
            png = STRAT_DIR / "plots" / f"equity_{kind}.png"
            if png.exists():
                st.image(str(png), caption=f"Equity — {kind}")
        st.caption("Independent reconstruction of Collective2 #141158577. Params hand-tuned on "
                   "this history; walk-forward pending. Not investment advice.")
