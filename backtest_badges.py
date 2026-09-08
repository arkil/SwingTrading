"""
Small read-only helpers that turn each short-DTE options strategy's
validated_config.json (or, for 0DTE sell-premium, the spy_0dte_iron_condor
findings) into a caption/banner string for the live scanner UI.

Never re-runs a backtest -- just reads whatever the strategy folder's
walkforward.py already wrote, so this is cheap to call on every dashboard
render. Falls back to "not yet backtested" text if a file is missing so the
page never breaks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_REPO_DIR = Path(__file__).parent


def _load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _fold_summary(result: dict) -> str:
    """One-line per-fold breakdown, e.g. '+290%(n149) +703%(n55) +940%(n92)'."""
    parts = []
    for fold in result.get("folds", []):
        if fold.get("status") != "ok":
            parts.append("skip")
            continue
        tm = fold["test_metrics"]
        e, n = tm.get("expectancy"), tm.get("n_trades")
        if e is None or (isinstance(e, float) and e != e):  # NaN
            parts.append(f"n={n}(no trades)")
        else:
            parts.append(f"{'+' if e >= 0 else ''}{e:.0f}%(n={n})")
    return "  ".join(parts)


def _badge_text(label: str, result: dict) -> str:
    n = result.get("avg_test_n_trades", 0)
    exp = result.get("avg_test_expectancy")
    folds = _fold_summary(result)
    exp_s = f"{exp:+.0f}%" if exp is not None else "n/a"
    if result.get("consistently_profitable"):
        verdict = "✅ Consistently profitable across all 3 walk-forward test folds"
    else:
        verdict = ("⚠️ Positive expectancy in every test fold so far, but sample sizes "
                    "are thin in later folds (< 100 trades) -- directional signal, "
                    "not a fully validated edge yet")
    return (f"📊 **{label} — walk-forward backtested** (3 expanding-window folds, "
            f"Black-Scholes-simulated pricing, no real option chain history). "
            f"Avg out-of-sample expectancy/trade: **{exp_s}**, avg {n} trades/fold. "
            f"Per-fold test results: {folds}. {verdict}")


def cheap_calls_badge() -> str:
    # Try both locations (relative to this file, or from dashboard)
    paths_to_try = [
        _REPO_DIR / "strategies" / "cheap_calls_weekly_0_7dte" / "validated_config.json",
        Path("/Users/arkilthakkar/workplace/Scripts/SwingTrading/strategies/cheap_calls_weekly_0_7dte/validated_config.json"),
    ]
    data = None
    for path in paths_to_try:
        data = _load(path)
        if data:
            break

    if not data or "cheap_calls" not in data:
        return "📊 Not yet backtested — treat signals as unvalidated."

    return _badge_text("Cheap Calls", data["cheap_calls"])



def weekly_opts_badge() -> str:
    data = _load(_REPO_DIR / "strategies" / "cheap_calls_weekly_0_7dte" / "validated_config.json")
    if not data or "weekly_opts" not in data:
        return "📊 Not yet backtested — treat signals as unvalidated."
    return _badge_text("Weekly Options", data["weekly_opts"])


def weekly_puts_badge() -> str:
    data = _load(_REPO_DIR / "strategies" / "weekly_puts_0_3dte" / "validated_config.json")
    if not data or "weekly_puts" not in data:
        return "📊 Not yet backtested — treat signals as unvalidated."
    return _badge_text("Weekly Puts", data["weekly_puts"])


def zero_dte_directional_badge(side: str = "call") -> str:
    data = _load(_REPO_DIR / "strategies" / "zero_dte_0_3dte" / "validated_config.json")
    if not data or side not in data:
        return "📊 Not yet backtested — treat signals as unvalidated."
    label = "0DTE Buy Calls (1-3 DTE Intraday Close)" if side == "call" else "0DTE Buy Puts (1-3 DTE Intraday Close)"
    return _badge_text(label, data[side])


def swing_options_badge() -> str:
    """Swing options 45-60 DTE strategy badge from latest backtest run (exp_005)."""
    metrics = _load(_REPO_DIR.parent / "strategies" / "swing_options_45_60d" / "experiments" / "exp_005" / "metrics.json")
    if not metrics:
        return "📊 Not yet backtested — treat signals as unvalidated."

    ret = metrics.get("total_return_pct", 0)
    wr = metrics.get("win_rate_pct", 0)
    sharpe = metrics.get("sharpe", 0)
    n_trades = metrics.get("total_trades", 0)
    max_dd = metrics.get("max_drawdown_pct", 0)

    return (f"📊 **Swing Options 45-60 DTE — Validated backtest** (2022–present, $25K capital, "
            f"Black-Scholes Greeks validation, 3 max concurrent positions). "
            f"**{ret:+.1f}% return** on ${metrics.get('initial_capital', 25000):,}, "
            f"**{wr:.1f}% win rate** across {n_trades} trades. "
            f"Sharpe **{sharpe:.2f}**, max drawdown **{max_dd:.1f}%**. "
            f"⚠️ Note: No walk-forward validation yet (unlike 0-7 DTE strategies) — sample represents "
            f"single-run (exp_005) performance. Consider this a directional signal with documented edge.")


def render_options_hub_validation() -> None:
    """Display all options hub mode validation badges in a collapsible expander (call from render_options_hub)."""
    import streamlit as st

    with st.expander("📊 Backtest Validation Across All Modes", expanded=False):
        st.markdown("### Validation Status by DTE Range")

        # Create columns for badge display
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Short-DTE (0-7 days)")
            st.caption(weekly_opts_badge())
            st.caption(cheap_calls_badge())
            st.caption(weekly_puts_badge())

        with col2:
            st.markdown("#### Directional Entry")
            st.caption(zero_dte_directional_badge("call"))
            st.caption(zero_dte_directional_badge("put"))

        st.divider()
        st.markdown("#### Swing Options (45-60 days)")
        st.caption(swing_options_badge())


def zero_dte_sell_premium_banner() -> str:
    """Prominent warning for the Sell Premium mode -- see zero_dte_backtest_runner.py."""
    try:
        from zero_dte_backtest_runner import get_sell_premium_findings
        f = get_sell_premium_findings()
    except Exception:
        return ("⚠️ Sell Premium mode has not been backtested in this dashboard session "
                 "— treat signals as unvalidated.")
    oos = f.get("out_of_sample", {})
    return (
        f"🚨 **Backtested and KILLED ({f.get('decision_date', 'n/a')}).** Real Alpaca "
        f"minute-bar SPY 0DTE data, {f.get('overfitting_check', 'n/a')} overfitting check. "
        f"No parameter combination (delta target x exit style, 23 combos across 2 rounds) "
        f"showed a positive edge, even at zero transaction cost. Best config found "
        f"(delta={f.get('best_config', {}).get('delta_target', '?')}, "
        f"{f.get('best_config', {}).get('exit_style', '?')}): out-of-sample "
        f"{oos.get('period', ['?', '?'])[0]}→{oos.get('period', ['?', '?'])[1]} "
        f"return **{oos.get('total_return_pct', '?')}%**, Sharpe **{oos.get('sharpe_ratio', '?')}**, "
        f"win rate {oos.get('win_rate_pct', '?')}% (win rate alone isn't enough — 0DTE gamma "
        f"causes losses larger than the win rate implies). "
        f"Full writeup: `{f.get('findings_doc', 'strategies/spy_0dte_iron_condor/OPTIMIZATION_FINDINGS.md')}`. "
        f"**Recommendation: do not trade this mode with real capital as currently configured.**"
    )
