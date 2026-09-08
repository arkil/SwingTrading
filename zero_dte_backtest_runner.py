"""
Dashboard adapter for zero_dte scanner backtest results.

Two independent sources, matching the 3 modes in dashboard.py's
render_0dte_scanner():

  - "directional": Buy Directional / Intraday Close (long calls/puts).
    Source: strategies/zero_dte_0_3dte/validated_config.json (walk-forward
    validated in this repo, see that folder's walkforward.py).

  - "sell_premium": credit spreads / iron condors. Source: the REAL
    Alpaca-minute-bar backtest in
    /Users/arkilthakkar/workplace/strategies/spy_0dte_iron_condor/ (a
    sibling directory outside this repo, same convention
    options_backtest_runner.py already uses for swing_options_45_60d).
    That strategy was already run to a KILLED conclusion -- no parameter
    combination showed positive edge, even at zero transaction cost. This
    adapter reads its structured output rather than re-running anything.

Usage (as a library, called from dashboard.py):
    from zero_dte_backtest_runner import get_directional_findings, get_sell_premium_findings
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

_REPO_DIR = Path(__file__).parent
_ZERO_DTE_DIR = _REPO_DIR / "strategies" / "zero_dte_0_3dte"
_IRON_CONDOR_DIR = _REPO_DIR.parent.parent / "strategies" / "spy_0dte_iron_condor"


def get_directional_findings() -> Optional[dict]:
    """Walk-forward validated_config.json for Buy Directional / Intraday Close."""
    f = _ZERO_DTE_DIR / "validated_config.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def get_sell_premium_findings() -> dict:
    """
    Structured summary of strategies/spy_0dte_iron_condor's KILLED finding.
    Reads its optimize_report2.json (round-2: delta-target x exit-style grid,
    in-sample 2024-06-01 to 2025-03-31, out-of-sample 2025-04-01 to
    2025-10-31) if available; falls back to a hardcoded summary of the same
    numbers from OPTIMIZATION_FINDINGS.md if the JSON is missing (e.g. this
    repo checked out without the sibling strategies/ directory).
    """
    report_path = _IRON_CONDOR_DIR / "experiments" / "optimize_report2.json"
    findings_doc = _IRON_CONDOR_DIR / "OPTIMIZATION_FINDINGS.md"

    fallback = {
        "status": "KILLED",
        "decision_date": "2026-07-01",
        "best_config": {"delta_target": 0.13, "exit_style": "hold_to_eod (no intraday stop)"},
        "in_sample": {"period": ["2024-06-01", "2025-03-31"], "win_rate_pct": 68.7,
                       "total_return_pct": -29.35, "sharpe_ratio": -1.28},
        "out_of_sample": {"period": ["2025-04-01", "2025-10-31"], "win_rate_pct": 68.1,
                            "total_return_pct": -6.81, "sharpe_ratio": -0.52},
        "overfitting_check": "PASS",
        "source": "hardcoded fallback (optimize_report2.json not found)",
        "findings_doc": str(findings_doc),
        "summary": (
            "No parameter combination tested (delta target x exit style, 23 combos "
            "across 2 optimization rounds, real Alpaca minute-bar SPY 0DTE option "
            "data) produced a positive edge -- holds even at zero transaction cost. "
            "0DTE gamma causes far more intraday stop-outs than the delta-implied "
            "expiry probability would suggest. Strategy abandoned 2026-07-01."
        ),
    }

    if not report_path.exists():
        return fallback

    try:
        report = json.loads(report_path.read_text())
        return {
            "status": "KILLED",
            "decision_date": "2026-07-01",
            "best_config": report.get("best_in_sample_config"),
            "in_sample": {
                "period": report.get("in_sample_period"),
                "win_rate_pct": report.get("in_sample_metrics", {}).get("win_rate_pct"),
                "total_return_pct": report.get("in_sample_metrics", {}).get("total_return_pct"),
                "sharpe_ratio": report.get("in_sample_metrics", {}).get("sharpe_ratio"),
            },
            "out_of_sample": {
                "period": report.get("out_of_sample_period"),
                "win_rate_pct": report.get("out_of_sample_metrics", {}).get("win_rate_pct"),
                "total_return_pct": report.get("out_of_sample_metrics", {}).get("total_return_pct"),
                "sharpe_ratio": report.get("out_of_sample_metrics", {}).get("sharpe_ratio"),
            },
            "overfitting_check": report.get("overfitting_check"),
            "source": str(report_path),
            "findings_doc": str(findings_doc),
            "summary": fallback["summary"],
        }
    except Exception:
        return fallback


if __name__ == "__main__":
    print("Directional (Buy/Intraday Close):")
    print(json.dumps(get_directional_findings(), indent=2, default=str)[:1000])
    print("\nSell Premium (credit spreads / iron condors):")
    print(json.dumps(get_sell_premium_findings(), indent=2, default=str))
