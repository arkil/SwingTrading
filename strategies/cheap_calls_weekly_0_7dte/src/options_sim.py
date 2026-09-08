"""
Black-Scholes option pricing for cheap_calls_weekly_0_7dte -- thin re-export
of the shared strategies/_shared/options_sim.py.
"""

from __future__ import annotations
import sys
from pathlib import Path

_SHARED_PARENT = Path(__file__).resolve().parents[2]
if str(_SHARED_PARENT) not in sys.path:
    sys.path.insert(0, str(_SHARED_PARENT))

from _shared.options_sim import (  # noqa: E402,F401
    MIN_ENTRY_PREMIUM,
    bs_call_price,
    bs_put_price,
    bs_delta,
    bs_put_delta,
    simulate_option_pnl,
    simulate_put_pnl,
)
