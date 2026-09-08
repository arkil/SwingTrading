"""
Data loader for cheap_calls_weekly_0_7dte -- thin wrapper around the shared
strategies/_shared/data_loader.py, pinned to this strategy's own Data/ cache
dir and FOCUSED_UNIVERSE default.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_SHARED_DIR = Path(__file__).resolve().parents[2] / "_shared"
if str(_SHARED_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR.parent))

from _shared.data_loader import (  # noqa: E402
    FOCUSED_UNIVERSE,
    BENCHMARK,
    pivot_to_wide,
    load_universe as _load_universe,
    load_spy as _load_spy,
)

DATA_DIR = Path(__file__).parent.parent / "Data"


def load_universe(tickers: Optional[list[str]] = None, start: str = "2023-01-01",
                   end: Optional[str] = None, force_refresh: bool = False):
    return _load_universe(DATA_DIR, tickers=tickers or FOCUSED_UNIVERSE, start=start,
                           end=end, force_refresh=force_refresh)


def load_spy(start: str = "2023-01-01", end: Optional[str] = None, force_refresh: bool = False):
    return _load_spy(DATA_DIR, start=start, end=end, force_refresh=force_refresh)
