"""
Feature engineering for cheap_calls_weekly_0_7dte -- thin re-export of the
shared strategies/_shared/features.py (call-side GO Score).
"""

from __future__ import annotations
import sys
from pathlib import Path

_SHARED_PARENT = Path(__file__).resolve().parents[2]
if str(_SHARED_PARENT) not in sys.path:
    sys.path.insert(0, str(_SHARED_PARENT))

from _shared.features import (  # noqa: E402,F401
    compute_features,
    compute_base_features,
    compute_call_go_score,
)
