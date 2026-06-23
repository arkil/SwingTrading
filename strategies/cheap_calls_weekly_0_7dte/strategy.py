"""
Strategy orchestrator for cheap_calls_weekly_0_7dte.
Loads data, computes features, aligns SPY, and exposes the signals DataFrame.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import Optional
import numpy as np
import polars as pl

from src.data_loader import load_universe, load_spy, pivot_to_wide, FOCUSED_UNIVERSE
from src.features     import compute_features
from src.signals      import generate_signals


class Strategy:
    def __init__(self, config: dict):
        self.cfg     = config
        self.params  = config.get("strategy_params", {})
        self._arrays: Optional[dict]  = None
        self._feats:  Optional[dict]  = None
        self._spy_close: Optional[np.ndarray] = None

    # ── Data ──────────────────────────────────────────────────────────────────

    def load(self, force_refresh: bool = False) -> None:
        """Fetch + cache OHLCV universe and SPY."""
        start = self.cfg["time"].get("start_date") or "2023-01-01"
        end   = self.cfg["time"].get("end_date")

        univ_df = load_universe(
            tickers=FOCUSED_UNIVERSE, start=start, end=end,
            force_refresh=force_refresh,
        )
        spy_df  = load_spy(start=start, end=end, force_refresh=force_refresh)

        # Align SPY to universe dates — join avoids Polars is_in() datetime mismatch
        spy_df = (
            spy_df
            .join(univ_df.select("date").unique(), on="date", how="inner")
            .sort("date")
        )

        self._arrays = pivot_to_wide(univ_df)

        # SPY close aligned to universe dates (fill forward any gaps)
        all_dates = self._arrays["dates"]
        spy_date_to_close = dict(zip(spy_df["date"].to_list(), spy_df["close"].to_list()))
        spy_close = np.array([spy_date_to_close.get(d, np.nan) for d in all_dates])
        # Forward-fill NaN
        last = np.nan
        for i in range(len(spy_close)):
            if not np.isnan(spy_close[i]):
                last = spy_close[i]
            elif not np.isnan(last):
                spy_close[i] = last
        self._spy_close = spy_close

    # ── Features ─────────────────────────────────────────────────────────────

    def compute(self) -> None:
        """Build feature arrays from loaded data."""
        if self._arrays is None:
            raise RuntimeError("Call load() first")
        self._feats = compute_features(
            self._arrays,
            self._spy_close,
            iv_scaling=self.params.get("iv_assumption", 0.45) /
                       # Use params iv_assumption as an absolute floor / fallback;
                       # compute_features derives actual IV from HV. When HV is
                       # unavailable we want to fall back to the config value.
                       # We pass iv_scaling=1.2 and handle the absolute floor inside.
                       1.0,
        )

    # ── Signals ───────────────────────────────────────────────────────────────

    def get_signals(
        self,
        go_threshold:         float = 0.0,
        dte_target:           int   = 7,
        otm_pct:              float = 5.0,
        max_premium:          float | None = None,
        delta_min:            float | None = None,
        delta_max:            float | None = None,
        require_chg_positive: bool  = False,
        require_above_ema20:  bool  = False,
        require_spy_bull:     bool  = False,
        min_mom5d:            float = -999.0,
        min_rsi:              float = 0.0,
        max_rsi:              float = 100.0,
        min_vol_ratio:        float = 0.0,
        min_rel_str:          float = -999.0,
    ):
        """Return signal DataFrame for given threshold, DTE, and hard filters."""
        if self._feats is None:
            raise RuntimeError("Call compute() first")
        p = self.params
        return generate_signals(
            arrays                = self._arrays,
            features              = self._feats,
            go_threshold          = go_threshold,
            max_premium           = max_premium  if max_premium is not None else p.get("max_premium", 1.50),
            delta_min             = delta_min    if delta_min   is not None else p.get("delta_min",   0.05),
            delta_max             = delta_max    if delta_max   is not None else p.get("delta_max",   0.45),
            otm_pct               = otm_pct,
            dte_target            = dte_target,
            require_chg_positive  = require_chg_positive,
            require_above_ema20   = require_above_ema20,
            require_spy_bull      = require_spy_bull,
            min_mom5d             = min_mom5d,
            min_rsi               = min_rsi,
            max_rsi               = max_rsi,
            min_vol_ratio         = min_vol_ratio,
            min_rel_str           = min_rel_str,
        )
