"""
Parameter sweep over guardrail combinations.
Tests each guardrail config against the full ticker universe and ranks by win rate × expectancy.
Uses Polars for fast aggregation of sweep results.
"""

import itertools
from dataclasses import asdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import polars as pl

from .signals import Guardrails


# ── Default sweep grid ────────────────────────────────────────────────────────

SWEEP_GRID = {
    "min_score":    [6, 7, 8],
    "max_ext_pct":  [15.0, 20.0, 30.0],
    "rsi_bull_max": [60.0, 65.0, 70.0],
    "min_adx":      [15.0, 20.0, 25.0],
    "min_vol_ratio":[1.0, 1.2, 1.5],
}

# Narrow grid for fast sweeps (2^3 = 8 combos)
NARROW_GRID = {
    "min_score":    [7, 8],
    "max_ext_pct":  [20.0, 30.0],
    "rsi_bull_max": [65.0, 70.0],
}


def _grid_combos(grid: Dict[str, list]) -> List[Dict[str, Any]]:
    keys   = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def run_sweep(
    all_trades_fn,          # callable(guardrails) → pd.DataFrame of trades
    grid: Dict = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run backtest for every parameter combo in grid.
    all_trades_fn is called with each Guardrails instance.

    Returns a Polars DataFrame ranked by expectancy.
    """
    if grid is None:
        grid = SWEEP_GRID

    combos = _grid_combos(grid)
    if verbose:
        print(f"Sweep: {len(combos)} combinations …")

    rows = []
    for i, params in enumerate(combos):
        g = Guardrails(**params)
        trades = all_trades_fn(g)

        if trades.empty or len(trades) < 5:
            n, wr, exp, pf = 0, 0.0, 0.0, 0.0
        else:
            pnl  = trades["PnL_R"]
            wins = trades["Win"]
            wr   = wins.mean()
            avg_w = pnl[wins].mean() if wins.any() else 0.0
            avg_l = pnl[~wins].mean() if (~wins).any() else -1.0
            exp  = wr * avg_w + (1 - wr) * avg_l
            pf   = (pnl[wins].sum() / abs(pnl[~wins].sum())
                    if (~wins).any() and pnl[~wins].sum() != 0 else np.inf)
            n    = len(trades)

        row = {**params, "n_trades": n, "win_rate": round(wr * 100, 1),
               "expectancy": round(exp, 3), "profit_factor": round(pf, 2)}
        rows.append(row)
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(combos)} done …")

    df = pl.DataFrame(rows)
    df = df.sort("expectancy", descending=True)
    return df


def best_params(sweep_result: pl.DataFrame) -> Guardrails:
    """Extract the top-ranked Guardrails from a sweep result."""
    top = sweep_result.head(1).to_dicts()[0]
    fields = {k: v for k, v in top.items()
              if k in Guardrails.__dataclass_fields__}
    return Guardrails(**fields)


def print_sweep(df: pl.DataFrame, top_n: int = 10) -> None:
    cols = [c for c in df.columns if c != "n_trades" or True]
    print("\nTop guardrail combinations (by expectancy):")
    print(df.head(top_n).to_pandas().to_string(index=False))
