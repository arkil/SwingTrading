"""
Trade simulator and 2-D sweep runner for cheap_calls_weekly_0_7dte.

For each (go_threshold, dte_target) cell:
  1. Generate signals
  2. Simulate each option trade via Black-Scholes repricing
  3. Compute win rate, avg return, expectancy

Usage:
    python backtest.py                    # run full sweep + save CSV
    python backtest.py --go 3 --dte 7     # single cell
    python backtest.py --baseline         # unfiltered (go=0, dte=7) baseline
    python backtest.py --iterate          # win-rate iteration sweep at GO≥2 DTE=5
"""

from __future__ import annotations

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from itertools import product
from datetime import datetime

from strategy       import Strategy
from src.options_sim import simulate_option_pnl


STRATEGY_DIR = Path(__file__).parent
EXP_DIR      = STRATEGY_DIR / "experiments"
TRADES_DIR   = STRATEGY_DIR / "trades"

# ── Sweep grid ────────────────────────────────────────────────────────────────
GO_THRESHOLDS = [0, 1, 2, 3, 4, 5]
DTE_TARGETS   = [0, 1, 2, 3, 5, 7]


# ── Trade simulation ──────────────────────────────────────────────────────────

def simulate_trades(
    signals: pd.DataFrame,
    arrays: dict,
    params: dict,
    stop_pct: float = 0.50,
    target_pct: float = 2.00,
) -> pd.DataFrame:
    """
    Simulate every signal in the signals DataFrame.
    For each signal we look up the forward price series for that symbol
    and run simulate_option_pnl to get entry/exit premium and return %.
    """
    if signals.empty:
        return pd.DataFrame()

    close    = arrays["close"]
    dates    = arrays["dates"]
    max_hold = max(DTE_TARGETS) + 3   # a few extra days of buffer

    results = []
    for _, row in signals.iterrows():
        i    = int(row["date_idx"])
        j    = int(row["sym_idx"])
        dte  = int(row["dte"])

        # Forward spot series: from entry date through dte + buffer
        end_idx = min(i + dte + 3, len(dates))
        spot_series = close[i:end_idx, j].copy()

        if len(spot_series) < 2 or np.any(np.isnan(spot_series[:2])):
            continue

        result = simulate_option_pnl(
            spot_series  = spot_series,
            strike       = float(row["strike"]),
            entry_dte    = dte,
            iv           = float(row["iv"]),
            max_premium  = params.get("max_premium", 1.50),
            stop_pct     = stop_pct,
            target_pct   = target_pct,
            r            = 0.05,
        )

        if result["exit_reason"] == "filtered":
            continue

        results.append({
            "date":          row["date"],
            "symbol":        row["symbol"],
            "spot":          row["spot"],
            "strike":        row["strike"],
            "dte":           dte,
            "iv":            row["iv"],
            "entry_premium": result["entry_premium"],
            "exit_premium":  result["exit_premium"],
            "return_pct":    result["return_pct"],
            "exit_reason":   result["exit_reason"],
            "days_held":     result["days_held"],
            "go_score":      row["go_score"],
            "vol_ratio":     row["vol_ratio"],
            "rsi14":         row["rsi14"],
            "rel_str":       row["rel_str"],
            "winner":        result["return_pct"] > 0 if not np.isnan(result["return_pct"]) else False,
        })

    return pd.DataFrame(results)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0, "win_rate": np.nan, "avg_return": np.nan,
                "avg_winner": np.nan, "avg_loser": np.nan, "expectancy": np.nan,
                "profit_factor": np.nan}

    r = trades["return_pct"].dropna()
    winners = r[r > 0]
    losers  = r[r <= 0]
    win_rate = len(winners) / len(r) * 100 if len(r) > 0 else np.nan
    avg_w    = winners.mean() if len(winners) > 0 else np.nan
    avg_l    = losers.mean()  if len(losers) > 0  else np.nan
    expect   = r.mean() if len(r) > 0 else np.nan
    pf       = (winners.sum() / abs(losers.sum())) if losers.sum() != 0 else np.inf

    return {
        "n_trades":     len(r),
        "win_rate":     round(win_rate, 1),
        "avg_return":   round(r.mean(), 1),
        "avg_winner":   round(avg_w, 1) if not np.isnan(avg_w) else np.nan,
        "avg_loser":    round(avg_l, 1) if not np.isnan(avg_l) else np.nan,
        "expectancy":   round(expect, 1),
        "profit_factor": round(pf, 2) if pf != np.inf else np.inf,
    }


# ── Single run ────────────────────────────────────────────────────────────────

def run_single(strategy: Strategy, go_threshold: float, dte_target: int,
               params: dict, save_trades: bool = False,
               stop_pct: float = 0.50, target_pct: float = 2.00,
               **filter_kwargs) -> dict:
    signals = strategy.get_signals(
        go_threshold=go_threshold,
        dte_target=dte_target,
        **filter_kwargs,
    )
    trades  = simulate_trades(signals, strategy._arrays, params, stop_pct=stop_pct, target_pct=target_pct)
    metrics = compute_metrics(trades)
    metrics["go_threshold"] = go_threshold
    metrics["dte_target"]   = dte_target

    if save_trades and not trades.empty:
        TRADES_DIR.mkdir(exist_ok=True)
        fname = TRADES_DIR / f"trades_go{go_threshold}_dte{dte_target}.csv"
        trades.to_csv(fname, index=False)
        print(f"  Trades saved → {fname}")

    return metrics


# ── Iteration sweep ───────────────────────────────────────────────────────────

# Each entry: (label, filter_kwargs)
_ITER_COMBOS: list[tuple[str, dict]] = [
    ("Baseline (no filters)",                  {}),
    ("1. Today up (chg>0)",                    dict(require_chg_positive=True)),
    ("2. Above EMA20",                         dict(require_above_ema20=True)),
    ("3. SPY bull",                            dict(require_spy_bull=True)),
    ("4. 5-day mom ≥0%",                      dict(min_mom5d=0.0)),
    ("5. RSI 40-70",                           dict(min_rsi=40.0, max_rsi=70.0)),
    ("6. Chg + EMA20 + SPY bull",             dict(require_chg_positive=True,
                                                    require_above_ema20=True,
                                                    require_spy_bull=True)),
    ("7. All filters + mom",                   dict(require_chg_positive=True,
                                                    require_above_ema20=True,
                                                    require_spy_bull=True,
                                                    min_mom5d=0.0)),
    ("8. All filters + RSI 40-70",             dict(require_chg_positive=True,
                                                    require_above_ema20=True,
                                                    require_spy_bull=True,
                                                    min_mom5d=0.0,
                                                    min_rsi=40.0, max_rsi=70.0)),
    ("9. ATM (otm=1%)",                        dict(otm_pct=1.0)),
    ("10. ATM + all filters",                  dict(otm_pct=1.0,
                                                    require_chg_positive=True,
                                                    require_above_ema20=True,
                                                    require_spy_bull=True,
                                                    min_mom5d=0.0,
                                                    min_rsi=40.0, max_rsi=70.0)),
]

# Batch 2: sweep OTM ladder, delta floor, and DTE — anchored at GO≥2
_ITER_COMBOS_2: list[tuple[str, int, dict]] = [
    # OTM % ladder (DTE=5)
    ("OTM 0%  (pure ATM)",        5,  dict(otm_pct=0.0)),
    ("OTM 1%",                    5,  dict(otm_pct=1.0)),
    ("OTM 2%",                    5,  dict(otm_pct=2.0)),
    ("OTM 3%",                    5,  dict(otm_pct=3.0)),
    ("OTM 5%  (baseline)",        5,  dict(otm_pct=5.0)),
    # Delta floor sweep at ATM (OTM=1%, DTE=5)
    ("ATM + delta ≥0.10",        5,  dict(otm_pct=1.0, delta_min=0.10)),
    ("ATM + delta ≥0.20",        5,  dict(otm_pct=1.0, delta_min=0.20)),
    ("ATM + delta ≥0.25",        5,  dict(otm_pct=1.0, delta_min=0.25)),
    ("ATM + delta ≥0.30",        5,  dict(otm_pct=1.0, delta_min=0.30)),
    ("ATM + delta ≥0.35",        5,  dict(otm_pct=1.0, delta_min=0.35)),
    # Premium cap sweep at ATM (OTM=1%, DTE=5)
    ("ATM + prem ≤$0.50",        5,  dict(otm_pct=1.0, max_premium=0.50)),
    ("ATM + prem ≤$0.75",        5,  dict(otm_pct=1.0, max_premium=0.75)),
    ("ATM + prem ≤$1.00",        5,  dict(otm_pct=1.0, max_premium=1.00)),
    # DTE sweep at ATM (OTM=1%)
    ("ATM + DTE=1",               1,  dict(otm_pct=1.0)),
    ("ATM + DTE=2",               2,  dict(otm_pct=1.0)),
    ("ATM + DTE=3",               3,  dict(otm_pct=1.0)),
    ("ATM + DTE=5",               5,  dict(otm_pct=1.0)),
    ("ATM + DTE=7",               7,  dict(otm_pct=1.0)),
    # Best combo: ATM + high delta + EMA20 filter at DTE=5 and DTE=7
    ("ATM + d≥0.25 + EMA20",     5,  dict(otm_pct=1.0, delta_min=0.25, require_above_ema20=True)),
    ("ATM + d≥0.25 + EMA20 DTE7",7,  dict(otm_pct=1.0, delta_min=0.25, require_above_ema20=True)),
    ("ATM + d≥0.25 + SPY bull",  5,  dict(otm_pct=1.0, delta_min=0.25, require_spy_bull=True)),
    ("ATM + d≥0.30 + EMA20 + SPY", 5, dict(otm_pct=1.0, delta_min=0.30,
                                             require_above_ema20=True, require_spy_bull=True)),
]


# Batch 3: stop-loss / target / vol-surge / rel-strength sweep
# Anchored at best known config: GO≥2, DTE=7, OTM=1%, RSI 55-70
_BEST_BASE = dict(otm_pct=1.0, min_rsi=55.0, max_rsi=70.0)

# (label, stop_pct, target_pct, extra_filters)
_ITER_COMBOS_3: list[tuple[str, float, float, dict]] = [
    # ── Stop-loss sweep (target fixed 2x) ───────────────────────────────────
    ("Stop 35% / Target 2x  (tight stop)",   0.35, 2.00, {}),
    ("Stop 40% / Target 2x",                 0.40, 2.00, {}),
    ("Stop 50% / Target 2x  (baseline)",     0.50, 2.00, {}),
    ("Stop 60% / Target 2x  (loose stop)",   0.60, 2.00, {}),
    ("Stop 75% / Target 2x  (very loose)",   0.75, 2.00, {}),
    # ── Target sweep (stop fixed 50%) ────────────────────────────────────────
    ("Stop 50% / Target 1.5x",               0.50, 1.50, {}),
    ("Stop 50% / Target 2x   (baseline)",    0.50, 2.00, {}),
    ("Stop 50% / Target 3x",                 0.50, 3.00, {}),
    ("Stop 50% / Target 5x",                 0.50, 5.00, {}),
    # ── Combo: tight stop + large target ─────────────────────────────────────
    ("Stop 40% / Target 3x",                 0.40, 3.00, {}),
    ("Stop 40% / Target 5x",                 0.40, 5.00, {}),
    # ── Volume surge filter (ATM, DTE=7, RSI 55-70) ──────────────────────────
    ("Vol surge ≥1.0× (no filter)",          0.50, 2.00, dict(min_vol_ratio=0.0)),
    ("Vol surge ≥1.5×",                      0.50, 2.00, dict(min_vol_ratio=1.5)),
    ("Vol surge ≥2.0×",                      0.50, 2.00, dict(min_vol_ratio=2.0)),
    ("Vol surge ≥2.5×",                      0.50, 2.00, dict(min_vol_ratio=2.5)),
    # ── Relative strength filter ─────────────────────────────────────────────
    ("Rel str ≥0%  (outperforming SPY)",     0.50, 2.00, dict(min_rel_str=0.0)),
    ("Rel str ≥0.5%",                        0.50, 2.00, dict(min_rel_str=0.5)),
    ("Rel str ≥1.0%",                        0.50, 2.00, dict(min_rel_str=1.0)),
    # ── Best known + stop/target tuning ─────────────────────────────────────
    ("Best (RSI 55-70) + Stop 40%/Tgt 3x",  0.40, 3.00, {}),
    ("Best (RSI 55-70) + Stop 40%/Tgt 5x",  0.40, 5.00, {}),
    ("Best + VolSurge 1.5x + Stop40/Tgt3x", 0.40, 3.00, dict(min_vol_ratio=1.5)),
    ("Best + RelStr 0.5% + Stop40/Tgt3x",   0.40, 3.00, dict(min_rel_str=0.5)),
]


def run_iterations_3(
    strategy: Strategy,
    params: dict,
    go_threshold: float = 2.0,
    dte_target: int = 7,
) -> pd.DataFrame:
    """Sweep stop/target levels, vol-surge, and rel-strength filters at the best known config."""
    EXP_DIR.mkdir(exist_ok=True)
    rows = []
    base_label = "Stop 50% / Target 2x  (baseline)"
    print(f"\nIteration-3 sweep at GO≥{go_threshold}, DTE={dte_target}  [stop / target / vol / relstr]")
    print("=" * 82)

    groups = [
        ("── Stop-loss sweep (target 2x) ──",    [c for c in _ITER_COMBOS_3 if "Target 2x" in c[0] and "Tgt" not in c[0] and "surge" not in c[0].lower() and "rel" not in c[0].lower() and "Best" not in c[0]]),
        ("── Target sweep (stop 50%) ──",         [c for c in _ITER_COMBOS_3 if c[0].startswith("Stop 50%")]),
        ("── Tight stop + large target ──",       [c for c in _ITER_COMBOS_3 if c[0].startswith("Stop 40%") and "Tgt" not in c[0]]),
        ("── Volume surge filter ──",             [c for c in _ITER_COMBOS_3 if "surge" in c[0].lower()]),
        ("── Relative strength filter ──",        [c for c in _ITER_COMBOS_3 if "rel str" in c[0].lower()]),
        ("── Best combos ──",                     [c for c in _ITER_COMBOS_3 if c[0].startswith("Best")]),
    ]

    baseline_wr = None
    for group_label, combos in groups:
        print(f"\n{group_label}")
        for label, stop_p, tgt_p, extra in combos:
            filters = {**_BEST_BASE, **extra}
            signals = strategy.get_signals(go_threshold=go_threshold, dte_target=dte_target, **filters)
            trades  = simulate_trades(signals, strategy._arrays, params, stop_pct=stop_p, target_pct=tgt_p)
            m       = compute_metrics(trades)
            wr  = m["win_rate"]
            n   = m["n_trades"]
            exp = m["expectancy"]
            aw  = m["avg_winner"]
            al  = m["avg_loser"]
            pf  = m["profit_factor"]

            if label == base_label or baseline_wr is None:
                baseline_wr = wr

            delta_wr = round(wr - baseline_wr, 1) if (not np.isnan(wr) and baseline_wr is not None and not np.isnan(baseline_wr)) else np.nan
            rows.append({"label": label, "stop_pct": stop_p, "target_pct": tgt_p,
                         "n_trades": n, "win_rate": wr, "delta_wr": delta_wr,
                         "avg_winner": aw, "avg_loser": al,
                         "expectancy": exp, "profit_factor": pf, **extra})

            wr_s  = f"{wr:.1f}%" if not np.isnan(wr) else "—"
            exp_s = f"{exp:.0f}%" if not np.isnan(exp) else "—"
            pf_s  = f"{pf:.2f}" if not np.isnan(pf) and pf != np.inf else ("∞" if pf == np.inf else "—")
            dw_s  = (f"+{delta_wr:.1f}%" if delta_wr > 0 else f"{delta_wr:.1f}%") if not np.isnan(delta_wr) else ""
            print(f"  {label:<45}  n={n:<6,}  WR={wr_s:<7}  E={exp_s:<8}  PF={pf_s}  {dw_s}")

    df = pd.DataFrame(rows)
    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = EXP_DIR / f"iterations3_{ts}.csv"
    df.to_csv(out, index=False)

    print(f"\n{'=' * 82}")
    print("Top 5 by EXPECTANCY (≥100 trades):")
    best_exp = df[df["n_trades"] >= 100].sort_values("expectancy", ascending=False).head(5)
    for _, r in best_exp.iterrows():
        print(f"  {r['label']:<45}  WR={r['win_rate']:.1f}%  E={r['expectancy']:.0f}%  n={int(r['n_trades'])}")
    print("\nTop 5 by WIN RATE (≥100 trades):")
    best_wr = df[df["n_trades"] >= 100].sort_values("win_rate", ascending=False).head(5)
    for _, r in best_wr.iterrows():
        print(f"  {r['label']:<45}  WR={r['win_rate']:.1f}%  E={r['expectancy']:.0f}%  n={int(r['n_trades'])}")
    print(f"\nResults saved → {out}")
    return df


def run_iterations(
    strategy: Strategy,
    params: dict,
    go_threshold: float = 2.0,
    dte_target: int = 5,
) -> pd.DataFrame:
    """
    Run all _ITER_COMBOS at the fixed (go_threshold, dte_target) anchor point
    and print a comparison table sorted by win rate.
    """
    EXP_DIR.mkdir(exist_ok=True)
    rows = []
    print(f"\nIteration sweep at GO≥{go_threshold}, DTE={dte_target}")
    print("=" * 68)

    for label, filters in _ITER_COMBOS:
        signals = strategy.get_signals(
            go_threshold=go_threshold,
            dte_target=dte_target,
            **filters,
        )
        trades  = simulate_trades(signals, strategy._arrays, params)
        m       = compute_metrics(trades)
        wr      = m["win_rate"]
        n       = m["n_trades"]
        exp     = m["expectancy"]
        aw      = m["avg_winner"]
        al      = m["avg_loser"]
        delta_wr = (wr - rows[0]["win_rate"]) if rows else 0.0

        rows.append({
            "label":     label,
            "n_trades":  n,
            "win_rate":  wr,
            "delta_wr":  round(delta_wr, 1) if rows else 0.0,
            "avg_winner": aw,
            "avg_loser":  al,
            "expectancy": exp,
        })

        wr_str  = f"{wr:.1f}%" if not np.isnan(wr) else "—"
        n_str   = f"{n:,}"
        exp_str = f"{exp:.0f}%" if not np.isnan(exp) else "—"
        dw_str  = f"+{delta_wr:.1f}%" if delta_wr > 0 else (f"{delta_wr:.1f}%" if delta_wr < 0 else "")
        if not rows or len(rows) == 1:
            dw_str = ""
        print(f"  {label:<45}  n={n_str:<6}  WR={wr_str:<7}  E={exp_str}  {dw_str}")

    df = pd.DataFrame(rows)
    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = EXP_DIR / f"iterations_{ts}.csv"
    df.to_csv(out, index=False)
    print(f"\n{'=' * 68}")
    print(f"Best by win rate:")
    best = df[df["n_trades"] >= 50].sort_values("win_rate", ascending=False).head(3)
    for _, r in best.iterrows():
        print(f"  {r['label']:<45}  WR={r['win_rate']:.1f}%  n={int(r['n_trades'])}")
    print(f"\nIteration results saved → {out}")
    return df


def run_iterations_2(
    strategy: Strategy,
    params: dict,
    go_threshold: float = 2.0,
) -> pd.DataFrame:
    """Sweep OTM%, delta floor, premium cap, and DTE at fixed GO threshold."""
    EXP_DIR.mkdir(exist_ok=True)
    rows = []
    print(f"\nIteration-2 sweep at GO≥{go_threshold}  (OTM / delta / premium / DTE)")
    print("=" * 78)

    groups = [
        ("── OTM % ladder ──", [c for c in _ITER_COMBOS_2 if c[0].startswith("OTM")]),
        ("── Delta floor (ATM, DTE=5) ──", [c for c in _ITER_COMBOS_2 if "delta" in c[0] and "EMA" not in c[0] and "SPY" not in c[0]]),
        ("── Premium cap (ATM, DTE=5) ──", [c for c in _ITER_COMBOS_2 if "prem" in c[0]]),
        ("── DTE sweep (ATM) ──", [c for c in _ITER_COMBOS_2 if "DTE=" in c[0]]),
        ("── Best combos ──", [c for c in _ITER_COMBOS_2 if "EMA20" in c[0] or "SPY bull" in c[0]]),
    ]

    for group_label, combos in groups:
        print(f"\n{group_label}")
        for label, dte, filters in combos:
            signals = strategy.get_signals(
                go_threshold=go_threshold,
                dte_target=dte,
                **filters,
            )
            trades = simulate_trades(signals, strategy._arrays, params)
            m = compute_metrics(trades)
            wr  = m["win_rate"]
            n   = m["n_trades"]
            exp = m["expectancy"]
            aw  = m["avg_winner"]
            al  = m["avg_loser"]

            rows.append({"label": label, "dte": dte, "n_trades": n,
                         "win_rate": wr, "expectancy": exp,
                         "avg_winner": aw, "avg_loser": al, **filters})

            wr_s  = f"{wr:.1f}%" if not np.isnan(wr)  else "—"
            exp_s = f"{exp:.0f}%" if not np.isnan(exp) else "—"
            print(f"  {label:<40}  DTE={dte}  n={n:<6,}  WR={wr_s:<7}  E={exp_s}")

    df = pd.DataFrame(rows)
    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = EXP_DIR / f"iterations2_{ts}.csv"
    df.to_csv(out, index=False)

    print(f"\n{'=' * 78}")
    print("Top 5 by win rate (≥50 trades):")
    best = df[df["n_trades"] >= 50].sort_values("win_rate", ascending=False).head(5)
    for _, r in best.iterrows():
        print(f"  {r['label']:<40}  DTE={int(r['dte'])}  WR={r['win_rate']:.1f}%  n={int(r['n_trades'])}")
    print(f"\nResults saved → {out}")
    return df


# ── 2-D sweep ─────────────────────────────────────────────────────────────────

def run_sweep(strategy: Strategy, params: dict) -> pd.DataFrame:
    EXP_DIR.mkdir(exist_ok=True)
    rows = []
    total = len(GO_THRESHOLDS) * len(DTE_TARGETS)
    done  = 0
    for go_t, dte_t in product(GO_THRESHOLDS, DTE_TARGETS):
        done += 1
        print(f"  [{done:02d}/{total}] GO≥{go_t}  DTE={dte_t}...", end="  ", flush=True)
        m = run_single(strategy, go_t, dte_t, params, save_trades=(go_t == 0))
        rows.append(m)
        print(f"n={m['n_trades']}  WR={m['win_rate']}%  E={m['expectancy']}%")

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = EXP_DIR / f"sweep_{ts}.csv"
    df.to_csv(out, index=False)
    print(f"\nSweep saved → {out}")
    return df


def print_sweep_table(df: pd.DataFrame) -> None:
    print("\n── Win Rate % ──────────────────────────────────────")
    piv = df.pivot(index="go_threshold", columns="dte_target", values="win_rate")
    print(piv.to_string())
    print("\n── Expectancy (avg return %) ───────────────────────")
    piv = df.pivot(index="go_threshold", columns="dte_target", values="expectancy")
    print(piv.to_string())
    print("\n── Trade Count ──────────────────────────────────────")
    piv = df.pivot(index="go_threshold", columns="dte_target", values="n_trades")
    print(piv.to_string())


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--go",       type=float, default=None)
    parser.add_argument("--dte",      type=int,   default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--iterate",  action="store_true", help="Run win-rate iteration combos at GO≥2 DTE=5")
    parser.add_argument("--iterate2", action="store_true", help="OTM/delta/premium/DTE sweep at GO≥2")
    parser.add_argument("--iterate3", action="store_true", help="Stop/target/vol/relstr sweep at GO≥2 DTE=7 (best config)")
    parser.add_argument("--refresh",  action="store_true", help="Force re-download data")
    parser.add_argument("--config",   default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Loading strategy…")
    strat = Strategy(config)
    strat.load(force_refresh=args.refresh)
    strat.compute()
    params = config.get("strategy_params", {})

    n_dates   = len(strat._arrays["dates"])
    n_symbols = len(strat._arrays["symbols"])
    print(f"Universe: {n_symbols} symbols × {n_dates} dates\n")

    if args.baseline:
        args.go, args.dte = 0, 7

    if args.iterate:
        run_iterations(strat, params, go_threshold=2.0, dte_target=5)
    elif args.iterate2:
        run_iterations_2(strat, params, go_threshold=2.0)
    elif args.iterate3:
        run_iterations_3(strat, params, go_threshold=2.0, dte_target=7)
    elif args.go is not None and args.dte is not None:
        m = run_single(strat, args.go, args.dte, params, save_trades=True)
        print(f"\nGO≥{args.go}  DTE={args.dte}")
        for k, v in m.items():
            print(f"  {k}: {v}")
    else:
        sweep_df = run_sweep(strat, params)
        print_sweep_table(sweep_df)


if __name__ == "__main__":
    main()
