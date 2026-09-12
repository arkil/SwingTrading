"""
Qullamaggie Breakout — parameter optimization with an out-of-sample check
==========================================================================

The one-at-a-time sensitivity sweep in backtest.py found the edge concentrates
in high-ADR names (adr_min>=5). This script does a small, deliberately
*coarse* grid search around that finding, combined with the other levers that
moved Sharpe/DD in the sweep — and reports every candidate BOTH in-sample
(train: 2016 - 2022) and out-of-sample (test: 2023 - now), from the same
equity/trade stream, so a config can't look good only because it curve-fit
the whole history.

Run:
    python optimize.py                  # stage 1: signal-quality grid
    python optimize.py --stage2         # stage 2: capital-deployment grid
                                         #   (needs a --base "k=v,k=v" from stage 1's winner)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt

SPLIT = pd.Timestamp("2023-01-01")


def slice_metrics(eq: pd.Series, tr: pd.DataFrame, spy: pd.DataFrame, start=None, end=None) -> dict:
    e = eq.loc[start:end] if (start or end) else eq
    if len(e) < 30:
        return dict(n_trades=0)
    t = tr
    if len(t):
        ed = pd.to_datetime(t.entry_date)
        if start is not None:
            t = t[ed >= start]
        if end is not None:
            t = t[ed <= end]
    return bt.metrics(e, t, spy, {})


def fmt(tag: str, m: dict) -> str:
    if m.get("n_trades", 0) < 5:
        return f"  {tag:<10} too few trades ({m.get('n_trades', 0)})"
    return (f"  {tag:<10} CAGR {m['cagr']*100:6.1f}%  DD {m['max_dd']*100:6.1f}%  "
            f"trades {m['n_trades']:4d}  win {m.get('win_rate', 0)*100:4.0f}%  "
            f"exp {m.get('expectancy_r', 0):+.2f}R  PF {m.get('profit_factor', 0):5.2f}  "
            f"Sharpe {m['sharpe']:.2f}")


def eval_one(data, spy, mom_pct, prm: dict, label: str):
    eq, tr = bt.run(data, spy, mom_pct, prm, verbose=False)
    train = slice_metrics(eq, tr, spy, end=SPLIT)
    test = slice_metrics(eq, tr, spy, start=SPLIT)
    full = slice_metrics(eq, tr, spy)
    print(f"\n{label}  {prm}")
    print(fmt("train", train))
    print(fmt("test ", test))
    print(fmt("full ", full))
    return dict(label=label, prm=prm, train=train, test=test, full=full)


def stage1(data, spy, mom_pct):
    print("═" * 78)
    print("STAGE 1 — signal quality grid (adr_min × cons_len × mom_pctile)")
    print("Fixed: partial_day=4, max_positions=10, risk=0.5%")
    print("═" * 78)
    results = []
    for adr_min in [3.5, 4.5, 5.0, 5.5, 6.5]:
        for cons_len in [12, 16, 20]:
            for mom_pctile in [0.90, 0.95]:
                prm = dict(bt.P)
                prm.update(adr_min=adr_min, cons_len=cons_len, mom_pctile=mom_pctile)
                label = f"adr{adr_min} len{cons_len} mom{mom_pctile}"
                results.append(eval_one(data, spy, mom_pct, prm, label))

    print("\n" + "═" * 78)
    print("STAGE 1 RANKED — by TRAIN Sharpe (min 40 train trades)")
    print("═" * 78)
    ranked = sorted(
        (r for r in results if r["train"].get("n_trades", 0) >= 40),
        key=lambda r: -r["train"]["sharpe"],
    )
    for r in ranked[:10]:
        tr_, te_ = r["train"], r["test"]
        print(f"  {r['label']:<28} train Sharpe {tr_['sharpe']:.2f} exp {tr_.get('expectancy_r',0):+.2f}R"
              f"  |  test Sharpe {te_.get('sharpe',0):.2f} exp {te_.get('expectancy_r',0):+.2f}R"
              f" n={te_.get('n_trades',0)}")
    return results


def stage2(data, spy, mom_pct, base: dict):
    print("═" * 78)
    print(f"STAGE 2 — capital deployment grid, base = {base}")
    print("═" * 78)
    results = []
    for risk in [0.005, 0.0075, 0.01]:
        for max_pos in [10, 15, 20]:
            prm = dict(bt.P); prm.update(base)
            prm.update(risk_per_trade=risk, max_positions=max_pos)
            label = f"risk{risk} maxpos{max_pos}"
            results.append(eval_one(data, spy, mom_pct, prm, label))
    for pday in [3, 5]:
        prm = dict(bt.P); prm.update(base)
        prm.update(partial_day=pday)
        label = f"partial_day{pday}"
        results.append(eval_one(data, spy, mom_pct, prm, label))

    print("\n" + "═" * 78)
    print("STAGE 2 RANKED — by TRAIN Sharpe (min 30 train trades)")
    print("═" * 78)
    ranked = sorted(
        (r for r in results if r["train"].get("n_trades", 0) >= 30),
        key=lambda r: -r["train"]["sharpe"],
    )
    for r in ranked[:10]:
        tr_, te_ = r["train"], r["test"]
        print(f"  {r['label']:<28} train Sharpe {tr_['sharpe']:.2f} exp {tr_.get('expectancy_r',0):+.2f}R"
              f"  |  test Sharpe {te_.get('sharpe',0):.2f} exp {te_.get('expectancy_r',0):+.2f}R"
              f" n={te_.get('n_trades',0)}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--base", type=str, default="adr_min=5.0,cons_len=16,mom_pctile=0.9",
                    help="k=v,k=v from stage 1's pick, used as the fixed base for stage 2")
    args = ap.parse_args()

    prices = bt.load_prices()
    data, spy, mom_pct = bt.build(prices)
    print(f"universe: {len(data)} symbols · train < {SPLIT.date()} · test >= {SPLIT.date()}\n")

    if args.stage2:
        base = {}
        for kv in args.base.split(","):
            k, v = kv.split("=")
            base[k] = float(v) if "." in v else int(v)
        stage2(data, spy, mom_pct, base)
    else:
        stage1(data, spy, mom_pct)


if __name__ == "__main__":
    main()
