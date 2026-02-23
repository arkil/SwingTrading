from __future__ import annotations

from pathlib import Path

import pandas as pd

from daytrade_backtester.engine.backtester import TradeResult


def _pick_example_trades(trades: list[TradeResult], count: int) -> list[TradeResult]:
    if not trades:
        return []

    winners = [t for t in trades if t.hit_target]
    losers = [t for t in trades if not t.hit_target]

    picked: list[TradeResult] = []
    if winners:
        picked.append(winners[0])
    if losers and (not picked or losers[0].entry_time != picked[0].entry_time):
        picked.append(losers[0])

    i = 0
    while len(picked) < count and i < len(trades):
        t = trades[i]
        if all(t.entry_time != p.entry_time for p in picked):
            picked.append(t)
        i += 1

    return picked[:count]


def save_trade_charts(prepared_df: pd.DataFrame, trades: list[TradeResult], output_dir: str = "artifacts/trade_charts", count: int = 2) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = _pick_example_trades(trades, count)
    saved: list[str] = []

    for idx, t in enumerate(examples, start=1):
        if t.entry_time not in prepared_df.index or t.exit_time not in prepared_df.index:
            continue

        entry_pos = prepared_df.index.get_loc(t.entry_time)
        exit_pos = prepared_df.index.get_loc(t.exit_time)
        if isinstance(entry_pos, slice) or isinstance(exit_pos, slice):
            continue

        start = max(0, int(entry_pos) - 20)
        end = min(len(prepared_df), int(exit_pos) + 21)
        w = prepared_df.iloc[start:end].copy()

        x = range(len(w))
        entry_x = int(entry_pos) - start
        exit_x = int(exit_pos) - start

        fig, (ax_price, ax_rsi) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        ax_price.plot(x, w["close"], label="Close", color="black", linewidth=1.5)
        ax_price.plot(x, w["bb_upper"], label="BB Upper", color="tab:blue", alpha=0.8)
        ax_price.plot(x, w["bb_basis"], label="BB Basis", color="tab:gray", alpha=0.8)
        ax_price.plot(x, w["bb_lower"], label="BB Lower", color="tab:blue", alpha=0.8)

        ax_price.scatter(entry_x, t.entry_price, marker="^", s=110, color="green", label="Entry")
        ax_price.scatter(exit_x, t.exit_price, marker="v", s=110, color="red", label="Exit")
        ax_price.set_title(
            f"Trade {idx}: {t.side.upper()} | {t.entry_time.strftime('%Y-%m-%d %H:%M')} -> {t.exit_time.strftime('%H:%M')} | "
            f"Exit: {t.exit_reason} | OptRet: {t.option_return_pct * 100:.2f}%"
        )
        ax_price.legend(loc="upper left", ncol=5)
        ax_price.grid(alpha=0.2)

        ax_rsi.plot(x, w["rsi"], label="RSI", color="tab:orange")
        ax_rsi.axhline(70, color="red", linestyle="--", linewidth=1)
        ax_rsi.axhline(30, color="green", linestyle="--", linewidth=1)
        ax_rsi.scatter(entry_x, float(w.iloc[entry_x]["rsi"]), marker="o", s=60, color="green")
        ax_rsi.scatter(exit_x, float(w.iloc[exit_x]["rsi"]), marker="o", s=60, color="red")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.grid(alpha=0.2)

        step = max(1, len(w) // 8)
        tick_idx = list(range(0, len(w), step))
        ax_rsi.set_xticks(tick_idx)
        ax_rsi.set_xticklabels([w.index[i].strftime("%m-%d %H:%M") for i in tick_idx], rotation=25, ha="right")

        plt.tight_layout()
        out_file = out_dir / f"trade_{idx}_{t.entry_time.strftime('%Y%m%d_%H%M')}_{t.side}.png"
        fig.savefig(out_file, dpi=140)
        plt.close(fig)
        saved.append(str(out_file))

    return saved
