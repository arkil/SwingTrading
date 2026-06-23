from __future__ import annotations

import argparse
import datetime
import os
import sys

from daytrade_backtester.config.loader import load_config
from daytrade_backtester.data.yahoo import load_intraday_bars
from daytrade_backtester.engine.backtester import count_signals, run_backtest
from daytrade_backtester.engine.options_enrich import enrich_trades_with_option_prices
from daytrade_backtester.reporting.charts import save_trade_charts
from daytrade_backtester.reporting.console import print_run_config, print_summary, print_trade_log
from daytrade_backtester.reporting.export import save_trade_log_csv
from daytrade_backtester.strategies.registry import get_strategy


class _Tee:
    """Write to multiple file-like objects simultaneously."""

    def __init__(self, *files):
        self._files = files

    def write(self, data: str) -> None:
        for f in self._files:
            f.write(data)

    def flush(self) -> None:
        for f in self._files:
            f.flush()

    def isatty(self) -> bool:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven day trading backtester")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore local cache and refetch all data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_cache:
        os.environ["DTB_REFRESH_CACHE"] = "1"
        print("Cache mode: REFRESH (bypass local cache for this run)")

    cfg = load_config(args.config)

    # ── Tee stdout → timestamped log file ────────────────────────────────────
    log_dir = "artifacts/logs"
    os.makedirs(log_dir, exist_ok=True)
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{ts_str}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    try:
        _run(cfg, args)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()

    print(f"\nFull log saved → {log_path}")


def _run(cfg, args) -> None:
    print_run_config(cfg)
    print(f"\nLoading data: {cfg.data.symbol} {cfg.data.interval} ({cfg.data.period}) from Yahoo Finance...")
    df = load_intraday_bars(cfg.data)
    print(f"Loaded {len(df):,} bars  |  {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}\n")

    strategy = get_strategy(cfg.strategy.name)

    # Count raw signal opportunities before any backtest filtering.
    print("Scanning bars for raw signals...", end=" ", flush=True)
    signal_stats = count_signals(df, strategy, cfg)
    print(
        f"found {signal_stats['total_signals']} signals "
        f"(long={signal_stats['long_signals']}, short={signal_stats['short_signals']}) "
        f"across {signal_stats['bars_scanned']:,} bars."
    )

    trades = run_backtest(df, strategy, cfg)
    print(f"Backtest complete: {len(trades)} trade(s) simulated.")

    trades = enrich_trades_with_option_prices(trades, cfg)
    print(f"Option enrichment complete: {len(trades)} trade(s) with pricing.\n")

    print_trade_log(trades)
    print_summary(trades, cfg=cfg, signal_stats=signal_stats)

    csv_path = save_trade_log_csv(trades, cfg, output_path="artifacts/trades/trade_log.csv")
    print(f"\nCSV saved → {csv_path}")

    prepared = strategy.prepare(df, cfg.strategy.params).dropna().copy()
    chart_paths = save_trade_charts(prepared, trades, output_dir="artifacts/trade_charts", count=2)
    if chart_paths:
        print("Charts saved:")
        for p in chart_paths:
            print(f"  → {p}")
    else:
        print("Charts skipped (matplotlib not installed or no trades).")


if __name__ == "__main__":
    main()
