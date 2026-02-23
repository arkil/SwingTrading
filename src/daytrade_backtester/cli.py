from __future__ import annotations

import argparse

from daytrade_backtester.config.loader import load_config
from daytrade_backtester.data.yahoo import load_intraday_bars
from daytrade_backtester.engine.backtester import run_backtest
from daytrade_backtester.engine.options_enrich import enrich_trades_with_option_prices
from daytrade_backtester.reporting.charts import save_trade_charts
from daytrade_backtester.reporting.console import print_run_config, print_summary, print_trade_log
from daytrade_backtester.reporting.export import save_trade_log_csv
from daytrade_backtester.strategies.registry import get_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven day trading backtester")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    print_run_config(cfg)
    print(f"Loading data: {cfg.data.symbol} {cfg.data.interval} ({cfg.data.period}) from Yahoo...")
    df = load_intraday_bars(cfg.data)
    strategy = get_strategy(cfg.strategy.name)

    trades = run_backtest(df, strategy, cfg)
    trades = enrich_trades_with_option_prices(trades, cfg)

    print_trade_log(trades)
    print_summary(trades)

    csv_path = save_trade_log_csv(trades, cfg, output_path="artifacts/trades/trade_log.csv")
    print(f"Saved trade log CSV: {csv_path}")

    prepared = strategy.prepare(df, cfg.strategy.params).dropna().copy()
    chart_paths = save_trade_charts(prepared, trades, output_dir="artifacts/trade_charts", count=2)
    if chart_paths:
        print("Saved trade screenshots:")
        for p in chart_paths:
            print(f"  - {p}")
    else:
        print("Trade screenshots were not generated (matplotlib not installed).")


if __name__ == "__main__":
    main()
