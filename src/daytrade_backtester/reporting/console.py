from __future__ import annotations

from collections import defaultdict

from daytrade_backtester.config.models import BacktestConfig
from daytrade_backtester.engine.backtester import TradeResult


def _pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def print_run_config(cfg: BacktestConfig) -> None:
    print("=" * 78)
    print(" " * 31 + "RUN CONFIG")
    print("=" * 78)
    print(f"Symbol:                     {cfg.data.symbol}")
    print(f"Interval / Period:          {cfg.data.interval} / {cfg.data.period}")
    print(f"Session:                    {cfg.data.session_start} - {cfg.data.session_end} ({cfg.data.timezone})")
    print(f"Strategy:                   {cfg.strategy.name}")
    print(f"Strategy Params:            {cfg.strategy.params}")
    print(
        "Risk Params:                "
        f"capital={cfg.risk.capital_per_trade}, "
        f"target={cfg.risk.option_target_pct}, "
        f"hold_bars={cfg.risk.hold_bars}, "
        f"leverage={cfg.risk.option_leverage}, "
        f"stop_atr_mult={cfg.risk.stop_atr_mult}, "
        f"commission={cfg.risk.commission_per_trade}"
    )
    print(
        "Options Data:               "
        f"provider={cfg.options.provider}, dte_target_days={cfg.options.dte_target_days}, "
        f"otm_steps={cfg.options.otm_steps}, use_real_prices_for_pnl={cfg.options.use_real_prices_for_pnl}, require_real_prices={cfg.options.require_real_prices}"
    )
    print("=" * 78)


def print_trade_log(trades: list[TradeResult]) -> None:
    print("\n" + "=" * 240)
    print(" " * 113 + "TRADE LOG")
    print("=" * 240)
    header = (
        "#   Entry Time           Exit Time            Side  SpotIn    SpotOut   OptSym                      OptIn    OptOut   "
        "Mode        Lookup Status                  Exit Reason     Opt Ret%   PnL($)    R"
    )
    print(header)
    print("-" * 240)

    for i, t in enumerate(trades, start=1):
        opt_sym = t.option_contract or "-"
        opt_in = f"{t.option_entry_price:.2f}" if t.option_entry_price is not None else "-"
        opt_out = f"{t.option_exit_price:.2f}" if t.option_exit_price is not None else "-"
        print(
            f"{i:<3} {t.entry_time.strftime('%Y-%m-%d %H:%M'):<19} "
            f"{t.exit_time.strftime('%Y-%m-%d %H:%M'):<19} "
            f"{t.side:<5} "
            f"{t.entry_price:>8.2f}  {t.exit_price:>8.2f}  "
            f"{opt_sym:<26} "
            f"{opt_in:>7}  {opt_out:>7}  "
            f"{t.pricing_mode:<10} "
            f"{t.option_lookup_status:<30} "
            f"{t.exit_reason:<13} "
            f"{(t.option_return_pct * 100):>8.2f}%  "
            f"{t.pnl_usd:>8.2f}  "
            f"{t.r_multiple:>6.2f}"
        )

    print("=" * 240)


def print_summary(trades: list[TradeResult]) -> None:
    print("=" * 78)
    print(" " * 25 + "BACKTEST SUMMARY")
    print("=" * 78)

    total = len(trades)
    wins = sum(1 for t in trades if t.hit_target)
    losses = total - wins
    net_pnl = sum(t.pnl_usd for t in trades)
    total_r = sum(t.r_multiple for t in trades)
    avg_r = (total_r / total) if total else 0.0
    win_rate = (wins / total) if total else 0.0

    print(f"Total Setups Found:          {total}")
    print(f"Completed Trades:            {total}")
    print("Incomplete Trades (No Exit): 0")
    print("-" * 78)
    print(f"Total Net P&L:             ${net_pnl:,.2f}")
    print(f"Total R Multiple Achieved:  {total_r:,.2f} R")
    print("-" * 78)
    print(f"Wins / Losses:              {wins} / {losses}")
    print(f"Win Rate:                   {_pct(win_rate)}")
    print(f"Avg R per Completed Trade:  {avg_r:.2f} R")
    print("=" * 78)
    print("PERFORMANCE BIFURCATED BY TREND (10 EMA vs 20 EMA):")
    print()

    grouped: dict[str, list[TradeResult]] = defaultdict(list)
    for t in trades:
        grouped[t.trend].append(t)

    print("Condition        | Total | Wins/Losses | Win Rate   | Total R    | Avg R")
    print("-" * 78)

    for trend in ("Uptrend", "Downtrend"):
        bucket = grouped.get(trend, [])
        bt = len(bucket)
        bw = sum(1 for t in bucket if t.hit_target)
        bl = bt - bw
        br = sum(t.r_multiple for t in bucket)
        ba = (br / bt) if bt else 0.0
        wr = (bw / bt) if bt else 0.0
        print(f"{trend:<15} | {bt:<5} | {bw:>3} /{bl:<7} | {_pct(wr):<9} | {br:>8.2f} R | {ba:>5.2f} R")

    print("=" * 78)
