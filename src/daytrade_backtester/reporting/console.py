from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from daytrade_backtester.config.models import BacktestConfig

from daytrade_backtester.engine.backtester import TradeResult

W = 90  # default section width


def _fmt_pnl(v: float) -> str:
    return f"${v:>10,.2f}"


def _fmt_pct(v: float) -> str:
    return f"{v * 100:>6.1f}%"


def _fmt_r(v: float) -> str:
    return f"{v:>+6.2f} R"


def _win_rate(wins: int, total: int) -> float:
    return wins / total if total else 0.0


def _profit_factor(trades: list[TradeResult]) -> str:
    gross_win = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
    gross_loss = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0))
    if gross_loss == 0:
        return "∞" if gross_win > 0 else "N/A"
    return f"{gross_win / gross_loss:.2f}"


def _max_drawdown(trades: list[TradeResult]) -> float:
    """Peak-to-trough drawdown on cumulative P&L series."""
    peak = 0.0
    dd = 0.0
    cum = 0.0
    for t in trades:
        cum += t.pnl_usd
        if cum > peak:
            peak = cum
        dd = min(dd, cum - peak)
    return dd


def _streaks(trades: list[TradeResult]) -> tuple[int, int]:
    """Returns (max_win_streak, max_loss_streak)."""
    max_w = max_l = cur_w = cur_l = 0
    for t in trades:
        if t.hit_target:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def _section(title: str, width: int = W) -> None:
    print("-" * width)
    print(f"  {title}")
    print("-" * width)


def print_run_config(cfg: BacktestConfig) -> None:
    print("=" * W)
    print(f"{'RUN CONFIG':^{W}}")
    print("=" * W)
    stop_str = f"{cfg.risk.stop_atr_mult}× ATR" if cfg.risk.stop_atr_mult > 0 else "disabled (time-exit only)"
    print(f"  Symbol / Interval / Period : {cfg.data.symbol}  {cfg.data.interval}  ({cfg.data.period})")
    print(f"  Session                    : {cfg.data.session_start} – {cfg.data.session_end}  ({cfg.data.timezone})")
    print(f"  Strategy                   : {cfg.strategy.name}")
    print(f"  BB / RSI                   : BB({cfg.strategy.params.get('bb_length', 20)}, {cfg.strategy.params.get('bb_std', 2.0)})  "
          f"RSI({cfg.strategy.params.get('rsi_length', 14)})  "
          f"oversold≤{cfg.strategy.params.get('rsi_oversold', 30)}  overbought≥{cfg.strategy.params.get('rsi_overbought', 70)}")
    avoid_open = cfg.strategy.params.get("avoid_open_minutes", 0)
    avoid_close = cfg.strategy.params.get("avoid_close_minutes", 0)
    print(f"  Session guards             : avoid open {avoid_open} min  |  avoid close {avoid_close} min")
    print(f"  Capital / Trade            : ${cfg.risk.capital_per_trade:,.0f}  |  commission ${cfg.risk.commission_per_trade:.2f}")
    print(f"  Option target              : {cfg.risk.option_target_pct * 100:.1f}%  |  leverage {cfg.risk.option_leverage}×")
    print(f"  Hold bars (max)            : {cfg.risk.hold_bars} bars  ({cfg.risk.hold_bars * 5} min)")
    print(f"  Stop loss                  : {stop_str}")
    cooldown = getattr(cfg.risk, "cooldown_bars", 0)
    max_tpd = getattr(cfg.risk, "max_trades_per_day", 0)
    print(f"  Cooldown / Max trades/day  : {cooldown} bars  |  {max_tpd if max_tpd else 'unlimited'}")
    print(f"  Options provider           : {cfg.options.provider}  |  exit_model={cfg.options.exit_model}")
    print(f"  DTE target / OTM steps     : {cfg.options.dte_target_days}d  |  {cfg.options.otm_steps} step(s) OTM")
    print(f"  Use real prices for P&L    : {cfg.options.use_real_prices_for_pnl}  |  require_real={cfg.options.require_real_prices}")
    slippage = cfg.options.provider_params.get("slippage_bps", 0)
    print(f"  Slippage                   : {slippage} bps")
    print("=" * W)


def print_trade_log(trades: list[TradeResult]) -> None:
    col = (
        f"{'#':<4} {'Entry':19} {'Exit':14} {'Side':5} "
        f"{'SpotIn':>8} {'SpotOut':>8}  "
        f"{'OptSym':26} {'OptIn':>7} {'OptOut':>7}  "
        f"{'Mode':12} {'Status':28} "
        f"{'Exit Rsn':14} {'RSI':>5} {'%B':>6}  "
        f"{'OptRet%':>8} {'P&L($)':>9} {'R':>6}"
    )
    divider = "=" * len(col)
    print(f"\n{divider}")
    print(f"{'TRADE LOG':^{len(col)}}")
    print(divider)
    print(col)
    print("-" * len(col))

    for i, t in enumerate(trades, start=1):
        opt_sym = (t.option_contract or "-")[:26]
        opt_in  = f"{t.option_entry_price:.2f}" if t.option_entry_price is not None else "-"
        opt_out = f"{t.option_exit_price:.2f}"  if t.option_exit_price  is not None else "-"
        rsi_str = f"{t.rsi_at_entry:.1f}"       if t.rsi_at_entry       is not None else "-"
        pctb_str = f"{t.bb_pct_at_entry:.3f}"   if t.bb_pct_at_entry    is not None else "-"
        exit_dt = t.exit_time.strftime("%m-%d %H:%M")
        print(
            f"{i:<4} {t.entry_time.strftime('%Y-%m-%d %H:%M'):<19} {exit_dt:<14} {t.side:<5} "
            f"{t.entry_price:>8.2f} {t.exit_price:>8.2f}  "
            f"{opt_sym:<26} {opt_in:>7} {opt_out:>7}  "
            f"{t.pricing_mode:<12} {t.option_lookup_status:<28} "
            f"{t.exit_reason:<14} {rsi_str:>5} {pctb_str:>6}  "
            f"{t.option_return_pct * 100:>7.2f}%  {t.pnl_usd:>9.2f}  {t.r_multiple:>+6.2f}"
        )

    print(divider)


def print_summary(
    trades: list[TradeResult],
    cfg: "BacktestConfig | None" = None,
    signal_stats: dict | None = None,
) -> None:
    print("\n" + "=" * W)
    print(f"{'BACKTEST SUMMARY':^{W}}")
    print("=" * W)

    if not trades:
        print("  No trades to summarise.")
        print("=" * W)
        return

    total = len(trades)
    wins  = sum(1 for t in trades if t.hit_target)
    losses = total - wins
    net_pnl  = sum(t.pnl_usd for t in trades)
    total_r  = sum(t.r_multiple for t in trades)
    avg_r    = total_r / total
    win_rate = _win_rate(wins, total)

    win_pnls  = [t.pnl_usd for t in trades if t.hit_target]
    loss_pnls = [t.pnl_usd for t in trades if not t.hit_target]
    avg_win   = sum(win_pnls)  / len(win_pnls)  if win_pnls  else 0.0
    avg_loss  = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
    max_win   = max(win_pnls,  default=0.0)
    max_loss  = min(loss_pnls, default=0.0)
    max_w_str, max_l_str = _streaks(trades)
    max_dd    = _max_drawdown(trades)
    pf_str    = _profit_factor(trades)
    expectancy = net_pnl / total

    avg_bars = sum(t.bars_held for t in trades) / total
    avg_atr  = sum(t.atr_at_entry for t in trades) / total

    rsi_vals = [t.rsi_at_entry for t in trades if t.rsi_at_entry is not None]
    pctb_vals = [t.bb_pct_at_entry for t in trades if t.bb_pct_at_entry is not None]
    avg_rsi  = sum(rsi_vals)  / len(rsi_vals)  if rsi_vals  else 0.0
    avg_pctb = sum(pctb_vals) / len(pctb_vals) if pctb_vals else 0.0

    # Date range from trade timestamps.
    first_entry = min(t.entry_time for t in trades)
    last_exit   = max(t.exit_time  for t in trades)

    # ── OVERVIEW ─────────────────────────────────────────────────────────────
    _section("OVERVIEW")
    print(f"  Date range         : {first_entry.strftime('%Y-%m-%d')} → {last_exit.strftime('%Y-%m-%d')}")
    if signal_stats:
        bars = signal_stats.get("bars_scanned", 0)
        sigs = signal_stats.get("total_signals", 0)
        long_s = signal_stats.get("long_signals", 0)
        short_s = signal_stats.get("short_signals", 0)
        hit_rate = f"{sigs / bars * 100:.2f}%" if bars else "N/A"
        print(f"  Bars scanned       : {bars:,}  |  raw signals: {sigs} ({hit_rate} of bars)")
        print(f"  Raw signals        : long={long_s}  short={short_s}  (before cooldown/day-limit)")
    print(f"  Trades executed    : {total}")

    # ── PROFITABILITY ─────────────────────────────────────────────────────────
    _section("PROFITABILITY")
    print(f"  Net P&L            : {_fmt_pnl(net_pnl)}")
    print(f"  Profit factor      : {pf_str:>10}  (gross_wins / gross_losses)")
    print(f"  Expectancy         : {_fmt_pnl(expectancy)} per trade")
    print(f"  Total R            : {_fmt_r(total_r)}")
    print(f"  Avg R / trade      : {_fmt_r(avg_r)}")

    # ── WIN / LOSS ────────────────────────────────────────────────────────────
    _section("WIN / LOSS")
    print(f"  Win rate           : {_fmt_pct(win_rate)}  ({wins} wins / {losses} losses)")
    print(f"  Avg win P&L        : {_fmt_pnl(avg_win)}")
    print(f"  Avg loss P&L       : {_fmt_pnl(avg_loss)}")
    print(f"  Best trade         : {_fmt_pnl(max_win)}")
    print(f"  Worst trade        : {_fmt_pnl(max_loss)}")
    print(f"  Max win streak     : {max_w_str} trades")
    print(f"  Max loss streak    : {max_l_str} trades")

    # ── RISK / EXECUTION ──────────────────────────────────────────────────────
    _section("RISK / EXECUTION")
    print(f"  Max drawdown       : {_fmt_pnl(max_dd)}")
    print(f"  Avg bars held      : {avg_bars:.1f} bars  ({avg_bars * 5:.0f} min)")
    print(f"  Avg ATR at entry   : {avg_atr:.4f}")
    print(f"  Avg RSI at entry   : {avg_rsi:.1f}")
    print(f"  Avg %B at entry    : {avg_pctb:.3f}  (0=@lower, 1=@upper; <0 or >1 = outside band)")

    # ── EXIT REASON BREAKDOWN ─────────────────────────────────────────────────
    _section("EXIT REASON BREAKDOWN")
    exit_groups: dict[str, list[TradeResult]] = defaultdict(list)
    for t in trades:
        exit_groups[t.exit_reason].append(t)

    hdr = f"  {'Exit Reason':<16} {'Count':>5}  {'Wins':>4}  {'Win%':>6}  {'Total P&L':>11}  {'Avg P&L':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for reason in ["profit_target", "time_exit", "early_exit", "day_end_exit", "atr_stop"]:
        bucket = exit_groups.get(reason, [])
        if not bucket:
            continue
        bt = len(bucket)
        bw = sum(1 for t in bucket if t.hit_target)
        bp = sum(t.pnl_usd for t in bucket)
        print(
            f"  {reason:<16} {bt:>5}  {bw:>4}  {_fmt_pct(_win_rate(bw, bt))}  "
            f"{_fmt_pnl(bp)}  {_fmt_pnl(bp / bt)}"
        )

    # ── SIDE BREAKDOWN ────────────────────────────────────────────────────────
    _section("SIDE BREAKDOWN  (long = calls / short = puts)")
    hdr = f"  {'Side':<7} {'Count':>5}  {'Wins':>4}  {'Win%':>6}  {'Total R':>8}  {'Total P&L':>11}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for side in ("long", "short"):
        bucket = [t for t in trades if t.side == side]
        if not bucket:
            continue
        bt = len(bucket)
        bw = sum(1 for t in bucket if t.hit_target)
        br = sum(t.r_multiple for t in bucket)
        bp = sum(t.pnl_usd for t in bucket)
        print(
            f"  {side:<7} {bt:>5}  {bw:>4}  {_fmt_pct(_win_rate(bw, bt))}  "
            f"{_fmt_r(br)}  {_fmt_pnl(bp)}"
        )

    # ── HOURLY BREAKDOWN ──────────────────────────────────────────────────────
    _section("HOURLY BREAKDOWN  (entry hour, Eastern Time)")
    hour_groups: dict[int, list[TradeResult]] = defaultdict(list)
    for t in trades:
        hour_groups[t.entry_time.hour].append(t)

    hdr = f"  {'Hour (ET)':<10} {'Count':>5}  {'Wins':>4}  {'Win%':>6}  {'Total P&L':>11}  {'Avg P&L':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for hour in sorted(hour_groups):
        bucket = hour_groups[hour]
        bt = len(bucket)
        bw = sum(1 for t in bucket if t.hit_target)
        bp = sum(t.pnl_usd for t in bucket)
        print(
            f"  {hour:02d}:xx       {bt:>5}  {bw:>4}  {_fmt_pct(_win_rate(bw, bt))}  "
            f"{_fmt_pnl(bp)}  {_fmt_pnl(bp / bt)}"
        )

    # ── PRICING MODE ──────────────────────────────────────────────────────────
    _section("PRICING MODE")
    mode_groups: dict[str, list[TradeResult]] = defaultdict(list)
    for t in trades:
        mode_groups[t.pricing_mode].append(t)

    hdr = f"  {'Mode':<14} {'Count':>5}  {'Wins':>4}  {'Win%':>6}  {'Total P&L':>11}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for mode, bucket in sorted(mode_groups.items()):
        bt = len(bucket)
        bw = sum(1 for t in bucket if t.hit_target)
        bp = sum(t.pnl_usd for t in bucket)
        print(
            f"  {mode:<14} {bt:>5}  {bw:>4}  {_fmt_pct(_win_rate(bw, bt))}  {_fmt_pnl(bp)}"
        )

    # ── TREND BREAKDOWN ───────────────────────────────────────────────────────
    _section("TREND AT ENTRY  (EMA-10 vs EMA-20)")
    trend_groups: dict[str, list[TradeResult]] = defaultdict(list)
    for t in trades:
        trend_groups[t.trend].append(t)

    hdr = f"  {'Trend':<12} {'Count':>5}  {'Wins':>4}  {'Win%':>6}  {'Total R':>8}  {'Avg R':>7}  {'Total P&L':>11}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for trend in ("Uptrend", "Downtrend"):
        bucket = trend_groups.get(trend, [])
        if not bucket:
            continue
        bt = len(bucket)
        bw = sum(1 for t in bucket if t.hit_target)
        br = sum(t.r_multiple for t in bucket)
        bp = sum(t.pnl_usd for t in bucket)
        print(
            f"  {trend:<12} {bt:>5}  {bw:>4}  {_fmt_pct(_win_rate(bw, bt))}  "
            f"{_fmt_r(br)}  {_fmt_r(br / bt)}  {_fmt_pnl(bp)}"
        )

    print("=" * W)
