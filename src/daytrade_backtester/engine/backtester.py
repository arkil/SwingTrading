from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from daytrade_backtester.config.models import BacktestConfig
from daytrade_backtester.strategies.base import BaseStrategy, Signal


@dataclass
class TradeResult:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    bars_held: int
    hit_target: bool
    option_return_pct: float
    pnl_usd: float
    r_multiple: float
    trend: str
    reason: str
    exit_reason: str
    atr_at_entry: float = 0.0
    rsi_at_entry: float | None = None
    bb_pct_at_entry: float | None = None
    option_contract: str | None = None
    option_entry_price: float | None = None
    option_exit_price: float | None = None
    pricing_mode: str = "synthetic"
    option_lookup_status: str = "synthetic"


def _estimate_option_return(side: str, entry_price: float, current_price: float, option_leverage: float) -> float:
    direction = 1.0 if side == "long" else -1.0
    underlying_return = (current_price - entry_price) / entry_price
    return direction * underlying_return * option_leverage


def _target_price(side: str, entry_price: float, option_target_pct: float, option_leverage: float) -> float:
    if option_leverage <= 0:
        return entry_price
    underlying_move = option_target_pct / option_leverage
    if side == "long":
        return entry_price * (1 + underlying_move)
    return entry_price * (1 - underlying_move)


def _entry_context(entry_row: pd.Series) -> tuple[float | None, float | None]:
    rsi_value = entry_row.get("rsi")
    rsi_at_entry = None if pd.isna(rsi_value) else float(rsi_value)

    bb_upper = entry_row.get("bb_upper")
    bb_lower = entry_row.get("bb_lower")
    close = entry_row.get("close")
    bb_pct_at_entry: float | None = None
    if not pd.isna(bb_upper) and not pd.isna(bb_lower) and not pd.isna(close):
        width = float(bb_upper) - float(bb_lower)
        if width > 0:
            bb_pct_at_entry = (float(close) - float(bb_lower)) / width

    return rsi_at_entry, bb_pct_at_entry


def count_signals(df: pd.DataFrame, strategy: BaseStrategy, cfg: BacktestConfig) -> dict[str, object]:
    prepared = strategy.prepare(df, cfg.strategy.params).dropna().copy()
    if prepared.empty:
        return {
            "bars_scanned": 0,
            "total_signals": 0,
            "long_signals": 0,
            "short_signals": 0,
            "signals_by_reason": {},
        }

    max_i = max(0, len(prepared) - cfg.risk.hold_bars - 1)
    total = 0
    long_count = 0
    short_count = 0
    by_reason: dict[str, int] = {}

    for i in range(max_i):
        signal = strategy.signal(prepared, i, cfg.strategy.params)
        if signal is None:
            continue

        total += 1
        if signal.side == "long":
            long_count += 1
        elif signal.side == "short":
            short_count += 1
        by_reason[signal.reason] = by_reason.get(signal.reason, 0) + 1

    return {
        "bars_scanned": max_i,
        "total_signals": total,
        "long_signals": long_count,
        "short_signals": short_count,
        "signals_by_reason": by_reason,
    }


def run_backtest(df: pd.DataFrame, strategy: BaseStrategy, cfg: BacktestConfig) -> list[TradeResult]:
    prepared = strategy.prepare(df, cfg.strategy.params).dropna().copy()
    if prepared.empty:
        return []

    risk = cfg.risk
    trades: list[TradeResult] = []
    i = 0

    provider = cfg.options.provider.strip().lower()
    exit_model = cfg.options.exit_model.strip().lower()
    options_native_exits = (
        exit_model == "option_native"
        and provider not in {"synthetic", "none"}
        and cfg.options.use_real_prices_for_pnl
    )

    cooldown_bars = max(int(getattr(risk, "cooldown_bars", 0)), 0)
    max_trades_per_day = max(int(getattr(risk, "max_trades_per_day", 0)), 0)
    cur_day = None
    trades_today = 0

    while i < len(prepared) - risk.hold_bars - 1:
        ts = prepared.index[i]
        day = ts.date()
        if day != cur_day:
            cur_day = day
            trades_today = 0

        if max_trades_per_day > 0 and trades_today >= max_trades_per_day:
            i += 1
            continue

        signal = strategy.signal(prepared, i, cfg.strategy.params)
        if signal is None:
            i += 1
            continue

        trade = _simulate_trade(prepared, i, signal, cfg, options_native_exits=options_native_exits)
        if trade is not None:
            trades.append(trade)
            trades_today += 1
            i += max(1, trade.bars_held + cooldown_bars)
        else:
            i += 1

    return trades


def _simulate_trade(
    df: pd.DataFrame,
    idx: int,
    signal: Signal,
    cfg: BacktestConfig,
    options_native_exits: bool = False,
) -> TradeResult | None:
    risk = cfg.risk
    entry_row = df.iloc[idx]
    entry_time = df.index[idx]
    entry_price = float(entry_row["close"])
    trade_day = entry_time.date()
    atr_value = float(entry_row.get("atr", 0.0))
    rsi_at_entry, bb_pct_at_entry = _entry_context(entry_row)

    max_exit_idx = min(len(df) - 1, idx + risk.hold_bars)
    exit_idx = max_exit_idx
    hit_target = False
    exit_reason = "time_exit"
    exit_price = float(df.iloc[exit_idx]["close"])

    if options_native_exits:
        for j in range(idx + 1, max_exit_idx + 1):
            if df.index[j].date() != trade_day:
                exit_idx = j - 1
                exit_price = float(df.iloc[exit_idx]["close"])
                exit_reason = "day_end_exit"
                break
            if j == max_exit_idx:
                exit_price = float(df.iloc[j]["close"])
                exit_reason = "time_exit"

        if exit_idx <= idx:
            return None

        exit_time = df.index[exit_idx]
        option_return_pct = _estimate_option_return(signal.side, entry_price, exit_price, risk.option_leverage)
        gross_pnl = option_return_pct * risk.capital_per_trade
        net_pnl = gross_pnl - risk.commission_per_trade
        r_multiple = option_return_pct / risk.option_target_pct if risk.option_target_pct > 0 else 0.0
        trend = "Uptrend" if float(entry_row["ema10"]) > float(entry_row["ema20"]) else "Downtrend"

        return TradeResult(
            entry_time=entry_time,
            exit_time=exit_time,
            side=signal.side,
            entry_price=entry_price,
            exit_price=exit_price,
            bars_held=exit_idx - idx,
            hit_target=hit_target,
            option_return_pct=option_return_pct,
            pnl_usd=net_pnl,
            r_multiple=r_multiple,
            trend=trend,
            reason=signal.reason,
            exit_reason=exit_reason,
            atr_at_entry=atr_value,
            rsi_at_entry=rsi_at_entry,
            bb_pct_at_entry=bb_pct_at_entry,
            pricing_mode="synthetic",
            option_lookup_status="synthetic",
        )

    target_price = _target_price(signal.side, entry_price, risk.option_target_pct, risk.option_leverage)
    stop_enabled = bool(risk.stop_atr_mult > 0 and atr_value > 0)
    if signal.side == "long":
        stop_price = entry_price - (risk.stop_atr_mult * atr_value)
    else:
        stop_price = entry_price + (risk.stop_atr_mult * atr_value)

    early_exit_bar = int(risk.early_exit_bar) if hasattr(risk, "early_exit_bar") else 0
    early_exit_pct = float(risk.early_exit_pct) if hasattr(risk, "early_exit_pct") else -1.0
    use_early_exit = early_exit_bar > 0 and early_exit_pct < 0.0

    for j in range(idx + 1, max_exit_idx + 1):
        if df.index[j].date() != trade_day:
            exit_idx = j - 1
            exit_price = float(df.iloc[exit_idx]["close"])
            exit_reason = "day_end_exit"
            break

        bar_high = float(df.iloc[j]["high"])
        bar_low  = float(df.iloc[j]["low"])
        bar_close = float(df.iloc[j]["close"])

        if signal.side == "long":
            stop_hit   = stop_enabled and (bar_low <= stop_price)
            target_hit = bar_high >= target_price
        else:
            stop_hit   = stop_enabled and (bar_high >= stop_price)
            target_hit = bar_low <= target_price

        if stop_hit:
            exit_idx = j
            exit_price = stop_price
            exit_reason = "atr_stop"
            break

        if target_hit:
            exit_idx = j
            exit_price = target_price
            hit_target = True
            exit_reason = "profit_target"
            break

        # Early exit: >= so missing one bar's data can't disable the cut entirely.
        if use_early_exit and (j - idx) >= early_exit_bar:
            opt_ret_now = _estimate_option_return(signal.side, entry_price, bar_close, risk.option_leverage)
            if opt_ret_now <= early_exit_pct:
                exit_idx = j
                exit_price = bar_close
                exit_reason = "early_exit"
                break

        if j == max_exit_idx:
            exit_price = bar_close
            exit_reason = "time_exit"

    if exit_idx <= idx:
        return None

    exit_time = df.index[exit_idx]
    option_return_pct = _estimate_option_return(signal.side, entry_price, exit_price, risk.option_leverage)

    gross_pnl = option_return_pct * risk.capital_per_trade
    net_pnl = gross_pnl - risk.commission_per_trade
    r_multiple = option_return_pct / risk.option_target_pct if risk.option_target_pct > 0 else 0.0

    trend = "Uptrend" if float(entry_row["ema10"]) > float(entry_row["ema20"]) else "Downtrend"

    return TradeResult(
        entry_time=entry_time,
        exit_time=exit_time,
        side=signal.side,
        entry_price=entry_price,
        exit_price=exit_price,
        bars_held=exit_idx - idx,
        hit_target=hit_target,
        option_return_pct=option_return_pct,
        pnl_usd=net_pnl,
        r_multiple=r_multiple,
        trend=trend,
        reason=signal.reason,
        exit_reason=exit_reason,
        atr_at_entry=atr_value,
        rsi_at_entry=rsi_at_entry,
        bb_pct_at_entry=bb_pct_at_entry,
        pricing_mode="synthetic",
        option_lookup_status="synthetic",
    )
