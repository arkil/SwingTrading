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


def run_backtest(df: pd.DataFrame, strategy: BaseStrategy, cfg: BacktestConfig) -> list[TradeResult]:
    prepared = strategy.prepare(df, cfg.strategy.params).dropna().copy()
    if prepared.empty:
        return []

    risk = cfg.risk
    trades: list[TradeResult] = []
    i = 0

    while i < len(prepared) - risk.hold_bars - 1:
        signal = strategy.signal(prepared, i, cfg.strategy.params)
        if signal is None:
            i += 1
            continue

        trade = _simulate_trade(prepared, i, signal, cfg)
        if trade is not None:
            trades.append(trade)
            i += max(1, trade.bars_held)
        else:
            i += 1

    return trades


def _simulate_trade(df: pd.DataFrame, idx: int, signal: Signal, cfg: BacktestConfig) -> TradeResult | None:
    risk = cfg.risk
    entry_row = df.iloc[idx]
    entry_time = df.index[idx]
    entry_price = float(entry_row["close"])
    trade_day = entry_time.date()
    atr_value = float(entry_row.get("atr", 0.0))
    target_price = _target_price(signal.side, entry_price, risk.option_target_pct, risk.option_leverage)

    if signal.side == "long":
        stop_price = entry_price - (risk.stop_atr_mult * atr_value)
    else:
        stop_price = entry_price + (risk.stop_atr_mult * atr_value)

    max_exit_idx = min(len(df) - 1, idx + risk.hold_bars)
    exit_idx = max_exit_idx
    hit_target = False
    exit_reason = "time_exit"
    exit_price = float(df.iloc[exit_idx]["close"])

    for j in range(idx + 1, max_exit_idx + 1):
        if df.index[j].date() != trade_day:
            exit_idx = j - 1
            exit_price = float(df.iloc[exit_idx]["close"])
            exit_reason = "day_end_exit"
            break

        bar_high = float(df.iloc[j]["high"])
        bar_low = float(df.iloc[j]["low"])

        if signal.side == "long":
            stop_hit = bar_low <= stop_price
            target_hit = bar_high >= target_price
        else:
            stop_hit = bar_high >= stop_price
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
        pricing_mode="synthetic",
        option_lookup_status="synthetic",
    )
