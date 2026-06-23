from __future__ import annotations

import os

import pandas as pd

from daytrade_backtester.config.models import BacktestConfig
from daytrade_backtester.data.options_polygon import (
    lookup_option_contract_and_bars_polygon,
    lookup_option_entry_exit_polygon,
)
from daytrade_backtester.data.options_yahoo import lookup_option_contract_and_bars, lookup_option_entry_exit
from daytrade_backtester.engine.backtester import TradeResult


def _slippage_bps(cfg: BacktestConfig) -> float:
    try:
        return max(float(cfg.options.provider_params.get("slippage_bps", 0.0)), 0.0)
    except Exception:
        return 0.0


def _min_option_volume(cfg: BacktestConfig) -> float:
    try:
        return max(float(cfg.options.provider_params.get("min_option_volume", 0.0)), 0.0)
    except Exception:
        return 0.0


def _min_option_price(cfg: BacktestConfig) -> float:
    try:
        return max(float(cfg.options.provider_params.get("min_option_price", 0.05)), 0.0)
    except Exception:
        return 0.05


def _apply_slippage(entry_price: float, exit_price: float, slippage_bps: float) -> tuple[float, float]:
    if slippage_bps <= 0:
        return entry_price, exit_price
    slip = slippage_bps / 10_000.0
    # Buy-to-open then sell-to-close; both sides penalized.
    return entry_price * (1.0 + slip), exit_price * (1.0 - slip)


def _apply_real_option_pnl(t: TradeResult, cfg: BacktestConfig, *, apply_slippage: bool = True) -> None:
    if t.option_entry_price is None or t.option_exit_price is None or t.option_entry_price <= 0:
        return

    entry = float(t.option_entry_price)
    exit_ = float(t.option_exit_price)
    if apply_slippage:
        entry, exit_ = _apply_slippage(entry, exit_, _slippage_bps(cfg))
        t.option_entry_price = entry
        t.option_exit_price = exit_

    option_return_pct = (exit_ / entry) - 1
    t.option_return_pct = option_return_pct
    t.pnl_usd = (option_return_pct * cfg.risk.capital_per_trade) - cfg.risk.commission_per_trade
    t.r_multiple = option_return_pct / cfg.risk.option_target_pct if cfg.risk.option_target_pct > 0 else 0.0
    t.hit_target = (t.exit_reason == "profit_target") or (option_return_pct >= cfg.risk.option_target_pct)


def _bar_near_time(df: pd.DataFrame, ts: pd.Timestamp) -> tuple[pd.Timestamp | None, pd.Series | None]:
    if df.empty:
        return None, None

    right = df.loc[df.index >= ts]
    if not right.empty:
        t = right.index[0]
        return t, right.iloc[0]

    left = df.loc[df.index <= ts]
    if not left.empty:
        t = left.index[-1]
        return t, left.iloc[-1]

    return None, None


def _simulate_option_native_exit(t: TradeResult, bars: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.Timestamp, float, str, bool, float]:
    bars = bars.sort_index()

    entry_ts, entry_row = _bar_near_time(bars, t.entry_time)
    if entry_ts is None or entry_row is None:
        raise ValueError("missing_entry_or_exit_price")

    entry_opt = float(entry_row.get("close", 0.0) or 0.0)
    if entry_opt <= 0:
        raise ValueError("missing_entry_or_exit_price")

    min_px = _min_option_price(cfg)
    if entry_opt < min_px:
        raise ValueError("option_price_too_low")

    min_vol = _min_option_volume(cfg)
    if min_vol > 0:
        entry_vol = float(entry_row.get("volume", 0.0) or 0.0)
        if entry_vol < min_vol:
            raise ValueError("insufficient_option_volume")

    end_bars = bars.loc[bars.index <= t.exit_time]
    if end_bars.empty:
        raise ValueError("missing_entry_or_exit_price")

    target_price = entry_opt * (1.0 + cfg.risk.option_target_pct)

    # Stop is disabled when stop_atr_mult <= 0 (original strategy: time exit only).
    use_stop = cfg.risk.stop_atr_mult > 0
    stop_price = 0.0
    if use_stop and t.entry_price > 0 and t.atr_at_entry > 0:
        stop_underlying_pct = (cfg.risk.stop_atr_mult * t.atr_at_entry) / t.entry_price
        stop_pct = stop_underlying_pct * cfg.risk.option_leverage
        if stop_pct > 0:
            stop_price = entry_opt * (1.0 - stop_pct)
        else:
            use_stop = False

    iter_bars = end_bars.loc[end_bars.index >= entry_ts]
    if iter_bars.empty:
        raise ValueError("missing_entry_or_exit_price")

    early_exit_bar = int(getattr(cfg.risk, "early_exit_bar", 0))
    early_exit_pct = float(getattr(cfg.risk, "early_exit_pct", -1.0))
    use_early_exit = early_exit_bar > 0 and early_exit_pct < 0.0

    prev_ts = entry_ts
    prev_close = entry_opt
    first = True
    bar_count = 0
    for ts, row in iter_bars.iterrows():
        if first:
            first = False
            continue

        bar_count += 1

        if ts.date() != entry_ts.date():
            return prev_ts, prev_close, "day_end_exit", False, entry_opt

        close_px = float(row.get("close", prev_close) or prev_close)
        high_px  = float(row.get("high",  close_px)   or close_px)
        low_px   = float(row.get("low",   close_px)   or close_px)

        # Stop check (skipped when stop_atr_mult=0).
        if use_stop and low_px <= stop_price:
            return ts, stop_price, "atr_stop", False, entry_opt

        if high_px >= target_price:
            return ts, target_price, "profit_target", True, entry_opt

        # Early exit: >= matches live position_monitor behavior.
        if use_early_exit and bar_count >= early_exit_bar:
            opt_return = (close_px / entry_opt) - 1.0
            if opt_return <= early_exit_pct:
                return ts, close_px, "early_exit", False, entry_opt

        prev_ts = ts
        prev_close = close_px

    return prev_ts, prev_close, "time_exit", False, entry_opt


def enrich_trades_with_option_prices(trades: list[TradeResult], cfg: BacktestConfig) -> list[TradeResult]:
    provider = cfg.options.provider.strip().lower()
    if provider in {"synthetic", "none"}:
        return trades

    exit_model = cfg.options.exit_model.strip().lower()
    option_native = (
        exit_model == "option_native"
        and provider not in {"synthetic", "none"}
        and cfg.options.use_real_prices_for_pnl
    )

    for t in trades:
        if provider == "yahoo":
            if option_native:
                contract, bars, status = lookup_option_contract_and_bars(
                    symbol=cfg.data.symbol,
                    side=t.side,
                    entry_time=t.entry_time,
                    end_time=t.exit_time,
                    spot_entry=t.entry_price,
                    interval=cfg.data.interval,
                    timezone=cfg.data.timezone,
                    dte_target_days=cfg.options.dte_target_days,
                    otm_steps=cfg.options.otm_steps,
                )
                if status == "ok":
                    try:
                        exit_ts, opt_exit, exit_reason, hit_target, opt_entry = _simulate_option_native_exit(t, bars, cfg)
                        t.exit_time = exit_ts
                        t.exit_reason = exit_reason
                        t.hit_target = hit_target
                        t.option_contract = contract
                        t.option_entry_price = opt_entry
                        t.option_exit_price = opt_exit
                        t.option_lookup_status = "ok"
                        t.pricing_mode = f"{provider}_real"
                        _apply_real_option_pnl(t, cfg, apply_slippage=True)
                    except ValueError as e:
                        t.option_contract = contract
                        t.option_lookup_status = str(e)
                        t.pricing_mode = "synthetic"
                else:
                    t.option_contract = contract
                    t.option_lookup_status = status
                    t.pricing_mode = "synthetic"
            else:
                contract, opt_entry, opt_exit, status = lookup_option_entry_exit(
                    symbol=cfg.data.symbol,
                    side=t.side,
                    entry_time=t.entry_time,
                    exit_time=t.exit_time,
                    spot_entry=t.entry_price,
                    interval=cfg.data.interval,
                    timezone=cfg.data.timezone,
                    dte_target_days=cfg.options.dte_target_days,
                    otm_steps=cfg.options.otm_steps,
                )
                t.option_contract = contract
                t.option_entry_price = opt_entry
                t.option_exit_price = opt_exit
                t.option_lookup_status = status
                t.pricing_mode = f"{provider}_real" if status == "ok" else "synthetic"
                if cfg.options.use_real_prices_for_pnl and status == "ok":
                    _apply_real_option_pnl(t, cfg, apply_slippage=True)

        elif provider == "polygon":
            pp = cfg.options.provider_params
            api_key = str(pp.get("api_key") or os.getenv("MASSIVE_API_KEY", ""))
            if option_native:
                contract, bars, status = lookup_option_contract_and_bars_polygon(
                    symbol=cfg.data.symbol,
                    side=t.side,
                    entry_time=t.entry_time,
                    end_time=t.exit_time,
                    spot_entry=t.entry_price,
                    interval=cfg.data.interval,
                    dte_target_days=cfg.options.dte_target_days,
                    otm_steps=cfg.options.otm_steps,
                    api_key=api_key,
                    base_url=str(pp.get("base_url", "https://api.massive.com")),
                    timeout_sec=float(pp.get("timeout_sec", 20.0)),
                    max_retries=int(pp.get("max_retries", 4)),
                    backoff_sec=float(pp.get("backoff_sec", 0.8)),
                    max_pages=int(pp.get("max_pages", 3)),
                )
                if status == "ok":
                    try:
                        exit_ts, opt_exit, exit_reason, hit_target, opt_entry = _simulate_option_native_exit(t, bars, cfg)
                        t.exit_time = exit_ts
                        t.exit_reason = exit_reason
                        t.hit_target = hit_target
                        t.option_contract = contract
                        t.option_entry_price = opt_entry
                        t.option_exit_price = opt_exit
                        t.option_lookup_status = "ok"
                        t.pricing_mode = f"{provider}_real"
                        _apply_real_option_pnl(t, cfg, apply_slippage=True)
                    except ValueError as e:
                        t.option_contract = contract
                        t.option_lookup_status = str(e)
                        t.pricing_mode = "synthetic"
                else:
                    t.option_contract = contract
                    t.option_lookup_status = status
                    t.pricing_mode = "synthetic"
            else:
                contract, opt_entry, opt_exit, status = lookup_option_entry_exit_polygon(
                    symbol=cfg.data.symbol,
                    side=t.side,
                    entry_time=t.entry_time,
                    exit_time=t.exit_time,
                    spot_entry=t.entry_price,
                    interval=cfg.data.interval,
                    dte_target_days=cfg.options.dte_target_days,
                    otm_steps=cfg.options.otm_steps,
                    api_key=api_key,
                    base_url=str(pp.get("base_url", "https://api.massive.com")),
                    timeout_sec=float(pp.get("timeout_sec", 20.0)),
                    max_retries=int(pp.get("max_retries", 4)),
                    backoff_sec=float(pp.get("backoff_sec", 0.8)),
                    max_pages=int(pp.get("max_pages", 3)),
                )
                t.option_contract = contract
                t.option_entry_price = opt_entry
                t.option_exit_price = opt_exit
                t.option_lookup_status = status
                t.pricing_mode = f"{provider}_real" if status == "ok" else "synthetic"
                if cfg.options.use_real_prices_for_pnl and status == "ok":
                    _apply_real_option_pnl(t, cfg, apply_slippage=True)

        elif provider in {"alpaca", "marketdata"}:
            t.option_contract, t.option_entry_price, t.option_exit_price, t.option_lookup_status = (
                None,
                None,
                None,
                "provider_not_implemented",
            )
            t.pricing_mode = "synthetic"
        else:
            t.option_contract, t.option_entry_price, t.option_exit_price, t.option_lookup_status = (
                None,
                None,
                None,
                f"unknown_provider:{provider}",
            )
            t.pricing_mode = "synthetic"

    if cfg.options.require_real_prices:
        real_trades = [t for t in trades if t.option_lookup_status == "ok"]
        if not real_trades:
            raise ValueError(
                "No trades with real option prices were found. "
                "Increase retries/backoff, reduce date range, or verify provider plan/permissions."
            )
        return real_trades

    return trades
