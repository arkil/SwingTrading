from __future__ import annotations

import os

from daytrade_backtester.config.models import BacktestConfig
from daytrade_backtester.data.options_polygon import lookup_option_entry_exit_polygon
from daytrade_backtester.data.options_yahoo import lookup_option_entry_exit
from daytrade_backtester.engine.backtester import TradeResult


def _apply_real_option_pnl(t: TradeResult, cfg: BacktestConfig) -> None:
    if t.option_entry_price is None or t.option_exit_price is None or t.option_entry_price <= 0:
        return

    option_return_pct = (t.option_exit_price / t.option_entry_price) - 1
    t.option_return_pct = option_return_pct
    t.pnl_usd = (option_return_pct * cfg.risk.capital_per_trade) - cfg.risk.commission_per_trade
    t.r_multiple = option_return_pct / cfg.risk.option_target_pct if cfg.risk.option_target_pct > 0 else 0.0
    t.hit_target = option_return_pct >= cfg.risk.option_target_pct


def enrich_trades_with_option_prices(trades: list[TradeResult], cfg: BacktestConfig) -> list[TradeResult]:
    provider = cfg.options.provider.strip().lower()
    if provider in {"synthetic", "none"}:
        return trades

    for t in trades:
        if provider == "yahoo":
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
        elif provider == "polygon":
            pp = cfg.options.provider_params
            api_key = str(pp.get("api_key") or os.getenv("MASSIVE_API_KEY", ""))
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
        elif provider in {"alpaca", "marketdata"}:
            contract, opt_entry, opt_exit, status = None, None, None, "provider_not_implemented"
        else:
            contract, opt_entry, opt_exit, status = None, None, None, f"unknown_provider:{provider}"

        t.option_contract = contract
        t.option_entry_price = opt_entry
        t.option_exit_price = opt_exit
        t.option_lookup_status = status
        t.pricing_mode = f"{provider}_real" if status == "ok" else "synthetic"

        if cfg.options.use_real_prices_for_pnl and status == "ok":
            _apply_real_option_pnl(t, cfg)

    if cfg.options.require_real_prices:
        real_trades = [t for t in trades if t.option_lookup_status == "ok"]
        if not real_trades:
            raise ValueError(
                "No trades with real option prices were found. "
                "Increase retries/backoff, reduce date range, or verify provider plan/permissions."
            )
        return real_trades

    return trades
