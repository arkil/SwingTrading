from __future__ import annotations

from pathlib import Path

import pandas as pd

from daytrade_backtester.config.models import BacktestConfig
from daytrade_backtester.engine.backtester import TradeResult


def _fallback_option_contract(symbol: str, side: str) -> str:
    right = "C" if side == "long" else "P"
    return f"{symbol}-SIM-{right}"


def save_trade_log_csv(trades: list[TradeResult], cfg: BacktestConfig, output_path: str = "artifacts/trades/trade_log.csv") -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i, t in enumerate(trades, start=1):
        rows.append(
            {
                "trade_id": i,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": t.side,
                "option_contract": t.option_contract or _fallback_option_contract(cfg.data.symbol, t.side),
                "option_entry_price": t.option_entry_price,
                "option_exit_price": t.option_exit_price,
                "underlying_entry_price": t.entry_price,
                "underlying_exit_price": t.exit_price,
                "bars_held": t.bars_held,
                "signal_reason": t.reason,
                "exit_reason": t.exit_reason,
                "hit_target": t.hit_target,
                "option_return_pct": t.option_return_pct,
                "pnl_usd": t.pnl_usd,
                "r_multiple": t.r_multiple,
                "trend": t.trend,
                "pricing_mode": t.pricing_mode,
                "option_lookup_status": t.option_lookup_status,
            }
        )

    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)
