from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataConfig:
    symbol: str = "SPY"
    interval: str = "5m"
    period: str = "60d"
    timezone: str = "America/New_York"
    session_start: str = "09:30"
    session_end: str = "16:00"
    date_start: str | None = None   # e.g. "2025-03-01" — overrides period when set
    date_end: str | None = None     # e.g. "2025-05-30" — inclusive end date


@dataclass
class RiskConfig:
    capital_per_trade: float = 1000.0
    option_target_pct: float = 0.05
    hold_bars: int = 3
    option_leverage: float = 20.0
    stop_atr_mult: float = 1.0
    commission_per_trade: float = 0.0
    cooldown_bars: int = 0
    max_trades_per_day: int = 0
    early_exit_bar: int = 0
    early_exit_pct: float = -1.0
    max_risk_usd: float = 1000.0  # max dollar loss per trade (sets dynamic stop_pct)
    # Pyramid scaling fields (used by dtb-pyramid runner)
    scale1_pct: float = 0.015    # +1.5%  → add 2nd contract
    scale2_pct: float = 0.027    # +2.7%  → add 3rd contract
    stop_pct: float = 0.10       # -10%   → close ALL
    target1_pct: float = 0.034   # +3.4%  → sell 1
    target2_pct: float = 0.064   # +6.4%  → sell 1
    target3_pct: float = 0.112   # +11.2% → sell last


@dataclass
class StrategyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptionsConfig:
    # provider: synthetic | yahoo | polygon | alpaca | marketdata
    provider: str = "yahoo"
    # exit_model: underlying (legacy) | option_native (option-bar exits)
    exit_model: str = "underlying"
    dte_target_days: int = 2
    otm_steps: int = 1
    use_real_prices_for_pnl: bool = False
    require_real_prices: bool = False
    provider_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    data: DataConfig
    strategy: StrategyConfig
    risk: RiskConfig
    options: OptionsConfig = field(default_factory=OptionsConfig)
