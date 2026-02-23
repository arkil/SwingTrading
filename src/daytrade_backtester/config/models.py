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


@dataclass
class RiskConfig:
    capital_per_trade: float = 1000.0
    option_target_pct: float = 0.05
    hold_bars: int = 3
    option_leverage: float = 20.0
    stop_atr_mult: float = 1.0
    commission_per_trade: float = 0.0


@dataclass
class StrategyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptionsConfig:
    # provider: synthetic | yahoo | polygon | alpaca | marketdata
    provider: str = "yahoo"
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
