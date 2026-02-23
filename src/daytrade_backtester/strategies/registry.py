from __future__ import annotations

from daytrade_backtester.strategies.base import BaseStrategy
from daytrade_backtester.strategies.bollinger_rsi_reversal import BollingerRsiReversalStrategy


REGISTRY: dict[str, type[BaseStrategy]] = {
    BollingerRsiReversalStrategy.name: BollingerRsiReversalStrategy,
}


def get_strategy(name: str) -> BaseStrategy:
    key = name.strip().lower()
    if key not in REGISTRY:
        supported = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown strategy '{name}'. Supported: {supported}")
    return REGISTRY[key]()
