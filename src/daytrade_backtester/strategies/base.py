from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Signal:
    side: str  # "long" for call buy, "short" for put buy
    reason: str


class BaseStrategy(ABC):
    name: str

    @abstractmethod
    def prepare(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Add indicator columns used by this strategy."""

    @abstractmethod
    def signal(self, df: pd.DataFrame, idx: int, params: dict) -> Signal | None:
        """Return signal for the bar index or None."""
