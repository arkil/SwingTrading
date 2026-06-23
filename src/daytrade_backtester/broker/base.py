from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OptionContract:
    symbol: str          # e.g. "SPY260314C00685000"
    underlying: str      # "SPY"
    expiration: str      # "2026-03-14"
    strike: float        # 685.0
    right: str           # "C" | "P"


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    qty: int
    side: str            # "buy" | "sell"
    status: str          # "accepted" | "filled" | "rejected"
    filled_price: float | None = None
    filled_at: datetime | None = None


class BrokerAdapter(ABC):
    """Abstract broker interface. Swap implementations (Alpaca, IBKR, etc.) without changing callers."""

    @abstractmethod
    def find_option_contract(
        self,
        underlying: str,
        right: str,          # "C" (call) | "P" (put)
        dte_target: int,     # minimum days to expiration
        otm_steps: int,      # strikes OTM from ATM
    ) -> OptionContract | None:
        """Find the nearest suitable option contract."""

    @abstractmethod
    def get_underlying_price(self, symbol: str) -> float:
        """Return latest trade/mid price for the underlying."""

    @abstractmethod
    def get_quote(self, option_symbol: str) -> Quote | None:
        """Return current bid/ask for an option contract."""

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,           # "buy" | "sell"
        order_type: str = "market",
    ) -> OrderResult:
        """Place an order. For options: symbol is the OCC contract string."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel a pending order."""

    @abstractmethod
    def close_position(self, symbol: str) -> OrderResult | None:
        """Sell all held qty of symbol at market."""

    @abstractmethod
    def get_open_positions(self) -> list[dict]:
        """Return list of open positions as dicts with keys: symbol, qty, avg_entry_price, current_price."""
