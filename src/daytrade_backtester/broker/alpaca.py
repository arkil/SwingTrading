from __future__ import annotations

import logging
import os
from datetime import date, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    AssetClass,
    ContractType,
    OrderSide,
    OrderType,
    TimeInForce,
)
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    MarketOrderRequest,
)

from daytrade_backtester.broker.base import BrokerAdapter, OptionContract, OrderResult, Quote

log = logging.getLogger(__name__)


class AlpacaBroker(BrokerAdapter):
    """
    Alpaca Markets broker adapter (paper or live).

    Authentication via environment variables:
        ALPACA_API_KEY    — Alpaca API key ID
        ALPACA_SECRET_KEY — Alpaca secret key

    Or pass api_key / secret_key explicitly.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ) -> None:
        key = api_key or os.environ.get("ALPACA_API_KEY", "")
        secret = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            raise ValueError(
                "Alpaca credentials missing. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )
        from alpaca.data.historical.option import OptionHistoricalDataClient
        self._trading = TradingClient(key, secret, paper=paper)
        self._data = StockHistoricalDataClient(key, secret)
        self._opt_data = OptionHistoricalDataClient(key, secret)
        self._paper = paper
        log.info("AlpacaBroker initialised (paper=%s)", paper)

    # ── Underlying price ──────────────────────────────────────────────────────

    def get_underlying_price(self, symbol: str) -> float:
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        resp = self._data.get_stock_latest_trade(req)
        price = float(resp[symbol].price)
        log.debug("Latest trade %s: %.4f", symbol, price)
        return price

    # ── Option contract lookup ────────────────────────────────────────────────

    def find_option_contract(
        self,
        underlying: str,
        right: str,
        dte_target: int,
        otm_steps: int,
    ) -> OptionContract | None:
        """
        Find the best-matching option contract.

        1. Get ATM strike from current underlying price.
        2. Select OTM strike by `otm_steps` (calls: strike > ATM, puts: strike < ATM).
        3. Find nearest expiry >= dte_target days from today.
        """
        spot = self.get_underlying_price(underlying)
        today = date.today()
        min_expiry = today + timedelta(days=dte_target)
        max_expiry = today + timedelta(days=dte_target + 14)  # 2-week window

        contract_type = ContractType.CALL if right.upper() == "C" else ContractType.PUT

        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            expiration_date_gte=min_expiry.isoformat(),
            expiration_date_lte=max_expiry.isoformat(),
            type=contract_type,
            status="active",
            limit=200,
        )
        try:
            resp = self._trading.get_option_contracts(req)
        except Exception as exc:
            log.error("Option contract lookup failed: %s", exc)
            return None

        # SDK returns OptionContractsResponse which iterates as
        # [('option_contracts', [list_of_dicts])].  Unwrap it.
        raw = list(resp)
        if raw and isinstance(raw[0], tuple):
            contracts = raw[0][1]
        else:
            contracts = raw

        if not contracts:
            log.warning(
                "No %s %s contracts found for dte_target=%d", underlying, right, dte_target
            )
            return None

        def _get(c, key):
            return c[key] if isinstance(c, dict) else getattr(c, key)

        # Sort by expiration (nearest first), then by strike closeness to spot
        contracts.sort(key=lambda c: (_get(c, "expiration_date"), abs(float(_get(c, "strike_price")) - spot)))

        nearest_expiry = _get(contracts[0], "expiration_date")
        expiry_group = [c for c in contracts if _get(c, "expiration_date") == nearest_expiry]

        if right.upper() == "C":
            otm_candidates = sorted(
                [c for c in expiry_group if float(_get(c, "strike_price")) >= spot],
                key=lambda c: float(_get(c, "strike_price")),
            )
        else:
            otm_candidates = sorted(
                [c for c in expiry_group if float(_get(c, "strike_price")) <= spot],
                key=lambda c: float(_get(c, "strike_price")),
                reverse=True,
            )

        if not otm_candidates:
            log.warning("No OTM candidates for %s %s near %.2f", underlying, right, spot)
            return None

        idx = min(otm_steps, len(otm_candidates) - 1)
        chosen = otm_candidates[idx]
        contract = OptionContract(
            symbol=_get(chosen, "symbol"),
            underlying=underlying,
            expiration=str(_get(chosen, "expiration_date")),
            strike=float(_get(chosen, "strike_price")),
            right=right.upper(),
        )
        log.info(
            "Selected contract %s  expiry=%s  strike=%.2f  spot=%.2f",
            contract.symbol, contract.expiration, contract.strike, spot,
        )
        return contract

    # ── Quotes ────────────────────────────────────────────────────────────────

    def get_quote(self, option_symbol: str) -> Quote | None:
        """Return latest bid/ask for an option contract."""
        from alpaca.data.requests import OptionLatestQuoteRequest
        try:
            req = OptionLatestQuoteRequest(symbol_or_symbols=option_symbol)
            resp = self._opt_data.get_option_latest_quote(req)
            q = resp[option_symbol]
            bid = float(q.bid_price) if q.bid_price else 0.0
            ask = float(q.ask_price) if q.ask_price else 0.0
            if bid <= 0 or ask <= 0:
                log.warning("One-sided or empty quote for %s: bid=%.4f ask=%.4f", option_symbol, bid, ask)
                return None
            return Quote(symbol=option_symbol, bid=bid, ask=ask)
        except Exception as exc:
            log.error("Quote fetch failed for %s: %s", option_symbol, exc)
            return None

    # ── Order placement ───────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
    ) -> OrderResult:
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            order = self._trading.submit_order(req)
            log.info("Order submitted: %s %s x%d  id=%s", side.upper(), symbol, qty, order.id)
            return OrderResult(
                order_id=str(order.id),
                symbol=symbol,
                qty=qty,
                side=side,
                status=str(order.status),
            )
        except Exception as exc:
            log.error("Order placement failed for %s: %s", symbol, exc)
            return OrderResult(
                order_id="",
                symbol=symbol,
                qty=qty,
                side=side,
                status="rejected",
            )

    def cancel_order(self, order_id: str) -> None:
        try:
            self._trading.cancel_order_by_id(order_id)
            log.info("Order cancelled: %s", order_id)
        except Exception as exc:
            log.warning("Cancel order %s failed: %s", order_id, exc)

    # ── Position management ───────────────────────────────────────────────────

    def close_position(self, symbol: str) -> OrderResult | None:
        try:
            resp = self._trading.close_position(symbol)
            log.info("Closed position: %s", symbol)
            return OrderResult(
                order_id=str(resp.id),
                symbol=symbol,
                qty=int(resp.qty or 0),
                side="sell",
                status=str(resp.status),
            )
        except Exception as exc:
            log.warning("Close position %s failed: %s", symbol, exc)
            return None

    def get_open_positions(self) -> list[dict]:
        try:
            positions = self._trading.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": int(p.qty),
                    "avg_entry_price": float(p.avg_entry_price or 0),
                    "current_price": float(p.current_price or 0),
                    "asset_class": str(p.asset_class),
                }
                for p in positions
            ]
        except Exception as exc:
            log.error("get_open_positions failed: %s", exc)
            return []
