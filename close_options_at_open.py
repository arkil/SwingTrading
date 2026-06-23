#!/usr/bin/env python3
"""Run once at market open to close all remaining 45-60 DTE option positions."""
from alpaca.trading.client import TradingClient
import os, time

KEY    = os.environ.get("ALPACA_API_KEY",    "PKB26M3XKTC4I35OWY5BCTSV6S")
SECRET = os.environ.get("ALPACA_SECRET_KEY", "6vryzcDiPpR1pvMxHu1K6MaNn3YqTgEK9L1S7PXEfcdw")

to_close = [
    "META260626P00645000",
    "NVDA260626C00225000",
    "QQQ260626C00710000",
    "TSLA260702C00435000",
]

tc = TradingClient(KEY, SECRET, paper=True)

for sym in to_close:
    try:
        resp = tc.close_position(sym)
        print(f"CLOSED {sym}  order_id={resp.id}")
    except Exception as e:
        print(f"SKIP   {sym}: {e}")

# Also close any remaining SPY pyramid contracts
positions = tc.get_all_positions()
for p in positions:
    if p.symbol.startswith("SPY2602") or p.symbol.startswith("SPY2605") or p.symbol.startswith("SPY2606"):
        try:
            resp = tc.close_position(p.symbol)
            print(f"CLOSED {p.symbol}  (spy pyramid cleanup)")
        except Exception as e:
            print(f"SKIP   {p.symbol}: {e}")
