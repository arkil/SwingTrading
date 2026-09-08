#!/usr/bin/env python3
"""
Expiry Safety Sweep
====================
Closes every open option position that expires today (or earlier — covers a
missed prior run) before Alpaca can auto-exercise an in-the-money long option
into a forced stock position. options_paper_trader.py, v6_live_daemon.py, and
the spy-pyramid bot are all long-only (buy calls/puts to open) — none of them
sell naked options — but a long option left open through expiration still
gets auto-exercised by the broker if it's ITM, which can force a stock
purchase the account doesn't have free cash to cover (i.e. margin).

Each bot's own monitor loop also force-closes its positions before expiry
(see options_paper_trader.py and pyramid_monitor.py) — this script is the
independent backstop that still runs even if a bot's daemon is stopped,
crashed, or restarted mid-session and a position gets left unwatched.

Scheduled via launchd (com.swingtrading.expiry-sweep) to run every weekday
shortly after market open. Safe to run manually or re-run — a no-op if
nothing is expiring.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))

from alpaca_trader import make_client, get_positions, close_position, parse_option_symbol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_env() -> None:
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def main() -> None:
    _load_env()
    api_key = os.environ.get("ALPACA_API_KEY", "")
    sec_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not sec_key:
        log.error("No Alpaca credentials — set ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
        sys.exit(1)

    client = make_client(api_key, sec_key, paper=True)
    today = date.today()

    positions = get_positions(client)
    to_close = []
    for p in positions:
        parsed = parse_option_symbol(p["symbol"])
        if parsed and parsed["expiration"] <= today:
            to_close.append((p["symbol"], parsed["expiration"]))

    if not to_close:
        log.info("No option positions expiring today or earlier — nothing to do.")
        return

    log.info("Found %d option position(s) at/past expiry — closing:", len(to_close))
    for sym, exp in to_close:
        result = close_position(client, sym)
        if result.get("ok"):
            log.info("  ✅  CLOSED  %-22s  (expired %s)  order_id=%s", sym, exp, result["order_id"])
        else:
            log.error("  ❌  FAILED  %-22s  (expired %s)  %s", sym, exp, result.get("error", ""))


if __name__ == "__main__":
    main()
