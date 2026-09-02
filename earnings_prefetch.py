#!/usr/bin/env python3
"""
Earnings IV-Crush — nightly prefetch
====================================
Runs the FULL upcoming-earnings scan once (pre-market, via a launchd agent) and
writes the result to disk so the dashboard page loads instantly and never hits
yfinance on a page view.

    python earnings_prefetch.py                      # default universe/window
    python earnings_prefetch.py --universe ndx --days 10
    python earnings_prefetch.py --force              # ignore the earnings-date cache too

Outputs (in Data/earnings_iv/):
    latest.json  — {"rows": [...], "details": {sym: analysis}, "meta": {...}}
                   plain JSON so ANY python (dashboard runs on system 3.9, the
                   prefetch on the .venv) can read it — no pickle / pandas pin
    latest.csv   — the table, for humans / other tools

Reliability: every yfinance call inside `_analyze_impl` already retries with
backoff (see earnings_iv_module._retry). This loop is SERIAL with a small
throttle so we never trigger the "getaddrinfo() thread failed to start"
resource exhaustion that kills a concurrent burst.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from datetime import datetime, timezone

_DIR = os.path.join(os.path.dirname(__file__), "Data", "earnings_iv")
_JSON = os.path.join(_DIR, "latest.json")
_CSV = os.path.join(_DIR, "latest.csv")
_PREM_JSON = os.path.join(_DIR, "premium_latest.json")
_PREM_CSV = os.path.join(_DIR, "premium_latest.csv")

DEFAULT_UNIVERSE = "both"   # S&P 500 + Nasdaq 100
DEFAULT_DAYS = 14
THROTTLE_SEC = 0.35         # gap between tickers — keeps yfinance happy


def _clean(obj):
    """Recursively make an analysis dict JSON-safe: Timestamps -> ISO strings,
    numpy scalars -> python, NaN/Inf -> None."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "isoformat"):           # datetime / pd.Timestamp
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item"):                # numpy scalar
        try:
            v = obj.item()
            return v if not (isinstance(v, float) and not math.isfinite(v)) else None
        except Exception:
            return str(obj)
    return obj


def _safe_write(path, csv_path, rows, details, meta, min_keep=5, max_age_h=72):
    """Persist the new scan — UNLESS it's degenerate (near-empty, i.e. a data
    outage) and a recent non-degenerate cache already exists, in which case keep
    the good one so a pre-market feed hiccup can't wipe the dashboard."""
    if len(rows) < min_keep and os.path.exists(path):
        try:
            with open(path) as f:
                old = json.load(f)
            om = old.get("meta", {})
            n_old = om.get("n_ideas", om.get("n_ok", 0))
            age_h = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(om["built_at"])).total_seconds() / 3600
            if n_old >= min_keep and age_h <= max_age_h:
                om.setdefault("attempts", []).append(
                    {"at": meta["built_at"], "n": len(rows),
                     "note": f"degenerate scan ({meta.get('n_err') or len(meta.get('skipped', []))} "
                             f"skipped — likely feed outage); kept the prior cache"})
                old["meta"] = om
                with open(path, "w") as f:
                    json.dump(old, f)
                print(f"  ↳ new scan had only {len(rows)} usable results — kept the prior "
                      f"cache ({n_old} rows, {age_h:.0f}h old)")
                return om
        except Exception:
            pass
    with open(path, "w") as f:
        json.dump({"rows": _clean(rows), "details": details, "meta": meta}, f)
    if rows:
        try:
            import pandas as pd
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        except Exception:
            pass
    return meta


def load_latest():
    """Return (rows_df_or_list, details, meta) from the last prefetch.
    `rows` comes back as a pandas DataFrame if pandas is importable, else a list
    of dicts. (None, None, None) if there is no cache yet."""
    try:
        with open(_JSON) as f:
            blob = json.load(f)
    except Exception:
        return None, None, None
    rows = blob.get("rows", [])
    try:
        import pandas as pd
        rows = pd.DataFrame(rows)
    except Exception:
        pass
    return rows, blob.get("details", {}), blob.get("meta", {})


def build_scan(universe: str = DEFAULT_UNIVERSE, days: int = DEFAULT_DAYS,
               force_earnings: bool = False, progress=None) -> dict:
    """Run the full scan and persist it. `progress(done, total, sym)` optional."""
    from earnings_cache import get_upcoming_earnings
    from earnings_iv_module import _analyze_impl, scan_row

    os.makedirs(_DIR, exist_ok=True)
    t0 = time.time()
    reporters = get_upcoming_earnings(days=days, universe=universe, force=force_earnings)
    total = len(reporters)
    print(f"[prefetch] {total} names reporting in next {days}d ({universe})")

    rows, details, errors = [], {}, []
    for i, rep in enumerate(reporters):
        sym = rep["symbol"]
        try:
            r = _analyze_impl(sym)
        except Exception as e:  # noqa: BLE001
            r = {"symbol": sym, "error": f"{type(e).__name__}: {e}"}
        if progress:
            try:
                progress(i + 1, total, sym)
            except Exception:
                pass
        if r.get("error"):
            errors.append(f"{sym} ({r['error']})")
            print(f"  [skip] {sym}: {r['error']}")
        else:
            details[sym] = _clean(r)
            rows.append(scan_row(rep, r))
        time.sleep(THROTTLE_SEC)

    # sort: actionable first, then by score
    def _key(row):
        act = 1 if ("SELL" in row["Signal"] or "BUY" in row["Signal"]) else 0
        return (act, row["Score"])
    rows.sort(key=_key, reverse=True)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe,
        "days": days,
        "n_reporters": total,
        "n_ok": len(rows),
        "n_err": len(errors),
        "errors": errors,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    meta = _safe_write(_JSON, _CSV, rows, details, meta, min_keep=1)
    print(f"[prefetch] done in {meta['elapsed_sec']}s — "
          f"{meta.get('n_ok', 0)} ok, {meta.get('n_err', 0)} skipped → {_JSON}")
    return meta


# --------------------------------------------------------------------------- #
# Premium (no-earnings) scan
# --------------------------------------------------------------------------- #
def load_premium():
    try:
        with open(_PREM_JSON) as f:
            blob = json.load(f)
    except Exception:
        return None, None, None
    rows = blob.get("rows", [])
    try:
        import pandas as pd
        rows = pd.DataFrame(rows)
    except Exception:
        pass
    return rows, blob.get("details", {}), blob.get("meta", {})


def build_premium_scan(tickers=None, dte_target: int = 35, progress=None) -> dict:
    from earnings_iv_module import _premium_impl, premium_row, CREDIBLE_TICKERS
    os.makedirs(_DIR, exist_ok=True)
    t0 = time.time()
    names = tickers or CREDIBLE_TICKERS
    total = len(names)
    print(f"[premium] scanning {total} credible names, ~{dte_target} DTE")

    rows, details, skipped = [], {}, []
    for i, sym in enumerate(names):
        try:
            r = _premium_impl(sym, dte_target=dte_target)
        except Exception as e:  # noqa: BLE001
            r = {"symbol": sym, "error": f"{type(e).__name__}: {e}"}
        if progress:
            try:
                progress(i + 1, total, sym)
            except Exception:
                pass
        if r.get("error") or r.get("skip"):
            skipped.append(f"{sym} ({r.get('error') or r.get('skip')})")
        else:
            details[sym] = _clean(r)
            rows.append(premium_row(r))
        time.sleep(THROTTLE_SEC)

    rows.sort(key=lambda x: x["Quality"], reverse=True)
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "dte_target": dte_target,
        "n_names": total,
        "n_ideas": len(rows),
        "n_good": sum(1 for x in rows if "GOOD" in x["Verdict"]),
        "skipped": skipped,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    meta = _safe_write(_PREM_JSON, _PREM_CSV, rows, details, meta, min_keep=5)
    print(f"[premium] done in {meta['elapsed_sec']}s — {meta.get('n_ideas', len(rows))} ideas, "
          f"{meta.get('n_good', 0)} GOOD, {len(skipped)} skipped → {_PREM_JSON}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["earnings", "premium", "both"])
    ap.add_argument("--universe", default=DEFAULT_UNIVERSE,
                    choices=["watchlist", "ndx", "sp500", "both"])
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    ap.add_argument("--force", action="store_true",
                    help="also refresh the earnings-date cache")
    a = ap.parse_args()
    try:
        if a.mode in ("earnings", "both"):
            build_scan(universe=a.universe, days=a.days, force_earnings=a.force)
        if a.mode in ("premium", "both"):
            build_premium_scan()
    except Exception:
        print("[prefetch] FATAL:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
