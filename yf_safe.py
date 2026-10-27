"""
Thread-safe yf.download for the long-running dashboard process.

yfinance's threaded download (via the `multitasking` package) is not safe in a server:
  - it starts one OS thread per ticker up front (the pool semaphore only limits how many
    run at once), so a 700-ticker scan briefly holds ~700 threads;
  - every download resets the module-global `yfinance.shared._DFS` and then spins on
    `while len(_DFS) < len(tickers)`, so two overlapping downloads (Home background scan +
    a page scan) can leave one spinning forever, holding its thread;
  - `multitasking.config["TASKS"]` keeps every Thread object ever started.
After ~9 days the dashboard hit macOS's 2048-thread cap and every page went blank with
"RuntimeError: can't start new thread". `install()` wraps yf.download so multi-ticker
downloads run one at a time, in chunks of CHUNK tickers, and prunes finished tasks.
"""

import threading

import pandas as pd

CHUNK = 50
_LOCK = threading.Lock()


def _ticker_list(tickers):
    if isinstance(tickers, (list, set, tuple)):
        items = list(tickers)
    else:
        items = str(tickers).replace(",", " ").split()
    seen, out = set(), []
    for t in items:
        u = str(t).upper()
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _prune_tasks():
    try:
        import multitasking
        multitasking.config["TASKS"] = [t for t in multitasking.config["TASKS"] if t.is_alive()]
    except Exception:
        pass


def _merge(parts, group_by):
    parts = [p for p in parts if isinstance(p, pd.DataFrame) and not p.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1, sort=True)
    out = out.loc[:, ~out.columns.duplicated()]
    if group_by != "ticker" and isinstance(out.columns, pd.MultiIndex):
        # keep yfinance's (Price, Ticker) layout: all Close columns together, etc.
        fields = list(dict.fromkeys(out.columns.get_level_values(0)))
        out = out[[c for f in fields for c in out.columns if c[0] == f]]
    return out


def make_safe_download(orig):
    def safe_download(tickers, *args, **kwargs):
        names = _ticker_list(tickers)
        if len(names) <= 1:
            return orig(tickers, *args, **kwargs)
        with _LOCK:
            try:
                if len(names) <= CHUNK:
                    return orig(tickers, *args, **kwargs)
                parts = [orig(names[i:i + CHUNK], *args, **kwargs)
                         for i in range(0, len(names), CHUNK)]
                return _merge(parts, kwargs.get("group_by", "column"))
            finally:
                _prune_tasks()

    safe_download.__wrapped__ = orig
    safe_download._yf_safe = True
    return safe_download


def install():
    """Patch yf.download in place (idempotent: the dashboard script re-runs per page load)."""
    import yfinance as yf
    if getattr(yf.download, "_yf_safe", False):
        return
    yf.download = make_safe_download(yf.download)
