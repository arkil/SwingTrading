from __future__ import annotations

from datetime import timedelta
import random
import time

import pandas as pd
import requests

from daytrade_backtester.data.cache import load_df_cache, load_json_cache, save_df_cache, save_json_cache


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _parse_interval(interval: str) -> tuple[int, str]:
    interval = interval.strip().lower()
    if interval.endswith("m"):
        return int(interval[:-1]), "minute"
    if interval.endswith("h"):
        return int(interval[:-1]), "hour"
    if interval.endswith("d"):
        return int(interval[:-1]), "day"
    return 5, "minute"


def _sleep_backoff(attempt: int, base_sec: float) -> None:
    jitter = random.uniform(0.0, 0.25)
    delay = (base_sec * (2**attempt)) + jitter
    time.sleep(delay)


def _get_json(
    url: str,
    params: dict,
    timeout_sec: float,
    max_retries: int,
    backoff_sec: float,
) -> tuple[dict | None, str]:
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout_sec)
            if r.status_code == 200:
                return r.json(), "ok"

            status = f"http_{r.status_code}"
            if r.status_code in RETRYABLE_STATUS and attempt < max_retries:
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(float(retry_after))
                    except Exception:
                        _sleep_backoff(attempt, backoff_sec)
                else:
                    _sleep_backoff(attempt, backoff_sec)
                continue
            return None, status
        except requests.RequestException:
            if attempt < max_retries:
                _sleep_backoff(attempt, backoff_sec)
                continue
            return None, "request_exception"

    return None, "request_exception"


def _fetch_reference_contracts(
    symbol: str,
    contract_type: str,
    entry_time: pd.Timestamp,
    api_key: str,
    base_url: str,
    timeout_sec: float,
    max_retries: int,
    backoff_sec: float,
    max_pages: int,
) -> tuple[list[dict], str]:
    cache_payload = {
        "symbol": symbol,
        "contract_type": contract_type,
        "as_of": entry_time.strftime("%Y-%m-%d"),
        "expiration_lte": (entry_time + pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
        "base_url": base_url,
        "max_pages": max_pages,
        "source": "polygon_reference_contracts",
    }
    cached = load_json_cache("options_polygon_reference", cache_payload)
    if cached is not None:
        return list(cached.get("results", [])), str(cached.get("status", "ok_cache"))

    params = {
        "underlying_ticker": symbol,
        "contract_type": contract_type,
        "as_of": entry_time.strftime("%Y-%m-%d"),
        "expiration_date.gte": entry_time.strftime("%Y-%m-%d"),
        "expiration_date.lte": (entry_time + pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
        "limit": 1000,
        "sort": "expiration_date",
        "order": "asc",
        "apiKey": api_key,
    }
    url = f"{base_url}/v3/reference/options/contracts"

    all_results: list[dict] = []
    page = 0
    last_status = "ok"

    while page < max_pages and url:
        payload, status = _get_json(url, params, timeout_sec, max_retries, backoff_sec)
        if payload is None:
            return all_results, status

        all_results.extend(payload.get("results", []))
        next_url = payload.get("next_url")
        if not next_url:
            break

        url = next_url
        params = {"apiKey": api_key}
        page += 1
        last_status = status

    if not all_results:
        return [], "reference_empty"

    save_json_cache(
        "options_polygon_reference",
        cache_payload,
        {
            "status": last_status,
            "results": all_results,
        },
    )

    return all_results, last_status


def _pick_contract(results: list[dict], side: str, spot: float, dte_target_days: int, entry_date: pd.Timestamp, otm_steps: int) -> dict | None:
    if not results:
        return None

    target_exp = (entry_date + pd.Timedelta(days=dte_target_days)).date()

    enriched = []
    for row in results:
        exp = row.get("expiration_date")
        strike = row.get("strike_price")
        if exp is None or strike is None:
            continue
        try:
            exp_date = pd.Timestamp(exp).date()
            strike_f = float(strike)
        except Exception:
            continue

        if side == "long" and strike_f < spot:
            continue
        if side == "short" and strike_f > spot:
            continue

        dte_gap = abs((exp_date - target_exp).days)
        moneyness_gap = abs(strike_f - spot)
        enriched.append((dte_gap, exp_date, moneyness_gap, strike_f, row))

    if not enriched:
        return None

    enriched.sort(key=lambda x: (x[0], x[1], x[2]))
    idx = min(max(otm_steps - 1, 0), len(enriched) - 1)
    return enriched[idx][4]


def _price_near_time(df: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if df.empty:
        return None

    right = df.loc[df.index >= ts]
    if not right.empty:
        return float(right.iloc[0]["close"])

    left = df.loc[df.index <= ts]
    if not left.empty:
        return float(left.iloc[-1]["close"])

    return None


def _fetch_aggs_bars(
    ticker: str,
    interval: str,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    api_key: str,
    base_url: str,
    timeout_sec: float,
    max_retries: int,
    backoff_sec: float,
) -> tuple[pd.DataFrame, str]:
    payload = {
        "ticker": ticker,
        "interval": interval,
        "start": entry_time.strftime("%Y-%m-%d"),
        "end": (exit_time + timedelta(days=1)).strftime("%Y-%m-%d"),
        "base_url": base_url,
        "source": "polygon_aggs_bars",
    }
    cached = load_df_cache("options_polygon_bars", payload)
    if cached is not None and not cached.empty:
        return cached, "ok_cache"

    mult, span = _parse_interval(interval)
    aggs_url = f"{base_url}/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{entry_time.strftime('%Y-%m-%d')}/{(exit_time + timedelta(days=1)).strftime('%Y-%m-%d')}"
    agg_payload, agg_status = _get_json(
        aggs_url,
        {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": api_key,
        },
        timeout_sec,
        max_retries,
        backoff_sec,
    )

    if not agg_payload or "results" not in agg_payload:
        return pd.DataFrame(), agg_status

    results = agg_payload.get("results", [])
    if not results:
        return pd.DataFrame(), "empty"

    bars = pd.DataFrame(results)
    bars["timestamp"] = pd.to_datetime(bars["t"], unit="ms", utc=True).dt.tz_convert(entry_time.tz)
    bars = bars.set_index("timestamp")
    bars = bars.rename(columns={"c": "close"})
    save_df_cache("options_polygon_bars", payload, bars)
    return bars, "ok"


def lookup_option_entry_exit_polygon(
    symbol: str,
    side: str,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    spot_entry: float,
    interval: str,
    dte_target_days: int,
    otm_steps: int,
    api_key: str,
    base_url: str = "https://api.massive.com",
    timeout_sec: float = 20.0,
    max_retries: int = 4,
    backoff_sec: float = 0.8,
    max_pages: int = 3,
) -> tuple[str | None, float | None, float | None, str]:
    if not api_key:
        return None, None, None, "missing_api_key"

    contract_type = "call" if side == "long" else "put"
    ref_results, ref_status = _fetch_reference_contracts(
        symbol=symbol,
        contract_type=contract_type,
        entry_time=entry_time,
        api_key=api_key,
        base_url=base_url,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
        max_pages=max_pages,
    )
    if not ref_results:
        return None, None, None, f"reference_api_error:{ref_status}"

    contract = _pick_contract(
        ref_results,
        side=side,
        spot=spot_entry,
        dte_target_days=dte_target_days,
        entry_date=entry_time,
        otm_steps=otm_steps,
    )
    if not contract:
        return None, None, None, "contract_not_found"

    ticker = contract.get("ticker")
    if not ticker:
        return None, None, None, "missing_contract_ticker"

    bars, bars_status = _fetch_aggs_bars(
        ticker=ticker,
        interval=interval,
        entry_time=entry_time,
        exit_time=exit_time,
        api_key=api_key,
        base_url=base_url,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )

    if bars.empty:
        return ticker, None, None, f"no_price_bars:{bars_status}"

    opt_entry = _price_near_time(bars, entry_time)
    opt_exit = _price_near_time(bars, exit_time)
    if opt_entry is None or opt_exit is None:
        return ticker, opt_entry, opt_exit, "missing_entry_or_exit_price"

    return ticker, opt_entry, opt_exit, "ok"
