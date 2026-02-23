from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


CACHE_ROOT = Path("artifacts/cache")


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()[:24]


def _path(namespace: str, payload: dict[str, Any], suffix: str) -> Path:
    key = _hash_payload(payload)
    directory = CACHE_ROOT / namespace
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.{suffix}"


def load_df_cache(namespace: str, payload: dict[str, Any]) -> pd.DataFrame | None:
    p = _path(namespace, payload, "pkl")
    if not p.exists():
        return None
    try:
        return pd.read_pickle(p)
    except Exception:
        return None


def save_df_cache(namespace: str, payload: dict[str, Any], df: pd.DataFrame) -> Path:
    p = _path(namespace, payload, "pkl")
    df.to_pickle(p)
    return p


def load_json_cache(namespace: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    p = _path(namespace, payload, "json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def save_json_cache(namespace: str, payload: dict[str, Any], data: dict[str, Any]) -> Path:
    p = _path(namespace, payload, "json")
    p.write_text(json.dumps(data))
    return p
