from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "audio": {
        "samplerate": 48000,
        "input_device": None,
        "record_seconds": 5.0,
    },
    "system_layout": "5.1",
    "analysis": {
        "criteria": "Medium (Δ<= 6dB) & (Best Corr >= 0.50)",
    },
    "thingsboard": {
        "enabled": False,
        "server": "thingsboard.cloud",
        "port": 1883,
        "token": "",
    },
    "paths": {
        "noise_wav": "data/noise.wav",
        "ref_wav": "data/ref.wav",
        "test_wav": "data/test.wav",
    },
}


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_CONFIG)
    if not isinstance(raw, dict):
        return dict(DEFAULT_CONFIG)
    return _deep_merge(DEFAULT_CONFIG, raw)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = _deep_merge(DEFAULT_CONFIG, cfg if isinstance(cfg, dict) else {})
    path.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
