"""Serialization helpers for reproducible result files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_json(path: Path, data: dict) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (np.floating, float)):
            number = float(value)
            return number if np.isfinite(number) else None
        return value

    path.write_text(json.dumps(clean(data), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
