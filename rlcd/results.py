"""Reading and writing the small result JSONs that every table and figure is generated from.

Rule for the repo: a number that appears in the README or a figure is read from a file under
``results/``; it is never typed by hand. Files are rounded to 6 significant places so a re-run
on the same seeds gives a byte-identical file (and a readable diff when it does not).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _plain(obj: Any) -> Any:
    """Convert numpy scalars/arrays (and nested containers) to JSON-friendly Python values."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return _plain(obj.item())
    if isinstance(obj, float):
        return float(f"{obj:.6g}") if np.isfinite(obj) else None
    return obj


def write_json(path: str | Path, obj: Any) -> Path:
    """Write ``obj`` as indented JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(obj), indent=1, sort_keys=False) + "\n")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def mean_range(values: Any) -> dict[str, float]:
    """Seed mean and range, the summary the pre-registration asks for."""
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}
