"""Item selection for the cynical-RLCD arms (A6 and its random control A6r).

The rule (decided 2026-10-06, see ``docs/decisions.md`` D5): score each candidate by its predictive
entropy normalised by ``log K`` (so a binary NOUL and a 10-level SCORE share one 0..1 scale), then
take the top ``N`` **within each answer type**, with each type's quota proportional to its share of
the candidate pool. A6r draws uniformly at random *inside the same per-type quotas*, so A6 vs A6r
tests mining and not a shift in the mix of answer types.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rlcd.scoring import ANSWER_TYPES


def normalised_entropy(rows: Sequence[Sequence[float]]) -> np.ndarray:
    """``H(p) / log K`` per distribution: 0 for one-hot, 1 for uniform. Ragged rows are allowed."""
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        p = np.asarray(r, dtype=np.float64)
        k = len(p)
        if k < 2:
            raise ValueError("a question needs at least 2 options")
        nz = p[p > 0]
        out[i] = -(nz * np.log(nz)).sum() / np.log(k)
    return out


def type_quotas(types: Sequence[str], n: int) -> dict[str, int]:
    """Split ``n`` picks over answer types in proportion to their counts (largest remainder)."""
    types_arr = np.asarray(types)
    counts = {t: int((types_arr == t).sum()) for t in ANSWER_TYPES if (types_arr == t).any()}
    total = sum(counts.values())
    if not 0 < n <= total:
        raise ValueError(f"cannot pick {n} items from a pool of {total}")
    exact = {t: n * c / total for t, c in counts.items()}
    quota = {t: int(np.floor(v)) for t, v in exact.items()}
    # hand out the leftover picks to the largest fractional parts (ties: fixed type order)
    for t in sorted(exact, key=lambda t: (-(exact[t] - quota[t]), ANSWER_TYPES.index(t))):
        if sum(quota.values()) == n:
            break
        quota[t] += 1
    return quota


def select_mined(rows: Sequence[Sequence[float]], types: Sequence[str], n: int) -> np.ndarray:
    """Indices of the ``n`` items to label: highest normalised entropy within each type's quota."""
    ent = normalised_entropy(rows)
    types_arr = np.asarray(types)
    picked: list[np.ndarray] = []
    for t, q in type_quotas(types, n).items():
        idx = np.flatnonzero(types_arr == t)
        order = np.argsort(-ent[idx], kind="stable")  # ties keep pool order
        picked.append(idx[order[:q]])
    return np.sort(np.concatenate(picked))


def select_random_matched(types: Sequence[str], n: int, rng: np.random.Generator) -> np.ndarray:
    """A6r: ``n`` random items inside the same per-type quotas as ``select_mined``."""
    types_arr = np.asarray(types)
    picked = [
        rng.choice(np.flatnonzero(types_arr == t), size=q, replace=False)
        for t, q in type_quotas(types, n).items()
    ]
    return np.sort(np.concatenate(picked))
