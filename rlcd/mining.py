"""Item selection for the cynical-RLCD arms (A6 and its random control A6r).

The rule (decided 2026-10-06, see ``docs/decisions.md`` D5): score each candidate by its predictive
entropy normalised by ``log K`` (so a binary NOUL and a 10-level SCORE share one 0..1 scale), then
take the top ``N`` **within each answer type**, with each type's quota proportional to its share of
the candidate pool. A6r draws uniformly at random *inside the same per-type quotas*, so A6 vs A6r
tests mining and not a shift in the mix of answer types.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence

import numpy as np

from rlcd.data import CandidatePool, LeakError, TypedQuestion, UnlabelledQuestion
from rlcd.scoring import ANSWER_TYPES

LABEL_NAMES = frozenset({"gold", "golds", "label", "labels", "y", "target", "targets", "answer"})


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


def assert_label_free(fn: Callable[..., object]) -> None:
    """Raise ``LeakError`` if ``fn`` has a parameter that could carry a label.

    Selection must happen before labelling, so the selection functions take distributions and answer
    types only. This guard is called on them in ``tests/test_label_hiding.py``: adding a ``gold``
    (or similar) parameter to one of them breaks the build.
    """
    bad = LABEL_NAMES & set(inspect.signature(fn).parameters)
    if bad:
        raise LeakError(f"{fn.__name__} takes label-like parameters {sorted(bad)}")


def mine_round(
    pool: CandidatePool,
    score_fn: Callable[[Sequence[UnlabelledQuestion]], Sequence[Sequence[float]]],
    n: int,
    mode: str,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[TypedQuestion]]:
    """One selection round: score the unlabelled candidates, pick ``n``, then reveal their labels.

    ``mode='mined'`` is A6 (``select_mined`` on the current model's distributions);
    ``mode='random'`` is A6r (``select_random_matched``, which needs ``rng``). ``score_fn`` sees
    unlabelled questions only. Returns the selected indices and the now-labelled questions.
    """
    if mode == "mined":
        idx = select_mined(score_fn(pool.questions), pool.types, n)
    elif mode == "random":
        if rng is None:
            raise ValueError("mode='random' needs an rng")
        idx = select_random_matched(pool.types, n, rng)
    else:
        raise ValueError("mode must be 'mined' or 'random'")
    return idx, pool.reveal(idx)
