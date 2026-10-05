"""Typed questions, deterministic splits, and the leakage checks that keep evaluation honest.

Splits are assigned by a salted hash of the question id, not by position or by an RNG stream. That
makes them reproducible, independent of file order, and *stable under growth*: the cynical-RLCD
loop (``mining``) adds items every round and none of the existing items change split.

Three kinds of separation are enforced here and asserted again in the tests:

* **items**: no id and no normalised-text duplicate may appear in two of train / cal / select / dev
  / JevBench (``assert_disjoint``);
* **seeds**: training, checkpoint-selection and final-evaluation seeds are disjoint (``SeedSets``);
* **use**: a checkpoint is chosen on the selection split or selection seeds, never on evaluation.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

ANSWER_TYPES = ("noul", "choice", "score")


@dataclass(frozen=True)
class TypedQuestion:
    """One question in the TypeSafe /v1/systemone shape: a state, a typed question, its options.

    ``noul`` has two options (false, true); ``choice`` 2..255 unordered options; ``score`` 2..10
    ordered levels. ``gold`` is the index of the correct option (or level).
    """

    qid: str
    state: str
    question: str
    answer_type: str
    options: tuple[str, ...]
    gold: int
    source: str = ""

    def __post_init__(self) -> None:
        if self.answer_type not in ANSWER_TYPES:
            raise ValueError(f"answer_type must be one of {ANSWER_TYPES}, got {self.answer_type!r}")
        if not 0 <= self.gold < len(self.options):
            raise ValueError(f"{self.qid}: gold {self.gold} outside {len(self.options)} options")


def content_key(q: TypedQuestion) -> str:
    """Hash of the question's normalised text (state, question, options), for duplicate detection.

    Case, punctuation-free whitespace runs and surrounding space are ignored, so a re-rendered or
    lightly re-formatted copy of a training question is still caught in an eval set.
    """
    text = " ".join([q.state, q.question, *q.options])
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _unit_hash(qid: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}|{qid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(qid: str, fractions: Mapping[str, float], salt: str = "rlcd-v1") -> str:
    """The split of ``qid``: the fraction interval its hash position in [0, 1) falls in."""
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    u, edge = _unit_hash(qid, salt), 0.0
    for name, frac in fractions.items():
        edge += frac
        if u < edge:
            return name
    return next(reversed(fractions))  # u can only reach 1.0 through float rounding


DEFAULT_POOL_SPLIT = {"train": 0.90, "cal": 0.05, "select": 0.05}


def make_splits(
    items: Iterable[TypedQuestion],
    fractions: Mapping[str, float] = DEFAULT_POOL_SPLIT,
    salt: str = "rlcd-v1",
) -> dict[str, list[TypedQuestion]]:
    """Split the teacher-written pool into train / calibration / selection by hashed id.

    ``cal`` is where temperatures are fitted, ``select`` is where learning rates and checkpoints are
    chosen. The dev set and JevBench are separate files that never pass through here.
    """
    out: dict[str, list[TypedQuestion]] = {name: [] for name in fractions}
    for q in items:
        out[assign_split(q.qid, fractions, salt)].append(q)
    return out


class LeakError(AssertionError):
    """Two splits that must be disjoint share an item."""


def assert_disjoint(splits: Mapping[str, Sequence[TypedQuestion]]) -> None:
    """Raise ``LeakError`` if any two named splits share a question id or a normalised-text copy.

    Duplicates *within* one split are allowed here (a pool can repeat itself); only cross-split
    overlap invalidates an evaluation.
    """
    ids = {name: {q.qid for q in qs} for name, qs in splits.items()}
    keys = {name: {content_key(q) for q in qs} for name, qs in splits.items()}
    names = list(splits)
    problems = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared_ids, shared_keys = ids[a] & ids[b], keys[a] & keys[b]
            if shared_ids or shared_keys:
                problems.append(
                    f"{a} / {b}: {len(shared_ids)} shared ids, {len(shared_keys)} shared texts"
                )
    if problems:
        raise LeakError("; ".join(problems))


@dataclass(frozen=True)
class SeedSets:
    """Seeds for training, checkpoint selection and final evaluation. They must not overlap.

    In Exp A "seeds" are training seeds (3 per arm) plus fixed split salts; in Exp B they also
    choose the Breakout game seeds, so the evaluation games are never games the agent trained on or
    was selected on.
    """

    train: tuple[int, ...]
    select: tuple[int, ...]
    evaluate: tuple[int, ...]

    def __post_init__(self) -> None:
        sets = {
            "train": set(self.train),
            "select": set(self.select),
            "evaluate": set(self.evaluate),
        }
        for a, b in (("train", "select"), ("train", "evaluate"), ("select", "evaluate")):
            shared = sets[a] & sets[b]
            if shared:
                raise LeakError(f"{a} and {b} seeds overlap: {sorted(shared)}")
