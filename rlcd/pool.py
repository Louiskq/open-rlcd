"""The teacher-written question pool as records, and its splits.

A *record* is one generated state with several typed questions about it. Splits are by record id, never by
question: two questions about the same state in different splits would leak the state between training and
evaluation (``rlcd.data`` makes the same point for items). The 90 / 5 / 5 train / calibration / selection split and
the halves of the training split used by A6 / A6r (``docs/decisions.md`` D-lean) are salted hashes of the record id,
so they are reproducible, independent of file order, and stable if records are added.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from rlcd.data import DEFAULT_POOL_SPLIT, HALVES_SALT, assign_split

ANSWER_TYPES = ("noul", "choice", "score")


@dataclass(frozen=True)
class Record:
    """One state and its typed questions, as written by the teacher pipeline (``questions`` are raw specs)."""

    id: str
    family: str
    sub: str
    domain: str
    state: Any
    questions: tuple[dict[str, Any], ...]

    def question_types(self) -> list[str]:
        return [q.get("type", "choice") for q in self.questions]


def read_records(paths: Iterable[str]) -> list[Record]:
    """Read JSONL files of records; ids must be unique across all files."""
    out: list[Record] = []
    seen: set[str] = set()
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                rid = str(r["id"])
                if rid in seen:
                    raise ValueError(f"duplicate record id {rid!r} in {path}")
                seen.add(rid)
                out.append(
                    Record(
                        rid,
                        r.get("family", ""),
                        r.get("sub", ""),
                        r.get("domain", ""),
                        r["state"],
                        tuple(r["questions"]),
                    )
                )
    return out


def split_records(
    records: Iterable[Record],
    fractions: dict[str, float] = DEFAULT_POOL_SPLIT,
    salt: str = "rlcd-v1",
) -> dict[str, list[Record]]:
    """Train / calibration / selection by hashed record id."""
    out: dict[str, list[Record]] = {name: [] for name in fractions}
    for r in records:
        out[assign_split(r.id, fractions, salt)].append(r)
    return out


def split_record_halves(
    records: Iterable[Record], salt: str = HALVES_SALT
) -> tuple[list[Record], list[Record]]:
    """Half A (what A6 / A6r start on) and half B (their held-back candidates) of the training split."""
    halves: dict[str, list[Record]] = {"A": [], "B": []}
    for r in records:
        halves[assign_split(r.id, {"A": 0.5, "B": 0.5}, salt)].append(r)
    return halves["A"], halves["B"]


def type_counts(records: Iterable[Record]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in records:
        c.update(r.question_types())
    return {t: c.get(t, 0) for t in ANSWER_TYPES}


def id_digest(records: Iterable[Record]) -> str:
    """A short digest of the set of record ids: lets a results file prove which split it used without listing ids."""
    h = hashlib.sha256()
    for rid in sorted(r.id for r in records):
        h.update(rid.encode())
    return h.hexdigest()[:16]


def manifest(splits: dict[str, Sequence[Record]]) -> dict[str, Any]:
    """Counts, answer-type mix, domains and an id digest per named split (what ``results/expA`` stores)."""
    out: dict[str, Any] = {}
    for name, recs in splits.items():
        out[name] = {
            "records": len(recs),
            "questions": sum(len(r.questions) for r in recs),
            "by_type": type_counts(recs),
            "domains": len({r.domain for r in recs}),
            "id_digest": id_digest(recs),
        }
    return out


def assert_record_disjoint(splits: dict[str, Sequence[Record]]) -> None:
    """No record id (hence no state) may appear in two splits."""
    from rlcd.data import LeakError

    names = list(splits)
    ids = {n: {r.id for r in splits[n]} for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = ids[a] & ids[b]
            if shared:
                raise LeakError(f"{a} and {b} share {len(shared)} record ids")
