"""Questions as scored *units*: the forward-pass rows that give one answer distribution.

* NOUL / CHOICE: one row (the context and one question); the answer distribution is the softmax over its options.
* SCORE: one row **per level**, each asking "does the proposed answer fit?" with options ``no`` / ``yes``
  (decider's isolated levels, ``docs/decisions.md`` D3). The distribution over levels is the normalised yes
  probabilities, so all of a question's rows must be processed together: ``plan_microbatches`` never splits a unit.

Options are shuffled once per row from a seed derived from the question id (the same rendering for every arm and
every training seed), and each row stores the permutation, so logits can always be mapped back to the original
option order (levels in order, NOUL as ``[false, true]``).

Everything here is torch-free. The tokenizer-dependent part is the ``build_row`` callable (the cluster passes a
wrapper around ``decider.prompt.build``; tests pass a stub).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from rlcd.pool import Record

PAD_BUCKET = 64


@dataclass
class Row:
    """One forward pass: token ids, the position of the answer slot, and how options were shuffled."""

    ids: np.ndarray  # int32 token ids
    slot: int  # position whose hidden state is read out
    nopts: int  # number of options shown
    perm: np.ndarray  # int16; perm[j] = original option index shown at position j


@dataclass
class Unit:
    """One question and the rows that score it. ``gold`` is the original option / level index, or None while hidden."""

    qid: str
    rec_id: str
    qtype: str
    family: str
    domain: str
    n: int  # original options (NOUL: 2, CHOICE: n options) or levels (SCORE)
    rows: list[Row]
    gold: int | None = None

    def with_gold(self, gold: int) -> Unit:
        if not 0 <= gold < self.n:
            raise ValueError(f"{self.qid}: gold {gold} outside {self.n} options")
        return replace(self, gold=int(gold))

    @property
    def max_len(self) -> int:
        return max(len(r.ids) for r in self.rows)

    @property
    def tokens(self) -> int:
        return sum(len(r.ids) for r in self.rows)


@dataclass(frozen=True)
class RenderedQuestion:
    qtype: str
    text: str
    options: tuple[str, ...]
    gold: int


BuildRow = Callable[[str, str, Sequence[str], random.Random], Row]
IsolatedRows = Callable[[str, Sequence[str]], list[tuple[str, list[str]]]]


def render_record(
    rec: Record, render_state: Callable[[Any], str], render_question: Callable[[dict], dict]
) -> tuple[str, list[RenderedQuestion]]:
    """The context string and rendered questions of a record, via decider's ``systemone`` functions."""
    context = render_state(rec.state)
    out = []
    for spec in rec.questions:
        r = render_question(spec)
        qtype = r["type"]
        gold = int(spec["answer"]) if qtype == "score" else r["names"].index(spec["answer"])
        out.append(RenderedQuestion(qtype, r["question"], tuple(r["options"]), gold))
    return context, out


def build_unit(
    rec: Record,
    k: int,
    rq: RenderedQuestion,
    context: str,
    build_row: BuildRow,
    isolated_rows: IsolatedRows,
    seed: int,
    hide_gold: bool = False,
) -> Unit:
    """Build the rows of question ``k`` of ``rec``. The shuffle seed depends only on ``seed`` and the question id."""
    qid = f"{rec.id}#{k}"
    rng = random.Random(f"{seed}:{qid}")
    if rq.qtype == "score":
        rows = [
            build_row(context, text, opts, rng) for text, opts in isolated_rows(rq.text, rq.options)
        ]
        n = len(rq.options)
    else:
        rows = [build_row(context, rq.text, rq.options, rng)]
        n = len(rq.options)
    return Unit(
        qid, rec.id, rq.qtype, rec.family, rec.domain, n, rows, None if hide_gold else rq.gold
    )


def build_units(
    records: Sequence[Record],
    render_state: Callable[[Any], str],
    render_question: Callable[[dict], dict],
    build_row: BuildRow,
    isolated_rows: IsolatedRows,
    seed: int = 0,
    hide_gold: bool = False,
) -> list[Unit]:
    """Units for every question of every record, in record order."""
    units: list[Unit] = []
    for rec in records:
        context, rqs = render_record(rec, render_state, render_question)
        for k, rq in enumerate(rqs):
            units.append(build_unit(rec, k, rq, context, build_row, isolated_rows, seed, hide_gold))
    return units


def padded_len(n: int, bucket: int = PAD_BUCKET) -> int:
    return ((n + bucket - 1) // bucket) * bucket


def plan_microbatches(
    units: Sequence[Unit], max_tokens: int, rng: random.Random, bucket: int = PAD_BUCKET
) -> list[list[int]]:
    """Group units into micro-batches of at most ``max_tokens`` padded tokens (rows x padded length).

    Units are bucketed by the padded length of their longest row so there are few distinct shapes, and a unit
    is never split across batches (its rows must be normalised together). A unit larger than ``max_tokens``
    on its own gets a batch of its own. Order is shuffled with ``rng`` (bucket members and the batch list).
    """
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, u in enumerate(units):
        buckets[padded_len(u.max_len, bucket)].append(i)
    batches: list[list[int]] = []
    for pad, members in buckets.items():
        rng.shuffle(members)
        cur: list[int] = []
        used = 0
        for i in members:
            cost = len(units[i].rows) * pad
            if cur and used + cost > max_tokens:
                batches.append(cur)
                cur, used = [], 0
            cur.append(i)
            used += cost
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def type_mix(units: Sequence[Unit]) -> dict[str, int]:
    out = {"noul": 0, "choice": 0, "score": 0}
    for u in units:
        out[u.qtype] += 1
    return out
