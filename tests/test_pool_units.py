"""Record-level splits, unit building and micro-batch planning (no tokenizer or GPU needed)."""

from __future__ import annotations

import json
import random
import re

import numpy as np
import pytest

from rlcd import data as D
from rlcd import pool as P
from rlcd import units as U

# ------------------------------------------------------------------ a fake of decider.systemone


def fake_render_state(state):
    return state if isinstance(state, str) else json.dumps(state)


def fake_render_question(spec):
    """Same shapes as decider.systemone.render_question for the three types."""
    t, crit = spec["type"], spec.get("criteria")
    if t == "choice":
        names = list(crit)
        opts = [n if crit[n] in (None, "") else f"{n}: {crit[n]}" for n in names]
    elif t == "score":
        names = list(range(len(crit)))
        opts = [f"{i}: {c}" for i, c in enumerate(crit)]
    else:
        names, opts = [False, True], ["no", "yes"]
    return dict(question=spec["instructions"], options=opts, type=t, names=names)


def fake_isolated_rows(question, levels):
    strip = lambda s: re.sub(r"^\s*-?\d+\s*:\s*", "", s)  # noqa: E731
    return [
        (f"{question}\nProposed answer: {strip(lv)}\nDoes the proposed answer fit?", ["no", "yes"])
        for lv in levels
    ]


def fake_build_row(context, text, options, rng):
    perm = list(range(len(options)))
    rng.shuffle(perm)
    n_ids = len(context.split()) + len(text.split()) + 6 * len(options)
    return U.Row(
        np.arange(n_ids, dtype=np.int32), n_ids - 1, len(options), np.asarray(perm, np.int16)
    )


def make_record(i: int, n_q: int = 3) -> P.Record:
    qs = [
        {"type": "noul", "instructions": f"is it {i}?", "answer": bool(i % 2)},
        {
            "type": "choice",
            "instructions": "which?",
            "criteria": {"a": None, "b": "second", "c": None},
            "answer": "b",
        },
        {
            "type": "score",
            "instructions": "how bad?",
            "criteria": ["low", "mid", "high", "worst"],
            "answer": 2,
        },
    ][:n_q]
    return P.Record(
        f"r{i}", "text:fam", "sub", f"domain{i % 4}", f"state number {i} " * 5, tuple(qs)
    )


def make_pool(n: int = 4000) -> list[P.Record]:
    return [make_record(i) for i in range(n)]


# ------------------------------------------------------------------------------ records


def test_read_records_and_duplicate_ids(tmp_path):
    recs = make_pool(5)
    path = tmp_path / "a.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": r.id,
                    "family": r.family,
                    "sub": r.sub,
                    "domain": r.domain,
                    "state": r.state,
                    "questions": list(r.questions),
                }
            )
            for r in recs
        )
        + "\n"
    )
    got = P.read_records([str(path)])
    assert [r.id for r in got] == [r.id for r in recs] and got[0].questions == recs[0].questions
    with pytest.raises(ValueError, match="duplicate"):
        P.read_records([str(path), str(path)])


def test_record_splits_are_disjoint_complete_and_keep_every_question_of_a_record_together():
    recs = make_pool()
    splits = P.split_records(recs)
    P.assert_record_disjoint(splits)
    assert sum(len(v) for v in splits.values()) == len(recs)
    assert len(splits["cal"]) == pytest.approx(0.05 * len(recs), rel=0.2)
    assert len(splits["train"]) == pytest.approx(0.9 * len(recs), rel=0.02)
    a, b = P.split_record_halves(splits["train"])
    P.assert_record_disjoint({"A": a, "B": b, "cal": splits["cal"], "select": splits["select"]})
    assert len(a) + len(b) == len(splits["train"]) and len(a) == pytest.approx(len(b), rel=0.06)


def test_splits_do_not_depend_on_order_and_are_stable_when_records_are_added():
    recs = make_pool(2000)
    s1 = P.split_records(recs)
    s2 = P.split_records(list(reversed(recs)))
    assert {k: P.id_digest(v) for k, v in s1.items()} == {k: P.id_digest(v) for k, v in s2.items()}
    grown = P.split_records(recs + [make_record(10_000 + i) for i in range(500)])
    before = {r.id: k for k, v in s1.items() for r in v}
    after = {r.id: k for k, v in grown.items() for r in v}
    assert all(after[i] == k for i, k in before.items())


def test_manifest_counts_types_and_a_leak_between_splits_is_caught():
    recs = make_pool(300)
    m = P.manifest({"all": recs})
    assert m["all"]["questions"] == 900 and m["all"]["by_type"] == {
        "noul": 300,
        "choice": 300,
        "score": 300,
    }
    with pytest.raises(D.LeakError):
        P.assert_record_disjoint({"x": recs[:10], "y": recs[5:15]})


# -------------------------------------------------------------------------------- units


def build_all(recs, seed=0, hide=False):
    return U.build_units(
        recs,
        fake_render_state,
        fake_render_question,
        fake_build_row,
        fake_isolated_rows,
        seed,
        hide,
    )


def test_golds_are_original_indices_noul_is_false_true_and_score_levels_are_in_order():
    recs = [make_record(1), make_record(2)]
    units = build_all(recs)
    by = {u.qid: u for u in units}
    assert by["r1#0"].gold == 1 and by["r2#0"].gold == 0  # [False, True].index(answer)
    assert by["r1#1"].gold == 1  # "b" is the second choice
    assert (
        by["r1#2"].gold == 2 and by["r1#2"].n == 4 and len(by["r1#2"].rows) == 4
    )  # one row per level
    assert all(r.nopts == 2 for r in by["r1#2"].rows)  # each isolated row offers no / yes
    assert by["r1#0"].qtype == "noul" and len(by["r1#0"].rows) == 1 and by["r1#0"].n == 2


def test_option_shuffles_are_a_function_of_seed_and_question_id_only():
    recs = make_pool(40)
    a, b = build_all(recs, 0), build_all(list(reversed(recs)), 0)
    pa = {u.qid: [r.perm.tolist() for r in u.rows] for u in a}
    pb = {u.qid: [r.perm.tolist() for r in u.rows] for u in b}
    assert pa == pb  # same rendering whatever the order, so every arm sees the same prompts
    c = build_all(recs, 1)
    assert pa != {u.qid: [r.perm.tolist() for r in u.rows] for u in c}
    assert all(sorted(r.perm.tolist()) == list(range(r.nopts)) for u in a for r in u.rows)


def test_hidden_gold_and_reveal():
    units = build_all([make_record(3)], hide=True)
    assert all(u.gold is None for u in units)
    u = units[1].with_gold(2)
    assert u.gold == 2 and units[1].gold is None  # a copy; the hidden unit is untouched
    with pytest.raises(ValueError):
        units[1].with_gold(3)  # outside the 3 options


# ------------------------------------------------------------------------ micro-batches


def test_every_unit_is_planned_once_and_never_split_across_batches():
    units = build_all(make_pool(200))
    batches = U.plan_microbatches(units, 4096, random.Random(0))
    seen = [i for b in batches for i in b]
    assert sorted(seen) == list(range(len(units)))
    assert len(set(seen)) == len(seen)


def test_batches_respect_the_token_budget_except_for_a_single_oversized_unit():
    units = build_all(make_pool(200))
    budget = 3000
    for b in U.plan_microbatches(units, budget, random.Random(1)):
        cost = sum(len(units[i].rows) * U.padded_len(units[i].max_len) for i in b)
        # padded to the *bucket* of each member; members of a batch share a bucket, so the sum is exact
        assert cost <= budget or len(b) == 1


def test_a_unit_larger_than_the_budget_gets_its_own_batch():
    units = build_all([make_record(7)])
    big = max(units, key=lambda u: u.tokens)
    batches = U.plan_microbatches(units, 10, random.Random(0))
    assert len(batches) == len(units) and big is not None


def test_planning_is_deterministic_for_a_seed_and_differs_between_seeds():
    units = build_all(make_pool(300))
    a = U.plan_microbatches(units, 5000, random.Random(5))
    b = U.plan_microbatches(units, 5000, random.Random(5))
    c = U.plan_microbatches(units, 5000, random.Random(6))
    assert a == b and a != c


def test_type_mix():
    assert U.type_mix(build_all(make_pool(10))) == {"noul": 10, "choice": 10, "score": 10}
