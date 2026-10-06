"""A6 / A6r select BEFORE labels are revealed (docs/decisions.md D12), and the H5b metric.

Labels arrive with generation, so "label the selected items" is simulated by hiding them until after
selection. These tests fail if any selection function can see a label.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from rlcd import data as D
from rlcd import mining as M
from rlcd import scoring as S


def make_pool(seed: int, n: int = 120, permute_labels: bool = False) -> D.CandidatePool:
    rng = np.random.default_rng(seed)
    kinds = rng.choice(["noul", "choice", "score"], size=n, p=[0.35, 0.52, 0.13])
    k = {"noul": 2, "choice": 4, "score": 5}
    items = [
        D.TypedQuestion(
            f"c{i}",
            f"state {i}",
            f"question {i}?",
            t,
            tuple(f"o{j}" for j in range(k[t])),
            int(rng.integers(0, k[t])),
        )
        for i, t in enumerate(kinds)
    ]
    if permute_labels:  # same questions, different gold labels
        shift = {t: np.random.default_rng(99).integers(1, k[t]) for t in k}
        items = [
            dataclasses.replace(q, gold=(q.gold + int(shift[q.answer_type])) % len(q.options))
            for q in items
        ]
    return D.CandidatePool(items)


def fixed_scorer(questions) -> list[list[float]]:
    """A deterministic 'model': distributions depend on the question text only."""
    rows = []
    for q in questions:
        r = np.random.default_rng(int(q.qid[1:]))
        z = r.normal(size=len(q.options)) * r.uniform(0.2, 3.0)
        e = np.exp(z - z.max())
        rows.append(list(e / e.sum()))
    return rows


# ----------------------------------------------------------------- the structural guard


@pytest.mark.parametrize(
    "fn", [M.select_mined, M.select_random_matched, M.type_quotas, M.normalised_entropy]
)
def test_selection_functions_take_no_label_like_parameter(fn):
    M.assert_label_free(fn)  # raises LeakError if a gold/label/y/... parameter ever appears
    assert not (M.LABEL_NAMES & set(inspect.signature(fn).parameters))


def test_the_guard_fails_on_a_selection_function_that_takes_a_label():
    def leaky_select(rows, types, n, gold):  # would let selection use the answer
        return np.arange(n)

    with pytest.raises(D.LeakError, match="label-like"):
        M.assert_label_free(leaky_select)


def test_the_scorer_is_handed_questions_with_no_gold_field():
    assert "gold" not in {f.name for f in dataclasses.fields(D.UnlabelledQuestion)}
    pool = make_pool(0)
    seen = []

    def spy(questions):
        seen.extend(questions)
        return fixed_scorer(questions)

    M.mine_round(pool, spy, 20, "mined")
    assert len(seen) == len(pool) and all(not hasattr(q, "gold") for q in seen)
    assert all(isinstance(q, D.UnlabelledQuestion) for q in pool.questions)


# --------------------------------------------------------------------- behaviour


def test_selection_does_not_depend_on_the_labels():
    """Two pools identical except for their gold labels select exactly the same items."""
    a, b = make_pool(1), make_pool(1, permute_labels=True)
    assert [q.gold for q in a._labelled] != [q.gold for q in b._labelled]
    ia, _ = M.mine_round(a, fixed_scorer, 30, "mined")
    ib, _ = M.mine_round(b, fixed_scorer, 30, "mined")
    np.testing.assert_array_equal(ia, ib)
    ra, _ = M.mine_round(a, fixed_scorer, 30, "random", np.random.default_rng(5))
    rb, _ = M.mine_round(b, fixed_scorer, 30, "random", np.random.default_rng(5))
    np.testing.assert_array_equal(ra, rb)


def test_labels_are_revealed_only_after_selection_and_only_for_the_selected():
    pool = make_pool(2)
    calls: list[str] = []
    real_reveal = pool.reveal

    def spy_reveal(indices):
        calls.append("reveal")
        return real_reveal(indices)

    pool.reveal = spy_reveal  # type: ignore[method-assign]

    def spy_scorer(questions):
        calls.append("score")
        return fixed_scorer(questions)

    idx, labelled = M.mine_round(pool, spy_scorer, 25, "mined")
    assert calls == ["score", "reveal"]  # scored first, labelled after
    assert [q.qid for q in labelled] == [pool.questions[i].qid for i in idx]
    assert all(isinstance(q, D.TypedQuestion) for q in labelled) and len(labelled) == 25


def test_random_mode_scores_nothing_and_needs_an_rng():
    pool = make_pool(3)

    def forbidden(_questions):
        raise AssertionError("A6r must not score the candidates")

    idx, labelled = M.mine_round(pool, forbidden, 20, "random", np.random.default_rng(0))
    assert len(idx) == len(labelled) == 20
    with pytest.raises(ValueError):
        M.mine_round(pool, forbidden, 20, "random")
    with pytest.raises(ValueError):
        M.mine_round(pool, fixed_scorer, 20, "greedy")


def test_a6_and_a6r_take_the_same_number_of_each_answer_type():
    pool = make_pool(4, n=300)
    mined, _ = M.mine_round(pool, fixed_scorer, 60, "mined")
    rand, _ = M.mine_round(pool, fixed_scorer, 60, "random", np.random.default_rng(1))
    types = np.asarray(pool.types)
    for t in ("noul", "choice", "score"):
        assert (types[mined] == t).sum() == (types[rand] == t).sum()


# --------------------------------------------------------------- H5b: per-domain ECE


def test_grouped_ece_matches_ece_per_group_and_averages_them():
    rng = np.random.default_rng(6)
    p = rng.dirichlet(np.ones(4) * 0.4, size=1200)
    y = rng.integers(0, 4, size=1200)
    groups = rng.choice(["a", "b", "c", "d"], size=1200)
    out = S.grouped_ece(p, y, groups)
    for g in "abcd":
        m = groups == g
        assert out[g] == pytest.approx(S.top_label_ece(p[m], y[m], 15, "mass"))
    assert out["mean"] == pytest.approx(np.mean([out[g] for g in "abcd"]))
    with pytest.raises(ValueError):
        S.grouped_ece(p, y, groups[:-1])


def test_grouped_ece_exposes_opposite_miscalibration_that_pooled_ece_hides():
    """Same stated confidence (0.7) in two groups: group A is right 90% of the time (under-
    confident), group B 50% (overconfident). Pooled, accuracy is 0.7 and ECE is about zero."""
    rng = np.random.default_rng(7)
    n = 30_000
    p = np.tile([0.7, 0.3], (2 * n, 1))
    acc = np.concatenate([np.full(n, 0.9), np.full(n, 0.5)])
    y = np.where(rng.random(2 * n) < acc, 0, 1)  # label 0 is the predicted option
    groups = np.array(["a"] * n + ["b"] * n)
    assert S.top_label_ece(p, y, 15, "width") < 0.01  # pooled: looks calibrated
    by_group = S.grouped_ece(p, y, groups)
    assert by_group["a"] > 0.15 and by_group["b"] > 0.15 and by_group["mean"] > 0.15
