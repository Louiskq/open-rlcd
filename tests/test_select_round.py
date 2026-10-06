"""The Exp A runner's selection round (A6 / A6r): hidden labels, per-type quotas, no repeats, determinism."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rlcd import mining as M  # noqa: E402
from rlcd import train_labels as T  # noqa: E402
from rlcd.units import Unit  # noqa: E402

K = {"noul": 2, "choice": 4, "score": 5}


def make_units(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    kinds = rng.choice(["noul", "choice", "score"], size=n, p=[0.34, 0.54, 0.12])
    full = [
        Unit(f"r{i}#0", f"r{i}", t, "fam", "dom", K[t], [], gold=int(rng.integers(0, K[t])))
        for i, t in enumerate(kinds)
    ]
    labels = {u.qid: u.gold for u in full}
    hidden = [replace(u, gold=None) for u in full]
    return hidden, labels


def scorer(seen=None):
    def score_units(units):
        if seen is not None:
            seen.extend(units)
        out = []
        for u in units:
            r = np.random.default_rng(int(u.qid[1:].split("#")[0]))
            z = r.normal(size=u.n) * r.uniform(0.2, 3.0)
            e = np.exp(z - z.max())
            out.append(e / e.sum())
        return out

    return score_units


def test_a6_scorer_never_sees_a_label_and_added_units_get_theirs_afterwards():
    hidden, labels = make_units(600)
    seen: list[Unit] = []
    added, rest = T.select_round("a6", hidden, labels, 100, scorer(seen), seed=0, r=0)
    assert len(seen) == 600 and all(u.gold is None for u in seen)
    assert len(added) == 100 and all(u.gold == labels[u.qid] for u in added)
    assert all(u.gold is None for u in rest)  # unselected candidates stay hidden
    assert {u.qid for u in added}.isdisjoint({u.qid for u in rest})
    assert len(rest) == 500


def test_a6_takes_the_highest_entropy_items_within_each_type_quota():
    hidden, labels = make_units(900, seed=1)
    score = scorer()
    added, _ = T.select_round("a6", hidden, labels, 150, score, seed=0, r=0)
    ent = dict(zip((u.qid for u in hidden), M.normalised_entropy(score(hidden)), strict=True))
    quotas = M.type_quotas([u.qtype for u in hidden], 150)
    for t, q in quotas.items():
        chosen = {u.qid for u in added if u.qtype == t}
        others = [ent[u.qid] for u in hidden if u.qtype == t and u.qid not in chosen]
        assert len(chosen) == q and min(ent[c] for c in chosen) >= max(others) - 1e-12


def test_a6r_never_scores_and_matches_a6_per_type_counts():
    hidden, labels = make_units(900, seed=2)

    def forbidden(_units):
        raise AssertionError("A6r must not score the candidates")

    r_added, _ = T.select_round("a6r", hidden, labels, 150, forbidden, seed=3, r=1)
    a_added, _ = T.select_round("a6", hidden, labels, 150, scorer(), seed=3, r=1)
    for t in K:
        assert sum(u.qtype == t for u in r_added) == sum(u.qtype == t for u in a_added)
    assert all(u.gold == labels[u.qid] for u in r_added)


def test_a6r_is_deterministic_per_seed_and_round_and_differs_between_them():
    hidden, labels = make_units(500, seed=4)

    def pick(seed, r):
        added, _ = T.select_round("a6r", hidden, labels, 80, None, seed, r)
        return sorted(u.qid for u in added)

    assert pick(1, 0) == pick(1, 0)
    assert pick(1, 0) != pick(1, 1) and pick(1, 0) != pick(2, 0)


@pytest.mark.parametrize("arm", ["a6", "a6r"])
def test_four_rounds_over_a_shrinking_pool_select_each_item_at_most_once(arm):
    hidden, labels = make_units(1000, seed=5)
    remaining, taken = hidden, []
    for r in range(4):
        added, remaining = T.select_round(arm, remaining, labels, 150, scorer(), seed=0, r=r)
        taken += [u.qid for u in added]
    assert len(taken) == 600 == len(set(taken)) and len(remaining) == 400
    assert not set(taken) & {u.qid for u in remaining}


def test_an_unknown_arm_is_rejected_and_n_is_capped_by_the_pool():
    hidden, labels = make_units(50)
    with pytest.raises(ValueError):
        T.select_round("a1", hidden, labels, 10, scorer(), 0, 0)
    added, rest = T.select_round("a6", hidden, labels, 500, scorer(), 0, 0)
    assert len(added) == 50 and rest == []
