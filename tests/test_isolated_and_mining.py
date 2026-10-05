"""SCORE isolated-level readout and the A6 / A6r selection rule."""

from __future__ import annotations

import numpy as np
import pytest

from rlcd import isolated as I
from rlcd import mining as M
from rlcd import scoring as S

# ------------------------------------------------------------------ isolated-level readout


def decider_combine_isolated(p_yes):
    """The formula of decider.systemone.combine_isolated, copied for comparison."""
    tot = sum(p_yes) or 1e-9
    return [x / tot for x in p_yes], tot


def test_combine_matches_deciders_combine_isolated():
    rng = np.random.default_rng(0)
    y = rng.random((20, 5)) * 0.9 + 0.05
    p, mass = I.combine(y)
    for i in range(len(y)):
        ref_p, ref_mass = decider_combine_isolated(list(y[i]))
        np.testing.assert_allclose(p[i], ref_p)
        assert mass[i] == pytest.approx(ref_mass)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_mass_deviation_hand_computed():
    # sums 0.6 + 0.3 = 0.9 and 0.9 + 0.8 = 1.7 -> |.9-1| = .1, |1.7-1| = .7 -> mean .4
    assert I.mass_deviation(np.array([[0.6, 0.3], [0.9, 0.8]])) == pytest.approx(0.4)
    assert I.mass_deviation(np.array([[0.7, 0.3]])) == pytest.approx(0.0, abs=1e-12)


def test_yes_probability_is_sigmoid_and_clipped():
    np.testing.assert_allclose(I.yes_probability(np.array([0.0])), [0.5])
    y = I.yes_probability(np.array([-1e3, 1e3]))
    assert 0 < y[0] < 1e-9 and 1 - 1e-9 < y[1] < 1


def test_all_zero_mass_is_rejected():
    with pytest.raises(ValueError):
        I.combine(np.zeros((1, 3)))


@pytest.mark.parametrize("rule,normalise", [("log", True), ("rps", True), ("rps", False)])
def test_score_gradient_through_normalisation_matches_finite_differences(rule, normalise):
    rng = np.random.default_rng(1)
    l = rng.normal(size=(7, 5)) * 1.5
    gold = rng.integers(0, 5, size=7)

    def score(ll):
        p, _ = I.combine(I.yes_probability(ll))
        if rule == "log":
            return S.log_score(p, gold)
        return -S.rps_per_item(p, gold, normalise=normalise)

    ana = I.score_grad_level_logits(l, gold, rule, normalise)
    eps = 1e-6
    for j in range(5):
        d = np.zeros_like(l)
        d[:, j] = eps
        np.testing.assert_allclose(ana[:, j], (score(l + d) - score(l - d)) / (2 * eps), atol=1e-6)


def test_gradient_raises_the_gold_level_and_lowers_the_others_on_average():
    l = np.zeros((1, 4))  # all levels 50% yes
    g = I.score_grad_level_logits(l, np.array([2]), "log")[0]
    assert g[2] > 0 and np.all(np.delete(g, 2) < 0)


def test_rps_prefers_probability_near_the_gold_level():
    l = np.zeros((1, 5))
    g = I.score_grad_level_logits(l, np.array([4]), "rps")[0]
    assert g[3] > g[0]  # the level next to the gold is pushed up more than a distant one


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        I.score_grad_level_logits(np.zeros((1, 3)), np.array([0]), "brier")


# ------------------------------------------------------------------------ A6 / A6r selection


def test_normalised_entropy_bounds_and_ragged_rows():
    h = M.normalised_entropy([[1.0, 0.0], [0.5, 0.5], [0.25] * 4, [1.0, 0, 0, 0, 0]])
    np.testing.assert_allclose(h, [0.0, 1.0, 1.0, 0.0])
    # a uniform 10-level SCORE and a uniform binary NOUL sit at the same normalised value
    assert M.normalised_entropy([[0.1] * 10])[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        M.normalised_entropy([[1.0]])


def test_quotas_sum_to_n_and_follow_the_pool_mix():
    types = ["noul"] * 500 + ["choice"] * 300 + ["score"] * 200
    q = M.type_quotas(types, 100)
    assert sum(q.values()) == 100 and q == {"noul": 50, "choice": 30, "score": 20}
    odd = M.type_quotas(["noul"] * 7 + ["choice"] * 5 + ["score"] * 3, 4)
    assert sum(odd.values()) == 4 and all(odd[t] >= 0 for t in odd)
    with pytest.raises(ValueError):
        M.type_quotas(types, 1001)


def test_mined_items_are_the_most_uncertain_within_each_type():
    rng = np.random.default_rng(2)
    types = np.array(["noul"] * 40 + ["choice"] * 40 + ["score"] * 20)
    rows = []
    for t in types:
        k = {"noul": 2, "choice": 4, "score": 5}[t]
        z = rng.normal(size=k) * rng.uniform(0.2, 3.0)
        e = np.exp(z - z.max())
        rows.append(e / e.sum())
    picked = M.select_mined(rows, types, 25)
    ent = M.normalised_entropy(rows)
    quota = M.type_quotas(types, 25)
    for t, q in quota.items():
        in_type = np.flatnonzero(types == t)
        chosen = np.intersect1d(picked, in_type)
        assert len(chosen) == q
        rest = np.setdiff1d(in_type, chosen)
        assert ent[chosen].min() >= ent[rest].max() - 1e-12
    assert len(set(picked)) == len(picked) == 25


def test_random_control_has_exactly_the_same_per_type_counts():
    types = np.array(["noul"] * 120 + ["choice"] * 60 + ["score"] * 20)
    rows = [[0.5, 0.5]] * 200  # content is irrelevant to the quotas
    mined = M.select_mined(rows, types, 40)
    rand = M.select_random_matched(types, 40, np.random.default_rng(3))
    for t in ("noul", "choice", "score"):
        assert (types[mined] == t).sum() == (types[rand] == t).sum()
    assert not np.array_equal(M.select_random_matched(types, 40, np.random.default_rng(4)), rand)
    np.testing.assert_array_equal(
        M.select_random_matched(types, 40, np.random.default_rng(3)), rand
    )  # reproducible per seed
