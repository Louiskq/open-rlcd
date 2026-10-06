"""Exp 0b (exploratory): the varying-difficulty problem is well-posed; the diagnostics add up."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from toy import exp0 as X
from toy.exp0b import SCALE_RANGE

TINY = replace(X.ToyConfig(), steps=1500, n_eval=6000, n_cal=6000, seeds=X.TRAIN_SEEDS[:2])


def test_policy_features_can_represent_the_true_logits_exactly():
    """z*(x) = s_lo W x + (s_hi - s_lo) W (u x) is linear in [x, u x]: well-specified."""
    prob = X.hetero_problem(TINY, *SCALE_RANGE)
    lo, hi = SCALE_RANGE
    theta = np.concatenate([lo * prob.w, (hi - lo) * prob.w], axis=1)
    x = np.random.default_rng(0).normal(size=(500, prob.dim))
    np.testing.assert_allclose(prob.features(x) @ theta.T, prob.true_logits(x), atol=1e-10)


def test_difficulty_is_uniform_and_scales_the_true_logits():
    prob = X.hetero_problem(TINY, *SCALE_RANGE)
    x = np.random.default_rng(1).normal(size=(20_000, prob.dim))
    u = prob.difficulty(x)
    assert u.min() > 0 and u.max() < 1 and u.mean() == pytest.approx(0.5, abs=0.01)
    spread = prob.true_logits(x).std(axis=1)
    hard, easy = spread[u < 0.1].mean(), spread[u > 0.9].mean()
    assert easy > 3 * hard  # easy inputs have a far sharper truth


def test_plain_problem_is_unchanged_by_the_generalisation():
    w = X.ground_truth(TINY)
    prob = X.as_problem(w)
    x = np.random.default_rng(2).normal(size=(10, prob.dim))
    assert prob.features(x) is x and prob.difficulty(x) is None
    np.testing.assert_array_equal(prob.true_logits(x), x @ w.T)
    assert X.make_dataset(w, 50, 7).difficulty is None


def test_terciles_partition_every_item_exactly_once():
    prob = X.hetero_problem(TINY, *SCALE_RANGE)
    data = X.make_dataset(prob, 5000, 3)
    masks = X._group_masks(data)
    assert np.sum(masks, axis=0).tolist() == [1] * 5000
    assert all(m.sum() > 1400 for m in masks)  # roughly a third each


@pytest.fixture(scope="module")
def runs():
    prob = X.hetero_problem(TINY, *SCALE_RANGE)
    ev = X.make_dataset(prob, TINY.n_eval, X.EVAL_SEED)
    cal = X.make_dataset(prob, TINY.n_cal, X.CAL_SEED)
    return {n: X.run_arm(X.ARMS[n], TINY, False, prob, ev, cal) for n in ("rlvr", "rlcd_bandit")}


def gap(runs, arm, g):
    return np.mean(
        [r.by_difficulty["ts"][g]["mean_conf"] - r.by_difficulty["ts"][g]["acc"] for r in runs[arm]]
    )


def test_one_temperature_leaves_rlvr_overconfident_on_hard_and_underconfident_on_easy(runs):
    assert gap(runs, "rlvr", 0) > 0.03 > -0.03 > gap(runs, "rlvr", 2)
    for g in range(3):
        assert abs(gap(runs, "rlcd_bandit", g)) < 0.03


def test_per_tercile_temperatures_differ_for_rlvr_but_not_for_rlcd(runs):
    def temps(arm):
        return np.mean([r.by_difficulty["group_temperature"] for r in runs[arm]], axis=0)

    t_rlvr, t_rlcd = temps("rlvr"), temps("rlcd_bandit")
    assert t_rlvr[0] > 1.3 * t_rlvr[2]  # hard inputs need much more softening than easy ones
    assert t_rlcd.max() / t_rlcd.min() < 1.15  # a calibrated model needs about the same everywhere


def test_plain_runs_carry_no_difficulty_diagnostics():
    w = X.ground_truth(TINY)
    ev, cal = X.make_dataset(w, 800, X.EVAL_SEED), X.make_dataset(w, 800, X.CAL_SEED)
    run = X.train(X.ARMS["supervised"], replace(TINY, steps=50), w, 100, ev, cal)
    assert run.by_difficulty is None
