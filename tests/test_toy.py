"""Exp 0 invariants on a tiny configuration (a few seconds): the claims the README makes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rlcd.data import LeakError
from toy import exp0 as X

TINY = replace(X.ToyConfig(), steps=1200, n_eval=4000, n_cal=4000, seeds=X.TRAIN_SEEDS[:2])


@pytest.fixture(scope="module")
def runs():
    w = X.ground_truth(TINY)
    ev = X.make_dataset(w, TINY.n_eval, X.EVAL_SEED)
    cal = X.make_dataset(w, TINY.n_cal, X.CAL_SEED)
    return {name: X.run_arm(arm, TINY, False, w, ev, cal) for name, arm in X.ARMS.items()}


def final(runs, arm, metric, ts=False):
    return float(
        np.mean([getattr(r, "final_ts" if ts else "final_raw")[metric] for r in runs[arm]])
    )


def test_rlcd_full_through_the_rl_path_matches_supervised_to_machine_precision(runs):
    for a, b in zip(runs["supervised"], runs["rlcd_full"], strict=True):
        for k in a.curves:
            np.testing.assert_allclose(a.curves[k], b.curves[k], atol=1e-9)


def test_rlvr_is_overconfident_and_rlcd_is_not(runs):
    assert final(runs, "rlvr", "mean_conf") - final(runs, "rlvr", "acc") > 0.2
    assert final(runs, "rlvr", "ece") > 5 * final(runs, "rlcd_bandit", "ece")
    assert final(runs, "rlvr", "log_loss") > final(runs, "rlcd_bandit", "log_loss") + 0.2
    assert final(runs, "rlcd_bandit", "kl_to_truth") < 0.05


def test_accuracy_is_the_same_for_every_arm(runs):
    accs = [final(runs, a, "acc") for a in X.ARMS]
    assert max(accs) - min(accs) < 0.02


def test_temperature_scaling_is_fitted_off_the_evaluation_set_and_raises_t_for_overconfidence(runs):
    assert np.mean([r.temperature for r in runs["rlvr"]]) > 2.0
    assert final(runs, "rlvr", "ece", ts=True) < final(runs, "rlvr", "ece")


def test_training_is_deterministic_per_seed():
    w = X.ground_truth(TINY)
    ev = X.make_dataset(w, 1000, X.EVAL_SEED)
    cal = X.make_dataset(w, 1000, X.CAL_SEED)
    cfg = replace(TINY, steps=200)
    a = X.train(X.ARMS["rlcd_bandit"], cfg, w, 100, ev, cal)
    b = X.train(X.ARMS["rlcd_bandit"], cfg, w, 100, ev, cal)
    assert a.curves["ece"] == b.curves["ece"]


def test_all_arms_share_the_same_data_stream_for_a_seed():
    """Arms differ only in their sampled answers: at step 1 the supervised and RLCD-full runs saw
    identical data, so they agree, while an RL arm (different gradient) does not."""
    w = X.ground_truth(TINY)
    ev = X.make_dataset(w, 1000, X.EVAL_SEED)
    cal = X.make_dataset(w, 1000, X.CAL_SEED)
    cfg = replace(TINY, steps=1, n_checkpoints=2)
    sup = X.train(X.ARMS["supervised"], cfg, w, 100, ev, cal)
    rlvr = X.train(X.ARMS["rlvr"], cfg, w, 100, ev, cal)
    assert sup.curves["log_loss"][-1] != rlvr.curves["log_loss"][-1]
    assert sup.curves["log_loss"][0] == rlvr.curves["log_loss"][0]  # same start


def test_seed_sets_are_disjoint_and_overlap_is_rejected():
    assert X.SEEDS.train == X.TRAIN_SEEDS  # SeedSets is built at import and would have raised
    assert {X.CAL_SEED, X.EVAL_SEED}.isdisjoint(X.TRAIN_SEEDS)
    with pytest.raises(LeakError):
        replace(X.ToyConfig(), seeds=(100, X.EVAL_SEED))


def test_checkpoint_grid_is_geometric_and_ends_at_the_last_step():
    steps = X.checkpoint_steps(4000, 48)
    assert steps[0] == 0 and steps[-1] == 4000 and steps == sorted(set(steps))
    assert steps[2] - steps[1] <= 2 and steps[-1] - steps[-2] > 500


def test_gradient_noise_shows_unbiased_bandit_biased_clip_and_improper_rlvr():
    noise = X.gradient_noise(TINY, n_batches=3000)  # batch 64, as in the full run
    for arm in ("supervised", "rlcd_full", "rlcd_bandit"):
        assert noise[arm]["bias_norm"] < 3 * noise[arm]["bias_se_norm"]
    assert (
        noise["rlcd_bandit_clip10"]["bias_norm"] > 3 * noise["rlcd_bandit_clip10"]["bias_se_norm"]
    )
    assert noise["rlvr"]["bias_norm"] > 10 * noise["rlvr"]["bias_se_norm"]
    assert noise["rlcd_bandit"]["noise_trace"] > noise["rlcd_bandit_clip10"]["noise_trace"]


def test_oracle_forecaster_sets_the_noise_floor():
    w = X.ground_truth(TINY)
    data = X.make_dataset(w, 20_000, X.EVAL_SEED)
    ref = X.reference_metrics(data)
    assert ref["kl_to_truth"] == pytest.approx(0.0, abs=1e-12)
    assert ref["true_gap"] == pytest.approx(0.0, abs=1e-12)
    assert ref["ece"] < 0.02 and 0.45 < ref["bayes_acc"] < 0.55
