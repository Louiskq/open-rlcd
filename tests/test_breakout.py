"""Exp B, CPU-only parts: belief labelling on scripted trajectories, advantages, belief metrics, margins and seeds."""

from __future__ import annotations

import random

import numpy as np
import pytest

from rlcd import breakout as BK


def _state(x=50, y=100, px=50, lives=5):
    return {"ball_x": x, "ball_y": y, "paddle_x": px, "lives": lives}


def _run(tracker, steps, ask_at=(0,), ctx="ctx"):
    """Drive a tracker over ``steps`` = [(state, score, done), ...]; a question is asked at the decisions in ask_at."""
    first = steps[0]
    tracker.start(first[0])
    for k, (state, score, _done) in enumerate(steps[:-1]):
        if k in ask_at:
            tracker.maybe_ask(k, state, score, ctx)
        nxt = steps[k + 1]
        tracker.observe(k + 1, nxt[0], nxt[1], nxt[2])
    return tracker.resolved


class _Fixed(random.Random):
    def __init__(self, kind):
        super().__init__(0)
        self.kind = kind

    def choice(self, seq):
        return self.kind if self.kind in seq else seq[0]


def test_brick30_true_only_if_a_brick_breaks_within_30_moves():
    traj = [(_state(y=100), 0, False) for _ in range(40)]
    traj[10] = (_state(y=100), 1, False)  # brick at decision 10
    traj[11:] = [(_state(y=100), 1, False)] * 29
    res = _run(BK.BeliefTracker(_Fixed("brick30"), every=1), traj, ask_at=(0,))
    assert [(r["kind"], r["gold"]) for r in res] == [("brick30", 1)]

    late = [(_state(y=100), 0, False) for _ in range(40)]
    late[35:] = [
        (_state(y=100), 1, False)
    ] * 5  # the first brick falls at step 35, outside 30 moves
    res = _run(BK.BeliefTracker(_Fixed("brick30"), every=1), late, ask_at=(0,))
    assert res[0]["gold"] == 0 and res[0]["t"] == 0


def test_life30_false_after_30_moves_true_if_a_life_is_lost():
    safe = [(_state(lives=5), 0, False) for _ in range(35)]
    res = _run(BK.BeliefTracker(_Fixed("life30"), every=1), safe, ask_at=(0,))
    assert res[0]["gold"] == 0
    lost = [(_state(lives=5), 0, False)] * 12 + [(_state(lives=4, y=0), 0, False)] * 23
    res = _run(BK.BeliefTracker(_Fixed("life30"), every=1), lost, ask_at=(0,))
    assert res[0]["gold"] == 1


def test_open_questions_at_truncation_are_dropped():
    traj = [(_state(), 0, False) for _ in range(10)]
    assert _run(BK.BeliefTracker(_Fixed("brick30"), every=1), traj, ask_at=(0,)) == []


def _return_traj(life_loss_after=None):
    """The ball descends to y=170 over the paddle, flips up (a return), then climbs."""
    ys = [100, 120, 140, 160, 170, 175, 170, 160, 140, 120]  # down then up: flips at index 5 -> 6
    traj = [(_state(y=y), 0, False) for y in ys]
    if life_loss_after is not None:
        traj = traj[: life_loss_after + 1] + [(_state(y=0, lives=4), 0, False)] * 3
    return traj


def test_return_is_true_on_a_paddle_flip_and_false_if_the_life_is_lost_first():
    res = _run(BK.BeliefTracker(_Fixed("return"), every=1), _return_traj(), ask_at=(0,))
    assert res[0]["kind"] == "return" and res[0]["gold"] == 1
    res = _run(
        BK.BeliefTracker(_Fixed("return"), every=1), _return_traj(life_loss_after=4), ask_at=(0,)
    )
    assert res[0]["gold"] == 0  # the ball is lost (y resets to 0, which is not a return)


def test_return_is_not_asked_when_no_ball_is_in_play():
    t = BK.BeliefTracker(_Fixed("return"), every=1)
    t.start(_state(y=0))
    kind = t.maybe_ask(0, _state(y=0), 0, "c")
    assert kind in ("brick30", "life30")  # 'return' needs a ball


def test_questions_are_asked_every_fourth_step():
    t = BK.BeliefTracker(random.Random(1), every=4)
    t.start(_state())
    asked = [t.maybe_ask(k, _state(), 0, "c") is not None for k in range(9)]
    assert asked == [True, False, False, False, True, False, False, False, True]


def test_returns_and_advantages():
    assert BK.returns([1.0, 0.0, 2.0], gamma=0.5) == pytest.approx(
        [1.0 + 0.5 * (0 + 0.5 * 2), 1.0, 2.0]
    )
    adv = BK.step_advantages([[1.0, 1.0], [3.0, 3.0]])
    assert adv[0][0] == pytest.approx(-adv[1][0])  # centred at each step
    flat = np.array([a for ep in adv for a in ep])
    assert flat.std() == pytest.approx(1.0, abs=1e-3)
    ragged = BK.step_advantages([[1.0, 2.0, 3.0], [1.0]])
    assert len(ragged[0]) == 3 and len(ragged[1]) == 1


def test_belief_metrics_and_skill_reference():
    gold = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    perfect = BK.belief_metrics(np.where(gold == 1, 0.999, 0.001), gold)
    assert perfect["acc"] == 1.0 and perfect["log_loss"] < 0.01
    base = BK.belief_metrics(np.full(8, gold.mean()), gold)
    assert base["log_loss"] == pytest.approx(BK.base_rate_log_loss(gold), abs=1e-6)
    assert base["base_rate"] == 0.5


def test_belief_temperature_recovers_a_known_scale():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(size=4000) * 1.5
    gold = (rng.random(4000) < BK.p_yes_from_logit_diff(true_logit)).astype(int)
    for scale in (3.0, 0.5):  # over- and under-confident reports
        T = BK.fit_belief_temperature(true_logit * scale, gold)
        assert abs(T - scale) / scale < 0.15


def test_select_iteration_keeps_the_earlier_on_ties_and_the_start_if_nothing_beats_it():
    assert BK.select_iteration([(0, 5.0), (5, 7.0), (10, 7.0), (15, 6.0)]) == 5
    assert BK.select_iteration([(0, 5.0), (5, 4.0), (10, 5.0)]) == 0


def test_delta_game_is_the_range_floored_at_five_percent_of_the_mean():
    d = BK.delta_game([20.0, 22.0, 21.0])
    assert d["delta_game"] == 2.0 and d["range"] == 2.0
    tiny = BK.delta_game([20.0, 20.1, 20.0])
    assert tiny["delta_game"] == pytest.approx(0.05 * np.mean([20.0, 20.1, 20.0]))


def test_paired_bootstrap_detects_a_consistent_difference():
    rng = np.random.default_rng(1)
    base = rng.random(500)
    out = BK.paired_bootstrap_diff(base - 0.1, base, B=500)
    assert out["mean_diff"] == pytest.approx(-0.1) and out["ci95"][1] < 0
    assert out["p_a_lower"] < 0.01
    same = BK.paired_bootstrap_diff(base, base, B=200)
    assert same["mean_diff"] == 0.0


def test_training_seeds_are_disjoint_from_selection_and_evaluation_seeds():
    for run_seed in (0, 1, 2):
        s = BK.seed_sets(run_seed)
        assert not set(s.train) & set(s.select) and not set(s.train) & set(s.evaluate)
    assert BK.train_env_seed(0, 0) != BK.train_env_seed(1, 0)
    assert set(BK.SELECT_SEEDS).isdisjoint(BK.EVAL_SEEDS)


def test_cluster_bootstrap_is_wider_than_the_item_bootstrap_when_items_are_correlated():
    rng = np.random.default_rng(0)
    episodes = np.repeat(np.arange(40), 25)  # 40 episodes of 25 questions
    episode_effect = np.repeat(
        rng.normal(size=40) * 0.5, 25
    )  # the difference is shared inside an episode
    a = 0.1 + episode_effect + rng.normal(size=1000) * 0.05
    b = np.zeros(1000)
    item = BK.paired_bootstrap_diff(a, b, B=500)
    clus = BK.cluster_bootstrap_diff(a, b, episodes, B=500)
    assert clus["mean_diff"] == pytest.approx(item["mean_diff"])
    assert (clus["ci95"][1] - clus["ci95"][0]) > 3 * (item["ci95"][1] - item["ci95"][0])
    assert clus["clusters"] == 40
