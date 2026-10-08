"""Exp B offline analysis on a synthetic run directory: delta_game, belief tables and the H7 decision rule."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rlcd import breakout_analysis as BA

N = 600


def _beliefs(rng, quality: float, scale: float = 1.0):
    """Answers whose logit has signal ``quality`` about the gold label (0 = noise), times ``scale``."""
    gold = (rng.random(N) < 0.4).astype(np.int8)
    d = (quality * (2 * gold - 1) + rng.normal(size=N)) * scale
    return dict(
        logit_diff=d,
        gold=gold,
        kind=np.array(["brick30", "life30", "return"])[rng.integers(0, 3, N)],
        seed=np.zeros(N, dtype=np.int32),
        t=np.zeros(N, dtype=np.int32),
    )


def _write_run(
    root, name, scores, selected_iter=5, sel_scores=(5.0, 9.0, 8.0), onpolicy=None, common=None
):
    run = root / name
    (run / "eval_best").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"selected_iter": selected_iter}))
    (run / "sel_evals.json").write_text(
        json.dumps(
            {
                "evals": list(zip([0, 5, 10], sel_scores, strict=True)),
                "selected_iter": selected_iter,
            }
        )
    )
    (run / "eval_best" / "games.json").write_text(
        json.dumps({"greedy_scores": list(scores), "sampled_scores": list(scores)})
    )
    if onpolicy is not None:
        np.savez(run / "eval_best" / "beliefs_onpolicy_eval.npz", **onpolicy)
        np.savez(run / "eval_best" / "beliefs_onpolicy_select.npz", **onpolicy)
    if common is not None:
        np.savez(run / "eval_best" / "beliefs_common.npz", **common)
    return run


def _layout(tmp_path, b3_score=20.0, b3_quality=1.2, b4_quality=0.4, seed=0):
    rng = np.random.default_rng(seed)
    gold_ref = _beliefs(rng, 0.0)
    common_gold = gold_ref["gold"]

    def with_gold(b):
        b = dict(b)
        b["gold"] = common_gold
        b["logit_diff"] = (b3_q_signal * (2 * common_gold - 1) + rng.normal(size=N)) * 1.0
        return b

    for s in BA.SEEDS:
        sel_scores = (5.0, 9.0 + 0.5 * s, 8.0)  # selected iteration 5 -> 9, 9.5, 10
        b1_common = _beliefs(rng, 0.2)
        _write_run(
            tmp_path,
            f"b1_s{s}",
            [10.0] * 10,
            5,
            sel_scores,
            onpolicy=_beliefs(rng, 0.2),
            common=b1_common,
        )
        b3_q_signal = b3_quality
        _write_run(
            tmp_path,
            f"b3_s{s}",
            [b3_score] * 10,
            5,
            sel_scores,
            onpolicy=_beliefs(rng, b3_quality),
            common=with_gold(_beliefs(rng, 0)),
        )
        b4 = tmp_path / f"b4_s{s}" / "eval"
        b4.mkdir(parents=True)
        b3_q_signal = b4_quality
        np.savez(b4 / "beliefs_common.npz", **with_gold(_beliefs(rng, 0)))
        np.savez(b4 / "beliefs_sel_b1.npz", **_beliefs(rng, b4_quality))
    return tmp_path


def test_delta_game_uses_selected_selection_scores_and_the_floor(tmp_path):
    _layout(tmp_path)
    d = BA.compute_delta_game(tmp_path)
    assert d["selected_selection_scores"] == [9.0, 9.5, 10.0]
    assert d["range"] == pytest.approx(1.0)
    assert d["delta_game"] == pytest.approx(max(1.0, 0.05 * 9.5))


def test_h7_holds_when_b3_has_better_beliefs_and_a_matching_game_score(tmp_path):
    _layout(tmp_path, b3_score=10.5, b3_quality=1.4, b4_quality=0.3)
    delta = {"delta_game": 1.0}
    out = BA.h7(tmp_path, delta, B=300)
    assert out["status"] == "complete"
    assert out["b3_raw_vs_b4_ts"]["ci95"][1] < 0
    assert out["belief_condition_met"] and out["game_condition_met"] and out["h7_holds"]
    assert set(out["common_metrics"]) == {"b1_raw", "b2", "b3_raw", "b3_ts", "b4_ts"}
    assert out["common_metrics"]["b3_raw"]["log_loss"]["n"] == 3


def test_h7_fails_when_b3_loses_too_much_game_score(tmp_path):
    _layout(tmp_path, b3_score=5.0, b3_quality=1.4, b4_quality=0.3)  # B1 scores 10; margin 1
    out = BA.h7(tmp_path, {"delta_game": 1.0}, B=200)
    assert out["belief_condition_met"] and not out["game_condition_met"] and not out["h7_holds"]


def test_h7_fails_when_the_beliefs_are_no_better(tmp_path):
    _layout(tmp_path, b3_score=10.0, b3_quality=0.3, b4_quality=1.4)
    out = BA.h7(tmp_path, {"delta_game": 1.0}, B=200)
    assert not out["belief_condition_met"] and not out["h7_holds"]


def test_h7_reports_incomplete_when_b4_is_missing(tmp_path):
    _layout(tmp_path)
    import shutil

    shutil.rmtree(tmp_path / "b4_s1")
    assert BA.h7(tmp_path, {"delta_game": 1.0}, B=50)["status"] == "incomplete"


def test_temperature_scaling_in_the_belief_block_fixes_overconfidence(tmp_path):
    rng = np.random.default_rng(3)
    b = _beliefs(rng, 1.0, scale=4.0)  # four times too confident
    raw = BA.belief_block(b)
    T = BA.fit_temperature_on(b)
    fixed = BA.belief_block(b, T)
    assert T > 2.0
    assert fixed["log_loss"] < raw["log_loss"] and fixed["ece"] < raw["ece"]
    assert set(raw["by_kind"]) == {"brick30", "life30", "return"}


def test_analyse_runs_end_to_end(tmp_path):
    _layout(tmp_path, b3_score=10.5, b3_quality=1.4, b4_quality=0.3)
    (tmp_path / "delta.json").write_text(json.dumps({"delta_game": 1.0}))
    out = BA.analyse(tmp_path, tmp_path / "delta.json", B=100)
    assert set(out["arms"]) == {"b1", "b3", "b5"}
    assert out["arms"]["b3"]["best"]["greedy"]["n"] == 3
    assert "common_raw" in out["arms"]["b3"]["best"]
