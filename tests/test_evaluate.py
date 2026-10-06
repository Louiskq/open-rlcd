"""Offline analysis of saved readouts: probabilities, temperature recovery per type, and the report structure."""

from __future__ import annotations

import numpy as np
import pytest

from rlcd import estimators as E
from rlcd import evaluate as EV
from rlcd import isolated as I


def synthetic(n_per_type: int = 6000, scale: dict[str, float] | None = None, seed: int = 0):
    """Readouts z = scale * z_true, with labels drawn from the TRUE distributions (so the right T is 'scale')."""
    scale = scale or {"noul": 1.0, "choice": 1.0, "score": 1.0}
    rng = np.random.default_rng(seed)
    rows = []
    for code, (name, k) in {0: ("noul", 2), 1: ("choice", 4), 2: ("score", 5)}.items():
        for _ in range(n_per_type):
            z_true = rng.normal(size=k) * 1.5
            p = I.combine(I.yes_probability(z_true))[0][0] if code == 2 else E.softmax(z_true)
            gold = int(rng.choice(k, p=p))
            rows.append((code, k, gold, z_true * scale[name], f"dom{code}"))
    kmax = 5
    z = np.full((len(rows), kmax), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        z[i, : r[1]] = r[3]
    return {
        "qid": np.array([f"q{i}" for i in range(len(rows))]),
        "qtype": np.array([r[0] for r in rows], dtype=np.int8),
        "n": np.array([r[1] for r in rows], dtype=np.int16),
        "gold": np.array([r[2] for r in rows], dtype=np.int16),
        "domain": np.array([r[4] for r in rows]),
        "z": z,
    }


def test_probabilities_are_softmax_for_list_types_and_normalised_yes_for_score():
    p = synthetic(3)
    probs = EV.probabilities(p)
    for i, q in enumerate(probs):
        z = p["z"][i, : p["n"][i]].astype(float)
        assert q.sum() == pytest.approx(1.0)
        if p["qtype"][i] == 2:
            np.testing.assert_allclose(q, I.combine(I.yes_probability(z))[0][0], atol=1e-7)
        else:
            np.testing.assert_allclose(q, E.softmax(z), atol=1e-7)


def test_temperature_changes_a_score_row_the_way_decider_applies_it():
    z = np.array([[2.0, -1.0, 0.5, np.nan, np.nan]], dtype=np.float32)
    preds = {
        "qid": np.array(["a"]),
        "qtype": np.array([2], dtype=np.int8),
        "n": np.array([3], dtype=np.int16),
        "gold": np.array([0], dtype=np.int16),
        "z": z,
    }
    (p,) = EV.probabilities(preds, 2.0)
    np.testing.assert_allclose(
        p, I.combine(I.yes_probability(z[0, :3].astype(float) / 2.0))[0][0], atol=1e-7
    )


@pytest.mark.parametrize(
    "scale",
    [{"noul": 1.0, "choice": 1.0, "score": 1.0}, {"noul": 2.0, "choice": 0.6, "score": 1.8}],
)
def test_fit_temperatures_recovers_the_per_type_scale(scale):
    temps = EV.fit_temperatures(synthetic(8000, scale))
    for name, s in scale.items():
        assert temps[name] == pytest.approx(s, rel=0.07)


def test_summarize_split_reports_every_type_the_pooled_block_and_per_domain_ece():
    out = EV.summarize_split(synthetic(500))
    assert {"noul", "choice", "score", "all", "per_domain_ece"} <= set(out)
    assert {"rps", "level_mae"} <= set(out["score"]) and "classwise_ece" in out["choice"]
    assert out["all"]["n"] == 1500 and set(out["per_domain_ece"]) == {
        "dom0",
        "dom1",
        "dom2",
        "mean",
    }


def test_scaling_fixes_an_overconfident_model_and_leaves_accuracy_alone():
    preds = synthetic(6000, {"noul": 2.5, "choice": 2.5, "score": 2.5})
    temps = EV.fit_temperatures(synthetic(6000, {"noul": 2.5, "choice": 2.5, "score": 2.5}, seed=1))
    raw, ts = EV.summarize_split(preds, 1.0), EV.summarize_split(preds, temps)
    for t in ("noul", "choice", "score"):
        assert ts[t]["log_loss"] < raw[t]["log_loss"] and ts[t]["ece"] < raw[t]["ece"]
        assert ts[t]["acc"] == raw[t]["acc"]


def test_score_mass_deviation_matches_the_numpy_definition():
    preds = synthetic(200)
    idx = preds["qtype"] == 2
    z = [preds["z"][i, : preds["n"][i]].astype(float) for i in np.flatnonzero(idx)]
    ref = np.mean(
        [
            abs(1 / (1 + np.exp(-x)).sum() - 1) if False else abs((1 / (1 + np.exp(-x))).sum() - 1)
            for x in z
        ]
    )
    assert EV.score_mass_deviation(preds) == pytest.approx(ref, rel=1e-5)
    assert EV.score_mass_deviation({"qtype": np.array([0, 1], dtype=np.int8)}) is None
