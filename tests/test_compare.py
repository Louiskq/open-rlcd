"""The pre-registered comparisons: p-value definitions, Holm, and detection on synthetic arms of known quality."""

from __future__ import annotations

import numpy as np
import pytest

from rlcd import compare as C

DELTAS = {
    "raw": {f"all/{m}": 0.02 for m in C.METRICS},
    "ts": {f"all/{m}": 0.02 for m in C.METRICS},
}


def test_holm_matches_the_textbook_example():
    adj, rej = C.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert rej == [True, False, False, True]


def test_holm_is_monotone_and_capped():
    adj, _ = C.holm([0.9, 0.001, 0.5])
    assert max(adj) <= 1.0
    assert adj[1] == pytest.approx(0.003)


def test_directional_p_is_two_sided_equivalent_of_the_interval():
    d = np.array([0.1] * 975 + [-0.1] * 25)  # 2.5% of replicates on the wrong side
    assert C.p_directional(d, "a_higher") == pytest.approx(0.05)
    assert C.p_directional(-d, "a_lower") == pytest.approx(0.05)
    assert C.p_directional(np.zeros(100), "a_higher") == 1.0  # ties count against the claim


def test_equivalence_p_is_inside_the_margin_test():
    d = np.linspace(-0.01, 0.01, 1001)
    assert C.p_equivalence(d, 0.02) == 0.0
    assert C.p_equivalence(d, 0.005) == pytest.approx(min(1.0, 2 * 0.25), abs=0.01)
    assert C.p_equivalence(d + 0.05, 0.02) == 1.0


def _arm(arm: str, seed: int, mat: np.ndarray) -> C.Arm:
    return C.Arm(arm, [C.RunProbs(f"{arm}_s{seed}", seed, {}, mat, mat)])


def _synthetic(n: int = 400, k: int = 4, sharp: float = 1.4, seed: int = 0):
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, k, n)
    domain = np.array([f"d{i % 4}" for i in range(n)])
    # latent evidence on the gold option; the same noise for every arm so the arms are paired
    noise = rng.normal(size=(n, k))
    items = C.Items(np.arange(n), gold, np.zeros(n, dtype=int), domain)

    def probs(strength: float, temperature: float = 1.0) -> np.ndarray:
        z = noise.copy()
        z[np.arange(n), gold] += strength
        z = z / temperature
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    return items, probs, sharp


def test_identical_arms_are_equivalent_and_not_different():
    items, probs, _ = _synthetic()
    arms = {"a1": _arm("a1", 0, probs(1.4)), "a3": _arm("a3", 0, probs(1.4))}
    tests = [
        C.Test("t-ece", "primary", "H1", ("a3", None), ("a1", None), "raw", "ece", "a_higher"),
        C.Test("t-eq", "primary", "H1", ("a3", None), ("a1", None), "raw", "log_loss", "equiv"),
    ]
    out = C.run_tests(arms, items, DELTAS, tests, B=200, seed=1)
    assert out[0]["observed_difference"] == 0.0 and not out[0]["passes"]
    assert out[1]["observed_difference"] == 0.0 and out[1]["passes"]


def test_a_sharper_overconfident_arm_is_detected():
    items, probs, _ = _synthetic(n=800)
    arms = {"a1": _arm("a1", 0, probs(1.4)), "a3": _arm("a3", 0, probs(1.4, temperature=0.4))}
    tests = [
        C.Test("t-ll", "primary", "H1", ("a3", None), ("a1", None), "raw", "log_loss", "a_higher"),
        C.Test("t-ece", "primary", "H1", ("a3", None), ("a1", None), "raw", "ece", "a_higher"),
        C.Test("t-acc", "primary", "H1", ("a3", None), ("a1", None), "raw", "acc", "equiv"),
    ]
    out = {r["id"]: r for r in C.run_tests(arms, items, DELTAS, tests, B=300, seed=2)}
    assert out["t-ll"]["observed_difference"] > 0 and out["t-ll"]["passes"]
    assert out["t-ece"]["observed_difference"] > 0
    # temperature does not change the argmax, so accuracy is exactly equivalent
    assert out["t-acc"]["observed_difference"] == 0.0 and out["t-acc"]["passes"]


def test_seed_mean_and_explicit_seed_selection():
    items, probs, _ = _synthetic()
    a1 = C.Arm(
        "a1",
        [
            C.RunProbs("a1_s0", 0, {}, probs(1.0), probs(1.0)),
            C.RunProbs("a1_s1", 1, {}, probs(2.0), probs(2.0)),
        ],
    )
    a5 = _arm("a5", 0, probs(1.0))
    arms = {"a1": a1, "a5": a5}
    t_seed0 = C.Test("s0", "primary", "H2", ("a5", (0,)), ("a1", (0,)), "raw", "log_loss", "equiv")
    t_all = C.Test("all", "primary", "H2", ("a5", (0,)), ("a1", None), "raw", "log_loss", "equiv")
    out = {r["id"]: r for r in C.run_tests(arms, items, DELTAS, [t_seed0, t_all], B=50, seed=3)}
    assert out["s0"]["observed_difference"] == 0.0
    assert (
        out["all"]["observed_difference"] > 0
    )  # seed 1 is sharper, so the seed mean is lower-loss than a5


def test_ensemble_mode_averages_probabilities_and_keeps_explicit_seeds():
    items, probs, _ = _synthetic()
    a1 = C.Arm(
        "a1",
        [
            C.RunProbs("a1_s0", 0, {}, probs(1.0), probs(1.0)),
            C.RunProbs("a1_s1", 1, {}, probs(2.0), probs(2.0)),
        ],
    )
    a5 = _arm("a5", 0, probs(1.0))
    tests = [
        C.Test("ens", "primary", "H2", ("a5", (0,)), ("a1", None), "raw", "log_loss", "equiv"),
        C.Test("s0", "primary", "H2", ("a5", (0,)), ("a1", (0,)), "raw", "log_loss", "equiv"),
    ]
    out = {
        r["id"]: r
        for r in C.run_tests(
            {"a1": a1, "a5": a5}, items, DELTAS, tests, B=50, seed=4, ensemble=True
        )
    }
    assert out["s0"]["observed_difference"] == 0.0
    assert "ensemble" in out["ens"]["b"]
    assert out["ens"]["observed_difference"] != 0.0


def test_preregistered_test_list_matches_the_amendment():
    tests = C.preregistered_tests()
    primary = [t for t in tests if t.family == "primary"]
    assert len(primary) == 17 and len([t for t in tests if t.family == "H5b"]) == 2
    assert len({t.id for t in tests}) == len(tests)
    assert {t.id for t in tests if t.hypothesis == "H2"} == {
        f"C2-{m}" for m in ("log_loss", "ece", "acc", "brier", "reliability", "resolution")
    }
    # H5b and C5 are on temperature-scaled rows, everything else is raw
    assert {t.row for t in tests if t.hypothesis in ("H5", "H5b")} == {"ts"}
    assert {t.row for t in tests if t.hypothesis in ("H1", "H2", "H3", "H4")} == {"raw"}


def test_hypothesis_verdicts_follow_amendment_item_7():
    rows = [
        {"hypothesis": "H4", "id": "C4-ece", "passes": False},
        {"hypothesis": "H4", "id": "C4-acc", "passes": False},
        {"hypothesis": "H1", "id": "C1-ece", "passes": True},
        {"hypothesis": "H1", "id": "C1-acc", "passes": False},
    ]
    v = C.hypothesis_verdicts(rows)
    assert v["H4"]["holds"] is True  # no C4 test passes
    assert v["H1"]["holds"] is False  # needs every H1 test to pass
