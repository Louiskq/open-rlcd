"""Metric correctness: hand-computed values, independent loop references, and identities."""

from __future__ import annotations

import numpy as np
import pytest

from rlcd import scoring as S


def random_probs(rng: np.random.Generator, n: int, k: int, sharp: float = 1.5) -> np.ndarray:
    z = rng.normal(size=(n, k)) * sharp
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def sample_labels(rng: np.random.Generator, p: np.ndarray) -> np.ndarray:
    return (rng.random((len(p), 1)) > np.cumsum(p, axis=1)).sum(axis=1).clip(max=p.shape[1] - 1)


# ----------------------------------------------------------------------------- proper scores


def test_log_loss_hand_computed():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    assert S.log_loss(p, np.array([0, 2])) == pytest.approx(-(np.log(0.7) + np.log(0.8)) / 2)


def test_log_score_is_clipped_not_infinite():
    p = np.array([[1.0, 0.0]])
    assert np.isfinite(S.log_score(p, np.array([1]))).all()


def test_brier_hand_computed():
    assert S.brier(np.array([[0.7, 0.2, 0.1]]), np.array([0])) == pytest.approx(0.14)


def test_rps_hand_computed_all_gold_levels():
    p = np.array([[0.2, 0.5, 0.3]] * 3)
    y = np.array([0, 1, 2])
    # CDF = [0.2, 0.7]; O = [1,1], [0,1], [0,0]
    raw = np.array([0.8**2 + 0.3**2, 0.2**2 + 0.3**2, 0.2**2 + 0.7**2])
    np.testing.assert_allclose(S.rps_per_item(p, y, normalise=False), raw)
    np.testing.assert_allclose(S.rps_per_item(p, y), raw / 2)


def test_rps_with_two_levels_is_binary_brier():
    rng = np.random.default_rng(1)
    p = random_probs(rng, 50, 2)
    y = rng.integers(0, 2, size=50)
    assert S.rps(p, y) == pytest.approx(S.brier(p, y) / 2)


def test_rps_is_ordinal_but_brier_is_not():
    far = np.array([[1.0, 0, 0, 0, 0]])
    near = np.array([[0.0, 0, 0, 1.0, 0]])
    y = np.array([4])
    assert S.rps(near, y) < S.rps(far, y)
    assert S.brier(near, y) == pytest.approx(S.brier(far, y))


def test_rps_ignores_zero_padding_given_n_options():
    p3 = np.array([[0.2, 0.5, 0.3]])
    padded = np.array([[0.2, 0.5, 0.3, 0.0, 0.0]])
    y = np.array([1])
    assert S.rps(padded, y, n_options=np.array([3])) == pytest.approx(S.rps(p3, y))


def test_level_mae_and_expected_level():
    p = np.array([[0.0, 0.5, 0.5]])
    assert S.expected_level(p)[0] == pytest.approx(1.5)
    assert S.level_mae(p, np.array([0])) == pytest.approx(1.5)


@pytest.mark.parametrize("rule", ["log", "neg_brier", "neg_rps"])
def test_scores_are_strictly_proper(rule):
    """Expected score under the true q is maximised by reporting q (not by any other p)."""
    rng = np.random.default_rng(2)
    k = 5
    ys = np.arange(k)

    def expected_score(p_report: np.ndarray, q: np.ndarray) -> float:
        reports = np.repeat(p_report[None, :], k, axis=0)
        if rule == "log":
            per_y = S.log_score(reports, ys)
        elif rule == "neg_brier":
            per_y = -S.brier_per_item(reports, ys)
        else:
            per_y = -S.rps_per_item(reports, ys)
        return float(q @ per_y)

    for _ in range(25):
        q = random_probs(rng, 1, k)[0]
        p = random_probs(rng, 1, k)[0]
        assert expected_score(q, q) > expected_score(p, q)


@pytest.mark.parametrize("normalise", [True, False])
def test_score_gradients_wrt_probs_match_directional_finite_differences(normalise):
    rng = np.random.default_rng(3)
    p = random_probs(rng, 6, 5)
    y = rng.integers(0, 5, size=6)
    d = rng.normal(size=p.shape)
    d -= d.mean(axis=1, keepdims=True)  # tangent to the simplex: rows still sum to 1
    eps = 1e-6
    # log score
    num = (S.log_score(p + eps * d, y) - S.log_score(p - eps * d, y)) / (2 * eps)
    np.testing.assert_allclose(num, (S.log_score_grad_probs(p, y) * d).sum(1), rtol=1e-5)
    # -RPS
    fd = (
        -S.rps_per_item(p + eps * d, y, normalise=normalise)
        + S.rps_per_item(p - eps * d, y, normalise=normalise)
    ) / (2 * eps)
    ana = (S.rps_score_grad_probs(p, y, normalise=normalise) * d).sum(1)
    np.testing.assert_allclose(-fd, -ana, rtol=1e-5, atol=1e-9)


# ------------------------------------------------------------------------------ calibration


def ref_ece(conf, correct, bins=15):
    """The loop in decider/metrics.py: (lo, hi] equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def ref_ece_mass(conf, correct, bins=15):
    order = np.argsort(conf, kind="stable")
    e = 0.0
    for chunk in np.array_split(order, bins):
        e += len(chunk) / len(conf) * abs(conf[chunk].mean() - correct[chunk].mean())
    return float(e)


def test_ece_hand_computed():
    conf = np.array([0.9, 0.9, 0.6, 0.6])
    correct = np.array([1, 0, 1, 1])
    # bin of 0.9: acc 0.5 (gap 0.4, weight 0.5); bin of 0.6: acc 1.0 (gap 0.4, weight 0.5)
    assert S.ece(conf, correct) == pytest.approx(0.4)


def test_ece_matches_decider_reference_and_mass_reference():
    rng = np.random.default_rng(4)
    for k in (2, 5, 9):
        p = random_probs(rng, 3000, k, sharp=2.0)
        y = rng.integers(0, k, size=3000)
        conf, correct = p.max(1), p.argmax(1) == y
        assert S.ece(conf, correct) == pytest.approx(ref_ece(conf, correct.astype(float)))
        assert S.ece(conf, correct, binning="mass") == pytest.approx(
            ref_ece_mass(conf, correct.astype(float))
        )
        assert S.top_label_ece(p, y) == pytest.approx(S.ece(conf, correct))


def test_value_on_a_bin_edge_goes_to_the_lower_bin():
    # 0.2 with 15 bins sits exactly on the edge between bin 2 (2/15, 3/15] and bin 3
    r = S.reliability_bins(np.array([0.2]), np.array([1.0]), bins=15)
    assert r["count"].argmax() == 2


def test_ece_is_small_for_a_calibrated_forecaster_and_large_for_an_overconfident_one():
    rng = np.random.default_rng(5)
    q = random_probs(rng, 200_000, 5)
    y = sample_labels(rng, q)
    assert S.top_label_ece(q, y) < 0.01
    sharp = q**4
    sharp /= sharp.sum(1, keepdims=True)
    assert S.top_label_ece(sharp, y) > 0.05


def test_classwise_ece_matches_loop_reference():
    rng = np.random.default_rng(6)
    p = random_probs(rng, 2000, 4)
    y = rng.integers(0, 4, size=2000)
    ref = np.mean([ref_ece(p[:, k], (y == k).astype(float)) for k in range(4)])
    assert S.classwise_ece(p, y) == pytest.approx(ref)


def test_reliability_bins_counts_sum_to_n():
    rng = np.random.default_rng(7)
    conf = rng.random(500)
    r = S.reliability_bins(conf, rng.random(500) < conf)
    assert r["count"].sum() == 500


# ------------------------------------------------------------------------------------ Murphy


@pytest.mark.parametrize("bins", [3, 15, 50])
def test_murphy_identity_is_exact(bins):
    rng = np.random.default_rng(8)
    p = random_probs(rng, 4000, 4, sharp=2.0)
    y = sample_labels(rng, random_probs(rng, 4000, 4))  # deliberately miscalibrated
    m = S.murphy_decomposition(p, y, bins)
    assert m["reliability"] - m["resolution"] + m["uncertainty"] + m["within_bin"] == pytest.approx(
        m["brier"], abs=1e-12
    )
    assert m["reliability"] >= 0 and m["resolution"] >= 0 and m["uncertainty"] >= 0


def ref_murphy_binary_class(p: np.ndarray, o: np.ndarray, bins: int):
    """Textbook binary Murphy decomposition with explicit loops (no within-bin term)."""
    n = len(p)
    base = o.mean()
    rel = res = 0.0
    for b in range(bins):
        m = (p > b / bins) & (p <= (b + 1) / bins) if b else (p >= 0) & (p <= 1 / bins)
        if m.any():
            rel += m.sum() * (p[m].mean() - o[m].mean()) ** 2 / n
            res += m.sum() * (o[m].mean() - base) ** 2 / n
    return rel, res, base * (1 - base)


def test_murphy_matches_textbook_loops_when_forecasts_are_discrete():
    rng = np.random.default_rng(9)
    values = np.array([0.05, 0.25, 0.45, 0.65, 0.85])  # one value per 5-bin cell
    p0 = rng.choice(values, size=5000)
    y = (rng.random(5000) > p0).astype(int)  # y=0 with probability p0
    p = np.stack([p0, 1 - p0], axis=1)
    m = S.murphy_decomposition(p, y, bins=5)
    assert m["within_bin"] == pytest.approx(0.0, abs=1e-12)
    rel = res = unc = 0.0
    for k in (0, 1):
        r, s, u = ref_murphy_binary_class(p[:, k], (y == k).astype(float), 5)
        rel, res, unc = rel + r, res + s, unc + u
    assert (m["reliability"], m["resolution"], m["uncertainty"]) == pytest.approx((rel, res, unc))


def test_climatology_has_zero_resolution_and_brier_equals_uncertainty():
    rng = np.random.default_rng(10)
    y = rng.choice(3, size=3000, p=[0.5, 0.3, 0.2])
    base = np.bincount(y, minlength=3) / len(y)
    m = S.murphy_decomposition(np.tile(base, (len(y), 1)), y)
    assert m["resolution"] == pytest.approx(0.0, abs=1e-12)
    assert m["reliability"] == pytest.approx(0.0, abs=1e-12)
    assert m["brier"] == pytest.approx(m["uncertainty"])


# --------------------------------------------------------------------------- risk-coverage


def test_aurc_matches_decider_formula_without_ties():
    rng = np.random.default_rng(11)
    conf = rng.random(400)
    correct = (rng.random(400) < conf).astype(float)
    order = np.argsort(-conf)
    ref = (np.cumsum(1 - correct[order]) / np.arange(1, 401)).mean()
    assert S.aurc(conf, correct) == pytest.approx(ref)


def test_risk_coverage_endpoints_and_monotone_coverage():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 1])
    cov, risk = S.risk_coverage(conf, correct)
    np.testing.assert_allclose(cov, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(risk, [0, 0, 1 / 3, 1 / 4])
    assert S.aurc(conf, correct) == pytest.approx((0 + 0 + 1 / 3 + 1 / 4) / 4)


def test_aurc_with_tied_confidences_does_not_depend_on_input_order():
    conf = np.array([0.5, 0.5, 0.5, 0.5, 0.9])
    correct = np.array([1, 0, 0, 1, 1])
    base = S.aurc(conf, correct)
    rng = np.random.default_rng(12)
    for _ in range(10):
        perm = rng.permutation(5)
        assert S.aurc(conf[perm], correct[perm]) == pytest.approx(base)
    # all tied -> risk is the overall error rate at every coverage
    assert S.aurc(np.full(4, 0.7), np.array([1, 0, 0, 1])) == pytest.approx(0.5)


def test_oracle_ranking_has_lower_aurc_than_random_and_adversarial():
    rng = np.random.default_rng(13)
    correct = (rng.random(1000) < 0.7).astype(float)
    oracle = correct + 0.01 * rng.random(1000)
    adversarial = -correct + 0.01 * rng.random(1000)
    assert (
        S.aurc(oracle, correct) < S.aurc(rng.random(1000), correct) < S.aurc(adversarial, correct)
    )


def test_selective_accuracy():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 0])
    assert S.selective_accuracy(conf, correct, 0.5) == 1.0
    assert S.selective_accuracy(conf, correct, 1.0) == 0.5


# ---------------------------------------------------------------------------------- summaries


def test_summarize_keys_by_answer_type():
    rng = np.random.default_rng(14)
    p = random_probs(rng, 300, 4)
    y = rng.integers(0, 4, size=300)
    base = {"acc", "log_loss", "ece", "ece_mass", "aurc", "brier", "reliability", "resolution"}
    assert base <= set(S.summarize(p, y, "noul")) and "classwise_ece" not in S.summarize(
        p, y, "noul"
    )
    assert "classwise_ece" in S.summarize(p, y, "choice")
    assert {"rps", "level_mae"} <= set(S.summarize(p, y, "score"))
    with pytest.raises(ValueError):
        S.summarize(p, y, "essay")


def test_summarize_by_type_handles_ragged_options():
    rows = [[0.7, 0.3], [0.2, 0.8], [0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [0.2, 0.3, 0.3, 0.2]]
    y = [0, 1, 0, 2, 3]
    out = S.summarize_by_type(rows, y, ["noul", "noul", "choice", "choice", "score"])
    assert set(out) == {"noul", "choice", "score"}
    assert out["noul"]["n"] == 2 and out["choice"]["n"] == 2 and out["score"]["n"] == 1
    assert out["noul"]["acc"] == 1.0


def test_inputs_are_validated():
    with pytest.raises(ValueError):
        S.log_loss(np.array([[0.5, 0.6]]), np.array([0]))  # does not sum to 1
    with pytest.raises(ValueError):
        S.log_loss(np.array([[0.5, 0.5]]), np.array([2]))  # gold out of range
    with pytest.raises(ValueError):
        S.log_loss(np.array([[0.5, 0.5]]), np.array([0.0]))  # gold not an integer
