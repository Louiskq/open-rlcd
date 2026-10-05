"""Estimator unbiasedness: the RLCD-bandit gradient equals the log-likelihood gradient on average.

Three levels of evidence, from exact to statistical:

1. exact enumeration over the sampled answer ``a`` (no randomness, tolerance 1e-12),
2. Monte Carlo through the same code path training uses, within 5 standard errors,
3. a negative control: the same Monte Carlo check *rejects* RLVR, so the test has teeth.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlcd import estimators as E
from rlcd import scoring as S

K = 5


def make_policy(seed: int, sharp: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=K) * sharp
    return z, E.softmax(z)


def enumerate_expected_gradient(p: np.ndarray, y: int, reward_fn) -> np.ndarray:
    """``sum_a p(a) * g(a)`` for the single-sample, no-baseline estimator, by enumerating ``a``."""
    total = np.zeros(K)
    for a in range(K):
        acts = np.array([[a]])
        r = reward_fn(acts, np.array([y]), p[None])
        total += p[a] * E.policy_gradient_logits(p[None], acts, r, baseline="none")[0]
    return total


def loglik_grad(p: np.ndarray, y: int) -> np.ndarray:
    onehot = np.eye(K)[y]
    return onehot - p


# ------------------------------------------------------------------- 1. exact enumeration


@pytest.mark.parametrize("seed", range(5))
def test_bandit_estimator_is_exactly_unbiased_by_enumeration(seed):
    _, p = make_policy(seed, sharp=1.5)
    for y in range(K):
        got = enumerate_expected_gradient(p, y, lambda a, yy, pp: E.reward_rlcd_bandit(a, yy, pp))
        np.testing.assert_allclose(got, loglik_grad(p, y), atol=1e-12)


def test_clipped_bandit_expectation_is_the_shrunk_gradient():
    """With clip c the expectation is min(1, c p_y) (e_y - p): biased exactly when p_y < 1/c."""
    _, p = make_policy(0, sharp=2.0)
    c = 10.0
    for y in range(K):
        got = enumerate_expected_gradient(
            p, y, lambda a, yy, pp: E.reward_rlcd_bandit(a, yy, pp, clip=c)
        )
        np.testing.assert_allclose(got, min(1.0, c * p[y]) * loglik_grad(p, y), atol=1e-12)
    assert (p < 1 / c).any() and (p > 1 / c).any()  # the fixture exercises both regimes


def test_rlvr_expectation_is_the_gradient_of_p_y_not_of_log_p_y():
    z, p = make_policy(1)
    y = 2
    got = enumerate_expected_gradient(p, y, lambda a, yy, pp: E.reward_rlvr(a, yy))
    np.testing.assert_allclose(got, p[y] * loglik_grad(p, y), atol=1e-12)
    eps = 1e-6  # and that is d p_y / d z by finite differences
    fd = np.array(
        [
            (E.softmax(z + eps * np.eye(K)[j])[y] - E.softmax(z - eps * np.eye(K)[j])[y])
            / (2 * eps)
            for j in range(K)
        ]
    )
    np.testing.assert_allclose(got, fd, atol=1e-8)


# ------------------------------------------------------------------ 2. Monte Carlo (the spec)


def monte_carlo_mean_and_se(arm: E.Arm, z: np.ndarray, y: np.ndarray, seed: int):
    """Mean and standard error of the per-context gradient over many independent contexts."""
    rng = np.random.default_rng(seed)
    g = E.logit_gradient(arm, np.tile(z, (len(y), 1)), y, rng)
    return g.mean(axis=0), g.std(axis=0, ddof=1) / np.sqrt(len(y))


def test_rlcd_bandit_mean_gradient_matches_loglik_gradient_within_monte_carlo_error():
    """The pre-registered unit test: fixed outcome y, 400k samples, 5 standard errors."""
    z, p = make_policy(3, sharp=1.5)
    y = int(
        np.argsort(p)[1]
    )  # a not-too-likely outcome, so the importance weights are heavy-tailed
    n = 400_000
    arm = E.Arm("bandit", "rlcd_bandit", samples=1, baseline="none")
    mean, se = monte_carlo_mean_and_se(arm, z, np.full(n, y), seed=0)
    assert np.all(np.abs(mean - loglik_grad(p, y)) <= 5 * se)
    assert np.all(se > 0)


def test_rlcd_bandit_with_sampled_outcomes_matches_population_loglik_gradient():
    """y ~ q: E[gradient] = grad_z E_q[log p_y] = q - p."""
    z, p = make_policy(4)
    q = E.softmax(np.random.default_rng(40).normal(size=K))
    n = 400_000
    rng = np.random.default_rng(1)
    y = E.sample_categorical(np.tile(q, (n, 1)), 1, rng)[:, 0]
    arm = E.Arm("bandit", "rlcd_bandit", samples=1, baseline="none")
    g = E.logit_gradient(arm, np.tile(z, (n, 1)), y, rng)
    mean, se = g.mean(axis=0), g.std(axis=0, ddof=1) / np.sqrt(n)
    assert np.all(np.abs(mean - (q - p)) <= 5 * se)


def test_leave_one_out_baseline_stays_unbiased_and_group_mean_shrinks_by_s_minus_1_over_s():
    z, p = make_policy(5)
    y = 1
    n, s = 100_000, 8
    target = loglik_grad(p, y)
    loo = E.Arm("loo", "rlcd_bandit", samples=s, baseline="loo")
    gm = E.Arm("gm", "rlcd_bandit", samples=s, baseline="group_mean")
    m_loo, se_loo = monte_carlo_mean_and_se(loo, z, np.full(n, y), seed=2)
    m_gm, se_gm = monte_carlo_mean_and_se(gm, z, np.full(n, y), seed=3)
    assert np.all(np.abs(m_loo - target) <= 5 * se_loo)
    assert np.all(np.abs(m_gm - (s - 1) / s * target) <= 5 * se_gm)


# ------------------------------------------------------------------ 3. negative controls


def test_the_check_rejects_rlvr_as_an_estimator_of_the_loglik_gradient():
    z, p = make_policy(3, sharp=1.5)
    y = int(np.argsort(p)[1])
    arm = E.Arm("rlvr", "rlvr", samples=1, baseline="none")
    mean, se = monte_carlo_mean_and_se(arm, z, np.full(200_000, y), seed=4)
    assert np.any(np.abs(mean - loglik_grad(p, y)) > 5 * se)


def test_clipping_bounds_the_per_sample_weight_and_so_the_gradient_norm():
    p = np.array([0.45, 0.30, 0.17, 0.06, 0.02])
    n, y = 50_000, 4  # p_y = 0.02: unclipped weight 50, clipped weight 10
    z = np.tile(np.log(p), (n, 1))
    yy = np.full(n, y)
    unclipped = E.Arm("u", "rlcd_bandit", samples=1, baseline="none")
    clipped = E.Arm("c", "rlcd_bandit", clip=10.0, samples=1, baseline="none")
    g_un = E.logit_gradient(unclipped, z, yy, np.random.default_rng(7))  # same draws for both arms
    g_cl = E.logit_gradient(clipped, z, yy, np.random.default_rng(7))
    norm_un, norm_cl = np.linalg.norm(g_un, axis=1), np.linalg.norm(g_cl, axis=1)
    assert norm_cl.max() <= 10.0 * np.sqrt(2)
    assert norm_un.max() > 40.0
    assert g_un.var(axis=0).sum() > 2 * g_cl.var(axis=0).sum()


def test_rlvr_is_improper_calibration_is_not_a_fixed_point_but_it_is_for_rlcd():
    """At p = q the expected RLVR gradient is nonzero and sharpens; the RLCD gradient is zero."""
    q = np.array([0.5, 0.25, 0.15, 0.07, 0.03])
    rlvr = np.zeros(K)
    rlcd = np.zeros(K)
    for y in range(K):
        rlvr += q[y] * enumerate_expected_gradient(q, y, lambda a, yy, pp: E.reward_rlvr(a, yy))
        rlcd += q[y] * enumerate_expected_gradient(
            q, y, lambda a, yy, pp: E.reward_rlcd_bandit(a, yy, pp)
        )
    np.testing.assert_allclose(rlcd, 0.0, atol=1e-12)
    assert rlvr[0] > 0 > rlvr[-1]  # pushes mass onto the argmax, away from the tail
    assert (q**2).sum() < q.max()  # expected RLVR reward is higher for one-hot-on-argmax than for q


# ------------------------------------------------------------------ the full-distribution path


def test_rlcd_full_log_score_gradient_equals_supervised_gradient():
    rng = np.random.default_rng(8)
    z = rng.normal(size=(64, K)) * 2
    y = rng.integers(0, K, size=64)
    p = E.softmax(z)
    np.testing.assert_allclose(
        E.full_score_grad_logits(p, y, "log"), E.supervised_grad_logits(p, y), atol=1e-12
    )
    # and through the arm dispatcher, which is the code path training uses
    full = E.logit_gradient(E.Arm("full", "rlcd_full"), z, y, rng)
    sup = E.logit_gradient(E.Arm("sup", "supervised"), z, y, rng)
    np.testing.assert_allclose(full, sup, atol=1e-12)


@pytest.mark.parametrize("rule", ["log", "rps"])
def test_full_score_gradient_matches_finite_differences_in_logit_space(rule):
    rng = np.random.default_rng(9)
    z = rng.normal(size=(6, K))
    y = rng.integers(0, K, size=6)
    ana = E.full_score_grad_logits(E.softmax(z), y, rule)
    eps = 1e-6
    for j in range(K):
        dz = np.zeros_like(z)
        dz[:, j] = eps
        fd = (
            E.reward_rlcd_full(E.softmax(z + dz), y, rule)
            - E.reward_rlcd_full(E.softmax(z - dz), y, rule)
        ) / (2 * eps)
        np.testing.assert_allclose(ana[:, j], fd, atol=1e-7)


def test_rps_gradient_ranks_nearby_levels_above_distant_ones():
    """For gold level 4 the RPS gradient favours level 3 over 0; the log score cannot tell."""
    p = np.full((1, K), 1 / K)
    g_rps = E.full_score_grad_logits(p, np.array([4]), "rps")[0]
    assert g_rps[3] > g_rps[0]
    g_log = E.full_score_grad_logits(p, np.array([4]), "log")[0]
    assert g_log[3] == pytest.approx(g_log[0])


# ------------------------------------------------------------------------------ KL gradients


@pytest.mark.parametrize("direction", ["reverse", "forward"])
def test_kl_gradient_matches_finite_differences(direction):
    rng = np.random.default_rng(10)
    z, zr = rng.normal(size=(1, K)), rng.normal(size=(1, K))
    pr = E.softmax(zr)

    def kl(zz):
        p = E.softmax(zz)
        a, b = (p, pr) if direction == "reverse" else (pr, p)
        return float((a * (np.log(a) - np.log(b))).sum())

    eps = 1e-6
    fd = np.array(
        [(kl(z + eps * np.eye(K)[j]) - kl(z - eps * np.eye(K)[j])) / (2 * eps) for j in range(K)]
    )
    np.testing.assert_allclose(E.kl_grad_logits(E.softmax(z), pr, direction)[0], fd, atol=1e-7)


def test_kl_gradient_is_zero_at_the_reference():
    p = E.softmax(np.random.default_rng(11).normal(size=(3, K)))
    for d in ("reverse", "forward"):
        np.testing.assert_allclose(E.kl_grad_logits(p, p, d), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------------- plumbing


def test_advantages_and_sampler_basics():
    r = np.array([[1.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(E.advantages(r, "group_mean"), [[0.5, -0.5, -0.5, 0.5]])
    np.testing.assert_allclose(
        E.advantages(r, "loo"), [[1 - 1 / 3, 0 - 2 / 3, 0 - 2 / 3, 1 - 1 / 3]]
    )
    np.testing.assert_allclose(E.advantages(r, "none"), r)
    with pytest.raises(ValueError):
        E.advantages(r[:, :1], "loo")
    p = np.array([[0.1, 0.2, 0.7]])
    draws = E.sample_categorical(np.tile(p, (200_000, 1)), 1, np.random.default_rng(0))[:, 0]
    np.testing.assert_allclose(np.bincount(draws, minlength=3) / len(draws), p[0], atol=0.005)


def test_arm_validation_and_reward_mixture_endpoints():
    with pytest.raises(ValueError):
        E.Arm("bad", "ppo")
    with pytest.raises(ValueError):
        E.Arm("bad", "rlvr", baseline="median")
    rng = np.random.default_rng(12)
    p = E.softmax(rng.normal(size=(4, K)))
    a = E.sample_categorical(p, 3, rng)
    y = rng.integers(0, K, size=4)
    np.testing.assert_allclose(E.reward_mixture(a, y, p, 1.0), E.reward_rlvr(a, y))
    np.testing.assert_allclose(E.reward_mixture(a, y, p, 0.0), E.reward_rlcd_bandit(a, y, p))
    with pytest.raises(ValueError):
        E.reward_mixture(a, y, p, 1.5)


def test_reward_rlcd_full_matches_scoring_module():
    rng = np.random.default_rng(13)
    p = E.softmax(rng.normal(size=(5, K)))
    y = rng.integers(0, K, size=5)
    np.testing.assert_allclose(E.reward_rlcd_full(p, y, "log"), S.log_score(p, y))
    np.testing.assert_allclose(E.reward_rlcd_full(p, y, "rps"), -S.rps_per_item(p, y))
