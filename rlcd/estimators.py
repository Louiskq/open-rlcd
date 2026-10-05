"""Rewards and gradients for RLVR, RLCD-bandit and RLCD-full.

All gradients are taken with respect to the *logits* ``z`` of a softmax policy ``p = softmax(z)``
K typed options, and point in the direction of **increasing** expected reward (gradient ascent).
For a linear policy ``z = theta x`` the parameter gradient is the outer product ``dz x^T``.

The objective RLCD argues for is a proper score of the stated distribution against outcome ``y``::

    max  E_{x, y} [ S(p(. | x), y) ]  -  beta * KL(p_ref || p)

and the three ways of estimating its gradient are:

``rlcd_full``    the whole distribution is scored, ``r = S(p, y)``; the gradient is ``grad S``
                 (needs the label ``y`` for *every* option, i.e. full feedback).
``rlcd_bandit``  only the sampled answer ``a ~ p`` can be checked. ``r = 1[a = y] / p(a)`` with a
                 stop-gradient on ``p(a)`` makes REINFORCE unbiased for ``grad log p(y)``::

                     E_a[ 1[a=y]/p(a) * grad log p(a) ] = sum_a p(a) 1[a=y]/p(a) grad log p(a)
                                                        = grad log p(y)

                 Clipping the weight at ``c`` trades that unbiasedness for lower variance.
``rlvr``         ``r = 1[a = y]`` is *not* a proper score: its expectation ``sum_a p(a) q(a)`` is
                 maximised by a one-hot ``p`` on the most likely answer, so it sharpens the model
                 beyond what the outcomes justify.

With fixed labels, ``rlcd_full`` has exactly the gradient of supervised log loss (log score) or RPS
loss (ordinal score); RL is only *needed* for bandit feedback or when outcomes depend on the
model's own actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlcd.scoring import (
    EPS,
    log_score,
    log_score_grad_probs,
    rps_per_item,
    rps_score_grad_probs,
)

BASELINES = ("none", "group_mean", "loo")


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def sample_categorical(probs: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` i.i.d. samples from each row of ``probs`` ([B, K]) -> [B, n] integer array."""
    cdf = np.cumsum(probs, axis=-1)
    u = rng.random((len(probs), n, 1)) * cdf[:, None, -1:]
    return np.minimum((u > cdf[:, None, :]).sum(axis=-1), probs.shape[-1] - 1)


# ---------------------------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------------------------


def reward_rlvr(actions: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Verifiable 0/1 reward ``1[a = y]`` for sampled ``actions`` ([B, S]) and gold ``y`` ([B])."""
    return (np.asarray(actions) == np.asarray(y)[:, None]).astype(np.float64)


def reward_rlcd_bandit(
    actions: np.ndarray, y: np.ndarray, probs: np.ndarray, clip: float | None = None
) -> np.ndarray:
    """Importance-weighted reward ``1[a = y] / p(a)``, optionally clipped at ``clip``.

    ``probs`` are treated as constants (stop-gradient): this function returns plain numbers, and the
    gradient flows only through ``log p(a)`` in ``policy_gradient_logits``.
    """
    actions = np.asarray(actions)
    p_a = np.take_along_axis(np.asarray(probs), actions, axis=1)
    r = reward_rlvr(actions, y) / np.maximum(p_a, EPS)
    return r if clip is None else np.minimum(r, clip)


def reward_mixture(
    actions: np.ndarray,
    y: np.ndarray,
    probs: np.ndarray,
    rlvr_mix: float,
    clip: float | None = None,
) -> np.ndarray:
    """``(1 - m) * RLCD-bandit + m * RLVR`` reward; ``m = 0`` is RLCD, ``m = 1`` is RLVR.

    The estimator is linear in the reward, so this is the same as mixing the two gradients.
    """
    if not 0.0 <= rlvr_mix <= 1.0:
        raise ValueError("rlvr_mix must lie in [0, 1]")
    return (1.0 - rlvr_mix) * reward_rlcd_bandit(actions, y, probs, clip) + rlvr_mix * reward_rlvr(
        actions, y
    )


def reward_rlcd_full(
    probs: np.ndarray, y: np.ndarray, rule: str = "log", n_options: np.ndarray | None = None
) -> np.ndarray:
    """Proper score ``S(p, y)`` of the stated distribution (higher is better): log score or -RPS."""
    if rule == "log":
        return log_score(probs, y)
    if rule == "rps":
        return -rps_per_item(probs, y, n_options)
    raise ValueError("rule must be 'log' or 'rps'")


# ---------------------------------------------------------------------------------------------
# gradients w.r.t. logits
# ---------------------------------------------------------------------------------------------


def advantages(rewards: np.ndarray, baseline: str = "group_mean") -> np.ndarray:
    """Subtract a per-context baseline from ``rewards`` ([B, S]).

    ``group_mean`` uses the mean over the S samples of the same context (including the sample
    itself); it shrinks the gradient by exactly ``(S-1)/S`` but keeps its direction. ``loo`` uses
    the mean of the *other* S-1 samples, which is exactly unbiased. ``none`` is plain REINFORCE.
    """
    r = np.asarray(rewards, dtype=np.float64)
    if baseline == "none":
        return r
    s = r.shape[1]
    if baseline == "group_mean":
        return r - r.mean(axis=1, keepdims=True)
    if baseline == "loo":
        if s < 2:
            raise ValueError("leave-one-out baseline needs at least 2 samples per context")
        return r - (r.sum(axis=1, keepdims=True) - r) / (s - 1)
    raise ValueError(f"baseline must be one of {BASELINES}")


def policy_gradient_logits(
    probs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    baseline: str = "group_mean",
) -> np.ndarray:
    """REINFORCE estimate of ``d E[reward] / d z``: ``mean_i adv_i (e_{a_i} - p)`` per context.

    ``probs`` [B, K], ``actions`` [B, S], ``rewards`` [B, S] -> [B, K]. Uses
    ``grad_z log p(a) = e_a - p`` for a softmax policy.
    """
    probs = np.asarray(probs, dtype=np.float64)
    adv = advantages(rewards, baseline)
    s = adv.shape[1]
    onehot = np.eye(probs.shape[1])[np.asarray(actions)]  # [B, S, K]
    return np.einsum("bs,bsk->bk", adv, onehot) / s - probs * adv.mean(axis=1, keepdims=True)


def probs_grad_to_logits_grad(probs: np.ndarray, grad_probs: np.ndarray) -> np.ndarray:
    """Chain rule through the softmax: ``J^T g = p * (g - <p, g>)``."""
    p = np.asarray(probs, dtype=np.float64)
    g = np.asarray(grad_probs, dtype=np.float64)
    return p * (g - (p * g).sum(axis=-1, keepdims=True))


def full_score_grad_logits(
    probs: np.ndarray, y: np.ndarray, rule: str = "log", n_options: np.ndarray | None = None
) -> np.ndarray:
    """Gradient w.r.t. logits of the proper score ``S(softmax(z), y)``: the RLCD-full path.

    Computed generically as (gradient of the scoring rule w.r.t. ``p``) chained through the softmax,
    so it exercises the same code for the log score and for RPS.
    """
    if rule == "log":
        g = log_score_grad_probs(probs, y)
    elif rule == "rps":
        g = rps_score_grad_probs(probs, y, n_options)
    else:
        raise ValueError("rule must be 'log' or 'rps'")
    return probs_grad_to_logits_grad(probs, g)


def supervised_grad_logits(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of the supervised log likelihood ``log p_y`` w.r.t. logits: ``e_y - p``."""
    p = np.asarray(probs, dtype=np.float64)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(p)), np.asarray(y)] = 1.0
    return onehot - p


def kl_grad_logits(
    probs: np.ndarray, ref_probs: np.ndarray, direction: str = "forward"
) -> np.ndarray:
    """Gradient w.r.t. the policy logits of a KL penalty to a reference distribution.

    ``forward`` is ``KL(p_ref || p)``: the project's choice (``docs/decisions.md`` D1) and the
    direction of decider's retention term and of the old Breakout code. It is large when ``p`` puts
    little mass on an option the reference supports, so it penalises collapse onto one option and
    partly counteracts RLVR's sharpening. ``reverse`` is ``KL(p || p_ref)``, the usual RLHF
    direction, kept for comparison. Subtract ``beta`` times this from the reward gradient.
    """
    p = np.asarray(probs, dtype=np.float64)
    q = np.asarray(ref_probs, dtype=np.float64)
    if direction == "forward":
        return p - q
    if direction == "reverse":
        logr = np.log(np.clip(p, EPS, 1.0)) - np.log(np.clip(q, EPS, 1.0))
        kl = (p * logr).sum(axis=-1, keepdims=True)
        return p * (logr - kl)
    raise ValueError("direction must be 'reverse' or 'forward'")


# ---------------------------------------------------------------------------------------------
# one entry point per training arm
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """A training arm of Exp 0 / Exp A, reduced to what determines its logit gradient.

    kind          'supervised' (analytic ``e_y - p``), 'rlcd_full' (score through the RL code path),
                  'rlcd_bandit' (sampled answers, importance-weighted reward), 'rlvr' (0/1).
    clip          weight clip for ``rlcd_bandit`` (None = unclipped).
    rlvr_mix      share of the RLVR reward mixed into ``rlcd_bandit`` (the notebook slider).
    samples       K answers sampled per context for the RL arms.
    baseline      'group_mean' (default, as specified), 'loo' or 'none'.
    rule          scoring rule for ``rlcd_full``: 'log' or 'rps'.
    """

    name: str
    kind: str
    clip: float | None = None
    rlvr_mix: float = 0.0
    samples: int = 8
    baseline: str = "group_mean"
    rule: str = "log"

    def __post_init__(self) -> None:
        if self.kind not in ("supervised", "rlcd_full", "rlcd_bandit", "rlvr"):
            raise ValueError(f"unknown arm kind {self.kind!r}")
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}")


def logit_gradient(arm: Arm, z: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ascent-direction gradient of the arm's objective w.r.t. logits ``z`` [B, K], gold ``y``."""
    p = softmax(z)
    if arm.kind == "supervised":
        return supervised_grad_logits(p, y)
    if arm.kind == "rlcd_full":
        return full_score_grad_logits(p, y, arm.rule)
    actions = sample_categorical(p, arm.samples, rng)
    if arm.kind == "rlvr":
        rewards = reward_rlvr(actions, y)
    else:
        rewards = reward_mixture(actions, y, p, arm.rlvr_mix, arm.clip)
    return policy_gradient_logits(p, actions, rewards, arm.baseline)
