"""SCORE answers read out as decider serves them: one yes/no pass per level ("isolated levels").

Each level ``k`` of an ordinal question is judged in its own row, ``"{question} / Proposed answer:
{level k} / Does the proposed answer fit?"``, and decider's training rows label it 1 exactly when
``k`` is the gold level (``decider.data.augment.isolated``). So a pass asks "is the score *equal to*
k", not "is it at least k": there is no monotonicity to check, and the per-level yes probabilities
``y_k`` need not sum to 1. They are turned into one distribution by normalisation (decider's
``combine_isolated``)::

    p_k = y_k / sum_j y_j          mass = sum_j y_j

The log score and the RPS of a SCORE question are computed on ``p``. ``mass`` is a diagnostic: its
mean absolute distance from 1 says how far the per-level passes are from being a distribution on
their own.

Everything works on the per-level logit difference ``l_k = z_yes - z_no`` at temperature 1, so
``y_k = sigmoid(l_k)``. Gradients are with respect to ``l`` and point towards higher score.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from rlcd.scoring import EPS, log_score_grad_probs, rps_score_grad_probs


def yes_probability(level_logit_diff: np.ndarray) -> np.ndarray:
    """``y = sigmoid(z_yes - z_no)`` for each level row, clipped away from 0 and 1."""
    l = np.asarray(level_logit_diff, dtype=np.float64)
    return np.clip(expit(l), EPS, 1.0 - EPS)


def combine(yes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalise per-level yes probabilities ``[N, L]`` into distributions, and return the mass."""
    y = np.atleast_2d(np.asarray(yes, dtype=np.float64))
    mass = y.sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError("a SCORE question has no level with positive yes probability")
    return y / mass[:, None], mass


def mass_deviation(yes: np.ndarray) -> float:
    """Mean absolute deviation of the unnormalised sum of yes probabilities from 1."""
    return float(np.abs(combine(yes)[1] - 1.0).mean())


def score_grad_level_logits(
    level_logit_diff: np.ndarray, gold: np.ndarray, rule: str = "rps", normalise_rps: bool = True
) -> np.ndarray:
    """Gradient of the proper score of the *normalised* distribution w.r.t. the level logits.

    ``S`` is the log score (``rule='log'``) or ``-RPS`` (``rule='rps'``, divided by K-1 when
    ``normalise_rps``). Chain: ``dS/dp`` (the scoring rule) through the normalisation
    ``dS/dy_j = (g_j - <p, g>) / mass``, then through the sigmoid ``dy/dl = y (1 - y)``.
    """
    y = yes_probability(level_logit_diff)
    p, mass = combine(y)
    gold = np.asarray(gold)
    if rule == "log":
        g = log_score_grad_probs(p, gold)
    elif rule == "rps":
        g = rps_score_grad_probs(p, gold, normalise=normalise_rps)
    else:
        raise ValueError("rule must be 'log' or 'rps'")
    d_yes = (g - (p * g).sum(axis=1, keepdims=True)) / mass[:, None]
    return d_yes * y * (1.0 - y)
