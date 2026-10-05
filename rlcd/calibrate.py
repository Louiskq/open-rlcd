"""Post-hoc temperature scaling, one temperature per answer type.

``p = softmax(z / T)``: ``T > 1`` softens an over-confident model, ``T < 1`` sharpens an
under-confident one, and the argmax (so accuracy) never changes. ``T`` is fitted by minimising NLL
on a calibration split that is disjoint from every evaluation set (see ``rlcd.data``).

This mirrors ``decider.calibrate`` (Mapika, Apache 2.0), whose ``decider_config.json`` stores
``temperature_by_type = {"choice": ..., "noul": ..., "score": ...}``. Not covered yet: decider's
"isolated levels" readout for SCORE questions (one yes/no forward pass per level, combined by
normalisation); whether Exp A trains and serves SCORE that way is an open question for phase 2.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

T_MIN, T_MAX = 0.05, 20.0


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """``softmax(logits / T)`` over the last axis; ``-inf`` logits (padding) get probability 0."""
    if not temperature > 0:
        raise ValueError("temperature must be > 0")
    z = np.asarray(logits, dtype=np.float64) / temperature
    return np.exp(z - logsumexp(z, axis=-1, keepdims=True))


def mean_nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    """Mean negative log-likelihood of the gold options at temperature ``T``."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    gold = z[np.arange(len(y)), np.asarray(y)]
    return float((logsumexp(z, axis=-1) - gold).mean())


def fit_temperature(
    logits: np.ndarray, y: np.ndarray, t_min: float = T_MIN, t_max: float = T_MAX
) -> float:
    """The ``T`` in ``[t_min, t_max]`` that minimises mean NLL.

    NLL is convex in ``1/T``, so it is unimodal in ``log T`` and a bounded scalar search finds the
    global minimum. A perfectly separable calibration set pushes ``T`` down until the NLL underflows
    to zero (around 0.2 for logit gaps of 10); any smaller ``T`` gives the same predictions.
    """
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y)
    if len(y) == 0:
        raise ValueError("no calibration items")
    res = minimize_scalar(
        lambda lt: mean_nll(logits, y, math.exp(lt)),
        bounds=(math.log(t_min), math.log(t_max)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(math.exp(res.x))


@dataclass
class TemperatureScaler:
    """One temperature per answer type, fitted on a calibration split.

    ``fit`` takes ``{type: (logits, y)}``; types with fewer than ``min_rows`` items fall back to the
    temperature fitted on all items pooled (as decider does with its single "temperature").
    """

    min_rows: int = 50
    temperatures: dict[str, float] = field(default_factory=dict)
    pooled: float = 1.0

    def fit(self, data: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> TemperatureScaler:
        self.temperatures = {}
        pooled_z, pooled_y = [], []
        for t, (z, y) in data.items():
            z = np.asarray(z, dtype=np.float64)
            if len(y) >= self.min_rows:
                self.temperatures[t] = fit_temperature(z, y)
            pooled_z.append(_pad_to(z, max(np.asarray(zz).shape[1] for zz, _ in data.values())))
            pooled_y.append(np.asarray(y))
        self.pooled = fit_temperature(np.concatenate(pooled_z), np.concatenate(pooled_y))
        return self

    def temperature(self, answer_type: str) -> float:
        return self.temperatures.get(answer_type, self.pooled)

    def transform(self, logits: np.ndarray, answer_type: str) -> np.ndarray:
        return apply_temperature(logits, self.temperature(answer_type))

    def to_config(self) -> dict[str, object]:
        """The shape of decider's ``decider_config.json`` temperature entries."""
        return {
            "temperature_by_type": {k: round(v, 3) for k, v in self.temperatures.items()},
            "temperature": round(self.pooled, 3),
        }


def _pad_to(z: np.ndarray, k: int) -> np.ndarray:
    if z.shape[1] == k:
        return z
    out = np.full((z.shape[0], k), -np.inf)
    out[:, : z.shape[1]] = z
    return out
