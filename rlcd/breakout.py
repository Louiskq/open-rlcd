"""Exp B, the parts that need no GPU: belief questions and their labels, returns and advantages, belief metrics,
temperature scaling for yes/no beliefs, checkpoint selection and the game-score margin of the H7 rule.

The Breakout state is the four RAM readings of ``decider.games.envs.Breakout._s()``: ``ball_x``, ``ball_y``,
``paddle_x`` and ``lives`` (``ball_y == 0`` means no ball in play). A belief question is asked about the
situation at step ``t`` and resolved from what happens next. ``BeliefTracker`` is the step-by-step labeller that
the earlier results used; ``docs/breakout_label_validation.md`` checked it against an independent re-derivation
(0 disagreements in 10,942 resolved questions).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rlcd import scoring as S
from rlcd.calibrate import fit_temperature
from rlcd.data import SeedSets

BELIEFS = {
    "return": "Will the paddle return the ball the next time the ball reaches the bottom?",
    "brick30": "Will at least one brick be broken within the next 30 moves?",
    "life30": "Will a life be lost within the next 30 moves?",
}
PADDLE_Y = 160  # a down -> up flip of the ball at or below this y is a paddle hit (bricks flip it at y ~ 72-87)
HORIZON = 30
BELIEF_EVERY = 4
GAMMA = 0.97

# Seeds (docs/decisions.md, PREREGISTRATION.md): training env seeds come from a private range per training seed.
SELECT_SEEDS = tuple(range(1000, 1050))
EVAL_SEEDS = tuple(range(2000, 2050))
LABEL_CHECK_SEEDS = tuple(range(3000, 3160))
TRAIN_SEED_BLOCK = 100_000


def train_env_seed(run_seed: int, counter: int) -> int:
    """Env seed of the ``counter``-th training episode of training seed ``run_seed``: disjoint from every other set."""
    return TRAIN_SEED_BLOCK * (run_seed + 1) + counter


def seed_sets(run_seed: int, max_episodes: int = 20_000) -> SeedSets:
    """The disjointness check (``SeedSets`` raises ``LeakError`` on overlap), including the label-check seeds."""
    train = tuple(train_env_seed(run_seed, i) for i in range(max_episodes))
    SeedSets(train, SELECT_SEEDS, EVAL_SEEDS + LABEL_CHECK_SEEDS)
    return SeedSets(train, SELECT_SEEDS, EVAL_SEEDS)


# ---------------------------------------------------------------------------------------------
# belief questions
# ---------------------------------------------------------------------------------------------


@dataclass
class _Pending:
    kind: str
    t: int
    ctx: str
    score: float
    lives: int


@dataclass
class BeliefTracker:
    """Asks a belief question every ``every`` steps and resolves each from what happens next (one per episode).

    Call ``maybe_ask(k, state, score, ctx)`` before the action of decision ``k`` and ``observe(k_after, state,
    score, done)`` after the environment step, where ``k_after`` is the number of decisions taken so far. Questions
    still open at truncation are dropped (``resolved`` contains only questions whose outcome is known).
    """

    rng: random.Random
    every: int = BELIEF_EVERY
    pending: list[_Pending] = field(default_factory=list)
    resolved: list[dict[str, Any]] = field(default_factory=list)
    _prev: dict[str, int] | None = None
    _vy: int = 0

    def start(self, state: dict[str, int]) -> None:
        self._prev, self._vy = dict(state), 0

    def maybe_ask(self, k: int, state: dict[str, int], score: float, ctx: str) -> str | None:
        if k % self.every:
            return None
        kinds = ["brick30", "life30"] + (["return"] if state["ball_y"] > 0 else [])
        kind = self.rng.choice(kinds)
        self.pending.append(_Pending(kind, k, ctx, score, state["lives"]))
        return kind

    def observe(self, k_after: int, state: dict[str, int], score: float, done: bool) -> None:
        prev = self._prev
        assert prev is not None, "call start() first"
        vy = state["ball_y"] - prev["ball_y"]
        # a paddle return: the ball turns from down to up near the bottom AND stays in play with no life lost; when the
        # ball is lost its y resets to 0, which looks like an upward turn, but that is a miss
        hit = (
            self._vy > 0
            and vy < 0
            and prev["ball_y"] >= PADDLE_Y
            and state["ball_y"] > 0
            and state["lives"] == prev["lives"]
        )
        still = []
        for b in self.pending:
            out: bool | None
            if b.kind == "return":
                out = True if hit else (False if state["lives"] < b.lives or done else None)
            elif b.kind == "brick30":
                out = (
                    True
                    if score > b.score
                    else (False if (k_after - b.t >= HORIZON or done) else None)
                )
            else:  # life30
                out = (
                    True
                    if state["lives"] < b.lives
                    else (
                        False
                        if (k_after - b.t >= HORIZON or (done and state["lives"] >= b.lives))
                        else None
                    )
                )
            if out is None:
                still.append(b)
            else:
                self.resolved.append({"kind": b.kind, "t": b.t, "ctx": b.ctx, "gold": int(out)})
        self.pending = still
        self._prev = dict(state)
        self._vy = vy if vy != 0 else self._vy


# ---------------------------------------------------------------------------------------------
# returns and advantages (PPO actor term)
# ---------------------------------------------------------------------------------------------


def returns(rewards: Sequence[float], gamma: float = GAMMA) -> list[float]:
    g, out = 0.0, []
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return out[::-1]


def step_advantages(all_returns: Sequence[Sequence[float]]) -> list[list[float]]:
    """Per-step baselines (the mean return at the same step over episodes), scaled by the overall s.d.

    This is the plain-PPO advantage of ``decider.games.rl``: one value per decision; at each step the baseline uses
    the episodes still running.
    """
    t_max = max(len(g) for g in all_returns)
    base = [float(np.mean([g[t] for g in all_returns if t < len(g)])) for t in range(t_max)]
    flat = [g[t] - base[t] for g in all_returns for t in range(len(g))]
    sd = float(np.std(flat)) + 1e-6
    return [[(g[t] - base[t]) / sd for t in range(len(g))] for g in all_returns]


# ---------------------------------------------------------------------------------------------
# belief metrics
# ---------------------------------------------------------------------------------------------


def p_yes_from_logit_diff(d: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(d, dtype=np.float64) / temperature))


def belief_metrics(p_yes: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    """Log loss, Brier, top-label ECE (15 equal-width bins as in Exp A), accuracy and base rate of yes/no beliefs."""
    p = np.clip(np.asarray(p_yes, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(gold).astype(int)
    probs = np.stack([1.0 - p, p], axis=1)
    return {
        "n": int(len(y)),
        "log_loss": S.log_loss(probs, y),
        "brier": float(S.brier(probs, y)),
        "ece": S.top_label_ece(probs, y, 15, "width"),
        "acc": float((probs.argmax(1) == y).mean()),
        "base_rate": float(y.mean()),
    }


def base_rate_log_loss(gold: np.ndarray) -> float:
    """Log loss of always predicting the empirical base rate (the skill reference)."""
    y = np.asarray(gold).astype(int)
    q = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    return float(-(y * math.log(q) + (1 - y) * math.log(1 - q)).mean())


def fit_belief_temperature(logit_diff: np.ndarray, gold: np.ndarray) -> float:
    """One temperature for yes/no beliefs: minimises log loss of ``sigmoid(d / T)`` on held-out rollouts."""
    d = np.asarray(logit_diff, dtype=np.float64)
    return fit_temperature(np.stack([np.zeros_like(d), d], axis=1), np.asarray(gold).astype(int))


# ---------------------------------------------------------------------------------------------
# checkpoint selection and the H7 game-score margin
# ---------------------------------------------------------------------------------------------


def select_iteration(evals: Sequence[tuple[int, float]]) -> int:
    """Iteration of the highest mean greedy score on the selection seeds; ties keep the earlier one.

    ``evals`` is ``[(iteration, mean score), ...]`` with iteration 0 the start model, which is chosen when training
    never beats it.
    """
    best_it, best = evals[0]
    for it, sc in evals[1:]:
        if sc > best:
            best_it, best = it, sc
    return int(best_it)


def delta_game(seed_scores: Sequence[float], floor_fraction: float = 0.05) -> dict[str, float]:
    """H7 rule (b): the range of B1's seeds' mean greedy selection-seed scores, floored at 5% of their mean."""
    v = np.asarray(seed_scores, dtype=float)
    rng_ = float(v.max() - v.min())
    floor = floor_fraction * float(v.mean())
    return {
        "range": rng_,
        "floor": floor,
        "delta_game": max(rng_, floor),
        "mean": float(v.mean()),
        "n": int(v.size),
    }


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, B: int = 2000, seed: int = 20261008
) -> dict[str, Any]:
    """Mean of ``a - b`` per item, with a 95% percentile interval and the share of replicates with ``a - b >= 0``.

    ``a`` and ``b`` are per-item losses of two arms on the same items (shape ``(n,)``, or ``(seeds, n)`` for
    seed-averaged arms: items are resampled once and every seed scored on them, as in ``rlcd.compare``).
    """
    a2, b2 = np.atleast_2d(a), np.atleast_2d(b)
    d = a2.mean(axis=0) - b2.mean(axis=0)
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        boots[i] = d[idx].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "mean_diff": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "share_not_below_zero": float((boots >= 0).mean()),
        "p_a_lower": float(min(1.0, 2.0 * (boots >= 0).mean())),
        "n": int(n),
    }


def item_log_loss(p_yes: np.ndarray, gold: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p_yes, dtype=np.float64), 1e-12, 1 - 1e-12)
    y = np.asarray(gold).astype(int)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def by_kind(
    fn: Callable[[np.ndarray, np.ndarray], Any], p: np.ndarray, gold: np.ndarray, kinds: np.ndarray
) -> dict[str, Any]:
    return {k: fn(p[kinds == k], gold[kinds == k]) for k in BELIEFS if np.any(kinds == k)}
