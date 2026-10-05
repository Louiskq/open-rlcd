"""Exp 0: a contextual bandit with known true probabilities.

A linear-softmax policy ``p_theta(. | x) = softmax(theta x)`` over 5 options is trained on labels
``y ~ q(. | x) = softmax(W* x)`` where ``W*`` is fixed. Because the policy class contains the truth,
"calibrated" has an exact meaning here (``p = q``) and every metric can be checked against it.

Arms (all start from the uniform policy, use Adam with the same learning rate, and see the same
stream of contexts and labels for a given seed; only the sampled answers differ):

``supervised``          log loss on the label (the analytic gradient ``e_y - p``)
``rlcd_full``           the same objective through the RL path: reward ``S(p, y)``, chain rule
``rlcd_bandit``         sampled answers, reward ``1[a=y] / p(a)``, unclipped
``rlcd_bandit_clip10``  the same with the weight clipped at 10
``rlvr``                sampled answers, reward ``1[a=y]`` (improper)

Seeds are split three ways and checked by ``SeedSets``: training seeds draw the training stream and
the actions; one fixed calibration seed draws the set temperature scaling is fitted on; one fixed
evaluation seed draws the set every number is reported on. ``W*`` has its own seed.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace

import numpy as np

from rlcd import calibrate as C
from rlcd import estimators as E
from rlcd import scoring as S
from rlcd.data import SeedSets

GROUND_TRUTH_SEED = 0
CAL_SEED = 7000
EVAL_SEED = 9000
TRAIN_SEEDS = tuple(range(100, 110))
SEEDS = SeedSets(train=TRAIN_SEEDS, select=(CAL_SEED,), evaluate=(EVAL_SEED,))
BINS = 15

ARMS: dict[str, E.Arm] = {
    "supervised": E.Arm("supervised", "supervised"),
    "rlcd_full": E.Arm("rlcd_full", "rlcd_full"),
    "rlcd_bandit": E.Arm("rlcd_bandit", "rlcd_bandit"),
    "rlcd_bandit_clip10": E.Arm("rlcd_bandit_clip10", "rlcd_bandit", clip=10.0),
    "rlvr": E.Arm("rlvr", "rlvr"),
}


@dataclass(frozen=True)
class ToyConfig:
    """Everything that defines a run besides the arm and the seed."""

    dim: int = 8
    n_options: int = 5
    logit_scale: float = 1.6  # std of the ground-truth logits of each option
    steps: int = 4000
    batch: int = 64
    lr: float = 0.01  # Adam, identical for every arm
    n_checkpoints: int = 48  # evaluation points, geometrically spaced: the dynamics are early
    n_eval: int = 20_000
    n_cal: int = 20_000
    seeds: tuple[int, ...] = TRAIN_SEEDS

    def __post_init__(self) -> None:
        SeedSets(train=self.seeds, select=(CAL_SEED,), evaluate=(EVAL_SEED,))  # raises on overlap


QUICK = {"steps": 1500, "n_eval": 5000, "n_cal": 5000, "seeds": TRAIN_SEEDS[:3]}


def checkpoint_steps(steps: int, n: int) -> list[int]:
    """Step 0 plus about ``n - 1`` geometrically spaced steps ending at ``steps``."""
    grid = np.unique(np.round(np.geomspace(1, steps, n - 1)).astype(int))
    return [0, *map(int, grid)]


# ---------------------------------------------------------------------------------------------
# the problem
# ---------------------------------------------------------------------------------------------


def ground_truth(cfg: ToyConfig) -> np.ndarray:
    """The fixed ``W*`` ([K, d]); entries scaled so each option's logit has std ``logit_scale``."""
    rng = np.random.default_rng(GROUND_TRUTH_SEED)
    return rng.normal(size=(cfg.n_options, cfg.dim)) * cfg.logit_scale / math.sqrt(cfg.dim)


@dataclass
class Dataset:
    """Contexts, sampled labels and the true distributions ``q`` behind them."""

    x: np.ndarray
    y: np.ndarray
    q: np.ndarray


def make_dataset(w_true: np.ndarray, n: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, w_true.shape[1]))
    q = E.softmax(x @ w_true.T)
    return Dataset(x, E.sample_categorical(q, 1, rng)[:, 0], q)


def reference_metrics(data: Dataset) -> dict[str, float]:
    """What a perfect forecaster (``p = q``) scores: the noise floor for every metric."""
    m = score_logits(np.log(data.q), data)
    m["bayes_acc"] = float((data.q.argmax(axis=1) == data.y).mean())
    m["entropy"] = float(-(data.q * np.log(data.q)).sum(axis=1).mean())
    m["frac_options_below_0.1"] = float((data.q < 0.1).mean())
    return m


# ---------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------


def score_logits(z: np.ndarray, data: Dataset, full: bool = False) -> dict[str, float]:
    """Metrics of the policy ``softmax(z)`` on a dataset, including the two that need the truth:
    ``kl_to_truth`` = mean KL(q || p) and ``true_gap`` = mean |confidence - q(predicted option)|."""
    p = E.softmax(z)
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    correct = pred == data.y
    out = {
        "acc": float(correct.mean()),
        "ece": S.ece(conf, correct, BINS),
        "log_loss": S.log_loss(p, data.y),
        "mean_conf": float(conf.mean()),
        "kl_to_truth": float((data.q * (np.log(data.q) - E.log_softmax(z))).sum(axis=1).mean()),
        "true_gap": float(np.abs(conf - data.q[np.arange(len(pred)), pred]).mean()),
    }
    if full:
        out.update(
            {k: v for k, v in S.summarize(p, data.y, "choice", bins=BINS).items() if k not in out}
        )
    return out


def reliability(z: np.ndarray, data: Dataset) -> dict[str, np.ndarray]:
    """Per-bin count, confidence sum and correct count (sums, so they can be pooled over seeds)."""
    p = E.softmax(z)
    r = S.reliability_bins(p.max(axis=1), p.argmax(axis=1) == data.y, BINS)
    count = r["count"]
    return {
        "count": count,
        "conf_sum": np.nan_to_num(r["conf"]) * count,
        "correct_sum": np.nan_to_num(r["acc"]) * count,
    }


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------


class Adam:
    """Plain Adam on a single array (no weight decay)."""

    def __init__(self, shape: tuple[int, ...], lr: float, b1: float = 0.9, b2: float = 0.999):
        self.lr, self.b1, self.b2, self.t = lr, b1, b2, 0
        self.m, self.v = np.zeros(shape), np.zeros(shape)

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """One descent step on ``grad``, the gradient of the loss (minus the reward gradient)."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        mhat, vhat = self.m / (1 - self.b1**self.t), self.v / (1 - self.b2**self.t)
        return theta - self.lr * mhat / (np.sqrt(vhat) + 1e-8)


@dataclass
class Run:
    """One (arm, seed) training run: curves at every checkpoint plus final raw / scaled metrics."""

    arm: str
    seed: int
    steps: list[int]
    curves: dict[str, list[float]]
    reliability: dict[str, np.ndarray]  # [n_checkpoints, BINS] each
    final_raw: dict[str, float]
    final_ts: dict[str, float]
    temperature: float
    final_reliability_raw: dict[str, np.ndarray]
    final_reliability_ts: dict[str, np.ndarray]


def train(
    arm: E.Arm,
    cfg: ToyConfig,
    w_true: np.ndarray,
    seed: int,
    eval_set: Dataset,
    cal_set: Dataset,
) -> Run:
    """Train one arm for ``cfg.steps`` Adam steps from the uniform policy, evaluating on the way."""
    data_rng = np.random.default_rng([seed, 0])  # contexts + labels: shared by all arms
    act_rng = np.random.default_rng([seed, 1])  # sampled answers: only the RL arms use it
    theta = np.zeros((cfg.n_options, cfg.dim))
    opt = Adam(theta.shape, cfg.lr)
    steps: list[int] = []
    curves: dict[str, list[float]] = {}
    rel: dict[str, list[np.ndarray]] = {"count": [], "conf_sum": [], "correct_sum": []}

    def checkpoint(step: int) -> None:
        z = eval_set.x @ theta.T
        for k, v in score_logits(z, eval_set).items():
            curves.setdefault(k, []).append(v)
        for k, v in reliability(z, eval_set).items():
            rel[k].append(v)
        steps.append(step)

    marks = set(checkpoint_steps(cfg.steps, cfg.n_checkpoints))
    checkpoint(0)
    for step in range(1, cfg.steps + 1):
        x = data_rng.normal(size=(cfg.batch, cfg.dim))
        y = E.sample_categorical(E.softmax(x @ w_true.T), 1, data_rng)[:, 0]
        g_z = E.logit_gradient(arm, x @ theta.T, y, act_rng)  # ascent direction on the reward
        theta = opt.step(theta, -(g_z.T @ x) / cfg.batch)
        if step in marks:
            checkpoint(step)

    z_eval, z_cal = eval_set.x @ theta.T, cal_set.x @ theta.T
    temperature = C.fit_temperature(z_cal, cal_set.y)  # fitted on the calibration set only
    z_ts = z_eval / temperature
    return Run(
        arm=arm.name,
        seed=seed,
        steps=steps,
        curves=curves,
        reliability={k: np.stack(v) for k, v in rel.items()},
        final_raw=score_logits(z_eval, eval_set, full=True),
        final_ts=score_logits(z_ts, eval_set, full=True),
        temperature=temperature,
        final_reliability_raw=reliability(z_eval, eval_set),
        final_reliability_ts=reliability(z_ts, eval_set),
    )


def run_arm(
    arm: E.Arm, cfg: ToyConfig, verbose: bool = True, w_true=None, eval_set=None, cal_set=None
) -> list[Run]:
    """All seeds of one arm."""
    w_true = ground_truth(cfg) if w_true is None else w_true
    eval_set = eval_set or make_dataset(w_true, cfg.n_eval, EVAL_SEED)
    cal_set = cal_set or make_dataset(w_true, cfg.n_cal, CAL_SEED)
    runs = []
    t0 = time.time()
    for seed in cfg.seeds:
        runs.append(train(arm, cfg, w_true, seed, eval_set, cal_set))
    if verbose:
        ece = np.mean([r.final_raw["ece"] for r in runs])
        print(f"  {arm.name:<20} {len(runs)} seeds  final ECE {ece:.4f}  ({time.time() - t0:.1f}s)")
    return runs


def mix_sweep(
    cfg: ToyConfig, mixes: tuple[float, ...], clip: float | None = None, verbose: bool = True
) -> dict[float, list[Run]]:
    """RLCD-bandit with a share ``m`` of the RLVR reward mixed in, for each ``m`` in ``mixes``."""
    w_true = ground_truth(cfg)
    eval_set = make_dataset(w_true, cfg.n_eval, EVAL_SEED)
    cal_set = make_dataset(w_true, cfg.n_cal, CAL_SEED)
    out = {}
    for m in mixes:
        arm = E.Arm(f"mix_{m:g}", "rlcd_bandit", clip=clip, rlvr_mix=m)
        out[m] = run_arm(arm, cfg, verbose, w_true, eval_set, cal_set)
    return out


def with_lr(cfg: ToyConfig, lr: float) -> ToyConfig:
    return replace(cfg, lr=lr)


def config_dict(cfg: ToyConfig) -> dict[str, object]:
    d = asdict(cfg)
    d["seeds"] = list(cfg.seeds)
    return d


def gradient_noise(
    cfg: ToyConfig, n_batches: int = 3000, seed: int = EVAL_SEED + 1
) -> dict[str, dict[str, float]]:
    """Bias and noise of each arm's per-batch parameter gradient *at the truth* ``theta = W*``.

    At the truth a proper, unbiased objective has zero expected gradient (calibration is a fixed
    point), so ``bias_norm`` should sit at the Monte Carlo floor ``bias_se_norm``; a clipped or
    improper estimator pushes the model away from it. ``noise_trace`` is the trace of the covariance
    of the per-batch gradient: the quantity that sets how noisy training is. All arms see the same
    contexts and labels in each batch.
    """
    w_true = ground_truth(cfg)
    data_rng = np.random.default_rng(seed)
    act_rngs = {name: np.random.default_rng([seed, i]) for i, name in enumerate(ARMS)}
    grads: dict[str, list[np.ndarray]] = {name: [] for name in ARMS}
    for _ in range(n_batches):
        x = data_rng.normal(size=(cfg.batch, cfg.dim))
        z = x @ w_true.T
        y = E.sample_categorical(E.softmax(z), 1, data_rng)[:, 0]
        for name, arm in ARMS.items():
            g_z = E.logit_gradient(arm, z, y, act_rngs[name])
            grads[name].append((g_z.T @ x / cfg.batch).ravel())
    out = {}
    for name, g in grads.items():
        g = np.stack(g)
        se = g.std(axis=0, ddof=1) / math.sqrt(len(g))
        out[name] = {
            "bias_norm": float(np.linalg.norm(g.mean(axis=0))),
            "bias_se_norm": float(np.linalg.norm(se)),
            "noise_trace": float(g.var(axis=0, ddof=1).sum()),
        }
    return out
