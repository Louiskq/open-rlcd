"""Proper scoring rules, calibration metrics and selective-prediction metrics for typed answers.

Everything here works on a batch: ``probs`` is ``[N, K]`` (each row a distribution over K options,
zero-padded when questions have different numbers of options) and ``y`` is ``[N]`` integer gold
indices. For ordinal SCORE questions the options are ordered levels ``0 .. K-1``.

Conventions follow ``decider/metrics.py`` (Mapika, Apache 2.0) so numbers are comparable with the
start model's own evaluation: Brier is the *sum* over options, ECE uses 15 equal-width bins with
``(lo, hi]`` edges, probabilities are clipped at 1e-12 inside logs. Differences are noted inline.

Scores are *higher is better* when called ``*_score`` (they are rewards) and *lower is better* when
called ``*_loss`` / ``brier`` / ``rps`` (they are losses).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EPS = 1e-12
ANSWER_TYPES = ("noul", "choice", "score")


# ---------------------------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------------------------


def as_probs(probs: np.ndarray, check: bool = True) -> np.ndarray:
    """Return ``probs`` as a float64 ``[N, K]`` array; raise if it is not distributions."""
    p = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    if p.ndim != 2:
        raise ValueError(f"probs must be [N, K], got shape {p.shape}")
    if check:
        if not np.all(np.isfinite(p)) or p.min() < -1e-9:
            raise ValueError("probs contain NaN/inf or negative entries")
        if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("each row of probs must sum to 1")
    return p


def as_labels(y: np.ndarray, k: int) -> np.ndarray:
    """Return ``y`` as an int array of gold indices, checked against ``k`` options."""
    y = np.asarray(y)
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y must be a 1-D integer array of gold option indices")
    if y.size and (y.min() < 0 or y.max() >= k):
        raise ValueError(f"gold index out of range for {k} options")
    return y


def pad_probs(rows: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Stack ragged distributions into a zero-padded ``[N, K_max]`` array and the true K of each."""
    n_options = np.array([len(r) for r in rows])
    out = np.zeros((len(rows), int(n_options.max())))
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out, n_options


# ---------------------------------------------------------------------------------------------
# proper scoring rules (and the analytic gradient of each w.r.t. the probability vector)
# ---------------------------------------------------------------------------------------------


def log_score(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-item log score ``log p_y`` (<= 0, higher is better): the reward for NOUL / CHOICE."""
    p = as_probs(probs)
    y = as_labels(y, p.shape[1])
    return np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))


def log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log score (NLL)."""
    return float(-log_score(probs, y).mean())


def brier_per_item(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-item multiclass Brier score ``sum_k (p_k - 1[y=k])^2`` (range 0..2)."""
    p = as_probs(probs)
    y = as_labels(y, p.shape[1])
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return ((p - onehot) ** 2).sum(axis=1)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean multiclass Brier score (sum over options, as in decider/metrics.py)."""
    return float(brier_per_item(probs, y).mean())


def _n_options(p: np.ndarray, n_options: np.ndarray | None) -> np.ndarray:
    if n_options is None:
        return np.full(len(p), p.shape[1])
    n = np.asarray(n_options)
    if n.shape != (len(p),) or n.min() < 2 or n.max() > p.shape[1]:
        raise ValueError("n_options must be [N] with 2 <= n_options <= K")
    return n


def rps_per_item(
    probs: np.ndarray,
    y: np.ndarray,
    n_options: np.ndarray | None = None,
    normalise: bool = True,
) -> np.ndarray:
    """Per-item ranked probability score of an ordinal distribution (lower is better).

    ``RPS = sum_{k=0}^{K-2} (P_k - O_k)^2`` with ``P`` the predicted CDF and ``O_k = 1[y <= k]``.
    With ``normalise`` (default) it is divided by ``K - 1`` so it lies in [0, 1] whatever the number
    of levels. ``n_options`` gives each item's true K when rows are zero-padded; padding adds zero.
    For K = 2 and ``normalise`` it equals the binary Brier score.
    """
    p = as_probs(probs)
    y = as_labels(y, p.shape[1])
    k = _n_options(p, n_options)
    cdf = np.cumsum(p, axis=1)[:, :-1]
    levels = np.arange(p.shape[1] - 1)
    obs = (levels[None, :] >= y[:, None]).astype(np.float64)
    r = ((cdf - obs) ** 2).sum(axis=1)
    return r / (k - 1) if normalise else r


def rps(
    probs: np.ndarray,
    y: np.ndarray,
    n_options: np.ndarray | None = None,
    normalise: bool = True,
) -> float:
    """Mean ranked probability score."""
    return float(rps_per_item(probs, y, n_options, normalise).mean())


def expected_level(probs: np.ndarray) -> np.ndarray:
    """Expected level ``sum_k k p_k`` of each ordinal distribution."""
    p = as_probs(probs)
    return p @ np.arange(p.shape[1])


def level_mae(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean absolute error between the expected level and the gold level (SCORE questions)."""
    p = as_probs(probs)
    y = as_labels(y, p.shape[1])
    return float(np.abs(expected_level(p) - y).mean())


def log_score_grad_probs(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``d log p_y / d p``: ``1/p_y`` at the gold option, 0 elsewhere (clipped like the score)."""
    p = as_probs(probs)
    y = as_labels(y, p.shape[1])
    g = np.zeros_like(p)
    idx = np.arange(len(y))
    g[idx, y] = 1.0 / np.clip(p[idx, y], EPS, 1.0)
    return g


def rps_score_grad_probs(
    probs: np.ndarray,
    y: np.ndarray,
    n_options: np.ndarray | None = None,
    normalise: bool = True,
) -> np.ndarray:
    """``d (-RPS) / d p`` for each item: the gradient of the reward used for SCORE questions."""
    p = as_probs(probs)
    y = as_labels(y, p.shape[1])
    k = _n_options(p, n_options)
    cdf = np.cumsum(p, axis=1)[:, :-1]
    levels = np.arange(p.shape[1] - 1)
    obs = (levels[None, :] >= y[:, None]).astype(np.float64)
    d = 2.0 * (cdf - obs)  # d RPS_unnormalised / d P_k
    # P_k = p_0 + ... + p_k, so d/dp_j sums d_k over k >= j (the last option has no CDF term)
    g = np.zeros_like(p)
    g[:, :-1] = np.cumsum(d[:, ::-1], axis=1)[:, ::-1]
    if normalise:
        g = g / (k - 1)[:, None]
    return -g


# ---------------------------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------------------------


def _width_bins(values: np.ndarray, bins: int) -> np.ndarray:
    """Bin index of each value in ``(lo, hi]`` equal-width bins over [0, 1] (0 falls in bin 0).

    The 1e-9 slack keeps a value that sits exactly on an edge (e.g. 0.2 with 15 bins, where
    ``0.2 * 15`` is 3.0000000000000004 in floating point) in the lower bin, as ``(lo, hi]`` says.
    """
    return np.clip(np.ceil(np.asarray(values) * bins - 1e-9).astype(int) - 1, 0, bins - 1)


def _mass_bins(values: np.ndarray, bins: int) -> np.ndarray:
    """Bin index when sorting by value and cutting into ``bins`` groups of (near-)equal size."""
    order = np.argsort(values, kind="stable")
    idx = np.empty(len(values), dtype=int)
    for b, chunk in enumerate(np.array_split(order, bins)):
        idx[chunk] = b
    return idx


def _bin_ids(conf: np.ndarray, bins: int, binning: str) -> np.ndarray:
    if binning == "width":
        return _width_bins(conf, bins)
    if binning == "mass":
        return _mass_bins(conf, bins)
    raise ValueError("binning must be 'width' or 'mass'")


def reliability_bins(
    conf: np.ndarray, correct: np.ndarray, bins: int = 15, binning: str = "width"
) -> dict[str, np.ndarray]:
    """Per-bin count, mean confidence and accuracy (NaN if empty): reliability-diagram data."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    ids = _bin_ids(conf, bins, binning)
    count = np.bincount(ids, minlength=bins).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.bincount(ids, weights=conf, minlength=bins) / count
        acc = np.bincount(ids, weights=correct, minlength=bins) / count
    return {"count": count, "conf": mean_conf, "acc": acc}


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15, binning: str = "width") -> float:
    """Expected calibration error ``sum_b (n_b / N) |acc_b - conf_b|``.

    ``binning='width'`` is the usual 15 equal-width bins; ``'mass'`` uses equal-count bins as a
    robustness check (it is less sensitive to where the confidences happen to pile up).
    """
    r = reliability_bins(conf, correct, bins, binning)
    n = r["count"].sum()
    mask = r["count"] > 0
    return float((r["count"][mask] / n * np.abs(r["acc"][mask] - r["conf"][mask])).sum())


def top_label_ece(
    probs: np.ndarray, y: np.ndarray, bins: int = 15, binning: str = "width"
) -> float:
    """Top-label ECE: confidence is the largest probability; correct means argmax equals gold."""
    p = as_probs(probs)
    y = as_labels(y, p.shape[1])
    return ece(p.max(axis=1), p.argmax(axis=1) == y, bins, binning)


def classwise_ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Classwise ECE: mean over options of the ECE of ``p_k`` against ``1[y = k]`` (CHOICE).

    Columns that are all zero (pure padding) are skipped. Group items by number of options first if
    a mix of K would make the zero padding matter.
    """
    p = as_probs(probs)
    y = as_labels(y, p.shape[1])
    n = len(y)
    errs = []
    for k in range(p.shape[1]):
        if not np.any(p[:, k] > 0):
            continue
        r = reliability_bins(p[:, k], y == k, bins, "width")
        mask = r["count"] > 0
        errs.append((r["count"][mask] / n * np.abs(r["acc"][mask] - r["conf"][mask])).sum())
    return float(np.mean(errs))


def grouped_ece(
    probs: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    bins: int = 15,
    binning: str = "mass",
) -> dict[str, float]:
    """Top-label ECE within each group (e.g. a dev domain), and the unweighted mean as ``mean``.

    Equal-mass bins by default (pre-registered H5b): groups can be small and uneven in how their
    confidences spread. Pooled ECE can hide opposite miscalibration in different groups; this
    cannot (Exp 0b).
    """
    p = as_probs(probs)
    y = as_labels(y, p.shape[1])
    g = np.asarray(groups)
    if g.shape != y.shape:
        raise ValueError("groups must have one entry per item")
    out = {
        str(name): top_label_ece(p[g == name], y[g == name], bins, binning) for name in np.unique(g)
    }
    out["mean"] = float(np.mean(list(out.values())))
    return out


def murphy_decomposition(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> dict[str, float]:
    """Murphy decomposition of the multiclass Brier score (sum over options).

    Each option k is treated as a binary forecast ``p_k`` of the event ``y = k`` and binned on
    ``p_k`` (equal width). Summed over options::

        brier = reliability - resolution + uncertainty + within_bin

    * ``reliability``: calibration error, lower is better (0 for a calibrated forecaster).
    * ``resolution``: how far the per-bin outcome rates are from the base rates, higher is better
      (sharpness that is justified by the outcomes).
    * ``uncertainty``: ``sum_k base_k (1 - base_k)``, a property of the data alone.
    * ``within_bin``: the residual that appears when forecasts inside a bin are not identical
      (Bröcker 2009). It shrinks as bins get finer and is exactly 0 for forecasts that take one
      value per bin, so the identity above is exact for any binning.
    """
    p = as_probs(probs)
    y = as_labels(y, p.shape[1])
    n, k = p.shape
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    rel = res = unc = wbv = 0.0
    for j in range(k):
        pj, oj = p[:, j], onehot[:, j]
        ids = _width_bins(pj, bins)
        cnt = np.bincount(ids, minlength=bins).astype(np.float64)
        full = cnt > 0
        pbar = np.zeros(bins)
        obar = np.zeros(bins)
        pbar[full] = np.bincount(ids, weights=pj, minlength=bins)[full] / cnt[full]
        obar[full] = np.bincount(ids, weights=oj, minlength=bins)[full] / cnt[full]
        base = oj.mean()
        rel += float((cnt * (pbar - obar) ** 2).sum() / n)
        res += float((cnt * (obar - base) ** 2).sum() / n)
        unc += float(base * (1.0 - base))
        dp, do = pj - pbar[ids], oj - obar[ids]
        wbv += float((dp**2 - 2.0 * dp * do).sum() / n)
    return {
        "brier": brier(p, y),
        "reliability": rel,
        "resolution": res,
        "uncertainty": unc,
        "within_bin": wbv,
    }


# ---------------------------------------------------------------------------------------------
# selective prediction: risk-coverage and AURC
# ---------------------------------------------------------------------------------------------


def _tie_averaged_errors(conf: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Per-item error indicator in descending-confidence order, averaged inside groups of tied
    confidences. Averaging gives the expected curve under random tie-breaking, so the result does
    not depend on the arbitrary order of equal confidences (which matters for coarse forecasts)."""
    conf = np.asarray(conf, dtype=np.float64)
    err = 1.0 - np.asarray(correct, dtype=np.float64)
    order = np.argsort(-conf, kind="stable")
    c, e = conf[order], err[order]
    _, start, counts = np.unique(-c, return_index=True, return_counts=True)
    group_mean = np.add.reduceat(e, start) / counts
    return np.repeat(group_mean, counts)


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve: ``coverage[i] = (i+1)/N`` and ``risk[i]`` = error rate among the ``i+1``
    most confident answers (ties averaged, see ``_tie_averaged_errors``)."""
    e = _tie_averaged_errors(conf, correct)
    n = len(e)
    return np.arange(1, n + 1) / n, np.cumsum(e) / np.arange(1, n + 1)


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better). Equals decider's ``aurc`` when no two
    confidences tie."""
    return float(risk_coverage(conf, correct)[1].mean())


def selective_accuracy(conf: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Accuracy when answering only the ``coverage`` fraction of items the model is most sure of."""
    e = _tie_averaged_errors(conf, correct)
    m = max(1, int(round(coverage * len(e))))
    return float(1.0 - e[:m].mean())


# ---------------------------------------------------------------------------------------------
# one-call summaries
# ---------------------------------------------------------------------------------------------


def summarize(
    probs: np.ndarray,
    y: np.ndarray,
    answer_type: str = "choice",
    n_options: np.ndarray | None = None,
    bins: int = 15,
) -> dict[str, float]:
    """All headline metrics for one batch of one answer type.

    Primary: ``log_loss``, ``brier`` and its Murphy parts. Secondary: ``acc``, ``ece`` (equal
    width), ``ece_mass``, ``classwise_ece`` (CHOICE only). Decision: ``aurc``, ``acc_at_80``,
    ``acc_at_50``. SCORE adds ``rps`` and ``level_mae``.
    """
    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"answer_type must be one of {ANSWER_TYPES}")
    p = as_probs(probs)
    y = as_labels(y, p.shape[1])
    conf = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    out: dict[str, float] = {
        "n": int(len(y)),
        "acc": float(correct.mean()),
        "log_loss": log_loss(p, y),
        "ece": ece(conf, correct, bins, "width"),
        "ece_mass": ece(conf, correct, bins, "mass"),
        "aurc": aurc(conf, correct),
        "acc_at_80": selective_accuracy(conf, correct, 0.8),
        "acc_at_50": selective_accuracy(conf, correct, 0.5),
        "mean_conf": float(conf.mean()),
    }
    out.update(murphy_decomposition(p, y, bins))
    if answer_type == "choice":
        out["classwise_ece"] = classwise_ece(p, y, bins)
    if answer_type == "score":
        out["rps"] = rps(p, y, n_options)
        out["level_mae"] = level_mae(p, y)
    return out


def summarize_by_type(
    rows: Sequence[Sequence[float]], y: Sequence[int], types: Sequence[str]
) -> dict[str, dict[str, float]]:
    """``summarize`` per answer type for ragged data: ``rows[i]`` is item i's distribution."""
    types_arr = np.asarray(types)
    y_arr = np.asarray(y, dtype=int)
    out: dict[str, dict[str, float]] = {}
    for t in ANSWER_TYPES:
        idx = np.flatnonzero(types_arr == t)
        if idx.size == 0:
            continue
        p, n_opt = pad_probs([rows[i] for i in idx])
        out[t] = summarize(p, y_arr[idx], t, n_opt)
    return out
