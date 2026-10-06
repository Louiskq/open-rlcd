"""Offline analysis of saved Exp A predictions (``rlcd.gpu_eval.save_predictions``): numpy and scipy only.

A run's ``preds_<split>.npz`` holds the raw readout ``z`` of every question at temperature 1. Everything else is
computed here, so one forward pass serves every row of the results table:

* **raw**: T = 1 (``docs/decisions.md`` D2).
* **as served**: the start checkpoint's stored temperature (1.05 for ``decider9b-sft``), one for all types.
* **+TS**: one temperature per answer type fitted from scratch on the calibration split (never composed with the
  stored temperature), then applied to the evaluation split.

NOUL / CHOICE probabilities are ``softmax(z / T)``. SCORE probabilities are the normalised yes probabilities of the
level rows, ``y_k = sigmoid(l_k / T)`` (decider applies the temperature to each row's yes / no pair), ``p = y / sum y``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_expit, logsumexp

from rlcd import scoring as S

TYPE_NAMES = {0: "noul", 1: "choice", 2: "score"}
SERVED_TEMPERATURE = 1.05  # decider9b-sft/decider_config.json


def load_preds(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def _logp(z: np.ndarray, qtype: int, temperature: float) -> np.ndarray:
    """Log answer distribution for one question from its stored readout at the given temperature."""
    z = np.asarray(z, dtype=np.float64) / temperature
    if qtype == 2:
        logy = log_expit(z)
        return logy - logsumexp(logy)
    return z - logsumexp(z)


def probabilities(
    preds: dict[str, np.ndarray], temperature: float | dict[str, float] = 1.0
) -> list[np.ndarray]:
    """Answer distributions in original option order; ``temperature`` is one number or one per answer type."""
    out = []
    for i in range(len(preds["qid"])):
        qtype = int(preds["qtype"][i])
        t = temperature if isinstance(temperature, (int, float)) else temperature[TYPE_NAMES[qtype]]
        out.append(np.exp(_logp(preds["z"][i, : preds["n"][i]], qtype, t)))
    return out


def score_mass_deviation(preds: dict[str, np.ndarray]) -> float | None:
    """Mean |sum_k y_k - 1| over SCORE questions at T = 1 (D3): how far the level rows are from a distribution."""
    idx = np.flatnonzero(preds["qtype"] == 2)
    if idx.size == 0:
        return None
    masses = [expit(preds["z"][i, : preds["n"][i]].astype(np.float64)).sum() for i in idx]
    return float(np.mean(np.abs(np.asarray(masses) - 1.0)))


def mean_nll(preds: dict[str, np.ndarray], qtype: int, temperature: float) -> float:
    idx = np.flatnonzero(preds["qtype"] == qtype)
    return float(
        np.mean(
            [
                -_logp(preds["z"][i, : preds["n"][i]], qtype, temperature)[int(preds["gold"][i])]
                for i in idx
            ]
        )
    )


def fit_temperatures(
    preds_cal: dict[str, np.ndarray], t_min: float = 0.05, t_max: float = 20.0
) -> dict[str, float]:
    """One temperature per answer type minimising calibration-split NLL (convex in 1/T: a bounded search is exact)."""
    out = {}
    for code, name in TYPE_NAMES.items():
        if not np.any(preds_cal["qtype"] == code):
            continue
        res = minimize_scalar(
            lambda lt, c=code: mean_nll(preds_cal, c, math.exp(lt)),
            bounds=(math.log(t_min), math.log(t_max)),
            method="bounded",
            options={"xatol": 1e-6},
        )
        out[name] = float(math.exp(res.x))
    return out


def _padded(dists: list[np.ndarray], idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = [dists[i] for i in idx]
    n = np.array([len(p) for p in sub])
    mat = np.zeros((len(sub), int(n.max())))
    for r, p in enumerate(sub):
        mat[r, : len(p)] = p
    return mat, n


def summarize_split(
    preds: dict[str, np.ndarray],
    temperature: float | dict[str, float] = 1.0,
    domains: bool = True,
) -> dict[str, Any]:
    """Every metric by answer type, plus pooled (``all``) log loss, resolution and per-domain ECE (H5b)."""
    dists = probabilities(preds, temperature)
    gold = preds["gold"].astype(int)
    out: dict[str, Any] = {}
    for code, name in TYPE_NAMES.items():
        idx = np.flatnonzero(preds["qtype"] == code)
        if idx.size == 0:
            continue
        mat, n = _padded(dists, idx)
        out[name] = S.summarize(mat, gold[idx], name, n_options=n if name == "score" else None)
    everything = np.arange(len(dists))
    mat, _ = _padded(dists, everything)
    y_all = gold[everything]
    out["all"] = {
        "n": int(len(y_all)),
        "log_loss": S.log_loss(mat, y_all),
        "acc": float((mat.argmax(1) == y_all).mean()),
        "ece": S.top_label_ece(mat, y_all, 15, "width"),
        "ece_mass": S.top_label_ece(mat, y_all, 15, "mass"),
        **{k: v for k, v in S.murphy_decomposition(mat, y_all).items()},
    }
    if domains and "domain" in preds and len(set(preds["domain"].tolist())) > 1:
        out["per_domain_ece"] = S.grouped_ece(mat, y_all, preds["domain"], 15, "mass")
    return out


def analyse_run(
    run_dir: str | Path,
    splits_eval: tuple[str, ...] = ("select", "dev"),
    served: float | None = None,
    fixed_temperatures: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The full results block of one run: raw, +TS (fitted on cal), and optionally as served."""
    run_dir = Path(run_dir)
    cal = load_preds(run_dir / "preds_cal.npz")
    temps = fixed_temperatures or fit_temperatures(cal)
    result: dict[str, Any] = {"temperature_by_type": temps}
    for split in splits_eval:
        path = run_dir / f"preds_{split}.npz"
        if not path.exists():
            continue
        p = load_preds(path)
        block = {"raw": summarize_split(p, 1.0), "ts": summarize_split(p, temps)}
        if served is not None:
            block["served"] = summarize_split(p, served)
        md = score_mass_deviation(p)
        if md is not None:
            block["score_mass_deviation"] = md
        result[split] = block
    return result
