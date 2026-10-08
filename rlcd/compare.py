"""The pre-registered Exp A comparisons (``PREREGISTRATION.md``, Amendment 1): numpy and scipy only.

Every test compares two arms on the dev set through a paired bootstrap over items. A bootstrap replicate resamples the
items once (within each domain for per-domain ECE) and scores *every seed of every arm and row* on those items, then
averages each metric over seeds ("seed-averaged" is read as the mean over seeds of the per-seed metric, Amendment 1;
``ensemble=True`` averages the seeds' probabilities instead, a robustness check).

p-values (Amendment 1, item 4)

* a claim "A is higher/lower than B": ``p = 2 x`` the share of replicates with the difference on the wrong side of 0
  or equal to 0, capped at 1, so that ``p < 0.05`` is the same as the 95% percentile interval excluding 0;
* equivalence within +-delta: ``p = 2 x max(share d <= -delta, share d >= delta)``, so that ``p < 0.05`` is the same
  as the 95% interval lying inside (-delta, delta).

Holm's step-down correction is applied to each family at 0.05.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from rlcd import evaluate as EV
from rlcd import scoring as S

METRICS = ("log_loss", "acc", "ece", "brier", "reliability", "resolution")
HIGHER_IS_BETTER = {"acc", "resolution"}
B_DEFAULT = 2000
SEED_DEFAULT = 20261008


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------


@dataclass
class RunProbs:
    """One run's dev (or selection) predictions as padded probability matrices, raw and +TS."""

    name: str
    seed: int
    temperatures: dict[str, float]
    raw: np.ndarray
    ts: np.ndarray


@dataclass
class Arm:
    arm: str
    runs: list[RunProbs] = field(default_factory=list)


@dataclass
class Items:
    """The evaluation items shared by every run."""

    qid: np.ndarray
    gold: np.ndarray
    qtype: np.ndarray
    domain: np.ndarray


def _padded_probs(preds: dict[str, np.ndarray], temperature: Any, kmax: int) -> np.ndarray:
    dists = EV.probabilities(preds, temperature)
    mat = np.zeros((len(dists), kmax))
    for i, p in enumerate(dists):
        mat[i, : len(p)] = p
    return mat


def load_run(
    run_dir: str | Path, seed: int, split: str = "dev"
) -> tuple[RunProbs, Items, dict[str, np.ndarray]]:
    run_dir = Path(run_dir)
    cal = EV.load_preds(run_dir / "preds_cal.npz")
    temps = EV.fit_temperatures(cal)
    preds = EV.load_preds(run_dir / f"preds_{split}.npz")
    kmax = int(preds["z"].shape[1])
    items = Items(
        preds["qid"], preds["gold"].astype(int), preds["qtype"].astype(int), preds["domain"]
    )
    run = RunProbs(
        run_dir.name,
        seed,
        temps,
        _padded_probs(preds, 1.0, kmax),
        _padded_probs(preds, temps, kmax),
    )
    return run, items, preds


def load_arms(
    runs_dir: str | Path, spec: dict[str, Sequence[tuple[str, int]]], split: str = "dev"
) -> tuple[dict[str, Arm], Items]:
    """``spec`` maps an arm id to ``[(run directory name, seed), ...]``. Items must match across runs."""
    arms: dict[str, Arm] = {}
    ref: Items | None = None
    for arm_id, runs in spec.items():
        arm = Arm(arm_id)
        for name, seed in runs:
            run, items, _ = load_run(Path(runs_dir) / name, seed, split)
            if ref is None:
                ref = items
            elif not np.array_equal(ref.qid, items.qid) or not np.array_equal(ref.gold, items.gold):
                raise ValueError(f"{name}: evaluation items differ from the first run's")
            arm.runs.append(run)
        arms[arm_id] = arm
    if ref is None:
        raise ValueError("empty spec")
    return arms, ref


# ---------------------------------------------------------------------------------------------
# metrics on a resample
# ---------------------------------------------------------------------------------------------


class _Cache:
    """Per-matrix quantities that do not depend on the resample."""

    def __init__(self, mat: np.ndarray, gold: np.ndarray) -> None:
        n = len(gold)
        self.mat = mat
        self.gold = gold
        self.conf = mat.max(axis=1)
        self.correct = (mat.argmax(axis=1) == gold).astype(np.float64)
        self.logscore = -np.log(np.clip(mat[np.arange(n), gold], 1e-12, 1.0))
        onehot = np.zeros_like(mat)
        onehot[np.arange(n), gold] = 1.0
        self.brier_i = ((mat - onehot) ** 2).sum(axis=1)


def _metrics(c: _Cache, idx: np.ndarray, want: Iterable[str]) -> dict[str, float]:
    want = set(want)
    out: dict[str, float] = {}
    if "log_loss" in want:
        out["log_loss"] = float(c.logscore[idx].mean())
    if "acc" in want:
        out["acc"] = float(c.correct[idx].mean())
    if "brier" in want:
        out["brier"] = float(c.brier_i[idx].mean())
    if "ece" in want:
        out["ece"] = S.ece(c.conf[idx], c.correct[idx].astype(bool), 15, "width")
    if want & {"reliability", "resolution"}:
        m = S.murphy_decomposition(c.mat[idx], c.gold[idx])
        out["reliability"] = m["reliability"]
        out["resolution"] = m["resolution"]
    return out


def _domain_ece(c: _Cache, idx: np.ndarray, domain: np.ndarray) -> float:
    """Mean over domains of the equal-mass 15-bin ECE (the H5b metric) on the resampled items."""
    d = domain[idx]
    vals = [
        S.ece(c.conf[idx][d == g], c.correct[idx][d == g].astype(bool), 15, "mass")
        for g in np.unique(d)
    ]
    return float(np.mean(vals))


# ---------------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Test:
    id: str  # e.g. "C1-ece"
    family: str  # "primary" or "H5b"
    hypothesis: str
    a: tuple[str, tuple[int, ...] | None]  # (arm, seeds); None = all seeds
    b: tuple[str, tuple[int, ...] | None]
    row: str  # "raw" or "ts"
    metric: str  # one of METRICS or "domain_ece"
    kind: str  # "a_higher", "a_lower", "equiv", or "diff" (exploratory: difference and interval, no p)
    scope: str = "all"  # "all" or an answer type: "noul", "choice", "score"


def preregistered_tests() -> list[Test]:
    """The tests of Amendment 1, items 5 and 6."""
    t: list[Test] = []
    A3, A1 = ("a3", None), ("a1", None)
    for m, kind in (("ece", "a_higher"), ("log_loss", "a_higher"), ("acc", "equiv")):
        t.append(Test(f"C1-{m}", "primary", "H1", A3, A1, "raw", m, kind))
    a5, a1s0 = ("a5", (0,)), ("a1", (0,))
    for m in ("log_loss", "ece", "acc", "brier", "reliability", "resolution"):
        t.append(Test(f"C2-{m}", "primary", "H2", a5, a1s0, "raw", m, "equiv"))
    for m in ("ece", "log_loss"):
        t.append(Test(f"C3-{m}", "primary", "H3", ("a4", None), A1, "raw", m, "equiv"))
    for m, kind in (("ece", "a_lower"), ("log_loss", "a_lower"), ("acc", "a_higher")):
        t.append(Test(f"C4-{m}", "primary", "H4", ("a6", None), ("a6r", None), "raw", m, kind))
    t.append(Test("C5-a3", "primary", "H5", A3, A1, "ts", "ece", "equiv"))
    t.append(Test("C5-a4", "primary", "H5", ("a4", None), A1, "ts", "ece", "equiv"))
    t.append(Test("C5-a5", "primary", "H5", a5, a1s0, "ts", "ece", "equiv"))
    t.append(Test("H5b-domain_ece", "H5b", "H5b", A3, A1, "ts", "domain_ece", "a_higher"))
    t.append(Test("H5b-resolution", "H5b", "H5b", A3, A1, "ts", "resolution", "a_lower"))
    return t


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm's step-down adjusted p-values and rejections at ``alpha``."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist(), (adj < alpha).tolist()


def p_directional(d: np.ndarray, kind: str) -> float:
    """``a_higher`` claims d > 0, ``a_lower`` claims d < 0 (Amendment 1, item 4)."""
    wrong = (d <= 0) if kind == "a_higher" else (d >= 0)
    return float(min(1.0, 2.0 * wrong.mean()))


def p_equivalence(d: np.ndarray, delta: float) -> float:
    return float(min(1.0, 2.0 * max((d <= -delta).mean(), (d >= delta).mean())))


def _seed_runs(arm: Arm, seeds: tuple[int, ...] | None) -> list[RunProbs]:
    runs = [r for r in arm.runs if seeds is None or r.seed in seeds]
    if not runs:
        raise ValueError(f"{arm.arm}: no runs for seeds {seeds}")
    return runs


def _ens_spec(
    spec: tuple[str, tuple[int, ...] | None], arms: dict[str, Arm]
) -> tuple[str, tuple[int, ...] | None]:
    name, seeds = spec
    if seeds is None and f"{name}#ens" in arms:
        return (f"{name}#ens", None)
    return spec


def _ensemble(arm: Arm) -> Arm:
    """One run whose probabilities are the mean over the arm's seeds (robustness check)."""
    raw = np.mean([r.raw for r in arm.runs], axis=0)
    ts = np.mean([r.ts for r in arm.runs], axis=0)
    first = arm.runs[0]
    return Arm(arm.arm, [RunProbs(f"{arm.arm}-ensemble", -1, first.temperatures, raw, ts)])


def run_tests(
    arms: dict[str, Arm],
    items: Items,
    deltas: dict[str, dict[str, float]],
    tests: Sequence[Test] | None = None,
    B: int = B_DEFAULT,
    seed: int = SEED_DEFAULT,
    ensemble: bool = False,
) -> list[dict[str, Any]]:
    """Run ``tests``: observed difference, 95% percentile interval, p, Holm-adjusted p, verdict.

    ``deltas`` is ``results/expA/delta.json``'s ``delta`` block: ``deltas["raw"]["all/ece"]`` and so on.
    """
    tests = list(tests if tests is not None else preregistered_tests())
    if (
        ensemble
    ):  # a spec with seeds=None uses the arm's seed-ensemble; an explicit seed keeps the single run
        arms = {**arms, **{f"{k}#ens": _ensemble(a) for k, a in arms.items() if len(a.runs) > 1}}
        tests = [replace(x, a=_ens_spec(x.a, arms), b=_ens_spec(x.b, arms)) for x in tests]
    gold, domain = items.gold, items.domain
    n = len(gold)
    caches = {
        (k, r.seed, row): _Cache(getattr(r, row), gold)
        for k, a in arms.items()
        for r in a.runs
        for row in ("raw", "ts")
    }

    def estimate(
        spec: tuple[str, tuple[int, ...] | None],
        row: str,
        metric: str,
        idx: np.ndarray,
        didx: np.ndarray,
    ) -> float:
        runs = _seed_runs(arms[spec[0]], spec[1])
        vals = []
        for r in runs:
            c = caches[(spec[0], r.seed, row)]
            vals.append(
                _domain_ece(c, didx, domain)
                if metric == "domain_ece"
                else _metrics(c, idx, [metric])[metric]
            )
        return float(np.mean(vals))

    rng = np.random.default_rng(seed)
    # a separate stream for scoped tests: adding them never changes the primary draws
    rng_scope = np.random.default_rng(seed + 1)
    by_domain = [np.flatnonzero(domain == g) for g in np.unique(domain)]
    scopes = {name: np.flatnonzero(items.qtype == code) for code, name in EV.TYPE_NAMES.items()}
    scopes["all"] = np.arange(n)
    used = sorted({t.scope for t in tests} - {"all"})
    full = np.arange(n)
    obs = np.array(
        [
            estimate(t.a, t.row, t.metric, scopes[t.scope], full)
            - estimate(t.b, t.row, t.metric, scopes[t.scope], full)
            for t in tests
        ]
    )
    boot = np.empty((B, len(tests)))
    for b in range(B):
        idx = rng.integers(0, n, n)
        didx = np.concatenate([rng.choice(g, len(g), replace=True) for g in by_domain])
        sidx = {"all": idx}
        for s in used:
            sidx[s] = rng_scope.choice(scopes[s], len(scopes[s]), replace=True)
        for j, t in enumerate(tests):
            boot[b, j] = estimate(t.a, t.row, t.metric, sidx[t.scope], didx) - estimate(
                t.b, t.row, t.metric, sidx[t.scope], didx
            )

    results: list[dict[str, Any]] = []
    for j, t in enumerate(tests):
        d = boot[:, j]
        lo, hi = np.quantile(d, [0.025, 0.975])
        delta = None
        if t.kind == "diff":
            p = None
        elif t.kind == "equiv":
            delta = float(deltas[t.row][f"all/{t.metric}"]) if t.metric in METRICS else None
            p = p_equivalence(d, delta)
        else:
            p = p_directional(d, t.kind)
        results.append(
            {
                "id": t.id,
                "family": t.family,
                "hypothesis": t.hypothesis,
                "a": (t.a[0] if t.a[1] is None else f"{t.a[0]} seed {t.a[1][0]}").replace(
                    "#ens", " (seed ensemble)"
                ),
                "b": (t.b[0] if t.b[1] is None else f"{t.b[0]} seed {t.b[1][0]}").replace(
                    "#ens", " (seed ensemble)"
                ),
                "row": t.row,
                "metric": t.metric,
                "scope": t.scope,
                "kind": t.kind,
                "delta": delta,
                "observed_difference": float(obs[j]),
                "ci95": [float(lo), float(hi)],
                "p": p,
            }
        )
    for fam in sorted({r["family"] for r in results}):
        rows = [r for r in results if r["family"] == fam and r["p"] is not None]
        if not rows:
            continue
        adj, rej = holm([r["p"] for r in rows])
        for r, a, ok in zip(rows, adj, rej, strict=True):
            r["p_holm"], r["passes"] = a, bool(ok)
    return results


def exploratory_tests() -> list[Test]:
    """Descriptive per-answer-type differences (no p-values, no hypotheses): not in the pre-registration."""
    out = []
    for a, b in (("a3", "a1"), ("a4", "a1"), ("a6", "a6r"), ("a5", "a1")):
        seeds = (0,) if a == "a5" else None
        for scope in ("noul", "choice", "score"):
            for row in ("raw", "ts"):
                for m in ("log_loss", "ece", "acc"):
                    tid = f"X-{a}-{b}-{scope}-{row}-{m}"
                    out.append(
                        Test(tid, "exploratory", "X", (a, seeds), (b, seeds), row, m, "diff", scope)
                    )
    return out


def hypothesis_verdicts(results: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Amendment 1, item 7: what counts as each hypothesis holding (H4 holds if NO C4 test passes)."""
    by_h: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        if r.get("family") != "exploratory":
            by_h.setdefault(r["hypothesis"], []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for h, rows in by_h.items():
        passes = [r["passes"] for r in rows]
        holds = (not any(passes)) if h == "H4" else all(passes)
        out[h] = {"holds": bool(holds), "tests": {r["id"]: r["passes"] for r in rows}}
    return out
