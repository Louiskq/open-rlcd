"""Data for the landing page's interactive toy: the RLVR-mix sweep of Exp 0, with the reliability diagram at every
checkpoint, from the repository's own toy code (CPU, about a minute).

    python scripts/make_site_data.py [--seeds 5] [--out site/data/toy_mix.json]

For each share of the 0/1 "be right" reward mixed into the RLCD-bandit reward (0 = pure RLCD, 1 = pure RLVR) it stores,
at each of 43 training checkpoints, the accuracy, the ECE and the reliability bins pooled over seeds, and the final
state after one fitted temperature. The supervised control is stored as a reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlcd import estimators as E  # noqa: E402
from toy import exp0 as X  # noqa: E402
from toy.run import MIXES  # noqa: E402


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def pooled(runs: list[X.Run], key: str) -> dict[str, list]:
    """Per-checkpoint reliability bins pooled over seeds: arrays [checkpoints][bins]."""
    count = np.sum([r.reliability["count"] for r in runs], axis=0)
    conf = np.sum([r.reliability["conf_sum"] for r in runs], axis=0)
    corr = np.sum([r.reliability["correct_sum"] for r in runs], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c, a = conf / count, corr / count
    c[count == 0], a[count == 0] = np.nan, np.nan
    return {
        "count": [[int(v) for v in row] for row in count],
        "conf": [[_r(v) for v in row] for row in c],
        "acc": [[_r(v) for v in row] for row in a],
    }[key]


def final_bins(runs: list[X.Run], attr: str) -> dict[str, list]:
    cnt = np.sum([getattr(r, attr)["count"] for r in runs], axis=0)
    cs = np.sum([getattr(r, attr)["conf_sum"] for r in runs], axis=0)
    ks = np.sum([getattr(r, attr)["correct_sum"] for r in runs], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        conf, acc = cs / cnt, ks / cnt
    return {
        "count": [int(v) for v in cnt],
        "conf": [_r(v) if c > 0 else None for v, c in zip(conf, cnt, strict=True)],
        "acc": [_r(v) if c > 0 else None for v, c in zip(acc, cnt, strict=True)],
    }


def arm_block(runs: list[X.Run]) -> dict:
    ece = np.mean([r.curves["ece"] for r in runs], axis=0)
    acc = np.mean([r.curves["acc"] for r in runs], axis=0)
    return {
        "ece": [_r(v) for v in ece],
        "acc": [_r(v) for v in acc],
        "reliability": {k: pooled(runs, k) for k in ("count", "conf", "acc")},
        "final_raw": {
            "ece": _r(np.mean([r.final_raw["ece"] for r in runs])),
            "acc": _r(np.mean([r.final_raw["acc"] for r in runs])),
            "bins": final_bins(runs, "final_reliability_raw"),
        },
        "final_ts": {
            "ece": _r(np.mean([r.final_ts["ece"] for r in runs])),
            "acc": _r(np.mean([r.final_ts["acc"] for r in runs])),
            "bins": final_bins(runs, "final_reliability_ts"),
        },
        "temperature": _r(np.mean([r.temperature for r in runs])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="site/data/toy_mix.json")
    a = ap.parse_args()
    cfg = X.ToyConfig()
    cfg = replace(cfg, seeds=cfg.seeds[: a.seeds])
    w_true = X.ground_truth(cfg)
    eval_set = X.make_dataset(w_true, cfg.n_eval, X.EVAL_SEED)
    cal_set = X.make_dataset(w_true, cfg.n_cal, X.CAL_SEED)
    out: dict = {
        "description": "Exp 0 (toy): RLCD-bandit with a share of the RLVR reward mixed in; reliability pooled over seeds.",
        "config": X.config_dict(cfg),
        "seeds": list(cfg.seeds),
        "bins": X.BINS,
        "mixes": list(MIXES),
        "arms": {},
    }
    runs = X.run_arm(E.Arm("supervised", "supervised"), cfg, True, w_true, eval_set, cal_set)
    out["steps"] = runs[0].steps
    out["supervised"] = arm_block(runs)
    for m in MIXES:
        arm = E.Arm(f"mix_{m:g}", "rlcd_bandit", rlvr_mix=m)
        out["arms"][f"{m:g}"] = arm_block(X.run_arm(arm, cfg, True, w_true, eval_set, cal_set))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
