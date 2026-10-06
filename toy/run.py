"""Run Exp 0 and write ``results/toy/*.json`` and the figures.

python -m toy.run                 # full: 10 seeds x 4,000 steps, about 1-2 minutes on a laptop CPU
python -m toy.run --quick         # CI: 3 seeds x 1,500 steps
python -m toy.run --out /tmp/x    # write to /tmp/x/results/toy and /tmp/x/figures
"""

from __future__ import annotations

import argparse
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import rlcd
from rlcd import estimators as E
from rlcd.results import mean_range, write_json
from toy import exp0 as X

LR_GRID = (0.003, 0.01, 0.03)
MIXES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
SWEEP_SEEDS = 3
FINAL_METRICS = (
    "acc",
    "ece",
    "ece_mass",
    "log_loss",
    "brier",
    "kl_to_truth",
    "true_gap",
    "mean_conf",
)


def _stack(runs: list[X.Run], attr: str, key: str) -> list[float]:
    return [getattr(r, attr)[key] for r in runs]


def pooled_reliability(count, conf_sum, correct_sum) -> dict[str, list]:
    """Pool per-seed bin sums ([seeds, ..., bins]) into per-bin confidence and accuracy."""
    n = np.sum(count, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.where(n > 0, np.sum(conf_sum, axis=0) / n, np.nan)
        acc = np.where(n > 0, np.sum(correct_sum, axis=0) / n, np.nan)
    return {"count": n, "conf": conf, "acc": acc}


def arm_json(arm: E.Arm, cfg: X.ToyConfig, runs: list[X.Run]) -> dict:
    """One arm's result file: per-seed curves and finals, plus reliability pooled over seeds."""
    rel = pooled_reliability(
        np.stack([r.reliability["count"] for r in runs]),
        np.stack([r.reliability["conf_sum"] for r in runs]),
        np.stack([r.reliability["correct_sum"] for r in runs]),
    )
    final_rel = {}
    for tag, attr in (("raw", "final_reliability_raw"), ("ts", "final_reliability_ts")):
        final_rel[tag] = pooled_reliability(
            np.stack([getattr(r, attr)["count"] for r in runs]),
            np.stack([getattr(r, attr)["conf_sum"] for r in runs]),
            np.stack([getattr(r, attr)["correct_sum"] for r in runs]),
        )
    out = {
        "arm": arm.name,
        "kind": arm.kind,
        "clip": arm.clip,
        "rlvr_mix": arm.rlvr_mix,
        "samples": arm.samples,
        "baseline": arm.baseline,
        "config": X.config_dict(cfg),
        "seeds": [r.seed for r in runs],
        "steps": runs[0].steps,
        "curves": {k: [r.curves[k] for r in runs] for k in runs[0].curves},
        "final": {
            "raw": {k: _stack(runs, "final_raw", k) for k in runs[0].final_raw},
            "ts": {k: _stack(runs, "final_ts", k) for k in runs[0].final_ts},
            "temperature": [r.temperature for r in runs],
        },
        "reliability": {"steps": runs[0].steps, **rel},
        "reliability_final": final_rel,
    }
    if runs[0].by_difficulty is not None:  # Exp 0b only; absent from Exp 0 files
        out["by_difficulty"] = [r.by_difficulty for r in runs]
    return out


def final_summary(runs: list[X.Run], metrics: tuple[str, ...] = FINAL_METRICS) -> dict:
    """Seed mean/range of the headline metrics, raw and after temperature scaling."""
    return {
        "raw": {k: mean_range(_stack(runs, "final_raw", k)) for k in metrics},
        "ts": {k: mean_range(_stack(runs, "final_ts", k)) for k in metrics},
        "temperature": mean_range([r.temperature for r in runs]),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="3 seeds x 1,500 steps (CI)")
    ap.add_argument("--out", type=Path, default=Path("."), help="root for results/toy and figures/")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    cfg = replace(X.ToyConfig(), **X.QUICK) if args.quick else X.ToyConfig()
    res_dir = args.out / "results" / "toy"
    fig_dir = args.out / "figures"
    t_start = time.time()

    w_true = X.ground_truth(cfg)
    eval_set = X.make_dataset(w_true, cfg.n_eval, X.EVAL_SEED)
    cal_set = X.make_dataset(w_true, cfg.n_cal, X.CAL_SEED)
    print(f"Exp 0: {len(cfg.seeds)} seeds x {cfg.steps} steps, lr {cfg.lr}, batch {cfg.batch}")

    print("arms")
    summaries, all_runs = {}, {}
    for name, arm in X.ARMS.items():
        all_runs[name] = X.run_arm(arm, cfg, True, w_true, eval_set, cal_set)
        write_json(res_dir / f"{name}.json", arm_json(arm, cfg, all_runs[name]))
        summaries[name] = final_summary(all_runs[name])

    print("gradient bias and noise at the truth")
    noise = X.gradient_noise(cfg)

    print(f"learning-rate robustness, lr in {LR_GRID}, {SWEEP_SEEDS} seeds")
    sweep_cfg = replace(cfg, seeds=cfg.seeds[:SWEEP_SEEDS])
    lr_table = {}
    for lr in LR_GRID:
        lr_cfg = X.with_lr(sweep_cfg, lr)
        lr_table[f"{lr:g}"] = {
            name: final_summary(X.run_arm(arm, lr_cfg, False, w_true, eval_set, cal_set))
            for name, arm in X.ARMS.items()
        }
        print(f"  lr {lr:g} done")
    write_json(res_dir / "lr_sensitivity.json", {"seeds": list(sweep_cfg.seeds), "lr": lr_table})

    print(f"RLVR-mix sweep, {len(MIXES)} mixes x {SWEEP_SEEDS} seeds")
    sweep = X.mix_sweep(sweep_cfg, MIXES)
    write_json(
        res_dir / "mix_sweep.json",
        {
            "seeds": list(sweep_cfg.seeds),
            "mixes": list(MIXES),
            "summary": {f"{m:g}": final_summary(runs) for m, runs in sweep.items()},
            "ece_curves": {f"{m:g}": [r.curves["ece"] for r in runs] for m, runs in sweep.items()},
            "steps": sweep[MIXES[0]][0].steps,
        },
    )

    write_json(
        res_dir / "meta.json",
        {
            "description": "Exp 0: linear-softmax contextual bandit with known true probabilities.",
            "config": X.config_dict(cfg),
            "seed_sets": {
                "train": list(cfg.seeds),
                "calibration": X.CAL_SEED,
                "evaluation": X.EVAL_SEED,
                "ground_truth": X.GROUND_TRUTH_SEED,
            },
            "oracle": X.reference_metrics(eval_set),
            "gradient_noise_at_truth": noise,
            "summary": summaries,
            "supervised_vs_rlcd_full_max_abs_diff": _max_curve_gap(
                all_runs["supervised"], all_runs["rlcd_full"]
            ),
            "software": {
                "open_rlcd": rlcd.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
            },
        },
    )
    print(f"results -> {res_dir}")

    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        from rlcd import plots

        plots.make_toy_figures(res_dir, fig_dir)
        print(f"figures -> {fig_dir}")
    print(f"done in {time.time() - t_start:.0f}s")


def _max_curve_gap(a: list[X.Run], b: list[X.Run]) -> float:
    """Largest absolute difference between two arms' curves over every metric, seed and step."""
    return float(
        max(
            np.abs(np.asarray(ra.curves[k]) - np.asarray(rb.curves[k])).max()
            for ra, rb in zip(a, b, strict=True)
            for k in ra.curves
        )
    )


if __name__ == "__main__":
    main()
