"""Exp 0b (EXPLORATORY, not pre-registered): a toy where difficulty varies by input.

    python -m toy.exp0b              # 10 seeds x 4,000 steps -> results/toy0b/, figures/
    python -m toy.exp0b --quick      # 3 seeds x 1,500 steps

In Exp 0 one temperature repairs RLVR because the sharpening is uniform across inputs. Here the
*scale* of the true logits varies with the input, ``z*(x) = s(x) W* x`` with ``s`` from 0.3 (hard,
near-uniform truth) to 3.0 (easy, near-deterministic); the mean scale matches Exp 0's. The policy
can represent the truth exactly (``toy.exp0.Problem``), so any remaining miscalibration comes from
the reward, not from the model class.

The question: after temperature scaling (one global temperature, fitted on a separate calibration
set), how do ECE and Murphy resolution compare for RLVR and RLCD? The scale range (0.3, 3.0) was
fixed in this file before the first run and has not been tuned. Everything is reported, including
diagnostics that were not asked for: calibration within difficulty terciles, and a per-tercile
temperature that uses the known difficulty (which a real model would not have).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from rlcd import scoring as S
from rlcd.results import mean_range, write_json
from toy import exp0 as X
from toy.run import FINAL_METRICS, arm_json, final_summary

SCALE_RANGE = (0.3, 3.0)  # fixed before the first run
METRICS = (*FINAL_METRICS, "resolution", "reliability", "uncertainty")


def _gap(group: dict[str, float]) -> float:
    """Mean confidence minus accuracy: positive is overconfident."""
    return group["mean_conf"] - group["acc"]


def _group_summary(runs: list[X.Run]) -> dict:
    """Seed mean/range of per-tercile ECE and signed confidence gap, raw and after the global T."""
    names = X.DIFFICULTY_NAMES
    out: dict[str, object] = {"names": list(names)}
    for key in ("raw", "ts"):
        out[key] = {
            "ece": [mean_range([r.by_difficulty[key][g]["ece"] for r in runs]) for g in range(3)],
            "gap": [
                mean_range(
                    [
                        r.by_difficulty[key][g]["mean_conf"] - r.by_difficulty[key][g]["acc"]
                        for r in runs
                    ]
                )
                for g in range(3)
            ],
        }
    out["group_temperature"] = [
        mean_range([r.by_difficulty["group_temperature"][g] for r in runs]) for g in range(3)
    ]
    out["ts_group_overall"] = {
        k: mean_range([r.by_difficulty["ts_group_overall"][k] for r in runs])
        for k in ("ece", "resolution", "log_loss", "kl_to_truth")
    }
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    cfg = replace(X.ToyConfig(), **X.QUICK) if args.quick else X.ToyConfig()
    res_dir = args.out / "results" / "toy0b"
    t0 = time.time()
    problem = X.hetero_problem(cfg, *SCALE_RANGE)
    eval_set = X.make_dataset(problem, cfg.n_eval, X.EVAL_SEED)
    cal_set = X.make_dataset(problem, cfg.n_cal, X.CAL_SEED)
    print(f"Exp 0b (exploratory): {len(cfg.seeds)} seeds x {cfg.steps} steps, scale {SCALE_RANGE}")

    summary, groups = {}, {}
    for name, arm in X.ARMS.items():
        runs = X.run_arm(arm, cfg, True, problem, eval_set, cal_set)
        write_json(res_dir / f"{name}.json", arm_json(arm, cfg, runs))
        summary[name] = final_summary(runs, METRICS)
        groups[name] = _group_summary(runs)

    oracle = X.reference_metrics(eval_set)
    oracle.update(
        {k: v for k, v in S.murphy_decomposition(eval_set.q, eval_set.y).items() if k != "brier"}
    )
    write_json(
        res_dir / "meta.json",
        {
            "description": "Exp 0b (exploratory): the logit scale varies with the input.",
            "exploratory": True,
            "config": X.config_dict(cfg),
            "scale_range": list(SCALE_RANGE),
            "seed_sets": {
                "train": list(cfg.seeds),
                "calibration": X.CAL_SEED,
                "evaluation": X.EVAL_SEED,
                "ground_truth": X.GROUND_TRUTH_SEED,
            },
            "oracle": oracle,
            "summary": summary,
            "by_difficulty": groups,
            "difficulty_terciles": list(X.DIFFICULTY_NAMES),
        },
    )
    print(f"results -> {res_dir}")
    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        from rlcd import plots

        plots.make_toy0b_figures(res_dir, args.out / "figures")
        print("figures written")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
