"""Exp A results: per-arm tables and the pre-registered comparisons (PREREGISTRATION.md, Amendment 1).

    python scripts/compare_arms.py RUNS_DIR --delta results/expA/delta.json --out results/expA/analysis.json

``RUNS_DIR`` holds the run directories copied from the cluster (``preds_*.npz`` per run). Writes one JSON with

* ``arms``: for every arm, every metric of every answer type and of the pooled ``all`` row, as seed mean and range, for
  the raw (T = 1), +TS and (A0 only) as-served rows, on the dev and selection splits;
* ``tests``: the 19 pre-registered tests with the observed difference, the 95% interval, p, the Holm-adjusted p and the
  verdict; ``hypotheses``: the verdict per hypothesis; ``robustness``: the same tests on seed-ensembled probabilities;
* ``learning_rate_sweeps`` and ``temperatures`` (fitted per run on the calibration split).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlcd import compare as C  # noqa: E402
from rlcd import evaluate as EV  # noqa: E402
from rlcd.results import mean_range, write_json  # noqa: E402

LR = "1e-4"
ARMS = {
    "a1": [(f"a1_lr{LR}_s{s}", s) for s in (0, 1, 2)],
    "a3": [(f"a3_lr{LR}_s{s}", s) for s in (0, 1, 2)],
    "a4": [(f"a4_lr{LR}_s{s}", s) for s in (0, 1, 2)],
    "a5": [(f"a5_lr{LR}_s0", 0)],
    "a6": [(f"a6_lr{LR}_s{s}", s) for s in (0, 1, 2)],
    "a6r": [(f"a6r_lr{LR}_s{s}", s) for s in (0, 1, 2)],
}
DESCRIPTIVE = {"a3beta0": [(f"a3beta0_lr{LR}_s0", 0)], "a0": [("a0", 0)]}
SWEEPS = {
    "a1": [f"a1_lr{lr}_s0" for lr in ("5e-5", "1e-4", "2e-4")],
    "a4": [f"a4_lr{lr}_s0" for lr in ("5e-5", "1e-4", "2e-4")],
}


def aggregate(blocks: list[dict]) -> dict:
    """Seed mean and range of every scalar in a list of ``summarize_split`` blocks."""
    out: dict = {}
    for scope in blocks[0]:
        if scope == "per_domain_ece":
            continue
        vals = {
            k: [b[scope][k] for b in blocks]
            for k, v in blocks[0][scope].items()
            if isinstance(v, (int, float))
        }
        out[scope] = {k: (mean_range(v) if k != "n" else int(v[0])) for k, v in vals.items()}
    if "per_domain_ece" in blocks[0]:
        keys = blocks[0]["per_domain_ece"].keys()
        out["per_domain_ece"] = {
            k: mean_range([b["per_domain_ece"][k] for b in blocks]) for k in keys
        }
    return out


def arm_table(runs_dir: Path, runs: list[tuple[str, int]], served: float | None) -> dict:
    per_run = [EV.analyse_run(runs_dir / name, ("select", "dev"), served) for name, _ in runs]
    out: dict = {
        "runs": [n for n, _ in runs],
        "temperatures": {
            n: r["temperature_by_type"] for (n, _), r in zip(runs, per_run, strict=True)
        },
    }
    for split in ("select", "dev"):
        out[split] = {}
        for row in ("raw", "ts", "served"):
            if row in per_run[0][split]:
                out[split][row] = aggregate([r[split][row] for r in per_run])
        out[split]["score_mass_deviation"] = mean_range(
            [
                r[split]["score_mass_deviation"]
                for r in per_run
                if "score_mass_deviation" in r[split]
            ]
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--delta", default="results/expA/delta.json")
    ap.add_argument("--out", default="results/expA/analysis.json")
    ap.add_argument("--B", type=int, default=C.B_DEFAULT)
    ap.add_argument("--no_tests", action="store_true")
    ap.add_argument("--no_tables", action="store_true")
    ap.add_argument("--split", default="dev", help="split the tests run on (pre-registered: dev)")
    ap.add_argument(
        "--exploratory",
        action="store_true",
        help="per-type differences, not the pre-registered tests",
    )
    a = ap.parse_args()
    runs_dir = Path(a.runs_dir)
    result: dict = {"arms": {}, "bootstrap": {"B": a.B, "seed": C.SEED_DEFAULT, "split": a.split}}
    if not a.no_tables:
        for arm, runs in {**ARMS, **DESCRIPTIVE}.items():
            result["arms"][arm] = arm_table(
                runs_dir, runs, EV.SERVED_TEMPERATURE if arm == "a0" else None
            )
            print(f"{arm}: tables done", flush=True)
        result["learning_rate_sweeps"] = {}
        for arm, names in SWEEPS.items():
            rows = {}
            for n in names:
                sel = EV.summarize_split(
                    EV.load_preds(runs_dir / n / "preds_select.npz"), 1.0, domains=False
                )["all"]
                rows[n] = {
                    "select_log_loss": sel["log_loss"],
                    "select_acc": sel["acc"],
                    "select_ece": sel["ece"],
                }
            result["learning_rate_sweeps"][arm] = rows
    if not a.no_tests:
        deltas = json.loads(Path(a.delta).read_text())["delta"]
        runs = [("tests", False)] if a.exploratory else [("tests", False), ("robustness", True)]
        for label, ens in runs:
            arms, items = C.load_arms(runs_dir, ARMS, a.split)
            tests = C.exploratory_tests() if a.exploratory else None
            res = C.run_tests(arms, items, deltas, tests, B=a.B, ensemble=ens)
            result[label] = res
            if not a.exploratory:
                key = "hypotheses" if label == "tests" else "hypotheses_ensemble"
                result[key] = C.hypothesis_verdicts(res)
            print(f"{label}: done ({len(res)} tests)", flush=True)
    write_json(a.out, result)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
