"""Collect the JevBench summaries of the Exp A models into ``results/expA/jevbench.json``.

    python scripts/collect_jevbench.py JEVBENCH_DIR [--out results/expA/jevbench.json]

``JEVBENCH_DIR`` holds one directory per run with ``summary-{easy,original,hard}.json`` (written by
``slurm/jevbench_expA.sh``). Per run: accuracy, mean Brier score and ECE of each public tier; per arm: seed mean
and range. JevBench hard has 111 items, so differences between arms are descriptive (standard error about 4 points).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlcd.results import mean_range, write_json  # noqa: E402

ARMS = {
    "a0": ["a0"],
    "a1": [f"a1_lr1e-4_s{s}" for s in (0, 1, 2)],
    "a3": [f"a3_lr1e-4_s{s}" for s in (0, 1, 2)],
    "a4": [f"a4_lr1e-4_s{s}" for s in (0, 1, 2)],
    "a5": ["a5_lr1e-4_s0"],
    "a6": [f"a6_lr1e-4_s{s}" for s in (0, 1, 2)],
    "a6r": [f"a6r_lr1e-4_s{s}" for s in (0, 1, 2)],
    "a3beta0": ["a3beta0_lr1e-4_s0"],
}
TIERS = ("easy", "original", "hard")


def tier(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    ece = s["ece"]["ece"] if isinstance(s["ece"], dict) else s["ece"]
    return {"n": s["n_scorable"], "accuracy": s["accuracy"], "brier": s["brier_mean"], "ece": ece}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jev_dir")
    ap.add_argument("--out", default="results/expA/jevbench.json")
    a = ap.parse_args()
    root = Path(a.jev_dir)
    out: dict = {"runs": {}, "arms": {}}
    for arm, runs in ARMS.items():
        per = []
        for r in runs:
            blk = {t: tier(root / r / f"summary-{t}.json") for t in TIERS}
            if all(v is not None for v in blk.values()):
                out["runs"][r] = blk
                per.append(blk)
        if per:
            out["arms"][arm] = {
                t: {m: mean_range([p[t][m] for p in per]) for m in ("accuracy", "brier", "ece")}
                | {"n": per[0][t]["n"]}
                for t in TIERS
            }
    write_json(a.out, out)
    print(f"wrote {a.out}: {len(out['runs'])} runs, arms {sorted(out['arms'])}")


if __name__ == "__main__":
    main()
