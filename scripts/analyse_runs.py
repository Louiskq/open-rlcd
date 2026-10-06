"""Compute ``metrics.json`` for Exp A run directories from their saved predictions (CPU, numpy + scipy only).

    python scripts/analyse_runs.py RUN_DIR [RUN_DIR ...] [--served 1.05]

Each run directory needs ``preds_cal.npz`` (to fit one temperature per answer type) and ``preds_select.npz`` and / or
``preds_dev.npz``. ``--served`` adds the "as served" row at the checkpoint's stored temperature (A0 only, D2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlcd.evaluate import analyse_run  # noqa: E402
from rlcd.results import write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--served", type=float, default=None)
    ap.add_argument("--splits", default="select,dev")
    a = ap.parse_args()
    for run in a.runs:
        result = analyse_run(run, tuple(a.splits.split(",")), a.served)
        write_json(Path(run) / "metrics.json", result)
        temps = {k: round(v, 3) for k, v in result["temperature_by_type"].items()}
        print(f"{run}: temperatures {temps}")
        for split in a.splits.split(","):
            if split not in result:
                continue
            for row in ("raw", "ts", "served"):
                if row in result[split]:
                    al = result[split][row]["all"]
                    print(
                        f"  {split:6s} {row:6s} log loss {al['log_loss']:.4f} acc {al['acc']:.4f} "
                        f"ece {al['ece']:.4f} resolution {al['resolution']:.4f}"
                    )


if __name__ == "__main__":
    main()
