"""Summarise the phase 2 smoke runs (10% of the steps, 1 seed, learning rate 1e-4) into one table.

    python scripts/summarize_smoke.py RUNS_DIR --out results/expA/phase2/smoke_summary.json

``RUNS_DIR`` holds ``a0`` and ``smoke_<arm>`` run directories with ``metrics.json`` (``scripts/analyse_runs.py``),
``run.json`` and ``metrics.jsonl``. The numbers are smoke-run numbers: one seed, a tenth of the steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlcd.results import write_json  # noqa: E402

ORDER = ["a0", "smoke_a1", "smoke_a1b", "smoke_a5", "smoke_a3", "smoke_a4", "smoke_a6", "smoke_a6r"]
LABEL = {
    "a0": "A0 start model",
    "smoke_a1": "A1 supervised",
    "smoke_a1b": "A1 rerun (same seed)",
    "smoke_a5": "A5 RLCD-full",
    "smoke_a3": "A3 RLVR",
    "smoke_a4": "A4 RLCD-bandit",
    "smoke_a6": "A6 mined",
    "smoke_a6r": "A6r random",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, rows = Path(a.runs_dir), {}
    for name in ORDER:
        d = root / name
        if not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        row = {"label": LABEL[name], "temperature_by_type": m["temperature_by_type"]}
        for split in ("select", "dev"):
            for kind in ("raw", "ts"):
                al = m[split][kind]["all"]
                row[f"{split}_{kind}"] = {
                    k: al[k] for k in ("log_loss", "acc", "ece", "resolution")
                }
            row[f"{split}_by_type_raw"] = {
                t: {k: m[split]["raw"][t][k] for k in ("log_loss", "acc", "ece")}
                for t in ("noul", "choice", "score")
            }
        if "score_mass_deviation" in m["select"]:
            row["score_mass_deviation_select"] = m["select"]["score_mass_deviation"]
        if (d / "metrics.jsonl").exists():
            last = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines()][-1]
            row["train_kl_last"] = last["all"]["kl"]
        if (d / "run.json").exists():
            r = json.loads((d / "run.json").read_text())
            row.update(
                {
                    "steps": r["steps"],
                    "diverged": r["diverged"],
                    "train_seconds": r["train_seconds"],
                    "peak_mem_gb": r["peak_mem_gb"],
                    "train_questions_end": r["train_questions_end"],
                }
            )
            gs = r.get("grad_share") or []
            if gs:
                row["grad_share_mean"] = {
                    t: sum(g[t]["share_of_norms"] for g in gs if t in g)
                    / max(1, sum(t in g for g in gs))
                    for t in ("noul", "choice", "score")
                }
        rows[name] = row
    write_json(a.out, {"setting": "10% of S = 734 steps, seed 0, lr 1e-4, beta 0.1", "runs": rows})
    print(
        f"{'run':24s} {'sel ll':>7s} {'sel acc':>7s} {'sel ece':>7s} {'sel ece+TS':>10s} {'dev ll':>7s} {'dev acc':>7s} {'dev ece':>7s} {'KL':>6s}"
    )
    for r in rows.values():
        print(
            f"{r['label']:24s} {r['select_raw']['log_loss']:7.4f} {r['select_raw']['acc']:7.4f} {r['select_raw']['ece']:7.4f} "
            f"{r['select_ts']['ece']:10.4f} {r['dev_raw']['log_loss']:7.4f} {r['dev_raw']['acc']:7.4f} {r['dev_raw']['ece']:7.4f} "
            f"{r.get('train_kl_last', float('nan')):6.3f}"
        )


if __name__ == "__main__":
    main()
