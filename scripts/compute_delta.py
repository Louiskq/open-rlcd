"""δ for the equivalence margins: twice the between-seed difference of A1 on the SELECTION split (decisions.md D5).

    python scripts/compute_delta.py RUN_SEED_A RUN_SEED_B --out results/expA/delta.json

Reads ``preds_cal.npz`` (to fit one temperature per answer type) and ``preds_select.npz`` of two runs. It never opens
``preds_dev.npz``: the dev predictions of these runs stay unread until the pre-registration is committed (D13).
For every metric, row (raw / +TS) and scope (all / each answer type): ``delta = 2 * |a - b|``; the accuracy delta is
floored at 1.0 point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlcd import evaluate as EV  # noqa: E402
from rlcd.results import write_json  # noqa: E402

ACC_FLOOR = 0.01


def flat(block: dict) -> dict[str, float]:
    """Scalar metrics of a ``summarize_split`` block as ``scope/metric`` -> value."""
    out = {}
    for scope, d in block.items():
        if isinstance(d, dict) and scope != "per_domain_ece":
            for k, v in d.items():
                if isinstance(v, (int, float)) and k != "n":
                    out[f"{scope}/{k}"] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = {}
    for tag, run in (("a", args.a), ("b", args.b)):
        run = Path(run)
        cal = EV.load_preds(run / "preds_cal.npz")
        sel = EV.load_preds(run / "preds_select.npz")  # the only evaluation split opened
        temps = EV.fit_temperatures(cal)
        rows[tag] = {
            "run": run.name,
            "temperature_by_type": temps,
            "raw": flat(EV.summarize_split(sel, 1.0, domains=False)),
            "ts": flat(EV.summarize_split(sel, temps, domains=False)),
        }
    delta = {}
    for row in ("raw", "ts"):
        delta[row] = {}
        for key, va in rows["a"][row].items():
            d = 2 * abs(va - rows["b"][row][key])
            if key.endswith("/acc"):
                d = max(d, ACC_FLOOR)
            delta[row][key] = d
    write_json(
        args.out,
        {
            "rule": "delta = 2 * |seed a - seed b| on the selection split; accuracy floored at 0.01",
            "runs": rows,
            "delta": delta,
        },
    )
    for key in ("all/ece", "all/log_loss", "all/acc", "all/brier", "all/resolution"):
        print(f"{key:18s} raw delta {delta['raw'][key]:.4f}   +TS delta {delta['ts'][key]:.4f}")


if __name__ == "__main__":
    main()
