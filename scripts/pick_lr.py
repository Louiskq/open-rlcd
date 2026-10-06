"""Pick the learning rate of an Exp A sweep: the lowest mean per-question log loss on the SELECTION split.

    python scripts/pick_lr.py RUNS_DIR PREFIX            # e.g. runs a1_lr5e-5_s0 a1_lr1e-4_s0 ... -> prints the rate

The rule is fixed in advance (``docs/decisions.md`` D-impl): selection split only, all answer types pooled, diverged
runs excluded. Reads ``run.json`` / ``eval_metrics.json`` and never opens the dev predictions.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> None:
    runs_dir, prefix = Path(sys.argv[1]), sys.argv[2]
    rows = []
    for d in sorted(runs_dir.glob(f"{prefix}_lr*_s0")):
        rj, em = d / "run.json", d / "eval_metrics.json"
        if not rj.exists() or not em.exists():
            continue
        run = json.loads(rj.read_text())
        lr = re.search(r"_lr([0-9.e+-]+)_s0$", d.name).group(1)
        nll = json.loads(em.read_text())["select"]["all"]["nll"]
        rows.append({"lr": lr, "select_nll": nll, "diverged": bool(run.get("diverged"))})
    ok = [r for r in rows if not r["diverged"]]
    if not ok:
        raise SystemExit(f"no finished, non-diverged runs for {prefix} in {runs_dir}")
    best = min(ok, key=lambda r: r["select_nll"])
    (runs_dir / f"sweep_{prefix}.json").write_text(
        json.dumps({"rows": rows, "chosen": best["lr"]}, indent=1)
    )
    print(best["lr"])


if __name__ == "__main__":
    main()
