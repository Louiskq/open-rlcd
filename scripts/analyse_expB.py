"""Exp B analysis: delta_game (before any B3 run), then the full tables and the H7 test.

    python scripts/analyse_expB.py delta-game RUNS_DIR --out results/expB/delta_game.json
    python scripts/analyse_expB.py analyse RUNS_DIR --delta results/expB/delta_game.json \
        --out results/expB/analysis.json

``delta-game`` reads only the selection-seed scores (``sel_evals.json``) of the three B1 runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlcd import breakout_analysis as BA  # noqa: E402
from rlcd.results import write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("delta-game")
    d.add_argument("runs_dir")
    d.add_argument("--out", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("runs_dir")
    a.add_argument("--delta", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    if args.cmd == "delta-game":
        res = BA.compute_delta_game(Path(args.runs_dir))
        write_json(args.out, res)
        print(
            f"delta_game = {res['delta_game']:.3f} (range {res['range']:.3f}, floor {res['floor']:.3f}, mean {res['mean']:.3f})"
        )
    else:
        res = BA.analyse(args.runs_dir, args.delta, args.B)
        write_json(args.out, res)
        print("H7:", res["h7"].get("status"), res["h7"].get("h7_holds"))


if __name__ == "__main__":
    main()
