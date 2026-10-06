"""The δ and learning-rate scripts: the rule is applied as written and the dev predictions are never opened."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.test_evaluate import synthetic

ROOT = Path(__file__).resolve().parent.parent


def write_run(
    path: Path, seed: int, scale: float, nll: float | None = None, diverged: bool = False
) -> None:
    path.mkdir(parents=True)
    for split in ("cal", "select"):  # note: no preds_dev.npz at all
        p = synthetic(
            300,
            {"noul": scale, "choice": scale, "score": scale},
            seed=seed + (split == "select") * 100,
        )
        np.savez_compressed(path / f"preds_{split}.npz", **p)
    if nll is not None:
        (path / "run.json").write_text(json.dumps({"diverged": diverged}))
        (path / "eval_metrics.json").write_text(json.dumps({"select": {"all": {"nll": nll}}}))


def test_compute_delta_uses_only_cal_and_select_and_floors_the_accuracy_margin(tmp_path):
    write_run(tmp_path / "a", 0, 1.0)
    write_run(tmp_path / "b", 1, 1.4)
    out = tmp_path / "delta.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compute_delta.py"),
            str(tmp_path / "a"),
            str(tmp_path / "b"),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
    )  # raises if it tried to open a dev file that does not exist
    d = json.loads(out.read_text())
    ra, rb = d["runs"]["a"]["raw"], d["runs"]["b"]["raw"]
    assert d["delta"]["raw"]["all/log_loss"] == pytest.approx(
        2 * abs(ra["all/log_loss"] - rb["all/log_loss"]), rel=1e-4
    )
    assert d["delta"]["raw"]["all/acc"] >= 0.01 and d["delta"]["ts"]["all/acc"] >= 0.01
    assert "all/ece" in d["delta"]["ts"] and "score/rps" in d["delta"]["raw"]


def test_pick_lr_takes_the_lowest_selection_loss_and_ignores_diverged_runs(tmp_path):
    write_run(tmp_path / "a1_lr5e-5_s0", 0, 1.0, nll=0.50)
    write_run(tmp_path / "a1_lr1e-4_s0", 1, 1.0, nll=0.47)
    write_run(
        tmp_path / "a1_lr2e-4_s0", 2, 1.0, nll=0.30, diverged=True
    )  # best number, but diverged
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/pick_lr.py"), str(tmp_path), "a1"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert r.stdout.strip() == "1e-4"
    saved = json.loads((tmp_path / "sweep_a1.json").read_text())
    assert saved["chosen"] == "1e-4" and len(saved["rows"]) == 3


def test_pick_lr_fails_loudly_when_nothing_finished(tmp_path):
    (tmp_path / "a1_lr1e-4_s0").mkdir()
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/pick_lr.py"), str(tmp_path), "a1"],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
