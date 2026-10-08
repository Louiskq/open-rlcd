"""Exp B offline analysis (numpy and scipy only): game scores, belief metrics, temperature scaling, delta_game and
the H7 test, from the files that ``rlcd.breakout_cli`` writes. Layout under the runs directory::

    b1_s0 ... b1_s2, b3_s0 ... b3_s2, b5_s0      run.json, sel_evals.json, eval_best/, eval_last/
    b4_s0 ... b4_s2                              eval/beliefs_common.npz, eval/beliefs_sel_b1.npz
                                                 (B4 answering B1's common-state and selection-seed questions)
    <run>/eval_<ckpt>/games.json, beliefs_onpolicy_{eval,select}.npz, beliefs_common.npz

``eval_best`` is the checkpoint chosen on the selection seeds (the start model if training never beat it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rlcd import breakout as BK

SEEDS = (0, 1, 2)


def _npz(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def _json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def mean_range(values: list[float]) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


# ---------------------------------------------------------------------------------------------
# delta_game (computed from B1's selection-seed scores, before any B3 run is submitted)
# ---------------------------------------------------------------------------------------------


def selected_selection_score(run_dir: Path) -> float:
    """Greedy mean score on the selection seeds of the run's selected checkpoint (the maximum of ``sel_evals``)."""
    info = json.loads((run_dir / "sel_evals.json").read_text())
    scores = dict((int(i), float(s)) for i, s in info["evals"])
    return scores[int(info["selected_iter"])]


def compute_delta_game(runs_dir: Path, arm: str = "b1") -> dict[str, Any]:
    names = [f"{arm}_s{s}" for s in SEEDS]
    per_seed = [selected_selection_score(runs_dir / n) for n in names]
    out = BK.delta_game(per_seed)
    out.update({"runs": names, "selected_selection_scores": per_seed, "seeds": list(SEEDS)})
    return out


# ---------------------------------------------------------------------------------------------
# belief evaluation
# ---------------------------------------------------------------------------------------------


def _p(b: dict[str, np.ndarray], temperature: float = 1.0) -> np.ndarray:
    return BK.p_yes_from_logit_diff(b["logit_diff"], temperature)


def fit_temperature_on(b: dict[str, np.ndarray]) -> float:
    return BK.fit_belief_temperature(b["logit_diff"], b["gold"])


def belief_block(
    b: dict[str, np.ndarray], temperature: float = 1.0, skill_gold: np.ndarray | None = None
) -> dict[str, Any]:
    """Metrics of one set of answers, pooled and by question kind; ``skill`` is 1 - log loss / base-rate log loss."""
    p, y = _p(b, temperature), b["gold"]
    out: dict[str, Any] = BK.belief_metrics(p, y)
    ref = BK.base_rate_log_loss(y if skill_gold is None else skill_gold)
    out["skill"] = float(1.0 - out["log_loss"] / ref)
    out["by_kind"] = BK.by_kind(BK.belief_metrics, p, y, b["kind"])
    return out


def _loss(b: dict[str, np.ndarray], temperature: float = 1.0) -> np.ndarray:
    return BK.item_log_loss(_p(b, temperature), b["gold"])


# ---------------------------------------------------------------------------------------------
# one arm
# ---------------------------------------------------------------------------------------------


def _ckpt_dir(run: Path, which: str) -> Path:
    return run / f"eval_{which}"


def run_summary(run: Path) -> dict[str, Any]:
    info = _json(run / "run.json") or {}
    out: dict[str, Any] = {"run": run.name, "selected_iter": info.get("selected_iter")}
    for which in ("best", "last"):
        d = _ckpt_dir(run, which)
        games = _json(d / "games.json")
        if not games:
            continue
        block: dict[str, Any] = {
            "greedy": float(np.mean(games["greedy_scores"])),
            "sampled": float(np.mean(games["sampled_scores"])),
            "greedy_scores": games["greedy_scores"],
        }
        ev, sel = _npz(d / "beliefs_onpolicy_eval.npz"), _npz(d / "beliefs_onpolicy_select.npz")
        if ev is not None:
            block["onpolicy_raw"] = belief_block(ev)
            if sel is not None:
                t = fit_temperature_on(sel)
                block["temperature"] = t
                block["onpolicy_ts"] = belief_block(ev, t)
        common = _npz(d / "beliefs_common.npz")
        if common is not None:
            block["common_raw"] = belief_block(common)
            if sel is not None:
                block["common_ts"] = belief_block(common, block["temperature"])
        out[which] = block
    return out


def aggregate(runs: list[dict[str, Any]], which: str) -> dict[str, Any]:
    """Seed mean and range of every scalar belief and game metric of one checkpoint kind."""
    have = [r[which] for r in runs if which in r]
    if not have:
        return {}
    out: dict[str, Any] = {
        "greedy": mean_range([h["greedy"] for h in have]),
        "sampled": mean_range([h["sampled"] for h in have]),
    }
    for key in ("onpolicy_raw", "onpolicy_ts", "common_raw", "common_ts"):
        blocks = [h[key] for h in have if key in h]
        if blocks:
            out[key] = {
                m: mean_range([b[m] for b in blocks])
                for m in ("log_loss", "brier", "ece", "acc", "skill", "base_rate")
            }
            out[key]["n"] = blocks[0]["n"]
    return out


# ---------------------------------------------------------------------------------------------
# H7
# ---------------------------------------------------------------------------------------------


def _agg_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        m: mean_range([b[m] for b in blocks]) for m in ("log_loss", "brier", "ece", "acc", "skill")
    }
    out["n"] = blocks[0]["n"]
    out["base_rate"] = float(np.mean([b["base_rate"] for b in blocks]))
    return out


def h7(runs_dir: Path, delta: dict[str, Any], B: int = 2000) -> dict[str, Any]:
    """PREREGISTRATION.md Amendment 2, item 9: B3 (selected checkpoint, T = 1) against B4 (+TS) on the common states,
    and the game-score condition. Also B3 + TS against B4, B3 against B2 (B1 + TS) and B1, descriptively."""
    b3 = [_npz(runs_dir / f"b3_s{s}" / "eval_best" / "beliefs_common.npz") for s in SEEDS]
    b4, b4_t = [], []
    for s in SEEDS:
        c = _npz(runs_dir / f"b4_s{s}" / "eval" / "beliefs_common.npz")
        sel = _npz(runs_dir / f"b4_s{s}" / "eval" / "beliefs_sel_b1.npz")
        b4.append(c)
        b4_t.append(fit_temperature_on(sel) if sel is not None else None)
    if any(x is None for x in b3) or any(x is None for x in b4) or any(t is None for t in b4_t):
        return {"status": "incomplete"}
    b3_t = []
    for s in SEEDS:
        sel = _npz(runs_dir / f"b3_s{s}" / "eval_best" / "beliefs_onpolicy_select.npz")
        b3_t.append(fit_temperature_on(sel) if sel is not None else 1.0)
    l3 = np.stack([_loss(b) for b in b3])
    l3_ts = np.stack([_loss(b, t) for b, t in zip(b3, b3_t, strict=True)])
    l4 = np.stack([_loss(b, t) for b, t in zip(b4, b4_t, strict=True)])
    primary = BK.paired_bootstrap_diff(l3, l4, B)
    out: dict[str, Any] = {
        "common_states": int(l3.shape[1]),
        "b3_raw_vs_b4_ts": primary,
        "b3_ts_vs_b4_ts": BK.paired_bootstrap_diff(l3_ts, l4, B),
        "temperatures": {"b3": b3_t, "b4": b4_t},
        "belief_log_loss": {
            "b3_raw": mean_range([float(x.mean()) for x in l3]),
            "b3_ts": mean_range([float(x.mean()) for x in l3_ts]),
            "b4_ts": mean_range([float(x.mean()) for x in l4]),
        },
    }
    out["common_metrics"] = {
        "b3_raw": _agg_blocks([belief_block(b) for b in b3]),
        "b3_ts": _agg_blocks([belief_block(b, t) for b, t in zip(b3, b3_t, strict=True)]),
        "b4_ts": _agg_blocks([belief_block(b, t) for b, t in zip(b4, b4_t, strict=True)]),
    }
    b1 = [_npz(runs_dir / f"b1_s{s}" / "eval_best" / "beliefs_common.npz") for s in SEEDS]
    b1_sel = [
        _npz(runs_dir / f"b1_s{s}" / "eval_best" / "beliefs_onpolicy_select.npz") for s in SEEDS
    ]
    if all(x is not None for x in b1) and all(x is not None for x in b1_sel):
        t1 = [fit_temperature_on(x) for x in b1_sel]
        l1 = np.stack([_loss(b) for b in b1])
        l2 = np.stack([_loss(b, t) for b, t in zip(b1, t1, strict=True)])
        out["common_metrics"]["b1_raw"] = _agg_blocks([belief_block(b) for b in b1])
        out["common_metrics"]["b2"] = _agg_blocks(
            [belief_block(b, t) for b, t in zip(b1, t1, strict=True)]
        )
        out["b3_raw_vs_b1_raw"] = BK.paired_bootstrap_diff(l3, l1, B)
        out["b3_raw_vs_b2"] = BK.paired_bootstrap_diff(l3, l2, B)
        out["belief_log_loss"]["b1_raw"] = mean_range([float(x.mean()) for x in l1])
        out["belief_log_loss"]["b2"] = mean_range([float(x.mean()) for x in l2])
    games = {}
    for arm in ("b1", "b3"):
        scores = [
            _json(runs_dir / f"{arm}_s{s}" / "eval_best" / "games.json")["greedy_scores"]
            for s in SEEDS
        ]
        games[arm] = mean_range([float(np.mean(x)) for x in scores])
    margin = float(delta["delta_game"])
    game_ok = games["b3"]["mean"] >= games["b1"]["mean"] - margin
    out["greedy_score_eval_seeds"] = games
    out["delta_game"] = margin
    out["game_condition_met"] = bool(game_ok)
    belief_ok = bool(primary["ci95"][1] < 0)
    out["belief_condition_met"] = belief_ok
    out["h7_holds"] = bool(belief_ok and game_ok)
    out["status"] = "complete"
    return out


def analyse(runs_dir: str | Path, delta_game_json: str | Path, B: int = 2000) -> dict[str, Any]:
    runs_dir = Path(runs_dir)
    delta = json.loads(Path(delta_game_json).read_text())
    summaries: dict[str, list[dict[str, Any]]] = {}
    for arm, seeds in (("b1", SEEDS), ("b3", SEEDS), ("b5", (0,))):
        summaries[arm] = [
            run_summary(runs_dir / f"{arm}_s{s}")
            for s in seeds
            if (runs_dir / f"{arm}_s{s}").exists()
        ]
    out: dict[str, Any] = {"delta_game": delta, "runs": summaries, "arms": {}}
    for arm, runs in summaries.items():
        out["arms"][arm] = {w: aggregate(runs, w) for w in ("best", "last")}
    out["h7"] = h7(runs_dir, delta, B)
    return out
