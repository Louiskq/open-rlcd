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


def selected_sampled_selection_score(run_dir: Path) -> float:
    """Mean sampled score on the 50 selection seeds of the run's selected checkpoint (Amendment 3, item 2)."""
    games = json.loads((run_dir / "eval_best" / "games.json").read_text())
    return float(np.mean(games["sampled_scores_select"]))


def compute_delta_game(runs_dir: Path, arm: str = "b1") -> dict[str, Any]:
    """delta_game (greedy, Amendment 2) and delta_sampled (Amendment 3) from the B1 runs' selection-seed scores."""
    names = [f"{arm}_s{s}" for s in SEEDS]
    greedy = [selected_selection_score(runs_dir / n) for n in names]
    out = BK.delta_game(greedy)
    out.update({"runs": names, "selected_selection_scores": greedy, "seeds": list(SEEDS)})
    sampled = [selected_sampled_selection_score(runs_dir / n) for n in names]
    ds = BK.delta_game(sampled)
    out["sampled"] = {
        "delta_sampled": ds["delta_game"],
        "range": ds["range"],
        "floor": ds["floor"],
        "mean": ds["mean"],
        "selected_sampled_selection_scores": sampled,
    }
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


def h7(runs_dir: Path, delta: dict[str, Any], B: int = 2000, ckpt: str = "last") -> dict[str, Any]:
    """PREREGISTRATION.md Amendment 2 item 9 as amended by Amendment 4: B3 (``ckpt``, by default the FINAL checkpoint,
    T = 1) against B4 (+TS) on the common states, and the greedy and sampled game-score conditions. Also B3 + TS
    against B4, B3 against B2 (B1 + TS) and B1, descriptively."""
    b3 = [_npz(runs_dir / f"b3_s{s}" / f"eval_{ckpt}" / "beliefs_common.npz") for s in SEEDS]
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
        sel = _npz(runs_dir / f"b3_s{s}" / f"eval_{ckpt}" / "beliefs_onpolicy_select.npz")
        b3_t.append(fit_temperature_on(sel) if sel is not None else 1.0)
    l3 = np.stack([_loss(b) for b in b3])
    l3_ts = np.stack([_loss(b, t) for b, t in zip(b3, b3_t, strict=True)])
    l4 = np.stack([_loss(b, t) for b, t in zip(b4, b4_t, strict=True)])
    l4_raw = np.stack([_loss(b) for b in b4])
    primary = BK.paired_bootstrap_diff(l3, l4, B)
    episodes = b3[0]["seed"]  # the episode (game seed) of each common-state question
    out: dict[str, Any] = {
        "common_states": int(l3.shape[1]),
        "b3_raw_vs_b4_ts": primary,
        # robustness: resample whole episodes, since questions within an episode are not independent
        "b3_raw_vs_b4_ts_clustered": BK.cluster_bootstrap_diff(l3, l4, episodes, B),
        "b3_raw_vs_b4_raw_clustered": BK.cluster_bootstrap_diff(l3, l4_raw, episodes, B),
        "b3_ts_vs_b4_ts": BK.paired_bootstrap_diff(l3_ts, l4, B),
        # descriptive: the sceptic's recipe without its temperature step (B4 + TS is the pre-registered comparator)
        "b3_raw_vs_b4_raw": BK.paired_bootstrap_diff(l3, l4_raw, B),
        "temperatures": {"b3": b3_t, "b4": b4_t},
        # the bootstrap resamples questions, not seeds: with 3 seeds per arm the per-seed values are the honest spread
        "per_seed_log_loss": {
            "b3_raw": [float(x.mean()) for x in l3],
            "b3_ts": [float(x.mean()) for x in l3_ts],
            "b4_ts": [float(x.mean()) for x in l4],
            "b4_raw": [float(x.mean()) for x in l4_raw],
        },
        "belief_log_loss": {
            "b3_raw": mean_range([float(x.mean()) for x in l3]),
            "b3_ts": mean_range([float(x.mean()) for x in l3_ts]),
            "b4_ts": mean_range([float(x.mean()) for x in l4]),
            "b4_raw": mean_range([float(x.mean()) for x in l4_raw]),
        },
    }
    out["common_metrics"] = {
        "b3_raw": _agg_blocks([belief_block(b) for b in b3]),
        "b3_ts": _agg_blocks([belief_block(b, t) for b, t in zip(b3, b3_t, strict=True)]),
        "b4_ts": _agg_blocks([belief_block(b, t) for b, t in zip(b4, b4_t, strict=True)]),
        "b4_raw": _agg_blocks([belief_block(b) for b in b4]),
    }
    b1 = [_npz(runs_dir / f"b1_s{s}" / f"eval_{ckpt}" / "beliefs_common.npz") for s in SEEDS]
    b1_sel = [
        _npz(runs_dir / f"b1_s{s}" / f"eval_{ckpt}" / "beliefs_onpolicy_select.npz") for s in SEEDS
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
            _json(runs_dir / f"{arm}_s{s}" / f"eval_{ckpt}" / "games.json")["greedy_scores"]
            for s in SEEDS
        ]
        games[arm] = mean_range([float(np.mean(x)) for x in scores])
    margin = float(delta["delta_game"])
    greedy_ok = games["b3"]["mean"] >= games["b1"]["mean"] - margin
    sampled_games = {}
    for arm in ("b1", "b3"):
        sc = [
            _json(runs_dir / f"{arm}_s{s}" / f"eval_{ckpt}" / "games.json")["sampled_scores"]
            for s in SEEDS
        ]
        sampled_games[arm] = mean_range([float(np.mean(x)) for x in sc])
    ds = delta.get("sampled", {}).get("delta_sampled")
    sampled_ok = (
        True
        if ds is None
        else sampled_games["b3"]["mean"] >= sampled_games["b1"]["mean"] - float(ds)
    )
    game_ok = bool(greedy_ok and sampled_ok)
    out["greedy_score_eval_seeds"] = games
    out["sampled_score_eval_seeds"] = sampled_games
    out["delta_game"] = margin
    out["delta_sampled"] = ds
    out["greedy_condition_met"] = bool(greedy_ok)
    out["sampled_condition_met"] = bool(sampled_ok)
    out["game_condition_met"] = game_ok
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
    out["h7"] = h7(runs_dir, delta, B, "last")  # pre-registered (Amendment 4): final checkpoints
    out["h7_selected"] = h7(
        runs_dir, delta, B, "best"
    )  # descriptive: the selected checkpoints (may be the start model)
    return out
