"""Exp B figures from ``results/expB/analysis.json`` and the run files in ``results/expB/runs/``. Same palette per
arm as every other figure: filled marker raw, hollow marker after temperature scaling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from rlcd import breakout as BK
from rlcd.plots import ARM_STYLES, THEMES, ArmStyle, Theme, color, rc, save_figure
from rlcd.results import read_json

ARM_STYLES.setdefault("B2", ArmStyle("B2 B1 + temperature scaling", "improper", "--", "s"))
SEEDS = (0, 1, 2)


def _dot(
    ax: Any, x: float, s: dict[str, float], arm: str, theme: Theme, hollow: bool = False
) -> None:
    st, col = ARM_STYLES[arm], color(arm, theme)
    ax.plot([x, x], [s["min"], s["max"]], color=col, lw=1.4, solid_capstyle="round", zorder=2)
    ax.plot(
        [x],
        [s["mean"]],
        ls="",
        marker=st.marker,
        ms=8,
        mfc=theme.surface if hollow else col,
        mec=col,
        mew=1.5,
        zorder=3,
    )


def _legend(fig: Any, theme: Theme, arms: list[str], y: float, extra: bool = True) -> None:
    handles = [
        Line2D(
            [],
            [],
            color=color(a, theme),
            marker=ARM_STYLES[a].marker,
            ls="",
            ms=7,
            mec=theme.surface,
            mew=1.2,
        )
        for a in arms
    ]
    labels = [ARM_STYLES[a].label for a in arms]
    if extra:
        handles += [
            Line2D([], [], color=theme.ink2, marker="o", ls="", ms=7),
            Line2D([], [], color=theme.ink2, marker="o", ls="", ms=7, mfc=theme.surface, mew=1.5),
        ]
        labels += ["raw (T = 1)", "after temperature scaling"]
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper left",
        bbox_to_anchor=(0.06, y),
        columnspacing=1.6,
        handletextpad=0.4,
    )


def _title(fig: Any, theme: Theme, title: str, sub: str) -> None:
    fig.text(0.065, 0.955, title, fontsize=13.5, fontweight="bold", color=theme.ink)
    fig.text(0.065, 0.918, sub, fontsize=9.5, color=theme.ink2, va="top", linespacing=1.5)


def fig_beliefs(data: dict[str, Any], runs: Path, theme: Theme) -> Any:
    """Belief log loss, ECE and the calibration curve on the common states."""
    cm = data["h7"]["common_metrics"]
    order = [
        ("b1_raw", "B1", False),
        ("b2", "B2", True),
        ("b3_raw", "B3", False),
        ("b3_ts", "B3", True),
        ("b4_ts", "B4", True),
    ]
    xl = ["B1\nraw", "B2\nB1+TS", "B3\nraw", "B3\n+TS", "B4\n+TS"]
    with plt.rc_context(rc(theme)):
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.0))
        fig.subplots_adjust(left=0.065, right=0.985, top=0.7, bottom=0.14, wspace=0.28)
        for ax, metric, ylabel in (
            (axes[0], "log_loss", "belief log loss, lower is better"),
            (axes[1], "ece", "belief ECE, lower is better"),
        ):
            for i, (key, arm, hollow) in enumerate(order):
                _dot(ax, i, cm[key][metric], arm, theme, hollow)
            ax.set_xticks(range(len(order)), xl, fontsize=8.5)
            ax.set_xlim(-0.5, len(order) - 0.5)
            ax.set_ylabel(ylabel)
            ax.grid(axis="x", visible=False)
        # base-rate reference line on the log-loss panel
        common = _common(runs)
        if common is not None:
            ref = BK.base_rate_log_loss(common["gold"])
            axes[0].axhline(ref, color=theme.muted, lw=0.9, ls="--")
            axes[0].text(
                0.02,
                ref,
                " always the base rate",
                transform=axes[0].get_yaxis_transform(),
                va="bottom",
                fontsize=8,
                color=theme.muted,
            )
        axes[0].set_title("Belief log loss, common states")
        axes[1].set_title("Belief ECE, common states")
        # calibration curve
        ax = axes[2]
        ax.plot([0, 1], [0, 1], color=theme.muted, lw=0.9, ls="--")
        for arm, rel, hollow in (("B1", "b1", False), ("B3", "b3", False), ("B4", "b4", True)):
            pts = _curve(runs, rel)
            if pts is not None:
                xs, ys, ns = pts
                col = color(arm, theme)
                ax.plot(xs, ys, color=col, lw=1.6, ls="--" if hollow else "-")
                ax.scatter(
                    xs,
                    ys,
                    s=18 + 90 * ns / ns.max(),
                    color=theme.surface if hollow else col,
                    edgecolor=col,
                    linewidth=1.4,
                    zorder=3,
                )
        ax.set_xlabel("stated probability of yes")
        ax.set_ylabel("how often it happened")
        ax.set_title("Calibration curve (seeds pooled)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        _title(
            fig,
            theme,
            "Exp B: belief quality on the same states",
            "Every arm answers the same questions: B1 seed 0's final policy's sampled play on the evaluation seeds. Mean over 3 seeds, bars are the seed range.\nB4 is B1's policy with beliefs fitted by supervised learning, then temperature-scaled. H7 asks whether B3 (raw) beats B4.",
        )
        _legend(fig, theme, ["B1", "B2", "B3", "B4"], 0.85, extra=True)
    return fig


def _common(runs: Path) -> dict[str, np.ndarray] | None:
    p = runs / "b1_s0" / "eval_best" / "beliefs_common.npz"
    if not p.exists():
        return None
    with np.load(p) as f:
        return {k: f[k] for k in f.files}


def _curve(
    runs: Path, arm: str, bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    ps, ys = [], []
    for s in SEEDS:
        base = runs / f"{arm}_s{s}"
        if arm == "b4":
            p, tsrc = base / "eval" / "beliefs_common.npz", base / "eval" / "beliefs_sel_b1.npz"
        else:
            p, tsrc = base / "eval_best" / "beliefs_common.npz", None
        if not p.exists():
            continue
        with np.load(p) as f:
            d, gold = f["logit_diff"], f["gold"]
        T = 1.0
        if tsrc is not None and tsrc.exists():
            with np.load(tsrc) as g:
                T = BK.fit_belief_temperature(g["logit_diff"], g["gold"])
        ps.append(BK.p_yes_from_logit_diff(d, T))
        ys.append(gold)
    if not ps:
        return None
    p, y = np.concatenate(ps), np.concatenate(ys).astype(float)
    ids = np.clip((p * bins).astype(int), 0, bins - 1)
    xs, fr, ns = [], [], []
    for b in range(bins):
        m = ids == b
        if m.sum() >= 20:
            xs.append(p[m].mean())
            fr.append(y[m].mean())
            ns.append(m.sum())
    return np.array(xs), np.array(fr), np.array(ns, dtype=float)


def fig_games(data: dict[str, Any], theme: Theme) -> Any:
    """Greedy and sampled game scores of the selected and the final checkpoints, with the H7 margin around B1."""
    arms = [("B1", "b1"), ("B3", "b3"), ("B5", "b5")]
    h = data["h7"]
    with plt.rc_context(rc(theme)):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
        fig.subplots_adjust(left=0.07, right=0.985, top=0.7, bottom=0.15, wspace=0.22)
        for ax, kind, title, delta in (
            (axes[0], "greedy", "Greedy play (one trajectory per checkpoint)", h.get("delta_game")),
            (axes[1], "sampled", "Sampled play, 50 evaluation seeds", h.get("delta_sampled")),
        ):
            x = 0
            ticks, labels = [], []
            for arm, rel in arms:
                for which, name in (("best", "selected"), ("last", "final")):
                    s = data["arms"][rel].get(which, {}).get(kind)
                    if s:
                        _dot(ax, x, s, arm, theme, hollow=(which == "last"))
                        ticks.append(x)
                        labels.append(f"{arm}\n{name}")
                        x += 1
                x += 0.6
            b1 = data["arms"]["b1"].get("best", {}).get(kind)
            if b1 and delta:
                ax.axhspan(
                    b1["mean"] - delta, b1["mean"] + delta, color=theme.grid, alpha=0.55, zorder=0
                )
                ax.axhline(b1["mean"], color=theme.muted, lw=0.9)
                ax.text(
                    0.99,
                    b1["mean"] + delta,
                    f"B1 selected ± δ ({delta:.1f})",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color=theme.muted,
                )
            ax.set_xticks(ticks, labels, fontsize=8.5)
            ax.set_ylabel("Breakout score (300 decisions)")
            ax.set_title(title)
            ax.grid(axis="x", visible=False)
        _title(
            fig,
            theme,
            "Exp B: game score",
            "Filled: the checkpoint chosen on the selection seeds. Hollow: the final checkpoint (no selection). The band is the pre-registered margin δ\naround B1 (range of B1's seeds on the selection seeds): H7 needs B3's mean not below it.",
        )
    return fig


def fig_training(runs: Path, theme: Theme) -> Any:
    """Training score, the belief and retention terms, and the selection-seed greedy score over iterations."""
    with plt.rc_context(rc(theme)):
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.7))
        fig.subplots_adjust(left=0.065, right=0.985, top=0.7, bottom=0.14, wspace=0.28)
        for arm, rel in (("B1", "b1"), ("B3", "b3"), ("B5", "b5")):
            rows = _metrics(runs, rel)
            if not rows:
                continue
            col, st = color(arm, theme), ARM_STYLES[arm]
            it = np.array([r["iter"] for r in rows[0]])
            sc = np.array([[r["sampled_score"] for r in run] for run in rows])
            axes[0].plot(it, sc.mean(0), color=col, lw=1.8, ls=st.linestyle)
            axes[0].fill_between(it, sc.min(0), sc.max(0), color=col, alpha=0.15, lw=0)
            if arm != "B1":
                bce = np.array([[r["belief_ce"] for r in run] for run in rows])
                axes[1].plot(it, bce.mean(0), color=col, lw=1.8, ls=st.linestyle)
            if arm == "B3":
                rk = np.array([[r["retention_kl"] for r in run] for run in rows])
                axes[1].plot(it, 5 * rk.mean(0), color=col, lw=1.2, ls=":")
            ev = _sel_evals(runs, rel)
            if ev:
                x = np.array([e[0] for e in ev[0]])
                y = np.array([[e[1] for e in run] for run in ev])
                axes[2].plot(
                    x, y.mean(0), color=col, lw=1.8, ls=st.linestyle, marker=st.marker, ms=5
                )
        axes[0].set_title("Training score (sampled play)")
        axes[0].set_xlabel("iteration")
        axes[0].set_ylabel("mean Breakout score of the rollouts")
        axes[1].set_title("Belief log loss during training")
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("belief cross-entropy (dotted: retention KL × 5)")
        axes[2].set_title("Selection-seed greedy score")
        axes[2].set_xlabel("iteration (0 = start model)")
        axes[2].set_ylabel("one trajectory per evaluation")
        for ax in axes:
            ax.grid(axis="x", visible=False)
        _title(
            fig,
            theme,
            "Exp B: training",
            "Mean over seeds (shaded: range). B1 has no belief term; B5 has no retention term. The greedy score is one trajectory per evaluation,\nso it moves in steps.",
        )
        _legend(fig, theme, ["B1", "B3", "B5"], 0.85, extra=False)
    return fig


def _metrics(runs: Path, arm: str) -> list[list[dict[str, Any]]]:
    out = []
    for s in SEEDS:
        p = runs / f"{arm}_s{s}" / "metrics.jsonl"
        if p.exists():
            out.append([json.loads(x) for x in p.read_text().splitlines()])
    return out


def _sel_evals(runs: Path, arm: str) -> list[list[list[float]]]:
    out = []
    for s in SEEDS:
        p = runs / f"{arm}_s{s}" / "sel_evals.json"
        if p.exists():
            out.append(json.loads(p.read_text())["evals"])
    return out


def make_expB_figures(analysis: str | Path, runs: str | Path, fig_dir: str | Path) -> list[Path]:
    mpl.rcParams["svg.fonttype"] = "path"
    data = read_json(analysis)
    runs = Path(runs)
    out: list[Path] = []
    for theme in THEMES:
        for stem, fig in (
            ("expB_beliefs", fig_beliefs(data, runs, theme)),
            ("expB_games", fig_games(data, theme)),
            ("expB_training", fig_training(runs, theme)),
        ):
            out += save_figure(fig, stem, fig_dir, theme)
            plt.close(fig)
    return out


if __name__ == "__main__":
    import sys

    make_expB_figures(
        sys.argv[1] if len(sys.argv) > 1 else "results/expB/analysis.json",
        sys.argv[2] if len(sys.argv) > 2 else "results/expB/runs",
        sys.argv[3] if len(sys.argv) > 3 else "figures",
    )
