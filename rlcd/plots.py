"""Figures with one consistent look: a palette per arm used everywhere, PNG + SVG, light + dark.

Colour encodes the *method family*, so the same arm has the same colour in every figure and the
colour means the same thing across experiments:

    blue     RLCD (bandit or joint on-policy): the claim
    orange   improper / plain RL: RLVR in Exp A, plain PPO in Exp B
    aqua     supervised log-loss control
    yellow   cynical RLCD (mining + proper loss, no RL)    magenta  its random-selection control
    green    RLCD-full through the RL code path             violet   ablations
    grey     start model and reference lines

Variants inside a family use line style and marker, never a new hue: dashed = clipped importance
weights, hollow marker = after temperature scaling. The hues are the dataviz skill's validated
categorical order (first four slots checked in light and dark); aqua and yellow fall below 3:1
on the light surface, so every figure carries a legend *and* direct labels as the relief channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from rlcd.results import read_json

# ---------------------------------------------------------------------------------------------
# theme and palette
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str  # primary text
    ink2: str  # secondary text
    muted: str  # axis labels, ticks
    grid: str  # hairline grid
    axis: str  # baseline / axis rule

    @property
    def suffix(self) -> str:
        return "" if self.name == "light" else f"-{self.name}"


LIGHT = Theme("light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
DARK = Theme("dark", "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#383835")
THEMES = (LIGHT, DARK)

# family -> (light, dark)
FAMILIES: dict[str, tuple[str, str]] = {
    "rlcd": ("#2a78d6", "#3987e5"),
    "improper": ("#eb6834", "#d95926"),
    "supervised": ("#1baf7a", "#199e70"),
    "cynical": ("#eda100", "#c98500"),
    "cynical_control": ("#e87ba4", "#d55181"),
    "rlcd_full": ("#008300", "#008300"),
    "ablation": ("#4a3aa7", "#9085e9"),
    "reference": ("#898781", "#898781"),
}


@dataclass(frozen=True)
class ArmStyle:
    label: str
    family: str
    linestyle: str = "-"  # dashed = clipped weights
    marker: str = "o"


ARM_STYLES: dict[str, ArmStyle] = {
    # Exp 0
    "supervised": ArmStyle("Supervised log loss", "supervised"),
    "rlcd_full": ArmStyle("RLCD-full", "rlcd_full"),
    "rlcd_bandit": ArmStyle("RLCD-bandit", "rlcd"),
    "rlcd_bandit_clip10": ArmStyle("RLCD-bandit, clip 10", "rlcd", "--", "s"),
    "rlvr": ArmStyle("RLVR", "improper"),
    # Exp A (registered now so the palette is fixed before those figures exist)
    "A0": ArmStyle("A0 start model", "reference"),
    "A1": ArmStyle("A1 supervised", "supervised"),
    "A3": ArmStyle("A3 RLVR", "improper"),
    "A4": ArmStyle("A4 RLCD-bandit", "rlcd"),
    "A5": ArmStyle("A5 RLCD-full", "rlcd_full"),
    "A6": ArmStyle("A6 cynical RLCD", "cynical"),
    "A6r": ArmStyle("A6r cynical, random", "cynical_control"),
    # Exp B
    "B1": ArmStyle("B1 plain PPO", "improper"),
    "B3": ArmStyle("B3 RLCD", "rlcd"),
    "B4": ArmStyle("B4 cynical", "cynical"),
    "B5": ArmStyle("B5 no retention", "rlcd", ":", "^"),
}


def color(key: str, theme: Theme) -> str:
    """Colour of an arm id (or a family name) on this theme's surface."""
    family = ARM_STYLES[key].family if key in ARM_STYLES else key
    light, dark = FAMILIES[family]
    return light if theme.name == "light" else dark


def rc(theme: Theme) -> dict[str, Any]:
    """Matplotlib rcParams for a theme: recessive chrome, thin marks, left-aligned titles."""
    return {
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "savefig.edgecolor": "none",
        "axes.edgecolor": theme.axis,
        "axes.linewidth": 0.8,
        "axes.labelcolor": theme.ink2,
        "axes.labelsize": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 9,
        "axes.titlecolor": theme.ink,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": theme.grid,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "text.color": theme.ink,
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": theme.ink2,
        "svg.fonttype": "path",
        "figure.dpi": 100,
        "savefig.dpi": 200,
    }


def save_figure(fig: Any, stem: str, out_dir: str | Path, theme: Theme) -> list[Path]:
    """Write ``stem[-dark].png`` and ``.svg`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}{theme.suffix}.{ext}" for ext in ("png", "svg")]
    for p in paths:
        fig.savefig(p, bbox_inches=None)
    return paths


def _hline_label(
    ax: Any, y: float, text: str, theme: Theme, x: float = 0.995, ha: str = "right"
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.get_yaxis_transform(),
        ha=ha,
        va="bottom",
        fontsize=8,
        color=theme.muted,
    )


def _log_steps(ax: Any, last: int) -> None:
    ax.set_xscale("log")
    ax.set_xlim(1, last)
    ax.set_xticks([1, 10, 100, 1000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("training step")


# ---------------------------------------------------------------------------------------------
# Exp 0 data
# ---------------------------------------------------------------------------------------------

TOY_ARMS = ("supervised", "rlcd_full", "rlcd_bandit", "rlcd_bandit_clip10", "rlvr")


def load_toy(res_dir: str | Path) -> dict[str, Any]:
    """Every Exp 0 result file as one dict: arms by id, plus meta, lr_sensitivity and mix_sweep."""
    res_dir = Path(res_dir)
    out: dict[str, Any] = {name: read_json(res_dir / f"{name}.json") for name in TOY_ARMS}
    for extra in ("meta", "lr_sensitivity", "mix_sweep"):
        out[extra] = read_json(res_dir / f"{extra}.json")
    return out


def _curve(
    arm: dict[str, Any], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Steps (without 0), seed mean, min and max of a per-step metric."""
    steps = np.asarray(arm["steps"])
    c = np.asarray(arm["curves"][metric], dtype=float)
    keep = steps > 0
    return steps[keep], c.mean(0)[keep], c.min(0)[keep], c.max(0)[keep]


def _plot_curve(ax: Any, arm_id: str, arm: dict[str, Any], metric: str, theme: Theme) -> None:
    st = ARM_STYLES[arm_id]
    x, mean, lo, hi = _curve(arm, metric)
    col = color(arm_id, theme)
    ax.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0)
    ax.plot(x, mean, color=col, lw=2.0, ls=st.linestyle, solid_capstyle="round")
    ax.plot(
        [x[-1]], [mean[-1]], marker=st.marker, ms=6.5, color=col, mec=theme.surface, mew=1.5, ls=""
    )


MIN_BIN_MASS = 0.002  # bins holding less of the pooled mass than this are not drawn (too noisy)


def _bin_points(rel: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf, acc, count = (np.asarray(rel[k], dtype=float) for k in ("conf", "acc", "count"))
    keep = count >= MIN_BIN_MASS * count.sum()
    return conf[keep], acc[keep], count


def hist_inset(ax: Any, n_bins: int, theme: Theme) -> Any:
    """A confidence-histogram strip under ``ax``, a child of it so the two always line up."""
    axh = ax.inset_axes([0, -0.27, 1, 0.19])
    axh.set_xlim(0, 1)
    axh.set_ylim(0, 1)
    axh.set_xticks([0, 0.5, 1])
    axh.tick_params(labelsize=8.5)
    axh.set_yticks([])
    axh.grid(False)
    axh.spines["left"].set_visible(False)
    axh.set_xlabel("confidence", labelpad=3)
    return axh


def style_reliability_axes(ax: Any) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelbottom=False)
    ax.grid(axis="x", visible=False)


def reliability_panel(
    ax: Any,
    rel: dict[str, Any],
    arm_id: str,
    theme: Theme,
    rel_ts: dict[str, Any] | None = None,
) -> None:
    """Top-label reliability diagram (filled = raw, hollow = after temperature scaling) over a
    confidence histogram that shows how sharp the forecasts are."""
    col = color(arm_id, theme)
    ax.plot([0, 1], [0, 1], color=theme.muted, lw=0.9, zorder=1)
    conf, acc, count = _bin_points(rel)
    ax.plot(conf, acc, color=col, lw=1.6, zorder=2)
    ax.plot(conf, acc, ls="", marker="o", ms=6.5, color=col, mec=theme.surface, mew=1.4, zorder=3)
    axh = hist_inset(ax, len(count), theme)
    centers = (np.arange(len(count)) + 0.5) / len(count)
    axh.bar(centers, count / count.sum(), width=0.8 / len(count), color=col, alpha=0.55, lw=0)
    tallest = count.max() / count.sum()
    if rel_ts is not None:
        c2, a2, n2 = _bin_points(rel_ts)
        ax.plot(c2, a2, color=col, lw=1.2, ls=(0, (1, 2)), zorder=2)
        ax.plot(c2, a2, ls="", marker="o", ms=6.5, mfc=theme.surface, mec=col, mew=1.6, zorder=4)
        axh.bar(centers, n2 / n2.sum(), width=0.8 / len(n2), fill=False, ec=col, lw=0.9)
        tallest = max(tallest, n2.max() / n2.sum())
    axh.set_ylim(0, max(tallest, 1e-9) * 1.08)
    style_reliability_axes(ax)


# ---------------------------------------------------------------------------------------------
# Figure 1: Exp 0 overview
# ---------------------------------------------------------------------------------------------

OVERVIEW_PANELS = ("rlvr", "rlcd_bandit", "rlcd_bandit_clip10", "supervised")
PANEL_TITLES = {
    "rlvr": "RLVR",
    "rlcd_bandit": "RLCD-bandit",
    "rlcd_bandit_clip10": "RLCD-bandit, clip 10",
    "supervised": "Supervised = RLCD-full",
}


def fig_toy_overview(data: dict[str, Any], theme: Theme) -> Any:
    """ECE and accuracy against training step, and a reliability diagram per arm."""
    meta = data["meta"]
    n_seeds = len(meta["seed_sets"]["train"])
    with plt.rc_context(rc(theme)):
        fig = plt.figure(figsize=(11.2, 7.8))
        outer = fig.add_gridspec(
            2,
            4,
            height_ratios=[1.0, 1.15],
            left=0.06,
            right=0.985,
            top=0.815,
            bottom=0.215,
            hspace=0.38,
            wspace=0.28,
        )
        ax_ece = fig.add_subplot(outer[0, 0:2])
        ax_acc = fig.add_subplot(outer[0, 2:4])
        last = max(data["rlvr"]["steps"])

        # ECE vs step
        for arm_id in ("rlvr", "rlcd_bandit_clip10", "rlcd_bandit", "supervised"):
            _plot_curve(ax_ece, arm_id, data[arm_id], "ece", theme)
        floor = meta["oracle"]["ece"]
        ax_ece.axhline(floor, color=theme.muted, lw=0.9)
        _hline_label(
            ax_ece,
            floor + 0.006,
            f"noise floor {floor:.3f}",
            theme,
            x=0.012,
            ha="left",
        )
        ax_ece.set_title("Calibration error (top-label ECE, 15 bins)")
        ax_ece.set_ylabel("ECE, lower is better")
        ax_ece.set_ylim(0, 0.43)
        _log_steps(ax_ece, last)
        for arm_id, dy, name in (("rlvr", 0.045, "RLVR"), ("rlcd_bandit_clip10", 0.04, "clip 10")):
            x, mean, *_ = _curve(data[arm_id], "ece")
            ax_ece.text(
                x[-1] * 0.9,
                mean[-1] + dy,
                name,
                ha="right",
                va="bottom",
                fontsize=9,
                color=theme.ink2,
            )

        # accuracy vs step
        for arm_id in ("rlvr", "rlcd_bandit_clip10", "rlcd_bandit", "supervised"):
            _plot_curve(ax_acc, arm_id, data[arm_id], "acc", theme)
        bayes = meta["oracle"]["bayes_acc"]
        ax_acc.axhline(bayes, color=theme.muted, lw=0.9)
        _hline_label(ax_acc, bayes + 0.012, f"Bayes-optimal accuracy, {bayes:.3f}", theme)
        ax_acc.set_title("Accuracy is the same for every arm")
        ax_acc.set_ylabel("accuracy on sampled labels")
        ax_acc.set_ylim(0.15, 0.58)
        _log_steps(ax_acc, last)

        # reliability row
        for j, arm_id in enumerate(OVERVIEW_PANELS):
            ax = fig.add_subplot(outer[1, j])
            rel = data[arm_id]["reliability_final"]
            reliability_panel(ax, rel["raw"], arm_id, theme, rel["ts"])
            fin = meta["summary"][arm_id]
            ax.set_title(PANEL_TITLES[arm_id], fontsize=10)
            ax.text(
                0.97,
                0.05,
                f"ECE {fin['raw']['ece']['mean']:.3f}\n{fin['ts']['ece']['mean']:.3f} at "
                f"T = {fin['temperature']['mean']:.1f}",
                ha="right",
                va="bottom",
                fontsize=8.5,
                color=theme.ink2,
                transform=ax.transAxes,
            )
            if j == 0:
                ax.set_ylabel("accuracy in bin")
        fig.text(
            0.06,
            0.953,
            "Rewarding a model for being right makes it overconfident",
            fontsize=15,
            fontweight="bold",
            color=theme.ink,
        )
        fig.text(
            0.06,
            0.924,
            "Exp 0: a linear-softmax policy over 5 options with known true probabilities. "
            "Only the reward differs.",
            fontsize=10,
            color=theme.ink2,
        )
        handles = [
            Line2D(
                [],
                [],
                color=color(a, theme),
                lw=2.0,
                ls=ARM_STYLES[a].linestyle,
                marker=ARM_STYLES[a].marker,
                ms=6.5,
                mec=theme.surface,
                mew=1.5,
            )
            for a in ("rlvr", "rlcd_bandit", "rlcd_bandit_clip10", "supervised")
        ]
        labels = ["RLVR", "RLCD-bandit", "RLCD-bandit, clip 10", "Supervised log loss = RLCD-full"]
        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="upper left",
            bbox_to_anchor=(0.055, 0.898),
            handlelength=2.4,
            columnspacing=1.8,
        )
        fig.text(
            0.06,
            0.014,
            f"{n_seeds} seeds, shaded = seed range; {meta['config']['n_eval']:,} held-out "
            "contexts. "
            "All arms start underconfident from a uniform policy. Reliability row: filled = raw "
            "policy, hollow = after temperature\nscaling (one temperature, fitted on a separate "
            "calibration set); bins with under 0.2% of the predictions are not drawn.",
            fontsize=8,
            color=theme.muted,
            linespacing=1.5,
        )
    return fig


# ---------------------------------------------------------------------------------------------
# Figure 2: the RLVR share sweep
# ---------------------------------------------------------------------------------------------


def fig_toy_mix(data: dict[str, Any], theme: Theme) -> Any:
    """Final ECE and log loss as a share of the RLVR reward is mixed into RLCD-bandit."""
    sweep, meta = data["mix_sweep"], data["meta"]
    mixes = np.asarray(sweep["mixes"], dtype=float)
    col = color("rlcd", theme)
    with plt.rc_context(rc(theme)):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
        fig.subplots_adjust(left=0.075, right=0.985, top=0.72, bottom=0.17, wspace=0.26)
        for ax, metric, title, ylabel in (
            (axes[0], "ece", "Calibration error", "ECE, lower is better"),
            (axes[1], "log_loss", "Log loss", "log loss, lower is better"),
        ):
            raw = [sweep["summary"][f"{m:g}"]["raw"][metric] for m in mixes]
            ts = [sweep["summary"][f"{m:g}"]["ts"][metric] for m in mixes]
            mean = np.array([r["mean"] for r in raw])
            ax.fill_between(
                mixes, [r["min"] for r in raw], [r["max"] for r in raw], color=col, alpha=0.12, lw=0
            )
            ax.plot(mixes, mean, color=col, lw=2.0)
            ax.plot(mixes, mean, ls="", marker="o", ms=6.5, color=col, mec=theme.surface, mew=1.5)
            ax.plot(mixes, [t["mean"] for t in ts], ls=(0, (1, 2)), lw=1.2, color=col)
            ax.plot(
                mixes,
                [t["mean"] for t in ts],
                ls="",
                marker="o",
                ms=6.5,
                mfc=theme.surface,
                mec=col,
                mew=1.6,
            )
            floor = meta["oracle"][metric]
            ax.axhline(floor, color=theme.muted, lw=0.9)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("share of RLVR reward mixed into RLCD-bandit")
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_ylim(bottom=0 if metric == "ece" else 1.1)
        handles = [
            Line2D([], [], color=col, lw=2.0, marker="o", ms=6.5, mec=theme.surface, mew=1.5),
            Line2D(
                [],
                [],
                color=col,
                lw=1.2,
                ls=(0, (1, 2)),
                marker="o",
                ms=6.5,
                mfc=theme.surface,
                mec=col,
                mew=1.6,
            ),
            Line2D([], [], color=theme.muted, lw=0.9),
        ]
        fig.legend(
            handles,
            ["raw policy", "after temperature scaling", "true probabilities"],
            ncol=3,
            loc="upper left",
            bbox_to_anchor=(0.068, 0.865),
            handlelength=2.4,
            columnspacing=1.8,
        )
        fig.text(
            0.075,
            0.935,
            "The more RLVR in the reward, the worse the calibration",
            fontsize=14,
            fontweight="bold",
            color=theme.ink,
        )
        fig.text(
            0.075,
            0.885,
            "reward = (1 - m) * 1[a=y]/p(a) + m * 1[a=y], so m = 0 is RLCD-bandit and m = 1 "
            "is RLVR",
            fontsize=9.5,
            color=theme.ink2,
        )
    return fig


# ---------------------------------------------------------------------------------------------
# notebook: explore the RLVR share
# ---------------------------------------------------------------------------------------------


def blend(c0: str, c1: str, t: float) -> tuple[float, float, float]:
    """Linear RGB blend from colour ``c0`` (t = 0) to ``c1`` (t = 1)."""
    a, b = np.array(mpl.colors.to_rgb(c0)), np.array(mpl.colors.to_rgb(c1))
    return tuple(float(v) for v in (1 - t) * a + t * b)


def _mean_ece(arm: dict[str, Any]) -> np.ndarray:
    return np.asarray(arm["curves"]["ece"], dtype=float).mean(axis=0)


def fig_toy_explorer(
    arms: dict[float, dict[str, Any]], mix: float, step_idx: int, theme: Theme = LIGHT
) -> Any:
    """Reliability diagram at one training step for one RLVR share, beside every share's ECE curve.

    ``arms`` maps an RLVR share to its arm JSON (``toy.run.arm_json``). Blue is pure RLCD-bandit,
    orange is pure RLVR, and mixtures blend between them.
    """
    arm = arms[mix]
    steps = np.asarray(arm["steps"])
    col = blend(color("rlcd", theme), color("rlvr", theme), mix)
    with plt.rc_context(rc(theme)):
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.9))
        fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.25, wspace=0.28)
        rel = {k: arm["reliability"][k][step_idx] for k in ("count", "conf", "acc")}
        conf, acc, count = _bin_points(rel)
        ax.plot([0, 1], [0, 1], color=theme.muted, lw=0.9, zorder=1)
        ax.plot(conf, acc, color=col, lw=1.8, zorder=2)
        ax.plot(conf, acc, ls="", marker="o", ms=7, color=col, mec=theme.surface, mew=1.5, zorder=3)
        axh = hist_inset(ax, len(count), theme)
        centers = (np.arange(len(count)) + 0.5) / len(count)
        axh.bar(centers, count / count.sum(), width=0.8 / len(count), color=col, alpha=0.6, lw=0)
        axh.set_ylim(0, 0.5)
        style_reliability_axes(ax)
        ax.set_ylabel("accuracy in bin")
        ece_now = _mean_ece(arm)[step_idx]
        ax.set_title(f"{mix:.0%} RLVR, step {steps[step_idx]:,}")
        ax.text(
            0.97,
            0.05,
            f"ECE {ece_now:.3f}",
            ha="right",
            va="bottom",
            fontsize=11,
            color=theme.ink,
            transform=ax.transAxes,
        )

        for other in arms.values():
            keep = np.asarray(other["steps"]) > 0
            ax2.plot(
                np.asarray(other["steps"])[keep],
                _mean_ece(other)[keep],
                color=theme.axis,
                lw=1.0,
                zorder=1,
            )
        keep = steps > 0
        ax2.plot(steps[keep], _mean_ece(arm)[keep], color=col, lw=2.2, zorder=3)
        if steps[step_idx] > 0:
            ax2.plot(
                [steps[step_idx]],
                [ece_now],
                marker="o",
                ms=8,
                color=col,
                mec=theme.surface,
                mew=1.8,
                zorder=4,
            )
        _log_steps(ax2, int(steps[-1]))
        ax2.set_ylim(0, 0.43)
        ax2.set_ylabel("ECE, lower is better")
        ax2.set_title("Calibration error through training (grey: other shares)")
        fig.text(
            0.07,
            0.935,
            "Mix RLVR into the reward and watch calibration break",
            fontsize=13,
            fontweight="bold",
            color=theme.ink,
        )
    return fig


def fig_toy_gallery(
    arms: dict[float, dict[str, Any]], mixes: tuple[float, ...], theme: Theme = LIGHT
) -> Any:
    """Final reliability diagrams for a few RLVR shares side by side (a static slider view)."""
    with plt.rc_context(rc(theme)):
        fig, axes = plt.subplots(1, len(mixes), figsize=(2.55 * len(mixes), 3.7))
        fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.30, wspace=0.18)
        for ax, m in zip(np.atleast_1d(axes), mixes, strict=True):
            arm = arms[m]
            col = blend(color("rlcd", theme), color("rlvr", theme), m)
            rel = {k: arm["reliability"][k][-1] for k in ("count", "conf", "acc")}
            conf, acc, count = _bin_points(rel)
            ax.plot([0, 1], [0, 1], color=theme.muted, lw=0.9, zorder=1)
            ax.plot(conf, acc, color=col, lw=1.6, zorder=2)
            ax.plot(
                conf, acc, ls="", marker="o", ms=6, color=col, mec=theme.surface, mew=1.4, zorder=3
            )
            axh = hist_inset(ax, len(count), theme)
            centers = (np.arange(len(count)) + 0.5) / len(count)
            axh.bar(
                centers, count / count.sum(), width=0.8 / len(count), color=col, alpha=0.6, lw=0
            )
            axh.set_ylim(0, 0.5)
            style_reliability_axes(ax)
            ax.set_title(f"{m:.0%} RLVR", fontsize=10)
            ax.text(
                0.97,
                0.05,
                f"ECE {_mean_ece(arm)[-1]:.3f}",
                ha="right",
                va="bottom",
                fontsize=9,
                color=theme.ink2,
                transform=ax.transAxes,
            )
        np.atleast_1d(axes)[0].set_ylabel("accuracy in bin")
        fig.text(
            0.05,
            0.94,
            "From RLCD to RLVR: the reliability diagram tilts off the diagonal",
            fontsize=12.5,
            fontweight="bold",
            color=theme.ink,
        )
    return fig


# ---------------------------------------------------------------------------------------------
# Exp 0b (exploratory)
# ---------------------------------------------------------------------------------------------

TOY0B_ARMS = ("rlvr", "rlcd_bandit", "rlcd_bandit_clip10", "supervised")


def load_toy0b(res_dir: str | Path) -> dict[str, Any]:
    res_dir = Path(res_dir)
    out: dict[str, Any] = {name: read_json(res_dir / f"{name}.json") for name in TOY_ARMS}
    out["meta"] = read_json(res_dir / "meta.json")
    return out


def _range_dot(
    ax: Any, x: float, s: dict[str, float], arm_id: str, theme: Theme, **kw: Any
) -> None:
    st = ARM_STYLES[arm_id]
    col = color(arm_id, theme)
    ax.plot([x, x], [s["min"], s["max"]], color=col, lw=1.4, solid_capstyle="round", zorder=2)
    ax.plot(
        [x],
        [s["mean"]],
        ls="",
        marker=st.marker,
        ms=8,
        color=col,
        mec=theme.surface,
        mew=1.6,
        zorder=3,
        **kw,
    )


def fig_toy0b(data: dict[str, Any], theme: Theme) -> Any:
    """After one temperature: pooled ECE, calibration by difficulty tercile, and resolution."""
    meta = data["meta"]
    summ, grp, oracle = meta["summary"], meta["by_difficulty"], meta["oracle"]
    n_seeds = len(meta["seed_sets"]["train"])
    labels = [ARM_STYLES[a].label.replace(" log loss", "") for a in TOY0B_ARMS]
    with plt.rc_context(rc(theme)):
        fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.9))
        fig.subplots_adjust(left=0.065, right=0.985, top=0.68, bottom=0.22, wspace=0.28)
        ax1, ax2, ax3 = axes

        # 1. pooled ECE after one global temperature
        for i, arm in enumerate(TOY0B_ARMS):
            _range_dot(ax1, i, summ[arm]["ts"]["ece"], arm, theme)
        ax1.axhline(oracle["ece"], color=theme.muted, lw=0.9)
        ax1.text(
            0.012,
            oracle["ece"] - 0.0003,
            f"noise floor {oracle['ece']:.3f}",
            transform=ax1.get_yaxis_transform(),
            ha="left",
            va="top",
            fontsize=8,
            color=theme.muted,
        )
        ax1.set_xticks(range(len(TOY0B_ARMS)), labels, fontsize=8, rotation=18, ha="right")
        ax1.set_xlim(-0.5, len(TOY0B_ARMS) - 0.5)
        ax1.set_ylim(0, 0.02)
        ax1.set_ylabel("pooled ECE, lower is better")
        ax1.set_title("Pooled ECE looks fixed")
        ax1.grid(axis="x", visible=False)

        # 2. signed confidence gap by difficulty tercile, after the same temperature
        names = grp[TOY0B_ARMS[0]]["names"]
        for arm in TOY0B_ARMS:
            col, st = color(arm, theme), ARM_STYLES[arm]
            gaps = grp[arm]["ts"]["gap"]
            ax2.plot(
                range(3), [g["mean"] for g in gaps], color=col, lw=2.0, ls=st.linestyle, zorder=2
            )
            for j, g in enumerate(gaps):
                _range_dot(ax2, j, g, arm, theme)
        ax2.axhline(0, color=theme.muted, lw=0.9)
        ax2.set_xticks(range(3), list(names), fontsize=9)
        ax2.set_xlabel("input difficulty (hard = true probabilities near-uniform)")
        ax2.set_xlim(-0.35, 2.35)
        ax2.set_ylabel("mean confidence \u2212 accuracy")
        ax2.set_title("by difficulty it is not")
        ax2.grid(axis="x", visible=False)
        ax2.text(
            0.98,
            0.97,
            "overconfident",
            transform=ax2.transAxes,
            fontsize=8,
            color=theme.muted,
            va="top",
            ha="right",
        )
        ax2.text(
            0.02, 0.03, "underconfident", transform=ax2.transAxes, fontsize=8, color=theme.muted
        )

        # 3. resolution after one temperature
        for i, arm in enumerate(TOY0B_ARMS):
            _range_dot(ax3, i, summ[arm]["ts"]["resolution"], arm, theme)
        ax3.axhline(oracle["resolution"], color=theme.muted, lw=0.9)
        _hline_label(
            ax3, oracle["resolution"] + 0.0003, "true probabilities", theme, x=0.99, ha="right"
        )
        ax3.set_xticks(range(len(TOY0B_ARMS)), labels, fontsize=8, rotation=18, ha="right")
        ax3.set_xlim(-0.5, len(TOY0B_ARMS) - 0.5)
        lo = min(summ[a]["ts"]["resolution"]["min"] for a in TOY0B_ARMS)
        ax3.set_ylim(lo - 0.006, oracle["resolution"] + 0.004)
        ax3.set_ylabel("Murphy resolution, higher is better")
        ax3.set_title("and resolution is lower")
        ax3.grid(axis="x", visible=False)

        fig.text(
            0.065,
            0.955,
            "Exploratory: pooled ECE after one temperature hides what RLVR got wrong",
            fontsize=13.5,
            fontweight="bold",
            color=theme.ink,
        )
        fig.text(
            0.065,
            0.915,
            "Exp 0b. The scale of the true logits changes with the input; the policy can represent "
            "the truth exactly.\nAll numbers are after one temperature fitted on a separate "
            "calibration set.",
            fontsize=9.5,
            color=theme.ink2,
            va="top",
            linespacing=1.5,
        )
        handles = [
            Line2D(
                [],
                [],
                color=color(a, theme),
                lw=2.0,
                ls=ARM_STYLES[a].linestyle,
                marker=ARM_STYLES[a].marker,
                ms=7,
                mec=theme.surface,
                mew=1.5,
            )
            for a in TOY0B_ARMS
        ]
        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="upper left",
            bbox_to_anchor=(0.06, 0.835),
            handlelength=2.4,
            columnspacing=1.8,
        )
        fig.text(
            0.065,
            0.025,
            f"Not pre-registered. {n_seeds} seeds; bars are the seed range. "
            "Supervised log loss and RLCD-full coincide.",
            fontsize=8,
            color=theme.muted,
        )
    return fig


def make_toy0b_figures(res_dir: str | Path, fig_dir: str | Path) -> list[Path]:
    mpl.rcParams["svg.fonttype"] = "path"
    data = load_toy0b(res_dir)
    out: list[Path] = []
    for theme in THEMES:
        fig = fig_toy0b(data, theme)
        out += save_figure(fig, "toy0b", fig_dir, theme)
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------------------------
# Hero animation
# ---------------------------------------------------------------------------------------------


def hero_animation(
    data: dict[str, Any], theme: Theme, path: str | Path, fps: int = 10, hold: int = 14
) -> Path:
    """Side-by-side reliability diagrams through training: RLVR overshoots, RLCD settles."""
    arms = ("rlvr", "rlcd_bandit")
    titles = ("RLVR", "RLCD-bandit")
    steps = data["rlvr"]["reliability"]["steps"]
    n_frames = len(steps)
    with plt.rc_context(rc(theme)):
        fig = plt.figure(figsize=(8.0, 5.8), dpi=100)
        outer = fig.add_gridspec(1, 2, left=0.085, right=0.975, top=0.79, bottom=0.32, wspace=0.18)
        lines, dots, bars, eces = [], [], [], []
        for j, arm_id in enumerate(arms):
            ax = fig.add_subplot(outer[0, j])
            col = color(arm_id, theme)
            ax.plot([0, 1], [0, 1], color=theme.muted, lw=0.9)
            (ln,) = ax.plot([], [], color=col, lw=2.0)
            (dt,) = ax.plot(
                [], [], ls="", marker="o", ms=7.5, color=col, mec=theme.surface, mew=1.5
            )
            style_reliability_axes(ax)
            ax.set_title(titles[j], fontsize=12)
            if j == 0:
                ax.set_ylabel("accuracy in bin")
            n_bins = len(data[arm_id]["reliability"]["count"][0])
            centers = (np.arange(n_bins) + 0.5) / n_bins
            axh = hist_inset(ax, n_bins, theme)
            bc = axh.bar(centers, np.zeros(n_bins), width=0.8 / n_bins, color=col, alpha=0.6, lw=0)
            txt = ax.text(
                0.97,
                0.05,
                "",
                ha="right",
                va="bottom",
                fontsize=11,
                color=theme.ink,
                transform=ax.transAxes,
            )
            lines.append(ln), dots.append(dt), bars.append(bc), eces.append(txt)
        fig.text(
            0.085,
            0.945,
            "RLVR overshoots into overconfidence. RLCD settles on the diagonal.",
            fontsize=14,
            fontweight="bold",
            color=theme.ink,
        )
        step_txt = fig.text(0.085, 0.895, "", fontsize=10.5, color=theme.ink2)
        fig.text(
            0.085,
            0.022,
            "Exp 0: same model, same data, same optimiser; only the reward differs. "
            "Bars show how confident the model is.",
            fontsize=8,
            color=theme.muted,
        )

        def draw(frame: int) -> list[Any]:
            i = min(frame, n_frames - 1)
            for j, arm_id in enumerate(arms):
                rel = data[arm_id]["reliability"]
                conf = np.asarray(rel["conf"][i], dtype=float)
                acc = np.asarray(rel["acc"][i], dtype=float)
                count = np.asarray(rel["count"][i], dtype=float)
                keep = count >= MIN_BIN_MASS * count.sum()
                lines[j].set_data(conf[keep], acc[keep])
                dots[j].set_data(conf[keep], acc[keep])
                frac = count / count.sum()
                for rect, h in zip(bars[j], frac / 0.5, strict=True):
                    rect.set_height(h)
                ece = float(np.mean(np.asarray(data[arm_id]["curves"]["ece"])[:, i]))
                eces[j].set_text(f"ECE {ece:.3f}")
            n_seeds = len(data["rlvr"]["seeds"])
            step_txt.set_text(f"training step {steps[i]:,}   ({n_seeds} seeds pooled)")
            return []

        anim = animation.FuncAnimation(fig, draw, frames=n_frames + hold, blit=False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=100)
        draw(n_frames - 1)
        fig.savefig(path.with_suffix(".png"), dpi=200)
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------------
# concept diagram
# ---------------------------------------------------------------------------------------------


def fig_concept(theme: Theme) -> Any:
    """State + typed questions -> one forward pass -> distributions -> proper-score reward."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    blue = color("rlcd", theme)
    with plt.rc_context(rc(theme)):
        fig, ax = plt.subplots(figsize=(11.2, 4.3))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 40)
        ax.set_aspect("equal")
        ax.axis("off")

        def box(x, y, w, h, edge=None, lw=1.0, dashed=False):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle="round,pad=0,rounding_size=1.4",
                    fc=theme.surface,
                    ec=edge or theme.axis,
                    lw=lw,
                    ls=(0, (3, 2)) if dashed else "-",
                )
            )

        def arrow(p0, p1, color_=None):
            ax.add_patch(
                FancyArrowPatch(
                    p0,
                    p1,
                    arrowstyle="-|>",
                    mutation_scale=11,
                    lw=1.3,
                    color=color_ or theme.muted,
                    shrinkA=0,
                    shrinkB=0,
                )
            )

        def title(x, y, text, sub=None):
            ax.text(x, y, text, fontsize=10.5, fontweight="bold", color=theme.ink, va="top")
            if sub:
                ax.text(x, y - 3.1, sub, fontsize=8.5, color=theme.ink2, va="top")

        # 1. state + typed questions
        box(1, 11, 25, 25)
        title(2.4, 34.4, "State + typed questions")
        mono = {"family": "monospace", "fontsize": 8.2, "color": theme.ink2, "va": "top"}
        ax.text(2.4, 29.2, 'state   "order arrived damaged"', **mono)
        ax.text(2.4, 25.0, "NOUL    refund eligible?", **mono)
        ax.text(2.4, 21.6, "CHOICE  billing | support", **mono)
        ax.text(2.4, 18.2, "        | legal", **mono)
        ax.text(2.4, 14.8, "SCORE   urgency 0 .. 4", **mono)

        # 2. one forward pass
        box(31, 15, 17.5, 17)
        title(32.4, 30.4, "One forward pass", "option logits at\nthe answer slot")

        # 3. a distribution per question
        box(53, 6, 27, 30, edge=blue, lw=1.6)
        title(54.4, 34.4, "A distribution per question")
        groups = [
            ("NOUL", [0.7, 0.3]),
            ("CHOICE", [0.55, 0.3, 0.15]),
            ("SCORE", [0.05, 0.15, 0.4, 0.3, 0.1]),
        ]
        gx, width, gap = 55.4, 1.6, 0.4
        for name, probs in groups:
            for i, pr in enumerate(probs):
                ax.add_patch(
                    plt.Rectangle(
                        (gx + i * (width + gap), 11.5), width, pr * 14, fc=blue, alpha=0.85, lw=0
                    )
                )
            span = len(probs) * (width + gap) - gap
            ax.plot([gx - 0.3, gx + span + 0.3], [11.5, 11.5], color=theme.axis, lw=0.8)
            ax.text(gx + span / 2, 9.6, name, fontsize=8.2, color=theme.ink2, ha="center", va="top")
            gx += span + 2.6

        # 4. proper-score reward
        box(86, 12, 13.5, 14, edge=blue, lw=1.6)
        title(87.2, 24.4, "Proper-score", "reward S(p, y)")
        ax.text(87.2, 13.4, "log score, or −RPS", fontsize=8.2, color=theme.muted, va="bottom")

        # outcome
        box(80.5, 30, 19, 9, dashed=True)
        ax.text(82, 37.6, "outcome y", fontsize=9.5, fontweight="bold", color=theme.ink, va="top")
        ax.text(
            82,
            34.3,
            "label, verifier, teacher\nor environment",
            fontsize=8.2,
            color=theme.ink2,
            va="top",
            linespacing=1.3,
        )

        arrow((26.4, 23.5), (30.6, 23.5))
        arrow((49.0, 23.5), (52.6, 23.5))
        arrow((80.4, 19.0), (85.6, 19.0), blue)
        arrow((92.7, 29.6), (92.7, 26.4))

        # learning signal back to the model
        ax.plot([92.7, 92.7, 39.75], [11.6, 3.2, 3.2], color=blue, lw=1.3)
        arrow((39.75, 3.2), (39.75, 14.6), blue)
        ax.text(
            66,
            0.6,
            r"gradient ascent on  $\mathbb{E}[S(p,y)] - \beta\,"
            r"\mathrm{KL}(p\,\Vert\,p_{\mathrm{ref}})$",
            fontsize=9.5,
            color=theme.ink2,
            ha="center",
            va="bottom",
        )
    return fig


def make_concept_figures(fig_dir: str | Path) -> list[Path]:
    mpl.rcParams["svg.fonttype"] = "path"
    out: list[Path] = []
    for theme in THEMES:
        fig = fig_concept(theme)
        out += save_figure(fig, "concept", fig_dir, theme)
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------


def make_toy_figures(res_dir: str | Path, fig_dir: str | Path) -> list[Path]:
    """Render every Exp 0 figure (light and dark) from the result JSONs."""
    mpl.rcParams["svg.fonttype"] = "path"
    data = load_toy(res_dir)
    written: list[Path] = []
    for theme in THEMES:
        for stem, fn in (("toy_overview", fig_toy_overview), ("toy_mix", fig_toy_mix)):
            fig = fn(data, theme)
            written += save_figure(fig, stem, fig_dir, theme)
            plt.close(fig)
        written.append(hero_animation(data, theme, Path(fig_dir) / f"toy_hero{theme.suffix}.gif"))
    return written


if __name__ == "__main__":
    import sys

    mpl.use("Agg")
    print(*make_concept_figures(sys.argv[1] if len(sys.argv) > 1 else "figures"), sep="\n")
