"""Exp A figures, drawn from ``results/expA/analysis*.json`` (``scripts/compare_arms.py``). Same palette per arm as
every other figure (``rlcd.plots``); filled markers are raw (T = 1), hollow markers are after temperature scaling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from rlcd.plots import ARM_STYLES, THEMES, ArmStyle, Theme, color, rc, save_figure
from rlcd.results import read_json

ARM_STYLES.setdefault("A3b0", ArmStyle("A3 β = 0 (exploratory)", "ablation", ":", "^"))
ARMS = ["A0", "A1", "A3", "A4", "A5", "A6", "A6r"]
KEY = {
    "A0": "a0",
    "A1": "a1",
    "A3": "a3",
    "A4": "a4",
    "A5": "a5",
    "A6": "a6",
    "A6r": "a6r",
    "A3b0": "a3beta0",
}
SHORT = {
    "A0": "A0\nstart",
    "A1": "A1\nsupervised",
    "A3": "A3\nRLVR",
    "A4": "A4\nRLCD-bandit",
    "A5": "A5\nRLCD-full",
    "A6": "A6\nmined",
    "A6r": "A6r\nrandom",
    "A3b0": "A3 β=0\n(exploratory)",
}


def _stat(
    arms: dict[str, Any], arm: str, split: str, row: str, scope: str, metric: str
) -> dict[str, float]:
    s = arms[KEY[arm]][split][row][scope][metric]
    if "mean" not in s:  # a single run
        return {"mean": s, "min": s, "max": s}
    return s


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
        mew=1.6 if hollow else 1.4,
        zorder=3,
    )


def _legend_rows(fig: Any, theme: Theme, arms: list[str], y: float) -> None:
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
    handles += [
        Line2D([], [], color=theme.ink2, marker="o", ls="", ms=7),
        Line2D([], [], color=theme.ink2, marker="o", ls="", ms=7, mfc=theme.surface, mew=1.5),
    ]
    labels = [ARM_STYLES[a].label for a in arms] + ["raw (T = 1)", "after temperature scaling"]
    fig.legend(
        handles,
        labels,
        ncol=5,
        loc="upper left",
        bbox_to_anchor=(0.06, y),
        columnspacing=1.6,
        handletextpad=0.4,
    )


def fig_overview(data: dict[str, Any], theme: Theme) -> Any:
    """ECE, log loss and accuracy of every arm: dev (the pre-registered test set) above, selection below."""
    arms = data["arms"]
    metrics = [
        ("ece", "ECE (15 bins), lower is better"),
        ("log_loss", "log loss, lower is better"),
        ("acc", "accuracy"),
    ]
    with plt.rc_context(rc(theme)):
        fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.6), sharex=True)
        fig.subplots_adjust(
            left=0.065, right=0.985, top=0.71, bottom=0.095, wspace=0.27, hspace=0.30
        )
        for r, (split, title) in enumerate(
            (
                ("dev", "Dev set (pre-registered test set)"),
                ("select", "Selection split (exploratory)"),
            )
        ):
            for c, (metric, ylabel) in enumerate(metrics):
                ax = axes[r, c]
                for i, a in enumerate(ARMS):
                    for row, dx, hollow in (("raw", -0.12, False), ("ts", 0.12, True)):
                        if row not in arms[KEY[a]][split]:
                            continue
                        _dot(
                            ax, i + dx, _stat(arms, a, split, row, "all", metric), a, theme, hollow
                        )
                ax.set_xticks(range(len(ARMS)), ARMS, fontsize=9)
                ax.set_xlim(-0.55, len(ARMS) - 0.45)
                ax.set_ylabel(ylabel)
                ax.grid(axis="x", visible=False)
                if c == 0:
                    ax.set_title(title, loc="left", fontsize=10)
                    ax.title.set_position((0, 1.0))
        fig.text(
            0.065,
            0.955,
            "Exp A: continued training from the start model, by arm",
            fontsize=13.5,
            fontweight="bold",
            color=theme.ink,
        )
        fig.text(
            0.065,
            0.918,
            "Mean over 3 seeds (A5: 1; A0 is the start model); bars are the seed range. A1, A5, A6 and A6r train SCORE with the "
            "ranked\nprobability score (D3); A3 and A4 sample answers, which changes the pooled log loss through SCORE (next figure).",
            fontsize=9.5,
            color=theme.ink2,
            va="top",
            linespacing=1.5,
        )
        _legend_rows(fig, theme, ARMS, 0.85)
        fig.text(
            0.065,
            0.02,
            f"{data['bootstrap']['B']}-replicate bootstrap tests are in the table, not here. Source: results/expA/analysis.json.",
            fontsize=8,
            color=theme.muted,
        )
    return fig


def fig_pertype(data: dict[str, Any], theme: Theme, split: str = "dev") -> Any:
    """Log loss and ECE per answer type, raw and after temperature scaling."""
    arms = data["arms"]
    types = [("noul", "NOUL"), ("choice", "CHOICE"), ("score", "SCORE")]
    with plt.rc_context(rc(theme)):
        fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.6), sharex=True)
        fig.subplots_adjust(
            left=0.065, right=0.985, top=0.71, bottom=0.095, wspace=0.27, hspace=0.30
        )
        for r, (metric, ylabel) in enumerate((("log_loss", "log loss"), ("ece", "ECE"))):
            for c, (scope, name) in enumerate(types):
                ax = axes[r, c]
                for i, a in enumerate(ARMS):
                    for row, dx, hollow in (("raw", -0.12, False), ("ts", 0.12, True)):
                        if row in arms[KEY[a]][split]:
                            _dot(
                                ax,
                                i + dx,
                                _stat(arms, a, split, row, scope, metric),
                                a,
                                theme,
                                hollow,
                            )
                ax.set_xticks(range(len(ARMS)), ARMS, fontsize=9)
                ax.set_xlim(-0.55, len(ARMS) - 0.45)
                ax.set_ylabel(f"{ylabel}, lower is better")
                ax.grid(axis="x", visible=False)
                if r == 0:
                    ax.set_title(name, loc="left")
        which = "Dev set" if split == "dev" else "Selection split"
        fig.text(
            0.065,
            0.955,
            f"Exp A by answer type ({which.lower()})",
            fontsize=13.5,
            fontweight="bold",
            color=theme.ink,
        )
        fig.text(
            0.065,
            0.918,
            "A1, A5, A6 and A6r train SCORE with the ranked probability score (D3); A3 and A4 sample answers. The SCORE column\n"
            "therefore differs by design; NOUL and CHOICE are the like-for-like comparison.",
            fontsize=9.5,
            color=theme.ink2,
            va="top",
            linespacing=1.5,
        )
        _legend_rows(fig, theme, ARMS, 0.85)
    return fig


def fig_tests(data: dict[str, Any], theme: Theme) -> Any:
    """The 19 pre-registered tests: equivalence tests in units of their margin delta, difference tests in raw units."""
    rows = data["tests"]
    eq = [r for r in rows if r["kind"] == "equiv"]
    di = [r for r in rows if r["kind"] != "equiv"]
    with plt.rc_context(rc(theme)):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12.2, 8.2), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.5}
        )
        fig.subplots_adjust(left=0.185, right=0.975, top=0.84, bottom=0.085)

        def label(r: dict[str, Any]) -> str:
            return f"{r['id']}  {r['a']} vs {r['b']} ({r['row']})".replace("seed 0", "s0")

        for ax, group, equiv in ((ax1, eq, True), (ax2, di, False)):
            ys = list(range(len(group)))[::-1]
            for y, r in zip(ys, group, strict=True):
                scale = r["delta"] if equiv else 1.0
                lo, hi, d = (
                    r["ci95"][0] / scale,
                    r["ci95"][1] / scale,
                    r["observed_difference"] / scale,
                )
                col = color(
                    "A3"
                    if r["hypothesis"] in ("H1", "H5b")
                    else "A4"
                    if r["hypothesis"] in ("H3", "H5")
                    else "A5"
                    if r["hypothesis"] == "H2"
                    else "A6",
                    theme,
                )
                ax.plot([lo, hi], [y, y], color=col, lw=1.8, solid_capstyle="round", zorder=2)
                ax.plot(
                    [d],
                    [y],
                    marker="o",
                    ms=7.5,
                    mfc=col if r["passes"] else theme.surface,
                    mec=col,
                    mew=1.6,
                    ls="",
                    zorder=3,
                )
            ax.set_yticks(ys, [label(r) for r in group], fontsize=7.8)
            ax.axvline(0, color=theme.muted, lw=0.9)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", visible=True)
            if equiv:
                ax.axvspan(-1, 1, color=theme.grid, alpha=0.6, zorder=0)
                ax.set_xlabel(
                    "difference in units of the equivalence margin δ (shaded band: equivalent)"
                )
                ax.set_xlim(-2.6, 2.6)
                ax.set_title("Equivalence tests (H1 accuracy, H2, H3, H5)")
            else:
                ax.set_xlabel("difference, A − B (raw units of the metric)")
                ax.set_title("Difference tests (H1, H4, H5b)")
        fig.text(
            0.02,
            0.965,
            "The 19 pre-registered tests, dev set",
            fontsize=13.5,
            fontweight="bold",
            color=theme.ink,
        )
        fig.text(
            0.02,
            0.93,
            "Dot = observed mean difference, line = 95% bootstrap interval. Filled dot: the test passed after Holm correction "
            "(equivalence: interval inside the band;\ndifference: interval excludes 0 in the claimed direction). Hollow: not shown.",
            fontsize=9.5,
            color=theme.ink2,
            va="top",
            linespacing=1.5,
        )
    return fig


def fig_beta0(data: dict[str, Any], theme: Theme) -> Any:
    """Exploratory E1: RLVR without the KL anchor against with it, on the dev set."""
    arms = data["arms"]
    show = ["A1", "A3", "A3b0"]
    with plt.rc_context(rc(theme)):
        fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.7))
        fig.subplots_adjust(left=0.065, right=0.985, top=0.7, bottom=0.2, wspace=0.3)
        for ax, (metric, ylabel, title) in zip(
            axes,
            (
                ("ece", "ECE, lower is better", "calibration error"),
                ("log_loss", "log loss, lower is better", "log loss"),
                ("acc", "accuracy", "accuracy"),
            ),
            strict=True,
        ):
            for i, a in enumerate(show):
                for row, dx, hollow in (("raw", -0.1, False), ("ts", 0.1, True)):
                    _dot(ax, i + dx, _stat(arms, a, "dev", row, "all", metric), a, theme, hollow)
            ax.set_xticks(range(len(show)), [SHORT[a] for a in show], fontsize=8.5)
            ax.set_xlim(-0.5, len(show) - 0.5)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(axis="x", visible=False)
        fig.text(
            0.065,
            0.955,
            "Exploratory (E1): without the KL anchor, RLVR becomes overconfident beyond repair",
            fontsize=13,
            fontweight="bold",
            color=theme.ink,
        )
        fig.text(
            0.065,
            0.9,
            "One run without the penalty (β = 0). The fitted temperature hits the search bound (20) for NOUL, so the hollow marker\n"
            "understates the damage: temperature scaling cannot repair it.",
            fontsize=9.5,
            color=theme.ink2,
            va="top",
            linespacing=1.5,
        )
    return fig


def make_expA_figures(analysis: str | Path, fig_dir: str | Path) -> list[Path]:
    mpl.rcParams["svg.fonttype"] = "path"
    data = read_json(analysis)
    out: list[Path] = []
    for theme in THEMES:
        for stem, fn in (
            ("expA_overview", fig_overview),
            ("expA_pertype_dev", lambda d, t: fig_pertype(d, t, "dev")),
            ("expA_pertype_select", lambda d, t: fig_pertype(d, t, "select")),
            ("expA_tests", fig_tests),
            ("expA_beta0", fig_beta0),
        ):
            fig = fn(data, theme)
            out += save_figure(fig, stem, fig_dir, theme)
            plt.close(fig)
    return out


if __name__ == "__main__":
    import sys

    make_expA_figures(
        sys.argv[1] if len(sys.argv) > 1 else "results/expA/analysis.json",
        sys.argv[2] if len(sys.argv) > 2 else "figures",
    )
