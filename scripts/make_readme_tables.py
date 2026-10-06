"""Fill the generated blocks of README.md from results/**/*.json.

    python scripts/make_readme_tables.py           # rewrite README.md in place
    python scripts/make_readme_tables.py --check   # exit 1 if README.md is out of date (CI)

A block looks like ``<!-- BEGIN gen:NAME --> ... <!-- END gen:NAME -->``. Everything between the
markers is replaced; nothing outside them is touched. Every number links to the JSON it came from.
Bold marks the best arm *and* any arm whose mean lies within the best arm's seed range, so a lead
that is inside the noise is not presented as a win.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlcd.plots import ARM_STYLES  # noqa: E402
from rlcd.results import read_json  # noqa: E402

TOY = "results/toy"
BLOCK = re.compile(r"(<!-- BEGIN gen:(\w+) -->\n)(.*?)(<!-- END gen:\2 -->)", re.S)
PENDING = "_pending_"
TOY_ROWS = ("rlvr", "rlcd_bandit", "rlcd_bandit_clip10", "supervised", "rlcd_full")


def link(text: str, rel: str) -> str:
    return f"[{text}]({rel})"


def num(s: dict[str, float], rel: str, nd: int = 3, rng: bool = True) -> str:
    """``mean (min–max)`` over seeds, as a link to the JSON it came from."""
    txt = f"{s['mean']:.{nd}f}"
    if rng and s.get("n", 1) > 1:
        txt += f" ({s['min']:.{nd}f}–{s['max']:.{nd}f})"
    return link(txt, rel)


def bold_winners(stats: list[dict[str, float]], lower_is_better: bool = True) -> list[bool]:
    """True for the best arm and every arm whose mean is within the best arm's seed range."""
    key = (lambda s: s["mean"]) if lower_is_better else (lambda s: -s["mean"])
    best = min(stats, key=key)
    slack = best["max"] - best["min"]
    return [key(s) <= key(best) + slack + 1e-12 for s in stats]


def pending(reason: str) -> str:
    return f"{PENDING}: {reason}"


def load_toy() -> dict | None:
    path = ROOT / TOY / "meta.json"
    return read_json(path) if path.exists() else None


# ------------------------------------------------------------------------------ generators


def gen_toy_setup() -> str:
    meta = load_toy()
    if meta is None:
        return pending("run `make toy`") + "\n"
    c, o, seeds = meta["config"], meta["oracle"], meta["seed_sets"]
    rel = TOY + "/meta.json"
    ece = link(f"{o['ece']:.3f}", rel)
    bayes = link(f"{o['bayes_acc']:.3f}", rel)
    return (
        f"{len(seeds['train'])} seeds × {c['steps']:,} Adam steps per arm, evaluated on "
        f"{c['n_eval']:,} held-out contexts. The true probabilities themselves score ECE "
        f"{ece} on this sample (the noise floor) and the Bayes-optimal accuracy is {bayes}.\n"
    )


def gen_toy_table() -> str:
    meta = load_toy()
    if meta is None:
        return pending("run `make toy`") + "\n"
    summ = meta["summary"]
    cols = [("ece", True), ("acc", None), ("log_loss", True), ("kl_to_truth", True)]
    win: dict[str, list[bool]] = {}
    for metric, lower in cols:
        if lower is not None:
            win[metric] = bold_winners([summ[a]["raw"][metric] for a in TOY_ROWS])
    ts_win = bold_winners([summ[a]["ts"]["ece"] for a in TOY_ROWS])
    head = (
        "| arm | ECE | accuracy | log loss | KL(truth ‖ model) | ECE after temperature scaling |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for i, arm in enumerate(TOY_ROWS):
        rel = f"{TOY}/{arm}.json"
        cells = []
        for metric, lower in cols:
            txt = num(summ[arm]["raw"][metric], rel)
            cells.append(f"**{txt}**" if lower is not None and win[metric][i] else txt)
        ts = num(summ[arm]["ts"]["ece"], rel) + f" (T = {summ[arm]['temperature']['mean']:.1f})"
        cells.append(f"**{ts}**" if ts_win[i] else ts)
        rows.append(f"| {ARM_STYLES[arm].label} | " + " | ".join(cells) + " |")
    o = meta["oracle"]
    m = TOY + "/meta.json"
    o_cells = [link(f"{o[k]:.3f}", m) for k in ("ece", "acc", "log_loss")]
    rows.append(f"| _the true probabilities_ | {' | '.join(o_cells)} | 0 | |")
    gap = link(f"{meta['supervised_vs_rlcd_full_max_abs_diff']:.1e}", m)
    note = (
        "\nSeed mean (min–max). Bold: best, or within the best arm's seed range. Supervised log "
        "loss and RLCD-full are the same objective through two code paths; their curves differ by "
        f"at most {gap} over every metric, seed and step.\n"
    )
    return head + "\n".join(rows) + "\n" + note


def gen_toy_noise() -> str:
    meta = load_toy()
    if meta is None:
        return pending("run `make toy`") + "\n"
    g = meta["gradient_noise_at_truth"]
    rel = TOY + "/meta.json"
    head = (
        "| arm | bias ‖E[g]‖ | Monte Carlo floor | bias / floor | noise (trace of covariance) |\n"
        "|---|---|---|---|---|\n"
    )
    rows = []
    for arm in TOY_ROWS:
        d = g[arm]
        ratio = d["bias_norm"] / d["bias_se_norm"]
        bias = link(f"{d['bias_norm']:.4f}", rel)
        noise = link(f"{d['noise_trace']:.3f}", rel)
        floor = f"{d['bias_se_norm']:.4f}"
        rows.append(f"| {ARM_STYLES[arm].label} | {bias} | {floor} | {ratio:.1f}× | {noise} |")
    return head + "\n".join(rows) + "\n"


def gen_toy_mix() -> str:
    path = ROOT / TOY / "mix_sweep.json"
    if not path.exists():
        return pending("run `make toy`") + "\n"
    sweep = read_json(path)
    rel = f"{TOY}/mix_sweep.json"
    head = (
        "| share of RLVR in the reward | ECE | ECE after temperature scaling | log loss |\n"
        "|---|---|---|---|\n"
    )
    rows = []
    for m in sweep["mixes"]:
        s = sweep["summary"][f"{m:g}"]
        rows.append(
            f"| {m:.0%} | {num(s['raw']['ece'], rel)} | {num(s['ts']['ece'], rel)} | "
            f"{num(s['raw']['log_loss'], rel)} |"
        )
    return head + "\n".join(rows) + "\n"


def gen_toy_lr() -> str:
    path = ROOT / TOY / "lr_sensitivity.json"
    if not path.exists():
        return pending("run `make toy`") + "\n"
    d = read_json(path)
    rel = f"{TOY}/lr_sensitivity.json"
    lrs = list(d["lr"])
    intro = f"{len(d['seeds'])} seeds per cell. Seed mean (min–max).\n\n"
    head = intro + "| arm (final ECE) | " + " | ".join(f"lr {x}" for x in lrs) + " |\n"
    head += "|---|" + "---|" * len(lrs) + "\n"
    rows = [
        f"| {ARM_STYLES[arm].label} | "
        + " | ".join(num(d["lr"][x][arm]["raw"]["ece"], rel) for x in lrs)
        + " |"
        for arm in TOY_ROWS
    ]
    return head + "\n".join(rows) + "\n"


def gen_tldr() -> str:
    meta = load_toy()
    head = (
        "| setting | arm | ECE (lower is better) | accuracy | log loss |\n|---|---|---|---|---|\n"
    )
    rows = []
    if meta is None:
        rows.append(f"| Exp 0 (toy) | | {PENDING} | | |")
    else:
        summ = meta["summary"]
        show = ("rlvr", "rlcd_bandit", "rlcd_bandit_clip10", "supervised")
        win = bold_winners([summ[a]["raw"]["ece"] for a in show])
        for i, arm in enumerate(show):
            rel = f"{TOY}/{arm}.json"
            ece = num(summ[arm]["raw"]["ece"], rel)
            label = ARM_STYLES[arm].label + (" (= RLCD-full)" if arm == "supervised" else "")
            rows.append(
                f"| Exp 0 (toy) | {label} | {'**' + ece + '**' if win[i] else ece} | "
                f"{num(summ[arm]['raw']['acc'], rel)} | {num(summ[arm]['raw']['log_loss'], rel)} |"
            )
    rows.append(
        f"| Exp A (9B decider, teacher labels) | A1–A6r | {PENDING} | {PENDING} | {PENDING} |"
    )
    rows.append(f"| Exp B (Breakout, on-policy) | B1–B5 | {PENDING} | {PENDING} | {PENDING} |")
    return head + "\n".join(rows) + "\n"


def load_toy0b() -> dict | None:
    path = ROOT / "results/toy0b/meta.json"
    return read_json(path) if path.exists() else None


def signed(s: dict[str, float], rel: str) -> str:
    return link(f"{s['mean']:+.3f}".replace("-", "\u2212"), rel)


def gen_toy0b_setup() -> str:
    meta = load_toy0b()
    if meta is None:
        return pending("run `python -m toy.exp0b`") + "\n"
    c, lo_hi = meta["config"], meta["scale_range"]
    return (
        f"{len(meta['seed_sets']['train'])} seeds × {c['steps']:,} Adam steps per arm, "
        f"{c['n_eval']:,} held-out contexts, true-logit scale between {lo_hi[0]} (hard) and "
        f"{lo_hi[1]} (easy).\n"
    )


def gen_toy0b_table() -> str:
    meta = load_toy0b()
    if meta is None:
        return pending("run `python -m toy.exp0b`") + "\n"
    rel0 = "results/toy0b/meta.json"
    summ, grp, o = meta["summary"], meta["by_difficulty"], meta["oracle"]
    head = (
        "| arm | ECE | ECE after one temperature | resolution after | KL(truth ‖ model) after | "
        "confidence − accuracy after, hard / medium / easy |\n|---|---|---|---|---|---|\n"
    )
    rows = []
    for arm in TOY_ROWS:
        rel = f"results/toy0b/{arm}.json"
        t = summ[arm]["ts"]
        gaps = " / ".join(signed(g, rel) for g in grp[arm]["ts"]["gap"])
        rows.append(
            f"| {ARM_STYLES[arm].label} | {num(summ[arm]['raw']['ece'], rel)} | "
            f"{num(t['ece'], rel)} (T = {summ[arm]['temperature']['mean']:.1f}) | "
            f"{num(t['resolution'], rel)} | {num(t['kl_to_truth'], rel)} | {gaps} |"
        )
    ece = link(f"{o['ece']:.3f}", rel0)
    res = link(f"{o['resolution']:.3f}", rel0)
    rows.append(f"| _the true probabilities_ | {ece} | | {res} | 0 | |")
    return head + "\n".join(rows) + "\n"


def gen_toy0b_group() -> str:
    meta = load_toy0b()
    if meta is None:
        return pending("run `python -m toy.exp0b`") + "\n"
    grp = meta["by_difficulty"]
    head = (
        "| arm | temperature per tercile (hard / medium / easy) | ECE after | resolution after | "
        "KL(truth ‖ model) after |\n|---|---|---|---|---|\n"
    )
    rows = []
    for arm in TOY_ROWS:
        rel = f"results/toy0b/{arm}.json"
        g = grp[arm]
        temps = " / ".join(f"{t['mean']:.1f}" for t in g["group_temperature"])
        t = g["ts_group_overall"]
        rows.append(
            f"| {ARM_STYLES[arm].label} | {temps} | {num(t['ece'], rel)} | "
            f"{num(t['resolution'], rel)} | {num(t['kl_to_truth'], rel)} |"
        )
    return head + "\n".join(rows) + "\n"


GENERATORS = {
    "tldr": gen_tldr,
    "toy_setup": gen_toy_setup,
    "toy_table": gen_toy_table,
    "toy_noise": gen_toy_noise,
    "toy_mix": gen_toy_mix,
    "toy_lr": gen_toy_lr,
    "toy0b_setup": gen_toy0b_setup,
    "toy0b_table": gen_toy0b_table,
    "toy0b_group": gen_toy0b_group,
}


def render(text: str) -> str:
    def fill(m: re.Match[str]) -> str:
        name = m.group(2)
        if name not in GENERATORS:
            raise SystemExit(f"README.md has an unknown generated block: gen:{name}")
        return m.group(1) + GENERATORS[name]() + m.group(4)

    return BLOCK.sub(fill, text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if README.md is out of date")
    args = ap.parse_args()
    path = ROOT / "README.md"
    old = path.read_text()
    new = render(old)
    if args.check:
        if new != old:
            print("README.md is out of date with results/. Run: make readme", file=sys.stderr)
            raise SystemExit(1)
        print("README.md is up to date")
        return
    path.write_text(new)
    print(f"updated {sum(1 for _ in BLOCK.finditer(new))} generated blocks in README.md")


if __name__ == "__main__":
    main()
