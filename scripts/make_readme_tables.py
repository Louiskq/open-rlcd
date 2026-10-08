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
EXPA = "results/expA"
EXPA_ARMS = (
    ("a1", "A1 supervised"),
    ("a3", "A3 RLVR"),
    ("a4", "A4 RLCD-bandit"),
    ("a5", "A5 RLCD-full"),
    ("a6", "A6 mined"),
    ("a6r", "A6r random"),
)
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
    ea = load_expA()
    if ea is None:
        rows.append(f"| Exp A (9B decider, dev set) | A1–A6r | {PENDING} | {PENDING} | {PENDING} |")
    else:
        rel = f"{EXPA}/analysis.json"
        a_ = ea["arms"]
        win = bold_winners([a_[a]["dev"]["raw"]["all"]["ece"] for a, _ in EXPA_ARMS])
        for i, (arm, label) in enumerate((("a0", "A0 start model"), *EXPA_ARMS)):
            blk = a_[arm]["dev"]["raw"]["all"]
            rng_ = blk["ece"].get("n", 1) > 1
            ece = num(blk["ece"], rel, rng=rng_)
            bold = i > 0 and win[i - 1]
            rows.append(
                f"| Exp A (9B decider, dev set) | {label} | {'**' + ece + '**' if bold else ece} | "
                f"{num(blk['acc'], rel, rng=rng_)} | {num(blk['log_loss'], rel, rng=rng_)} |"
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


# ------------------------------------------------------------------------------ Exp A
def load_expA(name: str = "analysis") -> dict | None:
    path = ROOT / EXPA / f"{name}.json"
    return read_json(path) if path.exists() else None


def _p(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _diff(r: dict) -> str:
    return f"{r['observed_difference']:+.4f} ({r['ci95'][0]:+.4f} to {r['ci95'][1]:+.4f})"


def _diff_sentence(r: dict) -> str:
    return (
        f"{r['observed_difference']:+.4f}, 95% interval {r['ci95'][0]:+.4f} to {r['ci95'][1]:+.4f}"
    )


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def _expA_split_table(split: str) -> str:
    d = load_expA()
    if d is None:
        return pending("run scripts/compare_arms.py")
    rel = f"{EXPA}/analysis.json"
    arms = d["arms"]

    def cell(arm: str, row: str, metric: str, nd: int = 3) -> str:
        return num(
            arms[arm][split][row]["all"][metric],
            rel,
            nd,
            rng=arms[arm][split][row]["all"][metric].get("n", 1) > 1,
        )

    trained = [a for a, _ in EXPA_ARMS]
    ece_win = bold_winners([arms[a][split]["raw"]["all"]["ece"] for a in trained])
    ll_win = bold_winners([arms[a][split]["raw"]["all"]["log_loss"] for a in trained])
    rows = [
        "| arm | log loss | accuracy | ECE | ECE after TS | log loss after TS |",
        "|---|---|---|---|---|---|",
        f"| A0 start model, T = 1 | {cell('a0', 'raw', 'log_loss')} | {cell('a0', 'raw', 'acc')} | {cell('a0', 'raw', 'ece')} | {cell('a0', 'ts', 'ece')} | {cell('a0', 'ts', 'log_loss')} |",
        f"| A0 as served, T = 1.05 | {cell('a0', 'served', 'log_loss')} | {cell('a0', 'served', 'acc')} | {cell('a0', 'served', 'ece')} | | |",
    ]
    for i, (arm, label) in enumerate(EXPA_ARMS):
        ll, ece = cell(arm, "raw", "log_loss"), cell(arm, "raw", "ece")
        rows.append(
            f"| {label}{' (= A2 after TS)' if arm == 'a1' else ''} | {'**' + ll + '**' if ll_win[i] else ll} | "
            f"{cell(arm, 'raw', 'acc')} | {'**' + ece + '**' if ece_win[i] else ece} | {cell(arm, 'ts', 'ece')} | {cell(arm, 'ts', 'log_loss')} |"
        )
    rows.append(
        f"| A3 with \u03b2 = 0 (exploratory, 1 run) | {cell('a3beta0', 'raw', 'log_loss')} | {cell('a3beta0', 'raw', 'acc')} | "
        f"{cell('a3beta0', 'raw', 'ece')} | {cell('a3beta0', 'ts', 'ece')} | {cell('a3beta0', 'ts', 'log_loss')} |"
    )
    return "\n".join(rows) + "\n"


def gen_expA_dev() -> str:
    return _expA_split_table("dev")


def gen_expA_select() -> str:
    return _expA_split_table("select")


def gen_expA_tests() -> str:
    d = load_expA()
    if d is None:
        return pending("run scripts/compare_arms.py")
    rows = [
        "| test | comparison (row) | difference A \u2212 B (95% interval) | margin \u03b4 | Holm p | passed |",
        "|---|---|---|---|---|---|",
    ]
    for r in d["tests"]:
        delta = f"{r['delta']:.4f}" if r["delta"] else "\u2013"
        what = {"a_higher": "A higher", "a_lower": "A lower", "equiv": "equivalent"}[r["kind"]]
        rows.append(
            f"| {r['id']} | {r['a']} vs {r['b']} ({r['row']}), {what} | {_diff(r)} | {delta} | "
            f"{_p(r['p_holm'])} | {'yes' if r['passes'] else 'no'} |"
        )
    return "\n".join(rows) + "\n"


def gen_expA_verdicts() -> str:
    d = load_expA()
    if d is None:
        return pending("run scripts/compare_arms.py")
    t, h = _by_id(d["tests"]), d["hypotheses"]
    he = d["hypotheses_ensemble"]
    sel = load_expA("analysis_select_prereg_tests")
    hs = sel["hypotheses"] if sel else {}

    def mark(v: dict | None) -> str:
        return "\u2013" if v is None else ("holds" if v["holds"] else "does not hold")

    def passed(*ids: str) -> str:
        return ", ".join(f"{i} {'passed' if t[i]['passes'] else 'did not pass'}" for i in ids)

    rows = [
        "| hypothesis (pre-registered) | what the dev set shows | verdict | same test, seed-ensembled | same test, selection split (exploratory) |",
        "|---|---|---|---|---|",
        f"| **H1** A3 (RLVR) has higher ECE and log loss than A1, accuracy within \u00b1\u03b4 | ECE {_diff(t['C1-ece'])}, log loss {_diff(t['C1-log_loss'])}; {passed('C1-acc')} | **{mark(h['H1'])}** | {mark(he['H1'])} | {mark(hs.get('H1'))} |",
        f"| **H2** A5 matches A1 on every metric | {passed('C2-log_loss', 'C2-ece', 'C2-acc', 'C2-brier', 'C2-reliability', 'C2-resolution')} | **{mark(h['H2'])}** | {mark(he['H2'])} | {mark(hs.get('H2'))} |",
        f"| **H3** A4 reaches A1's calibration | ECE {_diff(t['C3-ece'])}, log loss {_diff(t['C3-log_loss'])}; {passed('C3-ece', 'C3-log_loss')} | **{mark(h['H3'])}** | {mark(he['H3'])} | {mark(hs.get('H3'))} |",
        f"| **H4** A6 is no better than A6r | ECE {_diff(t['C4-ece'])}, log loss {_diff(t['C4-log_loss'])}, accuracy {_diff(t['C4-acc'])}; none passed after Holm | **{mark(h['H4'])}** | {mark(he['H4'])} | {mark(hs.get('H4'))} |",
        f"| **H5** with TS, A1 is within noise of every RL arm on ECE | {passed('C5-a3', 'C5-a4', 'C5-a5')} (margin {t['C5-a3']['delta']:.4f}) | **{mark(h['H5'])}** | {mark(he['H5'])} | {mark(hs.get('H5'))} |",
        f"| **H5b** with TS, A3 has worse per-domain ECE and lower resolution than A1 | per-domain ECE {_diff(t['H5b-domain_ece'])}, resolution {_diff(t['H5b-resolution'])} | **{mark(h['H5b'])}** | {mark(he['H5b'])} | {mark(hs.get('H5b'))} |",
    ]
    return "\n".join(rows) + "\n"


def gen_expA_pertype() -> str:
    d = load_expA()
    if d is None:
        return pending("run scripts/compare_arms.py")
    rel = f"{EXPA}/analysis.json"
    arms = d["arms"]
    rows = [
        "| arm | NOUL log loss | NOUL ECE | CHOICE log loss | CHOICE ECE | SCORE log loss | SCORE ECE |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, label in (("a0", "A0 start model"), *EXPA_ARMS):
        cells = []
        for scope in ("noul", "choice", "score"):
            for m in ("log_loss", "ece"):
                s = arms[arm]["dev"]["raw"][scope][m]
                cells.append(num(s, rel, rng=s.get("n", 1) > 1))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def gen_expA_temps() -> str:
    d = load_expA()
    if d is None:
        return pending("run scripts/compare_arms.py")
    import numpy as np

    arms = d["arms"]
    rows = ["| arm | NOUL | CHOICE | SCORE |", "|---|---|---|---|"]
    for arm, label in (("a0", "A0 start model"), *EXPA_ARMS, ("a3beta0", "A3 with \u03b2 = 0")):
        temps = list(arms[arm]["temperatures"].values())
        cells = []
        for k in ("noul", "choice", "score"):
            v = [x[k] for x in temps]
            cells.append(
                f"{np.mean(v):.2f}" + (f" ({min(v):.2f}\u2013{max(v):.2f})" if len(v) > 1 else "")
            )
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def gen_expA_headline() -> str:
    d = load_expA()
    if d is None:
        return pending("run scripts/compare_arms.py")
    arms, t = d["arms"], _by_id(d["tests"])
    m = lambda a, s, r, k: arms[a][s][r]["all"][k]["mean"]  # noqa: E731
    return (
        f"On the independent dev set, continued training from the start model takes log loss from "
        f"{m('a0', 'dev', 'raw', 'log_loss'):.3f} to between {min(m(a, 'dev', 'raw', 'log_loss') for a, _ in EXPA_ARMS):.3f} and "
        f"{max(m(a, 'dev', 'raw', 'log_loss') for a, _ in EXPA_ARMS):.3f} and accuracy from {m('a0', 'dev', 'raw', 'acc'):.3f} to between "
        f"{min(m(a, 'dev', 'raw', 'acc') for a, _ in EXPA_ARMS):.3f} and {max(m(a, 'dev', 'raw', 'acc') for a, _ in EXPA_ARMS):.3f}, "
        f"whatever the arm. RLVR (A3) ends with an ECE of {m('a3', 'dev', 'raw', 'ece'):.3f} against {m('a1', 'dev', 'raw', 'ece'):.3f} for the supervised control "
        f"(difference {_diff_sentence(t['C1-ece'])}) at the same accuracy, and without a worse log loss (difference {_diff_sentence(t['C1-log_loss'])}). "
        f"After one temperature per answer type the ECEs are {m('a3', 'dev', 'ts', 'ece'):.3f} (A3) and {m('a1', 'dev', 'ts', 'ece'):.3f} (A1).\n"
    )


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
    "expA_headline": gen_expA_headline,
    "expA_verdicts": gen_expA_verdicts,
    "expA_dev": gen_expA_dev,
    "expA_select": gen_expA_select,
    "expA_tests": gen_expA_tests,
    "expA_pertype": gen_expA_pertype,
    "expA_temps": gen_expA_temps,
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
