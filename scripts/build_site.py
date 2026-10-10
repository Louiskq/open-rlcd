"""Build the static landing page in ``site/`` (deployable as-is to Vercel or GitHub Pages).

    python scripts/make_site_data.py     # once: the toy sweep behind the interactive demo (about 30 s)
    python scripts/build_site.py         # site/index.html, site/assets/, site/explorer.html, site/vercel.json
    python scripts/build_site.py --check # exit 1 if site/ is out of date with results/ (CI)

Every number on the page is read from ``results/**/*.json`` (the same files as the README tables); the toy demo's
data is embedded in the page so it works without a server. Nothing here needs a login: deploy with ``npx vercel``
from ``site/``.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_readme_tables as T  # noqa: E402

REPO = "https://github.com/Louiskq/open-rlcd"
TEMPLATE = ROOT / "scripts" / "site_index.template.html"
ASSETS = ("concept", "expA_overview", "expB_beliefs")
VERCEL = {
    "cleanUrls": True,
    "trailingSlash": False,
    "headers": [
        {
            "source": "/assets/(.*)",
            "headers": [{"key": "Cache-Control", "value": "public, max-age=86400"}],
        },
        {"source": "/(.*)", "headers": [{"key": "X-Content-Type-Options", "value": "nosniff"}]},
    ],
}


def f3(x: float) -> str:
    return f"{x:.3f}"


def values() -> dict[str, str]:
    """Every number on the page, read from the results files."""
    toy, ea, eb = T.load_toy(), T.load_expA(), T.load_expB()
    if toy is None or ea is None or eb is None:
        raise SystemExit(
            "run the experiments first: results/toy, results/expA and results/expB are needed"
        )
    s, arms, h7 = toy["summary"], ea["arms"], eb["h7"]
    cm, ps = h7["common_metrics"], h7["per_seed_log_loss"]
    b4_better = sum(1 for a, b in zip(ps["b4_ts"], ps["b3_raw"], strict=True) if a < b)
    return {
        "toy_rlvr_ece": f3(s["rlvr"]["raw"]["ece"]["mean"]),
        "toy_rlcd_ece": f3(s["rlcd_bandit"]["raw"]["ece"]["mean"]),
        "toy_sup_ece": f3(s["supervised"]["raw"]["ece"]["mean"]),
        "toy_rlvr_acc": f3(s["rlvr"]["raw"]["acc"]["mean"]),
        "toy_rlcd_acc": f3(s["rlcd_bandit"]["raw"]["acc"]["mean"]),
        "a3_dev_ece": f3(arms["a3"]["dev"]["raw"]["all"]["ece"]["mean"]),
        "a4_dev_ece": f3(arms["a4"]["dev"]["raw"]["all"]["ece"]["mean"]),
        "a1_dev_ece": f3(arms["a1"]["dev"]["raw"]["all"]["ece"]["mean"]),
        "a3_ts_ece": f3(arms["a3"]["dev"]["ts"]["all"]["ece"]["mean"]),
        "a4_ts_ece": f3(arms["a4"]["dev"]["ts"]["all"]["ece"]["mean"]),
        "a1_ts_ece": f3(arms["a1"]["dev"]["ts"]["all"]["ece"]["mean"]),
        "b1_ll": f3(cm["b1_raw"]["log_loss"]["mean"]),
        "b3_ll": f3(cm["b3_raw"]["log_loss"]["mean"]),
        "b4_ll": f3(cm["b4_ts"]["log_loss"]["mean"]),
        "b4_better_seeds": str(b4_better),
    }


def headline_rows() -> str:
    toy, ea, eb = T.load_toy(), T.load_expA(), T.load_expB()
    s, arms, cm = toy["summary"], ea["arms"], eb["h7"]["common_metrics"]
    rows = [
        (
            "Toy (truth known), ECE",
            [s[a]["raw"]["ece"] for a in ("rlvr", "rlcd_bandit", "supervised")],
        ),
        (
            "9B model, dev set, ECE",
            [arms[a]["dev"]["raw"]["all"]["ece"] for a in ("a3", "a4", "a1")],
        ),
        (
            "9B model, dev set, after one fitted temperature, ECE",
            [arms[a]["dev"]["ts"]["all"]["ece"] for a in ("a3", "a4", "a1")],
        ),
        (
            "9B model, harder split, ECE",
            [arms[a]["select"]["raw"]["all"]["ece"] for a in ("a3", "a4", "a1")],
        ),
        ("Breakout beliefs, log loss", [cm[k]["log_loss"] for k in ("b1_raw", "b3_raw", "b4_ts")]),
    ]
    out = []
    for label, stats in rows:
        win = T.bold_winners(stats)
        cells = "".join(
            f'<td class="{"best" if w else ""}">{f3(st["mean"])}</td>'
            for st, w in zip(stats, win, strict=True)
        )
        out.append(f"<tr><td>{html.escape(label)}</td>{cells}</tr>")
    return "\n".join(out)


def render() -> dict[str, str | bytes]:
    """The files of the site as {relative path: content}."""
    data_path = ROOT / "site" / "data" / "toy_mix.json"
    if not data_path.exists():
        raise SystemExit("run scripts/make_site_data.py first")
    page = TEMPLATE.read_text()
    subs = {
        **values(),
        "HEADLINE_ROWS": headline_rows(),
        "REPO": REPO,
        "TOY_JSON": json.dumps(json.loads(data_path.read_text()), separators=(",", ":")),
    }
    for k, v in subs.items():
        page = page.replace("{{" + k + "}}", v)
    if "{{" in page:
        left = page[page.index("{{") : page.index("{{") + 40]
        raise SystemExit(f"unfilled placeholder in the template: {left!r}")
    files: dict[str, str | bytes] = {
        "index.html": page,
        "vercel.json": json.dumps(VERCEL, indent=2) + "\n",
    }
    files["explorer.html"] = (ROOT / "docs" / "explorer.html").read_text()
    for stem in ASSETS:
        for suffix in ("", "-dark"):
            name = f"{stem}{suffix}.png"
            files[f"assets/{name}"] = (ROOT / "figures" / name).read_bytes()
    return files


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "site"))
    a = ap.parse_args()
    out = Path(a.out)
    files = render()
    if a.check:
        stale = []
        for rel, content in files.items():
            p = out / rel
            cur = p.read_bytes() if p.exists() else None
            new = content if isinstance(content, bytes) else content.encode()
            if cur != new:
                stale.append(rel)
        if stale:
            print(
                f"site/ is out of date with results/: {', '.join(stale)}. Run: make site",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print("site/ is up to date")
        return
    for rel, content in files.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {len(files)} files to {out} ({total / 1024:.0f} KB in all)")


if __name__ == "__main__":
    main()
