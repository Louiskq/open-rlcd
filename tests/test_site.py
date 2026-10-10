"""The landing page: built from the results files, no unfilled placeholders, numbers equal to the JSON, assets exist."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def build():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def files(build):
    return build.render()


def test_no_placeholder_is_left_and_the_page_is_self_contained(files):
    page = files["index.html"]
    assert not re.search(r"\{\{[A-Za-z_0-9]+\}\}", page)
    assert '<script type="application/json" id="toy-data">' in page
    assert "https://github.com/Louiskq/open-rlcd" in page


def test_numbers_on_the_page_are_the_ones_in_the_results_files(files, build):
    page = files["index.html"]
    toy = json.loads((ROOT / "results/toy/meta.json").read_text())["summary"]
    ea = json.loads((ROOT / "results/expA/analysis.json").read_text())["arms"]
    eb = json.loads((ROOT / "results/expB/analysis.json").read_text())["h7"]
    assert f"{toy['rlvr']['raw']['ece']['mean']:.3f}" in page
    assert f"{ea['a3']['dev']['raw']['all']['ece']['mean']:.3f}" in page
    assert f"{eb['common_metrics']['b3_raw']['log_loss']['mean']:.3f}" in page
    vals = build.values()
    assert vals["toy_rlvr_ece"] == f"{toy['rlvr']['raw']['ece']['mean']:.3f}"
    ps = eb["per_seed_log_loss"]
    assert vals["b4_better_seeds"] == str(
        sum(a < b for a, b in zip(ps["b4_ts"], ps["b3_raw"], strict=True))
    )


def test_the_embedded_toy_data_matches_the_repository_toy_results(files):
    page = files["index.html"]
    raw = re.search(r'id="toy-data">(.*?)</script>', page, re.S).group(1)
    data = json.loads(raw)
    assert len(data["mixes"]) == 9 and data["mixes"][0] == 0.0 and data["mixes"][-1] == 1.0
    assert set(data["arms"]) == {"0", "0.02", "0.05", "0.1", "0.2", "0.35", "0.5", "0.75", "1"}
    # pure RLVR reproduces the RLVR arm of Exp 0 (same code, 5 of the 10 seeds)
    full = json.loads((ROOT / "results/toy/meta.json").read_text())["summary"]["rlvr"]["raw"][
        "ece"
    ]["mean"]
    assert abs(data["arms"]["1"]["final_raw"]["ece"] - full) < 0.02
    # pure RLCD is calibrated, pure RLVR is not, at the same accuracy
    assert (
        data["arms"]["0"]["final_raw"]["ece"] < 0.02 < 0.3 < data["arms"]["1"]["final_raw"]["ece"]
    )
    assert abs(data["arms"]["0"]["final_raw"]["acc"] - data["arms"]["1"]["final_raw"]["acc"]) < 0.01
    n = len(data["steps"])
    assert all(
        len(a["ece"]) == n and len(a["reliability"]["count"]) == n for a in data["arms"].values()
    )


def test_every_asset_the_page_links_to_is_built(files):
    page = files["index.html"]
    refs = set(re.findall(r'(?:src|srcset|href)="((?:assets/)[^"]+|explorer\.html)"', page))
    assert refs and all(r in files for r in refs), [r for r in refs if r not in files]
    assert "vercel.json" in files and json.loads(files["vercel.json"])["cleanUrls"] is True


def test_the_page_states_the_not_typesafe_disclaimer_and_the_mixed_result(files):
    page = files["index.html"]
    assert "not TypeSafe's method" in page
    assert "no clear advantage" in page.replace("\n", " ")
