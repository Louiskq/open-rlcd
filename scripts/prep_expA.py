"""Build the Exp A question units (tokenised rows) for every split. Run on the cluster, in decider's venv:

    PYTHONPATH=$REPO $DECIDER/.venv/bin/python scripts/prep_expA.py \
        --model $PROJ/models/decider9b-sft --data_dir $PROJ/hardgen/data --out $PROJ/rlcd_expA/data

Writes ``units_<split>.pkl`` for train_A, train_B, cal, select and dev (lists of ``rlcd.units.Unit``, labels
included: the A6 runner hides half B's labels itself), ``splits.json`` (counts, type mix, domains, an id digest
per split) and ``dev_domains.json`` (the domain of every dev question, for H5b). Splits are by record id
(``rlcd.pool``); the dev set must share no record id and no state text with the pool.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np

from rlcd import pool as P
from rlcd import units as U

POOL_FILES = [
    "text_seeded_train.jsonl",
    "text_authored_train_a.jsonl",
    "text_authored_train_b.jsonl",
    "text_r2_seeded_a.jsonl",
    "text_r2_seeded_b.jsonl",
    "text_r2_authored_a.jsonl",
    "text_r2_authored_b.jsonl",
    "text_r2_authored_c.jsonl",
]
DEV_FILES = ["text_dev_authored_a.jsonl", "text_dev_authored_b.jsonl"]


def norm_state(state) -> str:
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True)
    return re.sub(r"\s+", " ", text.lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint dir (tokenizer is read from it)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="seeds the per-question option shuffles")
    ap.add_argument("--max_ctx", type=int, default=16384)
    ap.add_argument(
        "--limit", type=int, default=0, help="debug: keep only this many records per split"
    )
    a = ap.parse_args()

    from decider import data as D
    from decider import systemone as S1
    from decider.prompt import build
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)

    def build_row(context, text, options, rng):
        ex = D.Example(context, [D.Q(text, list(options), -1)], "expA")
        d = build(ex, tok, rng, max_options=255, max_ctx_tokens=a.max_ctx, layout="state_first")
        return U.Row(
            np.asarray(d["ids"], np.int32),
            int(d["slots"][0]),
            int(d["nopts"][0]),
            np.asarray(d["perms"][0], np.int16),
        )

    data = Path(a.data_dir)
    pool = P.read_records([str(data / f) for f in POOL_FILES])
    dev = P.read_records([str(data / f) for f in DEV_FILES])
    splits = P.split_records(pool)
    half_a, half_b = P.split_record_halves(splits["train"])
    named = {
        "train_A": half_a,
        "train_B": half_b,
        "cal": splits["cal"],
        "select": splits["select"],
        "dev": dev,
    }
    if a.limit:
        named = {k: v[: a.limit] for k, v in named.items()}

    P.assert_record_disjoint(named)
    pool_states = {norm_state(r.state) for r in pool}
    dev_overlap = sum(norm_state(r.state) in pool_states for r in dev)
    print(
        f"[prep] pool {len(pool)} records, dev {len(dev)} records, dev states also in the pool: {dev_overlap}"
    )
    if dev_overlap:
        raise SystemExit("dev and pool share state text: refusing to continue")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, recs in named.items():
        t0 = time.time()
        units = U.build_units(
            recs, S1.render_state, S1.render_question, build_row, S1.isolated_rows, a.seed
        )
        with open(out / f"units_{name}.pkl", "wb") as f:
            pickle.dump(units, f, protocol=pickle.HIGHEST_PROTOCOL)
        rows = sum(len(u.rows) for u in units)
        toks = sum(u.tokens for u in units)
        print(
            f"[prep] {name:8s} {len(recs):6d} records {len(units):6d} questions {rows:6d} rows "
            f"{toks / 1e6:6.2f}M tokens {U.type_mix(units)} in {time.time() - t0:.0f}s",
            flush=True,
        )
    manifest = P.manifest(named)
    manifest["_seed"] = a.seed
    with open(out / "splits.json", "w") as f:
        json.dump(manifest, f, indent=1)
    with open(out / "units_dev.pkl", "rb") as f:
        dev_units = pickle.load(f)
    with open(out / "dev_domains.json", "w") as f:
        json.dump({u.qid: u.domain for u in dev_units}, f)
    print("[prep] wrote", out, "files:", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
