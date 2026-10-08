"""Merge an Exp A LoRA adapter into the start model so decider can serve it (for JevBench).

    python scripts/merge_expA_adapter.py --adapter RUN/lora_adapter.pt --out MERGED_DIR [--temperature 1.0]

Run in decider's venv on a GPU node. The merged directory is a plain checkpoint plus a ``decider_config.json`` copied
from the start model with the temperature set explicitly: 1.0 for the "raw" rows (D2) and 1.05 for the start model
"as served". Per-answer-type temperatures fitted on the calibration split are not applied here: decider serves one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decider.model import DecisionModel  # noqa: E402

from rlcd.lora import LORA_TARGETS, add_lora, load_adapter, merge_lora  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="/projects/u6xe/louisk/models/decider9b-sft")
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=128.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()
    m = DecisionModel(a.base, grad_ckpt=False).cuda().eval()
    n = add_lora(m.lm, a.r, a.alpha, LORA_TARGETS.split(","))
    state = torch.load(a.adapter, map_location="cuda")
    load_adapter(m, state)
    print(
        f"[merge] {n} adapted layers, {len(state)} adapter tensors loaded onto {a.base}", flush=True
    )
    merge_lora(m.lm)
    os.makedirs(a.out, exist_ok=True)
    m.lm.save_pretrained(a.out)
    m.tok.save_pretrained(a.out)
    cfg = json.loads((Path(a.base) / "decider_config.json").read_text())
    cfg["temperature"] = a.temperature
    (Path(a.out) / "decider_config.json").write_text(json.dumps(cfg))
    for extra in ("chat_template.jinja",):
        if (Path(a.base) / extra).exists() and not (Path(a.out) / extra).exists():
            shutil.copy(Path(a.base) / extra, Path(a.out) / extra)
    print(f"[merge] saved {a.out} (temperature {a.temperature})", flush=True)


if __name__ == "__main__":
    main()
