"""Forward-only passes over units on the GPU: predictions, candidate scoring and the KL reference.

Predictions are saved as the raw readout ``z`` of each unit at temperature 1 (option logits for NOUL / CHOICE, level
logit differences ``z_yes - z_no`` for SCORE), in original option order. Temperature scaling, the SCORE
normalisation and every metric are computed offline from these arrays (``rlcd.evaluate``), so one saved file
serves "raw", "as served" (T = 1.05) and "+TS" rows without another forward pass.
"""

from __future__ import annotations

import pickle
import random
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from rlcd import torch_losses as TL
from rlcd.torch_batch import collate, original_order
from rlcd.units import Unit, plan_microbatches

QTYPE_CODE = {"noul": 0, "choice": 1, "score": 2}


@torch.no_grad()
def predict_units(
    model,
    units: Sequence[Unit],
    max_tokens: int = 24576,
    log=print,
    tag: str = "",
) -> list[np.ndarray]:
    """The readout ``z`` of every unit (float32, original option order), in the order of ``units``."""
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    pad_id = model.tok.pad_token_id
    out: dict[int, np.ndarray] = {}
    batches = plan_microbatches(units, max_tokens, random.Random(0))
    t0 = time.time()
    for n, idxs in enumerate(batches):
        sub = [units[i] for i in idxs]
        batch = collate(sub, pad_id).to(dev)
        logits = model.slot_logits(
            batch.input_ids, batch.attention_mask, batch.slot_idx, batch.slot_batch, batch.nopts
        )
        for i, z in zip(idxs, TL.unit_z(original_order(logits, batch), batch, sub), strict=True):
            out[i] = z.float().cpu().numpy()
        if n % 50 == 0:
            log(f"[predict]{tag} {n}/{len(batches)} batches {time.time() - t0:.0f}s")
    if was_training:
        model.train()
    return [out[i] for i in range(len(units))]


def distributions(units: Sequence[Unit], zs: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Answer distributions at T = 1 in original option order (SCORE: normalised yes probabilities)."""
    out = []
    for u, z in zip(units, zs, strict=True):
        logp, _ = TL.logp_from_z(u.qtype, torch.from_numpy(np.asarray(z, dtype=np.float64)))
        out.append(logp.exp().numpy())
    return out


def reference_logps(model, units: Sequence[Unit], log=print) -> dict[str, np.ndarray]:
    """``log p_ref`` of every unit under the frozen start model (adapters must be off), keyed by question id."""
    zs = predict_units(model, units, log=log, tag=" ref")
    out = {}
    for u, z in zip(units, zs, strict=True):
        lp, _ = TL.logp_from_z(u.qtype, torch.from_numpy(z.astype(np.float64)))
        out[u.qid] = lp.numpy().astype(np.float32)
    return out


def save_predictions(path: str | Path, units: Sequence[Unit], zs: Sequence[np.ndarray]) -> None:
    """One ``.npz``: ids, type codes, option counts, gold (-1 when hidden) and the padded readouts."""
    kmax = max(len(z) for z in zs)
    z_pad = np.full((len(zs), kmax), np.nan, dtype=np.float32)
    for i, z in enumerate(zs):
        z_pad[i, : len(z)] = z
    np.savez_compressed(
        path,
        qid=np.array([u.qid for u in units]),
        rec_id=np.array([u.rec_id for u in units]),
        qtype=np.array([QTYPE_CODE[u.qtype] for u in units], dtype=np.int8),
        n=np.array([u.n for u in units], dtype=np.int16),
        gold=np.array([-1 if u.gold is None else u.gold for u in units], dtype=np.int16),
        domain=np.array([u.domain for u in units]),
        z=z_pad,
    )


def load_units(path: str | Path) -> list[Unit]:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_reference(path: str | Path, ref: dict[str, np.ndarray]) -> None:
    with open(path, "wb") as f:
        pickle.dump(ref, f, protocol=pickle.HIGHEST_PROTOCOL)
