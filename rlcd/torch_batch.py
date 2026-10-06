"""Turn planned micro-batches of units into tensors for ``decider.model.DecisionModel.slot_logits``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rlcd.units import Unit, padded_len


@dataclass
class Batch:
    """Tensors for one micro-batch plus how its rows map back to units and to original option order."""

    input_ids: torch.Tensor  # [R, T]
    attention_mask: torch.Tensor  # [R, T]
    slot_idx: torch.Tensor  # [R] position read out in each row
    slot_batch: torch.Tensor  # [R] row index (arange)
    nopts: torch.Tensor  # [R] options shown per row
    inv: torch.Tensor  # [R, Kmax]: inv[r, i] = position at which ORIGINAL option i was shown
    spans: list[tuple[int, int]]  # rows [start, end) of each unit

    def to(self, device: torch.device | str) -> Batch:
        return Batch(
            self.input_ids.to(device),
            self.attention_mask.to(device),
            self.slot_idx.to(device),
            self.slot_batch.to(device),
            self.nopts.to(device),
            self.inv.to(device),
            self.spans,
        )


def collate(units: Sequence[Unit], pad_id: int) -> Batch:
    """Right-pad every row of every unit to one length (a multiple of 64) and build the index tensors."""
    rows = [r for u in units for r in u.rows]
    spans, start = [], 0
    for u in units:
        spans.append((start, start + len(u.rows)))
        start += len(u.rows)
    t = padded_len(max(len(r.ids) for r in rows))
    n_rows = len(rows)
    input_ids = torch.full((n_rows, t), pad_id, dtype=torch.long)
    attn = torch.zeros((n_rows, t), dtype=torch.long)
    kmax = max(r.nopts for r in rows)
    inv = torch.zeros((n_rows, kmax), dtype=torch.long)
    for b, r in enumerate(rows):
        n = len(r.ids)
        input_ids[b, :n] = torch.from_numpy(r.ids.astype("int64"))
        attn[b, :n] = 1
        for pos, orig in enumerate(r.perm.tolist()):
            inv[b, orig] = pos
    return Batch(
        input_ids,
        attn,
        torch.tensor([r.slot for r in rows], dtype=torch.long),
        torch.arange(n_rows, dtype=torch.long),
        torch.tensor([r.nopts for r in rows], dtype=torch.long),
        inv,
        spans,
    )


def original_order(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    """``logits`` [R, 255] in displayed order -> [R, Kmax] in original option order (columns >= nopts are junk)."""
    return torch.gather(logits, 1, batch.inv)
