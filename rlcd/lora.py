"""LoRA for a decider backbone, without ``peft`` (the cluster environment does not have it).

Copied, with small edits, from the LoRA patch that ``Louiskq/jev-class-9b`` applied to Mapika's decider
(``decider/train.py``, Apache 2.0). Copied rather than imported so this repository does not depend on that
repository at runtime.

``y = base(x) + (alpha / r) * B A x`` with the base frozen and ``A``, ``B`` trained in fp32; ``B`` starts at
zero, so the wrapped model equals the base model exactly until it is trained.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F

LORA_TARGETS = (
    "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj"
)


class LoRALinear(torch.nn.Module):
    """``base(x) + scale * B (A x)``; ``enabled = False`` gives the frozen base model's output."""

    def __init__(self, base: torch.nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.scale, self.enabled = base, alpha / r, True
        dev = base.weight.device
        self.A = torch.nn.Parameter(
            torch.empty(r, base.in_features, dtype=torch.float32, device=dev)
        )
        self.B = torch.nn.Parameter(
            torch.zeros(base.out_features, r, dtype=torch.float32, device=dev)
        )
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        return out + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale

    @torch.no_grad()
    def merged(self) -> torch.nn.Linear:
        w = self.base.weight
        w.copy_((w.float() + (self.B @ self.A) * self.scale).to(w.dtype))
        return self.base


def add_lora(lm: torch.nn.Module, r: int, alpha: float, targets: Iterable[str]) -> int:
    """Freeze ``lm`` and wrap every ``nn.Linear`` whose attribute name is in ``targets``; returns how many."""
    targets = set(targets)
    for p in lm.parameters():
        p.requires_grad_(False)
    n = 0
    for mod in list(lm.modules()):
        for name, child in list(mod.named_children()):
            if name in targets and type(child) is torch.nn.Linear:
                setattr(mod, name, LoRALinear(child, r, alpha))
                n += 1
    lm.enable_input_require_grads()  # gradient checkpointing needs a grad-requiring input with frozen embeddings
    return n


def set_adapters(model: torch.nn.Module, enabled: bool) -> None:
    """Switch every adapter on or off: off gives the frozen start model (the KL reference)."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


def adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The adapter weights only (small, fp32): what a run saves."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}


def load_adapter(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in unexpected]
    if bad:
        raise ValueError(f"unexpected adapter keys: {bad[:3]}")


def merge_lora(lm: torch.nn.Module) -> None:
    """Fold every adapter into its base weight and put the plain ``nn.Linear`` back."""
    _swap(lm, lambda m: m.merged() if isinstance(m, LoRALinear) else m)


def _swap(lm: torch.nn.Module, pick: Callable[[torch.nn.Module], torch.nn.Module]) -> None:
    for mod in list(lm.modules()):
        for name, child in list(mod.named_children()):
            new = pick(child)
            if new is not child:
                setattr(mod, name, new)
