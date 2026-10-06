"""Answer distributions and per-arm losses in torch (the training-side twin of ``rlcd.estimators``).

``z`` is the readout of one *unit* in original option order: the option logits for NOUL / CHOICE, and for SCORE the
level logit differences ``l_k = z_yes - z_no`` of its isolated rows (``docs/decisions.md`` D3). The answer
distribution is ``softmax(z)`` for NOUL / CHOICE and ``y_k / sum_j y_k`` with ``y = sigmoid(l)`` for SCORE.

Arms (the shared objective is ``E[S(p, y)] - beta KL(p_ref || p)``; every loss below is *minimised*):

``supervised`` (A1)   -log p_y (NOUL / CHOICE), RPS / (K - 1) (SCORE); gradient by autograd.
``rlcd_full`` (A5)    the same objective through the RL code path: the *analytic* gradient of ``S`` is computed
                      without autograd and injected as a surrogate. It must agree with ``supervised``.
``rlvr`` (A3)         K sampled answers, reward ``1[a = y]``, group-mean baseline.
``rlcd_bandit`` (A4)  K sampled answers, reward ``1[a = y] / p(a)`` (stop-gradient), group-mean baseline, unclipped;
                      on SCORE questions this optimises the log score (RPS is not recoverable from one indicator).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rlcd.torch_batch import Batch
from rlcd.units import Unit

EPS = 1e-12
ARMS = ("supervised", "rlcd_full", "rlvr", "rlcd_bandit")


def unit_z(orig: torch.Tensor, batch: Batch, units: Sequence[Unit]) -> list[torch.Tensor]:
    """The readout ``z`` of each unit from logits already in original option order (``[R, Kmax]``)."""
    out = []
    for u, (a, b) in zip(units, batch.spans, strict=True):
        if u.qtype == "score":
            rows = orig[a:b]
            out.append(rows[:, 1] - rows[:, 0])  # z_yes - z_no, one per level
        else:
            out.append(orig[a, : u.n])
    return out


def logp_from_z(qtype: str, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Log answer distribution over original options, and the SCORE mass ``sum_k y_k`` (1 for the others)."""
    if qtype == "score":
        logy = F.logsigmoid(z)
        lse = torch.logsumexp(logy, 0)
        return logy - lse, lse.exp()
    return F.log_softmax(z, 0), z.new_ones(())


def log_score_loss(logp: torch.Tensor, gold: int) -> torch.Tensor:
    return -logp[gold]


def rps_loss(logp: torch.Tensor, gold: int) -> torch.Tensor:
    """Ranked probability score divided by K - 1 (``docs/decisions.md`` D4)."""
    p = logp.exp()
    k = p.numel()
    cdf = torch.cumsum(p, 0)[:-1]
    obs = (torch.arange(k - 1, device=p.device) >= gold).to(p.dtype)
    return ((cdf - obs) ** 2).sum() / (k - 1)


def supervised_loss(qtype: str, z: torch.Tensor, gold: int) -> torch.Tensor:
    logp, _ = logp_from_z(qtype, z)
    return rps_loss(logp, gold) if qtype == "score" else log_score_loss(logp, gold)


def rps_score_grad_probs(p: torch.Tensor, gold: int) -> torch.Tensor:
    """``d(-RPS / (K-1)) / dp``: the gradient of the SCORE reward w.r.t. the level probabilities."""
    k = p.numel()
    cdf = torch.cumsum(p, 0)[:-1]
    obs = (torch.arange(k - 1, device=p.device) >= gold).to(p.dtype)
    d = 2.0 * (cdf - obs)
    g = torch.zeros_like(p)
    g[:-1] = torch.flip(torch.cumsum(torch.flip(d, [0]), 0), [0])  # sum over k >= j of dRPS/dP_k
    return -g / (k - 1)


@torch.no_grad()
def full_score_grad(qtype: str, z: torch.Tensor, gold: int) -> torch.Tensor:
    """Analytic ascent gradient of ``S(p, y)`` w.r.t. ``z``, derived without autograd.

    NOUL / CHOICE: ``e_y - p``. SCORE: ``dS/dp`` through the normalisation, ``(g - <p, g>) / mass``, then
    through the sigmoid, ``y (1 - y)`` (``rlcd.isolated.score_grad_level_logits`` in numpy).
    """
    if qtype != "score":
        p = torch.softmax(z, 0)
        g = -p
        g[gold] += 1.0
        return g
    y = torch.sigmoid(z).clamp(EPS, 1.0 - EPS)
    mass = y.sum()
    p = y / mass
    g_p = rps_score_grad_probs(p, gold)
    return (g_p - (p * g_p).sum()) / mass * y * (1.0 - y)


def full_score_loss(qtype: str, z: torch.Tensor, gold: int) -> torch.Tensor:
    """Surrogate whose gradient w.r.t. ``z`` is minus the analytic gradient of ``S`` (so minimising ascends ``S``)."""
    return -(z * full_score_grad(qtype, z, gold)).sum()


def sampled_policy_loss(
    arm: str,
    qtype: str,
    z: torch.Tensor,
    gold: int,
    k: int,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """REINFORCE with ``k`` sampled answers and a group-mean baseline (``rlcd.estimators.advantages``).

    The group mean includes the sample itself, so the gradient is shrunk by exactly ``(k - 1) / k`` and keeps its
    direction. ``rlcd_bandit`` weights the hit by ``1 / p(a)`` with a stop-gradient on ``p(a)``.
    """
    logp, _ = logp_from_z(qtype, z)
    p = logp.detach().exp()
    a = torch.multinomial(p, k, replacement=True, generator=gen)
    hit = (a == gold).to(p.dtype)
    if arm == "rlvr":
        reward = hit
    elif arm == "rlcd_bandit":
        reward = hit / p[a].clamp_min(EPS)
    else:
        raise ValueError(f"not a sampled-policy arm: {arm}")
    adv = reward - reward.mean()
    return -(adv * logp[a]).mean()


def forward_kl(logp_ref: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    """``KL(p_ref || p)`` (``docs/decisions.md`` D1): large when ``p`` drops an option the reference supports."""
    return (logp_ref.exp() * (logp_ref - logp)).sum()


@dataclass
class UnitLoss:
    loss: torch.Tensor  # what is backpropagated (arm term + beta * KL)
    nll: torch.Tensor  # -log p_y, detached, for logging
    kl: torch.Tensor  # forward KL to the reference, detached (0 without a reference)
    mass: torch.Tensor  # SCORE: sum of yes probabilities (1 otherwise), detached


def unit_loss(
    arm: str,
    qtype: str,
    z: torch.Tensor,
    gold: int,
    ref_logp: torch.Tensor | None,
    beta: float,
    k: int,
    gen: torch.Generator | None = None,
) -> UnitLoss:
    """The loss of one question for ``arm``, plus the forward-KL penalty ``beta * KL(p_ref || p)``."""
    if arm == "supervised":
        base = supervised_loss(qtype, z, gold)
    elif arm == "rlcd_full":
        base = full_score_loss(qtype, z, gold)
    elif arm in ("rlvr", "rlcd_bandit"):
        base = sampled_policy_loss(arm, qtype, z, gold, k, gen)
    else:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    logp, mass = logp_from_z(qtype, z)
    kl = forward_kl(ref_logp, logp) if ref_logp is not None else z.new_zeros(())
    loss = base + beta * kl if (beta > 0 and ref_logp is not None) else base
    return UnitLoss(loss, -logp[gold].detach(), kl.detach(), mass.detach())
