"""Exp A runner: one arm, one seed, one learning rate. Run on the cluster in decider's venv (see ``slurm/env.sh``).

    python -m rlcd.train_labels --arm a1 --lr 1e-4 --seed 0 --out $OUT            # train + evaluate
    python -m rlcd.train_labels --compute_reference --out $REF_DIR                 # log p_ref of every training unit
    python -m rlcd.train_labels --eval_only --adapters off --out $OUT              # A0: the start model
    python -m rlcd.train_labels --eval_only --adapter $RUN/lora_adapter.pt ...      # re-evaluate a saved adapter

Arms (``rlcd.torch_losses``): ``a1`` supervised log loss (RPS for SCORE); ``a3`` RLVR; ``a4`` RLCD-bandit (unclipped);
``a5`` RLCD-full through the analytic-gradient path; ``a6`` / ``a6r`` supervised loss on half A plus items mined (a6) or
drawn at random (a6r) from the held-back half B over rounds, with labels hidden until after selection
(``docs/decisions.md`` D11, D12, D-lean). Every arm gets the same total optimiser steps: S = ``epochs`` epochs over the
whole training split, whatever data the arm itself sees.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rlcd import gpu_eval as GE
from rlcd import mining as M
from rlcd import torch_losses as TL
from rlcd.data import CandidatePool, TypedQuestion
from rlcd.lora import LORA_TARGETS, adapter_state, add_lora, load_adapter, set_adapters
from rlcd.results import write_json
from rlcd.torch_batch import collate, original_order
from rlcd.units import Unit, plan_microbatches, type_mix

ARM_LOSS = {
    "a1": "supervised",
    "a3": "rlvr",
    "a4": "rlcd_bandit",
    "a5": "rlcd_full",
    "a6": "supervised",
    "a6r": "supervised",
}
TYPES = ("noul", "choice", "score")


class Stream:
    """An endless stream of micro-batches over a list of units, re-planned (and reshuffled) each epoch."""

    def __init__(self, units: list[Unit], max_tokens: int, rng: random.Random):
        self.max_tokens, self.rng = max_tokens, rng
        self.set_units(units)
        self.epochs = 0

    def set_units(self, units: list[Unit]) -> None:
        self.units, self._plan, self._pos = units, [], 0

    def next(self) -> list[Unit]:
        if self._pos >= len(self._plan):
            self._plan, self._pos = plan_microbatches(self.units, self.max_tokens, self.rng), 0
            self.epochs += 1
        batch = [self.units[i] for i in self._plan[self._pos]]
        self._pos += 1
        return batch


def select_round(
    arm: str,
    remaining: list[Unit],
    labels: dict[str, int],
    n: int,
    score_units,
    seed: int,
    r: int,
) -> tuple[list[Unit], list[Unit]]:
    """One A6 / A6r selection round over the not-yet-selected half-B units (whose ``gold`` is hidden).

    ``score_units(units)`` returns an answer distribution per unit (used by A6 only). Labels are looked up in
    ``labels`` only after the selection (``rlcd.data.CandidatePool.reveal``). Returns the newly selected units,
    now carrying their gold, and the units still remaining.
    """
    by_qid = {u.qid: u for u in remaining}
    pool = CandidatePool(
        [
            TypedQuestion(
                u.qid,
                u.rec_id,
                u.qid,
                u.qtype,
                tuple(str(i) for i in range(u.n)),
                labels[u.qid],
                u.rec_id,
            )
            for u in remaining
        ]
    )
    n = min(n, len(remaining))
    if arm == "a6":
        _, labelled = M.mine_round(
            pool, lambda qs: score_units([by_qid[q.qid] for q in qs]), n, "mined"
        )
    elif arm == "a6r":
        rng = np.random.default_rng([seed, r])
        _, labelled = M.mine_round(
            pool, lambda qs: score_units([by_qid[q.qid] for q in qs]), n, "random", rng
        )
    else:
        raise ValueError(f"not a mining arm: {arm}")
    chosen = {q.qid: q.gold for q in labelled}
    added = [by_qid[q].with_gold(g) for q, g in chosen.items()]
    return added, [u for u in remaining if u.qid not in chosen]


def steps_per_epoch(units: list[Unit], max_tokens: int, accum: int) -> int:
    return math.ceil(len(plan_microbatches(units, max_tokens, random.Random(0))) / accum)


def micro_losses(model, units: list[Unit], args, ref, gen):
    """Forward one micro-batch; returns per-unit ``UnitLoss`` objects."""
    dev = next(model.parameters()).device
    batch = collate(units, model.tok.pad_token_id).to(dev)
    logits = model.slot_logits(
        batch.input_ids, batch.attention_mask, batch.slot_idx, batch.slot_batch, batch.nopts
    )
    zs = TL.unit_z(original_order(logits, batch), batch, units)
    out = []
    for u, z in zip(units, zs, strict=True):
        ref_lp = torch.from_numpy(ref[u.qid]).to(dev) if ref is not None else None
        out.append(
            TL.unit_loss(ARM_LOSS[args.arm], u.qtype, z, u.gold, ref_lp, args.beta, args.k, gen)
        )
    return out, batch.input_ids.numel()


class Meter:
    """Running means of the logged quantities, by answer type."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = {t: {"nll": 0.0, "kl": 0.0, "mass": 0.0, "n": 0} for t in TYPES}
        self.tokens = 0

    def add(self, units, losses, tokens):
        for u, ul in zip(units, losses, strict=True):
            s = self.sum[u.qtype]
            s["nll"] += ul.nll.item()
            s["kl"] += ul.kl.item()
            s["mass"] += ul.mass.item()
            s["n"] += 1
        self.tokens += tokens

    def summary(self) -> dict:
        out = {}
        for t, s in self.sum.items():
            if s["n"]:
                out[t] = {"nll": s["nll"] / s["n"], "kl": s["kl"] / s["n"], "n": s["n"]}
                if t == "score":
                    out[t]["mass"] = s["mass"] / s["n"]
        n_all = sum(s["n"] for s in self.sum.values())
        out["all"] = {
            "nll": sum(s["nll"] for s in self.sum.values()) / max(1, n_all),
            "kl": sum(s["kl"] for s in self.sum.values()) / max(1, n_all),
        }
        return out


def measure_grad_share(model, params, batches: list[list[Unit]], args, ref, gen) -> dict:
    """Norm of the gradient each answer type contributes to one optimizer step (D4: report, do not correct).

    ``batches`` are the step's micro-batches. Each micro-batch loss is the mean over its questions, divided by the
    number of micro-batches, so the per-type losses (each summed and scaled the same way) add up to the step's loss and
    the per-type gradients add up to the step's gradient. Every type's gradient is accumulated over all the
    micro-batches before its norm is taken. Grads are left cleared.
    """
    out = {}
    n_all = sum(len(mb) for mb in batches)
    for t in TYPES:
        n_t = 0
        for mb in batches:
            sub = [u for u in mb if u.qtype == t]
            if not sub:
                continue
            losses, _ = micro_losses(model, sub, args, ref, gen)
            (torch.stack([ul.loss for ul in losses]).sum() / (len(mb) * len(batches))).backward()
            n_t += len(sub)
        if n_t:
            sq = sum((p.grad.float() ** 2).sum().item() for p in params if p.grad is not None)
            out[t] = {"grad_norm": math.sqrt(sq), "questions": n_t}
        for p in params:
            p.grad = None
    total = sum(v["grad_norm"] for v in out.values()) or 1.0
    for v in out.values():
        v["share_of_norms"] = v["grad_norm"] / total
        v["share_of_questions"] = v["questions"] / n_all
    return out


def quick_metrics(units: list[Unit], zs: list[np.ndarray]) -> dict:
    """Selection-split metrics: mean -log p_y and accuracy by type, RPS for SCORE, and the all-question mean nll."""
    dists = GE.distributions(units, zs)
    acc: dict[str, list] = {t: [] for t in TYPES}
    nll: dict[str, list] = {t: [] for t in TYPES}
    rps: list[float] = []
    for u, p in zip(units, dists, strict=True):
        nll[u.qtype].append(-math.log(max(float(p[u.gold]), 1e-12)))
        acc[u.qtype].append(float(np.argmax(p) == u.gold))
        if u.qtype == "score":
            cdf = np.cumsum(p)[:-1]
            obs = (np.arange(len(p) - 1) >= u.gold).astype(float)
            rps.append(float(((cdf - obs) ** 2).sum() / (len(p) - 1)))
    out = {
        t: {"nll": float(np.mean(nll[t])), "acc": float(np.mean(acc[t])), "n": len(nll[t])}
        for t in TYPES
        if nll[t]
    }
    if rps:
        out["score"]["rps"] = float(np.mean(rps))
    out["all"] = {"nll": float(np.mean([x for t in TYPES for x in nll[t]])), "n": len(units)}
    return out


def load_model(args, log):
    from decider.model import DecisionModel

    model = DecisionModel(args.model).cuda()
    n = add_lora(model.lm, args.lora_r, args.lora_alpha, LORA_TARGETS.split(","))
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(
        f"[lora] rank {args.lora_r} alpha {args.lora_alpha}: {n} layers wrapped, {ntr / 1e6:.1f}M trainable parameters"
    )
    return model


def evaluate_and_save(model, args, out: Path, log) -> dict:
    """Predict the evaluation splits, save the raw readouts, and return the selection-split metrics."""
    summary = {}
    for split in [x for x in args.eval_splits.split(",") if x]:
        units = GE.load_units(Path(args.data) / f"units_{split}.pkl")
        t0 = time.time()
        zs = GE.predict_units(model, units, log=log, tag=f" {split}")
        GE.save_predictions(out / f"preds_{split}.npz", units, zs)
        if split in ("select", "cal", "dev"):
            summary[split] = quick_metrics(units, zs)
        log(
            f"[eval] {split}: {len(units)} questions in {time.time() - t0:.0f}s {json.dumps(summary.get(split, {}).get('all', {}))}"
        )
    write_json(out / "eval_metrics.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", choices=list(ARM_LOSS), default="a1")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/projects/u6xe/louisk/rlcd_expA/data")
    ap.add_argument("--model", default="/projects/u6xe/louisk/models/decider9b-sft")
    ap.add_argument(
        "--epochs",
        type=float,
        default=2.0,
        help="S = epochs x steps per epoch over the whole training split",
    )
    ap.add_argument(
        "--steps_frac", type=float, default=1.0, help="fraction of S to run (smoke: 0.1)"
    )
    ap.add_argument(
        "--beta", type=float, default=0.1, help="forward-KL weight, identical across arms (D1)"
    )
    ap.add_argument("--k", type=int, default=8, help="sampled answers per question for the RL arms")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=float, default=128.0)
    ap.add_argument("--rounds", type=int, default=4, help="A6 / A6r rounds")
    ap.add_argument(
        "--n_per_round", type=int, default=1500, help="A6 / A6r items selected per round"
    )
    ap.add_argument(
        "--pool_frac", type=float, default=1.0, help="A6 / A6r: use this fraction of half B (smoke)"
    )
    ap.add_argument(
        "--reference", default="", help="pickle of log p_ref per question (needed when beta > 0)"
    )
    ap.add_argument("--eval_splits", default="cal,select,dev")
    ap.add_argument(
        "--grad_share_steps",
        default="",
        help="comma list of steps at which to measure per-type grad norms",
    )
    ap.add_argument(
        "--stop_step", type=int, default=0, help="debug: stop after this optimizer step"
    )
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--adapters", choices=["on", "off"], default="on")
    ap.add_argument("--adapter", default="", help="eval_only: load this saved adapter")
    ap.add_argument("--compute_reference", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"

    def log(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log("[args]", json.dumps(vars(args)))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    data = Path(args.data)
    model = load_model(args, log)

    if args.compute_reference or args.eval_only:
        if args.adapter:
            load_adapter(model, torch.load(args.adapter, map_location="cuda"))
        set_adapters(model, args.adapters == "on" and not args.compute_reference)
        if args.compute_reference:
            units = GE.load_units(data / "units_train_A.pkl") + GE.load_units(
                data / "units_train_B.pkl"
            )
            t0 = time.time()
            ref = GE.reference_logps(model, units, log=log)
            GE.save_reference(out / "ref_logp.pkl", ref)
            log(
                f"[reference] {len(ref)} questions in {time.time() - t0:.0f}s -> {out / 'ref_logp.pkl'}"
            )
        else:
            evaluate_and_save(model, args, out, log)
        return

    # ---------------------------------------------------------------- data and schedule
    train_a = GE.load_units(data / "units_train_A.pkl")
    train_b = GE.load_units(data / "units_train_B.pkl")
    all_train = train_a + train_b
    s_full = math.ceil(args.epochs * steps_per_epoch(all_train, args.max_tokens, args.accum))
    total = max(1, round(s_full * args.steps_frac))
    warmup = min(args.warmup, max(1, total // 10))
    log(
        f"[sched] S = {s_full} optimizer steps for {args.epochs} epochs over {len(all_train)} questions; running {total}"
    )

    ref = None
    if args.beta > 0:
        with open(args.reference, "rb") as f:
            ref = pickle.load(f)
        log(f"[ref] {len(ref)} reference distributions from {args.reference}")

    mining = args.arm in ("a6", "a6r")
    if mining:
        labels = {u.qid: u.gold for u in train_b}
        remaining = [replace(u, gold=None) for u in train_b]  # labels hidden until selection
        if args.pool_frac < 1.0:
            keep = random.Random(0).sample(
                range(len(remaining)), int(args.pool_frac * len(remaining))
            )
            remaining = [remaining[i] for i in sorted(keep)]
        train_units = list(train_a)
        round_len = total // args.rounds
        log(
            f"[mining] half A {len(train_a)} questions; half B {len(remaining)} candidates; {args.rounds} rounds of "
            f"{round_len} steps, {args.n_per_round} per round"
        )
    else:
        train_units = all_train
        round_len = 0
    stream = Stream(train_units, args.max_tokens, rng)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    gen = torch.Generator(device="cuda").manual_seed(args.seed + 12345)
    gen_measure = torch.Generator(device="cuda").manual_seed(args.seed + 54321)
    share_steps = {int(s) for s in args.grad_share_steps.split(",") if s}

    def lr_at(s: int) -> float:
        if s < warmup:
            return args.lr * (s + 1) / warmup
        return (
            args.lr
            * 0.5
            * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total - warmup))))
        )

    def mining_round(r: int, step: int) -> None:
        nonlocal remaining
        t0 = time.time()

        def score_units(us):
            return GE.distributions(us, GE.predict_units(model, us, log=log, tag=f" round{r}"))

        added, remaining = select_round(
            args.arm, remaining, labels, args.n_per_round, score_units, args.seed, r
        )
        stream.set_units(stream.units + added)
        log(
            f"[mining] round {r} step {step}: selected {len(added)} {type_mix(added)}; "
            f"{len(remaining)} candidates left; training set {len(stream.units)} questions "
            f"({time.time() - t0:.0f}s)"
        )

    # ----------------------------------------------------------------------- train
    model.train()
    meter, t_log = Meter(), time.time()
    t0 = time.time()
    share_log = []
    diverged = False
    for step in range(total):
        if mining and round_len and step % round_len == 0 and step // round_len < args.rounds:
            mining_round(step // round_len, step)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        micro_batches = [stream.next() for _ in range(args.accum)]
        if step in share_steps:
            sh = measure_grad_share(model, params, micro_batches, args, ref, gen_measure)
            share_log.append({"step": step, **sh})
            log(
                f"[grad-share] step {step} "
                + json.dumps({t: round(v["share_of_norms"], 3) for t, v in sh.items()})
                + " questions "
                + json.dumps({t: v["questions"] for t, v in sh.items()})
            )
        for units in micro_batches:
            losses, ntok = micro_losses(model, units, args, ref, gen)
            loss = torch.stack([ul.loss for ul in losses]).mean()
            if not torch.isfinite(loss):
                log(f"[diverged] non-finite loss at step {step}")
                diverged = True
                break
            (loss / args.accum).backward()
            meter.add(units, losses, ntok)
        if diverged:
            break
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        if done % 20 == 0 or done == total:
            now = time.time()
            s = meter.summary()
            tps = meter.tokens / (now - t_log)
            log(
                f"[train] step {done}/{total} nll {s['all']['nll']:.4f} kl {s['all']['kl']:.4f} gn {gn:.2f} "
                f"lr {lr_at(step):.2e} {(now - t0) / 60:.1f}min {tps:.0f}tok/s mem {torch.cuda.max_memory_allocated() / 1e9:.0f}GB "
                + json.dumps({t: round(v["nll"], 3) for t, v in s.items() if t != "all"})
            )
            with open(out / "metrics.jsonl", "a") as f:
                f.write(
                    json.dumps(
                        {"step": done, "gn": float(gn), "lr": lr_at(step), "tok_s": tps, **s}
                    )
                    + "\n"
                )
            meter.reset()
            t_log = now
        if args.stop_step and done >= args.stop_step:
            log(f"[stop] debug stop at step {done}")
            break

    # ------------------------------------------------------------------- finish
    train_seconds = time.time() - t0
    torch.save(adapter_state(model), out / "lora_adapter.pt")
    summary = {}
    if not diverged:
        summary = evaluate_and_save(model, args, out, log)
    write_json(
        out / "run.json",
        {
            "arm": args.arm,
            "lr": args.lr,
            "seed": args.seed,
            "S_full": s_full,
            "steps": total,
            "diverged": diverged,
            "train_seconds": train_seconds,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            "train_questions_end": len(stream.units),
            "grad_share": share_log,
            "eval": summary,
            "args": vars(args),
        },
    )
    log("[done]", out)


if __name__ == "__main__":
    main()
