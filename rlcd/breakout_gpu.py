"""Exp B on a GPU: Breakout rollouts, the PPO / belief / retention trainer behind B1, B3 and B5, the supervised belief
fit behind B4, and the evaluation of games and beliefs.

Needs decider (Apache 2.0, https://github.com/Mapika/decider) on ``PYTHONPATH`` for the model wrapper
(``decider.games.rl.Policy``), the Breakout environment (``decider.games.envs``) and the question builder. Everything
that does not need a GPU is in ``rlcd.breakout`` and is unit-tested.

One loop trains every RL arm; the arms differ only in the loss weights (``ARMS``), so B1 against B3 is a comparison of
loss terms and nothing else:

* **actor**: PPO clipped surrogate on the served action probability, per-step return baselines (decider's plain run);
* **belief**: log score of the model's yes / no answer to a question about what happens next, scored against what
  happened (a proper scoring rule);
* **retention**: KL(start model || current) on replayed supervised decider questions.

Weights actor : belief : retention are 1 : 0 : 0 (B1), 1 : 2 : 7 (B3) and 1 : 2 : 0 (B5); decider's own recipe uses
0.1 : 0.2 : 0.7. Rendering consistency and the retention gate of decider's recipe are omitted
(``docs/decisions.md`` D6).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch
import torch.nn.functional as F
from decider import data as D
from decider.games import envs as G
from decider.games.rl import REWARD_SCALE, NoShuffle, Policy, a_success
from decider.prompt import build

from rlcd import breakout as BK

ARMS = {  # actor, belief, retention weights
    "b1": (1.0, 0.0, 0.0),
    "b3": (1.0, 2.0, 7.0),
    "b5": (1.0, 2.0, 0.0),
}
MAX_T = 300


# ---------------------------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------------------------


def play(
    pol: Policy,
    seeds: list[int],
    rng: random.Random,
    max_t: int = MAX_T,
    greedy: bool = False,
    belief_every: int = BK.BELIEF_EVERY,
    keep_items: bool = True,
) -> list[dict[str, Any]]:
    """One Breakout episode per env seed, all stepped together; belief questions asked every ``belief_every`` steps."""
    envs = []
    for sd in seeds:
        g = G.GAMES["breakout"]()
        g.reset(seed=sd)
        tr = BK.BeliefTracker(rng, belief_every)
        tr.start(g._s())
        envs.append(
            dict(
                g=g,
                seed=sd,
                tr=tr,
                done=False,
                k=0,
                opt=g.options[0],
                items=[],
                acts=[],
                rews=[],
                logps=[],
                last=g.score(),
                nopt=len(g.options),
            )
        )
    pol.m.eval()
    while any(not e["done"] for e in envs):
        due = [e for e in envs if not e["done"]]
        items = []
        for e in due:
            ex, fr = pol.example(e["g"])
            items.append(pol.make_item(ex, fr))
            e["tr"].maybe_ask(
                e["k"],
                e["g"]._s(),
                e["g"].score(),
                f"{e['g'].intro}\n\nSituation: {e['g'].text()}",
            )
        with torch.no_grad():
            lg = pol.logits(items)
        for e, it, row in zip(due, items, lg, strict=True):
            p = torch.softmax(row[: e["nopt"]].float(), -1).cpu()
            a = int(p.argmax()) if greedy else int(torch.multinomial(p, 1))
            if keep_items:
                e["items"].append(it)
            e["acts"].append(a)
            e["logps"].append(float(torch.log(p[a] + 1e-12)))
            e["rews"].append(0.0)
            e["opt"] = e["g"].options[a]
        for e in due:
            _, done = e["g"].step(e["opt"])
            e["k"] += 1
            sc = e["g"].score()
            e["rews"][-1] += (sc - e["last"]) / REWARD_SCALE.get("breakout", 1.0)
            e["last"] = sc
            if done and e["g"].success():
                e["rews"][-1] += a_success
            e["tr"].observe(e["k"], e["g"]._s(), sc, done)
            if done or e["k"] >= max_t:
                e["done"] = True  # questions still open at truncation are dropped
    out = [
        dict(
            seed=e["seed"],
            items=e["items"],
            acts=e["acts"],
            rews=e["rews"],
            logps=e["logps"],
            score=float(e["g"].score()),
            nopt=e["nopt"],
            beliefs=[dict(b, seed=e["seed"]) for b in e["tr"].resolved],
            steps=e["k"],
        )
        for e in envs
    ]
    for e in envs:
        e["g"].close()
    return out


def mean_score(trajs: list[dict[str, Any]]) -> float:
    return float(np.mean([t["score"] for t in trajs]))


# ---------------------------------------------------------------------------------------------
# belief questions as model inputs
# ---------------------------------------------------------------------------------------------


def belief_items(pol: Policy, beliefs: list[dict[str, Any]]) -> list[Any]:
    # D.Example / D.Q with an explicit task: D.load_cache rebinds __main__.Example / Q (decider.data.core.load_cache),
    # so names imported elsewhere would silently change type once the replay rows are loaded.
    return [
        build(
            D.Example(b["ctx"], [D.Q(BK.BELIEFS[b["kind"]], ["no", "yes"], b["gold"])], "belief"),
            pol.tok,
            NoShuffle(),
            max_ctx_tokens=1024,
        )
        for b in beliefs
    ]


def belief_logit_diff(pol: Policy, beliefs: list[dict[str, Any]], bs: int = 64) -> np.ndarray:
    """``logit(yes) - logit(no)`` at T = 1 for each belief question (``sigmoid`` of it is the stated P(yes))."""
    items = belief_items(pol, beliefs)
    out: list[float] = []
    pol.m.eval()
    with torch.no_grad():
        for i in range(0, len(items), bs):
            lg = pol.logits(items[i : i + bs]).float()[:, :2]
            out += (lg[:, 1] - lg[:, 0]).cpu().tolist()
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------------------------


def replay_rows(pol: Policy, path: str, n: int, rng: random.Random) -> tuple[list[Any], list[Any]]:
    """``n`` single-question supervised rows with the start model's log-probs (a fixed reference)."""
    train, _ = D.load_cache(path)
    pick = [e for e in rng.sample(train, min(len(train), 20 * n)) if len(e.qs) == 1][: 4 * n]
    items = []
    for e in pick:
        it = build(e, pol.tok, random.Random(0), max_options=10, max_ctx_tokens=1024)
        if len(it["ids"]) <= 1024:
            items.append(it)
        if len(items) == n:
            break
    ref = []
    with torch.no_grad():
        for i in range(0, len(items), 32):
            ref += list(F.log_softmax(pol.logits(items[i : i + 32]).float(), -1).cpu())
    del train
    return items, ref


def retention_kl(pol: Policy, items: list[Any], ref: list[Any]) -> torch.Tensor:
    lp = F.log_softmax(pol.logits(items).float(), -1)
    r = torch.stack(ref).to(lp.device)
    ok = torch.isfinite(r) & torch.isfinite(lp)
    return (r.exp() * (r - lp)).masked_fill(~ok, 0.0).sum(1).mean()


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------


def _logger(out: Path):
    f = open(out / "rl.log", "a")  # noqa: SIM115 (kept open for the life of the run)

    def log(*s: Any) -> None:
        m = " ".join(str(x) for x in s)
        print(m, flush=True)
        f.write(m + "\n")
        f.flush()

    return log


def evaluate_select(pol: Policy, rng: random.Random, n: int = 50) -> float:
    """Mean greedy score on the selection seeds (the checkpoint-selection criterion and the delta_game input)."""
    torch.manual_seed(0)
    return mean_score(
        play(pol, list(BK.SELECT_SEEDS[:n]), rng, MAX_T, greedy=True, keep_items=False)
    )


def train(a: Any) -> None:
    """Train one B1 / B3 / B5 run (``a`` is the parsed ``breakout_cli train`` namespace)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = _logger(out)
    w_act, w_bel, w_ret = ARMS[a.arm]
    if a.belief_w is not None:
        w_bel = a.belief_w
    if a.ret_w is not None:
        w_ret = a.ret_w
    BK.seed_sets(a.seed)
    log("[args]", json.dumps(vars(a)), f"weights actor={w_act} belief={w_bel} retention={w_ret}")
    rng = random.Random(a.seed)
    eval_rng = random.Random(10_000 + a.seed)
    torch.manual_seed(a.seed)
    pol = Policy(a.init, False)
    t_start = time.time()
    rep_items: list[Any] = []
    rep_ref: list[Any] = []
    if w_ret > 0:
        t0 = time.time()
        rep_items, rep_ref = replay_rows(pol, a.replay, a.ret_rows, random.Random(0))
        log(
            f"[retention] {len(rep_items)} replay rows, reference from the start model in {time.time() - t0:.0f}s"
        )
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, betas=(0.9, 0.95))
    evals: list[tuple[int, float]] = [(0, evaluate_select(pol, eval_rng, a.select_n))]
    log(f"[eval] start greedy selection-seed score {evals[0][1]:.2f}")
    counter = 0
    belief_file = open(out / "beliefs.jsonl", "a")  # noqa: SIM115
    metrics_file = open(out / "metrics.jsonl", "a")  # noqa: SIM115
    for it in range(1, a.iters + 1):
        t0 = time.time()
        seeds = [BK.train_env_seed(a.seed, counter + j) for j in range(a.envs)]
        counter += a.envs
        trajs = play(pol, seeds, rng, a.max_t)
        t_roll = time.time() - t0
        rets = [BK.returns(t["rews"], a.gamma) for t in trajs]
        advs_by_ep = BK.step_advantages(rets)
        items, acts, advs, oldlp = [], [], [], []
        for t, adv in zip(trajs, advs_by_ep, strict=True):
            items += t["items"]
            acts += t["acts"]
            advs += adv
            oldlp += t["logps"]
        bel = [b for t in trajs for b in t["beliefs"]]
        for b in bel:
            belief_file.write(json.dumps(dict(b, iter=it)) + "\n")
        belief_file.flush()
        bitems = belief_items(pol, bel) if (bel and w_bel > 0) else []
        bgold = [b["gold"] for b in bel]
        order = list(range(len(items)))
        rng.shuffle(order)
        order = order[: a.update_samples]
        pol.m.train()
        chunks = [order[i :: a.ppo_steps] for i in range(a.ppo_steps)]
        kl_last = pg_tot = bl_tot = rk_tot = gn = 0.0
        nb = nsteps = 0
        for chunk in chunks:
            kls = []
            for i in range(0, len(chunk), a.batch):
                idx = chunk[i : i + a.batch]
                logp = F.log_softmax(pol.logits([items[j] for j in idx]).float(), -1)
                act = torch.tensor([acts[j] for j in idx], device="cuda")
                A = torch.tensor([advs[j] for j in idx], device="cuda")
                lp = logp.gather(1, act[:, None]).squeeze(1)
                olp = torch.tensor([oldlp[j] for j in idx], device="cuda")
                ratio = torch.exp(lp - olp)
                pg = -torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A).mean()
                valid = torch.isfinite(logp)
                lps = logp.masked_fill(~valid, 0.0)
                ent = -(lps.exp().masked_fill(~valid, 0.0) * lps).sum(1).mean()
                (w_act * (pg - a.entropy * ent) * len(idx) / len(chunk)).backward()
                pg_tot += pg.item() * len(idx)
                nb += len(idx)
                kls.append(float((olp - lp).mean()))
            if bitems:  # belief: log score against what happened
                bidx = rng.sample(range(len(bitems)), min(a.belief_batch, len(bitems)))
                for i in range(0, len(bidx), a.batch):
                    sub = bidx[i : i + a.batch]
                    lg = pol.logits([bitems[j] for j in sub]).float()[:, :2]
                    ce = F.cross_entropy(lg, torch.tensor([bgold[j] for j in sub], device="cuda"))
                    (w_bel * ce * len(sub) / len(bidx)).backward()
                    bl_tot += ce.item() * len(sub) / len(bidx)
            if w_ret > 0:  # retention: stay close to the start model
                ridx = rng.sample(range(len(rep_items)), a.ret_batch)
                rk = retention_kl(pol, [rep_items[j] for j in ridx], [rep_ref[j] for j in ridx])
                (w_ret * rk).backward()
                rk_tot += rk.item()
            nsteps += 1
            gn = float(torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            kl_last = float(np.mean(kls)) if kls else 0.0
            if kl_last > a.kl_stop:
                break
        row = dict(
            iter=it,
            sampled_score=round(mean_score(trajs), 3),
            decisions=len(items),
            beliefs=len(bel),
            pg=pg_tot / max(1, nb),
            kl=kl_last,
            belief_ce=bl_tot / max(1, nsteps),
            retention_kl=rk_tot / max(1, nsteps),
            grad_norm=gn,
            steps=nsteps,
            rollout_s=round(t_roll),
            total_s=round(time.time() - t0),
        )
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        log("[rl]", json.dumps(row))
        if it % a.eval_every == 0 or it == a.iters:
            sc = evaluate_select(pol, eval_rng, a.select_n)
            evals.append((it, sc))
            log(f"[eval] iter {it} greedy selection-seed score {sc:.2f}")
            if BK.select_iteration(evals) == it:
                pol.save(str(out / "model_best"))
                log(f"[save] selected checkpoint: iteration {it}")
            (out / "sel_evals.json").write_text(
                json.dumps({"evals": evals, "selected_iter": BK.select_iteration(evals)})
            )
    pol.save(str(out / "model_last"))
    sel = BK.select_iteration(evals)
    (out / "run.json").write_text(
        json.dumps(
            dict(
                arm=a.arm,
                seed=a.seed,
                weights=[w_act, w_bel, w_ret],
                iters=a.iters,
                selected_iter=sel,
                selection_evals=evals,
                train_seconds=round(time.time() - t_start),
                episodes=counter,
                args=vars(a),
            ),
            indent=1,
        )
    )
    log(f"[done] selected iteration {sel}, {time.time() - t_start:.0f}s")


# ---------------------------------------------------------------------------------------------
# B4: supervised belief fit on a B1 run's own rollouts
# ---------------------------------------------------------------------------------------------


def fit_beliefs(a: Any) -> None:
    """Fit the belief answers of a B1 run's final policy by supervised learning on the belief questions in that run's
    own training rollouts (``beliefs.jsonl``): log score only, no actor and no retention term. The result is used for
    belief answers; B4's *policy* is B1's (PREREGISTRATION.md H7 rule)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = _logger(out)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    data = [json.loads(x) for x in (Path(a.b1_run) / "beliefs.jsonl").read_text().splitlines()]
    log(f"[b4] {len(data)} resolved belief questions from {a.b1_run}")
    pol = Policy(str(Path(a.b1_run) / "model_last"), False)
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, betas=(0.9, 0.95))
    t0 = time.time()
    for step in range(1, a.steps + 1):
        pol.m.train()
        pick = rng.sample(data, min(a.belief_batch, len(data)))
        items = belief_items(pol, pick)
        gold = [b["gold"] for b in pick]
        tot = 0.0
        for i in range(0, len(items), a.batch):
            lg = pol.logits(items[i : i + a.batch]).float()[:, :2]
            ce = F.cross_entropy(lg, torch.tensor(gold[i : i + a.batch], device="cuda"))
            (ce * len(gold[i : i + a.batch]) / len(items)).backward()
            tot += ce.item() * len(gold[i : i + a.batch]) / len(items)
        gn = float(torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == 1:
            log(
                f"[b4] step {step}/{a.steps} belief_ce {tot:.4f} gn {gn:.2f} {time.time() - t0:.0f}s"
            )
    pol.save(str(out / "model_last"))
    (out / "run.json").write_text(
        json.dumps(
            dict(arm="b4", seed=a.seed, b1_run=str(a.b1_run), steps=a.steps, data=len(data)),
            indent=1,
        )
    )


# ---------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------


def _belief_arrays(pol: Policy, beliefs: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return dict(
        logit_diff=belief_logit_diff(pol, beliefs),
        gold=np.array([b["gold"] for b in beliefs], dtype=np.int8),
        kind=np.array([b["kind"] for b in beliefs]),
        seed=np.array([b["seed"] for b in beliefs], dtype=np.int32),
        t=np.array([b["t"] for b in beliefs], dtype=np.int32),
    )


def _dump_records(path: Path, beliefs: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(b) + "\n" for b in beliefs))


def evaluate_checkpoint(a: Any) -> None:
    """Games and beliefs of one checkpoint: 50 greedy and 50 sampled games on the evaluation seeds (their sampled games
    also give the on-policy belief set), and 50 sampled games on the selection seeds whose beliefs fit the temperature
    (B2 / B3 + TS). Raw logit differences are saved (``beliefs_<name>.npz``) and the questions themselves
    (``onpolicy_*.jsonl``), so another model can answer the same questions and every metric is computed offline.

    ``--score name=path[,name=path]`` also answers the questions in those jsonl files (for example the common states,
    or a B1 run's selection-seed questions for B4's temperature)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = _logger(out)
    pol = Policy(a.ckpt, False)
    rng = random.Random(a.seed)
    res: dict[str, Any] = {"ckpt": a.ckpt, "seed": a.seed}
    if not a.no_games:
        torch.manual_seed(a.seed)
        n = a.n_seeds
        greedy = play(pol, list(BK.EVAL_SEEDS[:n]), rng, MAX_T, greedy=True, keep_items=False)
        res["greedy_scores"] = [t["score"] for t in greedy]
        torch.manual_seed(a.seed + 1)
        sampled = play(pol, list(BK.EVAL_SEEDS[:n]), rng, MAX_T, greedy=False, keep_items=False)
        res["sampled_scores"] = [t["score"] for t in sampled]
        log(
            f"[eval] greedy {np.mean(res['greedy_scores']):.2f} sampled {np.mean(res['sampled_scores']):.2f}"
        )
        bel = [b for t in sampled for b in t["beliefs"]]
        _dump_records(out / "onpolicy_eval.jsonl", bel)
        np.savez(out / "beliefs_onpolicy_eval.npz", **_belief_arrays(pol, bel))
        torch.manual_seed(a.seed + 2)
        sel = play(pol, list(BK.SELECT_SEEDS[:n]), rng, MAX_T, greedy=False, keep_items=False)
        res["sampled_scores_select"] = [t["score"] for t in sel]
        bel_sel = [b for t in sel for b in t["beliefs"]]
        _dump_records(out / "onpolicy_select.jsonl", bel_sel)
        np.savez(out / "beliefs_onpolicy_select.npz", **_belief_arrays(pol, bel_sel))
    for spec in filter(None, a.score.split(",")):
        name, path = spec.split("=", 1)
        recs = [json.loads(x) for x in Path(path).read_text().splitlines()]
        np.savez(out / f"beliefs_{name}.npz", **_belief_arrays(pol, recs))
        log(f"[eval] answered {len(recs)} questions from {path} as '{name}'")
    if (
        not a.no_games or not (out / "games.json").exists()
    ):  # a scoring-only call keeps the existing game results
        (out / "games.json").write_text(json.dumps(res, indent=1))


def collect_common_states(a: Any) -> None:
    """The common states (H7 rule (a)): sampled rollouts of one frozen policy on the evaluation seeds, with the belief
    questions asked and resolved exactly as in training. Every arm's beliefs are scored on these same questions."""
    pol = Policy(a.ckpt, False)
    torch.manual_seed(a.seed)
    trajs = play(
        pol, list(BK.EVAL_SEEDS), random.Random(a.seed), MAX_T, greedy=False, keep_items=False
    )
    recs = [b for t in trajs for b in t["beliefs"]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(
        f"[common] {len(recs)} resolved questions from {len(trajs)} episodes, mean score {mean_score(trajs):.2f}"
    )
