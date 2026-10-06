"""Exp B label validation: are the Breakout belief labels what their questions say? (phase 2)

Run on a GPU in decider's venv:
    PYTHONPATH=$REPO:$DECIDER $PY scripts/validate_breakout_labels.py --init $PROJ/models/decider9b-sft \
        --episodes 160 --out $PROJ/rlcd_expA/breakout_labels \
        --score_models sft=$PROJ/models/decider9b-sft,v2_last=$PROJ/hardgen/runs/rl_breakout_cal_v2/model_last

What it does, in order:
1. Plays sampled-policy episodes (seeds 3000+, disjoint from the selection 1000-1049 and evaluation 2000-2049 seeds)
   and records the game state at every step. Belief questions are asked every 4th step exactly as in the old
   ``rl_calibrated.py`` and resolved by a copy of its step-by-step labelling logic ("legacy").
2. Re-derives every label from the recorded trajectory with an independent labeller (``independent_labels``), then
   compares the two. Disagreements are listed.
3. Physical checks that need neither labeller: where the ball turns around (a paddle return must happen at the
   bottom, horizontally over the paddle) and that no turning point sits in the ambiguous band of the 160 threshold.
4. Writes a readable timeline of 20 sampled episodes (events and belief questions with their labels) for inspection.
5. Scores the belief questions with each model in ``--score_models`` and reports the log-score skill against
   predicting the base rate: ``1 - nll_model / nll_base``, the base rate being leave-one-episode-out.

Legacy labelling is copied (with small edits) from ``Louiskq/jev-class-9b`` ``hardgen/rl_calibrated.py`` so the
check is of the labels that the earlier results used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np  # noqa: E402
import torch  # noqa: E402

BELIEFS = {
    "return": "Will the paddle return the ball the next time the ball reaches the bottom?",
    "brick30": "Will at least one brick be broken within the next 30 moves?",
    "life30": "Will a life be lost within the next 30 moves?",
}
PADDLE_Y = 160  # legacy: a down -> up flip at or below this y is a paddle hit


# --------------------------------------------------------------------------------------------- rollouts
def rollout(pol, n, seed0, max_t, rng, belief_every=4):
    """Sampled-policy episodes with per-step state and the legacy belief labels."""
    from decider.games import envs as G

    envs = []
    for j in range(n):
        g = G.GAMES["breakout"]()
        g.reset(seed=seed0 + j)
        envs.append(
            dict(
                g=g,
                done=False,
                terminal=False,
                k=0,
                opt=g.options[0],
                last=g.score(),
                pend=[],
                beliefs=[],
                prev=g._s(),
                vy=0,
                traj=[dict(g._s(), score=g.score())],
                acts=[],
                seed=seed0 + j,
            )
        )
    pol.m.eval()
    while any(not e["done"] for e in envs):
        due = [e for e in envs if not e["done"]]
        items = []
        for e in due:
            ex, fr = pol.example(e["g"])
            items.append(pol.make_item(ex, fr))
            if e["k"] % belief_every == 0:
                s = e["g"]._s()
                kinds = ["brick30", "life30"] + (["return"] if s["ball_y"] > 0 else [])
                e["pend"].append(
                    dict(
                        kind=rng.choice(kinds),
                        t=e["k"],
                        ctx=f"{e['g'].intro}\n\nSituation: {e['g'].text()}",
                        score=e["g"].score(),
                        lives=s["lives"],
                    )
                )
        with torch.no_grad():
            lg = pol.logits(items)
        for e, row in zip(due, lg, strict=True):
            p = torch.softmax(row[: len(e["g"].options)].float(), -1).cpu()
            a = int(torch.multinomial(p, 1))
            e["acts"].append(a)
            e["opt"] = e["g"].options[a]
        for e in due:
            _, done = e["g"].step(e["opt"])
            e["k"] += 1
            sc = e["g"].score()
            s = e["g"]._s()
            vy = s["ball_y"] - e["prev"]["ball_y"]
            # --- legacy labelling, step by step (copied from rl_calibrated.rollout) ---
            hit = (
                e["vy"] > 0
                and vy < 0
                and e["prev"]["ball_y"] >= PADDLE_Y
                and s["ball_y"] > 0
                and s["lives"] == e["prev"]["lives"]
            )
            still = []
            for b in e["pend"]:
                if b["kind"] == "return":
                    out = True if hit else False if s["lives"] < b["lives"] or done else None
                elif b["kind"] == "brick30":
                    out = (
                        True
                        if sc > b["score"]
                        else False
                        if (e["k"] - b["t"] >= 30 or done)
                        else None
                    )
                else:
                    out = (
                        True
                        if s["lives"] < b["lives"]
                        else False
                        if (e["k"] - b["t"] >= 30 or (done and s["lives"] >= b["lives"]))
                        else None
                    )
                if out is None:
                    still.append(b)
                else:
                    e["beliefs"].append(
                        dict(kind=b["kind"], t=b["t"], ctx=b["ctx"], legacy=int(out))
                    )
            e["pend"] = still
            e["prev"], e["vy"] = s, (vy if vy != 0 else e["vy"])
            e["traj"].append(dict(s, score=sc))
            if done or e["k"] >= max_t:
                e["done"] = True
                e["terminal"] = bool(
                    done
                )  # False: truncated at max_t (unresolved beliefs are dropped)
    for e in envs:
        e["g"].close()
    return envs


# ---------------------------------------------------------------------------------- independent labeller
def turning_points(traj, bottom=150, over=20):
    """Indices j where the ball turns from moving down to moving up at the bottom without a life being lost
    and horizontally over the paddle: a paddle return, found from positions alone."""
    out = []
    for j in range(2, len(traj)):
        a, b, c = traj[j - 2], traj[j - 1], traj[j]
        if (
            b["ball_y"] >= a["ball_y"]
            and b["ball_y"] > c["ball_y"]
            and b["ball_y"] >= bottom
            and c["ball_y"] > 0
            and c["lives"] == b["lives"]
            and abs(b["ball_x"] - b["paddle_x"]) <= over
        ):
            out.append(j)
    return out


def independent_labels(e, bottom=150, over=20):
    """Re-derive the label of every belief question of episode ``e`` from its recorded trajectory.

    brick30 / life30: the event happens within the next 30 moves; otherwise False if 30 moves (or the episode)
    elapse, and dropped when the episode was truncated first. return: the first thing that happens next is a
    paddle return (True) or a life loss / the end of the episode (False).
    """
    traj, T = e["traj"], len(e["traj"]) - 1
    hits = set(turning_points(traj, bottom, over))
    out = []
    for b in e["beliefs"]:
        t, kind, lab = b["t"], b["kind"], None
        s0 = traj[t]
        if kind in ("brick30", "life30"):
            happened = any(
                (traj[j]["score"] > s0["score"])
                if kind == "brick30"
                else (traj[j]["lives"] < s0["lives"])
                for j in range(t + 1, min(t + 30, T) + 1)
            )
            lab = True if happened else False if (t + 30 <= T or e["terminal"]) else None
        else:
            for j in range(t + 1, T + 1):
                if traj[j]["lives"] < s0["lives"]:
                    lab = False
                    break
                if j in hits:
                    lab = True
                    break
            if lab is None and e["terminal"]:
                lab = False
        out.append(None if lab is None else int(lab))
    return out


# -------------------------------------------------------------------------------------------- reporting
def summarise_events(e):
    """Events of an episode, from positions and score alone."""
    traj, ev = e["traj"], []
    for j in range(1, len(traj)):
        if traj[j]["score"] > traj[j - 1]["score"]:
            ev.append((j, f"BRICK score {traj[j - 1]['score']:.0f}->{traj[j]['score']:.0f}"))
        if traj[j]["lives"] < traj[j - 1]["lives"]:
            ev.append((j, f"LIFE LOST lives {traj[j - 1]['lives']}->{traj[j]['lives']}"))
    for j in turning_points(traj):
        b = traj[j - 1]
        ev.append((j, f"RETURN ball ({b['ball_x']},{b['ball_y']}) paddle x={b['paddle_x']}"))
    return sorted(ev)


def dump_episodes(envs, labels, path, n=20, seed=0):
    pick = random.Random(seed).sample(range(len(envs)), min(n, len(envs)))
    with open(path, "w") as f:
        for i in pick:
            e = envs[i]
            f.write(
                f"=== episode seed {e['seed']}: {len(e['traj']) - 1} steps, final score {e['traj'][-1]['score']:.0f}, "
                f"{'terminal' if e['terminal'] else 'truncated'}\n"
            )
            rows = [(t, txt) for t, txt in summarise_events(e)]
            for b, ind in zip(e["beliefs"], labels[i], strict=True):
                rows.append(
                    (b["t"], f"ASKED {b['kind']:8s} legacy={b['legacy']} independent={ind}")
                )
            for t, txt in sorted(rows, key=lambda r: (r[0], r[1].startswith("ASKED"))):
                f.write(f"  t={t:3d}  {txt}\n")


def score_beliefs(model_path, envs, batch_size=32):
    """P(yes) of every belief question under one model (a yes/no question, one forward pass)."""
    from decider import data as D
    from decider.games.rl import NoShuffle
    from decider.model import DecisionModel, collate
    from decider.prompt import build

    m = DecisionModel(model_path, grad_ckpt=False).cuda().eval()
    flat = [(i, b) for i, e in enumerate(envs) for b in e["beliefs"]]
    items = [
        build(
            D.Example(b["ctx"], [D.Q(BELIEFS[b["kind"]], ["no", "yes"], b["legacy"])], "belief"),
            m.tok,
            NoShuffle(),
            max_ctx_tokens=1024,
        )
        for _, b in flat
    ]
    p_yes = []
    with torch.no_grad():
        for s in range(0, len(items), batch_size):
            bt = collate(items[s : s + batch_size], m.tok.pad_token_id)
            lg = m.slot_logits(
                bt["input_ids"].cuda(),
                bt["attention_mask"].cuda(),
                bt["slot_idx"].cuda(),
                bt["slot_batch"].cuda(),
                bt["nopts"].cuda(),
            )
            p_yes += torch.softmax(lg[:, :2].float(), -1)[:, 1].cpu().tolist()
    del m
    torch.cuda.empty_cache()
    return [(i, b["kind"], b["legacy"], p) for (i, b), p in zip(flat, p_yes, strict=True)]


def skill_table(rows):
    """Per kind: n, base rate, model nll, base-rate nll (leave-one-episode-out) and skill = 1 - nll / nll_base."""
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r[1]].append(r)
    out = {}
    for kind, rs in by_kind.items():
        y = np.array([r[2] for r in rs], float)
        p = np.clip(np.array([r[3] for r in rs]), 1e-6, 1 - 1e-6)
        ep = np.array([r[0] for r in rs])
        nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        base_nll = []
        for k in range(len(rs)):
            others = ep != ep[k]
            q = float(np.clip(y[others].mean(), 1e-6, 1 - 1e-6)) if others.any() else 0.5
            base_nll.append(-(y[k] * math.log(q) + (1 - y[k]) * math.log(1 - q)))
        base = float(np.mean(base_nll))
        out[kind] = {
            "n": len(rs),
            "base_rate": float(y.mean()),
            "nll": nll,
            "nll_base_rate": base,
            "skill": 1 - nll / base,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--episodes", type=int, default=160)
    ap.add_argument("--max_t", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--score_models", default="", help="comma list of name=path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    torch.manual_seed(0)
    rng = random.Random(0)

    from decider.games import envs as G
    from decider.games.rl import Policy

    G.RENDER = False
    pol = Policy(a.init, False)
    envs = []
    for s in range(0, a.episodes, a.envs):
        envs += rollout(pol, min(a.envs, a.episodes - s), a.seed0 + s, a.max_t, rng)
        print(f"[rollout] {len(envs)}/{a.episodes} episodes", flush=True)
    del pol
    torch.cuda.empty_cache()

    labels = [independent_labels(e) for e in envs]
    report: dict = {
        "episodes": len(envs),
        "steps": int(sum(len(e["traj"]) - 1 for e in envs)),
        "mean_final_score": float(np.mean([e["traj"][-1]["score"] for e in envs])),
        "terminal_episodes": int(sum(e["terminal"] for e in envs)),
    }
    kinds = Counter(b["kind"] for e in envs for b in e["beliefs"])
    report["beliefs_asked_and_resolved"] = dict(kinds)
    # --- legacy vs independent
    agree, dis = Counter(), []
    dropped = Counter()
    for e, lab in zip(envs, labels, strict=True):
        for b, ind in zip(e["beliefs"], lab, strict=True):
            if ind is None:
                dropped[b["kind"]] += 1
                continue
            agree[(b["kind"], b["legacy"] == ind)] += 1
            if b["legacy"] != ind:
                dis.append(
                    dict(
                        seed=e["seed"],
                        t=b["t"],
                        kind=b["kind"],
                        legacy=b["legacy"],
                        independent=ind,
                    )
                )
    report["agreement"] = {
        k: {"agree": agree[(k, True)], "disagree": agree[(k, False)]} for k in BELIEFS
    }
    report["unresolvable_by_independent_labeller"] = dict(dropped)
    report["disagreements"] = dis[:40]
    # --- base rates from the legacy labels (what training used) and from the independent ones
    for name in ("legacy", "independent"):
        rates = defaultdict(list)
        for e, lab in zip(envs, labels, strict=True):
            for b, ind in zip(e["beliefs"], lab, strict=True):
                v = b["legacy"] if name == "legacy" else ind
                if v is not None:
                    rates[b["kind"]].append(v)
        report[f"base_rate_{name}"] = {
            k: {"rate": float(np.mean(v)), "n": len(v)} for k, v in rates.items()
        }
    # --- physical checks on paddle returns and the threshold
    dx, y_hit, ambiguous = [], [], 0
    for e in envs:
        for j in turning_points(e["traj"], bottom=100, over=1000):
            b = e["traj"][j - 1]
            if b["ball_y"] >= 150:
                dx.append(abs(b["ball_x"] - b["paddle_x"]))
                y_hit.append(b["ball_y"])
            elif b["ball_y"] > 100:
                ambiguous += 1
    report["returns"] = {
        "n": len(dx),
        "ball_y_at_return_min_median_max": [int(min(y_hit)), int(np.median(y_hit)), int(max(y_hit))]
        if y_hit
        else None,
        "abs_dx_to_paddle_median_p90_max": [
            float(np.median(dx)),
            float(np.percentile(dx, 90)),
            float(max(dx)),
        ]
        if dx
        else None,
        "turning_points_between_y100_and_150": ambiguous,
    }
    thr = {}
    for bottom in (140, 150, 160, 170):
        vals = [x for e in envs for x in independent_labels(e, bottom=bottom) if x is not None]
        thr[bottom] = float(np.mean(vals)) if vals else None
    report["mean_independent_label_by_threshold(all kinds)"] = thr
    dump_episodes(envs, labels, out / "episodes_inspect.txt")
    # --- model beliefs and skill against the base rate
    skill = {}
    for spec in [s for s in a.score_models.split(",") if s]:
        name, path = spec.split("=")
        rows = score_beliefs(path, envs)
        skill[name] = skill_table(rows)
        print(
            f"[skill] {name}: "
            + json.dumps({k: round(v["skill"], 3) for k, v in skill[name].items()}),
            flush=True,
        )
    report["model_skill_vs_base_rate"] = skill
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}, indent=1))
    print(
        f"[done] {len(dis)} disagreements; timeline of 20 episodes in {out / 'episodes_inspect.txt'}"
    )


if __name__ == "__main__":
    main()
