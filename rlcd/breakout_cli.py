"""Command line for Exp B (run in decider's venv on a GPU; ``PYTHONPATH`` must hold this repo and decider).

python -m rlcd.breakout_cli train --arm b3 --seed 0 --out RUN --init MODEL
python -m rlcd.breakout_cli eval --ckpt RUN/model_best --out RUN/eval_best [--score common=common_states.jsonl]
python -m rlcd.breakout_cli collect-common --ckpt B1_RUN/model_last --out common_states.jsonl
python -m rlcd.breakout_cli fit-beliefs --b1_run B1_RUN --out B4_RUN
"""

from __future__ import annotations

import argparse

from rlcd import breakout as BK


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="B1 / B3 / B5")
    t.add_argument("--arm", choices=["b1", "b3", "b5"], required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    t.add_argument("--init", default="/projects/u6xe/louisk/models/decider9b-sft")
    t.add_argument("--replay", default="/projects/u6xe/louisk/decider/data/mixture_full.pkl")
    t.add_argument("--iters", type=int, default=40)
    t.add_argument("--envs", type=int, default=16)
    t.add_argument("--max_t", type=int, default=300)
    t.add_argument("--lr", type=float, default=4e-6)
    t.add_argument("--gamma", type=float, default=BK.GAMMA)
    t.add_argument("--entropy", type=float, default=0.01)
    t.add_argument("--ppo_steps", type=int, default=4)
    t.add_argument("--clip", type=float, default=0.2)
    t.add_argument("--kl_stop", type=float, default=0.05)
    t.add_argument("--update_samples", type=int, default=6000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--eval_every", type=int, default=5)
    t.add_argument("--belief_w", type=float, default=None, help="override the arm's belief weight")
    t.add_argument("--ret_w", type=float, default=None, help="override the arm's retention weight")
    t.add_argument("--belief_batch", type=int, default=128)
    t.add_argument("--ret_rows", type=int, default=1000)
    t.add_argument("--ret_batch", type=int, default=32)
    t.add_argument(
        "--select_n", type=int, default=50, help="selection seeds per evaluation (smoke: 4)"
    )

    e = sub.add_parser("eval", help="games and beliefs of one checkpoint")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument(
        "--score", default="", help="name=path.jsonl[,name=path.jsonl]: also answer these questions"
    )
    e.add_argument("--no_games", action="store_true")
    e.add_argument("--n_seeds", type=int, default=50, help="games per set (smoke: 4)")

    c = sub.add_parser("collect-common", help="the common states of the H7 rule")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, default=0)

    f = sub.add_parser("fit-beliefs", help="B4: supervised belief fit on a B1 run's own rollouts")
    f.add_argument("--b1_run", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--steps", type=int, default=160, help="4 PPO steps x 40 iterations of B3")
    f.add_argument("--lr", type=float, default=4e-6)
    f.add_argument("--belief_batch", type=int, default=128)
    f.add_argument("--batch", type=int, default=32)

    a = ap.parse_args()
    from rlcd import breakout_gpu as GPU  # imports torch and decider: only on a GPU node

    {
        "train": GPU.train,
        "eval": GPU.evaluate_checkpoint,
        "collect-common": GPU.collect_common_states,
        "fit-beliefs": GPU.fit_beliefs,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
