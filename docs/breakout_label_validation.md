# Exp B: validation of the Breakout belief labels (phase 2, 2026-10-06)

**Question.** Do the labels of `brick30`, `life30` and `return` mean what their questions say? The earlier "v2" fix
showed that a label error can manufacture apparent skill, so this is checked before any B arm is trained.

**Verdict.** The labels pass every check below. One caveat shapes how B is evaluated: a belief that points at the
end of a truncated episode never resolves and is dropped (see "What this does not show").

Script: `scripts/validate_breakout_labels.py` (re-runnable). Raw output: `results/expB/label_validation/report.json` and
`episodes_inspect.txt`.

## Setup

160 episodes of the Stage 1 SFT policy sampling its own actions, seeds 3000–3159 (disjoint from the selection seeds
1000–1049 and the evaluation seeds 2000–2049), 300 decisions each: 48,000 steps, mean final score 6.04. Belief
questions are asked every 4th step with a random kind among those that make sense, exactly as in the old
`rl_calibrated.py`, and resolved by a copy of that script's step-by-step labelling logic ("legacy").
**All 160 episodes were truncated at 300 steps; none ended in game over.**

## Checks and results

| check | result |
|---|---|
| **Legacy labeller against an independent re-derivation** from the recorded trajectory (positions, lives and score only, written separately: `independent_labels`) | **0 disagreements in 10,942 resolved questions** (`return` 3,586, `life30` 3,663, `brick30` 3,693) |
| **Base rates** (identical under both labellers) | `return` **0.778**, `life30` **0.221**, `brick30` **0.635** |
| **Physical check of `return`**: where does the ball turn from moving down to moving up without a life lost? | 1,121 turnings, all at ball y = 175–182 (median 175), a median **9 px** (90th percentile 14, maximum 20) from the paddle centre; **no** turning point at all between y = 100 and 150 |
| **Threshold sensitivity** (the legacy "bottom" threshold of y = 160) | the independent labels are **identical for thresholds 140, 150, 160 and 170** |
| **Hand inspection of 20 sampled episodes** (1,365 questions; timelines in `episodes_inspect.txt`) | **0 mismatches**; I read episode 3098 line by line against the question definitions (for example, `brick30` asked at t = 0 is False because the first brick falls at t = 32, outside 30 moves; `return` asked at t = 140 and 152 is False because a life is lost at t = 155 before the next return; `return` asked at t = 156 is True because the return is at t = 173) and found every label correct |

## Log-score skill against the base rate

`skill = 1 − nll_model / nll_base`, the base rate being leave-one-episode-out, on the SFT policy's states:

| model | `return` | `life30` | `brick30` |
|---|---|---|---|
| Stage 1 SFT (the B1 start) | **−2.99** | **−1.68** | **−0.83** |
| old RLCD-style v2, final checkpoint | **+0.066** | **+0.052** | **+0.556** |

The SFT model's beliefs are much *worse* than saying the base rate, so any belief skill in the B arms has to be
earned by training. The old v2 model has real but small skill on `return` and `life30` and large skill on `brick30`.
These are off-policy states for v2 (it was trained on its own rollouts), so they are a floor on what a matched
evaluation would show.

## What this does not show

* **Both labellers use the same definitions and the same RAM readout** (ball and paddle position, lives, score).
  Disagreement would have caught an implementation error; it cannot catch a wrong *definition* or a wrong RAM
  address. The physical check (returns happen at the bottom, over the paddle; score steps are single bricks) is
  the independent evidence there. No frame-level check was made.
* **Truncation drops questions.** About 1,060 of the 12,000 questions asked (about 9%) were pointed at the last
  30 steps of an episode and never resolved, so the resolved set under-represents the very end of an episode.
  This affects which states are asked about, not the labels of those that are.
* One reference policy (the SFT policy) was used. Other policies visit other states, but the labels are functions of
  the trajectory, not of the policy.
* In the B protocol (D16) belief quality is measured on rollouts of B1 seed 0, so the same checks should be re-run on
  those rollouts before B3 results are read.
