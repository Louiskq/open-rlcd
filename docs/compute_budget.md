# Compute budget (Isambard-AI, one GH200 unless stated)

Written 2026-10-06 for the phase 2 go/no-go; revised the same day for decisions D9–D13 and again for the lean plan (D-lean). Every rate below is **measured** from a completed Slurm job of the
old project (job ids given, `sacct` and the run logs on the cluster) unless it is marked *assumed*. A
probe (`slurm/throughput_probe.sh`, job 7114048) confirmed today's training and scoring rates; its result is at
the end of this file. These are budget estimates, not results; the README does not quote them.

## Measured rates

| quantity | measured | source |
|---|---|---|
| LoRA training, r = 64, on the SFT checkpoint | about 6.2–6.5k tokens/s, 30 GB peak memory; a 2-epoch run on the 22.9k-question pool (19.1M tokens/epoch, 654 steps) took **1 h 52 min** including about 11 min of load, tokenisation, evals and save | job 7099024 (`arm_scaleAll`) |
| Dev-set evaluation, forward only | **4,780 questions in 285 s** (three dev sets, graph engine, batch 16), including three model loads: **at least 17 questions/s**, about 5 min per model | jobs 7099003–7099033 |
| Teacher generation, authored questions (Qwen3.8-27B, vLLM, writer plus three checks) | **about 570 kept questions per GPU-hour** (16,084 kept questions in 28.1 job-hours over nine jobs; the latest two: 1,389 in 2.48 h and 1,402 in 2.49 h) | jobs 6961774/5, 6961982, 7000341–3, 7099000/1 |
| Breakout, calibrated RL (B3-like) | **156 s per iteration** (rollout 30 s, update about 125 s), 40 iterations: about 1.9 GPU-h per run with evals | `runs/rl_breakout_cal_v2/rl.log` |
| Breakout, plain PPO (B1-like) | 1 h 35 min for 40 iterations | job 7000347 |
| Account usage so far | 227 GPU-hours since 2026-09-01 (`sreport`); no limit is visible from the user side, so **check the allocation on the Brics portal** | `sreport` |

**Pool composition** (raw teacher records, `arm_scale5k.jsonl` and `arm_scale2k.jsonl`): CHOICE about 52%, NOUL
about 35%, SCORE about 12.6% of questions; SCORE questions have 2–7 levels (mean about 4). Under the isolated-level
readout (`docs/decisions.md` D3) every level is its own forward pass over the full context, so SCORE is about 12–13%
of the questions but roughly **40% of the forward-pass rows**. That is a cost factor, and it also bears on D4: after
RPS is divided by K − 1, SCORE carries less gradient per question than the log score does.

## Measured in phase 2 (2026-10-06, Exp A on `decider9b-sft`)

| quantity | measured | source |
|---|---|---|
| **One full-length A run** (S = 734 steps, LoRA r = 64, 2 epochs of half the 90% training split) | training **6,565–6,605 s (1.83 h)**; whole job **1 h 55 min 03 s – 1 h 55 min 50 s (1.93 GPU-h)** including load, in-training gradient-share measurement and the calibration, selection and dev predictions | jobs 7125954/5/6 (`sacct`) |
| Training speed | 6.5k tokens/s, about 9 s per optimizer step, 35 GB peak (the 654-step job of the old project: 30 GB) | the three sweep runs' logs |
| Smoke run (73 steps) | about 0.35 GPU-h per arm, as estimated | `docs/phase2_smoke.md` |
| Queue wait | the dev-session allocation 7120786 (4 h 38 min used) hosted the smoke runs; the three sweep jobs started within the hour | `squeue`, `sacct` |

A full-length run is therefore **1.93 GPU-h**, 3% above the 1.87 used below. The estimate below is rescaled by that
factor and JevBench (deferred to phase 3; it needs a merged checkpoint served) is added.

**Revised estimate (2026-10-06, after the sweep): phase 2 about 13 (actual to date, including seed 1 of A1 in the
queue), phase 3 about 42 (including about 5 for JevBench), phase 4 about 16, total about 71 GPU-hours.** The ±30% band
and the assumptions below are unchanged.

## Estimate by phase (lean plan, D-lean, 2026-10-06)

| phase | GPU-hours | what dominates |
|---|---|---|
| **2. smoke run** | **about 11**, of which 7.5 are the four A1 runs that phase 3 needs anyway | A1's full-length sweep and δ seed |
| **3. Exp A in full** | **about 38** (beyond phase 2) | A4 (5 runs), A6 and A6r (6 runs), A3 (3 runs) |
| **4. Exp B in full** | **about 16** | 7 Breakout runs at 1.6–1.9 each |
| total | **about 65 GPU-hours** (about 2.7 GPU-days) | |

The earlier plan (generated candidate pools, 7 arms with clipping and A7, B6) was about **275** GPU-hours; its
single biggest item, about 140 GPU-hours of teacher generation, is gone because A6 and A6r now mine from a
held-back half of the existing pool.

Basis: one full-length A run is **1.87 GPU-h** (job 7099024, 654 steps); an A6 or A6r run is that plus about 0.1 per
extra segment for four segments of S/4 steps (**2.17**); a smoke run at 10% of the steps is **0.35**; evaluating a
final model on the dev sets, calibration and selection splits is **0.12**; Breakout runs are **1.6** (B1) and
**1.9** (B3). Candidate scoring for A6 is at no less than 17 questions/s.

### Phase 2 (about 11.2)

| GPU-h | item |
|---|---|
| 7.48 | **A1: three-value sweep (seed 0) plus the δ seed 1 at the chosen rate, all at full length** (D13); these four runs count as A1's phase 3 runs |
| 1.05 | A3, A4, A5 smoke at 10% of the steps, 1 seed |
| 1.14 | A6 and A6r smoke (four segments of a tenth of S/4) |
| 0.05 | A6 candidate scoring in the smoke run (a tenth of half B) |
| 0.72 | evaluation of A0 (T = 1, as served, +TS) and 5 smoke models |
| 0.12 | selection-split evaluation of the four A1 runs (their evaluation-split predictions are deferred) |
| 0.30 | per-answer-type gradient-norm share (D4) |
| 0.30 | Breakout label validation |

### Phase 3, beyond phase 2 (about 37.6)

| GPU-h | item |
|---|---|
| 1.87 | A1 seed 2 |
| 9.35 | A4: the shared A3/A4 sweep (3 rates) plus 2 more seeds, 5 runs |
| 5.61 | A3: 3 seeds at the shared rate |
| 1.87 | A5: 1 seed |
| 6.51 | A6: 3 seeds |
| 6.51 | A6r: 3 seeds |
| 1.58 | A6 candidate scoring: half B is about 10.3k; with N = 1,500 per round A6 scores about 32k questions per run |
| 1.87 | A3 with β = 0 (exploratory) |
| 2.42 | evaluation of 17 final models, the sweep selection splits and the A0 reference rows |

### Phase 4 (about 15.8)

| GPU-h | item |
|---|---|
| 4.80 | B1, 3 seeds |
| 5.70 | B3, 3 seeds |
| 1.80 | B5, 1 seed |
| 0.98 | final evaluation of 14 checkpoints (7 runs, selected and final) on 100 games each |
| 0.70 | common-state belief evaluation (D16): one reference rollout plus about ten model scorings |
| 0.30 | B2: temperature fits on held-out rollouts |
| 1.50 | B4: rollouts, supervised belief fit and temperature scaling, 3 seeds |

### Assumptions that change the numbers

1. **Step count.** Every training line scales with the shared step count S; the table assumes the old setting
   (2 epochs, 654 steps). The spec leaves S open. At twice that, the total is roughly 110.
2. **N per round for A6 / A6r** barely changes cost (steps are fixed) but changes what the comparison can show
   (D-lean). Scoring cost scales with half B and the number of rounds.
3. **The A3/A4 sweep is run on A4 only.** Sweeping A3 and A4 separately adds about 5.6.
4. **The δ runs count as A1 phase 3 seeds** only if they use the rate the A1 sweep selects (D13).
5. **Evaluation is counted conservatively**: the 1.87 per run already includes about 0.13 of in-training
   evaluation, so a little is counted twice.
6. **The range is about ±30%** (roughly 45 to 85), before queueing. On 2026-10-06 the cluster had about 965 nodes
   allocated and probe job 7114048 had waited for hours; wall-clock time will be set by the queue.

Teacher-generation throughput (about 570 kept questions per GPU-hour, above) is kept in the measured-rates table
for the record; the lean plan does not use it.

## Probe job 7114048 (ran 2026-10-06, node nid011115, 14 min 9 s)

It reproduced the measured rates, so the estimate above is **unchanged** (about 65 GPU-hours).

| quantity | probe | old measured | verdict |
|---|---|---|---|
| LoRA training, same settings as `arm_scaleAll` | 6.3k tokens/s at step 20, 6.6k at step 40; about 9.9 s per optimizer step; 30 GB peak | 6.2–6.45k tokens/s, 30 GB | within about 3% |
| model load, tokenise 19.1M tokens, 40 steps | 452 s in all (tokenising 30 s) | tokenising 29 s | as before |
| forward-only scoring of the 2,791-question dev set, batch 16 | **164 s including model load: 17.0 questions/s** | at least 17 questions/s | matches the figure the A6 scoring cost uses |
| the same at batch 48 | **221 s (12.6 questions/s): slower** | not measured | keep batch 16 |

The probe removed its own output directory. **Queue wait is the real constraint:** this single-GPU, 35-minute job
was submitted at about 00:20 UTC and started at 06:10 UTC, so it waited about 5 h 50 min. Plan phases 2–4 as a few
longer or batched jobs (for example job arrays, or one allocation running several arms in sequence) so that each
arm does not pay its own wait.
