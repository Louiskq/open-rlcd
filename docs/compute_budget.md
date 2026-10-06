# Compute budget (Isambard-AI, one GH200 unless stated)

Written 2026-10-06 for the phase 2 go/no-go. Every rate below is **measured** from a completed Slurm job of the
old project (job ids given, `sacct` and the run logs on the cluster) unless it is marked *assumed*. A
fresh probe (`slurm/throughput_probe.sh`, job 7114048) confirms today's training and scoring rates; its status
is recorded at the end of this file. These are budget estimates, not results; the README does not quote them.

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

## Estimate by phase

| phase | GPU-hours | what dominates |
|---|---|---|
| **2. smoke run** | **about 12**, of which about 7 are teacher generation that phase 3 reuses | see below |
| **3. Exp A in full** | **about 290** (range 200–330) | teacher candidate pools (about 136), A6/A6r training (about 75), the 30 supervised/RL runs (about 56) |
| **4. Exp B in full** | **about 38** | 15 Breakout runs at about 1.9 each |
| total | about 340 GPU-hours (about 14 GPU-days) | |

### Phase 2 in detail

- 9 training runs at 10% of the steps (A1 ×2 seeds for δ, A3, A4 ×2 clips, A5, A6, A6r, A7): about 15 min each, **2.3**.
- Evaluation of 10 models (adds A0 and A0 as served): **1.2**. Per-answer-type gradient-norm share: **0.3**.
- A6 / A6r smoke: 2 rounds, N = 500, a candidate pool of 2,000 per round, so 4,000 generated questions at
  570/GPU-h is **about 7 (reused in phase 3: it is the start of the real round-1 and round-2 pools)**; fine-tuning
  and scoring about **1.2**.
- Breakout label validation (20 sampled episodes per belief, base rates, log-score skill): **0.3**.
- If the two A1 seeds for δ are run at full length instead of 10% (a recommendation, see `docs/decisions.md` D5):
  **+3.4**.

### Phase 3 in detail

- **Candidate pools: 4 rounds × 4N = 80k questions at N = 5k, about 136 GPU-h.** This is independent of any model,
  so it can start as soon as N is fixed and run on up to 4 GPUs in parallel (about 34 h of wall time). It only
  stays at this size if **one pool per round is shared by A6, A6r and all seeds**; generating per run would
  multiply it by 6 (about 800 GPU-h) and is not feasible.
- 6 arms (A1, A3, A4 ∞, A4 clip 10, A5, A7) × (3 learning rates + 2 more seeds) = 30 runs at about 1.87 each (the
  old 654-step setting): **56**; the exploratory A3 with β = 0: **2**.
- A6 and A6r: 2 arms × 5 runs, each 4 fine-tune rounds. If every round trains for the shared step count that is
  about 7.5 per run, **75**; if rounds 1–3 use a quarter of the steps it is about 3.3 per run, **33**.
- Scoring candidates at the dev-eval rate (at least 17 questions/s, so at most about 20 min per 20k pool): 40 round-runs, **12**.
- Evaluation of about 45 models: **5**.

### Assumptions that change the numbers (each is a decision for the maintainer)

1. **Pools are shared** across A6, A6r and seeds (above). Without this the plan is out of budget.
2. **Labels come with generation** (the pipeline's writer supplies the intended answer and three checks verify it).
   If "label the selected items" means a separate teacher pass, add roughly 20–50 GPU-h.
3. **N = 5k.** N = 2.5k roughly halves the generation and A6 training cost (saving about 70–90 GPU-h).
4. **Steps per arm** follow the old setting (2 epochs, 654 steps); the spec leaves this open.
5. **The candidate pools use authored questions**; seeded (templated) families were cheaper per question in the
   old jobs but their yield was not logged, so they are not credited here.
6. Queueing is not included. On 2026-10-06 the cluster had about 965 nodes allocated, and probe job 7114048 sat
   pending on priority for more than 30 minutes; wall-clock time for phases 3–4 will be set by the queue, not by the
   GPU-hours above.

## Probe job 7114048

Status when this file was written: **pending (Priority)**. When it runs, its log will give today's tokens/s and
memory for the 40-step training probe and the wall time of forward-only scoring of the dev set at batch sizes 16
and 48. If any of those differ from the measured rates above by more than about 25%, the table above is
re-derived and this file is updated with a dated note.
