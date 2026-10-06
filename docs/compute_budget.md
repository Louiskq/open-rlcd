# Compute budget (Isambard-AI, one GH200 unless stated)

Written 2026-10-06 for the phase 2 go/no-go; revised the same day for decisions D9–D13. Every rate below is **measured** from a completed Slurm job of the
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

## Estimate by phase (revised 2026-10-06 for decisions D9–D13)

| phase | GPU-hours | what dominates |
|---|---|---|
| **2. smoke run** | **about 18**, of which about 7.5 are A1 runs phase 3 needs anyway and about 7 are teacher generation that phase 3 reuses | see below |
| **3. Exp A in full** | **about 220** (range 190–250) | teacher candidate pools (about 133), then the 25 non-A1 runs (about 47) and A6/A6r (about 24) |
| **4. Exp B in full** | **about 38** | 15 Breakout runs at about 1.9 each |
| total | **about 275 GPU-hours** (about 11.5 GPU-days) | |

With the fallback of one 40k candidate pool (D10) the pools cost about 70 instead of about 140, so phase 3 is
about 150 and the total about 205.

### Phase 2 in detail

- **A1 at full length (D13): a three-value learning-rate sweep (seed 0) plus a second seed at the chosen rate, 4 runs
  at about 1.87 each, 7.5.** These count as A1's phase 3 runs, so phase 3 needs one more A1 seed only.
- The other arms (A3, A4 ×2 clips, A5, A7) at 10% of the steps, 5 runs at about 15 min each: **1.25**.
- A6 / A6r smoke, 2 rounds at 10% scale (S/4 steps per round, D11): **0.6**. Its candidate slice is the first 2,000
  questions of each of the first two shared pools, generated at 570 per GPU-hour: **about 7, reused in phase 3**.
- Evaluation of about 10 models (selection split only for the two δ runs, D13): **1.2**; per-answer-type
  gradient-norm share: **0.3**; Breakout label validation: **0.3**.

### Phase 3 in detail

- **Candidate pools: 4 rounds × 20k = 80k questions, about 140 GPU-h in all (about 133 left after phase 2).** They
  are generated once and shared by A6, A6r and all seeds (D9); generating per run would multiply the cost by 6.
  The pools are independent of any model, so they can run on up to 4 GPUs in parallel (about 35 h of wall time) and
  can start as soon as the budget is confirmed.
- A1 needs 1 more seed: **1.9**. A3, A4 ∞, A4 clip 10, A5, A7: 5 arms × (3 learning rates + 2 more seeds) = 25 runs
  at about 1.87: **47**. Exploratory A3 with β = 0: **1.9**.
- **A6 and A6r (D11): each run has the shared step count in total (S/4 per round), about 1.87 plus roughly 0.5 for
  four segments' load, tokenise and save, so about 2.4 per run; 2 arms × 5 runs = 24.**
- **Scoring candidates is needed by A6 only** (A6r picks at random inside the quotas, D12): 5 runs × 4 rounds × 20k
  questions at no less than 17 questions/s, about 20 min each: **about 6**.
- Evaluation of about 45 models: **5**.

### Assumptions that change the numbers

1. **Step count per arm** follows the old setting (2 epochs, 654 steps); the spec leaves it open, and it scales the
   training lines above.
2. **The candidate pools use authored questions** at the measured rate (about 570 kept per GPU-hour); seeded
   (templated) families were cheaper per question in the old jobs but their yield was not logged, so they are not
   credited.
3. **Labels come with generation** (D12), so there is no separate labelling pass.
4. **The δ runs count as A1 phase 3 seeds** only if they use the learning rate the sweep selects (D13); if that fails,
   add about 3.7 for two more A1 seeds.
5. Queueing is not included. On 2026-10-06 the cluster had about 965 nodes allocated, and probe job 7114048 had
   waited more than an hour; wall-clock time for phases 3 and 4 will be set by the queue, not by the GPU-hours above.

## Probe job 7114048

Status when this file was written: **pending (Priority)**. When it runs, its log will give today's tokens/s and
memory for the 40-step training probe and the wall time of forward-only scoring of the dev set at batch sizes 16
and 48. If any of those differ from the measured rates above by more than about 25%, the table above is
re-derived and this file is updated with a dated note.
