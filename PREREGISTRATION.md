# Pre-registration

> **STATUS: FROZEN** at the commit that adds this file (2026-10-06), after the phase 2 smoke run and the A1 sweep
> and δ seed, and before any other full run. Per `docs/decisions.md` D5 it is **never edited**: changes go under
> [Amendments](#amendments) with a date and a reason. Design decisions are D1–D18, D-lean and D-impl in
> `docs/decisions.md`; the maintainer confirmed the readings listed there by instructing "go" on 2026-10-06. Before
> this file was written nothing in Exp B had been trained; the Exp 0 and Exp 0b toy results in `results/` are not
> pre-registered. What had been seen when it was frozen is listed under [Seen before freezing](#seen-before-freezing).

## Hypotheses

H1–H7 are from the project spec; H5b was added by the maintainer (D14). **H6 is dropped before any run (D-lean)
and will not be tested or reported as a hypothesis.**

- **H1.** A3 (RLVR) has higher ECE and log loss than A1. Its accuracy is within ±δ_acc of A1 (δ_acc is **2.2 points** on all questions, wider than the
  spec's ±1 point because it is set from the A1 seed spread, see below). *(A3 is run at the learning rate chosen for
  A4, D-lean, so H1 is a statement about RLVR at a rate tuned for RLCD-bandit.)*
- **H2.** A5 matches A1 on every metric within seed noise. This is a correctness check; if it fails, find the bug
  before interpreting anything else. *(A5 has one seed: evaluated as A5 seed 0 against A1 seed 0, equivalence within
  ±δ.)*
- **H3.** A4 reaches A1's calibration with higher variance across seeds. *(The second clause of the spec's H3, that
  clipping lowers variance and adds measurable bias, is not tested in Exp A: the clipped arm is dropped. Exp 0's
  gradient bias and noise table is the evidence for that mechanism.)*
- **H4.** A6 is no better than A6r beyond seed noise. This would agree with the earlier finding that
  hard-example mining adds nothing beyond volume.
- **H5.** With temperature scaling, A1 is within noise of every RL arm on ECE in Exp A. That is, fixed labels give
  RL no calibration advantage. *(The RL arms are A3, A4 and A5; "A1 plus temperature scaling" is A2.)*
- **H5b (pre-registered secondary).** After temperature scaling, A3 has worse **per-domain ECE** (equal-mass bins,
  the 4 dev domains) and lower **Murphy resolution** than A1. H5 is kept as written.
- **H7 (the open question).** In Breakout, B3 beats B4 on belief log loss at a matched game score. If it does,
  joint on-policy training matters and RLCD is more than the cynical reading. If not, the cynical reading wins in
  this setting too. Operationalised under [H7 rule](#h7-rule-d16).

**How the cynical reading is tested.** The main test is **A2 against the RL arms** (H5, H5b). **A6 against A6r**
(H4) tests only whether mining adds anything; B3 against B4 (H7) is the on-policy counterpart.

## Making "within noise" testable

- **"A is better than B" on metric M**: the paired-bootstrap 95% CI (over items, on seed-averaged predictions) of
  M(A) − M(B) excludes 0 in the stated direction, after Holm correction across the comparisons below.
- **"A matches B" (H2, H5)**: the same CI lies entirely inside an equivalence margin ±δ_M.
- **How δ is set (D5, D13).** **A1 is trained on 2 seeds at full length** (seed 0 is the learning-rate sweep's chosen
  run `a1_lr1e-4_s0`, seed 1 is `a1_lr1e-4_s1`). For each metric M,
  **δ_M = 2 × |M(seed 0) − M(seed 1)| measured on the selection split, never the evaluation split** (1,189 questions:
  413 NOUL, 634 CHOICE, 142 SCORE). Raw rows use T = 1; +TS rows use each run's own temperatures fitted on the
  calibration split. The accuracy δ is floored at **1.0 point** (the floor binds only for CHOICE, whose two seeds differ by 0.2 point). The same
  δ_acc is the accuracy margin of H1. **The values are fixed here** (`results/expA/delta.json`, computed by
  `scripts/compute_delta.py`, committed with this file):

| scope | metric | δ raw | δ +TS |
|---|---|---|---|
| all types | log_loss | 0.0325 | 0.0316 |
| all types | ece | 0.0178 | 0.0083 |
| all types | acc | 2.2 pt | 2.2 pt |
| all types | brier | 0.0191 | 0.0187 |
| all types | reliability | 0.0094 | 0.0051 |
| all types | resolution | 0.0278 | 0.0249 |
| NOUL | log_loss | 0.0484 | 0.0429 |
| NOUL | ece | 0.0135 | 0.0257 |
| NOUL | acc | 3.9 pt | 3.9 pt |
| NOUL | brier | 0.0352 | 0.0327 |
| NOUL | reliability | 0.0161 | 0.0167 |
| NOUL | resolution | 0.0501 | 0.0497 |
| CHOICE | log_loss | 0.0248 | 0.0251 |
| CHOICE | ece | 0.0158 | 0.0138 |
| CHOICE | acc | 1.0 pt | 1.0 pt |
| CHOICE | brier | 0.0066 | 0.0050 |
| CHOICE | reliability | 0.0097 | 0.0049 |
| CHOICE | resolution | 0.0190 | 0.0109 |
| SCORE | log_loss | 0.0209 | 0.0280 |
| SCORE | ece | 0.0759 | 0.0378 |
| SCORE | acc | 8.5 pt | 8.5 pt |
| SCORE | brier | 0.0280 | 0.0387 |
| SCORE | reliability | 0.0432 | 0.0103 |
| SCORE | resolution | 0.0628 | 0.0300 |

  The per-type SCORE rows rest on 142 questions and are very wide, so SCORE comparisons are descriptive. The ECE
  equivalence margin used by C5 on all questions is therefore **0.0083** after TS (0.0178 raw).
  - The two runs also count as two of A1's three phase 3 seeds. **Their evaluation-split (dev) predictions were
    written by the runner but not opened by anyone** (not read, no metric computed) until this file is committed;
    `scripts/pick_lr.py` and `scripts/compute_delta.py` read only the calibration and selection splits (the cluster copy
    of `eval_metrics.json` and `run.json` for these runs also holds dev metrics and was not printed).
  - They use the learning rate chosen for A1 (1e-4, the lowest selection-split log loss of the three-value sweep:
    0.341 against 0.360 at 2e-4 and 0.367 at 5e-5; the 5e-5 and 2e-4 runs are sweep runs and are not A1 seeds).
- **H3 variance**: the across-seed standard deviation of final ECE and log loss. With 3 seeds per arm this is
  descriptive (reported with its range), not a significance test.

## H7 rule (D16)

- **(a) Common states.** Belief quality is measured on rollouts from **one frozen reference policy, B1 seed 0, final
  checkpoint, on fresh evaluation seeds**, so a different state distribution cannot confound belief metrics.
- **(b) The rule.** H7 holds if **B3 beats B4 on belief log loss on those states** (paired-bootstrap CI excludes 0)
  **and** B3's mean greedy game score is **not lower than B1's by more than δ_game** (B4's policy is B1's).
  **δ_game is set from the B1 seed spread before any B3 result is seen** (B1 runs complete first): **δ_game = the range
  (maximum minus minimum) of the mean greedy game score of B1's three seeds on the selection seeds 1000–1049, floored at
  5% of their mean** (so that a chance small range of three seeds does not make the condition unreachable). It is
  computed and written to `results/expB/delta_game.json` before any B3 run is submitted.
- On-policy belief metrics (each arm's own rollouts) are secondary and reported as such.

## Comparisons

**Primary family (Holm-corrected together):**

| id | comparison | metrics |
|---|---|---|
| C1 | A3 vs A1 | ECE, log loss (one-sided: A3 worse); accuracy (equivalence ±δ_acc) |
| C2 | A5 seed 0 vs A1 seed 0 | every primary and secondary metric (equivalence) |
| C3 | A4 vs A1 | ECE, log loss; across-seed variance (descriptive) |
| C4 | A6 vs A6r | ECE, log loss, accuracy |
| C5 | A2 vs each of A3 / A4 / A5, each + TS | ECE (equivalence) |
| C7 | B3 vs B4 | belief log loss on the common states, with the game-score condition of the H7 rule |

(C6, the RPS-against-log-loss comparison, is deleted with H6.)

**Pre-registered secondary (H5b)**: A3 vs A1 after temperature scaling on two metrics, per-domain
ECE (one-sided: A3 worse) and Murphy resolution (one-sided: A3 lower), **Holm-corrected within this two-metric
family** and not entered in the primary family.

**Exploratory, outside every Holm family** (labelled as such wherever reported): **(E1)** one run of A3 with
β = 0, to show how much the forward KL was masking RLVR's overconfidence (D1); **(E2)** Exp 0b, the toy with
input-dependent difficulty (`results/toy0b/`). B5 (no retention KL) on one seed is descriptive and has no test.

## Metrics and statistics

Primary: log loss, Brier score and its Murphy decomposition (reliability, resolution, uncertainty).
Secondary: top-label ECE (15 equal-width bins; equal-mass as a robustness check), accuracy, classwise ECE for
CHOICE, **per-domain ECE with equal-mass bins (the H5b metric, `rlcd.scoring.grouped_ece`)**, RPS and mean
absolute error of the expected level for SCORE (reported descriptively; there is no RPS-versus-log-loss test).
Decision metric: risk–coverage curve and AURC. Every metric is reported per answer type (NOUL / CHOICE / SCORE).
Implemented and tested in `rlcd/scoring.py`.

3 seeds per arm (A5 and B5: 1); seed mean and range are reported. Pairwise comparisons: paired bootstrap 95% CIs
over items on seed-averaged predictions (for per-domain ECE, resampled within each domain), Holm-corrected as above.

**Probabilities and temperature scaling (D2).** "Raw" is `softmax(logits)` at **T = 1**. Every "+TS" row fits its own
temperature **from scratch**, one per answer type, on the calibration split. A separate reference row, **"A0 as
served (T = 1.05)"**, shows the start model as decider ships it (1.05 is the stored temperature of
`decider9b-sft`, read from its `decider_config.json`).

**SCORE readout (D3).** Decider's one yes/no pass per level (an exact-level question, "score = k"), per-level yes
probabilities normalised into one distribution before the log score or RPS. The mean absolute deviation of the
unnormalised sum from 1 is reported for every arm. **RPS is divided by K − 1 (D4)**, and the share of the total
gradient norm taken by each answer type is reported from the smoke run.

## Data, splits and seeds

- **Start model:** the Stage 1 SFT checkpoint of the Qwen3.5-9B decider (dev accuracy 0.815, per the project
  spec; not re-measured here). **Training pool:** the 22.9k teacher-written text questions.
- **Splits** (`rlcd.data.make_splits`, salted hash of the question id, salt `rlcd-v1`): train 90%, calibration 5%,
  selection 5% (D15, confirmed). The calibration split fits the temperatures and is disjoint from the dev set and
  from JevBench; the selection split chooses the learning rates, the checkpoints and the δ values.
- **Halves for A6 / A6r (D-lean):** the **training split** is halved by hashed id (`rlcd.data.split_halves`, salt
  `rlcd-v1-halves`) into half A and half B, 10,312 and 10,284 questions. "The pool" of D-lean is the training split. A1–A5 train on A ∪ B; A6 and A6r train on half A plus what they select from half B, so they
  never see more data than A1.
- **Evaluation:** the independent dev set (2,791 questions, 4 unseen domains) is the main test set; JevBench hard
  (111 items) is reported but is too small to separate the arms (standard error about 4.4 points). The dev set was
  written by the same teacher as the training data; this is a limitation, not a defence. The domain of each dev
  question for H5b comes from the dev records' `domain` field (verified in phase 2: 951 dev records, 4 domains).
- **Seeds** (`rlcd.data.SeedSets`, disjoint by construction): training **0, 1, 2**; selection **1000–1049**;
  evaluation **2000–2049** (Breakout game seeds). `assert_disjoint` runs on train / cal / select / dev / JevBench
  before every Exp A run, and on halves A, B and the held-out splits.

## Design fixed before any full run

- **Arms of Exp A:** A0 (start model; also "as served"), A1 (continued log loss, RPS for SCORE), A2 (A1 + TS),
  A3 (RLVR), A4 (RLCD-bandit, unclipped), A5 (RLCD-full through the RL path), A6, A6r, plus the exploratory A3
  with β = 0. **Dropped before any run (D-lean): the clipped A4 variant, A7.**
- **Shared across Exp A arms (D-impl):** LoRA rank 64 / alpha 128 on decider's usual targets;
  **S = 734 optimiser steps** for every arm (two epochs over the whole training split), 4 micro-batches of up to
  16,384 tokens per step, AdamW (0.9, 0.95), warm-up 20 then cosine, clip 1.0; **K = 8**; **β = 0.1**, identical
  across arms; 3 seeds except A5. The penalty is the forward KL(p_ref ‖ p_θ) (D1). The group-mean baseline of the RL arms
  shrinks the gradient by (K−1)/K (`rlcd.estimators.advantages`).
- **Learning-rate sweeps (D-lean):** three values, **5e-5, 1e-4, 2e-4**, chosen on the selection split by mean per-question
  log loss over all answer types (diverged runs excluded: `scripts/pick_lr.py`). **One sweep on A1
  chooses the rate for A1, A6 and A6r; one sweep on A4 (seed 0) chooses it for A3 and A4** (the swept arm for A3 / A4 is A4).
- **A6 / A6r (D-lean, D11, D12).** No new teacher generation. **R = 4 rounds**; the candidate set is the
  not-yet-selected part of half B (it shrinks). Each round: score the candidates with the model from the previous
  segment (A6 only; round 1 uses the start model), select **N per round** (**1,500**) by
  predictive entropy normalised by log K, top N within each answer type in proportion to that type's share of the
  candidates (A6), or at random inside the same per-type quotas (A6r); reveal the labels; train **S/4 steps** on
  half A plus everything selected so far, **continuing from the previous round's weights, never restarting**
  (one learning-rate schedule over all S steps, optimiser state carried across rounds). **Selection
  never receives labels**: labels are hidden until after selection (`rlcd.data.CandidatePool`,
  `rlcd.mining.mine_round`, guarded by `tests/test_label_hiding.py`). After the last round temperature scaling is
  applied. A6 and A6r share half B across arms and seeds, so their seed spread understates data variance.
- **A4 on SCORE questions** optimises the log score (the bandit reward cannot estimate RPS); A5 uses RPS. See D3.
- **Exp B:** arms **B1** (plain PPO), **B2** (B1 + temperature scaling on held-out rollouts), **B3** (actor +
  belief + retention, weights 1.0 / 2.0 / 7.0), **B4** (B1's policy, supervised belief fit on its rollouts, then
  temperature scaling), **B5** (B3 without retention KL; **1 seed**, descriptive). **B6 is dropped (D-lean).** The v2
  label fix; belief questions brick30, life30 and return. **Deviations from decider's `RL.md`, both deliberate:** no
  rendering-consistency term, no retention gate (D6, `docs/loss_weights.md`). 50 greedy + 50 sampled evaluation
  games on fresh seeds; checkpoint chosen on selection seeds, final checkpoint also reported.
- **Exp B label validation** (base rate against hand inspection of 20 episodes per belief, plus log-score skill
  against the base rate) ran *before* B-arm training (`docs/breakout_label_validation.md`): the legacy and an independent labeller agree on all
  10,942 resolved questions, base rates 0.778 (return), 0.221 (life30), 0.635 (brick30), the physical checks and 20
  hand-inspected episodes show no error. It cannot catch a wrong definition or RAM address, and truncation drops about
  9% of the asked questions.

## Seen before freezing

Stated so that readers can judge which hypotheses were tested blind. Nothing below is an Exp A result.

- **Smoke run** (`docs/phase2_smoke.md`; one seed, 10% of the steps, evaluated on the selection and dev splits). The
  A3 start showed the H1 pattern on the selection split (raw ECE 0.075 against 0.026–0.034, log loss 0.610 against
  0.538) and not on dev; A4 and A5 were indistinguishable from A1; A6 and A6r were not separable. These runs informed
  nothing in the design except that the pipeline works.
- **A0** (start model, T = 1 and as served) was evaluated on the selection and dev splits.
- **The A1 sweep and δ seed** were read on the **selection and calibration splits only** (log loss, accuracy, ECE and
  the other metrics above, and per-type gradient shares). Their dev predictions are unread.
- **Breakout label validation** was run (above); no B arm has been trained.

## Amendments

### Amendment 1 (2026-10-08): how the pre-registered comparisons are run

Written **before any Phase 3 run's calibration, selection or dev predictions were opened** (the only Phase 3 numbers
seen were the three A4 sweep selection log losses the launcher used to pick the A4 rate: 0.340 at 1e-4, 0.364 at
5e-5, 0.384 at 2e-4; all 17 jobs finished and none diverged). It fixes details the text above leaves open; it changes
no hypothesis, margin or comparison.

1. **Split.** Every test below uses the **dev set** (2,791 questions, 4 domains). The selection split and JevBench are
   reported descriptively and are not tested. Scope is the pooled `all` row of the δ table; per-type results are
   descriptive.
2. **"Seed-averaged predictions" means the mean over seeds of each metric, computed per seed.** For arms with three
   seeds, a bootstrap replicate resamples the items once and every seed of every arm is scored on the same resampled
   items, then the metric is averaged over seeds. (The δ margins are single-run differences, which this matches.) Averaging
   the probabilities of the seeds first, an ensemble, is reported as a robustness check only. "+TS" rows use each run's own
   temperatures fitted on the calibration split (D2).
3. **Bootstrap.** 2,000 replicates, random seed 20261008, items resampled with replacement, the same indices for all
   arms, seeds and rows; for per-domain ECE items are resampled within each domain.
4. **p-values.** For a claim "A is worse/better than B" with difference d = M(A) − M(B): p = 2 × (share of replicates
   with d on the wrong side of 0 or equal to 0), capped at 1, so that p < 0.05 is the same as the 95% interval excluding 0 in the
   stated direction. For equivalence within ±δ: p_eq = 2 × max(share of replicates with d ≤ −δ, share with d ≥ δ), so
   that p_eq < 0.05 is the same as the 95% interval lying inside (−δ, δ). Holm's step-down correction is applied to
   these p-values at 0.05, **separately for the primary family and the H5b family**. Unadjusted 95% percentile
   intervals are reported too.
5. **The primary family (17 tests, δ from the table above: raw rows for raw comparisons, +TS rows for +TS comparisons).**
   - C1, A3 against A1, raw: ECE higher; log loss higher; accuracy equivalent within ±δ_acc (2.2 points).
   - C2, A5 seed 0 against A1 seed 0, raw: equivalence for log loss, ECE, accuracy, Brier, reliability and resolution.
   - C3, A4 against A1, raw: equivalence for ECE and log loss. (Across-seed variance is descriptive.)
   - C4, A6 against A6r, raw: A6 better, that is ECE lower, log loss lower, accuracy higher (one direction each).
   - C5, +TS rows, equivalence of ECE within δ_ECE(+TS) = 0.0083: A1 against A3; A1 against A4; A1 seed 0 against A5 seed 0.
6. **The H5b family (2 tests, +TS rows):** A3 against A1, mean over the 4 dev domains of equal-mass (15 bins) ECE higher;
   A3 against A1, Murphy resolution lower.
7. **What counts as each hypothesis holding.** H1: C1's ECE and log-loss tests and its accuracy equivalence all pass.
   H2: all six C2 equivalences pass. H3: both C3 equivalences pass (and the A4 across-seed s.d. of ECE and log loss is
   reported next to A1's). H4: no C4 test passes (the intervals are reported, so a null is not read as an
   equivalence). H5: all three C5 equivalences pass. H5b: both H5b tests pass. Anything else is reported as
   "not shown", with its interval.
8. **Descriptive only, no test:** A0 (T = 1, as served, +TS), the A3 β = 0 run (E1), per-type results, the selection
   split, JevBench, risk-coverage and AURC, Murphy parts other than those above, and the learning-rate sweeps.

### Amendment 2 (2026-10-08): how Exp B is run and decided

Written **before any Exp B arm was trained** (only the label validation and a few-iteration smoke test of the code
exist). It fixes details the text above leaves open; it changes no hypothesis.

1. **One loop for all RL arms** (`rlcd/breakout_gpu.py`): B1, B3 and B5 differ only in the loss weights
   actor : belief : retention = 1 : 0 : 0, 1 : 2 : 7 and 1 : 2 : 0. Settings are those of decider's plain Breakout
   run: 40 iterations of 16 episodes, 300 decisions per episode, learning rate 4e-6, γ = 0.97, PPO clip 0.2, entropy
   0.01, four PPO chunks of up to 6,000 decisions per iteration, stop a chunk loop when the mean KL exceeds 0.05,
   gradient clip 1.0, AdamW (0.9, 0.95). Belief questions are asked every 4th decision in every arm (B1 records them
   but is not trained on them); retention uses 1,000 replayed decider questions, 32 per step.
2. **Horizon.** Every game (training, selection, evaluation) lasts 300 decisions; the score is the Atari score at the end.
3. **Seeds.** Training environments use seeds 100,000 × (seed + 1) + episode number; selection seeds are 1000–1049;
   evaluation seeds are 2000–2049; the label check used 3000–3159. `rlcd.breakout.seed_sets` raises on any overlap.
4. **Checkpoint selection.** The greedy mean score on the 50 selection seeds is measured for the start model and
   every 5 iterations; the **selected** checkpoint is the highest (ties keep the earlier; the start model if nothing
   beats it). The **final** checkpoint (iteration 40) is also evaluated. The selected checkpoint is the primary one for
   H7 and the tables.
5. **Evaluation.** 50 greedy and 50 sampled games on evaluation seeds 2000–2049 for the selected and the final checkpoint.
6. **δ_game** = range of B1's three seeds' selected-checkpoint greedy mean scores **on the selection seeds**, floored at
   5% of their mean (the rule above); `results/expB/delta_game.json` is written and committed before any B3 job is
   submitted.
7. **Common states.** B1 seed 0, final checkpoint, sampled play on the evaluation seeds with belief questions every 4th
   decision (brick30, life30, and return when a ball is in play), resolved by the validated labeller
   (`rlcd.breakout.BeliefTracker`). Every arm and seed answers exactly these questions.
8. **B2** = B1's selected checkpoint with one temperature fitted on its own beliefs in 50 sampled games on the selection
   seeds. **B4 seed i** = B1 seed i's final policy fine-tuned for 160 steps (the 4 × 40 belief batches of B3) of 128 questions
   by the log score only, on the belief questions in B1 seed i's own training rollouts, with no actor or retention term,
   then one temperature fitted on its answers to B1 seed i's selection-seed questions. **B4's game score is B1's.**
   B3 + TS is fitted like B2. B5 has one seed and no test.
9. **The H7 test (C7, a family of one, no Holm).** The statistic is the mean over seeds i of B3 seed i's belief log
   loss (selected checkpoint, T = 1) minus B4 seed i's (+TS) on the common-state questions, with a paired bootstrap over
   those questions (2,000 replicates, random seed 20261008). **H7 holds iff** the 95% interval of that difference lies
   below 0 **and** the mean over B3 seeds of the selected checkpoint's greedy score on the evaluation seeds is at least the B1
   mean minus δ_game. B3 + TS against B4, B3 against B2, B5, per-kind results, ECE, accuracy, skill against the base
   rate, on-policy beliefs and the final checkpoints are reported **descriptively**.
