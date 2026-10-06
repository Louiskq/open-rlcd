# Decisions

Amendments to the project spec, made by the maintainer on **2026-10-06** after reviewing phase 1. The spec
itself is not stored in this repository, so **this file is the authoritative record of where the repository
departs from it**. Each entry says what was decided, why, what it implies, and where it is implemented.
Decisions made later are appended; an entry is never silently rewritten.

## D1. KL direction: forward, KL(p_ref ‖ p_θ)

- **Spec said:** `max E[S(p_θ, y)] − β KL(p_θ ‖ p_ref)`.
- **Decided:** `max E[S(p_θ, y)] − β KL(p_ref ‖ p_θ)`, matching decider's retention term and the old Breakout
  code. The equation in the README is corrected. β is identical across all Exp A arms.
- **Why it matters:** the forward KL is large when the policy puts little probability on an option the reference
  supports, so it **penalises collapse onto one option and partly counteracts RLVR's overconfidence**. A3's
  measured overconfidence is therefore a *lower bound* on what the improper reward does by itself.
- **Exploratory, outside the pre-registered comparisons:** one extra run of A3 with β = 0, to show how much of
  the KL was masking. It is labelled exploratory everywhere it is reported and is not part of the Holm family.
- **Implemented:** `rlcd.estimators.kl_grad_logits` defaults to `direction="forward"` (`"reverse"` stays
  available for comparison).

## D2. "Raw" means temperature 1

- **Decided:** raw probabilities, and the distribution that is trained and scored, are `softmax(logits)` at
  **T = 1**. The stored 1.30 of the decider checkpoint is itself a post-hoc calibration, which is exactly what A2
  and the "+TS" rows measure, so it must not be baked into "raw".
- **Added reference row: "A0 as served (T = 1.30)"**, next to A0 at T = 1.
- **Every "+TS" row fits its own temperature from scratch** on the calibration split, one per answer type,
  from the T = 1 logits. It is never composed with the stored 1.30.
- **Implemented:** `rlcd.calibrate` fits from raw logits; the served-temperature row is evaluation-only.

*Correction, 2026-10-06: the stored temperature of this checkpoint (`decider9b-sft`, read from its
`decider_config.json`) is **1.05**, not 1.30. The 1.30 quoted above is the 2B model's value in decider's `RL.md`. The
"A0 as served" row therefore uses T = 1.05, and "+TS" rows still fit their own temperatures from scratch.*

## D3. SCORE readout: decider's isolated levels, normalised

- **Decided:** keep decider's one yes/no pass per level for **all arms**, because the start model was trained
  that way.
- **Checked first, as instructed.** Decider's pass asks `"{question} / Proposed answer: {level} / Does the
  proposed answer fit?"` and its training rows are labelled `1 if j == gold else 0`
  (`decider/systemone.py: ISOLATED`, `decider/data/augment.py: isolated`, read from `Mapika/decider` on
  2026-10-06). So each pass asks **"score = k", not "score ≥ k"**. The cumulative branch (RPS as a sum of
  per-threshold Brier scores, a non-monotonicity rate) therefore **does not apply**.
- **Applied branch ("score = k"):** per-level yes probabilities `y_k` (at T = 1, D2) are normalised into one
  distribution `p_k = y_k / Σ_j y_j` *before* the log score or RPS is computed, and the gradient flows through
  the normalisation. **Reported for every arm and the smoke run: the mean absolute deviation of `Σ_j y_j` from
  1.**
- **Consequences to keep in view:**
  - A SCORE question with L levels costs **L forward passes** per step (2 to 10), which enters the GPU budget.
  - **A4 on SCORE items optimises the log score, not RPS**: the bandit reward `1[a=y]/p(a)` is an unbiased
    estimator of `∇ log p(y)` and nothing else (RPS is not recoverable from the indicator of one sampled answer).
    A5 uses RPS through the RL code path. H6 is a comparison of A1 and A7 only, so it is unaffected.
  - A1 continues the start model with RPS on the normalised distribution, which is not the per-level
    binary cross-entropy the start model was trained with; that is a deliberate change of objective.
- **Implemented:** `rlcd.isolated` (`combine`, `mass_deviation`, `score_grad_level_logits`), tested against
  decider's `combine_isolated` formula and by finite differences.

*Amended by D-lean: A7 and H6 are dropped, so the statement that H6 is a comparison of A1 and A7 no longer applies. A4 on SCORE items still optimises the log score, A5 and A1 still use RPS.*

## D4. RPS stays divided by K − 1

- **Decided:** RPS is normalised by `K − 1`, so it lies in [0, 1].
- **Consequence, stated up front:** this sets the weight of SCORE questions *relative to NOUL and CHOICE* in the
  RL reward. The log score is unbounded and its gradient can be large on confident mistakes, while a normalised
  RPS is bounded, so **SCORE questions will carry less gradient than their share of the pool**.
- **Reported, not corrected:** the smoke run reports **each answer type's share of the total gradient norm**
  alongside its share of the pool.
- **Implemented:** `rlcd.scoring.rps(..., normalise=True)`.

## D5. Pre-registration

- **δ (equivalence margins).** In the phase 2 smoke run, train **A1 on 2 seeds** and set each δ to **twice the
  observed between-seed difference of that metric on the selection split, never the evaluation split**. The
  accuracy δ is floored at **1.0 point** (the old B1 seeds spread 0.9). All δ values are fixed **before** the
  pre-registration is committed.
- **A6 selection rule.** Predictive entropy **normalised by log K**; take the top N **within each answer type**,
  each type's quota proportional to its share of the candidate pool. **A6r uses the same per-type quotas** and
  draws at random inside them, so A6 against A6r tests mining and not a change in the mix of answer types.
  Implemented in `rlcd.mining` (`select_mined`, `select_random_matched`).
- **Seeds:** training 0, 1, 2; selection 1000–1049; evaluation 2000–2049 (as proposed).
- **When:** the pre-registration is committed **after the smoke run and before any full run**.
- *Open point on the δ procedure (raised in the phase 2 estimate):* the smoke run uses 10% of the steps, and seed
  spread at 10% of training may differ from the spread of a full run. **Resolved 2026-10-06 by D13.**

## D6. Exp B

- **B3 follows the spec's definition**: actor + belief + retention, with weights 1.0 / 2.0 / 7.0.
- **Two deliberate deviations from decider's `RL.md`, both recorded here:** B3 **omits the rendering-consistency
  term** and **omits the retention gate** (see `docs/loss_weights.md`).
- The old repo's single best-checkpoint Breakout number is **not reproduced anywhere in this repository**. The
  one run's final checkpoint scored 33 greedy (plain PPO: 34); the other number came from choosing a checkpoint
  on 3 fixed games, which is the selection problem the new protocol (separate selection seeds) exists to avoid.

## D7. Exploratory work (outside the pre-registered comparisons)

1. **A3 with β = 0** (D1).
2. **Exp 0b**: a toy in which the difficulty varies by input (the scale of the true logits changes with the
   context). Reported: ECE and resolution after temperature scaling, RLVR against RLCD. It asks whether a single
   global temperature can still repair RLVR when overconfidence differs from input to input. Code and results:
   `toy/exp0b.py`, `results/toy0b/`; clearly labelled exploratory in the README.

## D8. Repository

`github.com/Louiskq/open-rlcd` is created **private**, so CI, Colab and badge links resolve. It is made public
once the Exp A results are in.

## Still proposed (not yet confirmed by the maintainer)

- The number of optimiser steps, batch size and β shared across Exp A arms (to be fixed before the
  pre-registration is committed).
- Items marked *[to confirm]* in D10, D11, D13 and D14 below.

---

## Answers to the phase 2 / 3 questions (2026-10-06)

## D9. Candidate pools are shared

- **Decided:** one candidate pool per round, **shared by A6, A6r and all seeds**. Selection still happens per seed:
  A6 against that seed's current model, A6r at random inside the per-type quotas with that seed's generator.
- **Stated in the README:** seeds share candidate data, so the seed spread of A6 and A6r **understates data
  variance** (it reflects training and selection randomness, not a different draw of candidate questions).
- Because A6 and A6r draw from the same pool, A6 against A6r also isolates the selection rule from the quality of
  the candidates.

*Amended by D-lean: there is no generated pool any more. A6 and A6r select from the same held-back half B of the existing pool, shared by both arms and all seeds; the README caveat that seed spread understates data variance stands.*

## D10. N stays at 5k; the fallback is one 40k pool

- **Decided:** N = 5k, so 4N = 20k candidates per round and 80k over four rounds. H4 predicts a null result, and a
  null is only convincing if enough data was added for it to have mattered; D11 saves more compute than halving N
  would.
- **Fallback, only if the maintainer says the budget is tight after the allocation check:** ONE candidate pool of
  40k drawn without replacement across all four rounds. *Interpretation to confirm before it is used:* each round
  scores the not-yet-selected part of that pool and takes N, so selection pressure is weaker (about 2:1, not 4:1).
  **Not** N = 2.5k.

*Superseded by D-lean: no new teacher generation, so N = 5k per round, the 80k pool and the 40k-pool fallback are void. N per round is now proposed at 1,500 from the 10.3k held-back half.*

## D11. A6 and A6r get the same total optimiser steps as every other arm

- **Decided:** the spec requires the same number of optimiser steps for every arm, so the total across all four
  rounds equals the shared step count S. **Each round gets S/4, continuing from the previous round's weights, never
  restarting.** Same for A6r. Each round's training data is the base pool plus all items mined so far.
- *Implementation detail assumed, to confirm:* one learning-rate schedule over the whole S steps, with the
  optimiser state carried across rounds (what "continuing" means in practice).
- **Consequence:** an A6 run costs about as much as an A1 run, not four times as much, and any lead A6 has over A1
  comes from the added data and not from extra training.

## D12. Labels come with generation; selection never sees them

- **Decided:** use the labels that come with generation; there is no separate teacher labelling pass.
- To keep "selected before labelled", the selection functions must not receive the label field, and labelling is
  **simulated by hiding labels until after selection**. Implemented as `rlcd.data.CandidatePool` (the scorer and
  selection see `UnlabelledQuestion`, which has no gold field; `reveal(indices)` is the only way to get labels) and
  `rlcd.mining.mine_round`.
- **The test that fails if a selection function receives a label:** `tests/test_label_hiding.py` calls
  `rlcd.mining.assert_label_free` on every selection function (it raises if a `gold` / `label` / `y` / similar
  parameter appears), checks that two pools identical except for their labels select exactly the same items, and
  checks that labels are revealed after scoring and only for the selected items. Breaking it in five different
  ways (a label parameter, a second reveal, revealing every item, a pool that leaks labels, an emptied name list)
  fails the build.
- **Disclosed:** the generation cost for all candidates is paid up front, so the "teacher-call budget" is nominal:
  both arms reveal the labels of N items per round.

## D13. The two δ seeds run at full length and count as A1 seeds

- **Decided:** run both A1 seeds for δ at **full length**. They may also count as two of A1's three phase 3 seeds,
  because δ uses only the selection split and the evaluation split stays untouched: **the evaluation-split
  predictions of these two runs are not computed or looked at until the pre-registration is committed.**
- **Sequencing this implies, to confirm.** To count as phase 3 seeds the two runs must use the learning rate phase
  3 would choose for A1, which comes from A1's three-value sweep on the selection split. Proposed: in phase 2 run
  A1's sweep at full length (seed 0), pick the learning rate on selection log loss, run seed 1 at that rate, and
  set δ = 2 × |difference| on the selection split; phase 3 then needs one more A1 seed. The alternative (a
  pre-declared rate, 1e-4 as in the old runs) only counts as phase 3 seeds if the sweep later picks that rate.
- The δ rule itself is unchanged from D5: twice the observed between-seed difference on the selection split,
  accuracy floored at 1.0 point.

*Amended by D-lean: A1's three-value sweep is shared with A6 and A6r (one sweep chooses the learning rate for all three), which does not change the δ procedure.*

## D14. H5b, a pre-registered secondary hypothesis

- **Decided:** add **H5b**: *after temperature scaling, A3 has worse per-domain ECE (equal-mass bins, the 4 dev
  domains) and lower Murphy resolution than A1.* H5 stays as written. Motivation: Exp 0b showed pooled ECE after
  temperature scaling can hide opposite miscalibration across strata.
- Per-domain ECE is the mean over the four dev domains of the equal-mass-bin top-label ECE
  (`rlcd.scoring.grouped_ece`). The domain of each dev question comes from the `domain` field of the dev records
  (to verify when the smoke run reads them).
- *To confirm:* H5b is analysed as a **secondary**, Holm-corrected within its own two-metric family (per-domain ECE,
  resolution) and not entered into the primary Holm family of C1–C7, so that adding it does not cost power there.

## D15. Splits confirmed

Train 90% / calibration 5% / selection 5% (`rlcd.data.make_splits`, salt `rlcd-v1`).

## D16. H7: how "B3 beats B4 at a matched game score" is decided

- **(a) A common set of states.** Belief quality is measured on rollouts from **one frozen reference policy (B1,
  seed 0, final checkpoint) on fresh evaluation seeds**, so a different state distribution cannot confound the
  belief metrics.
- **(b) The rule.** H7 holds if B3 beats B4 on belief log loss on those states (paired-bootstrap CI excludes 0)
  **and** B3's mean greedy game score is **not lower than B1's** (B4's policy *is* B1's) **by more than δ_game**.
  δ_game is set from the B1 seed spread **before any B3 result is seen** (so B1 runs complete first); its exact
  formula (for example the range of B1's three seeds on the selection seeds) is fixed in the pre-registration.
- On-policy belief metrics (each arm's own rollouts) are secondary and reported as such.
- This replaces the earlier proposed matching rule in the pre-registration draft.

*Amended by D-lean: B6 is dropped and B5 runs on one seed; the H7 rule is unchanged.*

## D17. Commit identity before the repository goes public

- **Decided:** before the repository goes public, rewrite every commit's author and committer email to the
  maintainer's GitHub no-reply address, set it in this repository's git config, force-push, and confirm with
  `git log --format='%ae %ce' | sort -u`.
- **Status: done 2026-10-06.** All 18 commits on `main` were rewritten (author and committer email only; names,
  dates, messages and file contents unchanged, tree-identical to before) to the GitHub no-reply address, set in
  this repository's git config, and force-pushed with `--force-with-lease`. Verified with
  `git log --format='%ae %ce' | sort -u` (one line, the no-reply address) and against GitHub's API for all 18
  commits.
- **Caveat found afterwards:** GitHub still serves the pre-rewrite commits **by their hash** (checked for two of them),
  with the old address, and the Actions runs reference them. Nothing is exposed while the repository is private. If
  the old address must never be reachable, delete and recreate the repository and push the rewritten history **before
  it is made public**; that has not been done.
- **Two things to weigh when doing it (raised, not decided):** a rewrite changes every commit hash, so it is best done
  **before the pre-registration commit** is made, so that the commit that freezes the pre-registration never has
  its hash changed afterwards. And a force-push does not remove the old commits from GitHub at once: Actions runs and
  cached refs can still resolve the old SHAs for a while. If the old address must never be reachable, the safer
  route is to delete and recreate the repository, then push the rewritten history, before it is made public.

## D18. Phase 2 is on hold

Phase 2 is held until the maintainer confirms the remaining Isambard allocation. The only job submitted is the
~15 minute throughput probe requested earlier (job 7114048), which was still waiting in the queue at the time of
writing; nothing else has been submitted.

---

## D-lean. A lean plan before phase 2 (2026-10-06)

Decided by the maintainer to cut the cost of Exp A and Exp B. **Nothing has been submitted under this plan.**

1. **A6 / A6r: no new teacher generation.** Split the 22.9k pool in half by hash. **A6 starts on half A and mines
   from half B over 4 rounds, with labels hidden until selection** (the same label-blind test as D12). **A6r selects
   from half B at random with the same per-type quotas.** Same total optimiser steps as the other arms (D11).
2. **README framing:** A2 against the RL arms is the **main test of the cynical reading**; A6 against A6r tests
   **whether mining adds anything**.
3. **Dropped:** the clipped A4 variant (Exp 0 already shows the clip bias), A7, and H6.
4. **A5 runs on 1 seed** (correctness check only).
5. **Learning-rate sweeps:** one shared sweep for A1 / A6 / A6r, one shared sweep for A3 / A4.
6. **Exp B:** B6 dropped; B5 on 1 seed.
7. **The GPU-hour estimate is recomputed and reported before anything is submitted** (`docs/compute_budget.md`:
   about 11 / 38 / 16 GPU-hours for phases 2 / 3 / 4, about 65 in total, down from about 275).

### Readings made to carry this out *(each to confirm)*

- **Which pool is halved.** "The 22.9k pool" is read as its **90% training split** (D15), so the calibration and
  selection splits stay out of training. Halves are by hashed id (`rlcd.data.split_halves`, salt `rlcd-v1-halves`,
  independent of the 90/5/5 assignment): about 10.3k questions each.
- **What each arm trains on.** A1–A5 train on the whole training split (A ∪ B). A6 and A6r train on half A plus
  whatever they select from half B, so **they never see more data than A1**. A6 against A1 is therefore *not* a
  volume comparison and is **not pre-registered**; A6 against A6r is the clean contrast (same half A, same candidate
  half B, same quotas, only the selection rule differs).
- **Mechanics.** Four rounds. The candidate set is the not-yet-selected part of half B (it shrinks). Each round:
  score the candidates (A6 only) with the model after the previous segment (round 1: the start model); select N per
  round with the per-type quotas of D5; reveal their labels; train S/4 steps on half A plus everything selected so
  far, continuing from the previous weights (D11).
- **N per round: proposed 1,500** (6k of the 10.3k, so round 1 takes the top 15% of the candidates and round 4
  about the top 26%). This is a real trade-off: a larger N adds more data but is less selective, and at N ≈ 2,570
  all of half B would be taken by round 4, leaving no selection effect in the final data at all. A smaller N is
  more selective but adds less data, which is the "enough added data to have mattered" worry behind D10.
- **Which arm is swept.** One sweep on **A1** picks the learning rate for A1, A6 and A6r. One sweep on **A4** (three
  rates, seed 0) picks it for A3 and A4. *Risk to note:* RLVR is sensitive to the learning rate (a smaller rate
  means less collapse in a fixed number of steps), so A3 is run at a rate tuned for RLCD-bandit, not at its own best.
  H1 is then a statement about A3 at that rate. The alternative, sweeping A3 and A4 separately, costs about 5.6 more
  GPU-hours.
- **H2** is evaluated as **A5 seed 0 against A1 seed 0**, equivalence within ±δ. The two objectives are identical,
  so the paired difference should be near zero up to numerical non-determinism; δ comes from A1's seed spread (D5).
- **H3** loses its clipping clause in Exp A (the clipped arm is dropped); the toy gradient-noise table remains the
  evidence for the bias. Its first clause (A4 has higher across-seed variance than A1) stays, descriptively.
- **Comparisons.** C3 keeps only A4 against A1; C5 becomes A2 against each of A3, A4 and A5, each after temperature
  scaling; C6 is deleted. H5b is unchanged (D14), run with A3 at the shared learning rate above.
- **Exp B** runs B1 (3 seeds), B2 (temperature scaling on B1's rollouts, 3 seeds), B3 (3 seeds), B4 (from B1's
  policy, 3 seeds) and B5 (1 seed, descriptive, no pre-registered test). B6 and the belief-weight sensitivity are not
  tested.

---

## D-impl. Implementation choices made in phase 2 (2026-10-06)

Choices the design left open, made while building the Exp A runner (`rlcd/train_labels.py`). Items marked
*[to confirm]* fix the shared settings of every Exp A arm and are therefore part of what the pre-registration freezes.

**Data and rendering**
- **Records and splits.** The pool is 13,387 records / 22,961 questions (CHOICE 54%, NOUL 34%, SCORE 12%). Splits are
  by *record* (a state and its questions stay together): train 12,006 records / 20,596 questions, split into half A
  (6,014 records, 10,312 questions) and half B (5,992 records, 10,284); calibration 686 records / 1,176 questions;
  selection 695 / 1,189 (`results/expA/splits.json`). The dev set (951 records, 2,791 questions, **4 domains**, SCORE
  21.6% of its questions against 12% of the pool) shares no record id and no state text with the pool.
- **One row per question, no packing, no replay, no augmentation.** Each question is its own row (as the server scores
  it), state-first plain layout, no "none of the above" augmentation, no layout mixing, no replayed rows from decider's
  mixture (the forward-KL term, D1, plays that role). Options are shuffled once per question from a fixed seed, so every
  arm and every training seed sees the same prompts; each row stores its permutation, so readouts are mapped back to the
  original option order. SCORE is read only through its isolated rows (D3).

**Training** *[to confirm: all of these]*
- LoRA rank 64, alpha 128 on decider's usual target modules (as in the old text-LoRA runs), bf16 base, fp32 adapters;
  AdamW (0.9, 0.95), no weight decay, 20 warm-up steps then cosine to zero, gradient clip 1.0, 4 micro-batches of up to
  16,384 tokens per optimiser step.
- **S = 734 optimiser steps for every arm**: two epochs over the whole training split (A ∪ B), computed once from its
  micro-batch plan. A6 and A6r run S/4 per round on their own, smaller data (D11).
- **K = 8** sampled answers per question for A3 and A4, group-mean baseline. **β = 0.1** on the forward KL(p_ref ‖ p),
  identical across arms. Observed at 10% of S, learning rate 1e-4: the mean per-question KL to the start model at the end of training is
  about 0.17 nats for A1 and A5 (0.26 for A3, 0.29–0.32 for A6 / A6r), i.e. a penalty of about 0.017 against a log loss of about 0.5, so β = 0.1 is a mild
  regulariser for the supervised arms and a real constraint only for arms that move further.
- **Learning-rate grid 5e-5, 1e-4, 2e-4**, chosen on the **selection split by mean per-question log loss over all
  answer types**. The A1 sweep (full length, seed 0) is running; A6 / A6r take its rate, and A3 / A4 take the rate of an
  A4 sweep (D-lean).
- No checkpoint/resume: a run is under about 2.5 h. Runs save the LoRA adapter (`lora_adapter.pt`), not a merged model.

**Evaluation**
- A run saves the **raw readout of every question** at T = 1 on the calibration, selection and dev splits; everything
  else (SCORE normalisation, temperatures, all metrics, per-domain ECE) is computed offline from those arrays
  (`rlcd/evaluate.py`), so one forward pass serves raw, "as served" (T = 1.05) and "+TS" rows.
- **Temperatures are fitted on the calibration split, which is drawn from the training distribution and is harder than
  the dev set** (the start model scores 0.69 accuracy on the selection split and 0.81 on dev). A temperature fitted
  there is therefore not tuned to dev, and can slightly *worsen* dev log loss (seen for A1 in the smoke run).
- **JevBench is deferred to phase 3**: it needs a merged checkpoint served through decider (`jevbench_decider.sh`),
  which the adapter-only runs do not produce. Merging is a one-off step after the final runs.
- The start model scores **0.808 dev accuracy** in this pipeline, not the 0.815 of the old evaluation (different option
  shuffles and layout details); A0 here is the baseline for everything.

**Measurements (D4)**
- Per-type gradient share is the norm of each type's gradient accumulated over **all micro-batches of one optimiser
  step**, as a fraction of the sum of the three norms, recorded at several steps (`grad_share` in `run.json`). A first
  version measured a single micro-batch and fluctuated wildly (SCORE 1% to 64%); it was replaced before any number was
  reported.


## D19. The pre-registration is frozen and Phase 3 started (2026-10-06)

- **Pre-registration frozen at commit `112dc50`** (`PREREGISTRATION.md`, with `results/expA/delta.json`). From here it is
  edited only through its Amendments section. The maintainer's "go" on 2026-10-06 confirmed the open readings (halved
  training split; N = 1,500 per round; A4 the swept arm for A3 / A4; H2 as A5 seed 0 against A1 seed 0; one
  learning-rate schedule over S with optimiser state carried across A6 / A6r rounds).
- **A1 learning rate 1e-4** (lowest selection-split log loss, 0.341 against 0.360 and 0.367); **δ** is twice the
  difference of A1 seeds 0 and 1 on the selection split (1,189 questions), the accuracy floor binding only for CHOICE.
  **H1's accuracy margin therefore becomes ±2.2 points**, wider than the spec's ±1, which is stated in the
  pre-registration.
- **Dev predictions** of the sweep and the seed-1 run were written by the runner but not opened; `compute_delta.py` and
  `pick_lr.py` read calibration and selection predictions only.
- **Phase 3 wave 1** was submitted by `scripts/expA_phase3.sh` right after the freeze (jobs 7129659–7129669: A1 seed 2,
  the A4 sweep, A5, A6 and A6r × 3 seeds, all at 1e-4 except the A4 sweep). Wave 2 (A4 seeds 1–2, A3 × 3 at the A4
  rate, A3 with β = 0) is submitted by the same detached driver once the A4 sweep has finished.
- **Time limits:** 3 h 30 min per run, 4 h 30 min for A6 / A6r (candidate scoring adds about half an hour).
- **Commit email, checked rather than assumed:** the pre-rewrite commits are still served by GitHub with the maintainer's
  old Gmail address in author and committer fields (`gh api repos/Louiskq/open-rlcd/commits/<old sha>`, four old SHAs
  found through the Actions runs). GitHub's "keep my email private" setting does not rewrite commit metadata made from
  the command line (it applies to web edits, and the separate "block command line pushes that expose my email" option
  would have rejected those pushes); so D17's delete-and-recreate step still stands before the repository is made public.
