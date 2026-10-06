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

## D10. N stays at 5k; the fallback is one 40k pool

- **Decided:** N = 5k, so 4N = 20k candidates per round and 80k over four rounds. H4 predicts a null result, and a
  null is only convincing if enough data was added for it to have mattered; D11 saves more compute than halving N
  would.
- **Fallback, only if the maintainer says the budget is tight after the allocation check:** ONE candidate pool of
  40k drawn without replacement across all four rounds. *Interpretation to confirm before it is used:* each round
  scores the not-yet-selected part of that pool and takes N, so selection pressure is weaker (about 2:1, not 4:1).
  **Not** N = 2.5k.

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

## D17. Commit identity before the repository goes public

- **Decided:** before the repository goes public, rewrite every commit's author and committer email to the
  maintainer's GitHub no-reply address, set it in this repository's git config, force-push, and confirm with
  `git log --format='%ae %ce' | sort -u`.
- **Status:** waiting for the address. Nothing has been rewritten.
- **Two things to weigh when doing it (raised, not decided):** a rewrite changes every commit hash, so it is best done
  **before the pre-registration commit** is made, so that the commit that freezes the pre-registration never has
  its hash changed afterwards. And a force-push does not remove the old commits from GitHub at once: Actions runs and
  cached refs can still resolve the old SHAs for a while. If the old address must never be reachable, the safer
  route is to delete and recreate the repository, then push the rewritten history, before it is made public.

## D18. Phase 2 is on hold

Phase 2 is held until the maintainer confirms the remaining Isambard allocation. The only job submitted is the
~15 minute throughput probe requested earlier (job 7114048), which was still waiting in the queue at the time of
writing; nothing else has been submitted.
