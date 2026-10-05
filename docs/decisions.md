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
- *Open point on the δ procedure (raised in the phase 2 estimate, not yet decided):* the smoke run uses 10% of
  the steps, and seed spread at 10% of training may differ from the spread of a full run.

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

- Split fractions for the training pool: train 90% / calibration 5% / selection 5%.
- The operational definition of H7's "matched game score" (in `PREREGISTRATION.md`).
- The number of optimiser steps, batch size and β shared across Exp A arms (to be fixed after the throughput
  measurement and before the pre-registration is committed).
