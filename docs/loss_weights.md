# Verifying the Breakout loss weights against `decider/docs/RL.md`

**Question (spec, ground rules).** The old repo's comments put the Breakout weights at
actor 1.0 / belief 2.0 / retention 7.0. Check them against `decider/docs/RL.md` before reusing them
and record what the docs actually say.

**Checked on 2026-10-05** against `Mapika/decider` `docs/RL.md` (read from GitHub, default branch) and
the old repo's `hardgen/rl_calibrated.py` (commit `069af08` of `Louiskq/jev-class-9b`).

## What the docs say

`RL.md` describes the v8 → v10 calibration-aware RL stage. Its loss terms and coefficients:

| term | coefficient in `RL.md` | what it is |
|---|---|---|
| 1. Actor | **0.1** | PPO clipped surrogate (clip 0.2), terminal return, leave-one-replicate-out baseline over 4 repeats |
| 2. Belief | **0.2** | log score of the answer to a "what happens next" question, against the exact law |
| 3. Rendering consistency | **1.0** | KL between the served answer and the answer in three other renderings of the same question |
| 4. Retention | **0.7** | KL(v8 ‖ student) on 8 replayed supervised rows per step |

Optimiser: AdamW (FP32 master weights over BF16), betas 0.9 / 0.95, weight decay 0.01, peak learning
rate 1e-6 with 16 warm-up steps and a cosine schedule over 576 steps, gradient norm clipped at 1.0.

Two further facts in `RL.md` that matter here:

* **The environments are not Breakout.** They are live MiniWoB++ click tasks, exact 4x4 minesweeper, a
  5x5 slippery grid and draws from bags of known composition. `RL.md` says the training code "lives in a
  separate research repository"; the Breakout recipe is therefore *the old repo's adaptation*, not
  something `RL.md` specifies.
* **Retention is gated.** If the mean retention KL over the 8 rows exceeds 0.01 nats, or any row exceeds
  0.05 nats, that step drops the actor and belief terms, zeroes the optimiser's first moment and applies
  only the retention term (94 of 576 steps in the released run).

## What the old Breakout script does

`hardgen/rl_calibrated.py` defaults: `--belief_w 2.0` ("v10 ratio belief/actor = 0.2/0.1"),
`--ret_w 7.0` ("retention/actor = 0.7/0.1"), and an actor coefficient of 1 (the PPO term is added to the
loss unscaled, as in the plain run `decider.games.rl`).

## Verdict

**The ratios are confirmed; the absolute coefficients are not the docs'.** 1.0 / 2.0 / 7.0 is the docs'
0.1 / 0.2 / 0.7 divided by the actor coefficient 0.1, i.e. actor-normalised. That is a faithful way to
carry the *balance* of the terms over to a different actor loss (the plain PPO term has coefficient 1),
and Adam is close to scale-invariant, so the rescale is nearly harmless. It is not identical, for two
reasons that should be written down rather than assumed away:

* **Gradient clipping at norm 1.0 is not scale-invariant.** With every term ten times larger, the
  clip binds more often than in the docs' absolute scale.
* **Learning rates differ.** The old Breakout runs use a constant 4e-6 (the plain-PPO default); the docs
  use a peak of 1e-6 with warm-up and cosine decay.

## Differences between the old Breakout recipe and `RL.md` (none of these are bugs; all must be stated)

| | `RL.md` (v10) | old Breakout script |
|---|---|---|
| actor / belief / retention | 0.1 / 0.2 / 0.7 | 1.0 / 2.0 / 7.0 (same ratios) |
| rendering-consistency term (1.0 in `RL.md`, so 10 on the rescaled axis) | present | **absent** |
| retention gate (drop terms 1 and 2 above 0.01 mean / 0.05 per-row KL) | present | **absent** (`--kl_stop 0.05` is the PPO old-to-new policy KL on sampled actions, a different quantity) |
| retention rows per step | 8 | 32 (from a fixed pool of 1,000 replayed rows) |
| retention direction | KL(v8 ‖ student) | KL(start ‖ current): the same direction |
| actor baseline | leave-one-replicate-out over 4 repeats | per-timestep batch mean, advantages standardised |
| learning rate | 1e-6 peak, warm-up + cosine | 4e-6 constant |
| weight decay | 0.01 | PyTorch AdamW default (0.01) |

## Consequences for Exp B in this repo

1. **B3 reuses 1.0 / 2.0 / 7.0** (the ratio-faithful adaptation that produced the old results), and the
   README says so, including that Breakout is *not* an environment `RL.md` covers.
2. **B6 (belief weight halved and doubled)** therefore means belief 1.0 and 4.0 on this scale.
3. **B5 (no retention KL)** sets the retention weight to 0.
4. **The missing rendering-consistency term and retention gate are deliberate omissions to disclose**,
   not silent ones. If B3 shows forgetting that the old runs did not, the gate is the first suspect.
5. **The KL direction in the task spec differs from the code.** The spec's objective is written with
   KL(p_θ ‖ p_ref) (reverse); decider's retention term and the old Breakout code use KL(p_ref ‖ p_θ)
   (forward). `rlcd.estimators.kl_grad_logits` supports both. **Open decision for the maintainer:** which
   direction Exp A's β-penalty uses. B3/B5/B6 keep the forward direction to match the verified recipe.

## Caveat on what the old Breakout numbers can support

The old results (`hardgen/EXPERIMENTS.md`: greedy 54 for "v2 best" against 34 for plain PPO) are one run
per arm, 10 greedy games, with the "best" checkpoint chosen on 3 fixed greedy games; the final iterate of
the same run scores 33 greedy. That gap is what selection on a tiny, reused set produces. Exp B is
designed to avoid it: three seeds per arm, 50 greedy + 50 sampled games on fresh evaluation seeds, and
checkpoints chosen on separate selection seeds (`rlcd.data.SeedSets`).
