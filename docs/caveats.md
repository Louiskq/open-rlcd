# Caveats and results that went against a hypothesis

Part of [open-rlcd](../README.md).

## Honest caveats

The first group of caveats was written *before* the language-model results existed, so they could not be shaped by
them; the later ones were added as the results came in and say so.

- **The toy is well-specified, and that flatters temperature scaling.** The policy class contains the truth, so
  RLVR only distorts the scale of the logits and a single temperature undoes it. Exp 0b shows that stops being
  true when difficulty varies. Whether it holds for a real model is exactly what A2 / A3 and hypothesis **H5**
  test. The toy shows the *mechanism*, not the size of the
  effect in a 9B model.
- **All arms start underconfident** (a uniform policy), so the early part of every curve is a start-up effect
  shared by all arms, not an RLCD property.
- **ECE on sampled labels has a noise floor** (it is not zero even for the true probabilities); it is reported
  next to every table.
- **Group-mean baseline.** The specified baseline (mean over the K samples of a question, including the sample
  itself) shrinks the gradient by exactly (K−1)/K without changing its direction. A leave-one-out baseline is
  exactly unbiased; both are implemented and tested.
- **Decisions that change what the experiments mean** are recorded in [`docs/decisions.md`](decisions.md):
  the KL direction (forward), "raw" meaning temperature 1 (the checkpoint's stored 1.05 is itself a post-hoc
  calibration, so it appears only as a separate "as served" reference row), how `SCORE` answers are read out
  (decider's one yes/no pass per level, normalised into a distribution), and RPS normalisation (which gives
  `SCORE` questions less gradient than their share of the pool).
- **Seeds share candidate data.** A6 and A6r choose from the same held-back half of the pool, shared by both arms
  and by all seeds; selection still differs per seed. Their seed spread therefore understates data variance.
- **A6 never sees more data than A1**: it trains on half A plus what it selects from half B. A6 against A1 is
  therefore not a volume comparison and is not pre-registered; the clean contrast is A6 against A6r.
- **A lean experiment set, chosen for budget** ([`docs/decisions.md`](decisions.md) D-lean). Dropped: the
  clipped RLCD-bandit arm in Exp A (Exp 0 already measures the clipping bias), the RPS-versus-log-loss ablation on
  `SCORE` (and its hypothesis), and the belief-weight sensitivity in Exp B. Run on one seed only: A5 (a
  correctness check) and B5 (a descriptive ablation). Learning-rate sweeps are shared (A1 with A6 and A6r; A3 with
  A4), so RLVR is run at a rate tuned for RLCD-bandit, not at its own best.
- **Planned-experiment limitations, stated now:** the independent dev set was written by the same teacher as the
  training data; JevBench hard has 111 items (standard error about 4.4 points), too few to separate the arms.
- **Results that went against a pre-registered hypothesis, by name** (the numbers are in [results.md](results.md)):
  **H1** was not met as stated (RLVR's ECE is higher but its log loss is not, on the dev set); **H5** was not established
  (the equivalence margin, set from two A1 seeds, was narrower than the intervals could resolve); **H5b** was not met (it
  was met only on the exploratory selection split). **H3** and **H4** held, but H3 did not under the seed-ensemble reading,
  and A6's unadjusted log-loss lead over A6r did not survive correction. Where a result is mixed it is reported as mixed.
- **The pooled log loss mixes training objectives** for `SCORE` (A1, A5, A6, A6r use the ranked probability score; A3, A4
  sample answers), which was not anticipated in the pre-registration; the per-type table is the like-for-like view.
- **Greedy Breakout play is deterministic** (found while B1 trained, before any B3 result): a greedy "game" is one
  trajectory whatever the seed, so Amendment 3 adds a sampled-play condition to H7, making it stricter.
