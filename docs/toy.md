# Exp 0 and Exp 0b: toy experiments

Part of [open-rlcd](../README.md). Two small experiments where the true probabilities are known, so the effect of the
reward can be measured exactly. `make toy` and `make toy0b` regenerate everything here in a couple of minutes on a laptop CPU.

## Exp 0 results: a toy where the truth is known

<!-- BEGIN gen:toy_setup -->
10 seeds × 4,000 Adam steps per arm, evaluated on 20,000 held-out contexts. The true probabilities themselves score ECE [0.010](../results/toy/meta.json) on this sample (the noise floor) and the Bayes-optimal accuracy is [0.500](../results/toy/meta.json).
<!-- END gen:toy_setup -->

A linear-softmax policy over 5 options, labels drawn from a fixed ground-truth softmax, so the policy class
contains the truth and "calibrated" has an exact meaning. Every arm has the same model, optimiser, learning rate
and data stream per seed; only the reward differs.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../figures/toy_overview-dark.png">
    <img alt="ECE and accuracy against training step for four arms, and a reliability diagram per arm. The RLVR curve falls far below the diagonal while the RLCD and supervised curves sit on it." src="../figures/toy_overview.png" width="900">
  </picture>
</p>

<!-- BEGIN gen:toy_table -->
| arm | ECE | accuracy | log loss | KL(truth ‖ model) | ECE after temperature scaling |
|---|---|---|---|---|---|
| RLVR | [0.381 (0.380–0.382)](../results/toy/rlvr.json) | [0.499 (0.498–0.500)](../results/toy/rlvr.json) | [3.047 (3.037–3.057)](../results/toy/rlvr.json) | [1.837 (1.827–1.846)](../results/toy/rlvr.json) | **[0.010 (0.008–0.012)](../results/toy/rlvr.json) (T = 6.1)** |
| RLCD-bandit | **[0.011 (0.008–0.014)](../results/toy/rlcd_bandit.json)** | [0.498 (0.496–0.501)](../results/toy/rlcd_bandit.json) | **[1.221 (1.219–1.225)](../results/toy/rlcd_bandit.json)** | **[0.006 (0.004–0.010)](../results/toy/rlcd_bandit.json)** | **[0.010 (0.008–0.013)](../results/toy/rlcd_bandit.json) (T = 1.0)** |
| RLCD-bandit, clip 10 | [0.060 (0.056–0.066)](../results/toy/rlcd_bandit_clip10.json) | [0.499 (0.496–0.502)](../results/toy/rlcd_bandit_clip10.json) | [1.241 (1.238–1.244)](../results/toy/rlcd_bandit_clip10.json) | [0.026 (0.022–0.029)](../results/toy/rlcd_bandit_clip10.json) | **[0.010 (0.009–0.012)](../results/toy/rlcd_bandit_clip10.json) (T = 1.3)** |
| Supervised log loss | **[0.009 (0.006–0.012)](../results/toy/supervised.json)** | [0.498 (0.495–0.500)](../results/toy/supervised.json) | **[1.220 (1.218–1.222)](../results/toy/supervised.json)** | **[0.004 (0.003–0.007)](../results/toy/supervised.json)** | **[0.009 (0.005–0.012)](../results/toy/supervised.json) (T = 1.0)** |
| RLCD-full | **[0.009 (0.006–0.012)](../results/toy/rlcd_full.json)** | [0.498 (0.495–0.500)](../results/toy/rlcd_full.json) | **[1.220 (1.218–1.222)](../results/toy/rlcd_full.json)** | **[0.004 (0.003–0.007)](../results/toy/rlcd_full.json)** | **[0.009 (0.005–0.012)](../results/toy/rlcd_full.json) (T = 1.0)** |
| _the true probabilities_ | [0.010](../results/toy/meta.json) | [0.500](../results/toy/meta.json) | [1.215](../results/toy/meta.json) | 0 | |

Seed mean (min–max). Bold: best, or within the best arm's seed range. Supervised log loss and RLCD-full are the same objective through two code paths; their curves differ by at most [4.4e-16](../results/toy/meta.json) over every metric, seed and step.
<!-- END gen:toy_table -->

What the figure and table say:

- **Accuracy does not move.** RLVR is not worse at picking the answer; it is worse at knowing how sure to be.
- **RLVR is overconfident, and its log loss is worse than the uniform guess.** It passes through calibration early
  and keeps sharpening.
- **RLCD-bandit lands on the diagonal using only the sampled answer**, and tracks supervised log loss.
- **Clipping the weight at 10 is measurably biased**: options whose true probability is under 0.1 are
  under-weighted, so the policy ends up too sharp (next table).
- **Supervised log loss and RLCD-full are one curve**: with fixed labels they are the same objective.
- **One temperature repairs RLVR in this toy** (the hollow markers). See the caveats, and Exp 0b below for
  what happens when difficulty varies.

### Why: bias and noise of each gradient estimator at the truth

At the true parameters a proper, unbiased objective has zero expected gradient (calibration is a fixed point).
Measured over many batches, with all arms seeing the same contexts and labels:

<!-- BEGIN gen:toy_noise -->
| arm | bias ‖E[g]‖ | Monte Carlo floor | bias / floor | noise (trace of covariance) |
|---|---|---|---|---|
| RLVR | [0.0435](../results/toy/meta.json) | 0.0015 | 29.4× | [0.007](../results/toy/meta.json) |
| RLCD-bandit | [0.0062](../results/toy/meta.json) | 0.0059 | 1.0× | [0.106](../results/toy/meta.json) |
| RLCD-bandit, clip 10 | [0.0250](../results/toy/meta.json) | 0.0050 | 5.0× | [0.074](../results/toy/meta.json) |
| Supervised log loss | [0.0044](../results/toy/meta.json) | 0.0050 | 0.9× | [0.074](../results/toy/meta.json) |
| RLCD-full | [0.0044](../results/toy/meta.json) | 0.0050 | 0.9× | [0.074](../results/toy/meta.json) |
<!-- END gen:toy_noise -->

The importance-weighted reward is unbiased but noisier than supervised log loss. Clipping removes much of that
extra noise at the price of a bias well above the Monte Carlo floor. RLVR has a clearly nonzero gradient at the
truth, which is the formal sense in which it is improper.

### How much RLVR does it take?

`reward = (1 − m)·1[a=y]/p(a) + m·1[a=y]`: `m = 0` is RLCD-bandit, `m = 1` is RLVR.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../figures/toy_mix-dark.png">
    <img alt="ECE and log loss against the share of RLVR mixed into the reward: flat near the noise floor for small shares, then rising steeply." src="../figures/toy_mix.png" width="760">
  </picture>
</p>

<!-- BEGIN gen:toy_mix -->
| share of RLVR in the reward | ECE | ECE after temperature scaling | log loss |
|---|---|---|---|
| 0% | [0.011 (0.010–0.013)](../results/toy/mix_sweep.json) | [0.010 (0.009–0.013)](../results/toy/mix_sweep.json) | [1.220 (1.219–1.222)](../results/toy/mix_sweep.json) |
| 2% | [0.011 (0.010–0.014)](../results/toy/mix_sweep.json) | [0.009 (0.008–0.010)](../results/toy/mix_sweep.json) | [1.221 (1.220–1.222)](../results/toy/mix_sweep.json) |
| 5% | [0.010 (0.008–0.011)](../results/toy/mix_sweep.json) | [0.010 (0.009–0.012)](../results/toy/mix_sweep.json) | [1.220 (1.219–1.221)](../results/toy/mix_sweep.json) |
| 10% | [0.011 (0.009–0.015)](../results/toy/mix_sweep.json) | [0.010 (0.008–0.011)](../results/toy/mix_sweep.json) | [1.220 (1.219–1.222)](../results/toy/mix_sweep.json) |
| 20% | [0.015 (0.008–0.022)](../results/toy/mix_sweep.json) | [0.009 (0.006–0.010)](../results/toy/mix_sweep.json) | [1.221 (1.220–1.222)](../results/toy/mix_sweep.json) |
| 35% | [0.023 (0.011–0.030)](../results/toy/mix_sweep.json) | [0.010 (0.007–0.012)](../results/toy/mix_sweep.json) | [1.223 (1.222–1.224)](../results/toy/mix_sweep.json) |
| 50% | [0.049 (0.043–0.056)](../results/toy/mix_sweep.json) | [0.011 (0.009–0.012)](../results/toy/mix_sweep.json) | [1.231 (1.228–1.235)](../results/toy/mix_sweep.json) |
| 75% | [0.115 (0.110–0.118)](../results/toy/mix_sweep.json) | [0.008 (0.007–0.010)](../results/toy/mix_sweep.json) | [1.286 (1.281–1.289)](../results/toy/mix_sweep.json) |
| 100% | [0.382 (0.381–0.382)](../results/toy/mix_sweep.json) | [0.011 (0.011–0.012)](../results/toy/mix_sweep.json) | [3.052 (3.049–3.057)](../results/toy/mix_sweep.json) |
<!-- END gen:toy_mix -->

The damage grows gradually with the share. It is not a cliff, and a small share is hard to tell from the noise.

### Does the ordering depend on the learning rate?

Final ECE under several learning rates, so the comparison is not an artefact of one setting:

<!-- BEGIN gen:toy_lr -->
3 seeds per cell. Seed mean (min–max).

| arm (final ECE) | lr 0.003 | lr 0.01 | lr 0.03 |
|---|---|---|---|
| RLVR | [0.325 (0.325–0.326)](../results/toy/lr_sensitivity.json) | [0.382 (0.381–0.382)](../results/toy/lr_sensitivity.json) | [0.421 (0.420–0.423)](../results/toy/lr_sensitivity.json) |
| RLCD-bandit | [0.008 (0.008–0.009)](../results/toy/lr_sensitivity.json) | [0.011 (0.010–0.013)](../results/toy/lr_sensitivity.json) | [0.017 (0.009–0.021)](../results/toy/lr_sensitivity.json) |
| RLCD-bandit, clip 10 | [0.056 (0.055–0.058)](../results/toy/lr_sensitivity.json) | [0.059 (0.058–0.060)](../results/toy/lr_sensitivity.json) | [0.079 (0.069–0.087)](../results/toy/lr_sensitivity.json) |
| Supervised log loss | [0.010 (0.008–0.012)](../results/toy/lr_sensitivity.json) | [0.009 (0.006–0.012)](../results/toy/lr_sensitivity.json) | [0.016 (0.011–0.023)](../results/toy/lr_sensitivity.json) |
| RLCD-full | [0.010 (0.008–0.012)](../results/toy/lr_sensitivity.json) | [0.009 (0.006–0.012)](../results/toy/lr_sensitivity.json) | [0.016 (0.011–0.023)](../results/toy/lr_sensitivity.json) |
<!-- END gen:toy_lr -->

## Exp 0b (exploratory, not pre-registered): what if difficulty varies?

In Exp 0 one temperature repairs RLVR, partly because RLVR sharpens every input by about the same factor. Exp 0b
asks what happens when it cannot. The *scale* of the true logits now changes with the input (hard inputs have
near-uniform true probabilities, easy ones near-deterministic ones) while the policy can still represent the truth
exactly, so any miscalibration left after temperature scaling comes from the reward. The scale range was fixed in
the code before the first run and not tuned.

<!-- BEGIN gen:toy0b_setup -->
10 seeds × 4,000 Adam steps per arm, 20,000 held-out contexts, true-logit scale between 0.3 (hard) and 3.0 (easy).
<!-- END gen:toy0b_setup -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../figures/toy0b-dark.png">
    <img alt="Three panels after one temperature. Pooled ECE is at the noise floor for every arm. Confidence minus accuracy by difficulty tercile is large and of opposite signs for RLVR, near zero for the RLCD arms. RLVR has lower Murphy resolution." src="../figures/toy0b.png" width="900">
  </picture>
</p>

<!-- BEGIN gen:toy0b_table -->
| arm | ECE | ECE after one temperature | resolution after | KL(truth ‖ model) after | confidence − accuracy after, hard / medium / easy |
|---|---|---|---|---|---|
| RLVR | [0.405 (0.403–0.407)](../results/toy0b/rlvr.json) | [0.010 (0.007–0.012)](../results/toy0b/rlvr.json) (T = 7.2) | [0.158 (0.157–0.159)](../results/toy0b/rlvr.json) | [0.030 (0.029–0.031)](../results/toy0b/rlvr.json) | [+0.093](../results/toy0b/rlvr.json) / [−0.018](../results/toy0b/rlvr.json) / [−0.087](../results/toy0b/rlvr.json) |
| RLCD-bandit | [0.010 (0.006–0.018)](../results/toy0b/rlcd_bandit.json) | [0.008 (0.005–0.012)](../results/toy0b/rlcd_bandit.json) (T = 1.0) | [0.165 (0.161–0.167)](../results/toy0b/rlcd_bandit.json) | [0.010 (0.007–0.015)](../results/toy0b/rlcd_bandit.json) | [−0.000](../results/toy0b/rlcd_bandit.json) / [−0.002](../results/toy0b/rlcd_bandit.json) / [−0.011](../results/toy0b/rlcd_bandit.json) |
| RLCD-bandit, clip 10 | [0.050 (0.042–0.064)](../results/toy0b/rlcd_bandit_clip10.json) | [0.010 (0.007–0.014)](../results/toy0b/rlcd_bandit_clip10.json) (T = 1.3) | [0.166 (0.165–0.167)](../results/toy0b/rlcd_bandit_clip10.json) | [0.007 (0.005–0.008)](../results/toy0b/rlcd_bandit_clip10.json) | [−0.010](../results/toy0b/rlcd_bandit_clip10.json) / [−0.004](../results/toy0b/rlcd_bandit_clip10.json) / [−0.011](../results/toy0b/rlcd_bandit_clip10.json) |
| Supervised log loss | [0.009 (0.006–0.015)](../results/toy0b/supervised.json) | [0.009 (0.007–0.013)](../results/toy0b/supervised.json) (T = 1.0) | [0.166 (0.164–0.167)](../results/toy0b/supervised.json) | [0.007 (0.005–0.010)](../results/toy0b/supervised.json) | [+0.000](../results/toy0b/supervised.json) / [−0.003](../results/toy0b/supervised.json) / [−0.013](../results/toy0b/supervised.json) |
| RLCD-full | [0.009 (0.006–0.015)](../results/toy0b/rlcd_full.json) | [0.009 (0.007–0.013)](../results/toy0b/rlcd_full.json) (T = 1.0) | [0.166 (0.164–0.167)](../results/toy0b/rlcd_full.json) | [0.007 (0.005–0.010)](../results/toy0b/rlcd_full.json) | [+0.000](../results/toy0b/rlcd_full.json) / [−0.003](../results/toy0b/rlcd_full.json) / [−0.013](../results/toy0b/rlcd_full.json) |
| _the true probabilities_ | [0.006](../results/toy0b/meta.json) | | [0.169](../results/toy0b/meta.json) | 0 | |
<!-- END gen:toy0b_table -->

- **Judged by pooled ECE alone, temperature scaling "fixes" RLVR**: after one temperature every arm is at the noise
  floor.
- **The pooled number hides the problem.** RLVR is still too confident on hard inputs and not confident enough on
  easy ones; averaging the two cancels. The RLCD arms are close to calibrated in every stratum.
- **RLVR also ends with lower resolution and a larger distance to the true probabilities** after temperature
  scaling than the RLCD arms (table above).
- RLVR needs a *different* temperature in each stratum, while the RLCD arms need about the same everywhere:

<!-- BEGIN gen:toy0b_group -->
| arm | temperature per tercile (hard / medium / easy) | ECE after | resolution after | KL(truth ‖ model) after |
|---|---|---|---|---|
| RLVR | 11.5 / 6.7 / 5.3 | [0.007 (0.005–0.009)](../results/toy0b/rlvr.json) | [0.166 (0.165–0.167)](../results/toy0b/rlvr.json) | [0.009 (0.008–0.009)](../results/toy0b/rlvr.json) |
| RLCD-bandit | 1.0 / 1.0 / 1.0 | [0.008 (0.005–0.012)](../results/toy0b/rlcd_bandit.json) | [0.165 (0.161–0.166)](../results/toy0b/rlcd_bandit.json) | [0.009 (0.007–0.015)](../results/toy0b/rlcd_bandit.json) |
| RLCD-bandit, clip 10 | 1.3 / 1.3 / 1.3 | [0.010 (0.007–0.013)](../results/toy0b/rlcd_bandit_clip10.json) | [0.166 (0.165–0.167)](../results/toy0b/rlcd_bandit_clip10.json) | [0.007 (0.005–0.008)](../results/toy0b/rlcd_bandit_clip10.json) |
| Supervised log loss | 1.0 / 1.0 / 1.0 | [0.009 (0.006–0.011)](../results/toy0b/supervised.json) | [0.166 (0.164–0.167)](../results/toy0b/supervised.json) | [0.007 (0.005–0.010)](../results/toy0b/supervised.json) |
| RLCD-full | 1.0 / 1.0 / 1.0 | [0.009 (0.006–0.011)](../results/toy0b/rlcd_full.json) | [0.166 (0.164–0.167)](../results/toy0b/rlcd_full.json) | [0.007 (0.005–0.010)](../results/toy0b/rlcd_full.json) |
<!-- END gen:toy0b_group -->

That per-stratum temperature uses the known difficulty, which a real model does not have. The point is diagnosis,
not a recommended fix. This is one toy configuration and says nothing yet about a 9B model, but it is the reason
the plan pre-registers **H5b** (after temperature scaling, A3 has worse per-domain ECE and lower resolution than
A1) next to H5, which is stated on pooled ECE and could pass while the strata disagree.
