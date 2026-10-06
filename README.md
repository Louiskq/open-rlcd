# open-rlcd

**RLCD, reverse-engineered: why rewarding a model for being right makes it overconfident, and the one-line fix.**

> **This repo presents one reading of RLCD, not TypeSafe's method.** TypeSafe has not published RLCD. What
> follows is an open, pre-registered attempt to test whether the idea, as described, works, with the results
> that go against it reported too.
> *Name collision:* "RLCD" is also the name of Reinforcement Learning from Contrastive Distillation
> (Yang et al., [arXiv:2307.12950](https://arxiv.org/abs/2307.12950)), an unrelated method.

[![CI](https://github.com/Louiskq/open-rlcd/actions/workflows/ci.yml/badge.svg)](https://github.com/Louiskq/open-rlcd/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Louiskq/open-rlcd/blob/main/notebooks/toy_rlcd.ipynb)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/toy_hero-dark.gif">
    <img alt="Two reliability diagrams animated through training. With the RLVR reward the model's confidence overshoots its accuracy; with the RLCD reward the points settle on the diagonal." src="figures/toy_hero.gif" width="640">
  </picture>
</p>
<p align="center"><sub>Same model, same data, same optimiser. Only the reward differs. (Exp 0: a toy where the true probabilities are known.)</sub></p>

> **Status: phase 1 of 5 done.** The metrics, the estimators (with unbiasedness tests), the toy experiments and
> the recorded design [decisions](docs/decisions.md) are in place. Exp A (a 9B model, teacher labels) and Exp B
> (Breakout) have **not been run**; their rows say *pending* rather than guess. See [Status](#status).

## TL;DR

- **The idea.** Reward the model with a *proper scoring rule* applied to the distribution it states over typed
  options, instead of rewarding it for being right.
- **The one-line fix.** When only the sampled answer `a` can be checked, use reward `1[a=y] / p(a)` (no
  gradient through the denominator) instead of `1[a=y]`. That makes REINFORCE an unbiased estimator of the
  log-likelihood gradient, so calibration is recoverable from bandit feedback.
- **What is shown so far (toy only).** The 0/1 reward makes the model more confident than accurate while
  leaving accuracy unchanged; the importance-weighted reward does not.

<!-- BEGIN gen:tldr -->
| setting | arm | ECE (lower is better) | accuracy | log loss |
|---|---|---|---|---|
| Exp 0 (toy) | RLVR | [0.381 (0.380–0.382)](results/toy/rlvr.json) | [0.499 (0.498–0.500)](results/toy/rlvr.json) | [3.047 (3.037–3.057)](results/toy/rlvr.json) |
| Exp 0 (toy) | RLCD-bandit | **[0.011 (0.008–0.014)](results/toy/rlcd_bandit.json)** | [0.498 (0.496–0.501)](results/toy/rlcd_bandit.json) | [1.221 (1.219–1.225)](results/toy/rlcd_bandit.json) |
| Exp 0 (toy) | RLCD-bandit, clip 10 | [0.060 (0.056–0.066)](results/toy/rlcd_bandit_clip10.json) | [0.499 (0.496–0.502)](results/toy/rlcd_bandit_clip10.json) | [1.241 (1.238–1.244)](results/toy/rlcd_bandit_clip10.json) |
| Exp 0 (toy) | Supervised log loss (= RLCD-full) | **[0.009 (0.006–0.012)](results/toy/supervised.json)** | [0.498 (0.495–0.500)](results/toy/supervised.json) | [1.220 (1.218–1.222)](results/toy/supervised.json) |
| Exp A (9B decider, teacher labels) | A1–A7 | _pending_ | _pending_ | _pending_ |
| Exp B (Breakout, on-policy) | B1–B6 | _pending_ | _pending_ | _pending_ |
<!-- END gen:tldr -->

<sub>Seed mean (min–max). Bold: best, or within the best arm's seed range. Every number links to the JSON it
came from. Rows marked *pending* have not been run.</sub>

## What is RLCD?

RLCD is reinforcement learning whose reward is a **proper scoring rule applied to the model's stated
distribution over typed options**, judged against an outcome supplied by labels, a verifier, a teacher model or
an environment:

$$
\max_\theta\; \mathbb{E}_{s,\,q,\; y}\Big[\, S\big(p_\theta(\cdot\mid s,q),\, y\big) \Big] \;-\; \beta\,\mathrm{KL}\big(p_{\text{ref}} \,\|\, p_\theta\big)
$$

A question is *typed*: `NOUL` (true or false), `CHOICE` (one of several unordered options) or `SCORE` (an
ordered scale). The model gives a probability for every option in one forward pass. $S$ is the log score for
`NOUL` and `CHOICE`, and the negative ranked probability score (RPS) for `SCORE`. The penalty is the
*forward* KL, the direction decider's retention term uses. It is large when the policy puts little mass on an
option the reference supports, so it penalises collapse onto one option and partly counteracts RLVR's
overconfidence; one exploratory run without it will show by how much ([`docs/decisions.md`](docs/decisions.md) D1).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/concept-dark.png">
    <img alt="State and typed questions go through one forward pass to a distribution per question, which is scored against an outcome by a proper scoring rule; the score drives gradient ascent." src="figures/concept.png" width="860">
  </picture>
</p>

**The contrast case is RLVR**, where the reward is $\mathbb{1}[a=y]$ for a sampled answer $a\sim p_\theta$.
Its expected value, $\sum_a p(a)\,q(a)$, is maximised by putting all the probability on the most likely answer.
It is an *improper* score, so it pushes the model to be sharper than the outcomes justify.

**Bandit form.** Sometimes only the sampled answer can be checked. With reward
$r = \mathbb{1}[a=y]/p_\theta(a)$ (stop-gradient on the denominator), REINFORCE is an unbiased estimator of
$\nabla\log p_\theta(y)$:

$$
\mathbb{E}_{a\sim p}\Big[\tfrac{\mathbb{1}[a=y]}{p(a)}\,\nabla\log p(a)\Big] \;=\; \sum_a p(a)\,\tfrac{\mathbb{1}[a=y]}{p(a)}\,\nabla\log p(a) \;=\; \nabla\log p(y).
$$

[`tests/test_estimators.py`](tests/test_estimators.py) checks this by exact enumeration, by Monte Carlo, and by
showing the same check *rejects* RLVR. Clipping the weight trades that unbiasedness for lower variance; the
tests pin the bias exactly (`min(1, c·p_y)` times the true gradient).

**When is RL needed at all?** With fixed labels, the full-distribution RLCD objective has the same gradient as
supervised log loss. RL is necessary only when (a) feedback is bandit-only, or (b) outcomes depend on the
model's own actions (on-policy environments). The experiments are built around that claim: Exp A tests the
label setting, Exp B the on-policy one.

## Is RLCD just a data engine?

A sceptic can read RLCD as a **branded proper-scoring fine-tune with an on-policy data loop, and no RL in it**.
This repo tests that reading directly instead of arguing about it:

- **Exp A, A6 ("cynical RLCD")**: mine the questions the current model is least sure about, take their labels
  (which arrive with generation and stay hidden until after selection), fine-tune with plain log loss (RPS for
  `SCORE`), repeat. **A6r** is the same with random items inside the same per-type quotas. A6 sees more data than
  A1–A5, but it gets the **same total optimiser steps** (a quarter per round, continuing from the previous round's
  weights), so any lead over A1 is data, not extra training; *A6 against A6r* is the test of mining itself.
- **Exp B, B4 ("cynical")**: take B1's final policy, fit the belief heads by supervised log loss on that
  policy's rollouts, then temperature-scale. B3 trains the beliefs jointly and on-policy.

Pre-registered predictions: **H4** (A6 is no better than A6r beyond seed noise) and **H7** (B3 beats B4 on belief
log loss at a matched game score; if not, the cynical reading wins in this setting too).

**Result: pending** (phases 3 and 4). It will be reported here with the same prominence whichever way it goes.

## Try it

| | |
|---|---|
| **Colab notebook**, runs the toy on a CPU and has a slider that mixes the two rewards | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Louiskq/open-rlcd/blob/main/notebooks/toy_rlcd.ipynb) |
| **`make toy`**, the same experiment as a script; regenerates `results/toy/` and `figures/` | available |
| **`python -m rlcd.demo --state "..." --question "..."`**, a typed answer with a probability per option from the released model | *pending* (needs the phase 3 model release) |
| **Interactive reliability-diagram explorer** (arm × answer type × temperature scaling on/off) | *pending* (needs Exp A predictions) |

## Exp 0 results: a toy where the truth is known

<!-- BEGIN gen:toy_setup -->
10 seeds × 4,000 Adam steps per arm, evaluated on 20,000 held-out contexts. The true probabilities themselves score ECE [0.010](results/toy/meta.json) on this sample (the noise floor) and the Bayes-optimal accuracy is [0.500](results/toy/meta.json).
<!-- END gen:toy_setup -->

A linear-softmax policy over 5 options, labels drawn from a fixed ground-truth softmax, so the policy class
contains the truth and "calibrated" has an exact meaning. Every arm has the same model, optimiser, learning rate
and data stream per seed; only the reward differs.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/toy_overview-dark.png">
    <img alt="ECE and accuracy against training step for four arms, and a reliability diagram per arm. The RLVR curve falls far below the diagonal while the RLCD and supervised curves sit on it." src="figures/toy_overview.png" width="900">
  </picture>
</p>

<!-- BEGIN gen:toy_table -->
| arm | ECE | accuracy | log loss | KL(truth ‖ model) | ECE after temperature scaling |
|---|---|---|---|---|---|
| RLVR | [0.381 (0.380–0.382)](results/toy/rlvr.json) | [0.499 (0.498–0.500)](results/toy/rlvr.json) | [3.047 (3.037–3.057)](results/toy/rlvr.json) | [1.837 (1.827–1.846)](results/toy/rlvr.json) | **[0.010 (0.008–0.012)](results/toy/rlvr.json) (T = 6.1)** |
| RLCD-bandit | **[0.011 (0.008–0.014)](results/toy/rlcd_bandit.json)** | [0.498 (0.496–0.501)](results/toy/rlcd_bandit.json) | **[1.221 (1.219–1.225)](results/toy/rlcd_bandit.json)** | **[0.006 (0.004–0.010)](results/toy/rlcd_bandit.json)** | **[0.010 (0.008–0.013)](results/toy/rlcd_bandit.json) (T = 1.0)** |
| RLCD-bandit, clip 10 | [0.060 (0.056–0.066)](results/toy/rlcd_bandit_clip10.json) | [0.499 (0.496–0.502)](results/toy/rlcd_bandit_clip10.json) | [1.241 (1.238–1.244)](results/toy/rlcd_bandit_clip10.json) | [0.026 (0.022–0.029)](results/toy/rlcd_bandit_clip10.json) | **[0.010 (0.009–0.012)](results/toy/rlcd_bandit_clip10.json) (T = 1.3)** |
| Supervised log loss | **[0.009 (0.006–0.012)](results/toy/supervised.json)** | [0.498 (0.495–0.500)](results/toy/supervised.json) | **[1.220 (1.218–1.222)](results/toy/supervised.json)** | **[0.004 (0.003–0.007)](results/toy/supervised.json)** | **[0.009 (0.005–0.012)](results/toy/supervised.json) (T = 1.0)** |
| RLCD-full | **[0.009 (0.006–0.012)](results/toy/rlcd_full.json)** | [0.498 (0.495–0.500)](results/toy/rlcd_full.json) | **[1.220 (1.218–1.222)](results/toy/rlcd_full.json)** | **[0.004 (0.003–0.007)](results/toy/rlcd_full.json)** | **[0.009 (0.005–0.012)](results/toy/rlcd_full.json) (T = 1.0)** |
| _the true probabilities_ | [0.010](results/toy/meta.json) | [0.500](results/toy/meta.json) | [1.215](results/toy/meta.json) | 0 | |

Seed mean (min–max). Bold: best, or within the best arm's seed range. Supervised log loss and RLCD-full are the same objective through two code paths; their curves differ by at most [4.4e-16](results/toy/meta.json) over every metric, seed and step.
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
| RLVR | [0.0435](results/toy/meta.json) | 0.0015 | 29.4× | [0.007](results/toy/meta.json) |
| RLCD-bandit | [0.0062](results/toy/meta.json) | 0.0059 | 1.0× | [0.106](results/toy/meta.json) |
| RLCD-bandit, clip 10 | [0.0250](results/toy/meta.json) | 0.0050 | 5.0× | [0.074](results/toy/meta.json) |
| Supervised log loss | [0.0044](results/toy/meta.json) | 0.0050 | 0.9× | [0.074](results/toy/meta.json) |
| RLCD-full | [0.0044](results/toy/meta.json) | 0.0050 | 0.9× | [0.074](results/toy/meta.json) |
<!-- END gen:toy_noise -->

The importance-weighted reward is unbiased but noisier than supervised log loss. Clipping removes much of that
extra noise at the price of a bias well above the Monte Carlo floor. RLVR has a clearly nonzero gradient at the
truth, which is the formal sense in which it is improper.

### How much RLVR does it take?

`reward = (1 − m)·1[a=y]/p(a) + m·1[a=y]`: `m = 0` is RLCD-bandit, `m = 1` is RLVR.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/toy_mix-dark.png">
    <img alt="ECE and log loss against the share of RLVR mixed into the reward: flat near the noise floor for small shares, then rising steeply." src="figures/toy_mix.png" width="760">
  </picture>
</p>

<!-- BEGIN gen:toy_mix -->
| share of RLVR in the reward | ECE | ECE after temperature scaling | log loss |
|---|---|---|---|
| 0% | [0.011 (0.010–0.013)](results/toy/mix_sweep.json) | [0.010 (0.009–0.013)](results/toy/mix_sweep.json) | [1.220 (1.219–1.222)](results/toy/mix_sweep.json) |
| 2% | [0.011 (0.010–0.014)](results/toy/mix_sweep.json) | [0.009 (0.008–0.010)](results/toy/mix_sweep.json) | [1.221 (1.220–1.222)](results/toy/mix_sweep.json) |
| 5% | [0.010 (0.008–0.011)](results/toy/mix_sweep.json) | [0.010 (0.009–0.012)](results/toy/mix_sweep.json) | [1.220 (1.219–1.221)](results/toy/mix_sweep.json) |
| 10% | [0.011 (0.009–0.015)](results/toy/mix_sweep.json) | [0.010 (0.008–0.011)](results/toy/mix_sweep.json) | [1.220 (1.219–1.222)](results/toy/mix_sweep.json) |
| 20% | [0.015 (0.008–0.022)](results/toy/mix_sweep.json) | [0.009 (0.006–0.010)](results/toy/mix_sweep.json) | [1.221 (1.220–1.222)](results/toy/mix_sweep.json) |
| 35% | [0.023 (0.011–0.030)](results/toy/mix_sweep.json) | [0.010 (0.007–0.012)](results/toy/mix_sweep.json) | [1.223 (1.222–1.224)](results/toy/mix_sweep.json) |
| 50% | [0.049 (0.043–0.056)](results/toy/mix_sweep.json) | [0.011 (0.009–0.012)](results/toy/mix_sweep.json) | [1.231 (1.228–1.235)](results/toy/mix_sweep.json) |
| 75% | [0.115 (0.110–0.118)](results/toy/mix_sweep.json) | [0.008 (0.007–0.010)](results/toy/mix_sweep.json) | [1.286 (1.281–1.289)](results/toy/mix_sweep.json) |
| 100% | [0.382 (0.381–0.382)](results/toy/mix_sweep.json) | [0.011 (0.011–0.012)](results/toy/mix_sweep.json) | [3.052 (3.049–3.057)](results/toy/mix_sweep.json) |
<!-- END gen:toy_mix -->

The damage grows gradually with the share. It is not a cliff, and a small share is hard to tell from the noise.

### Does the ordering depend on the learning rate?

Final ECE under several learning rates, so the comparison is not an artefact of one setting:

<!-- BEGIN gen:toy_lr -->
3 seeds per cell. Seed mean (min–max).

| arm (final ECE) | lr 0.003 | lr 0.01 | lr 0.03 |
|---|---|---|---|
| RLVR | [0.325 (0.325–0.326)](results/toy/lr_sensitivity.json) | [0.382 (0.381–0.382)](results/toy/lr_sensitivity.json) | [0.421 (0.420–0.423)](results/toy/lr_sensitivity.json) |
| RLCD-bandit | [0.008 (0.008–0.009)](results/toy/lr_sensitivity.json) | [0.011 (0.010–0.013)](results/toy/lr_sensitivity.json) | [0.017 (0.009–0.021)](results/toy/lr_sensitivity.json) |
| RLCD-bandit, clip 10 | [0.056 (0.055–0.058)](results/toy/lr_sensitivity.json) | [0.059 (0.058–0.060)](results/toy/lr_sensitivity.json) | [0.079 (0.069–0.087)](results/toy/lr_sensitivity.json) |
| Supervised log loss | [0.010 (0.008–0.012)](results/toy/lr_sensitivity.json) | [0.009 (0.006–0.012)](results/toy/lr_sensitivity.json) | [0.016 (0.011–0.023)](results/toy/lr_sensitivity.json) |
| RLCD-full | [0.010 (0.008–0.012)](results/toy/lr_sensitivity.json) | [0.009 (0.006–0.012)](results/toy/lr_sensitivity.json) | [0.016 (0.011–0.023)](results/toy/lr_sensitivity.json) |
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
    <source media="(prefers-color-scheme: dark)" srcset="figures/toy0b-dark.png">
    <img alt="Three panels after one temperature. Pooled ECE is at the noise floor for every arm. Confidence minus accuracy by difficulty tercile is large and of opposite signs for RLVR, near zero for the RLCD arms. RLVR has lower Murphy resolution." src="figures/toy0b.png" width="900">
  </picture>
</p>

<!-- BEGIN gen:toy0b_table -->
| arm | ECE | ECE after one temperature | resolution after | KL(truth ‖ model) after | confidence − accuracy after, hard / medium / easy |
|---|---|---|---|---|---|
| RLVR | [0.405 (0.403–0.407)](results/toy0b/rlvr.json) | [0.010 (0.007–0.012)](results/toy0b/rlvr.json) (T = 7.2) | [0.158 (0.157–0.159)](results/toy0b/rlvr.json) | [0.030 (0.029–0.031)](results/toy0b/rlvr.json) | [+0.093](results/toy0b/rlvr.json) / [−0.018](results/toy0b/rlvr.json) / [−0.087](results/toy0b/rlvr.json) |
| RLCD-bandit | [0.010 (0.006–0.018)](results/toy0b/rlcd_bandit.json) | [0.008 (0.005–0.012)](results/toy0b/rlcd_bandit.json) (T = 1.0) | [0.165 (0.161–0.167)](results/toy0b/rlcd_bandit.json) | [0.010 (0.007–0.015)](results/toy0b/rlcd_bandit.json) | [−0.000](results/toy0b/rlcd_bandit.json) / [−0.002](results/toy0b/rlcd_bandit.json) / [−0.011](results/toy0b/rlcd_bandit.json) |
| RLCD-bandit, clip 10 | [0.050 (0.042–0.064)](results/toy0b/rlcd_bandit_clip10.json) | [0.010 (0.007–0.014)](results/toy0b/rlcd_bandit_clip10.json) (T = 1.3) | [0.166 (0.165–0.167)](results/toy0b/rlcd_bandit_clip10.json) | [0.007 (0.005–0.008)](results/toy0b/rlcd_bandit_clip10.json) | [−0.010](results/toy0b/rlcd_bandit_clip10.json) / [−0.004](results/toy0b/rlcd_bandit_clip10.json) / [−0.011](results/toy0b/rlcd_bandit_clip10.json) |
| Supervised log loss | [0.009 (0.006–0.015)](results/toy0b/supervised.json) | [0.009 (0.007–0.013)](results/toy0b/supervised.json) (T = 1.0) | [0.166 (0.164–0.167)](results/toy0b/supervised.json) | [0.007 (0.005–0.010)](results/toy0b/supervised.json) | [+0.000](results/toy0b/supervised.json) / [−0.003](results/toy0b/supervised.json) / [−0.013](results/toy0b/supervised.json) |
| RLCD-full | [0.009 (0.006–0.015)](results/toy0b/rlcd_full.json) | [0.009 (0.007–0.013)](results/toy0b/rlcd_full.json) (T = 1.0) | [0.166 (0.164–0.167)](results/toy0b/rlcd_full.json) | [0.007 (0.005–0.010)](results/toy0b/rlcd_full.json) | [+0.000](results/toy0b/rlcd_full.json) / [−0.003](results/toy0b/rlcd_full.json) / [−0.013](results/toy0b/rlcd_full.json) |
| _the true probabilities_ | [0.006](results/toy0b/meta.json) | | [0.169](results/toy0b/meta.json) | 0 | |
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
| RLVR | 11.5 / 6.7 / 5.3 | [0.007 (0.005–0.009)](results/toy0b/rlvr.json) | [0.166 (0.165–0.167)](results/toy0b/rlvr.json) | [0.009 (0.008–0.009)](results/toy0b/rlvr.json) |
| RLCD-bandit | 1.0 / 1.0 / 1.0 | [0.008 (0.005–0.012)](results/toy0b/rlcd_bandit.json) | [0.165 (0.161–0.166)](results/toy0b/rlcd_bandit.json) | [0.009 (0.007–0.015)](results/toy0b/rlcd_bandit.json) |
| RLCD-bandit, clip 10 | 1.3 / 1.3 / 1.3 | [0.010 (0.007–0.013)](results/toy0b/rlcd_bandit_clip10.json) | [0.166 (0.165–0.167)](results/toy0b/rlcd_bandit_clip10.json) | [0.007 (0.005–0.008)](results/toy0b/rlcd_bandit_clip10.json) |
| Supervised log loss | 1.0 / 1.0 / 1.0 | [0.009 (0.006–0.011)](results/toy0b/supervised.json) | [0.166 (0.164–0.167)](results/toy0b/supervised.json) | [0.007 (0.005–0.010)](results/toy0b/supervised.json) |
| RLCD-full | 1.0 / 1.0 / 1.0 | [0.009 (0.006–0.011)](results/toy0b/rlcd_full.json) | [0.166 (0.164–0.167)](results/toy0b/rlcd_full.json) | [0.007 (0.005–0.010)](results/toy0b/rlcd_full.json) |
<!-- END gen:toy0b_group -->

That per-stratum temperature uses the known difficulty, which a real model does not have. The point is diagnosis,
not a recommended fix. This is one toy configuration and says nothing yet about a 9B model, but it is the reason
the plan pre-registers **H5b** (after temperature scaling, A3 has worse per-domain ECE and lower resolution than
A1) next to H5, which is stated on pooled ECE and could pass while the strata disagree.

## Honest caveats

Written *before* the language-model results exist, so they cannot be shaped by them.

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
- **Decisions that change what the experiments mean** are recorded in [`docs/decisions.md`](docs/decisions.md):
  the KL direction (forward), "raw" meaning temperature 1 (the checkpoint's stored 1.30 is itself a post-hoc
  calibration, so it appears only as a separate "as served" reference row), how `SCORE` answers are read out
  (decider's one yes/no pass per level, normalised into a distribution), and RPS normalisation (which gives
  `SCORE` questions less gradient than their share of the pool).
- **Seeds share candidate data.** The cynical arms (A6, A6r) draw their candidate questions from one pool per
  round that is shared by both arms and by all seeds; selection still differs per seed. Their seed spread therefore
  understates data variance.
- **Planned-experiment limitations, stated now:** the independent dev set was written by the same teacher as the
  training data; JevBench hard has 111 items (standard error about 4.4 points), too few to separate the arms.
- **Results that went against a hypothesis** will be listed here by name. None yet, because no hypothesis has
  been tested yet.

## Status

| phase | what | state |
|---|---|---|
| 1 | scaffold, metrics, estimators, tests, Exp 0 (+ exploratory Exp 0b), notebook, README skeleton, loss-weight check, design decisions | **done** |
| 2 | Exp A smoke run (all arms, 1 seed, 10% of steps); Breakout label validation | not started |
| 3 | commit the pre-registration, then Exp A in full | not started |
| 4 | Exp B in full | not started |
| 5 | write-up, releases (Hugging Face), interactive explorer | not started |

## Reproduce

```bash
uv sync --extra dev        # Python 3.10+; installs the package and dev tools
make test                  # unit tests (metrics, estimators, splits, toy invariants)
make lint
make toy                   # Exp 0 -> results/toy/*.json, figures/
make toy0b                 # Exp 0b (exploratory) -> results/toy0b/*.json, figures/
make readme                # regenerate every table in this file from results/
make notebook              # rebuild and execute notebooks/toy_rlcd.ipynb
```

`make readme-check` fails if this file disagrees with `results/`; CI runs it. Exp A / Exp B configs and Slurm
scripts arrive with phases 2–4 (`configs/`, `slurm/`).

```
rlcd/        scoring.py (metrics, scoring rules) · estimators.py (RLVR / RLCD rewards and gradients)
             calibrate.py (temperature scaling) · data.py (splits, leakage checks) · isolated.py (SCORE readout)
             mining.py (A6 / A6r selection) · plots.py · results.py
toy/         Exp 0 and Exp 0b (exploratory)
tests/       estimator unbiasedness, metric correctness, split disjointness, toy invariants
results/     one JSON per run; every table and figure is generated from these
docs/        decisions.md (amendments to the spec) · loss_weights.md (the Breakout weight verification)
```

## Citation, acknowledgements, licence

```bibtex
@software{kearney2026openrlcd,
  author = {Kearney, Louis},
  title  = {open-rlcd: an open reading of Reinforcement Learning for Calibrated Decisions},
  year   = {2026},
  url    = {https://github.com/Louiskq/open-rlcd}
}
```

- The decider recipe, data conventions, serving code and evaluation suite this builds on are
  [Mapika's decider](https://github.com/Mapika/decider) (Apache 2.0). The metric conventions and the
  per-answer-type temperature scaling follow its code.
- The term "RLCD" and JevBench are TypeSafe's. This repository is independent of them and does not
  speak for them.
- No JevBench item is used for training or for choosing training data.

Apache 2.0, see [`LICENSE`](LICENSE).
