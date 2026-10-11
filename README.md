# open-rlcd

**An open, pre-registered test of RLCD (Reinforcement Learning for Calibrated Decisions).**

> **This is one reading of RLCD, not TypeSafe's method.** TypeSafe has not published RLCD. This repository tests the idea
> as it has been described, in the open, and reports the results that go against it too.

[![CI](https://github.com/Louiskq/open-rlcd/actions/workflows/ci.yml/badge.svg)](https://github.com/Louiskq/open-rlcd/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Louiskq/open-rlcd/blob/main/notebooks/toy_rlcd.ipynb)

**Live page: [rlcd.vercel.app](https://rlcd.vercel.app)**, the interactive toy and the findings on one page.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/toy_hero-dark.gif">
    <img alt="Two reliability diagrams animated through training. With the RLVR reward the model's confidence overshoots its accuracy; with the RLCD reward the points settle on the diagonal." src="figures/toy_hero.gif" width="640">
  </picture>
</p>
<p align="center"><sub>Same model, same data, same optimiser. Only the reward differs. (A toy where the true probabilities are known.)</sub></p>

## What this is

A model that's trained with RL to *be right* tends to become more confident than it deserves. The answers
stay as accurate, but its stated probabilities stop being trustworthy. RLCD is the idea of rewarding the model with a
**proper scoring rule applied to the probabilities it states over typed options** (true/false, pick-one, ordered scale)
instead. A proper rule pays best for stating the true probabilities.

With fixed labels, this is just supervised fine-tuning plus a temperature fitted afterwards. 

The experiments, planned and frozen before any full run ([`PREREGISTRATION.md`](PREREGISTRATION.md)), puts
RLCD against that sceptical baseline in three settings of increasing realism
- a toy
- a 9B language model trained on teacher-labelled questions,
- and a model playing Breakout

## What was found

- **The mechanism is real.** In the toy, RLVR's 0/1 reward makes the model overconfident at unchanged accuracy, and the
  importance-weighted RLCD reward does not.
- **In the 9B model, RLVR's overconfidence is real but modest** while a KL penalty ties it to its start, and RLCD is
  less affected. Remove the penalty and RLVR's probabilities collapse (one run).
- **But supervised fine-tuning plus one fitted temperature did as well as RLCD.** On the independent test set the arms are
  close after temperature scaling, so the sceptical reading is largely supported. The equivalence the plan asked for
  could not be established at its margin.
- **Choosing training questions by the model's uncertainty** was not shown to beat choosing at random.
- **In Breakout** (where outcomes depend on the model's own moves, the case RL is actually needed for), RLCD's joint
  training beat the sceptic's version by the pre-registered rule. The result is fragile: it rests on one seed, and a plain
  supervised fit is nearly as good.

<!-- BEGIN gen:headline -->
| setting | metric (lower is better) | RLVR / plain RL | RLCD | supervised control or sceptic's baseline |
|---|---|---|---|---|
| Toy (truth known) | ECE | [0.381 (0.380–0.382)](results/toy/rlvr.json) | **[0.011 (0.008–0.014)](results/toy/rlcd_bandit.json)** | **[0.009 (0.006–0.012)](results/toy/supervised.json)** |
| 9B model, dev set | ECE | [0.030 (0.024–0.034)](results/expA/analysis.json) | [0.021 (0.017–0.028)](results/expA/analysis.json) | **[0.015 (0.014–0.017)](results/expA/analysis.json)** |
| 9B model, dev set, after one temperature | ECE | **[0.021 (0.018–0.023)](results/expA/analysis.json)** | **[0.023 (0.020–0.026)](results/expA/analysis.json)** | **[0.019 (0.016–0.023)](results/expA/analysis.json)** |
| 9B model, harder split | ECE | [0.058 (0.055–0.061)](results/expA/analysis.json) | **[0.026 (0.020–0.031)](results/expA/analysis.json)** | **[0.027 (0.019–0.034)](results/expA/analysis.json)** |
| Breakout beliefs | log loss | [0.979 (0.635–1.536)](results/expB/analysis.json) | **[0.488 (0.467–0.524)](results/expB/analysis.json)** | [0.592 (0.432–0.900)](results/expB/analysis.json) |
<!-- END gen:headline -->

<sub>Seed mean (min&ndash;max). Bold: best, or within the best arm's seed range. Every number links to the file it came from.
Exp A is the pre-registered dev set unless stated.</sub>


RLCD fixes RLVR's overconfidence, but gave no clear advantage over supervised training plus temperature
scaling, except possibly in the on-policy Breakout case, where the evidence is weak.


## Read more

| | |
|---|---|
| [`docs/method.md`](docs/method.md) | what RLCD is, the estimator at its core, and the sceptic's reading the experiments test |
| [`docs/results.md`](docs/results.md) | the full results: every table, the 19 pre-registered tests, JevBench, the Breakout analysis |
| [`docs/toy.md`](docs/toy.md) | the two toy experiments (one exploratory) |
| [`docs/caveats.md`](docs/caveats.md) | what went against a hypothesis, and the known weak spots |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) · [`docs/decisions.md`](docs/decisions.md) | the frozen plan with its four dated amendments, and every design decision |
| [`docs/release.md`](docs/release.md) | what could be released as model weights, and what blocks it |

## Try it

| | |
|---|---|
| **Colab notebook**: the toy on a CPU, with a slider that mixes the two rewards | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Louiskq/open-rlcd/blob/main/notebooks/toy_rlcd.ipynb) |
| **`make toy`**: the same experiment as a script (about two minutes) | regenerates `results/toy/` and `figures/` |
| **Reliability explorer**: arm × answer type × split × temperature scaling, one self-contained file | open [`docs/explorer.html`](docs/explorer.html) in a browser |
| **Landing page**: the interactive toy and the findings on one page | live at [rlcd.vercel.app](https://rlcd.vercel.app); rebuild with `make site`, deploy notes in [`docs/site.md`](docs/site.md) |
| **`python -m rlcd.demo --model DIR --state "..." --question "..." --type choice --options a,b,c`**: a typed answer with a probability per option from a decider checkpoint | needs one GPU; no model is published yet |

## Reproduce

```bash
uv sync --extra dev        # Python 3.10+
make test                  # about 200 unit tests
make toy                   # toy experiment -> results/toy/, figures/
make expA-analysis         # the pre-registered Exp A tests, from the committed predictions (about 25 min, CPU)
make readme                # regenerate every table in this README and docs/ from results/
```

`make readme-check` fails if a page disagrees with `results/`; CI runs it. Training the models needs a GPU: the Slurm
scripts are in `slurm/` and the launchers in `scripts/`, assuming decider and the start checkpoint as laid out in
`slurm/env.sh`. The raw predictions of every Exp A run are committed in `results/expA/runs/`, so no GPU is needed to check
a number.

```
rlcd/        metrics and scoring rules · RLVR / RLCD estimators · the Exp A trainer and its analysis
             (compare.py runs the pre-registered tests) · the Breakout code · demo.py · figures
toy/         the toy experiments            tests/    estimator unbiasedness, metric correctness, splits, analyses
results/     one JSON per run + raw predictions; every table and figure is generated from these
docs/        the long-form pages above · explorer.html · analysis and compute notes
```

## Citation, credit, licence

```bibtex
@software{kearney2026openrlcd,
  author = {Kearney, Louis},
  title  = {open-rlcd: an open reading of Reinforcement Learning for Calibrated Decisions},
  year   = {2026},
  url    = {https://github.com/Louiskq/open-rlcd}
}
```

- The decider recipe, data conventions, serving code and evaluation suite this builds on are
  [Mapika's decider](https://github.com/Mapika/decider) (Apache 2.0). The metric conventions and the per-answer-type
  temperature scaling follow its code.
- The term "RLCD" and JevBench are TypeSafe's. This repository is independent of them and does not speak for them.
- No JevBench item is used for training or for choosing training data.

Apache 2.0, see [`LICENSE`](LICENSE).
