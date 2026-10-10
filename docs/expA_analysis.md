# Exp A: from predictions to the results page

Everything in the Exp A section of [results.md](results.md) is computed from `results/expA/runs/<run>/preds_{cal,select,dev}.npz`,
the readout of every question at temperature 1 that `rlcd.train_labels` saves after each run. No GPU is needed.

| step | command | writes |
|---|---|---|
| per-arm tables, the 19 pre-registered tests, the seed-ensemble robustness check | `make expA-analysis` (about 25 min) | `results/expA/analysis.json` |
| the same tests on the selection split (exploratory) | in `make expA-analysis` | `results/expA/analysis_select_prereg_tests.json` |
| per-answer-type differences (exploratory) | in `make expA-analysis` | `results/expA/analysis_exploratory_{dev,select}.json` |
| figures | `make expA-figures` | `figures/expA_*.{png,svg}` (light and dark) |
| tables | `make readme` | the generated blocks in `README.md`, `docs/results.md` and `docs/toy.md` |

What is computed, in `rlcd/evaluate.py` and `rlcd/compare.py`:

- **Probabilities.** `NOUL` and `CHOICE`: softmax of the stored readout. `SCORE`: one yes/no pass per level, the yes
  probabilities normalised into one distribution (D3). *Raw* is T = 1. *After TS* uses one temperature per answer type,
  fitted on the calibration split of the same run. *As served* (A0 only) is the checkpoint's stored 1.05.
- **Pre-registered tests.** A paired bootstrap over dev items (2,000 replicates, seed 20261008), one resample shared by
  every arm, seed and row; the metric is computed per seed and averaged over seeds. A difference test passes when the
  95% interval excludes 0 in the claimed direction, an equivalence test when it lies inside the margin δ
  (`results/expA/delta.json`, from the A1 seed pair on the selection split), both after Holm correction. Amendment 1 of
  `PREREGISTRATION.md` has the exact definitions and the list of 19 tests.
- **Learning rates.** `results/expA/runs/sweep_a1.json` and `sweep_a4.json`: lowest selection-split log loss over three
  rates. Every arm ended up at 1e-4.

Things to know when reading the numbers:

1. The pooled log loss mixes answer types trained with different objectives (D20). Read NOUL and CHOICE separately.
2. The dev set was written by the same teacher as the training data and has four domains; per-domain results rest on
   few questions per domain.
3. A5, A0 and the β = 0 run are single runs, so they show no seed range.
4. The A6 / A6r seeds share the held-back half of the pool, so their seed spread understates data variance.
