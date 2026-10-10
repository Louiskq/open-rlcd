# Release plan (nothing has been uploaded)

Everything the numbers on the README and the results page rest on is **already in this repository**: the raw predictions of every Exp A run
(`results/expA/runs/`, 2.4 MB), the analysis outputs (`results/expA/analysis*.json`, `results/expB/`), the figures and
`docs/explorer.html`. `make expA-analysis`, `make expA-figures` and `make readme` regenerate the tables and figures from
them without a GPU. What follows is only about model weights, which are too large for git.

## What exists on the cluster (not in the repo)

| artifact | size | notes |
|---|---|---|
| Exp A LoRA adapters, one per run (`lora_adapter.pt`, rank 64, fp32) | about 640 MB each, 21 runs | apply to the Stage 1 SFT checkpoint `decider9b-sft` |
| the start model `decider9b-sft` | about 18 GB | a Qwen3.5-9B decider after its supervised stage, trained in an earlier project; **not published** |
| Exp B checkpoints (full weights, selected and final per run) | about 18 GB each | B1 / B3 / B5; B4's fitted copies are deleted after their answers are saved |
| JevBench summaries per Exp A model | small | `$PROJ/rlcd_expA/jevbench/<run>/summary-*.json` |

## What a release could contain, and what blocks it

1. **Adapters for one seed per arm** (A1, A3, A4), published as LoRA adapters plus a model card with the training recipe
   and the figures. *Blocker:* an adapter is only usable with its base model, and `decider9b-sft` has not been published;
   either publish it (licence and provenance to be stated: Qwen3.5-9B, decider's recipe, Apache 2.0) or publish merged
   weights (about 18 GB per model).
2. **A merged A1 or A4 model** so that `python -m rlcd.demo --model DIR ...` works from a download. Same blocker.
3. **Breakout B3 seed 0** (the model that was trained with beliefs), with the Breakout wrapper from decider.
4. **The questions.** The Exp A training and dev questions were written by a teacher model; whether they can be
   redistributed depends on that model's terms. The repository releases predictions only (question ids, gold labels, the
   model's readout), not the question text.

JevBench items are not used for training or for choosing training data and are not redistributed here.

## Commands for whoever uploads (needs a Hugging Face token; run by the maintainer, not by the experiments)

```bash
# 1. merge an adapter into the start model (GPU, decider's venv); writes a decider checkpoint with decider_config.json
python scripts/merge_expA_adapter.py --adapter RUN/lora_adapter.pt --out MERGED_DIR --temperature 1.0
# 2. check it answers
python -m rlcd.demo --model MERGED_DIR --state "..." --question "..." --type choice --options a,b,c
# 3. upload
huggingface-cli upload YOUR_ORG/open-rlcd-a1 MERGED_DIR .
```

Decide first whether the repository goes public (it is private now; the commit-email history needs to be recreated first,
see `docs/decisions.md` D17 and D19) and which of items 1 to 4 to publish.
