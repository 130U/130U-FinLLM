# 130U-FinLLM

**A compact financial language-model training system, from random initialization to calibrated, asset-aware inference.**

130U-FinLLM brings model architecture, corpus preparation, pretraining, supervised learning, preference training, and evaluation into one native PyTorch project. Its task is precise: given a short English text and a target asset, produce **bearish / neutral / bullish** stance, class probabilities, a direction score, and review flags.

The project centers on the engineering behind a financial NLP model: defining the learning objective, keeping evaluation data separate, preserving complete training state, and connecting predictions to the exact tokenizer and checkpoint that produced them.

[Model implementation](src/finllm/model.py) · [Training engine](src/finllm/training.py) · [Saved engineering results](reports/) · [Data preparation record](docs/SEED_DATA.md)

## Project contributions

- **Asset-conditioned learning.** Text and target are encoded together. Full-parameter SFT and DPO supervise the answer-label position, with a frozen SFT reference for preference training. This gives the same sentence a separate prediction for each asset.
- **Data preparation with evaluation exclusions.** Document-aware partitioning, exact and lexical near-duplicate checks, and exclusions before vocabulary training connect raw sources to versioned, memory-mapped token shards.
- **Resumable training as a complete state machine.** Checkpoints preserve weights, optimizer, scheduler, scaler, random state, and data position. Hashes and completion markers support recovery; completed-budget reruns preserve existing artifacts.
- **Evaluation beyond the predicted label.** Separate temperature-fitting and review-threshold partitions feed macro-F1, log loss, Brier score, calibration error, and coverage reporting. Calibration artifacts bind to checkpoint and tokenizer hashes.

The implemented decoder uses **GQA, RoPE, Q/K RMSNorm, SwiGLU, and tied embeddings**. Its architecture is adapted from a [pinned MiniMind revision](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f); the project adds financial objectives, data controls, recovery, evaluation, and notebook execution.

## Recorded engineering results

| Deliverable | Recorded result | Evidence |
| --- | --- | --- |
| Model specification and execution | **24,255,744 parameters**; six layers, 512-token context; full-size CPU forward/backward and AdamW update verified | [Configuration](configs/model_24m.json) · [CPU check](reports/full_config_cpu_check.json) |
| Prepared seed corpus | **4,976,894 general + 341,309 financial training tokens**, with validation text stored separately | [Preparation report](reports/seed_data_preparation.json) |
| Self-trained vocabulary and exclusions | **8,192-token byte-level BPE**; **10,686 distinct candidate headlines excluded** before vocabulary training | [Preparation report](reports/seed_data_preparation.json) |
| Local notebook lifecycle | **120 tests passed; nine notebooks validated**, covering active-kernel imports, training, backup, rerun, and fresh-kernel restoration | [Recovery report](reports/colab_kernel_fix.json) |

These are architecture, data-preparation, and engineering-validation results. The repository provides the implemented system and its execution evidence.

## Run the system

For a small CPU exercise, use Python 3.10+ and PyTorch 2.5+:

```bash
git clone https://github.com/130U/130U-FinLLM.git
cd 130U-FinLLM
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -e ".[test]"
python -m finllm smoke --output runs/showcase-smoke --device cpu
```

The synthetic smoke exercise runs all four training stages, restores a checkpoint, fits calibration, and writes predictions, metrics, and `summary.json`. Use a fresh output directory for another run.

For the notebook path, start with [00_setup_and_smoke](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/00_setup_and_smoke.ipynb), then follow [seed preparation](notebooks/01a_seed_data.ipynb) → [24M pilot](notebooks/02a_seed_pilot.ipynb). The [notebooks](notebooks/) cover pretraining, financial domain training, SFT, DPO, and evaluation; [configs](configs/) specify each run.

## Prediction interface

```json
{"text": "Long $AAPL, short $TSLA.", "target": "AAPL"}
```

The output includes three probabilities, confidence, and `sentiment_score = 100 × (P(bullish) − P(bearish))`. The score expresses **textual stance toward the specified asset**. Input checks and configurable confidence thresholds produce explicit review reasons.

Inspect the [data guide](docs/DATA_GUIDE.md), [model card](docs/MODEL_CARD.md), [evaluation code](src/finllm/evaluation.py), and [tests](tests/) for the complete contract.

## Attribution and license

Code: [Apache License 2.0](LICENSE). MiniMind attribution and the project-specific changes are documented in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md). External data retains its source-specific terms.

