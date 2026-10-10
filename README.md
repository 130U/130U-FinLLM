# 130U-FinLLM

Train and evaluate a compact language model for asset-specific financial stance, starting from random weights.

Given a short English text and a target asset, the model is designed to classify **bearish / neutral / bullish** stance and return class probabilities, a direction score, and review flags. The training path covers pretraining, financial domain pretraining, full-parameter SFT, a DPO experiment, and calibration.

**Current status: a runnable training and evaluation pipeline; 24M training and real financial evaluation remain pending.** See [STATUS](docs/STATUS.md) and the [planned model card](docs/MODEL_CARD.md).

## Start with a small run

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/00_setup_and_smoke.ipynb)

Open **00_setup_and_smoke**, select a T4 runtime, and run the cells in order. For persistent backups, set `MOUNT_DRIVE = True` and authorize Drive. Setup pins a code commit and preserves Colab's CUDA PyTorch. `/content` is temporary; see the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

For a local CPU check, use Python 3.10+ and PyTorch 2.5+:

```bash
git clone https://github.com/130U/130U-FinLLM.git
cd 130U-FinLLM
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -e ".[test]"
python -m finllm environment
python -m finllm smoke --output runs/smoke --device cpu
```

The smoke run writes `summary.json`, metrics, predictions, and checkpoints. Choose a new output directory when repeating it. Its tiny model and overlapping synthetic examples exercise training, recovery, calibration, and prediction; their scores do not measure financial ability. Run the test suite with `python -m pytest -q`.

## What is implemented

- **Model and tokenizer:** a native PyTorch dense decoder with GQA, RoPE, Q/K RMSNorm, SwiGLU, tied embeddings, and self-trained byte-level BPE. No pretrained model weights are downloaded.
- **Data and training controls:** group-aware splits, exact and lexical near-duplicate filtering, held-out exclusions before tokenizer training, memory-mapped shards, and answer-position SFT/DPO.
- **Recovery and evaluation:** integrity-checked checkpoints retain optimizer, scheduler, scaler, RNG, and data-cursor state. Evaluation includes target-aware baselines, separate temperature and review-threshold calibration partitions, and JSONL prediction.

The decoder is adapted from [a pinned MiniMind commit](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f); attribution and changes are in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md).

## Continue with real data

[01a: prepare seed data](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/01a_seed_data.ipynb) → [02a: run the 24M pilot](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02a_seed_pilot.ipynb)

01a prepares bounded sources, exclusions, an 8192-token vocabulary, versioned shards, and backups on CPU. 02a restores matching artifacts for 61 T4 pretraining updates, approximately one million tokens.

The [saved preparation report](reports/seed_data_preparation.json) records **4,976,894 general training tokens**, **341,309 financial training tokens**, and **10,686 distinct candidate headlines excluded before tokenizer training**. These are local preparation results. Seed text, tokenizer files, and weights are generated outside Git.

The [seed execution record](docs/SEED_DATA.md) records **zero human-confirmed stance labels**. SEntFiN sentiment labels and AI proposals need target and stance review before formal SFT/DPO.

Complete `seed_v1` backups can be reused. For older directory-reuse, dependency, or active-kernel import failures, follow [COLAB_DEBUG](docs/COLAB_DEBUG.md), [COLAB_KERNEL_FIX](docs/COLAB_KERNEL_FIX.md), and [Issue #1](https://github.com/130U/130U-FinLLM/issues/1).

## Notebook guide

| Notebook | Purpose |
| --- | --- |
| [00_setup_and_smoke](notebooks/00_setup_and_smoke.ipynb) | Tiny training-to-prediction check |
| [01_prepare_data](notebooks/01_prepare_data.ipynb) | Review, split, exclude, tokenize, and pack formal data |
| [01a_seed_data](notebooks/01a_seed_data.ipynb) | Prepare and back up seed sources |
| [02a_seed_pilot](notebooks/02a_seed_pilot.ipynb) | Short 24M T4 pretraining pilot |
| [02_pretrain](notebooks/02_pretrain.ipynb) | General pretraining from random weights |
| [03_domain_pretrain](notebooks/03_domain_pretrain.ipynb) | Financial pretraining with general replay |
| [04_sft](notebooks/04_sft.ipynb) | Full-parameter target-stance learning |
| [05_dpo](notebooks/05_dpo.ipynb) | Preference training against a frozen SFT reference |
| [06_evaluate_and_predict](notebooks/06_evaluate_and_predict.ipynb) | Model selection, calibration, final test, and prediction |

The planned model has **24,255,744 parameters**, six layers, and an 8192-token vocabulary with 512-token context. The cumulative budget is 100M general tokens plus 25M domain tokens, including 5M general replay; replay does not imply new independent data.

Read the [data guide](docs/DATA_GUIDE.md), [execution plan](docs/PLAN.md), and [configs](configs/) for prerequisites, architecture, training commands, and continuation rules. Missing inputs fail explicitly. Select SFT or DPO using development evidence.

## Prediction contract

Each JSONL row supplies one text and one target:

```json
{"text": "Long $AAPL, short $TSLA.", "target": "AAPL"}
```

Evaluate another target separately. A cashtag or audited `target_aliases` mapping must establish the asset link.

```text
sentiment_score = 100 * (P(bullish) - P(bearish))
confidence = max(P(bearish), P(neutral), P(bullish))
```

The −100 to 100 score describes stance, not returns or price-increase probability. Unsupported target links, clearly unsupported language/script, and overlong prompts return empty scores and review reasons; language detection is limited.

After training, with matching tokenizer and calibration artifacts:

```bash
python -m finllm predict --checkpoint runs/sft/best.pt --tokenizer tokenizer/english \
  --input data/raw/predict.jsonl --calibration reports/private/calibration.json \
  --output reports/private/predictions.jsonl
```

Calibration is optional; without it, `calibrated: false` is recorded. Low confidence triggers review and does not automatically mean neutral.

## Validation evidence

Saved checks cover synthetic four-stage CPU training, one 24M random-token optimizer step, and persistent local Jupyter installation, training, backup, rerun, and restoration. The [kernel recovery report](reports/colab_kernel_fix.json) records 120 passing local tests with explicit substitutions. STATUS separately records a user-confirmed tiny Colab T4 smoke run.

Formal labels, GPU resource measurements, real SFT/DPO, independent calibration, and held-out financial testing remain open. Keep checkpoint files with their `.json` markers, and load only trusted checkpoints: full-state restoration uses Python-object deserialization.

## License

Code: [Apache License 2.0](LICENSE). External data and future weights require their own provenance and applicable terms.

