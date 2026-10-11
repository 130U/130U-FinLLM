# 130U-FinLLM

A financial sentence can express opposite views about different assets. “Long $AAPL, short $TSLA” needs one interpretation for Apple and another for Tesla. Assigning a single sentiment label to the sentence loses that distinction.

I built **130U-FinLLM** to train and evaluate a compact language model from random initialization. The system takes English text and a specified asset, and produces **bearish, neutral, or bullish** stance, class probabilities, and a direction score.

## The model and learning problem

The model specification is a **24.26M-parameter decoder** implemented in native PyTorch, with a self-trained byte-level BPE tokenizer. The training path covers general pretraining, financial-domain pretraining, full-parameter supervised fine-tuning, and a preference-training experiment.

My task design puts the asset inside the model input and places supervision at the answer label. This keeps the learning objective focused on the text’s stance toward that asset. Preference training compares alternative stance labels against a frozen supervised reference model.

The prediction score is `100 × (P(bullish) − P(bearish))`: a measure of textual direction toward the target. The interface also returns confidence and review reasons when the asset connection or input cannot be established.

## What I built

The completed work is a working training and evaluation stack, together with a prepared first corpus of general English text and financial source material. The preparation pipeline separates training and validation documents, excludes candidate evaluation text before vocabulary training, and converts the remaining material into versioned token shards.

I designed the evaluation around questions that matter for this task: does the model distinguish the stance classes, are its probabilities calibrated, and which predictions should be reviewed? Model selection, probability calibration, review-threshold selection, and final testing use separate partitions. The implementation also includes a target-aware text baseline for comparison.

Training recovery is part of the experiment design. A resumed run restores the optimization state, random state, and position in the corpus. Calibration files are tied to the selected checkpoint and tokenizer. These choices keep comparisons connected to the model, data, and training history that actually produced them.

This provides a concrete setup for researching financial language models from scratch: the target, learning objective, training stages, and evaluation decisions are explicit and can be changed individually.

## Read or run

[Training code](src/finllm/training.py) · [Data preparation](docs/SEED_DATA.md) · [Evaluation](src/finllm/evaluation.py) · [Model specification](docs/MODEL_CARD.md)

Start with the [small training-to-prediction notebook](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/00_setup_and_smoke.ipynb). Continue through [seed preparation](notebooks/01a_seed_data.ipynb), the [model pilot](notebooks/02a_seed_pilot.ipynb), and [checkpoint continuation](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02b_seed_continue.ipynb).

A user-run Colab T4 experiment reached 300 cumulative updates and 4,914,941 supervised tokens, with validation loss decreasing from 7.2726 at the pilot to 5.5419. The continuation took about 19 minutes, and the final Drive backup reported successful verification. The [training record](docs/T4_PILOT.md) documents the user-supplied evidence, measurement limits, and next steps. Financial stance accuracy has not yet been evaluated.

The next stage uses [01b: additional data](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/01b_expand_data.ipynb) and [02c: expanded pretraining](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02c_expanded_pretrain.ipynb). It keeps the existing tokenizer, removes overlap with seed and evaluation documents, and transfers the 300-step model weights into a separately recorded training stage. The [expansion guide](docs/DATA_EXPANSION.md) explains the data limits, new optimizer state, recovery checks, and initial 610-update budget. This Colab stage is pending execution.

The [local preparation report](reports/expansion_local_validation_20261011.json) records 41,499,364 additional general training tokens and 972,438 financial training tokens with the unchanged 8K tokenizer. CPU preparation took 544.91 seconds, sampled process RSS peaked at 344.7 MiB, and the candidate, prepared, and tokenized outputs occupied 560.2 MiB, excluding the earlier seed and checkpoints. These are local preparation measurements; the new Colab training has not run. All 205 local tests and checks for 12 notebooks passed.

The decoder architecture is adapted from [MiniMind](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f). Attribution and project-specific changes appear in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md). Code is licensed under [Apache-2.0](LICENSE).

