# 24M public Trial | Task and training specification

This page describes the public 24M Trial's task, implementation, and accounting rules. See the [project overview](PROJECT_OVERVIEW.md) and [model card](MODEL_CARD.md) for the private 2.4B version. The public code and actual runs described here belong to the Trial; changing parameter settings alone does not implement a different architecture.

## Task definition

Given English text and a specified target, identify the investment stance expressed in the text: bearish, neutral, or bullish. A headline reporting improved operations, deteriorating finances, or a price change does not automatically express an author's explicit bullish or bearish stance. Target relationships, opinion attribution, and evidence scope are part of the label definition.

For eligible inputs, outputs retain three-class normalized probabilities, a predicted class, confidence, a direction score, and review status:

`score = 100 × (P(bullish) − P(bearish))`

A score near zero can reflect either high neutral probability or offsetting probabilities for the two directions, so the probability vector must be retained. Failed input eligibility is a review condition and should not be hidden behind a neutral label.

The actual Trial uses headline literal-entity target mode. This does not establish a verified security identifier or a personal investment judgment. Full-article context, event impact, and personal preferences require their own task definitions and label sources.

## Model and tokenizer

| Component | 24M Trial configuration |
|---|---:|
| Total parameters | 24,255,744 |
| Transformer layers | 6 |
| Hidden size | 512 |
| FFN intermediate size | 1,664 |
| Query heads / KV heads | 8 / 4 |
| Head dimension | 64 |
| Vocabulary size | 8,192 |
| Context length | 512 |
| Embeddings / output projection | Tied weights |

The backbone is a dense autoregressive decoder initialized at random, using GQA, RoPE, pre-RMSNorm, QK normalization, and SwiGLU. The tokenizer is a self-trained byte-level BPE. See the [model implementation](../src/finllm/model.py), [tokenizer](../src/finllm/tokenizer.py), and [model_24m.json](../configs/model_24m.json).

Architecture references and adaptation provenance are retained in [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md), including MiniMind revision [f659b55761b754d306bd140573493a6543cafd7f](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f). Licensing and attribution remain separate from version positioning and do not change with the model scale described.

## Stages and observed status

| Stage | Trial status | Main accounting units |
|---|---|---|
| Tokenizer and foundational language training | Executed | Tokenizer identity, valid supervised tokens, validation loss |
| Expanded pretraining and domain adaptation | Executed | Stage-specific cumulative updates; financial and general-retention validation sets |
| Target-conditioned SFT | Initial and balanced runs executed | Independent examples, label exposures, per-class performance, macro-F1 |
| Calibration and inference export | Executed | Temperature, threshold, two disjoint calibration subsets, weight binding |
| DPO | Code and synthetic pipeline checks available | Real financial preference training is not among the completed experiments |
| New independent final evaluation | No new result record yet | Event independence, time range, complete prediction pipeline |

Actual language training consumed 26,115,096 valid supervised tokens. The earlier 100M foundation plus 25M domain tokens were historical Trial budgets; consumption and budgets are recorded separately. [Status](STATUS.md), the [counting ledger](PROJECT_PROGRESS.md), and the [experiment report](experiment-results.md) document executed runs.

## Labels and partitions

Of 2,000 reviewed headlines, 1,993 decisions were confirmed, including 114 exclusions. 7 remained unresolved, leaving 1,879 three-class candidates. The Trial selected 1,008 examples and split them into 705 training, 101 development, 101 calibration, and 101 test records. The calibration pool was further divided into 51 temperature-fitting and 50 threshold-selection examples.

Human confirmation applies to headline-scope labels. 5 disagreements between headline and contextual AI judgments were retained; full-context AI comparisons are not full-article human gold. Security-identity checks, event grouping, and label review use different fields, and one completion flag cannot substitute for the others. See the [review definitions](TASK_REVIEW_NEXT.md) and [human-confirmation report](../reports/human_headline_review_20261030.json).

Data processing groups normalized text, near-duplicates, and supplied relationship identifiers, and freezes task-exclusion manifests. The existing Trial split is not a chronological holdout and does not establish complete semantic event isolation. Evaluation-content exclusion must also be checked for tokenizer training and language corpora, not just overlap between SFT files. See the [data implementation](../src/finllm/data.py) and [data guide](DATA_GUIDE.md).

External labels are not automatically equivalent to this task's labels: company sentiment, favorable or adverse events, and explicit investment stance can differ. New tasks and text scopes require explicit conversion rules and human review.

## Supervision and model selection

Language training applies next-token loss at valid prediction positions. SFT uses the shared vocabulary head and computes full-vocabulary cross-entropy at the position predicting the answer label; prompt and padding positions are not label-supervision targets. Evaluation then computes conditional probabilities over the three label tokens. These supervision denominators must not be conflated.

The initial training pool contained 491 neutral, 140 bullish, and 74 bearish examples. The balanced version brought each class to 491 exposures per epoch. The initial run used 2 epochs, 45 updates, and 1,410 supervised label positions; the balanced run used 3 epochs, 139 updates, and 4,419 supervised label positions. Sampling and budget changed together, so subsequent method comparisons must control or disclose both.

Language stages select models by validation loss. SFT selects by development macro-F1 and also reports class support, precision, recall, and the confusion structure. The development set participates in model selection. Previously examined test results cannot be presented as a new independent holdout. See the [trainer](../src/finllm/training.py) and [evaluation implementation](../src/finllm/evaluation.py).

## State and probability binding

Resuming the same run restores its complete training state. Initializing a new stage transfers only compatible weights and creates new optimization state. Counters distinguish input positions, supervised positions, and attempted/successful updates. AMP-skipped updates are recorded separately from successful optimizer steps. Gradient accumulation is normalized by actual valid supervised positions. Architecture changes require new parameters, vocabulary, and resource implementation.

Calibration fits probability scaling on the temperature subset and selects a review threshold on the threshold subset. Results are bound to the checkpoint, tokenizer, and target mode. The executed Trial produced a temperature of approximately 1.50634 and a threshold of approximately 0.57380; these are not default settings for 2.4B. Positive temperature scaling preserves the class argmax, and calibration does not replace risk-and-coverage evaluation under a new distribution. See the [export script](../scripts/kaggle_inference_export.py).

Public engineering checks cover label tokens, causal masks, supervision positions, finite values, checkpoint integrity, and restoration. Check records are kept separately from task results. See the [engineering handoff](HANDOFF.md) for continuation entry points.

## Data and references

The code license does not automatically grant permission to redistribute training data. Provenance and usage conditions are retained in the [data guide](DATA_GUIDE.md), [seed-corpus sources](research/corpus_sources_20261009.md), [expanded-corpus sources](research/expanded_corpus_sources_20261011.md), and [stance-task sources](research/stance_sources_20261009.md).

Method references: [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/quicktour), the [original DPO paper](https://arxiv.org/abs/2305.18290), and [temperature scaling and neural-network calibration](https://proceedings.mlr.press/v70/guo17a.html). Citing a method does not imply that its corresponding real-data training has been completed in this Trial.
