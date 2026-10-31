# 24M Trial: headline stance experiments

This report covers the 24M Trial branch of 130U-FinLLM: a 24,255,744-parameter decoder trained from random initialization, then adapted to finance and supervised on target-linked English headlines. These observed results belong to the Trial branch; the project's 2.4B architecture and method design are presented in the [project overview](../README.md). The measurements below assess headline stance, not market prediction or a validated Twitter classifier.

## Results

| System | Partition | Examples | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: | ---: |
| Initial SFT | Development | 101 | 75.25% | 0.2863 |
| Initial SFT | Test | 101 | 58.42% | 0.2458 |
| Class-weighted TF-IDF + logistic regression | Development | 101 | 75.25% | 0.6087 |
| Balanced SFT | Development | 101 | 78.22%* | 0.6549 |

\* Balanced SFT accuracy is calculated from the supplied per-class recall and support: 5 bearish, 62 neutral, and 12 bullish correct, or 79/101. Macro-F1 is the mean of the three class F1 values. The balanced SFT and baseline results are exploratory model-selection results on the same development set; they are not independent test estimates.

The initial SFT classified every development and test example as neutral, so its directional recall was zero. The development-set class breakdown for balanced SFT is:

| Label | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Bearish | 10 | 0.4545 | 0.5000 | 0.4762 |
| Neutral | 76 | 0.9394 | 0.8158 | 0.8732 |
| Bullish | 15 | 0.5000 | 0.8000 | 0.6154 |

## Data and training

The original SFT training split contains 705 rows: 491 neutral, 140 bullish, and 74 bearish. The balanced run samples only from this training split, repeating rows to obtain 491 exposures per class per epoch; the 1,473 exposures per epoch are not additional unique examples. The run used three epochs and reported 139 optimizer updates and 4,419 supervised label tokens. It initialized from the trained finance-domain model weights with a new optimizer, not from the first SFT optimizer.

The two SFT comparisons use the same 101-example development split. The baseline uses the original training rows, class-weighted logistic regression and word TF-IDF features with unigrams and bigrams. The prior test set was examined for the initial run; it has not been reported as a fresh test for the balanced model. A new, independent and preferably chronological, event-deduplicated holdout is needed for further evaluation.

The [balanced SFT experiment script](../scripts/kaggle_balanced_sft.py) records the training and development comparison. Running it requires the existing private 04e Kaggle session inputs and financial-domain checkpoint; the repository alone does not contain the data or model weights. The reported run used repository revision `75a90c3fca66ae08abf77742ccbd4368e50c7e25`.

## Evidence and limitations

The original run's checkpoint, tokenizer, reports, and input files were received in a verified private backup. The balanced run's counts, metrics, and completion are taken from a user-supplied Kaggle notebook log; its new checkpoint and archive have not been independently downloaded or verified here. No private headline, label, model weight, or backup is published in this report.

Samples are financial news headlines paired with literal target entities, not verified stock ticker mappings or Twitter posts. Splits are not chronological and event-level semantic independence has not been fully audited. Small and uneven class supports mean the reported scores have substantial uncertainty. These results do not measure investment returns or demonstrate production-ready performance. DPO code exists, but no DPO run is represented by the results above.
