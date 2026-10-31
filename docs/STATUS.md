# 24M public Trial | Implementation and validation status

This page records the implementation and actual runs of the 24,255,744-parameter Trial. The broader 130U-FinLLM project comprises a **private 2.4B model and a public 24M Trial**, described in the [project overview](PROJECT_OVERVIEW.md) and [model card](MODEL_CARD.md). All training counts and results below belong to the 24M Trial.

## Completed training path

Random initialization → seed pretraining → expanded-corpus continuation → financial-domain adaptation → headline entity-stance SFT → balanced-sampling SFT → calibration and inference export.

| Stage | Confirmed runs and counts | Evidence |
|---|---|---|
| Seed pretraining | 300 updates; 4,914,941 valid supervised tokens; validation loss 5.5419 | [Run report](../reports/colab_t4_continuation_20261011.json) |
| Expanded-corpus continuation | 1,220 cumulative updates; 19,988,195 valid supervised tokens; expanded validation loss 3.6929 | [Run report](../reports/colab_t4_expanded_continuation_20261030.json) |
| Financial-domain adaptation | 74 updates; 1,211,960 valid supervised tokens; financial validation loss 3.2375 | [Run report](../reports/colab_t4_domain_20261030.json) |
| Initial entity-stance SFT | 705 training examples, 2 epochs, 45 updates, 1,410 supervised label positions; all-neutral development predictions, macro-F1 0.2863 | [Experiment report](experiment-results.md) |
| Balanced-sampling SFT | Same training pool, 3 epochs, 139 updates, 4,419 supervised label positions; development macro-F1 0.6549, 79/101 correct | [Experiment report](experiment-results.md), [run script](../scripts/kaggle_balanced_sft.py) |
| Calibration and inference export | Completed; temperature approximately 1.50634, review threshold approximately 0.57380; private lightweight inference package generated | [Export and calibration script](../scripts/kaggle_inference_export.py) |

Language training totals **26,115,096 valid supervised tokens**. The expanded stage's first 610 updates are included in the cumulative 1,220 and must not be counted again. SFT label positions, input tokens, and independent examples are separate quantities.

The initial all-neutral failure remains in the experiment report. On the same development set, the balanced model correctly classified 17 more directional examples and 14 fewer neutral examples. Sampling, epochs, update budget, and the learning-rate schedule changed together, so this comparison does not isolate sampling. The previously examined test results belong to the initial SFT model.

## Data and task scope

The human-review task contains 2,000 headlines: 1,993 confirmed decisions, including 114 exclusions; 7 unresolved records; and 1,879 three-class candidates. Literal-target requirements yielded a Trial pool of 1,008 examples, split into 705 training, 101 development, 101 calibration, and 101 test records. The calibration pool was further split into 51 temperature-fitting and 50 threshold-selection examples. See the [human-confirmation record](../reports/human_headline_review_20261030.json), [review definitions](TASK_REVIEW_NEXT.md), and [entity SFT entry point](ENTITY_SFT_PILOT.md).

The confirmed labels describe stance within the headline. They do not establish security identity, full-article stance, or future price direction. The existing partitions group text and supplied relationship identifiers; they are not chronological holdouts. Event-level semantic independence requires separate review. The target expressions supported by current code must not be used to infer a different historical training-set size.

## Public implementation

- [Model](../src/finllm/model.py) and [24M configuration](../configs/model_24m.json): a 6-layer dense decoder with GQA, RoPE, QK normalization, SwiGLU, RMSNorm, and tied vocabulary weights.
- [Data processing](../src/finllm/data.py): task grouping, exclusion manifests, token packing, and memory-mapped reads. The [tokenizer](../src/finllm/tokenizer.py) is a self-trained 8K byte-level BPE.
- [Trainer](../src/finllm/training.py): language training, answer-position SFT, the DPO path, gradient accumulation, training counters, and checkpoint restoration.
- [Calibration](../src/finllm/evaluation.py) and [inference](../src/finllm/inference.py): three-class probabilities, direction scores, input checks, and review status.

DPO has an implementation and a small synthetic pipeline check, but no DPO result on the real financial data described here. The [Twitter weak-supervision script](../scripts/kaggle_twitter_weak_sft.py) is also outside the confirmed experiments above. Further evaluation should use new independent examples and report performance by class and among accepted predictions.

## Evidence and entry points

The initial run artifacts were received in a verified private backup. Completion of balanced SFT and inference export is supported by user-supplied run outputs. The public repository supplies code, configurations, and summary records; private headlines, labels, weights, and backups are not published with it. See the [experiment report](experiment-results.md) for result attribution.

The [CPU smoke](../reports/cpu_smoke_check.json), [full-configuration CPU check](../reports/full_config_cpu_check.json), [continuation checks](../reports/seed_continuation_checks.json), and [domain handoff checks](../reports/domain_handoff_checks_20261030.json) document engineering validation rather than task performance. Historical reports retain their original status. Early descriptions such as “awaiting review” or “SFT not started” are supplemented by subsequent run records.

Continue with the [training and data ledger](PROJECT_PROGRESS.md), [engineering handoff](HANDOFF.md), or [Trial technical specification](PLAN.md).
