# 24M public Trial | Training and data ledger

This page summarizes actual inputs and outputs of the public Trial. Completion against an old budget does not describe the project as a whole. See the [project overview](PROJECT_OVERVIEW.md) and [model card](MODEL_CARD.md) for the private 2.4B version. All run counts below belong to 24M.

## Language training

| Stage | Cumulative updates within stage | Valid supervised tokens in stage | Validation observations |
|---|---:|---:|---|
| Seed pretraining | 300 | 4,914,941 | Seed validation loss 5.5419 |
| Expanded-corpus continuation | 1,220 | 19,988,195 | Expanded validation loss 3.6929; seed-retention loss 4.4032 → 3.9842 |
| Financial-domain adaptation | 74 | 1,211,960 | Financial validation loss 3.2375; general-retention loss 3.9842 → 4.0574 |
| Total | Counted separately by stage | **26,115,096** | Losses from different validation sets are not directly comparable |

Sources: [seed run](../reports/colab_t4_continuation_20261011.json), [expanded run](../reports/colab_t4_expanded_continuation_20261030.json), and [domain run](../reports/colab_t4_domain_20261030.json).

The expanded stage's 1,220 updates include the earlier 610 updates. Cumulative values from those reports must not be added together. Valid supervised tokens count positions contributing to language loss, not unique text. The earlier 100M foundation plus 25M domain tokens were historical Trial budgets, not completed 2.4B training volume.

## Review and task partitions

| Quantity | Count | Meaning |
|---|---:|---|
| Headline review task | 2,000 | Original review workload |
| Confirmed review decisions | 1,993 | Includes 114 exclusions |
| Unresolved records | 7 | Not counted as confirmed labels |
| Three-class candidates | 1,879 | 426 bullish, 234 bearish, 1,219 neutral |
| Actual Trial task pool | 1,008 | Also satisfies the headline literal-target construction rules |
| Training / development / calibration / test | 705 / 101 / 101 / 101 | Four partitions of the same fixed task pool |
| Temperature / threshold subsets | 51 / 50 | Fit temperature and select the review threshold separately |

Sources: [human-confirmation record](../reports/human_headline_review_20261030.json), [review definitions](TASK_REVIEW_NEXT.md), and [entity-task documentation](ENTITY_SFT_PILOT.md).

A completed label review does not simultaneously validate full-article labels, security mappings, or event identity. The 1,008-example task pool is not the training set; there are 705 independent training examples.

## SFT comparison

| Quantity | Initial SFT | Balanced-sampling SFT |
|---|---:|---:|
| Independent training examples | 705 | 705 |
| Epochs | 2 | 3 |
| Label exposures per epoch | 705 | 1,473 |
| Total supervised label positions | 1,410 | 4,419 |
| Optimizer updates | 45 | 139 |
| Development macro-F1 | 0.2863 | 0.6549 |
| Correct development predictions | 76/101 | 79/101 |

The original training pool contained 491 neutral, 140 bullish, and 74 bearish examples. The balanced run sampled only within that partition, bringing each class to 491 exposures per epoch without adding independent labeled examples. It initialized from the domain parent weights with a new optimizer. Total label exposure was approximately 3.13 times the initial run, and the update count was approximately 3.09 times larger; the learning-rate schedule also changed with the budget.

The initial run predicted neutral for every development and original test example. On the development set, the balanced model correctly classified 17 more directional examples and 14 fewer neutral examples. Word-level TF-IDF with class-weighted logistic regression achieved development macro-F1 of 0.6087. These are exploratory comparisons on the same development set, not independent final tests. See the [experiment report](experiment-results.md) and [balanced-SFT script](../scripts/kaggle_balanced_sft.py).

## Calibration and delivery

Temperature fitting, threshold selection, and lightweight inference export were run on the best balanced-model weights. The fitted temperature was approximately 1.50634, and the review threshold was approximately 0.57380. The export contains weights, tokenizer, calibration configuration, and a structured prediction entry point. Training restoration still requires the full checkpoint. See the [export script](../scripts/kaggle_inference_export.py).

These calibration values belong only to the bound Trial weights, target mode, and small calibration pool. New model versions and task distributions require recalibration and risk-and-coverage evaluation. Real financial DPO, Twitter weak-supervision training, and a new independent final test are not included among the completed runs on this page.

Accounting consistently distinguishes prepared quantities from consumed quantities, input positions from supervised positions, repeated exposures from independent information, attempted updates from successful updates, and development-set selection from final evaluation. See the [trainer implementation](../src/finllm/training.py) for the exact counters.
