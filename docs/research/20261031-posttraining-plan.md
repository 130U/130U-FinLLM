# 24M Trial | Class Balancing, Preference Learning, and Probability Calibration

Scope: historical methodology research for the 24M Trial of 130U-FinLLM, compiled on 2026-10-31. The original filename is retained to preserve references. Initial SFT, balanced SFT, and subsequent inference calibration each have execution records; DPO remains a conditional methodological option. This document explains the methods and evidence, rather than serving as an implementation schedule for the overall 2.4B project. See the [project overview](../../README.md).

## Charter

- **Objective:** explain how the 24M Trial handles class imbalance, structures post-training comparisons, calibrates probabilities, and interprets measured results.
- **Scope / object / topic:** English financial-text classification / asset-target stance in the existing model / class-balanced SFT, optional DPO, and validation on unseen data.
- **Research type:** descriptive methodology research, using Trial training records to explain method selection, comparison conditions, and applicability.
- **Questions:** Q1: How should class imbalance be handled? Q2: What data does DPO require? Q3: How should a new model be assessed? Q4: Can external financial Twitter data be used?
- **Evidence approach:** prioritize original papers, original dataset cards, and official documentation. Project logs and code establish actual project behavior. Search date: 2026-10-31.

## Canonical Sources

| Tier | Source | Purpose |
|---|---|---|
| 1 | [Original DPO paper](https://arxiv.org/abs/2305.18290) | Preference-training method and data semantics |
| 1 | [BEA 2025 class-imbalance experiment](https://aclanthology.org/2025.bea-1.103/) | Examples of resampling and class weighting in an imbalanced task |
| 1 | [Twitter Financial News Sentiment dataset card](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | Declared fields, labels, license, and limitations of candidate external data |
| 1 | [Neural-network calibration paper](https://arxiv.org/abs/1706.04599) | Purpose of probability calibration |
| 2 | [scikit-learn F1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html); [model-evaluation documentation](https://scikit-learn.org/stable/modules/cross_validation.html) | Definition of macro averaging and importance of unseen samples |
| 3 | [Project results report](../experiment-results.md); [balanced SFT script](../../scripts/kaggle_balanced_sft.py) | Actual user-supplied logs, configuration, and current data scope |

## Findings Ledger

| ID | Finding | Evidence | Tier | Question |
|---|---|---|---:|---|
| F1 | The class-balanced model achieved macro-F1 of 0.6549 on the 101-record development set, compared with 0.2863 for the original SFT and 0.6087 for the class-weighted TF-IDF baseline. These scores were used for model selection. | [Project results report](../experiment-results.md), user-supplied Kaggle logs | 3 | Q1, Q3 |
| F2 | A three-class educational-text experiment used oversampling, class weighting, and augmentation. Its task and model differ from this project, so its results cannot be treated as this project's effects. | [Original BEA 2025 paper](https://aclanthology.org/2025.bea-1.103/) | 1 | Q1 |
| F3 | DPO targets relative preference signals. Without project-specific preference pairs, no new independent human-preference feedback can be claimed. | [Original DPO paper](https://arxiv.org/abs/2305.18290); existing project task data | 1, 3 | Q2 |
| F4 | `macro` F1 is the unweighted mean of per-class F1. After development-set model selection, unseen data is needed to assess generalization. | [F1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html); [model-evaluation documentation](https://scikit-learn.org/stable/modules/cross_validation.html) | 2 | Q3 |
| F5 | The Twitter financial dataset card describes three-class text classification and shows `text` and `label` fields, without a separate target-asset field. The card declares MIT, while its stated total size and split counts are inconsistent. | [Original dataset card](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | 1 | Q4 |
| F6 | Temperature scaling calibrates probabilities. Positive temperature scaling leaves the highest-scoring class unchanged, as follows directly from the operation's ordering, so it cannot repair three-class prediction collapse. | [Original calibration paper](https://arxiv.org/abs/1706.04599); mathematical deduction | 1, 3 | Q3 |

## Scope Filter

F2 is not generalized from educational dialogue into a guaranteed effect on financial text. Text-sentiment or headline scores are not interpreted as future returns or validated performance on security identifiers.

## Object Filter

F1 consists only of results on a non-temporal development split reused during training. An old test set already inspected cannot serve as a fresh independent test for a new model. The whole-text labels in F5 do not automatically provide a separate stance toward every asset mentioned. Multi-asset or ambiguous-target content requires exclusion or separate annotation. The dataset-card license must still be assessed against actual sources and intended use, and actual row counts should be checked first.

## Methodological Conclusions and Trial Application

1. **Class balancing and model selection.** The Trial resampled only within the training partition and completed development-set comparisons of initial SFT, balanced SFT, and simple baselines; actual scores are in F1. Macro-F1 and per-class recall expose all-neutral prediction that majority-class accuracy can conceal. Balanced SFT changed both training exposure and the update budget, so these results cannot isolate resampling as the cause. Attribution requires matched-budget comparisons. After checkpoint selection, a new independent test must assess generalization; development scores do not directly establish performance on actual Twitter data (F1, F4).
2. **Conditions for DPO.** Confirmed correct and incorrect training labels can form derived `chosen/rejected` pairs. With the SFT reference model frozen, an experiment can test whether the preference objective adds value. This supervision should be called **preferences derived from class labels**, rather than new human preferences (F3). The existing formal DPO entry point requires `preference_reviewed=true` for every pair; confirmation of class labels does not constitute pair-by-pair preference review. Derived pairs require a clearly distinguished exploratory workflow, or prior human preference review. This option has not been executed, and comparison must allow retention of a better-performing SFT model.
3. **Semantic alignment of external weak supervision.** Candidate Twitter data requires a fixed version and checks of availability, license, and label semantics. It is a weak-supervision candidate only when the target is unambiguously singular and the label matches the task. Multi-target samples and unclear directions must not default to genuine asset stance (F5). Subsequent external-data experiments should report their training sources and evaluation objects separately, rather than being merged into the headline-data comparisons in F1.
4. **Probability calibration after model selection.** The Trial's later inference export fitted temperature and a review threshold for the selected balanced-SFT model. The original 101-record calibration partition was split by group into 51 temperature-fitting and 50 threshold-selection records. See the [inference export script](../../scripts/kaggle_inference_export.py). Calibration adjusts the sharpness of the three-label conditional probabilities. Positive temperature does not change class ranking and therefore does not fix prediction collapse (F6). A review threshold fitted on a small sample is not a guarantee of future error rates.

## Reproduction Conditions and Applicability

1. Bind each experiment to its actual checkpoint, tokenizer, code version, training partition, and output. Record backup creation, download, and independent verification separately. A lightweight inference export does not replace a training-recovery package containing optimizer state.
2. SFT/DPO comparisons use fixed data, budgets, and the same development set. Evaluation partitions do not enter preference construction. Once an old test set has been used to inspect results, it cannot again be described as unseen.
3. Generalization evaluation uses new data that participated in neither training nor tuning, reviewed by target, event, and time. Report per-class sample counts, macro-F1, per-class recall, and review coverage. Report headline and external Twitter results separately.
4. The reproducible measured object here is the 24M Trial's headline-stance experiment. External-domain transfer, complete event isolation, and investment returns require corresponding experiments and cannot be inferred from this page's development comparisons.
