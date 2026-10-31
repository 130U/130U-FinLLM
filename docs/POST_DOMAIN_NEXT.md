# 24M public Trial | From domain weights to stance tasks

This page records stage transitions in the public Trial, whose completed path now includes balanced SFT, calibration, and inference export. See the [project overview](PROJECT_OVERVIEW.md) and [model card](MODEL_CARD.md) for the private 2.4B version. Results on this page belong only to 24M.

## Completed transitions

| Component | Recorded outcome |
|---|---|
| Domain parent model | 74 updates; 1,211,960 valid supervised tokens; financial validation loss 3.2375; general-retention loss 3.9842 → 4.0574 |
| Fixed task pool | 1,008 headline entity examples: 705 training, 101 development, 101 calibration, 101 test |
| Initial SFT | 2 epochs, 45 updates; all-neutral predictions on development and the original test set |
| Balanced SFT | 3 epochs, 139 updates, 4,419 supervised label positions; development macro-F1 0.6549 |
| Calibration and export | Completed; temperature approximately 1.50634, review threshold approximately 0.57380; lightweight inference package generated |

Sources: [domain run](../reports/colab_t4_domain_20261030.json), [entity-task entry point](ENTITY_SFT_PILOT.md), [experiment report](experiment-results.md), [balanced-SFT script](../scripts/kaggle_balanced_sft.py), and [export script](../scripts/kaggle_inference_export.py).

Both initial and balanced SFT created new optimization state from the domain parent weights. The balanced run increased class exposure and training budget; it did not change sampling alone or resume the initial SFT optimizer. Preparation-stage reports are retained to document inputs at the time, not to supersede later completed runs.

## Stage-transition checks

1. **Parent identity.** Match the checkpoint, model configuration, tokenizer hash, reports, and source revision. Weights must correspond to the selected domain stage.
2. **Data and exclusion manifests.** Verify task partitions and versions, distinguishing source-file integrity from exclusion-content coverage. A matching SHA establishes file identity; it does not independently establish absence of semantic event leakage.
3. **Task semantics.** Record label confirmation, target association, security mapping, and event grouping separately. Headline stance does not automatically extend to full-article views or future price direction.
4. **Training state.** Stage initialization and same-run resume have different state boundaries. Lightweight weight packages support a new stage or inference, not full training restoration.

Implementations: [SFT preflight](../src/finllm/sft_preflight.py), [SFT runtime](../src/finllm/sft_pilot_runtime.py), and [lightweight handoff](../src/finllm/kaggle_lite.py).

## Target modes and new tasks

The historical pool of 1,008 examples was built using the single-identifier literal-target rules in effect at the time. Current inference capabilities and historical construction rules must be interpreted separately. Supporting more literal-target expressions does not add historical training examples. Security mode also requires verification of asset identity, identifier type, aliases, and historical validity; a current ticker with the same name is not a substitute.

When adding labels or expanding semantic scope, freeze a new review version, task partitions, and exclusion manifests. Human CSV confirmation and review-workspace progress JSON are separate import paths, each with its own source-identity and field rules. See the [review definitions](TASK_REVIEW_NEXT.md).

Real DPO requires preference supervision with its own review; three-class headline labels cannot simply be called preference pairs. Further comparisons need a new independent evaluation set. The old test set, already examined for the initial model, is no longer a fresh holdout. Continue through the [engineering handoff](HANDOFF.md).
