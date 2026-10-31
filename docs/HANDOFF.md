# 24M public Trial | Engineering handoff

This page supports further work on the public 24M Trial code and records. The project's private 2.4B version is described in the [project overview](PROJECT_OVERVIEW.md) and [model card](MODEL_CARD.md). Replacing the repository's default configuration with that architecture does not complete the migration.

The Trial has completed pretraining from random initialization, financial adaptation, headline entity-stance SFT, a balanced-sampling comparison, and calibrated inference export. Language training totals 26,115,096 valid supervised tokens. The final SFT training pool has 705 independent examples; the balanced run consumed 4,419 supervised label positions. See [implementation status](STATUS.md), the [counting ledger](PROJECT_PROGRESS.md), and the [experiment report](experiment-results.md).

## Engineering entry points

| Work | Entry point | Requirements |
|---|---|---|
| Installation and synthetic smoke | [Dependencies](../pyproject.toml), [00 notebook](../notebooks/00_setup_and_smoke.ipynb), [CLI](../src/finllm/cli.py) | Checks the code pipeline; does not measure real financial task performance |
| Seed data and pretraining | [Seed data](SEED_DATA.md), [T4 pilot](T4_PILOT.md) | Matching corpora, tokenizer, and exclusion manifests |
| Expansion and continuation | [Data expansion](DATA_EXPANSION.md), [Expanded continuation](EXPANDED_CONTINUATION.md) | Distinguish intermediate checkpoints from later cumulative records |
| Domain adaptation and task transition | [Domain handoff](DOMAIN_HANDOFF.md), [Post-domain transition](POST_DOMAIN_NEXT.md) | Verify parent weights, vocabulary, data, and stage configuration |
| Entity-stance SFT | [Entity SFT pilot](ENTITY_SFT_PILOT.md), [04e notebook](../notebooks/04e_kaggle_lite_sft.ipynb) | Private task package and matching domain weights |
| Balanced-sampling comparison | [Run script](../scripts/kaggle_balanced_sft.py) | Prepared 04e session inputs; sampling confined to the training partition |
| Calibration and inference export | [Run script](../scripts/kaggle_inference_export.py) | Best balanced-SFT weights, original tokenizer, and two calibration subsets |
| Human review and import | [Task review](TASK_REVIEW_NEXT.md), [Review workspace](REVIEW_WORKSPACE.md) | Verify source version, label scope, and human-confirmation status |

The repository contains code, default configurations, and public summaries, without private headlines, labels, training weights, or complete run backups. Reproducing a historical real-data run requires its matching data, checkpoints, and runtime configuration. Generic configuration files are not complete manifests of every experiment.

## Three kinds of state transfer

**Resume the same run.** Use a complete checkpoint to restore model, optimizer, learning-rate schedule, random state, data position, and progress, checking configuration and provenance identities. The [trainer](../src/finllm/training.py) and [checkpoint module](../src/finllm/checkpoint.py) perform compatibility and integrity checks. Read-only restoration checks on a completed run do not add training volume.

**Start a new stage.** Initialize from compatible model weights while creating a new optimizer and data position. Domain adaptation to SFT, and the initial and balanced SFT comparisons starting from the same domain parent, must be distinguished from a full resume.

**Scale to a different architecture.** The 24M and 2.4B versions differ in vocabulary, dimensions, layer count, and parameter shapes. Task rules, data identities, and validation methods can be reused. Token IDs, model weights, optimizer state, temperature, and thresholds cannot be transferred as though the new architecture were the same run.

A lightweight inference package contains prediction weights and metadata, without full optimizer/RNG state. It does not replace a recovery checkpoint. Retain complete checkpoints, paired metadata, completion markers, and data versions together. Restoration across hardware does not imply bitwise numerical identity.

## Minimum ledger for further experiments

Use a separate output directory for every experiment. Record code revision, model configuration, tokenizer identity, parent checkpoint, data and partition versions, target semantics, sampling rules, input and supervision counts, attempted and successful updates, model-selection data, and calibration data. When comparing sampling policies, also compare label exposure, update counts, and learning-rate schedules so that extra training is not attributed to one method alone.

Freeze new examples for further independent evaluation. Previously examined test data cannot become a fresh holdout. Real DPO requires preference supervision; three-class headline labels are not inherently preference pairs. The existing DPO implementation and synthetic smoke remain engineering entry points.

## Preservation and publication

Retain historical reports, including failures and preparation-stage records. [Status](STATUS.md) and the [experiment report](experiment-results.md) supply later updates. Public code licensing and third-party attribution are in [LICENSE](../LICENSE) and [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md). Data use follows the [data guide](DATA_GUIDE.md) and the conditions of each source.
