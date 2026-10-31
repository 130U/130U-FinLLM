# 24M public Trial | Headline review and task construction

This page defines human-label and task-construction accounting for the public Trial. See the [project overview](PROJECT_OVERVIEW.md) and [model card](MODEL_CARD.md) for the private 2.4B version. All counts below refer to the 24M headline entity experiment.

## Review ledger

The original task contained 2,000 headlines. The user confirmed individual review of 1,993 records: 426 bullish, 234 bearish, 1,219 neutral, and 114 exclude. 7 remained unresolved. There were 1,879 three-class candidates, from which 1,008 Trial task examples were constructed.

The original CSV was matched record by record to the frozen source's 2,000 ordered IDs, texts, and targets. Confirmation and file identity are documented in the [human headline-review report](../reports/human_headline_review_20261030.json). The earlier [review-package report](../reports/task_review_2000_20261030.json) and [readiness report](../reports/task_review_readiness_20261030.json) describe AI pre-annotations and the absence of human confirmation at that time; they are not current completion counts.

## What each field establishes

| Field or record | What it represents | What it does not automatically establish |
|---|---|---|
| Source sentiment label | The original dataset's sentiment definition | This project's explicit investment stance |
| proposed_stance / ai_rationale | AI suggestions and rationale | Final human labels |
| human_stance / human_reviewed | Human-confirmed label and status | Full-article stance, security identity, or event identity |
| headline_only_stance scope | Judgment based only on the headline | A judgment using unseen article text |
| context_ai_stance | An AI comparison using context | Full-article human gold |
| Asset-identity and alias verification | Whether the target mapping is valid | Whether the text's stance is labeled correctly |
| Source, event, and relationship identifiers | Known relationships between examples | Discovery of all semantic duplicates |

5 disagreements between headline and contextual AI judgments were retained. Human confirmation applies to headline labels; contextual AI outputs cannot also be counted as reviewed gold. All event_id fields in the 2,000 review records are empty. An empty field means unknown, not “the same event” or “event independence has been verified.”

## From candidates to the Trial task pool

Historical construction first excluded 114 exclude decisions and 7 unresolved records, then applied the single-identifier target and literal-headline matching rules in effect at the time. Recorded construction counts were: 121 records failing the reviewed three-class requirement, 864 failing the single-identifier target requirement, and 7 failing literal matching, leaving 1,008 examples.

The pool was split into 705 training, 101 development, 101 calibration, and 101 test examples. Calibration was further divided into 51 temperature-fitting and 50 threshold-selection examples. Group checks used near-duplicate text and supplied relationship identifiers, but the pool was not a chronological holdout; empty event fields did not complete semantic event review. See the [entity SFT documentation](ENTITY_SFT_PILOT.md) and [task-construction script](../scripts/build_entity_pilot.py).

Training and calibration have run on this path. Balanced sampling repeats only the 705 training examples, adds no independent examples, and does not change development labels. Updates to target parsing likewise do not retroactively change the historical 1,008-example pool or its denominators.

## Additional review and import

For further review, fix the source-file identity, ordered example IDs, targets, and evidence scopes. Store AI suggestions, human decisions, and review reasons separately. Source changes require a new version; changing a hash to bypass a mismatch is not a valid import.

Human CSV confirmation and review-workspace progress JSON are two different import paths. Completion through one path does not mean the other was imported. See the [review workspace](REVIEW_WORKSPACE.md), [import implementation](../src/finllm/review_import.py), and [assembly script](../scripts/assemble_review_annotations.py).

New event-direction or personal-judgment tasks need their own definitions and fields for opinion holder, time, and evidence scope. Source-text stance labels cannot simply be copied as new gold. Security targets also require verification of identity and applicable time periods; completing headline review does not establish these conditions.

The public repository retains review statistics and construction methods without publishing private source text or row-level decisions. Data-use conditions are documented in the [data guide](DATA_GUIDE.md) and [source review](research/stance_sources_20261009.md). The repository's code license does not replace external data licenses.
