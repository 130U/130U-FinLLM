# 24M Trial | Data Preparation and Review

Scope: public engineering documentation for the 24M Trial, describing the existing pipeline's data formats, 8K vocabulary, and review rules. The original example budgets below are not the training budget for the overall project or the 2.4B model. See the [project overview](../README.md).

The task is the investment stance expressed by a given text toward a specified asset, rather than the sentence's general positive or negative sentiment. Statements about historical market movements default to neutral; subsequent price changes must not be used to fill in labels.

## Input Files

These files contain user-provided material or public material whose sources have been reviewed. The repository does not claim to possess a formal corpus that has not actually been obtained.

```text
data/raw/
  task.jsonl                 # Human-confirmed, target-conditioned labels
  general_train.jsonl         # General training text
  finance_train.jsonl         # Financial training text
  general_validation.jsonl    # Held-out text for pretraining diagnostics
  finance_validation.jsonl    # Held-out text for financial diagnostics
  tokenizer_train.jsonl       # Representative sample of general/financial training text
```

Each line is a JSON object. Formal source text requires `text, source, rights_note, training_allowed:true`. Labeled data requires `id,target,label`; recommended fields are `created_at` (ISO8601 with a timezone), `event_id`, and human-review information. If provided, `split` should be `validation` for validation text and `train` for training text. Missing fields must not be filled with invented dates or rights statements.

`training_allowed` records a completed source-rights review; setting the flag does not create permission. A third-party mirror or public availability does not automatically grant training rights. Review the original data card, source, and applicable terms before importing a candidate. X content is excluded from the first training round until clearly applicable training rights have been established. Authorized English financial material and independently authored technical samples can be used instead.

See `data/examples/annotation_template.jsonl` for a label-format example. It demonstrates the format and is not formal training data.

## Labels and Targets

- `bullish`: expresses buying, going long, an upward outlook, or a positive investment outlook for the target.
- `bearish`: expresses selling, going short, a downward outlook, or a negative investment outlook for the target.
- `neutral`: verifiably mentions the target without expressing a direction.
- Samples with unclear sarcasm, quotation attribution, or asset relationships remain pending review instead of being forced into neutral. Positive/negative labels in public datasets must be reviewed against these definitions.

Multiple targets from the same post share the original `id`; records from the same event or duplicate source should share `event_id/group_id`. Current automatic checks use normalized exact matching plus 64-bit SimHash (Hamming≤3). They do not guarantee detection of every semantic paraphrase; human event grouping supplements them.

## Required Order

1. Review approximately 200 rubric samples first, record disagreements, and obtain enough data to retain validation examples for every class. Independent review by two annotators and delayed relabeling by one annotator must be reported separately.
2. Run `prepare-task` for an approximately 70/10/10/10 split by duplicate/event group. Where reliable timestamps exist, sort by each group's latest time; otherwise identify the split as non-temporal. Split the calibration partition into two subsets by group.
3. Combine held-out task texts and both pretraining validation sets into `frozen_exclusions.jsonl`. Exclude those texts from the formal tokenizer and training shards. Pretraining validation shards exclude only held-out task texts, so they do not filter themselves out entirely.
4. Train the tokenizer from scratch and check that it actually contains 8192 tokens. If insufficient data produces a smaller vocabulary, add training material and rebuild in a new output directory. Do not add special tokens while retaining the old parameter count.
5. Encode shards and generate hash manifests. The default is 5M tokens per shard, approximately 10MB. The exact-deduplication index lives in SQLite on disk, tokens are stored as uint16, and only the current batch is converted to int64.
6. Use `mix-corpora` to produce exactly 20M financial + 5M general tokens. If the input is insufficient, the command cycles through the available sources. The manifest separately records source-token counts and cumulative mixed-token counts; replay is not independent new data.

Complete executable commands are in `01_prepare_data.ipynb`. If the shard directory already contains a manifest or remnants of an earlier build, choose a new version directory instead of overwriting data referenced by a checkpoint.

## Controlled Streaming Downloads

After `pip install -e '.[data]'`, use `fetch-corpus` to access an explicitly specified dataset at a fixed revision, stopping at document-count or byte limits. Review the applicable terms first. Initial defaults are 1000 documents and 20MB; the command does not download an entire large corpus.

```bash
python -m finllm fetch-corpus --dataset DATASET_ID --revision FULL_40_CHARACTER_COMMIT \
  --max-documents 1000 --max-bytes 20000000 --rights-note "Specific reviewed terms" \
  --training-rights-reviewed --output data/raw/general_candidate.jsonl
```

Replace `DATASET_ID` and the revision with verified values. This is an example, not a record of a completed download. Candidate review can begin with the [FineWeb-Edu data card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu); for financial sources, consult the [official SEC EDGAR access rules](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data). This version includes no X scraper and does not accept source terms on the user's behalf.

An individually oversized document in a streaming source is skipped; a later version can segment it if needed. Preserve existing negation, cashtags, numbers, and percentages. Avoid generic cleaning that removes them.

## DPO Data

```bash
python -m finllm make-preferences --input data/task/train.jsonl --output data/task/preferences_candidates.jsonl
```

This command constructs candidate chosen/rejected pairs from reviewed training labels, with `preference_reviewed:false` by default. After reviewing the pairs, save confirmed records separately as `preferences_train.jsonl` and set the flag to true. Formal training rejects unreviewed pairs, non-training partitions, identical answers, and unknown classes. Do not construct preferences from dev/cal/test.

## Evaluation

Use the development set to select models and training parameters. Freeze the final model before fitting temperature and the review threshold on their separate partitions. Report test results only at the end; do not inspect the test set to tune parameters. Actual performance reporting should include at least macro-F1, per-class performance, sample counts, NLL/Brier/ECE, and review coverage.

```bash
python -m finllm baseline --train data/task/train.jsonl --input data/task/development.jsonl --output reports/private/baseline_dev.json
```

This command includes majority-class and target-aware TF-IDF linear baselines. FinBERT comparisons, repeated seeds, event-group bootstrap, and unseen-asset stress tests still require actual data and further experiments; they were not completed as part of this small trial.
