# 24M Trial | First Real-Data Preparation Record

Scope: public engineering documentation for the 24M Trial, recording the first data preparation and corresponding stage results. See the [Trial experiment report](experiment-results.md) for subsequent training progress and the [project overview](../README.md).

Initial preparation record dated 2026-10-09. The downloads, splits, tokenizer training, and encoding below were executed locally. Each section preserves the execution status at the time it was recorded.

## Actual Artifacts

| Item | Completed result |
|---|---:|
| Fixed-revision Wikipedia subset | 825 documents; 19,985,370 JSONL bytes |
| Original Federal Reserve FOMC minutes snapshots | 32 documents; 1,791,245 JSONL bytes |
| General train / validation | 794 / 31 documents; 4,976,894 / 135,705 tokens |
| Financial train / validation | 31 / 1 documents; 341,309 / 9,209 tokens |
| Candidate task-text exclusion pool | 10,686 distinct headlines covering 10,753 source records |
| Byte-level BPE trained from scratch | 8192 tokens, including six special tokens |
| Payload of four token-shard sets | 10,926,234 bytes, approximately 10.9 MB |
| Actual sources, snapshots, intermediate files, tokenizer, and shards | Approximately 81 MB, plus download-library caches and dependencies |

Data is stored in local Git-ignored directories `data/raw/seed_v1/`, `data/processed/seed_v1/`, and `tokenizer/seed_v1/`. Review sheets are in `data/raw/sentfin_review_v1/`. GitHub publishes only scripts, notebooks without stored output, and aggregate evidence; it does not publish full corpus text or model weights.

Sources and supporting terms: [fixed-revision Wikipedia data card](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md), [Federal Reserve Board notice](https://www.federalreserve.gov/disclaimer.htm), and [fixed-revision SEntFiN author repository](https://github.com/pyRis/SEntFiN/tree/eba43610fe3cb57a3e5773cd6c037da1f8992ad5). See the [general/financial corpus review](research/corpus_sources_20261009.md) and [task-data review](research/stance_sources_20261009.md) for details, limitations, and source selection.

The general sample is a streamed prefix and is not claimed to represent all of Wikipedia. FOMC minutes use a macroeconomic policy register with a distribution different from tweets. The single financial validation document is only an initial diagnostic. This first batch is sufficient for an approximately 1M-token resource pilot. A unique 100M + 25M corpus has not been obtained; cycling through the same data is not new data.

## Processing and Validation

All candidate task texts are frozen first. Language-model validation documents are then selected by source-document hash. SQLite provides global exact deduplication, with the existing SimHash check excluding lexical near-duplicates. Tokenizer training uses only training text, capped at 12 MB of JSONL. Documents are encoded individually into uint16 shards; only the current batch is read and converted to int64 during training. These steps do not require the entire corpus in memory, although tokenizer frequency tables and exclusion indexes still consume RAM.

An actual 24,255,744-parameter model completed one CPU forward and backward pass on these real tokens, with finite loss and gradients. No optimizer update was performed and no weights were saved; this check does not establish financial capability. Evidence is in `reports/seed_model_input_check.json` and `reports/seed_data_preparation.json`.

Input scripts reject incomplete download manifests, hash mismatches, and empty validation partitions. Existing complete artifacts are reused read-only after manifest, configuration, and file-hash verification. Incomplete artifacts are preserved and an error is raised. Notebook subprocess output streams line by line. Both seed notebooks back up to `MyDrive/130U-FinLLM/seeds/seed_v1/`, without replacing summary manifests from earlier smoke tests or other seed versions.

## Continuing in Colab

1. Open [01a_seed_data.ipynb](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/01a_seed_data.ipynb), save a copy, and run in order. It first restores and verifies existing artifacts. Only the initial run downloads bounded sources, builds the exclusion pool, trains its own tokenizer, encodes data, and backs up. It can use CPU and does not start model training.
2. Open [02a_seed_pilot.ipynb](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02a_seed_pilot.ipynb), select a T4, and run in order. It restores the versioned snapshot and pretrains from random weights for 61 updates. If a checkpoint already exists, it verifies and restores progress first.
3. After this approximately 1M-token pretraining segment, inspect the GPU model and actual token count, throughput, peak GPU memory, and validation loss in `run_summary.json` before deciding the next budget. [02b_seed_continue.ipynb](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02b_seed_continue.ipynb) reuses the checkpoint and continues to 300 cumulative steps. On 2026-10-11, the user supplied its 4,914,941-token result and successful final backup verification. See the [T4 training record](T4_PILOT.md); existing results need no retraining. The next stage first expands the corpus and explicitly handles transfer across data versions, rather than automatically running to 100M.

A Colab reproduction records its own source snapshots and tokenizer hash. Web pages and library versions can change, so local hashes cannot unconditionally stand in for those from a future Colab run. Continuation must use the input set matching the original checkpoint. Rerunning 01a verifies and reuses artifacts. Choose a new version directory only when rebuilding is necessary; do not overwrite the existing tokenizer. See the [Colab fix notes](COLAB_DEBUG.md) for old rerun behavior and dependency conflicts.

01a/02a/02b are executed by the user in their own Colab account. Screenshots, logs, and backup output supplied on 2026-10-11 show that the 24M model completed 4,914,941 cumulative tokens of general pretraining and Drive snapshot verification. This does not establish completion of the full pretraining budget, financial post-training, or recovery on a new VM. The agent had not independently read the cloud checkpoint.

## Reviewing 200 Stance Records

`review_200.csv` is the original review sheet. `review_200_ai_proposals.csv` contains record-level AI proposals with Chinese rationales. Original sentiment labels, AI suggestions, and human confirmations are stored in separate columns; the original sheet was not modified. The AI proposals comprise 22 bullish, 16 bearish, 122 neutral, and 40 excluded records. They are not human ground truth and cannot support an accuracy claim.

During review, fill in `asset_identifier`, `human_stance`, and `human_reviewed`. If the asset code is unclear, leave it blank and add a note. `human_stance` can be bullish, bearish, neutral, or exclude. Already-observed price changes default to neutral; bullish/bearish requires an explicit investment direction or outlook. The original published dataset may label entities such as regulators, which cannot all be treated as investable targets.

When this first preparation record was created, zero records had human confirmation. Review later expanded to 2000 records, of which 1993 received explicit user confirmation; see the [review record](../reports/human_headline_review_20261030.json). Eligible literal-entity task samples subsequently formed the 1008-record Trial partitions used for SFT. Confirming headline stance and verifying security identity are separate steps. An AI proposal alone is not human confirmation, and DPO preference pairs require their own review.
