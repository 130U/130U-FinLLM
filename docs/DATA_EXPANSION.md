# 24M Trial | Corpus Expansion and Weight Transfer into a New Stage

Scope: public engineering documentation for the 24M Trial, covering the data and weight transfer for `expansion_v1`. See the [project overview](../README.md) for the broader project.

Created on 2026-10-11; expansion results updated on 2026-10-30. These entry points build on the completed 300-step `seed_v1` run. The original data, 8192-token vocabulary, checkpoint, and report archives are retained; new artifacts live separately under `expansion_v1`. The user completed 01b, 02c, and 02d, reaching 1220 cumulative expansion steps and 19,988,195 effective tokens, with successful verification reported for the final snapshot. See the [1220-step record read back from the supplied output](../reports/colab_t4_expanded_continuation_20261030.json). The subsequent [03a financial stage](DOMAIN_HANDOFF.md) also completed 74 steps and 1,211,960 effective tokens. Task SFT results are documented separately in the [Trial experiment report](experiment-results.md).

## Execution Order

1. Open [01b: corpus expansion](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/01b_expand_data.ipynb). A CPU runtime is sufficient. Authorize Drive and run the cells from the top.
2. Inspect the actual unique-document counts, deduplication and exclusion counts, and token counts in `expansion_summary.json`. Confirm `verified: true` for all three backups: `corpora`, `prepared`, and `processed`.
3. Open [02c: expanded general pretraining](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02c_expanded_pretrain.ipynb), select a T4, authorize access, and run in order.
4. 02c first measures the parent checkpoint's loss on a bounded portion of the original seed validation shards, verifies the first update of the new stage, and then trains to 610 cumulative updates in that stage. It finally remeasures loss on the same original validation shards and verifies and backs up the run snapshot.

Both entry points include complete environment installation and an import check in the current kernel. 01b installs the project and `[data]` dependencies in a single pip transaction while retaining Colab's PyTorch; 02c installs the training dependencies. There is no need to repeat 01a, 02a, or 02b. The paths in the old `03_domain_pretrain.ipynb` still do not match these versions; do not run it directly.

## Sources and the Actual Increment

General candidates use a larger prefix of the same fixed Wikipedia revision, with default limits of 200,000,000 bytes and 50,000 documents. Financial candidates follow actual links in the Board's official historical indexes and recent calendars to obtain 2011–2025 FOMC minutes, with default limits of 128 documents and 12,000,000 bytes. Rights information, attribution, actual publication dates, and original HTML snapshots are retained. See the [source review](research/expanded_corpus_sources_20261011.md).

The expansion candidates include the old seed prefix and 2021–2024 minutes, so source bytes are not a measure of new data. The preparation pipeline first excludes source identities, duplicates, and lexical near-duplicates from all old seed training and validation documents, together with the frozen task-text pool. Only the remaining documents are split into new train/dev sets. The original seed validation set remains unchanged; the expanded corpus has its own separately reported validation set.

This stage reuses the original hash of `tokenizer/seed_v1/tokenizer.json`; it does not train a new tokenizer. Text is processed line by line, token shards use `uint16`, and training reads them through mmap. The full English Wikipedia is never materialized as an in-memory list. JSONL output limits exclude Parquet buffers, HTML and restoration caches, cleaned text, and checkpoints, so they are not a guarantee of total disk usage.

The actual amount of new data must come from this run's `expansion_summary.json`. A small collection of macroeconomic minutes cannot automatically supply 25M unique financial tokens, and replay does not create new documents. 01b also prepares an 80% financial / 20% general `domain_mix` without cycling through source content; its size is limited by the financial source available. This 02c entry point does not use that mix to start the financial stage or generate bullish/bearish asset labels. Checkpoint handoff, configuration, and a notebook for the financial stage require a separate delivery.

## Actual Local Preparation Results

The default source limits were used for a real local download and CPU build; see the [machine-readable report](../reports/expansion_local_validation_20261011.json). From 28,486 Wikipedia and 120 FOMC candidates, the pipeline excluded the old seed's 825 general and 32 financial documents. The 10,686 frozen task texts were included in exclusion checks. The original seed manifests and 8192-token tokenizer hash remained unchanged.

| New shards | Documents | Tokens |
|---|---:|---:|
| General train | 26,267 | 41,499,364 |
| General validation | 1,394 | 2,235,968 |
| Financial train | 84 | 972,438 |
| Financial validation | 4 | 50,999 |

`domain_mix` contains 1,215,545 tokens: 972,436 financial and 243,109 general, an 80/20 ratio. It reorganizes the training sources above and must not be counted again as independent new data. The financial content consists of macroeconomic minutes; it does not yet cover tweet stance or human asset labels.

The local build took 544.91 seconds. Process RSS sampled every 0.1 seconds peaked at 361,443,328 bytes, approximately 344.7 MiB. Candidates and caches occupied 247,243,919 bytes, cleaned artifacts 248,168,041 bytes, and token artifacts 91,966,277 bytes: 587,378,237 bytes in total, approximately 560.2 MiB. These figures measure only the local CPU run and those three artifact categories; they exclude the old seed, model checkpoints, and external caches. Sampled RSS is neither an exact allocation peak nor T4 GPU memory.

At the 01b/02c delivery, all 205 local tests passed in 274.58 seconds, checks for 12 notebooks passed, and `pip check` found no broken dependencies. Code cells were executed in isolated Jupyter kernels with local substitute inputs. Separately, the user supplied successful 01b build output, verified backups for all three components, and 02c's 610-step training and final backup output. The agent did not independently read back Drive files or download cloud weights. These evidence sources are recorded separately; local tests are not treated as cloud reruns.

## Weight Transfer and Resume Semantics

The original `--resume` requires matching data hashes and cannot be used to disguise new data as the old training stream. 02c explicitly uses `pretrain_transition.kind = data_expansion`, freezes the parent checkpoint, and verifies its SHA256, formal pretraining identity, progress of at least 300 steps, and tokenizer.

On the first entry into the new stage, only the parent model weights are loaded. AdamW, the scheduler, AMP scaler, random state, and data cursor are initialized afresh; stage update and token counters start at zero. Reports list parent-model tokens, new-stage tokens, and the total processed along this model's training path. Architecture and tokenizer remain unchanged.

The segment defaults to 610 updates, with batch 4 × gradient accumulation 8 × sequence length 512, equivalent to 9,994,240 nominal positions. Padding and skipped AMP updates affect effective counts; actual tokens come from the training report. The new stage uses 50 warmup steps and a 6104-step schedule, with the current stopping budget set to 610 steps. The 100M-token ceiling remains a configuration budget. Training stops explicitly if the new general shards cannot support this segment without replay.

Once the new stage has created a checkpoint, reruns of the same version must resume under its strict provenance checks; they cannot restart from the parent weights. Changes to the tokenizer, corpus, precision, or semantic configuration raise an error and require a separate experiment while preserving the old version. Checkpoints sync to Drive every 10 steps, and reports are archived separately. If training, measurement on the original validation set, or backup fails, the notebook's success flag is false and the subsequent full-backup cell stops.

## Storage and Acceptance

| Artifact | Colab working directory | Under the Drive project directory |
|---|---|---|
| Candidate and cleaned text | `data/raw/expansion_v1/{corpora,prepared}` | `expansions/expansion_v1/{corpora,prepared}` |
| Token shards and build reports | `data/processed/expansion_v1` | `expansions/expansion_v1/processed` |
| Frozen parent checkpoint | `checkpoints/expansion_v1_parent` | `expansions/expansion_v1/parent` |
| New-stage run snapshot | `runs/expansion_v1_pretrain` | `expansions/expansion_v1/run` |
| Periodic checkpoints | Same as above | `live_checkpoints/expansion_v1_pretrain` |
| Before/after measurements on the original validation set and segment reports | Saved by the entry point | `expansions/expansion_v1/{retention,report_archive}` |

The Drive project root is `MyDrive/130U-FinLLM`. The original `seeds/seed_v1` is a read-only input and is not overwritten by these entry points.

01b is complete only with its actual build report and verified backups of all three components. 02c is complete with the target new-stage step count, before/after measurements on the original validation set, the formal segment report, and final `verified: true`. A connection dialog does not establish that background training has stopped: wait for reconnection, confirm the run has ended or stopped, and then restore from the first cell to avoid launching two training processes concurrently.

Delivery of these two entry points does not mean that all general pretraining, financial-domain training, or post-training is complete. Subsequent segments still depend on the actual corpus size and new-stage measurements, a matching financial-mixture configuration, target-asset mapping, and human stance labels.

The 610-step 02c record and its retention measurements are preserved separately. 02d then strictly resumed the same expansion experiment to 1220 steps. The read-back general validation loss was 3.6929; loss on the original seed retention subset changed from 4.4032 to 3.9842 across that continuation. The final backup output reported `verified: true` for seven files totaling 616,109,865 bytes. The two validation sets remain separately reported. The next stage, 03a, verifies and freezes the 1220-step model, then transfers its weights into a new financial-stage optimizer state. It does not overwrite the original 02c measurements or seed artifacts.
