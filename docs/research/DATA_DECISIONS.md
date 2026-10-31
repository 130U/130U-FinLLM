# 24M Trial | Initial Data Decision Record

Scope: historical engineering research for the 24M Trial, preserving findings and decisions from initial data preparation. GPU pretraining, domain training, and literal-entity SFT were completed subsequently; see the [Trial experiment report](../experiment-results.md) and [project overview](../../README.md).

## Charter

Objective: obtain a traceable starter corpus and produce verifiable inputs for the existing T4, 24M, English asset-stance task. Scope / object / unit of research: model-data preparation / the first real materials under limited resources / a fixed Wikipedia subset, original FOMC minutes, and the SEntFiN candidate task pool. Knowledge depth: working. Type: descriptive. Questions: Q1: Conditions of use; Q2: label fit to the target task; Q3: scale and memory; Q4: actual execution status.

## Canonical Sources

1. Fixed author repositories and data cards: Wikimedia and SEntFiN.
2. Official Wikimedia licensing terms, the Federal Reserve Board's official reuse notice, and FOMC calendars.
3. Fixed-file sizes, hashes, actual content, and local processing results.

Exact URLs, versions, and read-back findings are preserved in [corpus_sources_20261009.md](corpus_sources_20261009.md) and [stance_sources_20261009.md](stance_sources_20261009.md).

## Findings Ledger

| ID | Finding | Evidence | Answers |
|---|---|---|---|
| F1 | The fixed Wikipedia snapshot contains English text and article URLs, with CC-BY-SA/GFDL licensing records. Differences from the current terms are preserved. | Source review F1–F3; `rights_note/source_url` in source rows | Q1 |
| F2 | Original Board material may be copied unless explicitly excepted. Third-party material is not automatically included. | Official disclaimer; source review F5–F6 | Q1 |
| F3 | The SEntFiN author repository declares MIT, but entity sentiment differs from this project's investment stance. The original text provides no verified security-code mapping. | Author LICENSE, paper, and actual CSV; task-source review | Q1/Q2 |
| F4 | 825 general and 32 financial documents were actually obtained, split, and encoded, and the tokenizer was trained to 8192 tokens. | `reports/seed_data_preparation.json`, source and preprocessing manifests | Q3/Q4 |
| F5 | During initial preparation, record-level AI proposals for 200 samples included 40 exclusions and a neutral majority. No human labels for this batch had been confirmed at that time. | `ai_review_manifest.json` in the private candidate directory; not ground-truth labels | Q2 |
| F6 | A real-token CPU forward/backward check passed. The browser could not load the Colab Monaco editor during that session, and this research stage did not execute new GPU training. | `reports/seed_model_input_check.json`; actual browser responses from the original session | Q4 |

## Scope Filter

Excluded: paid news, X scraping, mirrors with unknown upstream training terms, whole-corpus downloads, and return backtests. The retained corpus is bounded and traceable. A database license does not automatically cover the original web pages.

## Object Filter

A Wikipedia prefix and the macroeconomic register of FOMC minutes do not generalize to the Twitter distribution. FOMC pages are retained as fetched snapshots, not claimed to be immutable remote revisions. SEntFiN sentiment requires a separate asset-stance review. Lexical near-duplicate checks do not establish the absence of semantic duplicates.

## Conclusions

- At this data-preparation acceptance point, the initial downloads, frozen exclusion pool, tokenizer, and shards were complete, meeting the conditions for a short 24M GPU pilot. [F1–F4]
- At that time, SFT/DPO still lacked human-confirmed stance and asset mapping. AI proposals could not substitute for that evidence. Subsequent label review and SFT results are recorded separately; source sentiment is not retroactively treated as the project's human ground truth. [F3/F5]
- This research records local preparation and CPU interface validation. See the [training record](../T4_PILOT.md) for subsequent T4 measurements. [F6]

## Execution Plan and Risks at the Time

01a would reproduce the verified preparation process in Colab and back it up; 02a would train approximately 1M tokens and record actual resource use. The plan was to review 200 records manually, then expand according to available scale. With only one financial validation document, biased source selection, and a narrow corpus distribution, this round would make no financial-performance claim or launch the long 100M/25M budget.

Research status: 1. Source ranking complete; 2. scope complete; 3. descriptive review and evidence table complete; 4. scope filtering complete; 5. object filtering complete; 6. conclusions, execution evidence, and open items complete. Final review was this team's source and consistency check, not external independent reproduction.
