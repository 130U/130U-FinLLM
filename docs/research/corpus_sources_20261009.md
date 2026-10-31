# 24M Trial | Initial corpus-source review

Scope: historical source research for the 24M Trial, preserving the evidence, selection decisions and stage plan from 2026-10-09. See the [Trial experiment report](../experiment-results.md) for subsequent execution results and the [overview](../../README.md) for the overall project.

Review date: 2026-10-09. The intended use is English pretraining from scratch and continued financial-domain pretraining; financial explanatory text is not automatically converted into bullish/bearish asset-stance labels. Bounded downloads followed this source review and are recorded below; source approval alone was not evidence of completed training.

## Research scope and sources

Knowledge depth: working. Objective: identify an initial real corpus with fixed versions, provenance and a reuse basis that can be imported within bounded Colab resources.

- Scope / Object / Subject: corpora for a language model trained from scratch / initial 1-5M-token engineering runs and a later 100M-token budget / Wikipedia, FineWeb-Edu and original English Federal Reserve Board minutes.
- Type: descriptive. The model and data pipeline already exist; this review checks executable source requirements.
- Q1: What supports reuse of the text? Q2: What are the pinned version, fields and acquisition endpoint? Q3: How are memory and disk use bounded?
- Canon: publisher data cards and APIs first; official Wikimedia, Federal Reserve Board and Common Crawl terms next; direct inspection of pinned shards and HTML last. No secondary recommendation articles were used.

| Priority | Primary source group | Purpose |
|---|---|---|
| 1 | [Wikimedia HF data card](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md), [dataset API](https://huggingface.co/api/datasets/wikimedia/wikipedia) | Pinned SHA, English configuration, fields and dataset license |
| 1 | [Wikimedia official terms, section 7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content) | Attribution, licensing, modification and redistribution requirements |
| 1 | [FineWeb-Edu data card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md), [Common Crawl terms](https://commoncrawl.org/terms-of-use) | Distinction between database licensing and original webpage rights |
| 1 | [Federal Reserve Board use notice](https://www.federalreserve.gov/disclaimer.htm) | Public-domain statement and exceptions for original Board material |
| 1 | [Official FOMC calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm), [minutes example](https://www.federalreserve.gov/monetarypolicy/fomcminutes20241218.htm) | Actual endpoints, release dates and body selector |
| 2 | [Hugging Face Datasets streaming documentation](https://huggingface.co/docs/datasets/stream) | Streaming iteration, buffering and shard behavior |

## Evidence ledger

| ID | Finding | Source / evidence | Question |
|---|---|---|---|
| F1 | The reviewed `wikimedia/wikipedia` SHA is `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`; English configuration `20231101.en`, split `train`; four string fields: `id, url, title, text`. The card declares `cc-by-sa-3.0` and `gfdl`. | Direct reads of the pinned card and API | Q1/Q2 |
| F2 | The English configuration contains 6,407,814 records and 11,630,929,031 download bytes. The first shard, `20231101.en/train-00000-of-00041.parquet`, is 420,296,449 bytes; its LFS SHA256 is `382e7f6f09e488b24793a7f7cfc659879d5a22da2cf2efec6491665f0c019677`. | Card and [pinned tree API](https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en) | Q2/Q3 |
| F3 | Wikimedia terms at review time use CC BY-SA 4.0, whereas the HF 2023 snapshot card names 3.0. Preserve article URLs for attribution, any additional imported attribution, modification notices and the applicable license. | Official section 7; retain this version distinction rather than rewriting the snapshot license | Q1 |
| F4 | FineWeb-Edu SHA: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`; `sample-10BT` maps to `sample/10BT/*`, with fields including `text/id/url/date`. The publisher names ODC-By 1.0 and references Common Crawl terms, which state that crawled content may have original-site terms and third-party copyright. | [FineWeb-Edu API](https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu), card and Common Crawl sections 2-3 | Q1/Q2 |
| F5 | The Federal Reserve Board notice states that information is public domain unless otherwise marked, with Board attribution requested. Non-Board images, graphics or other material require permission from the relevant third parties; links do not extend that permission. | Official disclaimer, marked updated 2024-08-02, read 2026-10-09 | Q1 |
| F6 | The calendar supplies actual 2021-2026 minutes links and release times. Sample `fomcminutes20241218.htm` returned HTTP 200 with 126,531 HTML bytes and an `#article` body node. The meeting was 2024-12-18; minutes were released 2025-01-08. | Official calendar and direct HTML inspection | Q2 |
| F7 | `streaming=True` creates an IterableDataset without first converting the entire source into a local Arrow dataset. Parquet still requires decoding and read buffers; streaming is not zero-memory processing. | Official Datasets streaming documentation | Q3 |

## Selection

Scope filter: exclude social-media stance labels, return backtests, paid financial news and full downloads of every candidate corpus. These are outside this unlabeled-corpus review.

Object filter: FineWeb-Edu is not a default rights-approved corpus; an ODC database license does not automatically establish training permission for every source webpage. Select only original Board text on `federalreserve.gov`; do not extend that treatment automatically to `federalreservehistory.org`, regional banks, third-party images or external links. Retain Wikipedia license records and article attribution; do not call Wikipedia public domain.

Decision: start with **a bounded Wikipedia subset for general text and original Federal Reserve Board FOMC minutes for a small financial corpus**. This implementation choice follows F1-F7. Encyclopedia and macroeconomic prose still differ substantially from Twitter asset-stance text.

## Minimum import plan

### General English

Use the existing `fetch-corpus` command without downloading all English Wikipedia. Start with `max_documents=5000` and a hard JSONL output cap of `20_000_000` bytes, stopping at either limit. Measure project-tokenizer tokens after encoding; bytes are not token counts.

```bash
python -m finllm fetch-corpus \
  --dataset wikimedia/wikipedia \
  --revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa \
  --config 20231101.en --split train --text-field text \
  --max-documents 5000 --max-bytes 20000000 \
  --training-rights-reviewed \
  --rights-note "Wikipedia contributors; original article URL retained; dataset card CC-BY-SA-3.0/GFDL; preserve applicable attribution and share-alike notices; see corpus_sources_20261009.md" \
  --output data/raw/wikipedia_en_pilot.jsonl
```

The importer preserves `source_url`, revision, rights_note, a manifest and file SHA256. Selection follows source order, producing a reproducible convenience subset rather than a representative random sample. Fix sampling seeds and shard order when expanding. Do not call `list(dataset)` or whole-table `to_pandas()` on the stream.

### Financial English

Initially use only 2021-2024 minutes links parsed from the actual calendar, sorted by meeting date: at most 32 documents, 8 MB of JSONL output, 2 MB per response and at least one second between serial requests. Do not guess URLs from dates or follow off-domain links. Record actual public release dates in `published_at`, not only meeting dates. This selection also prevents the sample from silently changing as later 2026 meetings are added. [F6]

Extract `#article`, retaining titles and paragraphs while removing navigation, footnote return links, participant lists and other unnecessary page structures; record this cleaning. Each record should preserve URL, title, meeting date, release date, UTC retrieval time, original HTML SHA256, extracted-text SHA256, extractor version, Board attribution, the public-domain basis URL and the copyright-exception check. Skip and log pages whose body separately marks third-party rights. Do not copy the disclaimer into training text.

Minutes form a small macroeconomic starting corpus. If actual tokens fall below 25M, report that amount rather than replaying text and calling it 25M independent tokens. Original Board Monetary Policy Reports and Financial Stability Reports can be reviewed separately for expansion; their combined size was not measured in this review.

## Storage and training-entry requirements

The following are engineering budgets, not measured Colab peaks: up to 20 MB general output plus 8 MB financial output and metadata; streaming still needs temporary caches. A `uint16` token file has a payload of `2 × tokens` bytes: 1M = 2 MB, 5M = 10 MB, 100M = 200 MB in decimal units, plus indices, manifests, source text and model checkpoints. Avoid retaining full HF download caches and redundant complete checkpoints.

Source approval addresses the basis for input use. Existing project entry requirements still apply:

1. Freeze a real stance-evaluation source-text exclusion pool before deduplication and leakage checks; missing labels do not justify treating an empty pool as complete.
2. Split language-model train/dev by document rather than by chunk; train the tokenizer only on training text.
3. Record actual independent documents, bytes, tokens, distributions and duplicate fractions after cleaning.
4. On T4, first measure a short 24M segment for peak memory, throughput and recovery before expanding the budget; this note does not establish runtime estimates.

## Verification status and risks

- Verified: two immutable public-API SHAs, Wikipedia fields and first-shard size, official reuse terms, the FOMC calendar and sample body selector.
- This source-review step did not run a tokenizer, T4 training or financial-judgment evaluation. Subsequent bounded download execution is recorded below.
- Remaining issues at that point: real asset-stance labels and their permissions, the Wikipedia current-terms/snapshot-card license distinction, financial-corpus coverage, actual cache peaks and training throughput.

Research steps completed: 1 Canon; 2 Charter; 3 descriptive-source reading and evidence table; 4 Scope filter; 5 Object filter; 6 conclusions and open issues.

## Subsequent actual download record

On 2026-10-09, `scripts/fetch_seed_corpora.py --output data/raw/seed_v1/corpora` was executed with `datasets==4.8.5`, `requests==2.34.2` and `beautifulsoup4==4.15.0`. HF access was public, unauthenticated streaming; the full English dataset was not downloaded. Script defaults were 20 MB / 10,000 general documents and 5 MB / 32 financial documents.

| Local artifact (source texts excluded from Git) | Actual documents | Actual bytes | SHA256 |
|---|---:|---:|---|
| `general_candidate.jsonl` | 825 | 19,985,370 | `cee8dfb34401588ca8ecde6ec8d13d1ffa275efb5e0ca1391499da16e6a1b22f` |
| `finance_candidate.jsonl` | 32 | 1,791,245 | `f864068487758cdcec3142a526c9abcc5da7e92f14f289f201078f34e424f1c6` |

Candidate JSONL files total 21,776,615 bytes. The complete output directory, including 32 original HTML pages, calendar HTML, actual links and manifests, totals 26,299,150 bytes. This is artifact storage, not process peak memory or external HF cache use. All 32 financial links succeeded with none skipped; meetings span 2021-2024 and actual release dates are retained. The final 2024-12-18 minutes were released 2025-01-08. Manifest status was `complete`; both JSONL hashes matched on readback.

Four offline tests in `tests/test_fetch_seed_corpora.py` passed: same-domain actual links/release dates, body extraction/copyright-marker rejection, general JSONL byte caps/article attribution, and rejection of existing-output overwrite. Two explicitly skip without optional HTML dependencies; install beautifulsoup4 in CI requiring complete parsing coverage.

These files remain unlabeled source candidates with `candidate_only=true`. Deduplication, train/dev splitting, the real asset-stance exclusion pool, tokenizer training and actual token counts belong to subsequent steps.
