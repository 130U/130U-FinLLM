# 24M Trial | Expanded-Corpus Source Review and Acquisition Contract

Scope: historical source research for the 24M Trial. The download scale and acquisition contract on this page belong to `expansion_v1`. See the [expansion record](../DATA_EXPANSION.md) for subsequent training results and the [project overview](../../README.md).

Review date: 2026-10-11. This work follows the completed 300-step `seed_v1` run. The file records only the sources, acquisition boundaries, and verification of new corpus candidates; it does not establish completion of new Colab training.

## Sources and Selection

This round continues with the two sources reviewed in the first round: a fixed English Wikipedia snapshot and original Federal Reserve Board FOMC minutes. It increases the bounded Wikipedia prefix and extends FOMC meetings to 2011–2025. Source order is fixed; this convenience subset is not described as a random or representative sample.

| Source | Primary evidence | Acquisition contract for this round |
|---|---|---|
| `wikimedia/wikipedia` | [Fixed data card](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md), [fixed raw data card](https://huggingface.co/datasets/wikimedia/wikipedia/resolve/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md) | Fixed revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, configuration `20231101.en`; stream `id/url/title/text` from `train`. Defaults: at most 200,000,000 JSONL bytes, 50,000 documents, and 1MB of body text per document. |
| Recent FOMC minutes | [Official calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) | Follow actual HTML links for 2021–2025 meetings and preserve the actual publication date from the page. Do not guess minutes URLs or treat a meeting date as a publication date. |
| Historical FOMC minutes | [Official historical-year index](https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm), [2011 page](https://www.federalreserve.gov/monetarypolicy/fomchistorical2011.htm), [2020 page](https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm) | Follow year links actually present in the index to obtain 2011–2020 pages, then follow their minutes HTML links. Defaults: at most 128 documents in total and 12,000,000 financial JSONL bytes. |

The current official calendar covers only recent meetings; reusing that page alone cannot obtain 2011–2020 minutes. Actual 2011 minutes use the older `#printThis.generalContentText` and `#leftText` page structure; newer pages use `#article`. The new extractor checks both structures separately. The older structure also requires an explicit FOMC minutes title. Articles that fail structural checks are logged as skipped rather than falling back to scraping the whole page. [Actual 2011 minutes](https://www.federalreserve.gov/monetarypolicy/fomcminutes20110126.htm)

## Reuse Basis and Attribution

The fixed raw Wikipedia data card was read directly on 2026-10-11: HTTP 200, 130,835 bytes. It records `cc-by-sa-3.0` and `gfdl`. Wikimedia's current terms use CC BY-SA 4.0 and allow attribution through an article URL. This version difference remains in each record's `license` and `rights_note`; the text is not reclassified as public domain. Original article URLs, snapshot revision, text hashes, and retrieval times are retained. [Official Wikimedia terms §7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content)

The Federal Reserve Board states that information on its website is in the public domain unless otherwise indicated and requests attribution to the Board. Non-Board material and external links do not automatically receive the same permission. This round therefore takes only original FOMC body text on the Board website, without following external links or collecting images, PDFs, or other meeting materials. Text with an explicit copyright notice is skipped with the reason retained. Each record preserves the Board URL, publication date, original HTML hash, text hash, extraction notes, and public-domain basis. [Board website notice](https://www.federalreserve.gov/disclaimer.htm)

This round did not scrape X/Twitter, paid financial articles, or new commercial datasets, and did not generate bullish/bearish labels from minutes. The sources remain weighted toward encyclopedic and macroeconomic policy writing. Asset-stance supervision requires a separate review.

## Memory, Storage, and Interruption Recovery

The entry point is `scripts/fetch_expansion_corpora.py --output data/raw/expansion_v1/corpora`. Its output contract comprises `general_candidate.jsonl`, `finance_candidate.jsonl`, and a schema 1 `manifest.json`, compatible with existing source-snapshot verification. `status: complete` is written atomically only after both components verify successfully. Reruns read back and verify complete results without overwriting files.

Wikipedia documents are written to JSONL one at a time. Every 100 source rows, a progress log commits cumulative byte counts and SHA256. After interruption, the fixed source is traversed again to the committed boundary, the committed prefix is verified, and only the uncommitted tail after that boundary is discarded before writing continues. This rereads some source data but does not accept committed articles twice. Initial setup and unverifiable corruption states produce explicit errors; the process cannot silently rebuild from an empty file.

Fed requests reuse the original `FedFetcher`: HTTPS `www.federalreserve.gov` only, sequential requests at no more than one per second, a maximum 2MB per HTML response, and bounded timeouts and retries. Each page first commits an atomic cache containing its URL, byte count, body SHA256, and HTML, then retains the original HTML snapshot. After interruption, verified page caches rebuild the unfinished financial component; completed components are not fetched again. A parameter or source-version change requires a new expansion version, preserving old output.

Streaming avoids downloading all of English Wikipedia or materializing it as an in-memory list. Parquet decoding and network reads still require buffers, however, so this is not zero-memory processing. [Official Hugging Face Datasets streaming documentation](https://huggingface.co/docs/datasets/stream)

The default 200MB + 12MB limits apply to candidate JSONL bytes, not tokens, and exclude some caches, cleaned text, and checkpoints. Formal artifacts also retain HTML, recovery caches, cleaned output, and `uint16` token files; the latter have a payload of two bytes per token. Actual storage and token counts are recorded after the build. These sources alone are not promised to supply 100M general or 25M unique financial tokens.

## Conditions Before Training

- Retain the original `seed_v1` tokenizer unchanged. New candidate data must not retrain or replace it.
- Exclude original seed train/validation document identities, duplicate and near-duplicate text, and frozen task-evaluation text. A larger Wikipedia prefix cannot itself be counted as entirely new documents.
- Split new train/dev data by document, retain the original validation set for comparable evaluation, and separately report new unique documents and actual tokenizer token counts.
- During weight transfer, record the original 300-step checkpoint hash and new-corpus provenance. New data cannot be presented as an exact optimizer/cursor continuation of the same stream. Subsequent recovery must verify the new stage's independent run contract.

## Validation Record

All 11 offline tests in `tests/test_expansion_sources.py` passed. They cover read-only reuse of complete results; restoration of committed Wikipedia prefixes, handling of uncommitted tails, and rejection of corrupted prefixes; interruption between the completion marker and atomic rename; allowlisting actual links and publication dates; extraction from older page structures and exclusion of copyright notices; financial-cache recovery and corruption rejection; rejection of changed parameters; and rejection of modified original HTML snapshots.

A small real-network check ran on 2026-10-11: general data capped at 1MB and 30 documents, and financial years 2011 and 2025 capped at two documents and 200KB. Financial records were selected chronologically, yielding two older-structure examples from 2011. All candidate files remained in the local Git-ignored directory `data/raw/expansion_network_probe_20261011/corpora`; source text was not uploaded.

| Small-scale candidates | Actual documents | Actual bytes | SHA256 |
|---|---:|---:|---|
| General English | 28 | 980,278 | `0dc2e5955a33057ef735e99cdea8dcfebc19c2746f4102d3a0e1a9312fb270e6` |
| FOMC financial English | 2 | 116,347 | `ecbc426ab4bbfec62f1ba03efe9bfde3e51037276406868216ee9bae8658516d` |

The small-scale directory, including recovery caches, totaled 2,262,672 bytes. No financial records were skipped; the manifest was `complete`. Rerunning with identical parameters returned `reused: true`. This verifies acquisition, parsing, and reuse only. It does not establish formal corpus scale, training throughput, or model performance. Subsequent actual statistics from the default bounded download and preparation belong in the stage report.

## Actual Results from the Default Bounded Download

The default 200MB Wikipedia limit and 2011–2025 FOMC acquisition were executed on 2026-10-11. The manifest completed at `2026-10-11T10:42:25.249898+00:00`. Local files are in `data/raw/expansion_v1/corpora`; original text was not committed to Git.

| Default candidates | Actual documents | Actual bytes | SHA256 |
|---|---:|---:|---|
| General English | 28,486 | 199,996,572 | `5b2b8def467868f177afd438c27ba4254fe41d746a5bbabb4bbbf3046769d547` |
| FOMC financial English | 120 | 6,977,681 | `5a9b357c4b979b5e85dc7622e28c5793068baeb67a48abbd218e81e7d8219a0a` |

The directory, including source HTML, recovery caches, and metadata, totaled 247,243,919 bytes. This is output-directory size; it excludes external HF buffers/caches and preparation artifacts, and is not peak process memory. Financial acquisition saved 12 discovery-page snapshots. All 120 minutes passed the reviewed old/new body-structure and copyright-notice checks, and `skipped` was empty.

These are source-candidate counts and include overlap with the old seed. They cannot be reported as new unique documents. Subsequent build reports provide new-document counts, partitions, and tokenizer token counts. This record did not start new-stage Colab training.
