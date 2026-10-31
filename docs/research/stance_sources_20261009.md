# 24M Trial | English asset-stance data review — 2026-10-09

Scope: historical source research for the 24M Trial, preserving decisions from the initial candidate-pool selection. See the [Trial experiment report](../experiment-results.md) for subsequent human review and SFT progress and the [overview](../../README.md) for the overall project.

This review used the Research skill's primary-source workflow to identify data with explicit use conditions and target entities that could support asset-stance annotation. Decision: **use SEntFiN first as a candidate pool for human review, excluding the entire pool of source texts before pretraining. Do not automatically map source positive/negative labels to this project's bullish/bearish labels.** This allows corpus preparation to continue alongside asset mapping and label review.

## Selected candidate: SEntFiN 1.0

The [author repository](https://github.com/pyRis/SEntFiN) is maintained by GitHub account pyRis; [GitHub user metadata](https://api.github.com/users/pyRis) returns Rishu Kumar, matching a paper author. The [pinned LICENSE](https://github.com/pyRis/SEntFiN/blob/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/LICENSE) is MIT, with copyright attributed to Ankur Sinha, Satishwar Kedas, Rishu Kumar and Pekka Malo. The repository explicitly releases the dataset and codes; besides README and LICENSE, its root contains two data files. This license evidence comes from the authors' release repository and does not establish additional rights on behalf of the original news publishers. This review uses it as a local research candidate pool, preserves copyright notices and excludes source text from the project GitHub repository.

[Section 3 of the author paper](https://arxiv.org/html/2305.12257) describes Indian financial news from 2011-2015. Three human annotators assessed entity-specific financial sentiment, followed by consensus and a small number of author adjudications. The paper reports 10,753 news items and 14,404 entity labels. Its neutral category also covers entities such as regulators for which financial sentiment is inapplicable. Original human sentiment annotation and project-confirmed asset stance therefore require separate fields; news sentiment and explicit investment direction are not automatically equivalent.

Pinned revision: `eba43610fe3cb57a3e5773cd6c037da1f8992ad5`.

| File | Pinned download | Bytes | SHA256 |
|---|---|---:|---|
| Main data | [SEntFiN.csv](https://raw.githubusercontent.com/pyRis/SEntFiN/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/SEntFiN.csv) | 1,118,162 | `570af7ba58f724f3fcc414dd5ba80da5b1a3b342134d76c6ebdf53a32c54a92f` |
| Entity-name table | [entity_list_comprehensive.csv](https://raw.githubusercontent.com/pyRis/SEntFiN/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/entity_list_comprehensive.csv) | 90,424 | `6ace045edc54c90bc0695ee5a78afec9f7a5294844ccfd0c9e85a14aeea4916b` |
| License | [LICENSE](https://raw.githubusercontent.com/pyRis/SEntFiN/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/LICENSE) | 1,110 | `961d32986ac3fa2fe2e55ebc7bbae1ba8d632412e98493d73bdf59f0ad8fc65f` |
| Introduction | [README.md](https://raw.githubusercontent.com/pyRis/SEntFiN/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/README.md) | 48 | `543e1779ccb4e13b23107b3c4443891f581041d56adf676daa8db173c730e03a` |

Independent local inspection found CSV fields `S No., Title, Decisions, Words`, 10,753 rows and 10,686 distinct news texts; 67 rows repeat text. `Decisions` contains Python-style dictionary strings mapping entities to `positive/negative/neutral`. Thirteen rows violate Python literal syntax because entity names contain unescaped apostrophes. Those rows were isolated from review sampling, while their source texts remained in the exclusion pool. The remaining rows yield 14,371 target labels. No dates were fabricated or entity names guessed into tickers. The source CSV supplies neither reliable per-record dates nor verified security identifiers.

## Preparation executed

Added `scripts/prepare_sentfin_review.py`, using only the Python standard library. Download mode accepts four pinned files totaling at most 1,209,744 bytes, validating each against fixed size and SHA256. Existing review output directories are not overwritten. Command:

```bash
python scripts/prepare_sentfin_review.py \
  --source-dir data/raw/candidates/sentfin \
  --output data/raw/sentfin_review_v1 --download
```

The original files were downloaded locally and the script was executed during this review. Files remain under Git-ignored `data/raw/`, outside the public repository.

Outputs:

- `review_200.csv`: UTF-8 BOM; 200 distinct news texts with original negative/neutral/positive counts of 67/67/66, including 85 multi-entity items. Sampling is stratified by source label with seed=130, not a simple random sample of the population.
- `task_exclusions.jsonl`: 10,686 distinct source texts, retaining source-ID membership for all 10,753 rows, including unsampled and malformed rows. It excludes tokenizer/pretraining leakage independently of whether project labels are ready.
- `manifest.json`: source version, hashes, counts, isolated rows, sample counts and artifact checksums.
- `SOURCE_LICENSE.txt`, `review_instructions.md`: original copyright/license notice and review rules.

The review sheet contains `id,text,source_entity,source_sentiment,source_entity_count,source_entities,entity_is_literal_substring,proposed_stance,asset_identifier,human_stance,human_reviewed,notes`. At generation, `proposed_stance`, `asset_identifier` and `human_stance` were empty and every `human_reviewed` was false. No human-confirmed or automatically guessed asset-stance labels were generated.

Reviewers should first confirm the target asset, then assign bullish/bearish/neutral under the project rules or record exclude with a reason. Regulators, unclear targets, sarcasm and ambiguous attribution should not be forced into neutral. A report of past price changes does not become investment stance merely because its source label is positive/negative. The first 200 rows identify label-definition and asset-mapping issues before expansion; they do not establish validated financial performance. If there is one reviewer, report single-reviewer annotation.

Dedicated tests cover exclusion of the full candidate-text pool, retention of malformed-row text, absence of fabricated human labels, reproducible sampling, rejection of review-output overwrite and rejection of altered source hashes: `tests/test_sentfin_review.py`, locally **3 passed**.

## Why other sources did not enter training directly

| Source | Primary evidence reviewed | Decision in this review |
|---|---|---|
| SemEval 2017 Task 5 | The [official data page](https://alt.qcri.org/semeval2017/task5/index.php?id=data-and-tools) requires a data release agreement before download and a separate form for the gold standard. The [task paper](https://aclanthology.org/S17-2089/) evaluates company/stock sentiment from -1 bearish to +1 bullish. | Closest task definition, but no completed source agreement was available. No download, agreement signed on the user's behalf or mirror workaround was attempted. |
| FiQA 2018 Task 1 | The [official homepage](https://sites.google.com/view/fiqa/home) specifies `sentence/info/target/sentiment_score/aspects`, scores from -1 to 1 and non-commercial use in both Train/Test Data sections. | Non-commercial research terms and target fields exist, but acquisition from the original link failed. Mirror license labels were not treated as stronger source-rights evidence. |
| Financial PhraseBank | The [original release uploaded by Pekka Malo](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10) identifies the author and archive. The original download was not obtained in this review; common mirrors label it CC BY-NC-SA 3.0. | Not imported based on a mirror license alone. Even if obtained, sentence/sentiment records still require entity and asset mapping. |
| zeroshot/twitter-financial-news-sentiment | The [publisher data card](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) labels it MIT, identifies the Twitter API as its source and lists only text/label fields. Card statements differ from page counts. | Not downloaded. No specified-target field; the publisher's MIT label does not remove upstream X use conditions. |

The official FiQA Task 1 training link is [Drive file 1icRTdnu8UcWyDIXtzpsYc2Hm6ACHT-Ch](https://drive.google.com/open?id=1icRTdnu8UcWyDIXtzpsYc2Hm6ACHT-Ch), which returned 404 on direct request in this review. The test link is [Drive file 1t3E2D7sBD3rG-OHNkvu2Fk7tbMzrMws0](https://drive.google.com/open?id=1t3E2D7sBD3rG-OHNkvu2Fk7tbMzrMws0); only its official linkage was confirmed, not acquisition or exact counts. FiQA Task 2 question-answering data are not Task 1 stance data.

The [X Developer Agreement, version 2026-04-27, III.A(k)](https://docs.x.com/developer-terms/agreement) reviewed here restricts foundation/frontier-model training or fine-tuning; the [Developer Policy, Content redistribution](https://docs.x.com/developer-terms/policy), also limits third-party redistribution. This supports the project decision not to treat API-derived mirrors as automatically approved training inputs; it is not a comprehensive judgment on every historical dataset or legal exception. No X posts were scraped or reacquired during this review.

## Stage decision on 2026-10-09

Q1, source conditions: the SEntFiN author repository provides an explicit MIT release and attribution, and the small pinned files were acquired. Q2, targets and labels: the source provides human entity-specific sentiment; at this research point, project asset mapping and investment-stance semantics still required human review. Q3, quantity: the candidate pool supported an initial 200-row definition review, but the eventual usable asset-stance count was unknown. Corpus preparation could proceed while SFT/DPO labels remained unaccepted. Literal-entity review and SFT were completed later; their results do not retroactively change this historical source decision.
