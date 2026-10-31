# 24M Trial | Local Asset-Stance Review Workspace

Scope: public engineering documentation for the 24M Trial, describing the first-round human-review tool and import contract. See the [Trial experiment report](experiment-results.md) for subsequent task partitions and training results, and the [project overview](../README.md).

Review AI-proposed labels in a self-contained HTML file without a server, account, or external scripts. Source text, suggestions, and human actions stay on the local computer; the page sends no network requests. Generated HTML and row-level data live in Git-ignored directories such as `runs/`.

## Recommended Review Order

1. First select the 40-record rubric-calibration option. It stratifies by AI class, prioritizes simple cases, tries to sample only one record per group, and enables blind review by default. Make your own judgment first, then disable blind review to compare it with the AI suggestion. These are rubric-calibration samples, not a final test set.
2. Once the rubric is stable, proceed in batches of 100. Use the priority-review / difficult-boundary filter to address easily misunderstood examples, or start with an easy batch to learn the interface. For each record, inspect the original text, specified target, and necessary reasoning evidence.
3. `A` confirms the current stance and advances; `1/2/3` selects bearish/neutral/bullish, followed by `A`; `X` explicitly excludes and confirms; `U` marks a record unresolved. `J/K` moves between records. `Z` undoes the last confirmation or unresolved action and returns to that record. Shortcuts do not fire while typing in notes or asset fields.
4. Confirm asset identity and stance separately. AI-proposed asset candidates remain drafts. Enter an identifier and select its verification checkbox after human verification. Stance can be confirmed first, with assets completed later using the stance-reviewed / asset-unverified filter. An asset without human verification is not exported as a verified `asset_identifier`.
5. After each batch, export progress JSON to your project directory; browser storage is not a backup. Reopening the HTML for the same data version usually restores progress automatically. Import the JSON when switching browsers.

Only an explicit confirmation sets `human_reviewed=true`. In this workspace, the flag means that a human has confirmed the stance. It **does not automatically establish acceptance of asset mapping, source rights, event deduplication, or data splits**. Assets have a separate `asset_verified` flag, and every merged row retains `formal_training_ready=false`. Formal data import requires independent gates.

If a record has a second AI review, the page shows the second suggestion and rationale and offers a filter for disagreement between the two AI suggestions. This only helps prioritize disagreements. A second AI pass is not human review and cannot establish human accuracy or a calibrated confidence probability.

## Files and Recovery

- **Progress JSON:** stores only human actions, the data SHA, and the exact row-number list. It restores draft, unresolved, and confirmed states. Import stops explicitly on a different data version, unknown or duplicate row numbers, invalid states, or conflicts with existing local actions, preserving current progress.
- **Confirmed CSV:** contains human fields only for stance-confirmed rows. Formula prefixes are escaped for Excel. Unverified asset fields appear only in `asset_identifier_draft`.
- **Confirmed JSONL:** merges original text, the initial AI proposal, and human decisions while preserving the original text. It supports later source, grouping, mapping, and label checks; it is not a final dataset ready for immediate training.
- **All-row JSONL:** also includes unreviewed and unresolved records without changing their state. Exporting or opening a file does not constitute approval.

Rows with `human_reviewed=false`, `exclude` rows, unverified asset identities, and rows that fail subsequent isolation checks cannot enter formal training solely on the strength of an AI proposal. The proportion of stances confirmed in the browser is neither AI-model accuracy nor an inter-annotator agreement score.

## Generation

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/build_review_workspace.py `
  --input runs/review_packages/review_2000_20261030/ai_annotations.jsonl `
  --output runs/review_packages/review_2000_20261030/review_workspace.html
```

Each JSON line must contain nonempty `id`, English `text`, `source_entity`, and `ai_stance` (or `proposed_stance`). Recommended fields include `ai_rationale`, verbatim `evidence_quotes`, `ai_asset_candidate`, `mapping_status`, `priority`, `difficulty`, `batch_id`, and grouping/source fields. Input accepts only AI records pending review; it must not carry preinserted human approval.

Page data and progress are bound to the input file's SHA256. The generator does not overwrite an existing HTML file with different content. Use a new filename when a new interface version or new data is needed, and retain exported progress JSON. An interface upgrade using the same input can restore progress from the progress file. Progress from input with changed bytes cannot be imported without verification.

## Before Handoff to Training

Prefer the `review_progress_*.json` downloaded through the progress-JSON backup action. It includes drafts, unresolved cases, exclusions, and explicit confirmations, preserving the full action record. The project's `scripts/import_review_progress.py` checks the exact source-file SHA and all ordered IDs before producing progress, label candidates, an asset-verification queue, and a manifest. Rows without decisions remain unreviewed; stance-reviewed rows with unverified assets are listed separately. All outputs remain barred from formal training.

See the [handoff after 03a](POST_DOMAIN_NEXT.md) for the complete command and next steps. The importer accepts actual progress JSON, not AI drafts renamed as human results. The current workspace data SHA is `becc92113c191f807c8a4d73fe8e792cfc7d0ea193f2b2627c4e66f80ac4070c`; other versions require their matching original data.
