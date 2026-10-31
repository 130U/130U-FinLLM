"""Export an unapproved asset-stance review package from frozen local inputs.

Author symbols are historical source candidates, never verified current assets.
No downloads, ticker inference, source-sentiment relabelling, or approvals occur.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path


REVISION = "eba43610fe3cb57a3e5773cd6c037da1f8992ad5"
ENTITY_SHA256 = "6ace045edc54c90bc0695ee5a78afec9f7a5294844ccfd0c9e85a14aeea4916b"
SOURCE_FIELDS = ("id", "text", "source_entity", "source_sentiment", "source_entity_count",
                 "source_entities", "entity_is_literal_substring")
HUMAN_FIELDS = ("asset_identifier", "human_stance", "human_reviewed")
EXTRA_FIELDS = ("source_symbol_candidates", "source_symbol_match", "source_headline_id",
                "text_group_id", "event_group_id")
STANCES = {"bullish", "bearish", "neutral", "exclude"}
REVIEW_NOTE = (
    "# 200-example asset-stance review package\n\n"
    "This directory contains material for review, not a formal training set. source_symbol_candidates comes only from the original historical entity table; "
    "it is not a verified current security identifier or necessarily a US equity. Neither source_sentiment nor proposed_stance is project human ground truth.\n\n"
    "Read text and source_entity for each row. Confirm that the target is a supported asset and verify its identity, market, and symbol, "
    "then fill asset_identifier. If it cannot be verified, set human_stance=exclude and explain in notes. "
    "Classify the specified target as bullish, bearish, neutral, or exclude in human_stance. "
    "Set human_reviewed=true only after completing review. Do not bulk-copy AI suggestions or treat source symbols as verified assets.\n\n"
    "A historical price or earnings change alone does not establish investment direction. Questions without conclusions, ambiguous targets, "
    "regulators, and unsupported assets require review or exclusion. Retain the conditions and targets of conditional bullish or bearish views.\n\n"
    "source_headline_id and text_group_id are draft source/exact-text groups. Review related news and assign a shared "
    "event_group_id (leave blank for investigation if unknown). Subsequent splitting must jointly consider source IDs, near-duplicate clusters, event groups, and multiple targets; "
    "do not split randomly by row. This package does not generate train/dev/calibration/test partitions or DPO pairs.\n\n"
    "This initial rule-review sample has sampling and class biases; it is not the full planned label set. "
    "Single-reviewer completion is not independent agreement between two reviewers.\n\n"
    "Retain this table and the original manifest.json. The script reuses only intact, unmodified initial packages; it will not overwrite edited ones. "
    "Continue editing the current table and use the explicit review-import step afterward.\n"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate CSV columns: {path}")
        if not set(required).issubset(reader.fieldnames):
            raise ValueError(f"Unexpected CSV schema: {path}")
        rows = list(reader)
        fields = reader.fieldnames
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Empty or malformed CSV rows: {path}")
    return fields, rows


def indexed(rows):
    result = {}
    for row in rows:
        if not row["id"] or row["id"] in result or not row["text"].strip() or not row["source_entity"].strip():
            raise ValueError("Review IDs must be unique; IDs, text and source entity must be nonempty")
        if any(row[key].strip() for key in ("asset_identifier", "human_stance")) or row["human_reviewed"].casefold() != "false":
            raise ValueError("This draft exporter accepts only unapproved inputs; existing human review must be preserved")
        result[row["id"]] = row
    return result


def alias_index(rows):
    aliases = defaultdict(set)
    for row in rows:
        symbol = row["Symbol"].strip()
        if not symbol or not row["Entity"].strip():
            raise ValueError("Entity rows need nonempty source Symbol and Entity")
        for alias in [symbol, *row["Entity"].split(",")]:
            if alias.strip():
                aliases[alias.strip().casefold()].add(symbol)
    return aliases


def prepare(review_dir, entity_file, output):
    review_dir, entity_file, output = Path(review_dir), Path(entity_file), Path(output)
    source_manifest = json.loads((review_dir / "manifest.json").read_text(encoding="utf-8"))
    ai_manifest = json.loads((review_dir / "ai_review_manifest.json").read_text(encoding="utf-8"))
    review_file = review_dir / "review_200.csv"
    proposal_file = review_dir / "review_200_ai_proposals.csv"
    if (source_manifest.get("source_repository") != "https://github.com/pyRis/SEntFiN"
            or source_manifest.get("source_revision") != REVISION
            or source_manifest["source_files"]["entity_list_comprehensive.csv"]["sha256"] != ENTITY_SHA256
            or digest(entity_file) != ENTITY_SHA256):
        raise ValueError("Pinned entity source provenance or hash differs")
    if (ai_manifest.get("original_manifest_sha256") != digest(review_dir / "manifest.json")
            or ai_manifest["source_csv"]["sha256"] != digest(review_file)
            or ai_manifest["output_csv"]["sha256"] != digest(proposal_file)
            or ai_manifest.get("not_for_formal_training") is not True
            or ai_manifest.get("human_reviewed_rows") != 0):
        raise ValueError("AI proposal provenance or unapproved state differs")
    base_fields, source_rows = read_csv(review_file, (*SOURCE_FIELDS, "proposed_stance", *HUMAN_FIELDS, "notes"))
    proposal_fields, proposal_rows = read_csv(proposal_file, (*base_fields, "ai_rationale", "ai_reviewer"))
    originals, proposals = indexed(source_rows), indexed(proposal_rows)
    if originals.keys() != proposals.keys() or len(originals) != source_manifest.get("review_rows") or len(originals) != ai_manifest.get("rows"):
        raise ValueError("Review and proposal row identities/counts differ")
    for item, original in originals.items():
        proposal = proposals[item]
        if any(original[key] != proposal[key] for key in (*SOURCE_FIELDS, "notes")):
            raise ValueError("AI proposal changed an original source field")
        if original["proposed_stance"] or proposal["proposed_stance"] not in STANCES:
            raise ValueError("Expected blank original stance and explicit AI-only proposal")
        if not item.startswith("sentfin:") or len(item.split(":", 2)) != 3:
            raise ValueError("Unexpected SEntFiN source row identifier")
    _, entity_rows = read_csv(entity_file, ("Symbol", "Entity"))
    aliases = alias_index(entity_rows)
    fields = [*proposal_fields, *EXTRA_FIELDS]
    if len(fields) != len(set(fields)):
        raise ValueError("Source schema already contains review-package columns")
    prepared, matches = [], Counter()
    for item, original in originals.items():
        candidates = sorted(aliases.get(original["source_entity"].strip().casefold(), ()))
        status = "unmapped" if not candidates else "unique_source_symbol" if len(candidates) == 1 else "ambiguous_source_symbols"
        matches[status] += 1
        normalized = " ".join(original["text"].split()).casefold()
        prepared.append({**proposals[item], "source_symbol_candidates": json.dumps(candidates, ensure_ascii=False),
                         "source_symbol_match": status, "source_headline_id": ":".join(item.split(":", 2)[:2]),
                         "text_group_id": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                         "event_group_id": ""})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(prepared)
    artifacts = {"review_200_to_complete.csv": stream.getvalue().encode("utf-8-sig"),
                 "READ_ME_FIRST.md": REVIEW_NOTE.encode("utf-8")}
    manifest = {
        "schema_version": 1, "status": "unapproved_review_draft", "source_revision": REVISION,
        "source_repository": source_manifest["source_repository"],
        "source_inputs": {"review_200.csv": digest(review_file), "review_200_ai_proposals.csv": digest(proposal_file),
                          "manifest.json": digest(review_dir / "manifest.json"),
                          "ai_review_manifest.json": digest(review_dir / "ai_review_manifest.json"),
                          "entity_list_comprehensive.csv": digest(entity_file)},
        "creation_counts": {"rows": len(prepared), "entity_source_rows": len(entity_rows),
                            "source_symbol_matches": {key: matches[key] for key in ("unique_source_symbol", "ambiguous_source_symbols", "unmapped")},
                            "ai_proposed_stances": dict(sorted(Counter(r["proposed_stance"] for r in prepared).items())),
                            "human_reviewed_rows": 0, "human_stance_labels": 0, "verified_asset_identifiers": 0},
        "formal_training_ready": False,
        "mapping_scope": "Exact trimmed casefold alias matches to author historical source symbols; no current ticker, exchange or security-identity verification.",
        "grouping_scope": "Source headline and normalized text IDs drafted; semantic near-duplicate/event grouping still required before split.",
        "artifacts": {name: {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()} for name, body in artifacts.items()},
        "limitations": ["Source sentiment and AI proposals are not human-approved asset stance labels.",
                        "Source symbol candidates must not be copied into verified asset identifiers without review.",
                        "No train/dev/calibration/test splits or DPO preference pairs are created.",
                        "The 200-row protocol sample is not the planned full-size labelled corpus."],
    }
    body = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    artifacts["manifest.json"] = body
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if output.is_dir() and all((output / name).is_file() and (output / name).read_bytes() == data for name, data in artifacts.items()):
            return manifest
        raise ValueError("Existing review package differs or has manual edits; preserve it and use a new directory for any new draft")
    output.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        (output / name).write_bytes(data)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=Path("data/raw/sentfin_review_v1"))
    parser.add_argument("--entity-file", type=Path, default=Path("data/raw/candidates/sentfin/entity_list_comprehensive.csv"))
    parser.add_argument("--output", type=Path, default=Path("runs/review_packages/review_200_20261030"))
    parser.add_argument("--report", type=Path, help="Optional aggregate report; never includes headline text or row labels")
    args = parser.parse_args()
    manifest = prepare(args.review_dir, args.entity_file, args.output)
    if args.report:
        aggregate = {**manifest, "date": "2026-10-30", "local_package": args.output.as_posix(),
                     "scope": "Local preparation from pinned author files and existing AI-only proposals; no external asset verification or human approval."}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["creation_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
