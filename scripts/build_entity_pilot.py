"""Build a small, explicitly entity-scoped, human-reviewed research pilot.

The result retains source headlines and must stay private. A literal entity is
not a verified security or a ticker. Event/semantic duplicates remain unaudited.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import tempfile

from finllm.data import file_sha256, normalized_text, prepare_task, read_jsonl, write_json, write_jsonl

SOURCE = "https://github.com/pyRis/SEntFiN/tree/eba43610fe3cb57a3e5773cd6c037da1f8992ad5"
RIGHTS = ("SEntFiN author repository publishes the dataset under MIT; source publication "
          "headlines may have third-party rights not independently verified. Private research pilot.")
LABELS = {"bullish", "bearish", "neutral"}


def build(review_csv, review_manifest, seed_exclusions, expansion_exclusions,
          seed_pretrain_manifest, expansion_pretrain_manifest, domain_manifest, output):
    review_csv, review_manifest, output = map(Path, (review_csv, review_manifest, output))
    if output.exists():
        raise FileExistsError("Output exists; preserve it and choose a new version")
    record = json.loads(review_manifest.read_text(encoding="utf-8"))
    if (record.get("format") != "130u-human-confirmed-headline-review-v1"
            or record.get("provenance", {}).get("type") != "explicit_user_conversation_confirmation"
            or record.get("output_csv", {}).get("sha256") != file_sha256(review_csv)):
        raise ValueError("CSV and explicit user confirmation provenance do not match")
    for path, manifest in ((seed_exclusions, seed_pretrain_manifest),
                           (expansion_exclusions, expansion_pretrain_manifest),
                           (expansion_exclusions, domain_manifest)):
        meta = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if meta.get("eval_exclusion_sha256") != file_sha256(path):
            raise ValueError("Pretraining manifest does not bind supplied frozen exclusions")
    seed_texts = {normalized_text(row["text"]) for row in read_jsonl(seed_exclusions)}
    expansion_texts = {normalized_text(row["text"]) for row in read_jsonl(expansion_exclusions)}
    with review_csv.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 2000 or len({r["id"] for r in source_rows}) != 2000:
        raise ValueError("Expected the complete unique 2000-row confirmed source")
    rows, skipped = [], Counter()
    for row in source_rows:
        label, entity, text = row["human_stance"], row["source_entity"].strip(), row["text"]
        if row["human_reviewed"] != "true" or label not in LABELS:
            skipped["unreviewed_or_excluded"] += 1
            continue
        if row["headline_only_stance"] != label:
            raise ValueError("Human label differs from confirmed headline label")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,14}", entity):
            skipped["target_not_single_identifier"] += 1
            continue
        if not re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", text, re.IGNORECASE):
            skipped["target_not_literal_boundary"] += 1
            continue
        if normalized_text(text) not in seed_texts or normalized_text(text) not in expansion_texts:
            raise ValueError("Candidate was not in both frozen pretraining exclusion sets")
        if not row["source_headline_id"]:
            raise ValueError("Missing original headline ID")
        rows.append({"id": row["id"], "text": text, "target": entity.upper(), "label": label,
                     "target_scope": "literal_entity_not_verified_security",
                     "human_reviewed": True, "asset_verified": False,
                     "training_allowed": True, "source": SOURCE, "rights_note": RIGHTS,
                     "group_id": row["source_headline_id"], "event_id": row["event_group_id"],
                     "source_headline_id": row["source_headline_id"], "synthetic": False})
    if len(rows) < 100:
        raise ValueError("Too few eligible reviewed literal-entity rows for the bounded pilot")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".entity-build-", dir=output.parent) as scratch:
        scratch = Path(scratch)
        dataset = scratch / "dataset"
        dataset.mkdir()
        write_jsonl(dataset / "reviewed_entity_candidates.jsonl", rows)
        result = prepare_task(dataset / "reviewed_entity_candidates.jsonl", dataset)
        parts = ("train", "development", "calibration", "test", "calibration_temperature", "calibration_threshold")
        counts = {name: Counter(r["label"] for r in read_jsonl(dataset / f"{name}.jsonl")) for name in parts}
        if any(not counts[name] for name in parts):
            raise ValueError("At least one held-out partition is empty")
        names = sorted(p.name for p in dataset.iterdir())
        manifest = {"format": "130u-entity-pilot-v1", "scope": "literal single-identifier entity; not a verified security",
                    "private_research_only": True, "source_url": SOURCE,
                    "rights_note": RIGHTS, "source_review_sha256": file_sha256(review_csv),
                    "source_review_manifest_sha256": file_sha256(review_manifest),
                    "seed_exclusions_sha256": file_sha256(seed_exclusions),
                    "expansion_exclusions_sha256": file_sha256(expansion_exclusions),
                    "source_rows": len(source_rows), "candidate_rows": len(rows),
                    "candidate_labels": dict(Counter(row["label"] for row in rows)),
                    "skipped": dict(skipped), "split": result, "class_counts": {k: dict(v) for k, v in counts.items()},
                    "user_verified_asset_count": 0, "semantic_event_audit_complete": False,
                    "test_interpretation": "non-temporal, lexical/source-group-isolated development pilot only; not final independent or X evaluation",
                    "cautions": ["Original author MIT license does not independently guarantee publisher rights.",
                                 "Target identity is literal text, not a verified or tradable security.",
                                 "Semantic event paraphrases and historical time are not verified.",
                                 "Production inference does not currently support identical bare entity aliases."],
                    "files": {name: {"sha256": file_sha256(dataset / name), "bytes": (dataset / name).stat().st_size}
                              for name in names}}
        write_json(dataset / "task_manifest.json", manifest)
        dataset.rename(output)
    return {"status": "pilot_data_prepared", "candidate_rows": len(rows),
            "labels": manifest["candidate_labels"], "class_counts": manifest["class_counts"],
            "output": str(output), "scope": manifest["scope"]}


def main():
    p = argparse.ArgumentParser()
    for name in ("review-csv", "review-manifest", "seed-exclusions", "expansion-exclusions",
                 "seed-pretrain-manifest", "expansion-pretrain-manifest", "domain-manifest", "output"):
        p.add_argument("--" + name, required=True)
    result = build(**vars(p.parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
