"""Import explicit offline review decisions without creating training data.

The JSON progress export is bound to the exact AI source bytes and ordered IDs.
Human stance approval and human asset verification remain separate facts. Even a
dual-confirmed candidate still requires asset registry, event grouping, rights,
and frozen split checks before it may become a formal training row.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


STANCES = frozenset({"bearish", "neutral", "bullish", "exclude"})
DECISION_FIELDS = frozenset({
    "id", "human_stance", "asset_identifier", "asset_verified", "notes",
    "event_group_id", "human_reviewed", "review_status", "reviewed_at",
})
ENVELOPE_FIELDS = frozenset({
    "format", "dataset_sha256", "row_ids", "exported_at",
    "formal_training_ready", "decisions",
})
PENDING_GATES = [
    "asset_registry_resolution", "event_semantic_grouping",
    "source_rights_review", "frozen_task_split",
]


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _jsonl_bytes(rows):
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True,
                               allow_nan=False) + "\n").encode("utf-8")
                    for row in rows)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object field")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON value is not allowed")


def _decode(raw):
    try:
        return json.loads(raw, object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid UTF-8 JSON input") from exc


def _text(record, key, *, nonempty=False):
    value = record.get(key)
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"{key} must be {'nonempty ' if nonempty else ''}text")
    return value


def _boolean(record, key):
    value = record.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be an exact JSON boolean")
    return value


def _load_source(raw):
    try:
        lines = raw.decode("utf-8-sig").splitlines()
    except UnicodeError as exc:
        raise ValueError("Source must be UTF-8 JSONL") from exc
    rows, ids = [], set()
    for line in lines:
        if not line.strip():
            continue
        row = _decode(line)
        if not isinstance(row, dict):
            raise ValueError("Source rows must be JSON objects")
        for key in ("id", "text", "source_entity"):
            _text(row, key, nonempty=True)
        if row["id"] in ids:
            raise ValueError("Duplicate source row ID")
        for key in ("human_reviewed", "human_asset_verified", "asset_verified",
                    "formal_training_ready"):
            if key in row and (_boolean(row, key) is not False):
                raise ValueError("AI source must not assert human approval or training readiness")
        if "human_stance" in row and _text(row, "human_stance") != "":
            raise ValueError("AI source must not contain an approved human stance")
        suggestion = row.get("ai_stance", row.get("proposed_stance"))
        if not isinstance(suggestion, str) or suggestion not in STANCES:
            raise ValueError("Source has no valid AI stance suggestion")
        rows.append(row)
        ids.add(row["id"])
    if not rows:
        raise ValueError("AI source is empty")
    return rows


def _clean_decision(record):
    if not isinstance(record, dict) or set(record) != DECISION_FIELDS:
        raise ValueError("Decision must contain exactly the UI decision fields")
    for key in ("id", "human_stance", "asset_identifier", "notes",
                "event_group_id", "review_status", "reviewed_at"):
        _text(record, key, nonempty=(key == "id"))
    reviewed = _boolean(record, "human_reviewed")
    verified = _boolean(record, "asset_verified")
    if record["human_stance"] not in STANCES | {""}:
        raise ValueError("Invalid human stance")
    if reviewed and record["human_stance"] not in STANCES:
        raise ValueError("Reviewed decision needs an explicit valid human stance")
    if verified and not record["asset_identifier"].strip():
        raise ValueError("Verified asset needs a nonempty asset identifier")
    status = record["review_status"]
    if status not in {"draft", "uncertain", "reviewed"} or reviewed != (status == "reviewed"):
        raise ValueError("Review status and human approval flag are inconsistent")
    return dict(record)


def _load_progress(raw, digest, ids):
    progress = _decode(raw)
    if not isinstance(progress, dict) or set(progress) != ENVELOPE_FIELDS:
        raise ValueError("Expected the exact UI progress JSON envelope")
    if progress["format"] != "130u-review-decisions-v1":
        raise ValueError("Unknown review progress format")
    if not isinstance(progress["dataset_sha256"], str) or progress["dataset_sha256"] != digest:
        raise ValueError("Review progress dataset SHA does not match the exact AI source")
    if not isinstance(progress["row_ids"], list) or progress["row_ids"] != ids:
        raise ValueError("Review progress ordered row IDs do not match the AI source")
    _text(progress, "exported_at", nonempty=True)
    if _boolean(progress, "formal_training_ready") is not False:
        raise ValueError("UI progress cannot assert formal training readiness")
    if not isinstance(progress["decisions"], list):
        raise ValueError("Progress decisions must be an array")
    allowed, decisions = set(ids), {}
    for item in progress["decisions"]:
        decision = _clean_decision(item)
        row_id = decision["id"]
        if row_id not in allowed or row_id in decisions:
            raise ValueError("Progress contains an unknown or duplicate decision ID")
        decisions[row_id] = decision
    return progress, decisions


def _blank_decision(row_id):
    return {"id": row_id, "human_stance": "", "asset_identifier": "",
            "asset_verified": False, "notes": "", "event_group_id": "",
            "human_reviewed": False, "review_status": "unreviewed", "reviewed_at": ""}


def _csv_cell(value):
    if type(value) is bool:
        return "true" if value else "false"
    value = str(value)
    # CSV is a convenience worksheet; the JSON products retain exact raw text.
    if value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _asset_csv(candidates):
    fields = ["id", "source_entity", "text", "source_headline_id", "human_stance",
              "asset_identifier", "asset_identifier_draft", "asset_verified",
              "event_group_id", "registry_target", "registry_aliases_json",
              "registry_reviewed", "event_group_reviewed", "source_rights_reviewed",
              "formal_training_ready"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in candidates:
        values = {key: row.get(key, "") for key in fields}
        for key in ("registry_reviewed", "event_group_reviewed", "source_rights_reviewed",
                    "formal_training_ready"):
            values[key] = False
        writer.writerow({key: _csv_cell(value) for key, value in values.items()})
    return stream.getvalue().encode("utf-8-sig")


def import_review_progress(source_path, progress_path, output_dir):
    """Create an immutable private audit package; never emits formal SFT rows.

    Existing complete byte-identical output can be reused. Different, additional,
    symlinked, or incomplete output is preserved and rejected, never repaired or
    overwritten. Use a new version directory when importing changed progress.
    """
    source_raw, progress_raw = Path(source_path).read_bytes(), Path(progress_path).read_bytes()
    digest = _sha(source_raw)
    rows = _load_source(source_raw)
    ids = [row["id"] for row in rows]
    progress, decisions = _load_progress(progress_raw, digest, ids)
    normalized, candidates = [], []
    for source in rows:
        row_id = source["id"]
        decision = decisions.get(row_id, _blank_decision(row_id))
        normalized.append({**decision, "explicit_decision": row_id in decisions,
                           "dataset_sha256": digest, "formal_training_ready": False})
        if not decision["human_reviewed"] or decision["human_stance"] == "exclude":
            continue
        candidate = {key: source[key] for key in (
            "id", "text", "source_entity", "source_headline_id", "text_group_id",
            "group_id", "source_url") if key in source}
        candidate.update({
            "human_stance": decision["human_stance"], "human_reviewed": True,
            "asset_identifier": decision["asset_identifier"] if decision["asset_verified"] else "",
            "asset_identifier_draft": "" if decision["asset_verified"] else decision["asset_identifier"],
            "asset_verified": decision["asset_verified"], "human_notes": decision["notes"],
            "event_group_id": decision["event_group_id"], "reviewed_at": decision["reviewed_at"],
            "dataset_sha256": digest, "source_row_sha256": _sha(_json_bytes(source)),
            "formal_training_ready": False, "pending_gates": list(PENDING_GATES),
        })
        candidates.append(candidate)
    statuses = Counter(item["review_status"] for item in normalized)
    reviewed = [item for item in normalized if item["human_reviewed"]]
    verified_candidates = sum(item["asset_verified"] for item in candidates)
    counts = {
        "source_rows": len(rows), "explicit_decisions": len(decisions),
        "unreviewed_rows": statuses["unreviewed"], "draft_rows": statuses["draft"],
        "uncertain_rows": statuses["uncertain"], "human_reviewed_rows": len(reviewed),
        "excluded_rows": sum(item["human_stance"] == "exclude" for item in reviewed),
        "reviewed_label_candidates": len(candidates),
        "asset_verified_rows": sum(item["asset_verified"] for item in normalized),
        "reviewed_candidates_asset_verified": verified_candidates,
        "reviewed_candidates_asset_pending": len(candidates) - verified_candidates,
        "pending_registry_resolution_rows": len(candidates), "formal_training_ready_rows": 0,
    }
    products = {
        "normalized_decisions.jsonl": _jsonl_bytes(normalized),
        "reviewed_label_candidates.jsonl": _jsonl_bytes(candidates),
        "pending_asset_resolution.csv": _asset_csv(candidates),
    }
    manifest = {
        "format": "130u-review-import-v1", "progress_format": progress["format"],
        "dataset_sha256": digest, "progress_sha256": _sha(progress_raw),
        "ordered_row_ids_sha256": _sha(_json_bytes(ids)),
        "progress_exported_at": progress["exported_at"], "counts": counts,
        "reviewed_label_counts": {label: sum(item["human_stance"] == label for item in reviewed)
                                  for label in sorted(STANCES)},
        "formal_training_ready": False, "pending_gates": list(PENDING_GATES),
        "privacy": "Contains source news and human decisions; keep outside public Git.",
        "products": {name: {"sha256": _sha(raw), "bytes": len(raw)}
                     for name, raw in products.items()},
    }
    products["manifest.json"] = _json_bytes(manifest)
    output = Path(output_dir)
    reused = False
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_dir():
            raise FileExistsError("Existing output is not a regular package directory; preserved")
        if set(path.name for path in output.iterdir()) != set(products):
            raise FileExistsError("Existing output is incomplete or has additional files; preserved; choose a new version")
        for name, raw in products.items():
            target = output / name
            if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
                raise FileExistsError("Existing output differs; preserved; choose a new version")
        reused = True
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name, raw in products.items():
            with (output / name).open("xb") as handle:
                handle.write(raw)
    # Safe for CLI stdout: counts/hashes only, never raw source or decisions.
    return {"format": manifest["format"], "dataset_sha256": digest,
            "progress_sha256": manifest["progress_sha256"], "counts": counts,
            "reviewed_label_counts": manifest["reviewed_label_counts"],
            "formal_training_ready": False, "reused": reused}
