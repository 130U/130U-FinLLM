import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from finllm.review_import import import_review_progress


def decision(row_id="row-1", **changes):
    result = {"id": row_id, "human_stance": "bullish", "asset_identifier": "NYSE:EXAMPLE",
              "asset_verified": True, "notes": "Human checked the target.", "event_group_id": "",
              "human_reviewed": True, "review_status": "reviewed", "reviewed_at": "2026-10-30T12:00:00Z"}
    result.update(changes)
    return result


def inputs(tmp_path, decisions=None):
    rows = [{"id": f"row-{i}", "text": f"Confidential example headline {i}",
             "source_entity": "Example", "ai_stance": "bearish", "source_sentiment": "bearish",
             "source_headline_id": f"headline-{i}", "human_reviewed": False,
             "human_asset_verified": False, "formal_training_ready": False}
            for i in range(1, 5)]
    source = tmp_path / "ai_annotations.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    progress = {"format": "130u-review-decisions-v1",
                "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "row_ids": [row["id"] for row in rows], "exported_at": "2026-10-30T12:01:00Z",
                "formal_training_ready": False, "decisions": decisions if decisions is not None else [decision()]}
    path = tmp_path / "review_progress.json"
    path.write_text(json.dumps(progress), encoding="utf-8")
    return source, path, progress


def write_progress(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_valid_ui_envelope_partial_progress_and_separate_asset_gate(tmp_path):
    source, progress, _ = inputs(tmp_path)
    output = tmp_path / "review_v1"
    result = import_review_progress(source, progress, output)
    assert result["counts"]["source_rows"] == 4
    assert result["counts"]["human_reviewed_rows"] == 1
    assert result["counts"]["unreviewed_rows"] == 3
    assert result["counts"]["asset_verified_rows"] == 1
    assert result["counts"]["reviewed_candidates_asset_verified"] == 1
    assert result["counts"]["pending_registry_resolution_rows"] == 1
    assert result["counts"]["formal_training_ready_rows"] == 0
    candidate, = read_jsonl(output / "reviewed_label_candidates.jsonl")
    assert candidate["human_stance"] == "bullish"  # Never copied from bearish AI/source label.
    assert candidate["asset_verified"] is True  # Stale source human_asset_verified=False is not an override.
    assert candidate["asset_identifier"] == "NYSE:EXAMPLE"
    assert candidate["formal_training_ready"] is False
    assert "label" not in candidate and "target" not in candidate
    normalized = read_jsonl(output / "normalized_decisions.jsonl")
    assert [row["id"] for row in normalized] == [f"row-{i}" for i in range(1, 5)]
    assert normalized[1]["explicit_decision"] is False
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, metadata in manifest["products"].items():
        raw = (output / name).read_bytes()
        assert metadata == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    assert import_review_progress(source, progress, output)["reused"] is True


def test_drafts_uncertain_excludes_and_unverified_identifier_are_preserved(tmp_path):
    decisions = [decision(human_stance="exclude", asset_identifier="", asset_verified=False),
                 decision("row-2", human_reviewed=False, review_status="draft", human_stance="", reviewed_at=""),
                 decision("row-3", human_reviewed=False, review_status="uncertain", reviewed_at=""),
                 decision("row-4", human_stance="neutral", asset_verified=False, asset_identifier="draft ID")]
    source, progress, _ = inputs(tmp_path, decisions)
    output = tmp_path / "review_v1"
    result = import_review_progress(source, progress, output)
    assert result["counts"]["excluded_rows"] == 1
    assert result["counts"]["draft_rows"] == 1
    assert result["counts"]["uncertain_rows"] == 1
    assert result["counts"]["reviewed_label_candidates"] == 1
    assert result["counts"]["reviewed_candidates_asset_pending"] == 1
    candidate, = read_jsonl(output / "reviewed_label_candidates.jsonl")
    assert candidate["id"] == "row-4"
    assert candidate["asset_identifier"] == ""
    assert candidate["asset_identifier_draft"] == "draft ID"
    assert read_jsonl(output / "normalized_decisions.jsonl")[2]["review_status"] == "uncertain"


def test_empty_progress_does_not_create_approvals(tmp_path):
    source, progress, _ = inputs(tmp_path, [])
    result = import_review_progress(source, progress, tmp_path / "review_v1")
    assert result["counts"]["explicit_decisions"] == 0
    assert result["counts"]["human_reviewed_rows"] == 0
    assert result["counts"]["reviewed_label_candidates"] == 0


@pytest.mark.parametrize("field,value", [
    ("human_reviewed", "true"), ("human_reviewed", 1), ("asset_verified", "false"),
    ("asset_verified", 0), ("human_stance", ["bullish"]), ("human_stance", "positive"),
    ("human_stance", ""), ("asset_identifier", None), ("asset_identifier", "  "),
    ("notes", 7), ("event_group_id", None), ("reviewed_at", None),
    ("review_status", "draft"), ("review_status", "unreviewed"), ("text", "altered"),
])
def test_forged_decision_fields_rejected_without_output(tmp_path, field, value):
    source, progress, envelope = inputs(tmp_path)
    envelope["decisions"][0][field] = value
    write_progress(progress, envelope)
    output = tmp_path / "review_v1"
    with pytest.raises(ValueError):
        import_review_progress(source, progress, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["sha", "id_order", "id_missing", "unknown", "repeated", "boolean", "readiness", "text", "missing"])
def test_mismatched_or_forged_envelopes_rejected(tmp_path, change):
    source, progress, envelope = inputs(tmp_path)
    if change == "sha":
        envelope["dataset_sha256"] = "0" * 64
    elif change == "id_order":
        envelope["row_ids"].reverse()
    elif change == "id_missing":
        envelope["row_ids"].pop()
    elif change == "unknown":
        envelope["decisions"][0]["id"] = "unknown"
    elif change == "repeated":
        envelope["decisions"].append(copy.deepcopy(envelope["decisions"][0]))
    elif change == "boolean":
        envelope["formal_training_ready"] = 0
    elif change == "readiness":
        envelope["formal_training_ready"] = True
    elif change == "text":
        envelope["exported_at"] = []
    elif change == "missing":
        del envelope["decisions"][0]["notes"]
    write_progress(progress, envelope)
    with pytest.raises(ValueError):
        import_review_progress(source, progress, tmp_path / "review_v1")


@pytest.mark.parametrize("change", ["changed_text", "duplicate_id", "source_approval", "json_duplicate", "nan"])
def test_changed_duplicate_or_invalid_source_rejected(tmp_path, change):
    source, progress, envelope = inputs(tmp_path)
    rows = read_jsonl(source)
    if change == "changed_text":
        rows[0]["text"] += " changed"
    elif change == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif change == "source_approval":
        rows[0]["human_reviewed"] = True
    elif change == "json_duplicate":
        progress.write_text(progress.read_text(encoding="utf-8").replace('"formal_training_ready": false', '"formal_training_ready": false, "formal_training_ready": false'), encoding="utf-8")
    elif change == "nan":
        envelope["decisions"][0]["notes"] = float("nan")
        write_progress(progress, envelope)
    if change in {"changed_text", "duplicate_id", "source_approval"}:
        source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        if change != "changed_text":
            envelope["dataset_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            envelope["row_ids"] = [row["id"] for row in rows]
            write_progress(progress, envelope)
    with pytest.raises(ValueError):
        import_review_progress(source, progress, tmp_path / "review_v1")


@pytest.mark.parametrize("change", ["different", "incomplete", "extra", "changed_progress"])
def test_existing_output_is_never_overwritten_or_repaired(tmp_path, change):
    source, progress, envelope = inputs(tmp_path)
    output = tmp_path / "review_v1"
    import_review_progress(source, progress, output)
    if change == "different":
        (output / "normalized_decisions.jsonl").write_text("preserve me", encoding="utf-8")
    elif change == "incomplete":
        (output / "pending_asset_resolution.csv").unlink()
    elif change == "extra":
        (output / "human_note.txt").write_text("preserve me", encoding="utf-8")
    else:
        envelope["decisions"][0]["notes"] = "new human note"
        write_progress(progress, envelope)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        import_review_progress(source, progress, output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_csv_formula_escaping_and_cli_only_prints_aggregates(tmp_path):
    source, progress, envelope = inputs(tmp_path, [decision(asset_identifier="=danger")])
    output = tmp_path / "review_v1"
    script = Path(__file__).resolve().parents[1] / "scripts" / "import_review_progress.py"
    run = subprocess.run([sys.executable, str(script), "--source", str(source),
                          "--progress", str(progress), "--output", str(output)],
                         capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["counts"]["human_reviewed_rows"] == 1
    assert "Confidential" not in run.stdout + run.stderr
    assert "danger" not in run.stdout + run.stderr
    assert "'=danger" in (output / "pending_asset_resolution.csv").read_text(encoding="utf-8-sig")
    envelope["decisions"][0]["human_stance"] = "Confidential secret malformed label"
    write_progress(progress, envelope)
    rejected = subprocess.run([sys.executable, str(script), "--source", str(source),
                               "--progress", str(progress), "--output", str(output)],
                              capture_output=True, text=True, encoding="utf-8")
    assert rejected.returncode == 2
    assert "Confidential" not in rejected.stdout + rejected.stderr
