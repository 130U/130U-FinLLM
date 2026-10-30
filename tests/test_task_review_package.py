"""Review provenance, ambiguous asset aliases and preservation of manual edits."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


_path = Path(__file__).resolve().parents[1] / "scripts/build_task_review_package.py"
_spec = importlib.util.spec_from_file_location("task_review_package", _path)
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _refresh_ai(root):
    manifest = json.loads((root / "ai_review_manifest.json").read_text())
    manifest["source_csv"]["sha256"] = review.digest(root / "review_200.csv")
    manifest["output_csv"]["sha256"] = review.digest(root / "review_200_ai_proposals.csv")
    _write_json(root / "ai_review_manifest.json", manifest)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    entities = root / "entities.csv"
    _write_csv(entities, [{"Symbol": "OLD_A", "Entity": "Acme,  ACME corp"},
                          {"Symbol": "OLD_B", "Entity": "Beta, Shared Alias"},
                          {"Symbol": "OLD_C", "Entity": "Gamma, Shared Alias"},
                          {"Symbol": "OLD_A", "Entity": "Acme"}])
    entity_hash = review.digest(entities)
    monkeypatch.setattr(review, "ENTITY_SHA256", entity_hash)
    rows = []
    for i, entity in enumerate(("acme", "ACME CORP", "Shared Alias", "Central regulator"), 1):
        rows.append({"id": f"sentfin:{i}:{entity}", "text": "Shared headline" if i < 3 else f"Headline {i}",
                     "source_entity": entity, "source_sentiment": "positive", "source_entity_count": "1",
                     "source_entities": json.dumps([entity]), "entity_is_literal_substring": "false",
                     "proposed_stance": "", "asset_identifier": "", "human_stance": "",
                     "human_reviewed": "false", "notes": ""})
    _write_csv(root / "review_200.csv", rows)
    proposed = [{**row, "proposed_stance": "exclude" if i == 3 else "neutral",
                 "ai_rationale": "AI-only draft", "ai_reviewer": "AI fixture"} for i, row in enumerate(rows)]
    _write_csv(root / "review_200_ai_proposals.csv", proposed)
    _write_json(root / "manifest.json", {"source_repository": "https://github.com/pyRis/SEntFiN",
                                         "source_revision": review.REVISION, "review_rows": len(rows),
                                         "source_files": {"entity_list_comprehensive.csv": {"sha256": entity_hash}}})
    _write_json(root / "ai_review_manifest.json", {"original_manifest_sha256": review.digest(root / "manifest.json"),
                "source_csv": {"sha256": review.digest(root / "review_200.csv")},
                "output_csv": {"sha256": review.digest(root / "review_200_ai_proposals.csv")},
                "not_for_formal_training": True, "human_reviewed_rows": 0, "rows": len(rows)})
    return root, entities


def test_exact_alias_candidates_preserve_unapproved_labels_and_groups(inputs, tmp_path):
    root, entities = inputs
    output = tmp_path / "out"
    manifest = review.prepare(root, entities, output)
    _, rows = review.read_csv(output / "review_200_to_complete.csv", review.EXTRA_FIELDS)
    assert manifest["creation_counts"]["source_symbol_matches"] == {
        "unique_source_symbol": 2, "ambiguous_source_symbols": 1, "unmapped": 1}
    assert json.loads(rows[0]["source_symbol_candidates"]) == ["OLD_A"]
    assert json.loads(rows[2]["source_symbol_candidates"]) == ["OLD_B", "OLD_C"]
    assert json.loads(rows[3]["source_symbol_candidates"]) == []
    assert rows[0]["text_group_id"] == rows[1]["text_group_id"]
    assert rows[0]["source_headline_id"] != rows[1]["source_headline_id"]
    assert rows[0]["source_sentiment"] == "positive" and rows[0]["proposed_stance"] == "neutral"
    assert all(row["human_stance"] == row["asset_identifier"] == row["event_group_id"] == "" for row in rows)
    assert all(row["human_reviewed"] == "false" for row in rows)
    assert manifest["formal_training_ready"] is False
    assert manifest["creation_counts"]["human_reviewed_rows"] == 0
    for name, record in manifest["artifacts"].items():
        assert review.digest(output / name) == record["sha256"]


@pytest.mark.parametrize("change", ["entity_bytes", "proposal_bytes", "changed_source_text", "human_approval", "duplicate_id"])
def test_invalid_or_approved_inputs_fail_before_creating_output(inputs, tmp_path, change):
    root, entities = inputs
    if change == "entity_bytes":
        entities.write_text(entities.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8")
    elif change == "proposal_bytes":
        with (root / "review_200_ai_proposals.csv").open("a", encoding="utf-8") as stream:
            stream.write("unexpected content\n")
    else:
        path = root / "review_200_ai_proposals.csv"
        _, rows = review.read_csv(path, review.SOURCE_FIELDS)
        if change == "changed_source_text":
            rows[0]["text"] = "Different original text"
        elif change == "human_approval":
            rows[0]["human_reviewed"] = "true"
            rows[0]["human_stance"] = "bullish"
        else:
            rows[1]["id"] = rows[0]["id"]
        _write_csv(path, rows)
        _refresh_ai(root)
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        review.prepare(root, entities, output)
    assert not output.exists()


def test_deterministic_readonly_reuse_and_manual_edits_are_never_overwritten(inputs, tmp_path):
    root, entities = inputs
    first, second = tmp_path / "a", tmp_path / "b"
    review.prepare(root, entities, first)
    review.prepare(root, entities, second)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in first.iterdir()}
    assert {p.name: p.read_bytes() for p in second.iterdir()} == {name: record[0] for name, record in before.items()}
    review.prepare(root, entities, first)
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in first.iterdir()}
    worksheet = first / "review_200_to_complete.csv"
    _, rows = review.read_csv(worksheet, review.HUMAN_FIELDS)
    rows[0]["human_stance"] = "neutral"
    rows[0]["human_reviewed"] = "true"
    rows[0]["asset_identifier"] = "manually-verified-fixture"
    _write_csv(worksheet, rows)
    manual = worksheet.read_bytes()
    with pytest.raises(ValueError, match="manual edits"):
        review.prepare(root, entities, first)
    assert worksheet.read_bytes() == manual
