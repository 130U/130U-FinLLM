"""Candidate isolation and honest review-state checks, using tiny source fixtures."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_sentfin_review.py"
_spec = importlib.util.spec_from_file_location("sentfin_review", _path)
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    rows = []
    for label in review.LABELS:
        for index in range(3):
            rows.append([str(len(rows) + 1), f"ACME and BETA headline {label} {index}",
                         repr({"ACME": label, "BETA": label}), "6"])
    rows.append(["10", rows[0][1], rows[0][2], "6"])
    rows.append(["11", "Osian's Fund headline is also frozen", "{'Osian's Fund': 'neutral'}", "7"])
    with (root / "SEntFiN.csv").open("w", encoding="utf8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["S No.", "Title", "Decisions", "Words"])
        writer.writerows(rows)
    for name in ("LICENSE", "README.md", "entity_list_comprehensive.csv"):
        (root / name).write_text("fixture\n", encoding="utf8")
    monkeypatch.setattr(review, "FILES", {p.name: (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
                                         for p in root.iterdir()})
    return root


def test_all_candidates_frozen_and_no_human_labels_invented(source, tmp_path):
    output = tmp_path / "review"
    result = review.prepare(source, output, count=6)
    assert result["source_headline_rows"] == 11
    assert result["unique_exact_headline_texts"] == 10
    assert result["quarantine_count"] == 1
    frozen = [json.loads(line) for line in (output / "task_exclusions.jsonl").read_text(encoding="utf8").splitlines()]
    assert any("Osian's" in row["text"] for row in frozen)
    assert any(len(row["source_ids"]) == 2 for row in frozen)
    assert len({row["text"] for row in frozen}) == 10
    assert (output / "review_200.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    with (output / "review_200.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({r["text"] for r in rows}) == 6
    assert result["review_source_sentiment_counts"] == {label: 2 for label in review.LABELS}
    for row in rows:
        assert row["human_reviewed"] == "false"
        assert row["human_stance"] == row["asset_identifier"] == row["proposed_stance"] == ""
    for name, record in result["artifacts"].items():
        assert review.digest(output / name) == record["sha256"]


def test_exports_deterministic_and_existing_reviews_preserved(source, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    review.prepare(source, first, count=6)
    review.prepare(source, second, count=6)
    for file in first.iterdir():
        assert file.read_bytes() == (second / file.name).read_bytes()
    with pytest.raises(ValueError, match="never overwritten"):
        review.prepare(source, first, count=6)


def test_source_tampering_rejected_before_export(source, tmp_path):
    with (source / "SEntFiN.csv").open("a", encoding="utf8") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="integrity"):
        review.prepare(source, tmp_path / "out", count=6)
    assert not (tmp_path / "out").exists()
