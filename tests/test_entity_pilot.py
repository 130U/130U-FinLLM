import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_entity_pilot import build


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "review.csv"
    fields = ["id", "human_stance", "human_reviewed", "source_entity", "text", "headline_only_stance", "source_headline_id", "event_group_id"]
    rows = []
    for i in range(2000):
        label = ("bullish", "bearish", "neutral")[i % 3] if i < 120 else "exclude"
        rows.append({"id": f"row:{i}", "human_stance": label, "human_reviewed": "true", "source_entity": "ACME", "text": f"ACME reports result unique{i} division {i * 313}", "headline_only_stance": label, "source_headline_id": f"headline:{i}", "event_group_id": "same-event" if i in (0, 1) else ""})
    with source.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"format": "130u-human-confirmed-headline-review-v1", "provenance": {"type": "explicit_user_conversation_confirmation"}, "output_csv": {"sha256": digest(source)}}), encoding="utf-8")
    exclusion = tmp_path / "exclusion.jsonl"
    exclusion.write_text("".join(json.dumps({"text": row["text"]}) + "\n" for row in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"eval_exclusion_sha256": digest(exclusion)}), encoding="utf-8")
    return dict(review_csv=source, review_manifest=review, seed_exclusions=exclusion,
                expansion_exclusions=exclusion, seed_pretrain_manifest=manifest,
                expansion_pretrain_manifest=manifest, domain_manifest=manifest,
                output=tmp_path / "output")


def test_build_uses_confirmed_labels_and_groups_events(inputs):
    result = build(**inputs)
    assert result["candidate_rows"] == 120
    out = inputs["output"]
    rows = [json.loads(s) for name in ("train", "development", "calibration", "test")
            for s in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    events = [r for r in rows if r["event_id"] == "same-event"]
    assert len(events) == 2 and len({r["split"] for r in events}) == 1
    assert all(r["human_reviewed"] and not r["asset_verified"] for r in rows)
    assert all(r["target_scope"] == "literal_entity_not_verified_security" for r in rows)
    m = json.loads((out / "task_manifest.json").read_text(encoding="utf-8"))
    assert not m["semantic_event_audit_complete"]
    assert all(digest(out / k) == v["sha256"] for k, v in m["files"].items())
    with pytest.raises(FileExistsError):
        build(**inputs)


@pytest.mark.parametrize("tamper", ["confirmation", "exclusions", "manifest"])
def test_provenance_and_frozen_exclusions_required(inputs, tamper):
    if tamper == "confirmation":
        inputs["review_csv"].write_text("tampered", encoding="utf-8")
    elif tamper == "exclusions":
        p = inputs["seed_exclusions"]
        lines = p.read_text(encoding="utf-8").splitlines()
        p.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
        # Rebind manifests to prove membership, not just file hash, is checked.
        inputs["seed_pretrain_manifest"].write_text(json.dumps({"eval_exclusion_sha256": digest(p)}), encoding="utf-8")
    else:
        inputs["seed_pretrain_manifest"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        build(**inputs)
    assert not inputs["output"].exists()
