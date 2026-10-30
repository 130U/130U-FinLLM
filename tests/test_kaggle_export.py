"""Transfer archive is complete, streamed, private, and never overwrites work."""
import json
from pathlib import Path
import zipfile

import pytest

import finllm.kaggle_export as export
from finllm.data import file_sha256
from finllm.runtime import sync_snapshot


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "drive"
    for i, snapshot in enumerate(export.SNAPSHOTS):
        source_dir = tmp_path / f"source-{i}"
        source_dir.mkdir()
        (source_dir / "payload.txt").write_text("content " + snapshot, encoding="utf-8")
        if snapshot == "tasks/entity_v1":
            (source_dir / "task_manifest.json").write_text('{"format":"fixture"}', encoding="utf-8")
        sync_snapshot(source_dir, root / snapshot)
    called = []
    monkeypatch.setattr(export, "audit_domain_parent", lambda *a, **k: called.append("parent") or {"updates": 74})
    monkeypatch.setattr(export, "audit_task_inputs", lambda *a, **k: called.append("task") or {"train": 1})
    monkeypatch.setattr(export, "_audit_training_exclusions", lambda *a, **k: called.append("exclusions") or {"seed_train": {"missing": 0}})
    task_sha = file_sha256(root / "tasks/entity_v1/task_manifest.json")
    return root, task_sha, called


def test_verified_snapshot_roundtrip_and_non_overwrite(source, tmp_path):
    root, task_sha, called = source
    output = tmp_path / "private.zip"
    result = export.export_kaggle_bundle(root, output, expected_task_manifest_sha256=task_sha)
    assert called == ["parent", "task", "exclusions"]
    assert result["sha256"] == file_sha256(output)
    with zipfile.ZipFile(output) as archive:
        import hashlib
        assert hashlib.sha256(archive.read("transfer_manifest.json")).hexdigest() == result["manifest_sha256"]
        manifest = json.loads(archive.read("transfer_manifest.json"))
        assert manifest["format"] == "130u-kaggle-transfer-v1"
        assert manifest["snapshots"] == list(export.SNAPSHOTS)
        assert set(archive.namelist()) == set(manifest["files"]) | {"transfer_manifest.json"}
        for name, entry in manifest["files"].items():
            raw = archive.read(name)
            assert len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        export.export_kaggle_bundle(root, output, expected_task_manifest_sha256=task_sha)
    assert output.read_bytes() == before


def test_changed_snapshot_and_task_are_rejected(source, tmp_path):
    root, task_sha, _ = source
    (root / "domains/domain_v1/run/payload.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError):
        export.export_kaggle_bundle(root, tmp_path / "bad.zip", expected_task_manifest_sha256=task_sha)
    assert not (tmp_path / "bad.zip").exists()


def test_unknown_sft_progress_blocks_export(source, tmp_path):
    root, task_sha, _ = source
    (root / "live_checkpoints/sft_entity_v1").mkdir(parents=True)
    with pytest.raises(ValueError, match="Existing SFT"):
        export.export_kaggle_bundle(root, tmp_path / "progress.zip", expected_task_manifest_sha256=task_sha)


def test_refuses_drive_output_or_partial_zip(source, tmp_path):
    root, task_sha, _ = source
    with pytest.raises(ValueError, match="outside"):
        export.export_kaggle_bundle(root, root / "output.zip", expected_task_manifest_sha256=task_sha)
    (tmp_path / "resume.zip.partial").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export.export_kaggle_bundle(root, tmp_path / "resume.zip", expected_task_manifest_sha256=task_sha)
    assert (tmp_path / "resume.zip.partial").read_text() == "do not overwrite"
