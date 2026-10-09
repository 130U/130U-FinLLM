import json

import pytest

from finllm.data import file_sha256
from finllm.runtime import sync_snapshot


def source_files(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "weights.txt").write_text("weights-v1", encoding="utf-8")
    (source / "config.json").write_text('{"seed": 42}', encoding="utf-8")
    (source / ".secret").write_text("not a project artifact", encoding="utf-8")
    return source


def test_snapshot_hash_manifest_and_restore_roundtrip(tmp_path):
    source = source_files(tmp_path)
    backup, restored = tmp_path / "backup", tmp_path / "restored"
    result = sync_snapshot(source, backup)
    assert result["verified"] is True and result["files"] == 2
    manifest = json.loads((backup / "SNAPSHOT_COMPLETE.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "finllm-snapshot-v1"
    for entry in manifest["files"]:
        assert file_sha256(backup / entry["path"]) == entry["sha256"]
        assert (backup / entry["path"]).stat().st_size == entry["bytes"]
    assert not (backup / ".secret").exists()
    (backup / "unlisted.txt").write_text("not part of this snapshot", encoding="utf-8")
    sync_snapshot(backup, restored)
    assert not (restored / "unlisted.txt").exists()
    for entry in manifest["files"]:
        assert (restored / entry["path"]).read_bytes() == (source / entry["path"]).read_bytes()


def test_working_copy_can_change_and_be_backed_up_as_a_new_version(tmp_path):
    original = source_files(tmp_path)
    first_backup, working, next_backup = (
        tmp_path / "first-backup", tmp_path / "working", tmp_path / "next-backup")
    sync_snapshot(original, first_backup)
    result = sync_snapshot(first_backup, working, working_copy=True)
    assert result["verified"] is True and result["working_copy"] is True
    assert not (working / "SNAPSHOT_COMPLETE.json").exists()
    assert not (working / "SNAPSHOT_INCOMPLETE.json").exists()
    (working / "nested" / "weights.txt").write_text("weights-v2", encoding="utf-8")
    (working / "new-log.txt").write_text("training continued", encoding="utf-8")
    next_result = sync_snapshot(working, next_backup)
    assert next_result["files"] == 3 and next_result["working_copy"] is False
    manifest = json.loads((next_backup / "SNAPSHOT_COMPLETE.json").read_text(encoding="utf-8"))
    assert {entry["path"] for entry in manifest["files"]} == {
        "config.json", "nested/weights.txt", "new-log.txt"}
    assert (next_backup / "nested" / "weights.txt").read_text(encoding="utf-8") == "weights-v2"
    assert (first_backup / "nested" / "weights.txt").read_text(encoding="utf-8") == "weights-v1"
    # A completed snapshot still behaves as immutable by default.
    (first_backup / "config.json").write_text("modified backup", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or changed"):
        sync_snapshot(first_backup, tmp_path / "rejected", working_copy=True)


def test_restore_rejects_corruption_before_writing_destination(tmp_path):
    backup = tmp_path / "backup"
    sync_snapshot(source_files(tmp_path), backup)
    (backup / "config.json").write_text("corrupted!", encoding="utf-8")
    restored = tmp_path / "restored"
    with pytest.raises(ValueError, match="incomplete or changed"):
        sync_snapshot(backup, restored)
    assert not restored.exists()


def test_snapshot_nested_paths_rejected(tmp_path):
    source = source_files(tmp_path)
    for target in (source, source / "backup", tmp_path):
        with pytest.raises(ValueError, match="non-nested"):
            sync_snapshot(source, target)


def test_failed_copy_invalidates_previous_complete_marker(tmp_path, monkeypatch):
    import finllm.runtime as runtime

    source, backup = source_files(tmp_path), tmp_path / "backup"
    sync_snapshot(source, backup)
    original_copy = runtime.shutil.copy2

    def corrupted_copy(source_path, target_path):
        original_copy(source_path, target_path)
        target_path.write_bytes(b"interrupted transfer")

    monkeypatch.setattr(runtime.shutil, "copy2", corrupted_copy)
    with pytest.raises(OSError, match="hash mismatch"):
        sync_snapshot(source, backup)
    assert not (backup / "SNAPSHOT_COMPLETE.json").exists()
    assert (backup / "SNAPSHOT_INCOMPLETE.json").exists()
    assert not list(backup.rglob("*.tmp"))
    with pytest.raises(ValueError, match="incomplete"):
        sync_snapshot(backup, tmp_path / "would-restore-incomplete")


def test_restore_rechecks_the_original_manifest_hash_during_copy(tmp_path, monkeypatch):
    import finllm.runtime as runtime

    backup = tmp_path / "backup"
    sync_snapshot(source_files(tmp_path), backup)
    original_copy = runtime.shutil.copy2

    def source_changes_after_validation(source_path, target_path):
        source_path.write_text("changed after snapshot validation", encoding="utf-8")
        return original_copy(source_path, target_path)

    monkeypatch.setattr(runtime.shutil, "copy2", source_changes_after_validation)
    restored = tmp_path / "restored"
    with pytest.raises(OSError, match="hash mismatch"):
        sync_snapshot(backup, restored)
    assert not (restored / "SNAPSHOT_COMPLETE.json").exists()


def test_restore_does_not_replace_the_frozen_hash_with_a_later_source_hash(tmp_path, monkeypatch):
    import finllm.runtime as runtime

    backup = tmp_path / "backup"
    sync_snapshot(source_files(tmp_path), backup)
    original_hash = runtime.file_sha256
    changed = False

    def change_after_first_verified_hash(path):
        nonlocal changed
        digest = original_hash(path)
        if path.parent == backup and not changed:
            # The verification read succeeded, but this source is changed
            # before the copy loop. Restore must retain the manifest's hash.
            path.write_text("a newer but uncertified snapshot version", encoding="utf-8")
            changed = True
        return digest

    monkeypatch.setattr(runtime, "file_sha256", change_after_first_verified_hash)
    with pytest.raises(OSError, match="hash mismatch"):
        sync_snapshot(backup, tmp_path / "restored")


@pytest.mark.parametrize("field,value", [("format", "unexpected-format"), ("bytes", 9999)])
def test_restore_rejects_invalid_manifest_metadata(tmp_path, field, value):
    backup = tmp_path / "backup"
    sync_snapshot(source_files(tmp_path), backup)
    marker = backup / "SNAPSHOT_COMPLETE.json"
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    if field == "format":
        manifest[field] = value
    else:
        manifest["files"][0][field] = value
    marker.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        sync_snapshot(backup, tmp_path / "restore")
