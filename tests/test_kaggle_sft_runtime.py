"""Private handoff integrity, trusted checkpoint ancestry, and recovery."""
import json
from pathlib import Path
import zipfile

import pytest

from finllm.checkpoint import save_checkpoint, sha256_file
from finllm.kaggle_sft_runtime import (export_recovery, install_recovery_archive, install_recovery_directory,
                                      install_transfer_archive, install_transfer_directory)
from finllm.runtime import sync_snapshot


def _model_chain(root, scratch):
    seed = scratch / "seed"
    seed.mkdir()
    save_checkpoint(seed / "latest.pt", {"provenance": {"stage": "pretrain"}})
    sync_snapshot(seed, root / "expansions/expansion_v1/parent")
    expansion = scratch / "expansion"
    expansion.mkdir()
    save_checkpoint(expansion / "latest.pt", {"provenance": {"parent_lineage": {
        "parent_checkpoint_sha256": sha256_file(seed / "latest.pt")}}})
    sync_snapshot(expansion, root / "domains/domain_v1/parent")
    domain = scratch / "domain"
    domain.mkdir()
    save_checkpoint(domain / "latest.pt", {"config": {"domain_transition": {
        "parent_checkpoint_sha256": sha256_file(expansion / "latest.pt")}}})
    sync_snapshot(domain, root / "domains/domain_v1/run")
    return sha256_file(domain / "latest.pt")


def _bundle(root, zip_path, extra=None):
    paths = [p for p in root.rglob("*") if p.is_file()]
    files = {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in paths}
    if extra:
        files[extra] = {"bytes": 1, "sha256": "0" * 64}
    manifest = {"format": "130u-kaggle-transfer-v1", "files": files, "snapshots": []}
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
        if extra:
            archive.writestr(extra, "x")
        archive.writestr("transfer_manifest.json", json.dumps(manifest))
    return sha256_file(zip_path)


@pytest.fixture
def transfer(tmp_path):
    root, scratch = tmp_path / "source", tmp_path / "scratch"
    root.mkdir()
    scratch.mkdir()
    domain = _model_chain(root, scratch)
    zip_path = tmp_path / "transfer.zip"
    digest = _bundle(root, zip_path)
    return root, zip_path, digest, domain


def test_verified_transfer_roundtrip_and_safe_idempotence(transfer, tmp_path):
    _, archive, digest, domain = transfer
    destination = tmp_path / "handoff"
    result = install_transfer_archive(archive, destination, expected_zip_sha256=digest, expected_domain_sha256=domain)
    assert result["reused"] is False and result["trusted_chain"]["domain_checkpoint_sha256"] == domain
    assert install_transfer_archive(archive, destination, expected_zip_sha256=digest, expected_domain_sha256=domain)["reused"] is True
    (destination / "domains/domain_v1/run/latest.pt").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="changed"):
        install_transfer_archive(archive, destination, expected_zip_sha256=digest, expected_domain_sha256=domain)


def test_wrong_archive_or_checkpoint_identity_fails_before_install(transfer, tmp_path):
    _, archive, digest, _ = transfer
    destination = tmp_path / "handoff"
    with pytest.raises(ValueError, match="ZIP SHA256"):
        install_transfer_archive(archive, destination, expected_zip_sha256="0" * 64)
    with pytest.raises(ValueError, match="known 74-step"):
        install_transfer_archive(archive, destination, expected_zip_sha256=digest, expected_domain_sha256="0" * 64)
    assert not destination.exists()


def test_archive_escape_and_incomplete_destination_rejected(transfer, tmp_path):
    root, _, _, domain = transfer
    malicious = tmp_path / "bad.zip"
    digest = _bundle(root, malicious, extra="../outside.txt")
    with pytest.raises(ValueError, match="path escapes"):
        install_transfer_archive(malicious, tmp_path / "target", expected_zip_sha256=digest, expected_domain_sha256=domain)
    assert not (tmp_path / "outside.txt").exists()
    _, archive, digest, _ = transfer
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        install_transfer_archive(archive, destination, expected_zip_sha256=digest, expected_domain_sha256=domain)
    assert (destination / "keep.txt").read_text() == "preserve"


def test_recovery_bundle_roundtrip_and_no_overwrite(transfer, tmp_path):
    _, archive, digest, domain = transfer
    first, second = tmp_path / "first", tmp_path / "second"
    for root in (first, second):
        install_transfer_archive(archive, root, expected_zip_sha256=digest, expected_domain_sha256=domain)
    live = first / "live_checkpoints/sft_kaggle_v1"
    live.mkdir(parents=True)
    save_checkpoint(live / "latest.pt", {"config": {"sft_pilot_binding": {"run_version": "sft_kaggle_v1"}},
                                         "progress": {"updates": 1}})
    report = first / "sft/sft_kaggle_v1/report_archive/report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")
    recovery = tmp_path / "recovery.zip"
    result = export_recovery(first, recovery)
    assert result["updates"] == 1 and result["download_or_save_version_required"] is True
    restored = install_recovery_archive(recovery, second, expected_zip_sha256=result["sha256"])
    assert restored["progress"]["updates"] == 1
    assert sha256_file(second / "live_checkpoints/sft_kaggle_v1/latest.pt") == sha256_file(live / "latest.pt")
    with pytest.raises(FileExistsError):
        install_recovery_archive(recovery, second, expected_zip_sha256=result["sha256"])
    third = tmp_path / "third"
    install_transfer_archive(archive, third, expected_zip_sha256=digest, expected_domain_sha256=domain)
    expanded = tmp_path / "recovery-extracted"
    with zipfile.ZipFile(recovery) as z:
        z.extractall(expanded)
    recovered = install_recovery_directory(expanded, third, expected_manifest_sha256=result["manifest_sha256"])
    assert recovered["progress"]["updates"] == 1
    assert sha256_file(third / "live_checkpoints/sft_kaggle_v1/latest.pt") == sha256_file(live / "latest.pt")


def test_actual_exporter_archive_is_accepted_by_receiver(tmp_path, monkeypatch):
    import finllm.kaggle_export as exporter
    root, scratch = tmp_path / "source-drive", tmp_path / "models"
    root.mkdir()
    scratch.mkdir()
    expected_domain = _model_chain(root, scratch)
    for i, snapshot in enumerate(exporter.SNAPSHOTS):
        if (root / snapshot).exists():
            continue
        folder = tmp_path / f"part-{i}"
        folder.mkdir()
        (folder / "fixture.txt").write_text(snapshot, encoding="utf-8")
        if snapshot == "tasks/entity_v1":
            (folder / "task_manifest.json").write_text('{}', encoding="utf-8")
        sync_snapshot(folder, root / snapshot)
    monkeypatch.setattr(exporter, "audit_domain_parent", lambda *a, **k: {"updates": 74})
    monkeypatch.setattr(exporter, "audit_task_inputs", lambda *a, **k: {"train": 1})
    monkeypatch.setattr(exporter, "_audit_training_exclusions", lambda *a, **k: {})
    archive = tmp_path / "export.zip"
    result = exporter.export_kaggle_bundle(root, archive, expected_domain_sha256=expected_domain,
                 expected_task_manifest_sha256=sha256_file(root / "tasks/entity_v1/task_manifest.json"))
    installed = install_transfer_archive(archive, tmp_path / "kaggle-root",
                 expected_zip_sha256=result["sha256"], expected_domain_sha256=expected_domain)
    assert installed["trusted_chain"]["domain_checkpoint_sha256"] == expected_domain
    extracted = tmp_path / "kaggle-input"
    with zipfile.ZipFile(archive) as z:
        z.extractall(extracted)
    assert sha256_file(extracted / "transfer_manifest.json") == result["manifest_sha256"]
    result2 = install_transfer_directory(extracted, tmp_path / "kaggle-extracted",
                 expected_manifest_sha256=result["manifest_sha256"], expected_domain_sha256=expected_domain)
    assert result2["trusted_chain"]["domain_checkpoint_sha256"] == expected_domain
    assert install_transfer_directory(extracted, tmp_path / "kaggle-extracted",
                 expected_manifest_sha256=result["manifest_sha256"], expected_domain_sha256=expected_domain)["reused"] is True
    with pytest.raises(ValueError, match="manifest SHA256"):
        install_transfer_directory(extracted, tmp_path / "wrong", expected_manifest_sha256="0" * 64,
                                   expected_domain_sha256=expected_domain)
    (extracted / "tasks/entity_v1/task_manifest.json").write_text("tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed"):
        install_transfer_directory(extracted, tmp_path / "tampered", expected_manifest_sha256=result["manifest_sha256"],
                                   expected_domain_sha256=expected_domain)


def test_real_cpu_pretraining_transfer_and_kaggle_sft_probe(tmp_path, monkeypatch):
    """Tiny real seed/expansion/domain checkpoints traverse both transfer ends."""
    import torch
    import finllm.kaggle_export as exporter
    import test_sft_pilot_runtime as sft_tests
    from finllm.sft_pilot_runtime import prepare_sft_pilot, run_sft_pilot
    from test_expansion_runtime import _runner

    # Reuse the real chained CPU training fixture, but suppress its final SFT
    # step so the export source represents a completed domain before SFT.
    monkeypatch.setattr(sft_tests, "run_sft_pilot", lambda *args: {"end_updates": 1})
    sft_tests.test_prepare_sft_pilot_reads_real_cloud_exclusions_and_checkpoint_chain(tmp_path, monkeypatch, False)
    drive = tmp_path / "drive"
    stage = drive / "sft/sft_entity_v1"
    assert stage.resolve().is_relative_to(tmp_path.resolve())
    stage.rename(tmp_path / "prior-test-stage")
    parent = drive / "domains/domain_v1/run/latest.pt"
    expected_domain = sha256_file(parent)
    expected_task = sha256_file(drive / "tasks/entity_v1/task_manifest.json")
    real_audit = exporter.audit_domain_parent
    def audit_tiny(*args, **kwargs):
        kwargs["expected_updates"] = 3
        return real_audit(*args, **kwargs)
    monkeypatch.setattr(exporter, "audit_domain_parent", audit_tiny)
    transfer = tmp_path / "actual-transfer.zip"
    built = exporter.export_kaggle_bundle(drive, transfer, expected_domain_sha256=expected_domain,
                                          expected_task_manifest_sha256=expected_task)
    root, project = tmp_path / "kaggle-root", tmp_path / "kaggle-project"
    project.mkdir()
    extracted = tmp_path / "kaggle-input-real"
    with zipfile.ZipFile(transfer) as z:
        z.extractall(extracted)
    install_transfer_directory(extracted, root, expected_manifest_sha256=built["manifest_sha256"],
                               expected_domain_sha256=expected_domain)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        state = prepare_sft_pilot(project, root, run_version="sft_kaggle_v1",
                                  expected_checkpoint_sha256=expected_domain,
                                  expected_domain_updates=3, expected_task_manifest_sha256=expected_task)
        assert state["parent_audit"]["actual_pretraining_exclusions"]["seed_train"]["missing"] == 0
        result = run_sft_pilot(state, "probe", _runner)
        assert result["end_updates"] == 1
        recovery = export_recovery(root, tmp_path / "kaggle-recovery.zip")
        assert recovery["updates"] == 1
        # Recreate the same Kaggle paths, as in a new matching session, and
        # recover the prior probe from the auto-extracted private data set.
        root.rename(tmp_path / "old-kaggle-root")
        project.rename(tmp_path / "old-kaggle-project")
        project.mkdir()
        install_transfer_directory(extracted, root, expected_manifest_sha256=built["manifest_sha256"],
                                   expected_domain_sha256=expected_domain)
        recovery_source = tmp_path / "kaggle-recovery-input"
        with zipfile.ZipFile(tmp_path / "kaggle-recovery.zip") as z:
            z.extractall(recovery_source)
        install_recovery_directory(recovery_source, root, expected_manifest_sha256=recovery["manifest_sha256"])
        resumed = prepare_sft_pilot(project, root, run_version="sft_kaggle_v1",
                                    expected_checkpoint_sha256=expected_domain,
                                    expected_domain_updates=3, expected_task_manifest_sha256=expected_task)
        assert resumed["selected"] is not None
        finished = run_sft_pilot(resumed, "continue", _runner)
        assert finished["start_updates"] == 1 and finished["end_updates"] == 2
    finally:
        torch.set_num_threads(old)
