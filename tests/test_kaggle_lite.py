"""Exercise a verified, weight-only transfer using genuine tiny checkpoints."""
import json
import zipfile
from pathlib import Path

import pytest

import finllm.kaggle_lite as lite
from finllm.checkpoint import sha256_file, load_checkpoint
from finllm.evaluation import run_evaluation
from finllm.training import run_training
from test_sft_pilot_runtime import pilot


@pytest.fixture
def mini_project(tmp_path, monkeypatch):
    fixture = pilot.__wrapped__(tmp_path, monkeypatch)
    project, drive, domain_hash, task_source = next(fixture)
    tokenizer_hash = sha256_file(drive / "seeds/seed_v1/tokenizer/tokenizer.json")
    task_hash = sha256_file(drive / "tasks/entity_v1/task_manifest.json")
    monkeypatch.setattr(lite, "DOMAIN_SHA", domain_hash)
    monkeypatch.setattr(lite, "TOKENIZER_SHA", tokenizer_hash)
    monkeypatch.setattr(lite, "TASK_MANIFEST_SHA", task_hash)
    yield project, drive, domain_hash, tokenizer_hash, task_hash
    try:
        next(fixture)
    except StopIteration:
        pass


def test_weight_only_export_install_and_real_sft(mini_project, tmp_path):
    project, drive, domain_hash, tokenizer_hash, task_hash = mini_project
    archive = tmp_path / "130u_kaggle_lite.zip"
    report = lite.export_lite_from_drive(drive, archive, expected_domain_sha256=domain_hash,
        expected_tokenizer_sha256=tokenizer_hash, expected_task_sha256=task_hash,
        expected_domain_updates=3, expected_task_rows=50)
    assert report["task_counts"]["train"] == 32
    extracted = tmp_path / "kaggle-private-input"
    with zipfile.ZipFile(archive) as z:
        z.extractall(extracted)
    installed = tmp_path / "kaggle-working/lite-input"
    result = lite.install_lite_directory(extracted, installed, expected_manifest_sha256=report["manifest_sha256"])
    assert not result["reused"]
    parent = load_checkpoint(installed / "weights/domain_weights.pt", allow_previous=False)
    original = load_checkpoint(drive / "domains/domain_v1/run/latest.pt", allow_previous=False)
    assert parent["source_checkpoint_sha256"] == domain_hash
    assert set(parent["model_state"]) == set(original["model_state"])
    import torch
    assert all(torch.equal(parent["model_state"][key], original["model_state"][key]) for key in parent["model_state"])
    run = lite.prepare_lite_sft(project, installed, tmp_path / "kaggle-working/sft",
                                expected_manifest_sha256=result["manifest_sha256"])
    run["config"]["train"]["max_updates"] = 1
    trained = run_training(run["config"])
    assert trained["updates"] == 1
    assert load_checkpoint(Path(trained["latest_checkpoint"]))["config"]["lite_binding"] == run["binding"]
    result_eval = run_evaluation(trained["latest_checkpoint"], installed / "tokenizer",
                                 installed / "task/development.jsonl", tmp_path / "eval.json",
                                 device="cpu", target_mode="literal_entity")
    assert result_eval["valid_count"] == 6 and result_eval["rejected_count"] == 0
    assert lite.install_lite_directory(extracted, installed, expected_manifest_sha256=report["manifest_sha256"])["reused"]
    installed_zip = tmp_path / "kaggle-archive/lite-input"
    assert lite.install_lite_archive(archive, installed_zip, expected_zip_sha256=report["sha256"])["manifest_sha256"] == report["manifest_sha256"]
    (installed / "task/train.jsonl").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        lite.prepare_lite_sft(project, installed, tmp_path / "kaggle-working/sft",
                              expected_manifest_sha256=report["manifest_sha256"])


def test_tampered_or_untrusted_data_is_rejected(mini_project, tmp_path):
    project, drive, domain_hash, tokenizer_hash, task_hash = mini_project
    archive = tmp_path / "lite.zip"
    with pytest.raises(ValueError, match="Domain checkpoint"):
        lite.export_lite_from_drive(drive, archive, expected_domain_sha256="a" * 64,
            expected_tokenizer_sha256=tokenizer_hash, expected_task_sha256=task_hash,
            expected_domain_updates=3, expected_task_rows=50)
    report = lite.export_lite_from_drive(drive, archive, expected_domain_sha256=domain_hash,
        expected_tokenizer_sha256=tokenizer_hash, expected_task_sha256=task_hash,
        expected_domain_updates=3, expected_task_rows=50)
    with pytest.raises(FileExistsError):
        lite.export_lite_from_drive(drive, archive, expected_domain_sha256=domain_hash,
            expected_tokenizer_sha256=tokenizer_hash, expected_task_sha256=task_hash,
            expected_domain_updates=3, expected_task_rows=50)
    extracted = tmp_path / "kaggle-private-input"
    with zipfile.ZipFile(archive) as z:
        z.extractall(extracted)
    with pytest.raises(ValueError, match="manifest"):
        lite.install_lite_directory(extracted, tmp_path / "wrong", expected_manifest_sha256="0" * 64)
    (extracted / "task/train.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or does not match"):
        lite.install_lite_directory(extracted, tmp_path / "tampered", expected_manifest_sha256=report["manifest_sha256"])


def test_existing_sft_progress_is_not_silently_discarded(mini_project, tmp_path):
    _, drive, domain_hash, tokenizer_hash, task_hash = mini_project
    (drive / "live_checkpoints/sft_entity_v1").mkdir(parents=True)
    with pytest.raises(ValueError, match="SFT progress"):
        lite.export_lite_from_drive(drive, tmp_path / "lite.zip", expected_domain_sha256=domain_hash,
            expected_tokenizer_sha256=tokenizer_hash, expected_task_sha256=task_hash,
            expected_domain_updates=3, expected_task_rows=50)
