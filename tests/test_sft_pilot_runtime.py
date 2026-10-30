"""Exercise real tiny domain-to-entity-SFT transfer and conservative gates."""
import json
import random
import zipfile
from pathlib import Path

import pytest
import torch

from finllm.checkpoint import sha256_file
from finllm.evaluation import run_evaluation
from finllm.runtime import sync_snapshot
from finllm.sft_pilot_runtime import _budget_complete, install_task_zip, prepare_sft_pilot, run_sft_pilot, backup_sft_pilot
from test_domain_runtime import _complete, _domain, _domain_fixture, _inventory
from test_expansion_runtime import _runner


@pytest.fixture
def pilot(tmp_path):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        project, drive = _domain_fixture(tmp_path)
        _complete(_domain(project, drive))
        source = tmp_path / "task_source"
        source.mkdir()
        randomizer = random.Random(130)
        groups = (("train", 32), ("development", 6), ("calibration_temperature", 4),
                  ("calibration_threshold", 4), ("test", 4))
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        for split, count in groups:
            rows = []
            for index in range(count):
                words = ["".join(randomizer.choices(alphabet, k=8)) for _ in range(8)]
                row = {"id": f"{split}:{index}", "text": "ACME " + " ".join(words), "target": "ACME",
                       "label": ["bullish", "neutral", "bearish"][index % 3],
                       "target_scope": "literal_entity_not_verified_security", "human_reviewed": True,
                       "asset_verified": False, "asset_identity_scope": "not_verified",
                       "training_allowed": True, "source": "fixture", "rights_note": "test-only fixture",
                       "group_id": f"{split}:{index}", "event_id": f"{split}:{index}",
                       "split": "calibration" if split.startswith("calibration_") else split}
                rows.append(row)
            (source / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        def record(path):
            return {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        seed = json.loads((project / "data/processed/seed_v1/general/manifest.json").read_text())
        expansion = json.loads((project / "data/processed/expansion_v1/general/manifest.json").read_text())
        manifest = {"format": "130u-entity-pilot-v1", "seed_exclusions_sha256": seed["eval_exclusion_sha256"],
                    "expansion_exclusions_sha256": expansion["eval_exclusion_sha256"],
                    "files": {p.name: record(p) for p in source.glob("*.jsonl")}}
        (source / "task_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        sync_snapshot(source, drive / "tasks/entity_v1")
        parent = drive / "domains/domain_v1/run/latest.pt"
        yield project, drive, sha256_file(parent), source
    finally:
        torch.set_num_threads(old)


def _prepare(pilot):
    project, drive, digest, source = pilot
    return prepare_sft_pilot(project, drive, expected_checkpoint_sha256=digest, expected_domain_updates=3,
                             expected_task_manifest_sha256=sha256_file(source / "task_manifest.json"))


def test_real_transfer_resume_complete_backup_and_noop(pilot):
    project, drive, digest, _ = pilot
    before = _inventory(drive / "domains/domain_v1")
    state = _prepare(pilot)
    probe = run_sft_pilot(state, "probe", _runner)
    result = run_sft_pilot(state, "continue", _runner)
    assert probe["start_updates"] == 0 and probe["end_updates"] == 1
    assert result["start_updates"] == 1 and result["end_updates"] == 2
    assert result["summary"]["validation"]["macro_f1"] >= 0
    output = Path(state["config"]["output_dir"])
    for name, checkpoint in (("parent", state["config"]["init_checkpoint"]), ("best", output / "best.pt")):
        metrics = run_evaluation(checkpoint, state["config"]["tokenizer"], state["task_local"] / "development.jsonl",
                                 output / (name + "_development.json"), device="cpu", target_mode="literal_entity")
        assert metrics["valid_count"] == 6 and metrics["rejected_count"] == 0
    assert backup_sft_pilot(state)["verified"] is True
    new = _prepare(pilot)
    assert run_sft_pilot(new, "probe", _runner)["resumed_no_training"] is True
    assert run_sft_pilot(new, "continue", _runner)["resumed_no_training"] is True
    assert _inventory(drive / "domains/domain_v1") == before


def test_uploaded_task_zip_is_verified_and_reused(pilot, tmp_path):
    _, _, _, source = pilot
    path = tmp_path / "upload.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for file in source.iterdir():
            archive.write(file, file.name)
    output = tmp_path / "new_drive"
    result = install_task_zip(path, output, sha256_file(path))
    assert result["verified"] is True
    assert install_task_zip(path, output, sha256_file(path))["reused"] is True
    with pytest.raises(ValueError, match="differs"):
        install_task_zip(path, output, "0" * 64)


def test_new_runtime_restores_live_probe_then_completes(pilot, tmp_path):
    _, drive, digest, source = pilot
    first = _prepare(pilot)
    run_sft_pilot(first, "probe", _runner)
    second_project = tmp_path / "fresh-colab"
    second_project.mkdir()
    resumed = prepare_sft_pilot(second_project, drive, expected_checkpoint_sha256=digest,
                                expected_domain_updates=3,
                                expected_task_manifest_sha256=sha256_file(source / "task_manifest.json"))
    assert resumed["selected"] is not None
    result = run_sft_pilot(resumed, "continue", _runner)
    assert result["start_updates"] == 1 and result["end_updates"] == 2
    assert backup_sft_pilot(resumed)["verified"] is True


def test_changed_local_task_is_rejected_before_update(pilot):
    state = _prepare(pilot)
    train = state["task_local"] / "train.jsonl"
    train.write_bytes(train.read_bytes() + b" ")
    with pytest.raises(ValueError, match="frozen experiment"):
        run_sft_pilot(state, "probe", _runner)


def test_amp_skipped_attempts_can_exhaust_epoch_budget_without_reaching_update_target():
    state = {"counts": {"train": 100}, "binding": {"epochs": 2}, "config": {"train": {"max_updates": 7}}}
    assert not _budget_complete({"updates": 6, "tokens": 199}, state)
    assert _budget_complete({"updates": 6, "tokens": 200}, state)
    assert _budget_complete({"updates": 7, "tokens": 190}, state)


@pytest.mark.parametrize("changed", ["review", "asset_scope", "entity", "split"])
def test_invalid_task_package_rejected_before_training(pilot, changed):
    project, drive, digest, source = pilot
    path = source / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if changed == "review": rows[0]["human_reviewed"] = False
    if changed == "asset_scope": rows[0]["asset_verified"] = True
    if changed == "entity": rows[0]["target"] = "NOTPRESENT"
    if changed == "split": rows[0]["split"] = "test"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest_path = source / "task_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sync_snapshot(source, drive / "tasks/entity_v1")
    with pytest.raises(ValueError):
        _prepare(pilot)
    assert not (drive / "live_checkpoints/sft_entity_v1").exists()
