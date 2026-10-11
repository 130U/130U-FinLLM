"""Real tiny formal training across versioned local/Drive lifecycle failures."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from finllm.checkpoint import load_checkpoint, sha256_file
from finllm.expansion_runtime import (prepare_expansion_training, run_expansion_segment,
                                      retention_check, backup_expansion_training)
from finllm.runtime import sync_snapshot
from finllm.tokenizer import train_tokenizer
from finllm.training import run_training
import finllm.expansion_runtime as runtime
import finllm.continuation as continuation


@pytest.fixture(autouse=True)
def few_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _expansion_fixture(tmp_path, monkeypatch=None):
    """Synthetic CPU fixture only; real data builder retains its strict 8K gate."""
    project, drive = tmp_path / "original-runtime", tmp_path / "drive"
    project.mkdir()
    tokenizer_path = project / "tokenizer/seed_v1"
    tokenizer = train_tokenizer([
        "Original financial neutral bullish bearish CPU integration fixture. " * 30,
        "Separate training and held out documents target asset earnings and market. " * 30,
    ], tokenizer_path, 384)
    tokenizer_hash = sha256_file(tokenizer_path / "tokenizer.json")

    def pack(root, name, offset, count, partition):
        directory = root / name
        directory.mkdir(parents=True)
        token_path = directory / "tokens.bin"
        ((np.arange(count, dtype=np.uint16) + offset) % (tokenizer.get_vocab_size() - 6) + 6).tofile(token_path)
        path = directory / "manifest.json"
        path.write_text(json.dumps({
            "format": "finllm-uint16-v1", "mode": "formal", "vocab_size": tokenizer.get_vocab_size(),
            "tokenizer_sha256": tokenizer_hash, "token_count": count, "partition": partition,
            "eval_exclusion_sha256": "frozen-synthetic-fixture-only", "source_sha256": f"{root.name}/{name}/{offset}",
            "shards": [{"path": token_path.name, "count": count, "sha256": sha256_file(token_path)}],
        }), encoding="utf-8")
        return path

    seed_processed = project / "data/processed/seed_v1"
    general = pack(seed_processed, "general", 0, 193, "train")
    general_validation = pack(seed_processed, "general_validation", 1, 65, "validation")
    finance_validation = pack(seed_processed, "finance_validation", 2, 49, "validation")
    config = {
        "mode": "formal", "stage": "pretrain", "tokenizer": str(tokenizer_path), "data": str(general),
        "validation_data": str(general_validation), "output_dir": str(project / "runs/seed_v1_pretrain"),
        "backup_dir": str(drive / "live_checkpoints/seed_v1_pretrain"),
        "model": {"vocab_size": tokenizer.get_vocab_size(), "hidden_size": 16, "num_hidden_layers": 1,
                  "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 32,
                  "max_position_embeddings": 256, "dropout": .2},
        "train": {"device": "cpu", "amp": False, "batch_size": 2, "grad_accumulation": 2, "seq_len": 16,
                  "max_updates": 2, "schedule_updates": 4, "warmup_updates": 0, "seed": 17,
                  "save_every": 1, "learning_rate": .001},
    }
    run_training(config)
    seed_drive = drive / "seeds/seed_v1"
    for source, name in ((seed_processed, "processed"), (tokenizer_path, "tokenizer"),
                         (Path(config["output_dir"]), "run")):
        sync_snapshot(source, seed_drive / name)
    processed = project / "data/processed/expansion_v1"
    pack(processed, "general", 3, 1025, "train")
    pack(processed, "general_validation", 4, 81, "validation")
    report = {"status": "expansion_inputs_prepared_training_not_started", "format_version": 1,
              "tokenizer_sha256": tokenizer_hash, "tokenizer_vocabulary": tokenizer.get_vocab_size(),
              "token_counts": {"general": 1025, "general_validation": 81},
              "retention_validation": {"general": {"manifest": general_validation.as_posix(), "sha256": sha256_file(general_validation)},
                                       "finance": {"manifest": finance_validation.as_posix(), "sha256": sha256_file(finance_validation)}},
              "fixture_scope": "Synthetic tiny CPU runtime fixture; not the real 8K expansion preparation"}
    (processed / "expansion_summary.json").write_text(json.dumps(report), encoding="utf-8")
    sync_snapshot(processed, drive / "expansions/expansion_v1/processed")
    return project, drive


def _prepare(project, drive):
    return prepare_expansion_training(project, drive, min_parent_updates=2, probe_updates=1, target_updates=4)


def _runner(*args):
    assert args[0:2] == ("train", "--config")
    config = json.loads(Path(args[2]).read_text(encoding="utf-8"))
    if len(args) > 3:
        assert args[3] == "--resume"
        config["resume"] = str(args[4])
        config.pop("init_checkpoint", None)
    return run_training(config)


def _files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _finish(state):
    before = retention_check(state, "before")
    probe = run_expansion_segment(state, "probe", _runner)
    completed = run_expansion_segment(state, "continue", _runner)
    after = retention_check(state, "after")
    return before, probe, completed, after


def test_expansion_lifecycle_fresh_vm_and_repeat_preserve_measurements_and_seed(tmp_path, monkeypatch):
    project, drive = _expansion_fixture(tmp_path)
    seed_before = _files(drive / "seeds/seed_v1")
    local_seed_before = _files(project / "data/processed/seed_v1")
    state = _prepare(project, drive)
    before, probe, completed, after = _finish(state)
    assert before["inputs"]["device"] == "cpu" and before["inputs"]["amp"] is False
    assert probe["start_updates"] == 0 and probe["end_updates"] == 1
    assert completed["start_updates"] == 1 and completed["end_updates"] == 4
    assert completed["summary"]["parent_lineage"]["parent_updates"] == 2
    assert completed["new_experiment_tokens"] == 256
    assert completed["model_path_total_tokens"] == 384
    assert after["inputs"]["data_sha256"] == before["inputs"]["data_sha256"]
    assert backup_expansion_training(state)["verified"]
    run = Path(state["config"]["output_dir"])
    measured = (run / "run_summary.json").read_bytes()
    checkpoint_hash = sha256_file(run / "latest.pt")
    fresh = tmp_path / "fresh-runtime"
    fresh.mkdir()
    restored = _prepare(fresh, drive)
    assert restored["restored_updates"] == 4
    assert (Path(restored["config"]["output_dir"]) / "run_summary.json").read_bytes() == measured

    def forbidden(*args, **kwargs):
        pytest.fail("Repeated completed lifecycle must not evaluate again or perform optimizer work")

    monkeypatch.setattr(runtime, "_validate", forbidden)
    assert retention_check(restored, "before") == before
    assert run_expansion_segment(restored, "probe", _runner)["new_updates"] == 0
    assert run_expansion_segment(restored, "continue", _runner)["new_updates"] == 0
    assert retention_check(restored, "after") == after
    assert backup_expansion_training(restored)["verified"]
    assert (Path(restored["config"]["output_dir"]) / "run_summary.json").read_bytes() == measured
    assert sha256_file(Path(restored["config"]["output_dir"]) / "latest.pt") == checkpoint_hash
    assert _files(drive / "seeds/seed_v1") == seed_before
    assert _files(project / "data/processed/seed_v1") == local_seed_before


def test_retention_saved_measurement_retries_interrupted_snapshot_without_reevaluation(tmp_path, monkeypatch):
    project, drive = _expansion_fixture(tmp_path)
    state = _prepare(project, drive)
    destination = drive / "expansions/expansion_v1/retention/before"
    original = runtime.sync_snapshot

    def fail(source, target, **kwargs):
        if Path(target) == destination:
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "SNAPSHOT_INCOMPLETE.json").write_text("{}")
            raise OSError("simulated retention snapshot interruption")
        return original(source, target, **kwargs)

    monkeypatch.setattr(runtime, "sync_snapshot", fail)
    with pytest.raises(OSError, match="retention snapshot interruption"):
        retention_check(state, "before")
    measured = destination.with_suffix(".json").read_bytes()
    assert destination.with_suffix(".pending.json").is_file()
    monkeypatch.setattr(runtime, "sync_snapshot", original)
    monkeypatch.setattr(runtime, "_validate", lambda *args, **kwargs: pytest.fail("Must reuse the real saved result"))
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    restored = _prepare(fresh, drive)
    assert retention_check(restored, "before") == json.loads(measured)
    assert (destination / "result.json").read_bytes() == measured
    assert not destination.with_suffix(".pending.json").exists()
    assert not (destination / "SNAPSHOT_INCOMPLETE.json").exists()


def test_incomplete_existing_run_without_any_checkpoint_refuses_fresh_initialization(tmp_path):
    project, drive = _expansion_fixture(tmp_path)
    run = drive / "expansions/expansion_v1/run"
    run.mkdir()
    (run / "SNAPSHOT_INCOMPLETE.json").write_text("{}")
    with pytest.raises(ValueError, match="refusing fresh initialization"):
        _prepare(project, drive)
    assert not (project / "runs/expansion_v1_pretrain/latest.pt").exists()


def test_training_failure_blocks_backup_and_preserves_seed(tmp_path):
    project, drive = _expansion_fixture(tmp_path)
    seed = _files(drive / "seeds/seed_v1")
    state = _prepare(project, drive)
    retention_check(state, "before")
    run_expansion_segment(state, "probe", _runner)

    def fail(*args):
        raise RuntimeError("simulated training failure")

    with pytest.raises(RuntimeError, match="training failure"):
        run_expansion_segment(state, "continue", fail)
    with pytest.raises(ValueError, match="did not complete successfully"):
        backup_expansion_training(state)
    assert not (drive / "expansions/expansion_v1/run").exists()
    assert _files(drive / "seeds/seed_v1") == seed


def test_partial_final_snapshot_recovers_live_checkpoint_and_archived_measurements(tmp_path, monkeypatch):
    project, drive = _expansion_fixture(tmp_path)
    state = _prepare(project, drive)
    _finish(state)
    measured = (Path(state["config"]["output_dir"]) / "run_summary.json").read_bytes()
    destination = drive / "expansions/expansion_v1/run"
    original = runtime.sync_snapshot

    def fail(source, target, **kwargs):
        if Path(target) == destination:
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "SNAPSHOT_INCOMPLETE.json").write_text("{}")
            raise OSError("simulated final snapshot interruption")
        return original(source, target, **kwargs)

    monkeypatch.setattr(runtime, "sync_snapshot", fail)
    with pytest.raises(OSError, match="final snapshot interruption"):
        backup_expansion_training(state)
    monkeypatch.setattr(runtime, "sync_snapshot", original)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    restored = _prepare(fresh, drive)
    assert restored["restored_updates"] == 4
    assert (Path(restored["config"]["output_dir"]) / "run_summary.json").read_bytes() == measured
    assert (destination / "SNAPSHOT_INCOMPLETE.json").exists()
    _finish(restored)
    assert backup_expansion_training(restored)["verified"]
    assert (destination / "run_summary.json").read_bytes() == measured


@pytest.mark.parametrize("changed", ["provenance", "parent_hash", "report_lineage"])
def test_report_archive_recovery_checks_provenance_and_lineage(tmp_path, changed):
    project, drive = _expansion_fixture(tmp_path)
    state = _prepare(project, drive)
    retention_check(state, "before")
    run_expansion_segment(state, "probe", _runner)
    archive = drive / "expansions/expansion_v1/report_archive/updates_000000_to_000001"
    metadata = json.loads((archive / "archive.json").read_text())
    report = json.loads((archive / "run_summary.json").read_text())
    if changed == "provenance":
        metadata["provenance"]["data_sha256"] = "0" * 64
    elif changed == "parent_hash":
        metadata["parent_sha256"] = "0" * 64
    else:
        report["parent_lineage"]["parent_checkpoint_sha256"] = "0" * 64
    source = tmp_path / "altered-archive"
    source.mkdir()
    (source / "run_summary.json").write_text(json.dumps(report), encoding="utf-8")
    (source / "archive.json").write_text(json.dumps(metadata), encoding="utf-8")
    sync_snapshot(source, archive)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    with pytest.raises(ValueError, match="archive provenance or parent lineage"):
        _prepare(fresh, drive)


def test_interrupted_report_archive_recovers_exact_measured_bytes(tmp_path, monkeypatch):
    project, drive = _expansion_fixture(tmp_path)
    state = _prepare(project, drive)
    retention_check(state, "before")
    original = continuation.sync_snapshot
    archive = drive / "expansions/expansion_v1/report_archive/updates_000000_to_000001"

    def fail(source, target, **kwargs):
        if Path(target) == archive:
            (archive / "SNAPSHOT_INCOMPLETE.json").write_text("{}")
            raise OSError("simulated archive interruption")
        return original(source, target, **kwargs)

    monkeypatch.setattr(continuation, "sync_snapshot", fail)
    with pytest.raises(OSError, match="archive interruption"):
        run_expansion_segment(state, "probe", _runner)
    measured = (Path(state["config"]["output_dir"]) / "run_summary.json").read_bytes()
    monkeypatch.setattr(continuation, "sync_snapshot", original)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    restored = _prepare(fresh, drive)
    assert restored["restored_updates"] == 1
    assert (archive / "run_summary.json").read_bytes() == measured
    assert (Path(restored["config"]["output_dir"]) / "run_summary.json").read_bytes() == measured
    assert run_expansion_segment(restored, "probe", _runner)["new_updates"] == 0


def test_corrupt_complete_snapshot_recovers_verified_live_without_replacing_bad_source(tmp_path):
    project, drive = _expansion_fixture(tmp_path)
    state = _prepare(project, drive)
    _finish(state)
    backup_expansion_training(state)
    snapshot = drive / "expansions/expansion_v1/run"
    (snapshot / "run_summary.json").write_text("corrupt copied summary", encoding="utf-8")
    preserved = _files(snapshot)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    restored = _prepare(fresh, drive)
    assert restored["restored_updates"] == 4
    assert _files(snapshot) == preserved
    report = json.loads((Path(restored["config"]["output_dir"]) / "run_summary.json").read_text())
    assert report["elapsed_seconds"] > 0 and report["tokens_per_second"] > 0


def test_training_requires_verified_expanded_processed_backup(tmp_path):
    project, drive = _expansion_fixture(tmp_path)
    snapshot = drive / "expansions/expansion_v1/processed"
    snapshot.rename(snapshot.with_name("processed-unavailable-fixture"))
    with pytest.raises(ValueError, match="complete verified snapshot"):
        _prepare(project, drive)
    assert not (project / "runs/expansion_v1_pretrain/latest.pt").exists()


def test_known_completed_probe_never_restarts_if_all_checkpoint_copies_are_lost(tmp_path):
    project, drive = _expansion_fixture(tmp_path)
    state = _prepare(project, drive)
    retention_check(state, "before")
    run_expansion_segment(state, "probe", _runner)
    for key in ("output_dir", "backup_dir"):
        directory = Path(state["config"][key])
        directory.rename(directory.with_name(directory.name + "-unavailable-fixture"))
    with pytest.raises(ValueError, match="refusing fresh initialization"):
        run_expansion_segment(state, "probe", _runner)
