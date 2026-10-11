"""Real tiny formal 02c -> strict 02d continuation and interruption lifecycle."""
import json
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from finllm.checkpoint import load_checkpoint, save_checkpoint, sha256_file
from finllm.expansion_runtime import backup_expansion_training
from finllm.expansion_continuation import (
    prepare_expansion_continuation, run_expansion_continuation,
    retention_continuation_check, backup_expansion_continuation,
)
from finllm.runtime import sync_snapshot
from finllm.training import run_training
import finllm.continuation as archive_runtime
import finllm.expansion_continuation as continuation
from test_expansion_runtime import _expansion_fixture, _prepare, _finish, _runner, _files


@pytest.fixture(autouse=True)
def few_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _completed(tmp_path):
    project, drive = _expansion_fixture(tmp_path)
    state = _prepare(project, drive)
    _finish(state)
    backup_expansion_training(state)
    return project, drive


def _continued(project, drive):
    return prepare_expansion_continuation(project, drive, completed_updates=4,
                                         probe_updates=5, target_updates=8, min_parent_updates=2)


def _complete(state):
    before = retention_continuation_check(state, "before")
    probe = run_expansion_continuation(state, "probe", _runner)
    trained = run_expansion_continuation(state, "continue", _runner)
    after = retention_continuation_check(state, "after")
    return before, probe, trained, after


def _inventory(directory):
    return {p.relative_to(directory).as_posix(): (sha256_file(p), p.stat().st_mtime_ns)
            for p in directory.rglob("*") if p.is_file()}


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            _assert_equal(first, second)
    else:
        assert actual == expected


def test_lifecycle_exact_optimizer_resume_archive_and_two_vm_noop(tmp_path, monkeypatch):
    project, drive = _completed(tmp_path)
    old_retention = _inventory(drive / "expansions/expansion_v1/retention")
    old_seed = _inventory(drive / "seeds/seed_v1")
    source = drive / "expansions/expansion_v1/run"
    old_run = _files(source)
    state = _continued(project, drive)
    archive = Path(state["predecessor_archive"])
    for name, body in old_run.items():
        if not name.startswith("SNAPSHOT_"):
            assert (archive / name).read_bytes() == body
    baseline_inventory = _inventory(archive)
    baseline = load_checkpoint(archive / "latest.pt", allow_previous=False)
    before = retention_continuation_check(state, "before")

    def strict_runner(*args):
        assert args[3] == "--resume"
        config = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        assert "init_checkpoint" not in config
        restored = load_checkpoint(args[4])
        assert restored["progress"]["updates"] == 4
        _assert_equal(restored["optimizer_state"], baseline["optimizer_state"])
        _assert_equal(restored["scheduler_state"], baseline["scheduler_state"])
        _assert_equal(restored["rng_state"], baseline["rng_state"])
        return _runner(*args)

    probe = run_expansion_continuation(state, "probe", strict_runner)
    fifth = load_checkpoint(Path(state["config"]["output_dir"]) / "latest.pt")
    assert {float(x["step"]) for x in baseline["optimizer_state"]["state"].values()} == {4.0}
    assert {float(x["step"]) for x in fifth["optimizer_state"]["state"].values()} == {5.0}
    assert fifth["progress"]["cursor"] == baseline["progress"]["cursor"] + 4
    result = run_expansion_continuation(state, "continue", _runner)
    after = retention_continuation_check(state, "after")
    assert probe["start_updates"] == 4 and probe["end_updates"] == 5
    assert result["start_updates"] == 5 and result["end_updates"] == 8
    assert result["new_updates"] == 3 and result["segment_updates"] == 4
    assert result["segment_tokens"] == 256 and result["new_tokens"] == 192
    assert result["new_experiment_tokens"] == 512 and result["parent_tokens"] == 128
    assert result["model_path_total_tokens"] == 640
    assert before["inputs"]["data_sha256"] == after["inputs"]["data_sha256"]
    assert backup_expansion_continuation(state)["verified"]
    output = Path(state["config"]["output_dir"])
    measurement = (output / "run_summary.json").read_bytes()
    final_inventory = _inventory(output)
    fresh = tmp_path / "fresh-runtime"
    fresh.mkdir()
    restored = _continued(fresh, drive)
    assert restored["restored_updates"] == 8
    monkeypatch.setattr(continuation, "_validate", lambda *args, **kwargs: pytest.fail("Saved retention must be reused"))
    assert retention_continuation_check(restored, "before") == before
    assert run_expansion_continuation(restored, "probe", _runner)["new_updates"] == 0
    assert run_expansion_continuation(restored, "continue", _runner)["new_updates"] == 0
    assert retention_continuation_check(restored, "after") == after
    assert backup_expansion_continuation(restored)["verified"]
    restored_output = Path(restored["config"]["output_dir"])
    assert (restored_output / "run_summary.json").read_bytes() == measurement
    assert sha256_file(restored_output / "latest.pt") == final_inventory["latest.pt"][0]
    assert _inventory(archive) == baseline_inventory
    assert _inventory(drive / "expansions/expansion_v1/retention") == old_retention
    assert _inventory(drive / "seeds/seed_v1") == old_seed


def test_retention_and_probe_preserve_exact_dropout_resume_numerics(tmp_path):
    project, drive = _completed(tmp_path)
    state = _continued(project, drive)
    archived = Path(state["predecessor_archive"]) / "latest.pt"
    assert load_checkpoint(archived)["model_config"]["dropout"] == .2
    direct = copy.deepcopy(state["config"])
    direct.pop("init_checkpoint")
    direct.update(resume=str(archived), output_dir=str(project / "runs/direct_comparison"),
                  backup_dir=str(drive / "direct_comparison_live"))
    run_training(direct)
    expected = load_checkpoint(Path(direct["output_dir"]) / "latest.pt")
    _complete(state)
    actual = load_checkpoint(Path(state["config"]["output_dir"]) / "latest.pt")
    for key in ("model_state", "optimizer_state", "scheduler_state", "scaler_state", "rng_state", "progress"):
        _assert_equal(actual[key], expected[key])
    assert actual["last_loss"] == expected["last_loss"]
    assert actual["last_validation"] == expected["last_validation"]


def test_newer_live_restores_without_rolling_back_archived_predecessor(tmp_path):
    project, drive = _completed(tmp_path)
    state = _continued(project, drive)
    old = _inventory(Path(state["predecessor_archive"]))
    retention_continuation_check(state, "before")
    run_expansion_continuation(state, "probe", _runner)
    # Simulate losing the VM while full shared backup is still the completed4.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    restored = _continued(fresh, drive)
    assert restored["restored_updates"] == 5
    result = run_expansion_continuation(restored, "continue", _runner)
    assert result["start_updates"] == 5 and result["end_updates"] == 8
    assert _inventory(Path(state["predecessor_archive"])) == old


@pytest.mark.parametrize("name", ["run_summary.json", "latest.pt", "latest.pt.json"])
def test_corrupt_completed_02c_does_not_create_archive_or_initialize(tmp_path, name):
    project, drive = _completed(tmp_path)
    source = drive / "expansions/expansion_v1/run"
    (source / name).write_bytes(b"corrupted certified input")
    preserved = _files(source)
    with pytest.raises(ValueError, match="missing or changed"):
        _continued(project, drive)
    assert _files(source) == preserved
    assert not (drive / "expansions/expansion_v1/continuations").exists()


def test_missing_original_after_blocks_continuation(tmp_path):
    project, drive = _completed(tmp_path)
    snapshot = drive / "expansions/expansion_v1/retention/after"
    snapshot.rename(snapshot.with_name("unavailable-after"))
    with pytest.raises((OSError, ValueError)):
        _continued(project, drive)
    assert load_checkpoint(project / "runs/expansion_v1_pretrain/latest.pt")["progress"]["updates"] == 4


def test_zero_time_predecessor_report_is_not_accepted_as_a_measured_run(tmp_path):
    project, drive = _completed(tmp_path)
    source = drive / "expansions/expansion_v1/run"
    modified = tmp_path / "modified-copy"
    sync_snapshot(source, modified, working_copy=True)
    report_path = modified / "run_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["elapsed_seconds"] = 0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    sync_snapshot(modified, source)
    with pytest.raises(ValueError, match="positive-time measured"):
        _continued(project, drive)
    assert not (drive / "expansions/expansion_v1/continuations").exists()


def test_corrupt_frozen_predecessor_is_not_replaced_by_stale_source(tmp_path):
    project, drive = _completed(tmp_path)
    state = _continued(project, drive)
    archive = Path(state["predecessor_archive"])
    (archive / "latest.pt").write_bytes(b"corrupt immutable predecessor")
    preserved = _files(archive)
    with pytest.raises(ValueError, match="missing or changed"):
        _continued(project, drive)
    assert _files(archive) == preserved


def test_runtime_drift_fails_before_freezing_or_advancing(tmp_path, monkeypatch):
    project, drive = _completed(tmp_path)
    old = _files(drive / "expansions/expansion_v1")
    monkeypatch.setattr(continuation.torch, "__version__", "unsupported-future-runtime")
    with pytest.raises(ValueError, match="runtime differs"):
        _continued(project, drive)
    assert _files(drive / "expansions/expansion_v1") == old


@pytest.mark.parametrize("changed", ["max_tokens", "validation_every", "save_every"])
def test_nonsemantic_training_settings_cannot_change_on_resume(tmp_path, changed):
    project, drive = _completed(tmp_path)
    state = _continued(project, drive)
    state["config"]["train"][changed] += 1
    with pytest.raises(ValueError, match="may change only max_updates"):
        retention_continuation_check(state, "before")
    with pytest.raises(ValueError, match="did not complete successfully"):
        backup_expansion_continuation(state)


def test_training_failure_does_not_publish_complete_backup(tmp_path):
    project, drive = _completed(tmp_path)
    state = _continued(project, drive)
    shared = _files(drive / "expansions/expansion_v1/run")
    retention_continuation_check(state, "before")
    run_expansion_continuation(state, "probe", _runner)
    with pytest.raises(RuntimeError, match="simulated train"):
        run_expansion_continuation(state, "continue", lambda *args: (_ for _ in ()).throw(RuntimeError("simulated train")))
    with pytest.raises(ValueError, match="did not complete successfully"):
        backup_expansion_continuation(state)
    assert _files(drive / "expansions/expansion_v1/run") == shared


def test_retention_failure_blocks_final_backup_and_can_retry(tmp_path, monkeypatch):
    project, drive = _completed(tmp_path)
    state = _continued(project, drive)
    retention_continuation_check(state, "before")
    run_expansion_continuation(state, "probe", _runner)
    run_expansion_continuation(state, "continue", _runner)
    original = continuation._validate
    monkeypatch.setattr(continuation, "_validate", lambda *args, **kwargs: {"loss": float("nan"), "supervised_tokens": 64})
    with pytest.raises(ValueError, match="invalid metrics"):
        retention_continuation_check(state, "after")
    with pytest.raises(ValueError, match="did not complete successfully"):
        backup_expansion_continuation(state)
    monkeypatch.setattr(continuation, "_validate", original)
    retention_continuation_check(state, "after")
    assert backup_expansion_continuation(state)["verified"]


def test_partial_predecessor_copy_retries_before_any_new_training(tmp_path, monkeypatch):
    project, drive = _completed(tmp_path)
    old = _files(drive / "expansions/expansion_v1/run")
    original = continuation.sync_snapshot

    def fail(source, target, **kwargs):
        if Path(target).name == "baseline_run":
            Path(target).mkdir(parents=True, exist_ok=True)
            (Path(target) / "SNAPSHOT_INCOMPLETE.json").write_text("{}")
            raise OSError("simulated predecessor copy failure")
        return original(source, target, **kwargs)

    monkeypatch.setattr(continuation, "sync_snapshot", fail)
    with pytest.raises(OSError, match="predecessor copy"):
        _continued(project, drive)
    assert _files(drive / "expansions/expansion_v1/run") == old
    monkeypatch.setattr(continuation, "sync_snapshot", original)
    assert _continued(project, drive)["restored_updates"] == 4


def test_partial_predecessor_retry_refuses_changed_source_inventory(tmp_path, monkeypatch):
    project, drive = _completed(tmp_path)
    original = continuation.sync_snapshot

    def fail(source, target, **kwargs):
        if Path(target).name == "baseline_run":
            Path(target).mkdir(parents=True, exist_ok=True)
            (Path(target) / "SNAPSHOT_INCOMPLETE.json").write_text("{}")
            raise OSError("simulated predecessor interruption")
        return original(source, target, **kwargs)

    monkeypatch.setattr(continuation, "sync_snapshot", fail)
    with pytest.raises(OSError, match="predecessor interruption"):
        _continued(project, drive)
    monkeypatch.setattr(continuation, "sync_snapshot", original)
    source = drive / "expansions/expansion_v1/run"
    modified = tmp_path / "modified-complete-copy"
    sync_snapshot(source, modified, working_copy=True)
    with (modified / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    sync_snapshot(modified, source)
    pending = drive / "expansions/expansion_v1/continuations/updates_000004_to_000008/predecessor_pending.json"
    preserved = pending.read_bytes()
    with pytest.raises(ValueError, match="different source bytes"):
        _continued(project, drive)
    assert pending.read_bytes() == preserved


def test_interrupted_retention_snapshot_reuses_real_saved_measurement(tmp_path, monkeypatch):
    project, drive = _completed(tmp_path)
    state = _continued(project, drive)
    destination = state["continuation_drive"] / "retention/before"
    import finllm.expansion_runtime as runtime
    original = runtime.sync_snapshot

    def fail(source, target, **kwargs):
        if Path(target) == destination:
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "SNAPSHOT_INCOMPLETE.json").write_text("{}")
            raise OSError("simulated retention IO failure")
        return original(source, target, **kwargs)

    monkeypatch.setattr(runtime, "sync_snapshot", fail)
    with pytest.raises(OSError, match="retention IO"):
        retention_continuation_check(state, "before")
    measured = destination.with_suffix(".json").read_bytes()
    assert not state["before_retention_succeeded"]
    monkeypatch.setattr(runtime, "sync_snapshot", original)
    monkeypatch.setattr(continuation, "_validate", lambda *args, **kwargs: pytest.fail("Do not remeasure a saved real result"))
    assert retention_continuation_check(state, "before") == json.loads(measured)
    assert (destination / "result.json").read_bytes() == measured


def test_interrupted_continuation_report_archive_recovers_measured_bytes(tmp_path, monkeypatch):
    project, drive = _completed(tmp_path)
    state = _continued(project, drive)
    retention_continuation_check(state, "before")
    original = archive_runtime.sync_snapshot

    def fail(source, target, **kwargs):
        if Path(target).is_relative_to(state["continuation_drive"] / "report_archive"):
            (Path(target) / "SNAPSHOT_INCOMPLETE.json").write_text("{}")
            raise OSError("simulated segment archive IO failure")
        return original(source, target, **kwargs)

    monkeypatch.setattr(archive_runtime, "sync_snapshot", fail)
    with pytest.raises(OSError, match="segment archive IO"):
        run_expansion_continuation(state, "probe", _runner)
    assert not state["continuation_succeeded"]
    measured = (Path(state["config"]["output_dir"]) / "run_summary.json").read_bytes()
    monkeypatch.setattr(archive_runtime, "sync_snapshot", original)
    restored = _continued(project, drive)
    assert run_expansion_continuation(restored, "probe", _runner)["new_updates"] == 0
    assert (Path(state["config"]["output_dir"]) / "run_summary.json").read_bytes() == measured


def test_final_snapshot_failure_recovers_live_without_losing_original_reports(tmp_path, monkeypatch):
    project, drive = _completed(tmp_path)
    state = _continued(project, drive)
    _complete(state)
    destination = drive / "expansions/expansion_v1/run"
    original = continuation.sync_snapshot

    def fail(source, target, **kwargs):
        if Path(target) == destination:
            (destination / "SNAPSHOT_INCOMPLETE.json").write_text("{}")
            raise OSError("simulated final copy failure")
        return original(source, target, **kwargs)

    monkeypatch.setattr(continuation, "sync_snapshot", fail)
    with pytest.raises(OSError, match="final copy"):
        backup_expansion_continuation(state)
    measurement = (Path(state["config"]["output_dir"]) / "run_summary.json").read_bytes()
    monkeypatch.setattr(continuation, "sync_snapshot", original)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    restored = _continued(fresh, drive)
    assert restored["restored_updates"] == 8
    _complete(restored)
    assert backup_expansion_continuation(restored)["verified"]
    assert (destination / "run_summary.json").read_bytes() == measurement


def test_epoch_replay_state_is_rejected_before_running(tmp_path):
    project, drive = _completed(tmp_path)
    state = _continued(project, drive)
    local = Path(state["config"]["output_dir"]) / "latest.pt"
    payload = load_checkpoint(local)
    payload["progress"]["epoch"] = 1
    save_checkpoint(local, payload)
    save_checkpoint(Path(state["config"]["backup_dir"]) / "latest.pt", payload)
    retention_continuation_check(state, "before")
    with pytest.raises(ValueError, match="insufficient without replay"):
        run_expansion_continuation(state, "probe", _runner)


@pytest.mark.parametrize("budgets", [(0, 1, 2), (4, 4, 8), (4, 5, 5), (True, 5, 8)])
def test_invalid_boundaries_write_nothing(tmp_path, budgets):
    with pytest.raises(ValueError, match="Invalid expansion continuation"):
        prepare_expansion_continuation(tmp_path / "project", tmp_path / "drive",
                                       completed_updates=budgets[0], probe_updates=budgets[1], target_updates=budgets[2])
    assert not list(tmp_path.iterdir())
