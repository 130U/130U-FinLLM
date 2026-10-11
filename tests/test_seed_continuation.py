"""Run the published continuation cells against real tiny CPU checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
import torch

from finllm.checkpoint import load_checkpoint, sha256_file
from finllm.continuation import prepare_seed_continuation, run_seed_segment, backup_seed_continuation
from finllm.runtime import sync_snapshot
from finllm.training import run_training
import finllm.continuation as continuation_module
from test_notebook_execution import (_cells, _cpu_config_cell, _drive_seed, _isolated_python,
                                     _kernel, _local_repository, _offline)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _pilot(tmp_path, monkeypatch):
    drive = _drive_seed(tmp_path)
    project = tmp_path / "original-runtime"
    project.mkdir()
    for source, destination in (("processed", "data/processed/seed_v1"), ("tokenizer", "tokenizer/seed_v1")):
        sync_snapshot(drive / "seeds/seed_v1" / source, project / destination, working_copy=True)
    monkeypatch.chdir(project)
    config = json.loads((project / "data/processed/seed_v1/pilot_config.json").read_text())
    config["backup_dir"] = str(drive / "live_checkpoints/seed_v1_pretrain")
    run_training(config)
    sync_snapshot(project / "runs/seed_v1_pretrain", drive / "seeds/seed_v1/run")
    return project, drive


def _runner(*args):
    assert args[0] == "train" and args[1] == "--config" and args[3] == "--resume"
    config = json.loads(Path(args[2]).read_text())
    config["resume"] = str(args[4])
    return run_training(config)


def _prepare(project, drive):
    return prepare_seed_continuation(project, drive, pilot_updates=2, probe_updates=3, target_updates=5)


def _continuation_cells():
    cells = _cells("02b_seed_continue.ipynb")
    setup = next(source for source in cells if "prepare_seed_continuation(" in source)
    setup = _cpu_config_cell(setup).replace("PILOT_UPDATES = 61", "PILOT_UPDATES = 2")
    setup = setup.replace("PROBE_UPDATES = 62", "PROBE_UPDATES = 3").replace("TARGET_UPDATES = 300", "TARGET_UPDATES = 5")
    probe = next(source for source in cells if 'probe_result = run_seed_segment' in source)
    continuation = next(source for source in cells if 'continuation_result = run_seed_segment' in source)
    backup = next(source for source in cells if 'backup_result = backup_seed_continuation' in source)
    return cells[0], setup, probe, continuation, backup


def test_generated_cells_preserve_pilot_and_repeat_completed_run(tmp_path, monkeypatch):
    project, drive = _pilot(tmp_path, monkeypatch)
    run = project / "runs/seed_v1_pretrain"
    pilot_bytes = (run / "run_summary.json").read_bytes()
    frozen_config = (project / "data/processed/seed_v1/pilot_config.json").read_bytes()
    _, setup, probe, continuation, backup = _continuation_cells()
    namespace = {"PROJECT": project, "DRIVE_ROOT": drive, "Path": Path, "json": json, "finllm": _runner}
    exec(setup, namespace)
    state = namespace["continuation"]
    assert (Path(state["pilot_archive"]) / "run_summary.json").read_bytes() == pilot_bytes
    with pytest.raises(RuntimeError, match="断点续训"):
        exec(continuation, namespace)
    with pytest.raises(RuntimeError, match="续训尚未"):
        exec(backup, namespace)
    exec(probe, namespace)
    assert namespace["probe_result"]["start_updates"] == 2
    assert namespace["probe_result"]["end_updates"] == 3
    exec(continuation, namespace)
    assert namespace["continuation_result"]["end_updates"] == 5
    assert namespace["continuation_result"]["tokens"] == 160
    exec(backup, namespace)
    assert namespace["CONTINUATION_BACKED_UP"] is True
    assert (Path(state["pilot_archive"]) / "run_summary.json").read_bytes() == pilot_bytes
    assert (project / "data/processed/seed_v1/pilot_config.json").read_bytes() == frozen_config
    for name in ("updates_000002_to_000003", "updates_000003_to_000005"):
        assert (drive / "seeds/seed_v1/report_archive" / name / "SNAPSHOT_COMPLETE.json").is_file()
    before = {path.name: (sha256_file(path), path.stat().st_mtime_ns) for path in run.iterdir()}
    exec(setup, namespace)
    exec(probe, namespace)
    exec(continuation, namespace)
    exec(backup, namespace)
    assert namespace["probe_result"]["new_updates"] == 0
    assert namespace["continuation_result"]["new_updates"] == 0
    assert before == {path.name: (sha256_file(path), path.stat().st_mtime_ns) for path in run.iterdir()}
    assert sha256_file(run / "latest.pt") == sha256_file(drive / "live_checkpoints/seed_v1_pretrain/latest.pt")


def test_missing_checkpoint_cannot_start_new_model(tmp_path, monkeypatch):
    project, drive = _pilot(tmp_path, monkeypatch)
    state = _prepare(project, drive)
    shutil.rmtree(project / "runs/seed_v1_pretrain")
    shutil.rmtree(drive / "seeds/seed_v1/run")
    shutil.rmtree(drive / "live_checkpoints/seed_v1_pretrain")
    with pytest.raises(ValueError, match="existing checkpoint"):
        _prepare(project, drive)
    calls = []
    with pytest.raises(ValueError, match="refusing random initialization"):
        run_seed_segment(state, "probe", lambda *args: calls.append(args))
    assert calls == []


def test_corrupt_inputs_and_pilot_archive_are_not_overwritten(tmp_path, monkeypatch):
    project, drive = _pilot(tmp_path, monkeypatch)
    state = _prepare(project, drive)
    frozen = project / "data/processed/seed_v1/pilot_config.json"
    original = frozen.read_bytes()
    frozen.write_text('{"broken": true}')
    with pytest.raises(ValueError, match="Local seed input differs"):
        _prepare(project, drive)
    with pytest.raises(ValueError, match="Frozen pilot configuration changed"):
        run_seed_segment(state, "probe", _runner)
    frozen.write_bytes(original)
    archive = Path(state["pilot_archive"]) / "run_summary.json"
    archive.write_text("changed")
    with pytest.raises(ValueError, match="missing or changed"):
        _prepare(project, drive)
    assert archive.read_text() == "changed"


def test_training_or_backup_failure_never_prints_success(tmp_path, monkeypatch):
    project, drive = _pilot(tmp_path, monkeypatch)
    _, setup, probe, continuation, backup = _continuation_cells()
    namespace = {"PROJECT": project, "DRIVE_ROOT": drive, "Path": Path, "json": json, "finllm": _runner}
    exec(setup, namespace)
    def fail(*args):
        raise RuntimeError("simulated CLI failure")
    namespace["finllm"] = fail
    with pytest.raises(RuntimeError, match="simulated CLI failure"):
        exec(probe, namespace)
    assert not namespace["RESUME_PROBE_SUCCEEDED"]
    with pytest.raises(RuntimeError, match="续训尚未"):
        exec(backup, namespace)
    namespace["finllm"] = _runner
    exec(probe, namespace)
    exec(continuation, namespace)
    namespace["backup_seed_continuation"] = fail
    with pytest.raises(RuntimeError, match="simulated CLI failure"):
        exec(backup, namespace)
    assert not namespace["CONTINUATION_BACKED_UP"]
    # The last persistent full run snapshot is still the completed pilot.
    assert json.loads((drive / "seeds/seed_v1/run/run_summary.json").read_text())["updates"] == 2


def test_runtime_change_still_fails_strict_trainer(tmp_path, monkeypatch):
    project, drive = _pilot(tmp_path, monkeypatch)
    state = _prepare(project, drive)
    before = sha256_file(project / "runs/seed_v1_pretrain/latest.pt")
    monkeypatch.setattr(torch, "__version__", str(torch.__version__) + "-different-runtime")
    with pytest.raises(ValueError, match="Resume provenance mismatch"):
        run_seed_segment(state, "probe", _runner)
    assert sha256_file(project / "runs/seed_v1_pretrain/latest.pt") == before


def test_interrupted_report_archive_recovers_in_a_new_runtime(tmp_path, monkeypatch):
    project, drive = _pilot(tmp_path, monkeypatch)
    state = _prepare(project, drive)
    original_sync = continuation_module.sync_snapshot
    archive = drive / "seeds/seed_v1/report_archive/updates_000002_to_000003"
    def interrupted(source, destination, **kwargs):
        if Path(destination) == archive:
            (archive / "SNAPSHOT_INCOMPLETE.json").write_text('{}')
            raise OSError("interrupted report copy")
        return original_sync(source, destination, **kwargs)
    monkeypatch.setattr(continuation_module, "sync_snapshot", interrupted)
    with pytest.raises(OSError, match="interrupted report"):
        run_seed_segment(state, "probe", _runner)
    assert load_checkpoint(project / "runs/seed_v1_pretrain/latest.pt")["progress"]["updates"] == 3
    assert (archive / "PENDING_REPORT.json").is_file()
    measured_probe = (project / "runs/seed_v1_pretrain/run_summary.json").read_bytes()
    monkeypatch.setattr(continuation_module, "sync_snapshot", original_sync)
    fresh = tmp_path / "fresh-runtime"
    fresh.mkdir()
    monkeypatch.chdir(fresh)
    restored = _prepare(fresh, drive)
    assert restored["restored_updates"] == 3
    assert not (archive / "PENDING_REPORT.json").exists()
    assert (archive / "run_summary.json").read_bytes() == measured_probe
    metadata = json.loads((archive / "archive.json").read_text())
    assert (metadata["start_updates"], metadata["end_updates"]) == (2, 3)
    assert run_seed_segment(restored, "probe", _runner)["new_updates"] == 0
    run_seed_segment(restored, "continue", _runner)
    assert backup_seed_continuation(restored)["verified"]


def test_interrupted_final_snapshot_recovers_live_and_archived_metrics(tmp_path, monkeypatch):
    project, drive = _pilot(tmp_path, monkeypatch)
    state = _prepare(project, drive)
    run_seed_segment(state, "probe", _runner)
    run_seed_segment(state, "continue", _runner)
    measured = (project / "runs/seed_v1_pretrain/run_summary.json").read_bytes()
    snapshot = drive / "seeds/seed_v1/run"
    original_sync = continuation_module.sync_snapshot
    def interrupted(source, destination, **kwargs):
        if Path(destination) == snapshot:
            (snapshot / "SNAPSHOT_COMPLETE.json").unlink()
            (snapshot / "SNAPSHOT_INCOMPLETE.json").write_text('{}')
            raise OSError("interrupted final backup")
        return original_sync(source, destination, **kwargs)
    monkeypatch.setattr(continuation_module, "sync_snapshot", interrupted)
    with pytest.raises(OSError, match="interrupted final"):
        backup_seed_continuation(state)
    monkeypatch.setattr(continuation_module, "sync_snapshot", original_sync)
    fresh = tmp_path / "fresh-runtime"
    fresh.mkdir()
    monkeypatch.chdir(fresh)
    restored = _prepare(fresh, drive)
    assert restored["restored_updates"] == 5
    assert (snapshot / "SNAPSHOT_INCOMPLETE.json").exists()
    assert (fresh / "runs/seed_v1_pretrain/run_summary.json").read_bytes() == measured
    assert run_seed_segment(restored, "probe", _runner)["new_updates"] == 0
    assert run_seed_segment(restored, "continue", _runner)["new_updates"] == 0
    assert backup_seed_continuation(restored)["verified"]
    assert not (snapshot / "SNAPSHOT_INCOMPLETE.json").exists()
    assert (snapshot / "run_summary.json").read_bytes() == measured


def test_new_kernels_restore_probe_and_finish_without_losing_pilot(tmp_path, monkeypatch):
    original, drive = _pilot(tmp_path, monkeypatch)
    pilot = (drive / "seeds/seed_v1/run/run_summary.json").read_bytes()
    # Bootstrap a fresh checkout in a real kernel, using the actual published
    # continuation cells. External Git/pip/Drive are substituted, not training.
    repository = _local_repository(tmp_path / "upstream")
    bootstrap, setup, probe, continuation, backup = _continuation_cells()
    first_project = tmp_path / "first" / "130U-FinLLM"
    first_project.parent.mkdir()
    python = _isolated_python(tmp_path / "python")
    first_kernel = tmp_path / "first-kernel"
    first_kernel.mkdir()
    with _kernel(python, first_kernel) as execute:
        execute(_offline(bootstrap, repository, first_project))
        execute(f"DRIVE_ROOT = Path({str(drive)!r})")
        execute(setup)
        execute(probe)
        execute("assert probe_result['start_updates'] == 2 and probe_result['end_updates'] == 3")
    # No final backup yet: snapshot is pilot2, live is probe3. A new VM must
    # select3, preserve the archived pilot2, and avoid re-executing the probe.
    assert json.loads((drive / "seeds/seed_v1/run/run_summary.json").read_text())["updates"] == 2
    second_project = tmp_path / "second" / "130U-FinLLM"
    second_project.parent.mkdir()
    second_kernel = tmp_path / "second-kernel"
    second_kernel.mkdir()
    with _kernel(python, second_kernel) as execute:
        execute(_offline(bootstrap, repository, second_project))
        execute(f"DRIVE_ROOT = Path({str(drive)!r})")
        execute(setup)
        execute("assert continuation['restored_updates'] == 3")
        execute(probe)
        execute("assert probe_result['new_updates'] == 0")
        execute(continuation)
        execute(backup)
        execute("assert continuation_result['start_updates'] == 3 and continuation_result['end_updates'] == 5\nassert CONTINUATION_BACKED_UP")
    archive = drive / "seeds/seed_v1/report_archive/pilot_updates_000002/run_summary.json"
    assert archive.read_bytes() == pilot
    result = load_checkpoint(second_project / "runs/seed_v1_pretrain/latest.pt")
    assert result["progress"]["updates"] == 5
    assert result["progress"]["tokens"] == 160
    print("CONTINUATION_NOTEBOOK_EVIDENCE=" + json.dumps({
        "device": "cpu", "synthetic_test_only": True, "pilot_updates": 2,
        "fresh_runtime_probe_updates": 3, "second_runtime_final_updates": 5,
        "final_tokens": 160, "pilot_report_bytes_retained": True,
        "full_backup_verified": True,
    }))
