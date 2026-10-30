"""Actual tiny CPU domain transfer, exact resume and persistent failure recovery."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from finllm.checkpoint import load_checkpoint, save_checkpoint, sha256_file
from finllm.domain_runtime import (backup_domain_training, prepare_domain_training,
                                   retention_domain_check, run_domain_segment)
from finllm.expansion_runtime import backup_expansion_training
from finllm.model import FinLLM, ModelConfig
from finllm.runtime import sync_snapshot
from finllm.training import run_training
import finllm.domain_runtime as runtime
import finllm.continuation as continuation
from test_expansion_runtime import _expansion_fixture, _prepare, _finish, _runner


@pytest.fixture(autouse=True)
def few_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _domain_fixture(tmp_path):
    """Synthetic fixture only; real prepared inputs retain their 8K/data audit."""
    project, drive = _expansion_fixture(tmp_path)
    parent = _prepare(project, drive)
    _finish(parent)
    assert backup_expansion_training(parent)["verified"]
    root = project / "data/processed/expansion_v1"
    summary_path = root / "expansion_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    general = json.loads((root / "general/manifest.json").read_text(encoding="utf-8"))

    def pack(name, count, offset, partition):
        directory = root / name
        directory.mkdir()
        data = directory / "tokens.bin"
        ((np.arange(count, dtype=np.uint16) + offset) % (general["vocab_size"] - 6) + 6).tofile(data)
        manifest = {"format": "finllm-uint16-v1", "mode": "formal", "vocab_size": general["vocab_size"],
                    "tokenizer_sha256": general["tokenizer_sha256"], "token_count": count, "partition": partition,
                    "eval_exclusion_sha256": general["eval_exclusion_sha256"],
                    "source_sha256": f"domain-fixture/{name}/{offset}",
                    "shards": [{"path": data.name, "count": count, "sha256": sha256_file(data)}]}
        if name == "domain_mix":
            manifest.update(mixture_token_counts={"general": 128, "finance": 512},
                            source_unique_token_counts={"general": 1025, "finance": 512})
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    pack("finance", 512, 8, "train")
    pack("domain_mix", 640, 6, "train")
    pack("finance_validation", 97, 7, "validation")
    mix_path = root / "domain_mix/manifest.json"
    mix = json.loads(mix_path.read_text(encoding="utf-8"))
    source_hashes = [sha256_file(root / name / "manifest.json") for name in ("general", "finance")]
    mix.update(source_manifests=source_hashes, seed=130,
               source_sha256=hashlib.sha256(json.dumps(source_hashes).encode()).hexdigest())
    mix_path.write_text(json.dumps(mix), encoding="utf-8")
    summary["token_counts"].update(finance=512, domain_mix=640, finance_validation=97)
    summary["settings"] = {"seed": 130}
    summary["seed_provenance"] = {"task_exclusion_sha256": json.loads(
        (project / "data/processed/seed_v1/general_validation/manifest.json").read_text())["eval_exclusion_sha256"]}
    summary["domain_mix"] = {"finance_fraction": .8, "repeated_source_tokens": 0,
                             "counts_are_not_additional_unique_data": True}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    sync_snapshot(root, drive / "expansions/expansion_v1/processed")
    return project, drive


def _domain(project, drive, **kwargs):
    return prepare_domain_training(project, drive, min_parent_updates=4, probe_updates=1, target_updates=3, **kwargs)


def _inventory(root):
    return {p.relative_to(root).as_posix(): (sha256_file(p), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def _complete(state):
    before = retention_domain_check(state, "before")
    probe = run_domain_segment(state, "probe", _runner)
    result = run_domain_segment(state, "continue", _runner)
    after = retention_domain_check(state, "after")
    assert backup_domain_training(state)["verified"]
    return before, probe, result, after


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _equal(a, b)
    else:
        assert actual == expected


def test_domain_weights_transfer_resets_optimizer_and_preserves_all_parent_artifacts(tmp_path, monkeypatch):
    project, drive = _domain_fixture(tmp_path)
    seed_before = _inventory(drive / "seeds/seed_v1")
    expansion_before = _inventory(drive / "expansions/expansion_v1")
    state = _domain(project, drive)
    parent = load_checkpoint(state["config"]["init_checkpoint"])
    expected = FinLLM(ModelConfig(**parent["model_config"]))
    expected.load_state_dict(parent["model_state"])
    captured = []
    original = torch.optim.AdamW.step

    def inspect(self, *args, **kwargs):
        if not captured:
            assert not self.state
            captured.extend(p.detach().clone() for group in self.param_groups for p in group["params"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", inspect)
    before, probe, result, after = _complete(state)
    for actual, wanted in zip(captured, expected.parameters()):
        assert torch.equal(actual, wanted)
    checkpoint = load_checkpoint(Path(state["config"]["output_dir"]) / "latest.pt")
    assert {float(item["step"]) for item in parent["optimizer_state"]["state"].values()} == {4.0}
    assert {float(item["step"]) for item in checkpoint["optimizer_state"]["state"].values()} == {3.0}
    assert checkpoint["progress"]["epoch"] == 0
    assert checkpoint["config"]["domain_transition"]["parent_checkpoint_sha256"] == state["parent"]["checkpoint_sha256"]
    assert probe["start_updates"] == 0 and probe["end_updates"] == 1
    assert result["start_updates"] == 1 and result["end_updates"] == 3
    assert result["parent_tokens"] == parent["progress"]["tokens"]
    assert result["seed_tokens"] == parent["provenance"]["parent_lineage"]["parent_tokens"]
    assert result["model_path_total_tokens"] == result["seed_tokens"] + result["parent_tokens"] + result["summary"]["tokens"]
    assert "not completion" in result["budget_scope"]
    assert before["inputs"]["data_sha256"] == after["inputs"]["data_sha256"]
    assert _inventory(drive / "seeds/seed_v1") == seed_before
    assert _inventory(drive / "expansions/expansion_v1") == expansion_before


def test_domain_dropout_direct_and_segmented_resume_are_numerically_identical(tmp_path):
    project, drive = _domain_fixture(tmp_path)
    state = _domain(project, drive)
    direct = copy.deepcopy(state["config"])
    direct.update(output_dir=str(project / "runs/direct_domain"), backup_dir=str(drive / "direct_domain_live"))
    run_training(direct)
    expected = load_checkpoint(Path(direct["output_dir"]) / "latest.pt")
    _complete(state)
    actual = load_checkpoint(Path(state["config"]["output_dir"]) / "latest.pt")
    for key in ("model_state", "optimizer_state", "scheduler_state", "scaler_state", "rng_state", "progress"):
        _equal(actual[key], expected[key])
    assert actual["last_loss"] == expected["last_loss"]
    assert actual["last_validation"] == expected["last_validation"]


def test_domain_fresh_directory_and_completed_noop_retain_measured_bytes(tmp_path, monkeypatch):
    project, drive = _domain_fixture(tmp_path)
    state = _domain(project, drive)
    before, _, _, after = _complete(state)
    output = Path(state["config"]["output_dir"])
    saved = (output / "run_summary.json").read_bytes()
    frozen = _inventory(state["paths"]["drive"] / "parent")
    original = _inventory(output)
    repeated = _domain(project, drive)
    for phase in ("before", "after"):
        monkeypatch.setattr(runtime, "_validate", lambda *args, **kwargs: pytest.fail("Do not repeat saved retention"))
    retention_domain_check(repeated, "before")
    assert run_domain_segment(repeated, "probe", _runner)["new_updates"] == 0
    assert run_domain_segment(repeated, "continue", _runner)["new_updates"] == 0
    retention_domain_check(repeated, "after")
    assert backup_domain_training(repeated)["verified"]
    assert _inventory(output) == original
    fresh = tmp_path / "second-runtime"
    fresh.mkdir()
    restored = _domain(fresh, drive)
    assert restored["restored_updates"] == 3
    assert retention_domain_check(restored, "before") == before
    assert run_domain_segment(restored, "probe", _runner)["new_updates"] == 0
    assert run_domain_segment(restored, "continue", _runner)["new_updates"] == 0
    assert retention_domain_check(restored, "after") == after
    assert backup_domain_training(restored)["verified"]
    assert (Path(restored["config"]["output_dir"]) / "run_summary.json").read_bytes() == saved
    assert _inventory(state["paths"]["drive"] / "parent") == frozen


@pytest.mark.parametrize("name", ["latest.pt", "latest.pt.json", "run_summary.json"])
def test_domain_corrupt_parent_fails_before_freezing_or_training(tmp_path, name):
    project, drive = _domain_fixture(tmp_path)
    (drive / "expansions/expansion_v1/run" / name).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        _domain(project, drive)
    assert not (drive / "domains/domain_v1/parent/SNAPSHOT_COMPLETE.json").exists()
    assert not (project / "runs/domain_v1").exists()


def test_domain_requires_matching_completed_parent_retention(tmp_path):
    project, drive = _domain_fixture(tmp_path)
    (drive / "expansions/expansion_v1/retention/after/result.json").write_text("{}")
    with pytest.raises(ValueError):
        _domain(project, drive)
    assert not (project / "runs/domain_v1").exists()


def test_domain_requires_completed_parent_budget_and_minimum(tmp_path):
    project, drive = _domain_fixture(tmp_path)
    with pytest.raises(ValueError, match="completed formal expansion"):
        prepare_domain_training(project, drive, min_parent_updates=5, probe_updates=1, target_updates=3)


@pytest.mark.parametrize("field", ["data_sha256", "validation_sha256"])
def test_domain_rejects_parent_from_different_expansion_inputs(tmp_path, field):
    project, drive = _domain_fixture(tmp_path)
    working = project / "runs/expansion_v1_pretrain"
    checkpoint = load_checkpoint(working / "latest.pt")
    checkpoint["provenance"][field] = "f" * 64
    save_checkpoint(working / "latest.pt", checkpoint)
    sync_snapshot(working, drive / "expansions/expansion_v1/run")
    with pytest.raises(ValueError, match="completed formal expansion"):
        _domain(project, drive)
    assert not (project / "runs/domain_v1").exists()


@pytest.mark.parametrize("field,value", [
    ("source_manifests", ["f" * 64, "e" * 64]),
    ("source_unique_token_counts", {"general": 9999, "finance": 512}),
    ("source_sha256", "f" * 64), ("eval_exclusion_sha256", "different-task-pool"),
    ("seed", 999),
])
def test_domain_rejects_mixture_from_different_sources_or_frozen_pool(tmp_path, field, value):
    project, drive = _domain_fixture(tmp_path)
    processed = project / "data/processed/expansion_v1"
    path = processed / "domain_mix/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    sync_snapshot(processed, drive / "expansions/expansion_v1/processed")
    with pytest.raises(ValueError, match="certified non-replayed 80/20"):
        _domain(project, drive)
    assert not (project / "runs/domain_v1").exists()


def test_domain_rejects_financial_validation_with_different_exclusion_boundary(tmp_path):
    project, drive = _domain_fixture(tmp_path)
    processed = project / "data/processed/expansion_v1"
    path = processed / "finance_validation/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["eval_exclusion_sha256"] = "different-frozen-pool"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    sync_snapshot(processed, drive / "expansions/expansion_v1/processed")
    with pytest.raises(ValueError, match="certified non-replayed 80/20"):
        _domain(project, drive)


def test_domain_old_parent_runtime_allows_new_weights_transfer(tmp_path, monkeypatch):
    actual = torch.__version__
    with monkeypatch.context() as old:
        old.setattr(torch, "__version__", "2.11.0+cpu")
        project, drive = _domain_fixture(tmp_path)
    state = _domain(project, drive)
    assert state["parent"]["parent_provenance"]["torch_version"] == "2.11.0+cpu"
    assert state["expected_provenance"]["torch_version"] == str(actual)
    _complete(state)


def test_domain_own_runtime_drift_refuses_exact_resume(tmp_path, monkeypatch):
    project, drive = _domain_fixture(tmp_path)
    _complete(_domain(project, drive))
    monkeypatch.setattr(torch, "__version__", "different-domain-runtime")
    with pytest.raises(ValueError, match="frozen experiment|runtime, settings"):
        _domain(project, drive)


def test_domain_newer_live_restores_without_rolling_back(tmp_path):
    project, drive = _domain_fixture(tmp_path)
    state = _domain(project, drive)
    retention_domain_check(state, "before")
    run_domain_segment(state, "probe", _runner)
    sync_snapshot(state["config"]["output_dir"], state["paths"]["drive"] / "run")
    run_domain_segment(state, "continue", _runner)
    fresh = tmp_path / "restored-newer-live"
    fresh.mkdir()
    restored = _domain(fresh, drive)
    assert restored["restored_updates"] == 3
    checkpoint = load_checkpoint(Path(restored["config"]["output_dir"]) / "latest.pt")
    assert checkpoint["progress"]["updates"] == 3
    assert json.loads((Path(restored["config"]["output_dir"]) / "run_summary.json").read_text())["updates"] == 3


def test_domain_failed_train_blocks_retention_after_and_final_backup(tmp_path):
    project, drive = _domain_fixture(tmp_path)
    state = _domain(project, drive)
    retention_domain_check(state, "before")

    def failed(*args):
        raise RuntimeError("simulated training failure")

    with pytest.raises(RuntimeError, match="training failure"):
        run_domain_segment(state, "probe", failed)
    with pytest.raises(ValueError, match="training did not complete"):
        retention_domain_check(state, "after")
    with pytest.raises(ValueError, match="did not complete"):
        backup_domain_training(state)


def test_domain_requires_before_and_probe_before_long_training(tmp_path):
    project, drive = _domain_fixture(tmp_path)
    state = _domain(project, drive)
    with pytest.raises((ValueError, FileNotFoundError)):
        run_domain_segment(state, "probe", _runner)
    retention_domain_check(state, "before")
    with pytest.raises(ValueError, match="transfer probe"):
        run_domain_segment(state, "continue", _runner)
    assert not (project / "runs/domain_v1").exists()


def test_domain_retention_failure_blocks_backup_and_retry_reuses_original(tmp_path, monkeypatch):
    project, drive = _domain_fixture(tmp_path)
    state = _domain(project, drive)
    retention_domain_check(state, "before")
    run_domain_segment(state, "probe", _runner)
    run_domain_segment(state, "continue", _runner)
    original = runtime._validate
    monkeypatch.setattr(runtime, "_validate", lambda *args: {"loss": float("nan"), "supervised_tokens": 1})
    with pytest.raises(ValueError, match="invalid metrics"):
        retention_domain_check(state, "after")
    with pytest.raises(ValueError, match="did not complete"):
        backup_domain_training(state)
    monkeypatch.setattr(runtime, "_validate", original)
    retention_domain_check(state, "after")
    assert backup_domain_training(state)["verified"]


def test_domain_changed_transition_and_epoch_replay_are_rejected(tmp_path):
    project, drive = _domain_fixture(tmp_path)
    state = _domain(project, drive)
    retention_domain_check(state, "before")
    run_domain_segment(state, "probe", _runner)
    path = Path(state["config"]["output_dir"]) / "latest.pt"
    checkpoint = load_checkpoint(path)
    checkpoint["config"]["domain_transition"]["parent_tokens"] += 1
    save_checkpoint(path, checkpoint)
    with pytest.raises(ValueError, match="parent transition"):
        run_domain_segment(state, "continue", _runner)
    checkpoint["config"]["domain_transition"]["parent_tokens"] -= 1
    checkpoint["progress"]["epoch"] = 1
    save_checkpoint(path, checkpoint)
    save_checkpoint(Path(state["config"]["backup_dir"]) / "latest.pt", checkpoint)
    with pytest.raises(ValueError, match="replay or rollback"):
        run_domain_segment(state, "continue", _runner)


def test_domain_changed_state_transition_is_rejected_before_any_optimizer_work(tmp_path):
    project, drive = _domain_fixture(tmp_path)
    state = _domain(project, drive)
    retention_domain_check(state, "before")
    state["config"]["domain_transition"]["parent_tokens"] += 1
    with pytest.raises(ValueError, match="frozen parent identity"):
        run_domain_segment(state, "probe", lambda *args: pytest.fail("Do not train after a changed transition"))
    assert not (project / "runs/domain_v1").exists()


def test_domain_interrupted_archive_recovers_genuine_report_before_new_steps(tmp_path, monkeypatch):
    project, drive = _domain_fixture(tmp_path)
    state = _domain(project, drive)
    retention_domain_check(state, "before")
    original = continuation.sync_snapshot

    def fail(source, target, **kwargs):
        if "report_archive" in Path(target).parts:
            raise OSError("simulated archive interruption")
        return original(source, target, **kwargs)

    monkeypatch.setattr(continuation, "sync_snapshot", fail)
    with pytest.raises(OSError, match="archive interruption"):
        run_domain_segment(state, "probe", _runner)
    saved = (Path(state["config"]["output_dir"]) / "run_summary.json").read_bytes()
    assert list((state["paths"]["drive"] / "report_archive").glob("*/PENDING_REPORT.json"))
    monkeypatch.setattr(continuation, "sync_snapshot", original)
    restored = _domain(project, drive)
    retention_domain_check(restored, "before")
    assert run_domain_segment(restored, "probe", _runner)["new_updates"] == 0
    assert (Path(restored["config"]["output_dir"]) / "run_summary.json").read_bytes() == saved
    run_domain_segment(restored, "continue", _runner)
    retention_domain_check(restored, "after")
    assert backup_domain_training(restored)["verified"]


def test_domain_final_snapshot_interruption_recovers_new_vm_from_live(tmp_path, monkeypatch):
    project, drive = _domain_fixture(tmp_path)
    state = _domain(project, drive)
    retention_domain_check(state, "before")
    run_domain_segment(state, "probe", _runner)
    run_domain_segment(state, "continue", _runner)
    retention_domain_check(state, "after")
    measured = (Path(state["config"]["output_dir"]) / "run_summary.json").read_bytes()
    original = runtime.sync_snapshot
    destination = state["paths"]["drive"] / "run"

    def fail(source, target, **kwargs):
        if Path(target) == destination:
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "SNAPSHOT_INCOMPLETE.json").write_text("{}")
            raise OSError("simulated final backup interruption")
        return original(source, target, **kwargs)

    monkeypatch.setattr(runtime, "sync_snapshot", fail)
    with pytest.raises(OSError, match="backup interruption"):
        backup_domain_training(state)
    monkeypatch.setattr(runtime, "sync_snapshot", original)
    fresh = tmp_path / "fresh-after-backup-failure"
    fresh.mkdir()
    restored = _domain(fresh, drive)
    retention_domain_check(restored, "before")
    run_domain_segment(restored, "probe", _runner)
    run_domain_segment(restored, "continue", _runner)
    retention_domain_check(restored, "after")
    assert backup_domain_training(restored)["verified"]
    assert (Path(restored["config"]["output_dir"]) / "run_summary.json").read_bytes() == measured


@pytest.mark.parametrize("budgets", [(0, 1, 3), (4, 0, 3), (4, 3, 3), (4, True, 3)])
def test_domain_invalid_budgets_write_nothing(tmp_path, budgets):
    with pytest.raises(ValueError, match="Invalid financial-domain"):
        prepare_domain_training(tmp_path / "project", tmp_path / "drive",
                                min_parent_updates=budgets[0], probe_updates=budgets[1], target_updates=budgets[2])
    assert not list(tmp_path.iterdir())
