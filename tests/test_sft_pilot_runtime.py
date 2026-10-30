"""Exercise real tiny domain-to-entity-SFT transfer and conservative gates."""
import json
import random
import zipfile
from pathlib import Path

import pytest
import torch
import finllm.sft_pilot_runtime as runtime

from finllm.checkpoint import save_checkpoint, sha256_file
from finllm.evaluation import run_evaluation
from finllm.runtime import sync_snapshot
from finllm.sft_pilot_runtime import _budget_complete, install_task_zip, prepare_sft_pilot, run_sft_pilot, backup_sft_pilot
from finllm.training import _data_digest
from test_domain_runtime import _complete, _domain, _domain_fixture, _inventory
from test_expansion_runtime import _runner


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Synthetic domain fixture has no real prepared raw snapshots. Its
        # SFT lifecycle tests mock only the separately tested exclusion audit.
        monkeypatch.setattr(runtime, "_audit_training_exclusions", lambda *args: {
            name: {"sha256": "fixture-" + name, "task_rows": 50,
                   "distinct_task_texts": 50, "missing": 0}
            for name in ("seed_train", "seed_validation", "expansion_train", "expansion_validation")})
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


def _interrupt_after_live_update(monkeypatch, desired):
    import finllm.training as training
    original = training.snapshot_to_drive
    def interrupt(path, destination):
        result = original(path, destination)
        if Path(path).name == "latest.pt" and runtime.load_checkpoint(path)["progress"]["updates"] == desired:
            raise RuntimeError("simulated interruption immediately after live checkpoint")
        return result
    monkeypatch.setattr(training, "snapshot_to_drive", interrupt)
    return training, original


def test_interrupted_periodic_checkpoint_can_resume_from_notebook_probe(pilot, tmp_path, monkeypatch):
    """Run a real 10-step live save without a matching segment report."""
    _, drive, digest, source = pilot
    path = source / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rng = random.Random(991)
    for index in range(320):
        words = ["".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=8)) for _ in range(8)]
        rows.append({**rows[index % 32], "id": f"additional:{index}",
                     "text": "ACME " + " ".join(words), "group_id": f"additional:{index}",
                     "event_id": f"additional:{index}"})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    task_manifest = source / "task_manifest.json"
    manifest = json.loads(task_manifest.read_text(encoding="utf-8"))
    manifest["files"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    task_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    sync_snapshot(source, drive / "tasks/entity_v1")
    state = _prepare(pilot)
    assert state["config"]["train"]["max_updates"] == 22
    run_sft_pilot(state, "probe", _runner)
    training, original = _interrupt_after_live_update(monkeypatch, 10)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_sft_pilot(state, "continue", _runner)
    monkeypatch.setattr(training, "snapshot_to_drive", original)
    fresh = tmp_path / "fresh-kaggle"
    fresh.mkdir()
    resumed = prepare_sft_pilot(fresh, drive, expected_checkpoint_sha256=digest,
                                expected_domain_updates=3, expected_task_manifest_sha256=sha256_file(task_manifest))
    probe = run_sft_pilot(resumed, "probe", _runner)
    assert probe["start_updates"] == 10 and probe["end_updates"] == 10
    assert probe["summary"] is None and probe["report_status"] == "checkpoint_only_no_matching_measured_report"
    result = run_sft_pilot(resumed, "continue", _runner)
    assert result["start_updates"] == 10 and result["end_updates"] == 22
    assert backup_sft_pilot(resumed)["verified"] is True


def test_interrupted_final_checkpoint_recovers_report_without_invented_metrics(pilot, tmp_path, monkeypatch):
    _, drive, digest, source = pilot
    state = _prepare(pilot)
    run_sft_pilot(state, "probe", _runner)
    training, original = _interrupt_after_live_update(monkeypatch, 2)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_sft_pilot(state, "continue", _runner)
    monkeypatch.setattr(training, "snapshot_to_drive", original)
    fresh = tmp_path / "fresh-kaggle"
    fresh.mkdir()
    resumed = prepare_sft_pilot(fresh, drive, expected_checkpoint_sha256=digest,
                                expected_domain_updates=3,
                                expected_task_manifest_sha256=sha256_file(source / "task_manifest.json"))
    probe = run_sft_pilot(resumed, "probe", _runner)
    assert probe["start_updates"] == 2 and probe["summary"] is None
    result = run_sft_pilot(resumed, "continue", _runner)
    assert result["start_updates"] == result["end_updates"] == 2
    assert result["summary"]["resumed_no_training"] is True
    assert result["summary"]["tokens_per_second"] is None and result["summary"]["peak_vram_bytes"] is None
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


@pytest.fixture
def actual_exclusion_fixture(tmp_path):
    """Actual snapshot files, manifests and checkpoint markers, no mocks."""
    root, drive = tmp_path / "work", tmp_path / "drive"
    seed_processed = root / "seed_processed"
    expansion_processed = root / "expansion_processed"
    task = root / "task"
    task.mkdir(parents=True)
    texts = ["Acme reports growth", "Beta shares decline", "Gamma issues guidance", "Delta gets approval", "Epsilon expands"]
    for split, title in zip(runtime.SPLITS, texts):
        (task / f"{split}.jsonl").write_text(json.dumps({"text": title}) + "\n", encoding="utf-8")
    seed_raw_source = root / "seed_raw"
    expansion_raw_source = root / "expansion_prepared"
    (seed_raw_source / "prepared").mkdir(parents=True)
    expansion_raw_source.mkdir()
    raw = "".join(json.dumps({"text": title, "cloud_metadata": "different from local"}) + "\n" for title in texts)
    for source in (seed_raw_source / "prepared/frozen_exclusions.jsonl",
                   seed_raw_source / "prepared/frozen_task_exclusions.jsonl",
                   expansion_raw_source / "frozen_exclusions.jsonl",
                   expansion_raw_source / "frozen_prior_exclusions.jsonl"):
        source.write_text(raw, encoding="utf-8")
    specs = ((seed_processed, "general", seed_raw_source / "prepared/frozen_exclusions.jsonl"),
             (seed_processed, "general_validation", seed_raw_source / "prepared/frozen_task_exclusions.jsonl"),
             (expansion_processed, "general", expansion_raw_source / "frozen_exclusions.jsonl"),
             (expansion_processed, "general_validation", expansion_raw_source / "frozen_prior_exclusions.jsonl"))
    for base, name, exclusion in specs:
        folder = base / name
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps({"eval_exclusion_sha256": sha256_file(exclusion)}), encoding="utf-8")
    for name, source in (("domain_mix", "general"), ("finance_validation", "general_validation")):
        folder = expansion_processed / name
        folder.mkdir()
        (folder / "manifest.json").write_bytes((expansion_processed / source / "manifest.json").read_bytes())
    seed_source = root / "seed_parent"
    seed_source.mkdir()
    seed_path = seed_source / "latest.pt"
    save_checkpoint(seed_path, {"provenance": {"data_sha256": _data_digest(seed_processed / "general/manifest.json"),
                                               "validation_sha256": _data_digest(seed_processed / "general_validation/manifest.json")}})
    exp_source = root / "exp_parent"
    exp_source.mkdir()
    exp_path = exp_source / "latest.pt"
    save_checkpoint(exp_path, {"provenance": {"data_sha256": _data_digest(expansion_processed / "general/manifest.json"),
                                              "validation_sha256": _data_digest(expansion_processed / "general_validation/manifest.json"),
                                              "parent_lineage": {"kind": "data_expansion", "parent_checkpoint_sha256": sha256_file(seed_path)}}})
    sync_snapshot(seed_raw_source, drive / "seeds/seed_v1/raw")
    sync_snapshot(expansion_raw_source, drive / "expansions/expansion_v1/prepared")
    sync_snapshot(seed_source, drive / "expansions/expansion_v1/parent")
    sync_snapshot(exp_source, drive / "domains/domain_v1/parent")
    paths = {"seed_processed": seed_processed, "processed": expansion_processed,
             "seed_drive": drive / "seeds/seed_v1", "drive": drive / "expansions/expansion_v1"}
    return paths, task, drive / "domains/domain_v1", texts


def test_cloud_exclusions_can_differ_from_local_bytes_but_must_cover_all_titles(actual_exclusion_fixture):
    paths, task, domain, texts = actual_exclusion_fixture
    result = runtime._audit_training_exclusions(paths, task, domain)
    assert set(result) == {"seed_train", "seed_validation", "expansion_train", "expansion_validation"}
    assert all(item["task_rows"] == 5 and item["missing"] == 0 for item in result.values())
    # A local pool with identical texts but different metadata has a different
    # byte hash; actual cloud pools remain tied to their own checkpoints.
    local = task / "local-exclusion.jsonl"
    local.write_text("".join(json.dumps({"text": t, "local_metadata": 1}) + "\n" for t in texts), encoding="utf-8")
    assert sha256_file(local) != result["seed_train"]["sha256"]


@pytest.mark.parametrize("missing", [False, True])
def test_prepare_sft_pilot_reads_real_cloud_exclusions_and_checkpoint_chain(tmp_path, monkeypatch, missing):
    """Exercise the actual preparation entry, including all four snapshot audits.

    Pretraining fixtures use synthetic data but real saved optimizer checkpoints,
    actual snapshot markers and the original 03a handoff implementation.
    """
    import test_domain_runtime as domain_tests
    import test_expansion_runtime as expansion_tests

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rng = random.Random(130)
    by_split = {}
    for split, count in (("train", 32), ("development", 6), ("calibration_temperature", 4),
                         ("calibration_threshold", 4), ("test", 4)):
        rows = []
        for index in range(count):
            words = ["".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=8)) for _ in range(8)]
            rows.append({"id": f"{split}:{index}", "text": "ACME " + " ".join(words), "target": "ACME",
                         "label": ["bullish", "neutral", "bearish"][index % 3],
                         "target_scope": "literal_entity_not_verified_security", "human_reviewed": True,
                         "asset_verified": False, "training_allowed": True, "source": "fixture",
                         "rights_note": "synthetic test fixture", "group_id": f"{split}:{index}",
                         "event_id": f"{split}:{index}",
                         "split": "calibration" if split.startswith("calibration_") else split})
        by_split[split] = rows
    exclusion_text = "".join(json.dumps({"text": row["text"], "metadata": "cloud"}) + "\n"
                             for rows in by_split.values() for row in rows)
    if missing:
        by_split["test"][0]["text"] = "ACME never excluded unmistakably unique text"
    pool_source = tmp_path / "cloud_pool"
    (pool_source / "prepared").mkdir(parents=True)
    for name in ("frozen_exclusions.jsonl", "frozen_task_exclusions.jsonl"):
        (pool_source / "prepared" / name).write_text(exclusion_text, encoding="utf-8")
    pool_hash = sha256_file(pool_source / "prepared/frozen_exclusions.jsonl")
    expansion_source = tmp_path / "cloud_expansion_pool"
    expansion_source.mkdir()
    for name in ("frozen_exclusions.jsonl", "frozen_prior_exclusions.jsonl"):
        (expansion_source / name).write_text(exclusion_text, encoding="utf-8")

    real_seed_train = expansion_tests.run_training
    real_expansion_fixture = domain_tests._expansion_fixture

    def seed_train_with_pool(config):
        if str(config["output_dir"]).endswith("seed_v1_pretrain"):
            for name in ("general", "general_validation", "finance_validation"):
                path = tmp_path / "original-runtime/data/processed/seed_v1" / name / "manifest.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                manifest["eval_exclusion_sha256"] = pool_hash
                path.write_text(json.dumps(manifest), encoding="utf-8")
            sync_snapshot(pool_source, tmp_path / "drive/seeds/seed_v1/raw")
        return real_seed_train(config)

    def expansion_fixture_with_pool(path):
        project, drive = real_expansion_fixture(path)
        processed = project / "data/processed/expansion_v1"
        for name in ("general", "general_validation"):
            manifest_path = processed / name / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["eval_exclusion_sha256"] = pool_hash
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        sync_snapshot(processed, drive / "expansions/expansion_v1/processed")
        sync_snapshot(expansion_source, drive / "expansions/expansion_v1/prepared")
        return project, drive

    monkeypatch.setattr(expansion_tests, "run_training", seed_train_with_pool)
    monkeypatch.setattr(domain_tests, "_expansion_fixture", expansion_fixture_with_pool)
    try:
        project, drive = _domain_fixture(tmp_path)
        _complete(_domain(project, drive))
        source = tmp_path / "task-source-real"
        source.mkdir()
        for split, rows in by_split.items():
            (source / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest = {"format": "130u-entity-pilot-v1", "seed_exclusions_sha256": "a" * 64,
                    "expansion_exclusions_sha256": "b" * 64,
                    "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                              for p in source.glob("*.jsonl")}}
        (source / "task_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        sync_snapshot(source, drive / "tasks/entity_v1")
        args = dict(expected_checkpoint_sha256=sha256_file(drive / "domains/domain_v1/run/latest.pt"),
                    expected_domain_updates=3, expected_task_manifest_sha256=sha256_file(source / "task_manifest.json"))
        if missing:
            with pytest.raises(ValueError, match="does not contain"):
                prepare_sft_pilot(project, drive, **args)
            assert not (drive / "live_checkpoints/sft_entity_v1").exists()
        else:
            state = prepare_sft_pilot(project, drive, **args)
            result = state["parent_audit"]["actual_pretraining_exclusions"]
            assert all(item["task_rows"] == 50 and item["missing"] == 0 for item in result.values())
            assert {item["sha256"] for item in result.values()} == {pool_hash}
            assert run_sft_pilot(state, "probe", _runner)["end_updates"] == 1
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("case", ["missing_title", "bad_snapshot", "wrong_manifest", "wrong_lineage", "wrong_pool_hash", "wrong_valid_lineage", "wrong_domain_pool"])
def test_unproven_cloud_exclusions_never_allow_training(actual_exclusion_fixture, case):
    paths, task, domain, _ = actual_exclusion_fixture
    if case == "missing_title":
        (task / "test.jsonl").write_text(json.dumps({"text": "Never excluded"}) + "\n", encoding="utf-8")
    elif case == "bad_snapshot":
        (paths["seed_drive"] / "raw/prepared/frozen_exclusions.jsonl").write_text("changed", encoding="utf-8")
    elif case == "wrong_manifest":
        manifest = paths["seed_processed"] / "general/manifest.json"
        manifest.write_text(json.dumps({"eval_exclusion_sha256": "0" * 64}), encoding="utf-8")
    elif case == "wrong_lineage":
        path = paths["drive"] / "parent/latest.pt"
        # Deliberately invalidate the independently snapshotted parent.
        path.write_bytes(b"untrusted different seed checkpoint")
    elif case == "wrong_pool_hash":
        folder = paths["seed_drive"] / "raw"
        path = folder / "prepared/frozen_exclusions.jsonl"
        path.write_bytes(path.read_bytes() + b" \n")
        marker = folder / "SNAPSHOT_COMPLETE.json"
        data = json.loads(marker.read_text(encoding="utf-8"))
        entry = next(item for item in data["files"] if item["path"] == "prepared/frozen_exclusions.jsonl")
        entry.update(bytes=path.stat().st_size, sha256=sha256_file(path))
        marker.write_text(json.dumps(data), encoding="utf-8")
    elif case == "wrong_valid_lineage":
        # A complete, self-consistent checkpoint snapshot still must reference
        # this exact seed parent; a new valid marker is not lineage proof.
        folder = domain / "parent"
        path = folder / "latest.pt"
        payload = runtime.load_checkpoint(path, allow_previous=False)
        payload["provenance"]["parent_lineage"]["parent_checkpoint_sha256"] = "0" * 64
        save_checkpoint(path, payload)
        marker = folder / "SNAPSHOT_COMPLETE.json"
        data = json.loads(marker.read_text(encoding="utf-8"))
        for item in data["files"]:
            target = folder / item["path"]
            item.update(bytes=target.stat().st_size, sha256=sha256_file(target))
        marker.write_text(json.dumps(data), encoding="utf-8")
    else:
        path = paths["processed"] / "domain_mix/manifest.json"
        path.write_text(json.dumps({"eval_exclusion_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises((ValueError, OSError)):
        runtime._audit_training_exclusions(paths, task, domain)


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
