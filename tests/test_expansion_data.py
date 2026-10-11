"""Real BPE/packing expansion, leakage exclusion, and repeat/interrupt safety."""
import importlib.util
import os
import random
import shutil
from pathlib import Path

import pytest

from finllm.data import TokenBlockDataset, file_sha256, read_jsonl, simhash, write_json, write_jsonl
from finllm.expansion import build_expansion_artifacts
from finllm.preparation import _digest
from finllm.runtime import sync_snapshot


PROJECT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("seed_build_for_expansion", PROJECT / "scripts/build_seed_artifacts.py")
seed_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed_build)


def _document(rng, domain, index, words=120):
    return {"text": " ".join("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=9)) for _ in range(words)),
            "source_url": f"https://example.org/{domain}/{index}", "source": "Original integration fixture",
            "training_allowed": True, "rights_note": "Original fixture material; not a downloadable training corpus"}


def _manifest(root):
    source = {"status": "complete", "format_version": 1}
    for domain in ("general", "finance"):
        path = root / f"{domain}_candidate.jsonl"
        source[domain] = {"file": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path),
                          "documents": sum(1 for _ in read_jsonl(path))}
    write_json(root / "manifest.json", source)


def _inventory(root):
    return {path.relative_to(root).as_posix(): (file_sha256(path), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def build_seed_expansion_fixture(root):
    """Reusable real-8K seed and new-source fixture for notebook cell tests."""
    root = Path(root)
    (root / "configs").mkdir()
    shutil.copy2(PROJECT / "configs/pretrain_pilot.json", root / "configs/pretrain_pilot.json")
    rng = random.Random(191)
    for domain in ("general", "finance"):
        write_jsonl(root / "seed_corpora" / f"{domain}_candidate.jsonl",
                    [_document(rng, "old-" + domain, index, 160) for index in range(160)])
    _manifest(root / "seed_corpora")
    task = {"text": "ACME profits narrowed sharply despite revenue growth in overseas branches."}
    write_jsonl(root / "task_pool.jsonl", [task])
    previous = Path.cwd()
    try:
        os.chdir(root)
        seed_build.build(corpora="seed_corpora", task_pool="task_pool.jsonl", prepared="seed_prepared",
                         tokenizer="tokenizer", processed="seed_processed", run_output="seed_run")
        for domain in ("general", "finance"):
            rows = [_document(rng, "additional-" + domain, index) for index in range(120)]
            # Every old source partition is protected, including the seed train
            # corpora: new token counts must describe additional documents.
            for old_domain in ("general", "finance"):
                for partition in ("train", "validation"):
                    old = next(read_jsonl(root / "seed_prepared" / f"{old_domain}_{partition}.jsonl"))
                    rows.append(dict(old, text=old["text"].upper(), source_url=f"https://example.org/alias/{domain}/{old_domain}/{partition}"))
            old = next(read_jsonl(root / "seed_prepared/general_train.jsonl"))
            rows.append(dict(_document(rng, "extra-" + domain, "old-document-new-chunk"), source_url=old["source_url"]))
            rows.append(dict(_document(rng, "extra-" + domain, "task-copy"), text=task["text"]))
            write_jsonl(root / "corpora" / f"{domain}_candidate.jsonl", rows)
        _manifest(root / "corpora")
    finally:
        os.chdir(previous)
    return root


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = build_seed_expansion_fixture(tmp_path_factory.mktemp("expansion-integration"))
    previous = Path.cwd()
    try:
        os.chdir(root)
        build_expansion_artifacts(**ARGS)
    finally:
        os.chdir(previous)
    return root


@pytest.fixture
def working(completed, tmp_path, monkeypatch):
    shutil.copytree(completed, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


ARGS = dict(corpora="corpora", task_pool="seed_prepared/frozen_task_exclusions.jsonl", seed_prepared="seed_prepared",
            seed_processed="seed_processed", tokenizer="tokenizer", prepared="prepared", processed="processed", shard_tokens=1000)


def test_real_expansion_reuses_tokenizer_and_excludes_all_old_partitions(working):
    before = _inventory(working)
    report = build_expansion_artifacts(**ARGS)
    assert _inventory(working) == before
    assert report["tokenizer_sha256"] == file_sha256(working / "tokenizer/tokenizer.json")
    assert report["tokenizer_vocabulary"] == 8192
    assert report["prior_exclusions"]["seed_documents"] == 320
    for domain in ("general", "finance"):
        assert report["source_statistics"][domain]["prior_group_excluded"] == 1
        assert report["source_statistics"][domain]["prior_text_excluded"] == 5
        for partition in ("train", "validation"):
            rows = list(read_jsonl(working / "prepared" / f"{domain}_{partition}.jsonl"))
            assert rows and all("additional-" in row["source_url"] for row in rows)
    assert report["retention_validation"]["general"]["sha256"] == file_sha256(working / "seed_processed/general_validation/manifest.json")
    assert report["domain_mix"]["finance_tokens"] == 4 * report["domain_mix"]["general_tokens"]
    assert report["domain_mix"]["repeated_source_tokens"] == 0
    for name, count in report["token_counts"].items():
        data = TokenBlockDataset(working / "processed" / name / "manifest.json", 32)
        assert data.total == count and data[0]["input_ids"].numel() == 32


@pytest.mark.parametrize("target", ["prepared/INCOMPLETE", "prepared/general_train.jsonl", "processed/expansion_summary.json",
                                   "processed/general/tokens-00000.bin", "processed/domain_mix/manifest.json",
                                   "tokenizer/tokenizer.json", "seed_prepared/general_validation.jsonl",
                                   "corpora/general_candidate.jsonl"])
def test_changed_or_partial_artifacts_preserved(working, target):
    path = working / target
    with path.open("ab") as stream:
        stream.write(b"interrupted")
    before = _inventory(working)
    with pytest.raises(ValueError, match="Existing files were preserved"):
        build_expansion_artifacts(**ARGS)
    assert _inventory(working) == before


def test_verified_snapshot_restore_is_reused_without_new_packing(working, monkeypatch):
    import finllm.expansion as expansion

    for name in ("prepared", "processed"):
        sync_snapshot(working / name, working / "backup" / name)
        shutil.rmtree(working / name)
        sync_snapshot(working / "backup" / name, working / name, working_copy=True)
    before = _inventory(working)
    monkeypatch.setattr(expansion, "_pack", lambda *args: pytest.fail("Verified reuse must not pack"))
    assert build_expansion_artifacts(**ARGS)["status"] == "expansion_inputs_prepared_training_not_started"
    assert _inventory(working) == before


def _absolute_args(root, prepared="prepared", processed="processed"):
    arguments = dict(ARGS, prepared=prepared, processed=processed)
    for name in ("corpora", "task_pool", "seed_prepared", "seed_processed", "tokenizer", "prepared", "processed"):
        arguments[name] = root / arguments[name]
    return arguments


def test_completed_backup_relocates_without_rewriting_historical_report(working, monkeypatch):
    import finllm.expansion as expansion

    relocated = working.parent / "relocated_complete"
    shutil.copytree(working, relocated)
    before = _inventory(relocated)
    report_bytes = (relocated / "processed/expansion_summary.json").read_bytes()
    monkeypatch.setattr(expansion, "_prepare", lambda *args: pytest.fail("Portable complete restore must not prepare"))
    monkeypatch.setattr(expansion, "_pack", lambda *args: pytest.fail("Portable complete restore must not pack"))
    report = build_expansion_artifacts(**_absolute_args(relocated))
    assert report["prepared"] == "prepared"  # Historical display path stays intact.
    assert (relocated / "processed/expansion_summary.json").read_bytes() == report_bytes
    assert _inventory(relocated) == before


def _interrupt_pack(expansion, monkeypatch, name="general", processed="new_processed"):
    original = expansion._atomic_json

    def interrupted(path, value):
        if path == Path(processed) / name / "manifest.json":
            raise OSError("simulated interrupted packing")
        return original(path, value)

    monkeypatch.setattr(expansion, "_atomic_json", interrupted)
    return original


def test_interrupted_pack_recovers_only_unfinished_phase_and_preserves_partials(working, monkeypatch):
    import finllm.expansion as expansion

    arguments = dict(ARGS, prepared="new_prepared", processed="new_processed")
    seed_before = {name: _inventory(working / name) for name in ("seed_prepared", "seed_processed", "tokenizer")}
    original = _interrupt_pack(expansion, monkeypatch)
    with pytest.raises(OSError, match="interrupted packing"):
        build_expansion_artifacts(**arguments)
    partial = _inventory(working / "new_processed/general")
    monkeypatch.setattr(expansion, "_atomic_json", original)
    monkeypatch.setattr(expansion, "_prepare", lambda *args: pytest.fail("Verified prepared files must be reused"))
    assert build_expansion_artifacts(**arguments)["status"] == "expansion_inputs_prepared_training_not_started"
    assert _inventory(working / "new_processed/recovery_archive/001_general") == partial
    assert seed_before == {name: _inventory(working / name) for name in seed_before}


def test_interrupted_preparation_recovers_under_same_input_contract(working, monkeypatch):
    import finllm.expansion as expansion

    original = expansion._rows

    def interrupted(path):
        for index, row in enumerate(original(path)):
            if Path(path) == Path("corpora/general_candidate.jsonl") and index == 3:
                raise OSError("simulated preparation interruption")
            yield row

    monkeypatch.setattr(expansion, "_rows", interrupted)
    arguments = dict(ARGS, prepared="new_prepared", processed="new_processed")
    with pytest.raises(OSError, match="preparation interruption"):
        build_expansion_artifacts(**arguments)
    partial = _inventory(working / "new_prepared")
    monkeypatch.setattr(expansion, "_rows", original)
    assert build_expansion_artifacts(**arguments)["tokenizer_vocabulary"] == 8192
    assert _inventory(working / "new_processed/recovery_archive/001_prepare") == partial


def test_incomplete_pack_snapshot_restores_and_recovers_on_new_vm(working, monkeypatch):
    import finllm.expansion as expansion

    original = _interrupt_pack(expansion, monkeypatch, "general_validation")
    arguments = dict(ARGS, prepared="new_prepared", processed="new_processed")
    with pytest.raises(OSError, match="interrupted packing"):
        build_expansion_artifacts(**arguments)
    completed_general = _inventory(working / "new_processed/general")
    for name in ("new_prepared", "new_processed"):
        sync_snapshot(working / name, working / "partial_backup" / name)
        shutil.rmtree(working / name)
        sync_snapshot(working / "partial_backup" / name, working / name, working_copy=True)
    monkeypatch.setattr(expansion, "_atomic_json", original)
    original_pack = expansion._pack

    def checked_pack(input_path, tokenizer, output, *args):
        assert output.name != "general", "Completed pack must be reused after VM restore"
        return original_pack(input_path, tokenizer, output, *args)

    monkeypatch.setattr(expansion, "_pack", checked_pack)
    assert build_expansion_artifacts(**arguments)["status"] == "expansion_inputs_prepared_training_not_started"
    assert _inventory(working / "new_processed/general") == completed_general


def test_incomplete_snapshot_recovers_at_different_absolute_project_path(working, monkeypatch):
    import finllm.expansion as expansion

    original = _interrupt_pack(expansion, monkeypatch, "general_validation")
    arguments = dict(ARGS, prepared="new_prepared", processed="new_processed")
    with pytest.raises(OSError, match="interrupted packing"):
        build_expansion_artifacts(**arguments)
    relocated = working.parent / "relocated_pending"
    for name in ("corpora", "seed_prepared", "seed_processed", "tokenizer", "new_prepared", "new_processed"):
        sync_snapshot(working / name, relocated / name, working_copy=True)
    general_before = _inventory(relocated / "new_processed/general")
    prepared_before = _inventory(relocated / "new_prepared")
    monkeypatch.setattr(expansion, "_atomic_json", original)
    report = build_expansion_artifacts(**_absolute_args(relocated, "new_prepared", "new_processed"))
    assert report["status"] == "expansion_inputs_prepared_training_not_started"
    assert _inventory(relocated / "new_processed/general") == general_before
    assert _inventory(relocated / "new_prepared") == prepared_before


def test_final_journal_write_failure_is_reused_read_only(working, monkeypatch):
    import finllm.expansion as expansion

    original = expansion._journal_write

    def interrupted(processed, journal):
        if journal["status"] == "complete":
            raise OSError("interrupted final journal commit")
        return original(processed, journal)

    arguments = dict(ARGS, prepared="new_prepared", processed="new_processed")
    monkeypatch.setattr(expansion, "_journal_write", interrupted)
    with pytest.raises(OSError, match="final journal commit"):
        build_expansion_artifacts(**arguments)
    assert not (working / "new_processed/INCOMPLETE").exists()
    assert expansion._read(working / "new_processed/build_journal.json")["status"] == "in_progress"
    before = _inventory(working)
    monkeypatch.setattr(expansion, "_journal_write", original)
    monkeypatch.setattr(expansion, "_prepare", lambda *args: pytest.fail("Complete artifacts must not prepare"))
    monkeypatch.setattr(expansion, "_pack", lambda *args: pytest.fail("Complete artifacts must not pack"))
    assert build_expansion_artifacts(**arguments)["status"] == "expansion_inputs_prepared_training_not_started"
    assert _inventory(working) == before


def test_interruption_after_summary_before_completion_marker_removal(working, monkeypatch):
    import finllm.expansion as expansion

    original = Path.unlink

    def interrupted(path, *args, **kwargs):
        if path == Path("new_processed/INCOMPLETE"):
            raise OSError("interrupted final readiness marker removal")
        return original(path, *args, **kwargs)

    arguments = dict(ARGS, prepared="new_prepared", processed="new_processed")
    monkeypatch.setattr(Path, "unlink", interrupted)
    with pytest.raises(OSError, match="readiness marker removal"):
        build_expansion_artifacts(**arguments)
    assert (working / "new_processed/expansion_summary.json").is_file()
    stable = {name: _inventory(working / name) for name in ("new_prepared", *("new_processed/" + name for name in expansion.PACK_NAMES))}
    report = (working / "new_processed/expansion_summary.json").read_bytes()
    monkeypatch.setattr(Path, "unlink", original)
    monkeypatch.setattr(expansion, "_prepare", lambda *args: pytest.fail("Complete prepared files must not rebuild"))
    monkeypatch.setattr(expansion, "_pack", lambda *args: pytest.fail("Complete packs must not rebuild"))
    assert build_expansion_artifacts(**arguments)["status"] == "expansion_inputs_prepared_training_not_started"
    assert (working / "new_processed/expansion_summary.json").read_bytes() == report
    assert stable == {name: _inventory(working / name) for name in stable}
    assert not (working / "new_processed/recovery_archive").exists()


@pytest.mark.parametrize("change", ["completed_pack", "unknown_partial_file", "settings"])
def test_recovery_refuses_changed_complete_output_or_unowned_files_without_writes(working, monkeypatch, change):
    import finllm.expansion as expansion

    original = _interrupt_pack(expansion, monkeypatch, "general_validation")
    arguments = dict(ARGS, prepared="new_prepared", processed="new_processed")
    with pytest.raises(OSError, match="interrupted packing"):
        build_expansion_artifacts(**arguments)
    monkeypatch.setattr(expansion, "_atomic_json", original)
    if change == "completed_pack":
        with (working / "new_processed/general/tokens-00000.bin").open("ab") as stream:
            stream.write(b"broken")
    elif change == "unknown_partial_file":
        (working / "new_processed/general_validation/user_notes.txt").write_text("Keep my notes")
    else:
        arguments["shard_tokens"] = 2000
    before = _inventory(working)
    with pytest.raises(ValueError, match="Existing files were preserved"):
        build_expansion_artifacts(**arguments)
    assert _inventory(working) == before


def test_task_pool_change_rejected_before_creating_outputs(working):
    write_jsonl(working / "other_tasks.jsonl", [{"text": "An unreviewed replacement evaluation pool"}])
    arguments = dict(ARGS, prepared="other_prepared", processed="other_processed", task_pool="other_tasks.jsonl")
    with pytest.raises(ValueError, match="original frozen seed task pool"):
        build_expansion_artifacts(**arguments)
    assert not (working / "other_prepared").exists() and not (working / "other_processed").exists()


def test_disk_index_detects_nonexact_lexical_near_duplicate(working):
    from finllm.expansion import _DiskExclusions
    import sqlite3

    # Verify a genuine nonexact lexical near duplicate in the disk index;
    # candidate URLs themselves differ so source identity cannot mask this test.
    row = next(read_jsonl(working / "seed_prepared/general_validation.jsonl"))
    text = row["text"]
    for addition in (".", ",", " !", " final"):
        changed = text + addition
        if (simhash(text) ^ simhash(changed)).bit_count() <= 3:
            break
    else:
        pytest.fail("Fixture did not produce a near duplicate")
    with sqlite3.connect(":memory:") as connection:
        index = _DiskExclusions(connection)
        index.add(text)
        assert index.matches(changed)


def test_new_validation_near_duplicate_is_filtered_in_real_build(working):
    rng = random.Random(390)
    row = _document(rng, "new-holdout-overlap", 1, 160)
    text = row["text"]
    for addition in (".", ",", " !", " final"):
        changed = text + addition
        if (simhash(text) ^ simhash(changed)).bit_count() <= 3:
            break
    else:
        pytest.fail("Fixture did not produce a near duplicate")
    urls = {"train": None, "validation": None}
    threshold = int(.05 * (1 << 256))
    for index in range(1000):
        url = f"https://example.org/holdout-overlap/{index}"
        partition = "validation" if int(_digest(130, "url:" + url), 16) < threshold else "train"
        urls[partition] = url
        if all(urls.values()):
            break
    rows = list(read_jsonl(working / "corpora/general_candidate.jsonl"))
    rows += [dict(row, text=text, source_url=urls["train"]),
             dict(row, text=changed, source_url=urls["validation"])]
    write_jsonl(working / "corpora/general_candidate.jsonl", rows)
    _manifest(working / "corpora")
    report = build_expansion_artifacts(**dict(ARGS, prepared="overlap_prepared", processed="overlap_processed"))
    assert report["source_statistics"]["general"]["validation_overlap_removed"] >= 1
    assert text not in {row["text"] for row in read_jsonl(working / "overlap_prepared/general_train.jsonl")}
    assert changed in {row["text"] for row in read_jsonl(working / "overlap_prepared/general_validation.jsonl")}


def test_oversized_document_fails_and_preserves_seed(working):
    from finllm.preparation import MAX_DOCUMENT_BYTES

    rows = list(read_jsonl(working / "corpora/general_candidate.jsonl"))
    rows.append(dict(rows[0], text="x" * (MAX_DOCUMENT_BYTES + 1), source_url="https://example.org/oversized"))
    write_jsonl(working / "corpora/general_candidate.jsonl", rows)
    _manifest(working / "corpora")
    seed_before = _inventory(working / "tokenizer")
    with pytest.raises(ValueError, match="document exceeds"):
        build_expansion_artifacts(**dict(ARGS, prepared="oversized_prepared", processed="oversized_processed"))
    assert (working / "oversized_prepared/INCOMPLETE").exists()
    assert (working / "oversized_processed/INCOMPLETE").exists()
    assert _inventory(working / "tokenizer") == seed_before
