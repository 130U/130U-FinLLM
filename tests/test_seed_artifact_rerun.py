"""Exercise a complete real BPE/packing build and safe repeat execution."""
import importlib.util
import json
import os
import random
import shutil
from pathlib import Path

import pytest

from finllm.data import file_sha256, write_json, write_jsonl

PROJECT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("seed_rerun", PROJECT / "scripts/build_seed_artifacts.py")
seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed)
ARGS = dict(corpora="corpora", task_pool="tasks.jsonl", prepared="prepared",
            tokenizer="tokenizer", processed="processed", run_output="run")


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("complete-seed")
    (root / "configs").mkdir()
    shutil.copy2(PROJECT / "configs/pretrain_pilot.json", root / "configs/pretrain_pilot.json")
    rng = random.Random(130)
    source = {"status": "complete"}
    for domain in ("general", "finance"):
        path = root / "corpora" / f"{domain}_candidate.jsonl"
        rows = [{"text": " ".join("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=9))
                                  for _ in range(160)),
                 "source_url": f"https://example.org/{domain}/{i}", "source": "original test fixture",
                 "training_allowed": True, "rights_note": "Generated integration test material"}
                for i in range(160)]
        write_jsonl(path, rows)
        source[domain] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    write_json(root / "corpora/manifest.json", source)
    write_jsonl(root / "tasks.jsonl", [{"text": "ACME's ambitious growth forecast faces execution risks."}])
    previous = Path.cwd()
    try:
        os.chdir(root)
        seed.build(**ARGS)
    finally:
        os.chdir(previous)
    return root


@pytest.fixture
def working(completed, tmp_path, monkeypatch):
    shutil.copytree(completed, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def inventory(root):
    return {p.relative_to(root).as_posix(): (file_sha256(p), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_repeat_completed_build_is_read_only_and_keeps_training_progress(working, monkeypatch):
    write_json(working / "run/progress.json", {"updates": 12})
    before = inventory(working)
    monkeypatch.setattr(seed, "finllm", lambda *a: pytest.fail("Rerun must not train or pack"))
    report = seed.build(**ARGS)
    assert report == json.loads((working / "processed/seed_summary.json").read_text())
    assert inventory(working) == before
    assert seed.build(**ARGS, validate_only=True) == report


@pytest.mark.parametrize("target,action", [
    ("processed/seed_summary.json", "remove"),
    ("prepared/INCOMPLETE", "change"),
    ("prepared/general_train.jsonl", "change"),
    ("tasks.jsonl", "change"),
    ("tokenizer/tokenizer.json", "change"),
    ("processed/general/tokens-00000.bin", "change"),
    ("processed/finance_validation/manifest.json", "remove"),
    ("processed/pilot_config.json", "change"),
])
def test_partial_or_changed_seed_fails_without_rewriting(working, target, action):
    path = working / target
    if action == "remove":
        path.unlink()
    else:
        with path.open("ab") as stream:
            stream.write(b"changed")
    before = inventory(working)
    with pytest.raises(ValueError, match="Existing files were preserved"):
        seed.build(**ARGS)
    assert inventory(working) == before


def test_different_requested_paths_do_not_reuse_seed(working):
    with pytest.raises(ValueError, match="requested paths changed"):
        seed.build(**dict(ARGS, run_output="other-run"))
    assert not (working / "other-run").exists()


def test_validation_does_not_create_missing_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="incomplete or changed"):
        seed.build(**ARGS, validate_only=True)
    assert list(tmp_path.iterdir()) == []
