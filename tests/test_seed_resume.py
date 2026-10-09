from pathlib import Path

import pytest
import torch

from finllm.checkpoint import load_checkpoint, marker_path, save_checkpoint
from finllm.runtime import restore_seed_checkpoint


def checkpoint(directory, attempted=4, tokens=128, updates=None, name="latest.pt", **changes):
    payload = {
        "model_config": {"hidden_size": 4, "num_hidden_layers": 1},
        "tokenizer_hash": "frozen-seed-tokenizer",
        "provenance": {"data_sha256": "frozen-seed-data", "stage": "pretrain", "seed": 130},
        "progress": {"attempted_updates": attempted, "tokens": tokens,
                     "updates": attempted if updates is None else updates},
        "model_state": {"weight": torch.tensor([float(attempted)])},
    }
    payload.update(changes)
    path = directory / name
    save_checkpoint(path, payload)
    return path


def test_new_run_only_when_no_checkpoint_artifacts_exist(tmp_path):
    local, live = tmp_path / "local", tmp_path / "live"
    assert restore_seed_checkpoint(local, live) is None
    assert not local.exists() and not live.exists()
    local.mkdir()
    (local / "metrics.jsonl").write_text("", encoding="utf-8")
    assert restore_seed_checkpoint(local, live) is None
    checkpoint(live, name="best.pt")
    with pytest.raises(ValueError, match="refusing fresh"):
        restore_seed_checkpoint(local, live)


def test_new_vm_restores_latest_and_best_without_touching_live(tmp_path):
    local, live = tmp_path / "local", tmp_path / "live"
    latest = checkpoint(live)
    checkpoint(live, attempted=3, tokens=96, name="best.pt")
    saved_bytes = latest.read_bytes()
    selected = restore_seed_checkpoint(local, live)
    assert selected == str((local / "latest.pt").resolve())
    assert load_checkpoint(selected)["progress"]["attempted_updates"] == 4
    assert load_checkpoint(local / "best.pt")["progress"]["attempted_updates"] == 3
    assert latest.read_bytes() == saved_bytes
    assert not (local / "SNAPSHOT_COMPLETE.json").exists()


def test_newer_live_wins_and_newer_local_is_never_rolled_back(tmp_path):
    local, live = tmp_path / "local", tmp_path / "live"
    checkpoint(local, attempted=2, tokens=64)
    checkpoint(live, attempted=4, tokens=128)
    assert load_checkpoint(restore_seed_checkpoint(local, live))["progress"]["attempted_updates"] == 4
    checkpoint(local, attempted=5, tokens=160)
    assert load_checkpoint(restore_seed_checkpoint(local, live))["progress"]["attempted_updates"] == 5
    assert load_checkpoint(live / "latest.pt")["progress"]["attempted_updates"] == 4


@pytest.mark.parametrize("change", [
    {"tokenizer_hash": "different-tokenizer"},
    {"model_config": {"hidden_size": 8}},
    {"provenance": {"data_sha256": "different-data"}},
])
def test_divergent_checkpoints_are_rejected_without_overwrite(tmp_path, change):
    local, live = tmp_path / "local", tmp_path / "live"
    original = checkpoint(local)
    checkpoint(live, attempted=5, tokens=160, **change)
    before = original.read_bytes()
    with pytest.raises(ValueError, match="provenance diverges"):
        restore_seed_checkpoint(local, live)
    assert original.read_bytes() == before


@pytest.mark.parametrize("attempted,tokens,updates", [(5, 100, 5), (4, 160, 4), (5, 160, 3)])
def test_incomparable_progress_cannot_be_automatically_selected(tmp_path, attempted, tokens, updates):
    local, live = tmp_path / "local", tmp_path / "live"
    checkpoint(local)
    checkpoint(live, attempted=attempted, tokens=tokens, updates=updates)
    with pytest.raises(ValueError, match="progress"):
        restore_seed_checkpoint(local, live)


def test_corrupt_pairs_refuse_fresh_and_valid_other_side_is_recovered(tmp_path):
    local, live = tmp_path / "local", tmp_path / "live"
    source = checkpoint(live)
    source.write_bytes(b"broken upload")
    with pytest.raises(ValueError, match="refusing fresh"):
        restore_seed_checkpoint(local, live)
    checkpoint(local, attempted=3, tokens=96)
    assert load_checkpoint(restore_seed_checkpoint(local, live))["progress"]["attempted_updates"] == 3
    (local / "latest.pt").write_bytes(b"broken local data")
    checkpoint(live, attempted=6, tokens=192)
    assert load_checkpoint(restore_seed_checkpoint(local, live))["progress"]["attempted_updates"] == 6


def test_interrupted_save_preserves_and_recovers_previous_pair(tmp_path, monkeypatch):
    import finllm.checkpoint as module

    local, live = tmp_path / "local", tmp_path / "live"
    checkpoint(live)
    original_replace = module.os.replace

    def fail_marker(source, target):
        if Path(target) == marker_path(live / "latest.pt"):
            raise OSError("simulated interrupted Drive marker")
        return original_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_marker)
    with pytest.raises(OSError, match="interrupted"):
        checkpoint(live, attempted=5, tokens=160)
    monkeypatch.setattr(module.os, "replace", original_replace)
    with pytest.warns(RuntimeWarning, match="Recovered previous"):
        selected = restore_seed_checkpoint(local, live)
    assert (local / "latest.pt.previous").exists()
    assert (local / "latest.pt.previous.json").exists()
    with pytest.warns(RuntimeWarning, match="Recovered previous"):
        assert load_checkpoint(selected)["progress"]["attempted_updates"] == 4


def test_local_checkpoint_alone_is_reused_and_nested_paths_are_refused(tmp_path):
    local, live = tmp_path / "local", tmp_path / "live"
    checkpoint(local)
    assert restore_seed_checkpoint(local, live) == str((local / "latest.pt").resolve())
    assert not live.exists()
    for nested in (local, local / "backup", tmp_path):
        with pytest.raises(ValueError, match="non-nested"):
            restore_seed_checkpoint(local, nested)
