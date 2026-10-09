import json
from pathlib import Path

import pytest
import torch

from finllm.checkpoint import capture_rng_state, load_checkpoint, marker_path, restore_rng_state, save_checkpoint, snapshot_to_drive


def test_complete_checkpoint_roundtrip_and_hash_rejection(tmp_path):
    path = tmp_path / "latest.pt"
    tensor = torch.arange(5, dtype=torch.float32)
    save_checkpoint(path, {"model_state": {"weight": tensor}, "progress": {"updates": 3}})
    loaded = load_checkpoint(path)
    assert loaded["complete"] is True
    assert torch.equal(loaded["model_state"]["weight"], tensor)
    marker = json.loads(marker_path(path).read_text())
    assert marker["updates"] == 3
    with path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        load_checkpoint(path)


def test_missing_completion_marker_is_rejected(tmp_path):
    path = tmp_path / "latest.pt"
    save_checkpoint(path, {"value": 1})
    marker_path(path).unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        load_checkpoint(path)


def test_interrupted_pair_replacement_recovers_previous(tmp_path, monkeypatch):
    import finllm.checkpoint as module
    path = tmp_path / "latest.pt"
    save_checkpoint(path, {"value": 1})
    original_replace = module.os.replace

    def fail_marker(source, target):
        if Path(target) == marker_path(path):
            raise OSError("simulated disconnect before completion marker")
        return original_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_marker)
    with pytest.raises(OSError, match="simulated disconnect"):
        save_checkpoint(path, {"value": 2})
    with pytest.warns(RuntimeWarning, match="Recovered previous"):
        assert load_checkpoint(path)["value"] == 1
    monkeypatch.setattr(module.os, "replace", original_replace)
    save_checkpoint(path, {"value": 3})
    assert load_checkpoint(path)["value"] == 3
    assert not path.with_name(path.name + ".previous").exists()


def test_rng_roundtrip_and_verified_backup(tmp_path):
    import random
    import numpy as np
    rng = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng_state(rng)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])
    source = tmp_path / "latest.pt"
    save_checkpoint(source, {"value": 7})
    destination = snapshot_to_drive(source, tmp_path / "persistent")
    assert load_checkpoint(destination)["value"] == 7
