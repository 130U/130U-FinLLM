import importlib.util
import json
from pathlib import Path

import pytest
from finllm.data import file_sha256

spec = importlib.util.spec_from_file_location("seed_artifacts", Path(__file__).parents[1] / "scripts/build_seed_artifacts.py")
seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed)


def test_partial_and_changed_downloads_cannot_enter_preparation(tmp_path):
    manifest = {"status": "in_progress"}
    for name in ("general", "finance"):
        path = tmp_path / f"{name}_candidate.jsonl"
        path.write_text('{"text":"' + name + '"}\n')
        manifest[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    marker = tmp_path / "manifest.json"
    marker.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        seed.validate_source_snapshot(tmp_path)
    manifest["status"] = "complete"
    marker.write_text(json.dumps(manifest))
    assert seed.validate_source_snapshot(tmp_path)["status"] == "complete"
    (tmp_path / "finance_candidate.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        seed.validate_source_snapshot(tmp_path)
