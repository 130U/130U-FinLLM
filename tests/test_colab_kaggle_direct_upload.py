"""No network/credentials: exercise safe direct-upload staging and failure paths."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("upload_direct", ROOT / "scripts/colab_kaggle_direct_upload.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def transfer(tmp_path):
    path = tmp_path / "130u_kaggle_transfer.zip"
    manifest = json.dumps({"format": "130u-kaggle-transfer-v1", "files": {}}).encode()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("transfer_manifest.json", manifest)
        archive.writestr("private.bin", b"private research content")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, {"output": str(path), "sha256": sha,
                  "manifest_sha256": hashlib.sha256(manifest).hexdigest()}


def test_upload_is_private_unique_and_does_not_leak_or_overwrite(transfer, tmp_path, capsys):
    path, report = transfer
    original = path.read_bytes()
    calls = []
    sentinel = "private-secret-not-for-display"

    def runner(args, **kwargs):
        calls.append((args, kwargs.get("env")))
        if args[:3] == ["kaggle", "datasets", "create"]:
            folder = Path(args[args.index("--path") + 1])
            meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
            assert meta["id"].startswith("theotheo66/130u-finllm-private-")
            assert meta["licenses"] == [{"name": "unknown"}]
            assert (folder / path.name).read_bytes() == original
            assert sentinel not in (folder / "dataset-metadata.json").read_text(encoding="utf-8")
            assert kwargs["env"]["KAGGLE_API_TOKEN"] == sentinel
        return SimpleNamespace(returncode=0)

    result = module.upload_existing_bundle(path, report, token_reader=lambda _: sentinel, runner=runner)
    assert result["status"] == "upload_submitted" and result["requested_private"] is True
    assert result["transfer_zip_sha256"] == report["sha256"]
    assert result["transfer_manifest_sha256"] == report["manifest_sha256"]
    assert len(calls) == 2 and "--public" not in calls[1][0] and "-u" not in calls[1][0]
    assert "--keep-tabular" in calls[1][0]
    assert all(sentinel not in str(args) for args, _ in calls)
    assert sentinel not in capsys.readouterr().out
    assert path.read_bytes() == original
    assert list(tmp_path.glob("130u-kaggle-upload-*")) == []


@pytest.mark.parametrize("bad", ["wrong_zip_hash", "wrong_manifest_hash", "wrong_path", "bad_owner"])
def test_invalid_input_refuses_before_network_or_token(transfer, bad):
    path, report = transfer
    if bad == "wrong_zip_hash": report["sha256"] = "0" * 64
    if bad == "wrong_manifest_hash": report["manifest_sha256"] = "0" * 64
    if bad == "wrong_path": report["output"] = str(path.parent / "elsewhere.zip")
    called = []
    with pytest.raises(ValueError):
        module.upload_existing_bundle(path, report, owner="../other" if bad == "bad_owner" else "theotheo66",
                                      token_reader=lambda _: called.append("token"),
                                      runner=lambda *a, **k: called.append("network"))
    assert called == []


def test_cli_failure_suppresses_output_and_preserves_source(transfer, tmp_path, capsys):
    path, report = transfer
    original = path.read_bytes()
    sentinel = "do-not-expose-token"
    def runner(args, **kwargs):
        return SimpleNamespace(returncode=1 if args[0] == "kaggle" else 0)
    with pytest.raises(RuntimeError, match="Kaggle command failed") as exc:
        module.upload_existing_bundle(path, report, token_reader=lambda _: sentinel, runner=runner)
    assert sentinel not in str(exc.value) + capsys.readouterr().out
    assert path.read_bytes() == original
    assert list(tmp_path.glob("130u-kaggle-upload-*")) == []
