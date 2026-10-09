"""Complete, integrity-checked local checkpoints.

Only load checkpoints you created or otherwise trust. ``torch.load`` uses pickle
for optimizer and RNG state; a SHA256 marker detects corruption, not malicious
code. Model-only third-party checkpoints must not be passed to this loader.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAT_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same number of CUDA devices.")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return value


def marker_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".json")


def _previous_path(path: Path) -> Path:
    return path.with_name(path.name + ".previous")


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _read_verified(path: Path, map_location: Any) -> dict[str, Any]:
    marker = marker_path(path)
    if not path.is_file() or not marker.is_file():
        raise ValueError(f"Incomplete checkpoint: both {path} and {marker} are required.")
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Invalid checkpoint completion marker: {marker}")
    if path.stat().st_size != manifest.get("size_bytes") or sha256_file(path) != manifest.get("sha256"):
        raise ValueError(f"Checkpoint integrity check failed: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION or payload.get("complete") is not True:
        raise ValueError(f"Unsupported or incomplete checkpoint payload: {path}")
    return payload


def load_checkpoint(path: str | Path, map_location: Any = "cpu", allow_previous: bool = True) -> dict[str, Any]:
    """Load a trusted checkpoint, refusing missing markers and corrupt copies.

    A crash between replacing the data file and marker can leave the previous
    complete generation alongside it. Recovery is explicit in a warning.
    """
    path = Path(path)
    try:
        return _read_verified(path, map_location)
    except (ValueError, OSError, json.JSONDecodeError) as current_error:
        previous = _previous_path(path)
        if allow_previous and previous.is_file() and marker_path(previous).is_file():
            try:
                payload = _read_verified(previous, map_location)
            except (ValueError, OSError, json.JSONDecodeError):
                raise current_error
            warnings.warn(f"Recovered previous complete checkpoint after interrupted save: {path}", RuntimeWarning)
            return payload
        raise


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Save at an optimizer boundary; retain one previous generation while replacing.

    Each file is replaced atomically on the same filesystem. A previous verified
    pair protects the two-file handover. Successful saves retain no history.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = marker_path(path)
    previous = _previous_path(path)
    snapshot = _cpu_snapshot(payload)
    snapshot.update(format_version=FORMAT_VERSION, complete=True)
    temporary: Path | None = None
    temporary_marker: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".incomplete", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(snapshot, handle)
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {
            "format_version": FORMAT_VERSION,
            "complete": True,
            "sha256": sha256_file(temporary),
            "size_bytes": temporary.stat().st_size,
            "updates": payload.get("progress", {}).get("updates"),
        }
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=marker.name + ".", suffix=".incomplete", dir=path.parent, delete=False) as handle:
            temporary_marker = Path(handle.name)
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_file() and marker.is_file():
            # Do not overwrite a recoverable previous pair with an incomplete save.
            try:
                existing = json.loads(marker.read_text(encoding="utf-8"))
                valid = existing.get("complete") is True and sha256_file(path) == existing.get("sha256")
            except (OSError, ValueError):
                valid = False
            if valid:
                _copy_or_link(path, previous)
                _copy_or_link(marker, marker_path(previous))
        os.replace(temporary, path)
        temporary = None
        os.replace(temporary_marker, marker)
        temporary_marker = None
        previous.unlink(missing_ok=True)
        marker_path(previous).unlink(missing_ok=True)
        return manifest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if temporary_marker is not None:
            temporary_marker.unlink(missing_ok=True)


def snapshot_to_drive(checkpoint_path: str | Path, backup_directory: str | Path) -> str:
    """Copy a trusted complete checkpoint pair to a mounted persistent directory.

    The caller mounts Google Drive and chooses the directory. This function does
    not authenticate, upload credentials, or assume Drive is available.
    """
    source = Path(checkpoint_path)
    # Validate before copying, then save through the same complete-pair protocol.
    target = Path(backup_directory) / source.name
    if source.resolve() == target.resolve():
        raise ValueError("Checkpoint backup destination must differ from its source.")
    payload = load_checkpoint(source)
    save_checkpoint(target, payload)
    load_checkpoint(target)
    return str(target)
