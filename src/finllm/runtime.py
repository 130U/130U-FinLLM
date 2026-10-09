"""Environment reporting and verified file snapshots for Colab/Drive."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import uuid
from pathlib import Path

import torch

from .data import file_sha256, write_json


def environment_report():
    packages = {}
    for name in ("torch", "numpy", "tokenizers", "scikit-learn", "datasets"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    ram = None
    if hasattr(os, "sysconf"):
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    return {"python": platform.python_version(), "platform": platform.system(), "packages": packages,
            "git_commit": commit, "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "system_ram_bytes": ram, "disk_free_bytes": shutil.disk_usage(Path.cwd()).free}


def sync_snapshot(source, destination, working_copy=False):
    """Copy and verify files; optionally restore a mutable working directory.

    Completed backup sources always undergo manifest verification. A restored
    working copy omits snapshot markers so later training can change its files
    and a subsequent backup can capture their new hashes.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise ValueError(f"Snapshot source is not a directory: {source}")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Snapshot source/destination must be separate non-nested directories")
    if (source / "SNAPSHOT_INCOMPLETE.json").exists():
        raise ValueError("Snapshot source is incomplete; refusing restore")
    marker = source / "SNAPSHOT_COMPLETE.json"
    if marker.exists():
        source_manifest = json.loads(marker.read_text(encoding="utf-8"))
        if source_manifest.get("format") != "finllm-snapshot-v1" or not isinstance(source_manifest.get("files"), list):
            raise ValueError("Invalid snapshot manifest format")
        selected = []
        seen = set()
        for entry in source_manifest["files"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError("Invalid snapshot file entry")
            relative = Path(entry["path"])
            path = (source / relative).resolve()
            if (relative.is_absolute() or not path.is_relative_to(source) or path in seen
                    or path.name in ("SNAPSHOT_COMPLETE.json", "SNAPSHOT_INCOMPLETE.json")
                    or not path.is_file() or not isinstance(entry.get("bytes"), int)
                    or path.stat().st_size != entry["bytes"] or file_sha256(path) != entry.get("sha256")):
                raise ValueError("Snapshot source incomplete or changed; refusing restore")
            seen.add(path)
            selected.append((path, entry["sha256"], entry["bytes"]))
    else:
        selected = [(path, None, None) for path in source.rglob("*") if path.is_file() and not path.is_symlink()
                    and not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(source).parts)
                    and path.name != "SNAPSHOT_COMPLETE.json"]
    if not selected:
        raise ValueError("Nothing to snapshot")
    destination.mkdir(parents=True, exist_ok=True)
    completed = destination / "SNAPSHOT_COMPLETE.json"
    incomplete = destination / "SNAPSHOT_INCOMPLETE.json"
    # Keep a failure marker so an interrupted backup cannot later masquerade as
    # an ordinary source directory and be restored without verification.
    write_json(incomplete, {"format": "finllm-snapshot-incomplete-v1"})
    # An old complete marker must not certify a partially replaced snapshot.
    if completed.exists():
        completed.unlink()
    files = []
    for path, expected_digest, expected_bytes in sorted(selected, key=lambda entry: entry[0]):
        relative = path.relative_to(source)
        target = destination / relative
        if not target.resolve().is_relative_to(destination):
            raise ValueError("Snapshot destination path escapes its directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = expected_digest if expected_digest is not None else file_sha256(path)
        size = expected_bytes if expected_bytes is not None else path.stat().st_size
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            shutil.copy2(path, temporary)
            if temporary.stat().st_size != size or file_sha256(temporary) != digest:
                raise OSError(f"Snapshot hash mismatch: {relative}")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        files.append({"path": relative.as_posix(), "sha256": digest, "bytes": target.stat().st_size})
    manifest = {"format": "finllm-snapshot-v1", "files": files}
    temporary_marker = completed.with_suffix(".tmp")
    write_json(temporary_marker, manifest)
    os.replace(temporary_marker, completed)
    if working_copy:
        completed.unlink()
    incomplete.unlink()
    return {"files": len(files), "bytes": sum(row["bytes"] for row in files),
            "verified": True, "working_copy": bool(working_copy)}


def restore_seed_checkpoint(local_dir, live_dir):
    """Select a verified latest seed checkpoint without restarting saved progress.

    Both directories must contain only this trusted training run. Existing
    complete checkpoint pairs, including the recoverable previous generation,
    are inspected with ``load_checkpoint``. A newer live directory is copied in
    full to preserve its best and previous pairs. Divergent run identity or
    incomparable progress refuses automatic selection. This helper never writes
    to the persistent live directory.

    Return the absolute local ``latest.pt`` path for ``--resume``, or ``None``
    only when neither location contains any checkpoint artifacts. Training still
    checks the selected checkpoint against the current configuration/runtime.
    """
    from .checkpoint import load_checkpoint

    local, live = Path(local_dir).resolve(), Path(live_dir).resolve()
    if local == live or local in live.parents or live in local.parents:
        raise ValueError("Local/live checkpoint directories must be separate and non-nested")

    def inspect(directory):
        if not directory.exists():
            return None, False, None
        if not directory.is_dir():
            raise ValueError(f"Checkpoint location is not a directory: {directory}")
        artifacts = list(directory.glob("*.pt*"))
        incomplete = (directory / "SNAPSHOT_INCOMPLETE.json").exists()
        if not artifacts and not incomplete:
            return None, False, None
        try:
            if incomplete:
                raise ValueError("Checkpoint directory has an incomplete snapshot marker")
            payload = load_checkpoint(directory / "latest.pt", map_location="cpu")
            for key in ("model_config", "provenance"):
                if not isinstance(payload.get(key), dict) or not payload[key]:
                    raise ValueError(f"Checkpoint requires nonempty {key}")
            if not isinstance(payload.get("tokenizer_hash"), str) or not payload["tokenizer_hash"]:
                raise ValueError("Checkpoint requires tokenizer_hash")
            progress = payload.get("progress")
            if not isinstance(progress, dict):
                raise ValueError("Checkpoint requires progress counters")
            for key in ("attempted_updates", "updates", "tokens"):
                value = progress.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"Invalid checkpoint progress counter: {key}")
            if progress["updates"] > progress["attempted_updates"]:
                raise ValueError("Successful updates cannot exceed attempted_updates")
            return payload, True, None
        except (OSError, ValueError, EOFError, RuntimeError) as error:
            return None, True, str(error)

    local_payload, local_artifacts, local_error = inspect(local)
    live_payload, live_artifacts, live_error = inspect(live)
    if local_payload is None and live_payload is None:
        if local_artifacts or live_artifacts:
            raise ValueError("Checkpoint artifacts exist but no recoverable latest checkpoint was found; "
                             f"refusing fresh training. Local: {local_error}; live: {live_error}")
        return None

    selected = local_payload
    restore_live = local_payload is None
    if local_payload is not None and live_payload is not None:
        for key in ("model_config", "tokenizer_hash", "provenance"):
            if local_payload[key] != live_payload[key]:
                raise ValueError(f"Local/live checkpoint provenance diverges ({key}); select the correct run explicitly")
        local_progress, live_progress = local_payload["progress"], live_payload["progress"]
        counters = ("attempted_updates", "tokens", "updates")
        differences = [live_progress[key] - local_progress[key] for key in counters]
        if any(value > 0 for value in differences) and any(value < 0 for value in differences):
            raise ValueError("Local/live checkpoint progress has inconsistent ordering")
        if differences[0] == 0 and any(differences[1:]):
            raise ValueError("Local/live checkpoint progress disagrees at the same attempted_updates")
        if not any(differences) and local_progress != live_progress:
            raise ValueError("Local/live checkpoint progress disagrees at the same counters")
        restore_live = any(value > 0 for value in differences)
    if restore_live:
        selected = live_payload
        sync_snapshot(live, local, working_copy=True)

    # The source could have changed while copying, or an old local fallback
    # could have survived a partial copy. Confirm selection again before resume.
    verified, _, error = inspect(local)
    if verified is None:
        raise ValueError(f"Selected local checkpoint failed verification: {error}")
    for key in ("model_config", "tokenizer_hash", "provenance", "progress"):
        if verified[key] != selected[key]:
            raise ValueError("Selected checkpoint changed during restoration; refusing resume")
    return str(local / "latest.pt")
