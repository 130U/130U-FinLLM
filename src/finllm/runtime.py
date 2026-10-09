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
