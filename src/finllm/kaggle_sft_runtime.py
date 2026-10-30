"""Private, verified handoff into a Kaggle working directory.

The input must be the user's own trusted checkpoint archive. PyTorch checkpoint
loading uses pickle and is not safe for arbitrary third-party archives.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from .checkpoint import load_checkpoint, sha256_file
from .continuation import _snapshot_file

DOMAIN_SHA256 = "2a6585bf18ed4753330221292d89b45ab17a52388ab7a0350eaf5d0dc4255c1f"
_ROOTS = {"seeds", "expansions", "domains", "tasks"}
_MAX_SIZE = 20 * 1024 ** 3


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest field")
        result[key] = value
    return result


def _path(name, roots):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("Invalid archive path")
    value = PurePosixPath(name)
    if (value.is_absolute() or any(part in ("", ".", "..") for part in name.split("/"))
            or value.parts[0] not in roots or ":" in name):
        raise ValueError("Archive path escapes permitted project folders")
    return value


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _read_manifest(archive, name, format_name, roots):
    infos = archive.infolist()
    names = [item.filename for item in infos]
    if len(names) != len(set(names)) or name not in names:
        raise ValueError("Archive contains duplicate entries or no manifest")
    item = archive.getinfo(name)
    if item.file_size > 8 * 1024 * 1024:
        raise ValueError("Archive manifest is too large")
    manifest = json.loads(archive.read(name), object_pairs_hook=_object)
    if manifest.get("format") != format_name or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported archive manifest")
    if set(names) != set(manifest["files"]) | {name}:
        raise ValueError("Archive files differ from manifest")
    if sum(item.file_size for item in infos) > _MAX_SIZE:
        raise ValueError("Archive exceeds maximum allowed size")
    for info in infos:
        if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError("Directories and symbolic links are not accepted in archives")
        if info.filename == name:
            continue
        _path(info.filename, roots)
        record = manifest["files"][info.filename]
        if (not isinstance(record, dict) or type(record.get("bytes")) is not int
                or record["bytes"] != info.file_size or not _digest(record.get("sha256"))):
            raise ValueError("Invalid file manifest entry")
    return manifest


def _copy_verified(archive, manifest, destination):
    destination = Path(destination)
    for name, metadata in manifest["files"].items():
        target = destination.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest, size = hashlib.sha256(), 0
        with archive.open(name) as source, target.open("xb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                size += len(block)
        if size != metadata["bytes"] or digest.hexdigest() != metadata["sha256"]:
            raise ValueError(f"Archive content hash mismatch: {name}")


def _verify_trusted_model_chain(root, expected_domain_sha256):
    # Pin every pickle checkpoint to a previously trusted checkpoint before
    # loading it: domain -> expansion -> seed.
    domain = Path(root) / "domains/domain_v1"
    trained = _snapshot_file(domain / "run", "latest.pt")
    if sha256_file(trained) != expected_domain_sha256:
        raise ValueError("Domain checkpoint differs from the known 74-step result")
    payload = load_checkpoint(trained, allow_previous=False)
    expansion = _snapshot_file(domain / "parent", "latest.pt")
    transition = payload.get("config", {}).get("domain_transition", {})
    if sha256_file(expansion) != transition.get("parent_checkpoint_sha256"):
        raise ValueError("Expansion parent does not match trusted domain checkpoint")
    parent = load_checkpoint(expansion, allow_previous=False)
    seed = _snapshot_file(Path(root) / "expansions/expansion_v1/parent", "latest.pt")
    if sha256_file(seed) != parent.get("provenance", {}).get("parent_lineage", {}).get("parent_checkpoint_sha256"):
        raise ValueError("Seed parent does not match trusted expansion checkpoint")
    return {"domain_checkpoint_sha256": expected_domain_sha256,
            "expansion_parent_sha256": sha256_file(expansion),
            "seed_parent_sha256": sha256_file(seed)}


def install_transfer_archive(zip_path, destination, *, expected_zip_sha256,
                             expected_domain_sha256=DOMAIN_SHA256):
    """Install complete source snapshots once; retain any previously created run."""
    zip_path, destination = Path(zip_path), Path(destination)
    if not _digest(expected_zip_sha256) or sha256_file(zip_path) != expected_zip_sha256:
        raise ValueError("Transfer ZIP SHA256 does not match the value printed in Colab")
    if not _digest(expected_domain_sha256):
        raise ValueError("Invalid trusted domain checkpoint SHA256")
    with zipfile.ZipFile(zip_path) as archive:
        manifest = _read_manifest(archive, "transfer_manifest.json", "130u-kaggle-transfer-v1", _ROOTS)
        transfer_identity = hashlib.sha256(archive.read("transfer_manifest.json")).hexdigest()
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError("Existing destination is not a regular directory")
            marker = destination / "TRANSFER_COMPLETE.json"
            if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")).get("zip_sha256") != expected_zip_sha256:
                raise FileExistsError("Existing transfer is incomplete or differs; preserved")
            for name, item in manifest["files"].items():
                file = destination.joinpath(*PurePosixPath(name).parts)
                if (file.is_symlink() or not file.resolve().is_relative_to(destination.resolve())
                        or not file.is_file() or file.stat().st_size != item["bytes"]
                        or sha256_file(file) != item["sha256"]):
                    raise ValueError("Existing transferred inputs changed; refusing reuse")
            return {"reused": True, "zip_sha256": expected_zip_sha256,
                    "trusted_chain": _verify_trusted_model_chain(destination, expected_domain_sha256)}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".kaggle-transfer-", dir=destination.parent) as temp:
            staging = Path(temp) / "handoff"
            staging.mkdir()
            _copy_verified(archive, manifest, staging)
            chain = _verify_trusted_model_chain(staging, expected_domain_sha256)
            (staging / "TRANSFER_COMPLETE.json").write_text(json.dumps({"format": "130u-transfer-installed-v1",
                "zip_sha256": expected_zip_sha256, "transfer_identity": transfer_identity}, indent=2) + "\n", encoding="utf-8")
            os.rename(staging, destination)
        return {"reused": False, "zip_sha256": expected_zip_sha256, "trusted_chain": chain}


def install_transfer_directory(source, destination, *, expected_manifest_sha256,
                               expected_domain_sha256=DOMAIN_SHA256):
    """Install a Kaggle Dataset after Kaggle has extracted the original ZIP.

    The ZIP bytes are no longer available. Instead compare the manifest bytes
    to the independent hash displayed by the exporter and then verify every
    extracted file against that manifest before trusting any checkpoint.
    """
    source, destination = Path(source), Path(destination)
    if not _digest(expected_manifest_sha256) or not _digest(expected_domain_sha256):
        raise ValueError("Supply the 64-character manifest and trusted domain SHA256 values")
    marker = source / "transfer_manifest.json"
    if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Extracted transfer manifest missing or invalid")
    raw = marker.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("Extracted manifest SHA256 differs from the Colab export")
    manifest = json.loads(raw, object_pairs_hook=_object)
    if manifest.get("format") != "130u-kaggle-transfer-v1" or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported extracted transfer manifest")
    if not manifest["files"]:
        raise ValueError("Extracted transfer has no files")
    total = 0
    for name, entry in manifest["files"].items():
        _path(name, _ROOTS)
        if (not isinstance(entry, dict) or type(entry.get("bytes")) is not int or entry["bytes"] < 0
                or not _digest(entry.get("sha256"))):
            raise ValueError("Invalid extracted transfer file entry")
        total += entry["bytes"]
    if total > _MAX_SIZE:
        raise ValueError("Extracted transfer exceeds allowed size")
    def verify_input_file(base, name, entry):
        file = base.joinpath(*PurePosixPath(name).parts)
        if (file.is_symlink() or not file.resolve().is_relative_to(base.resolve()) or not file.is_file()
                or file.stat().st_size != entry["bytes"] or sha256_file(file) != entry["sha256"]):
            raise ValueError(f"Transferred input changed or missing: {name}")
        return file
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("Existing destination is not a regular directory")
        installed = destination / "TRANSFER_COMPLETE.json"
        if (not installed.is_file() or json.loads(installed.read_text(encoding="utf-8")).get("transfer_identity")
                != expected_manifest_sha256):
            raise FileExistsError("Existing transfer is incomplete or differs; preserved")
        for name, entry in manifest["files"].items():
            verify_input_file(destination, name, entry)
        return {"reused": True, "manifest_sha256": expected_manifest_sha256,
                "trusted_chain": _verify_trusted_model_chain(destination, expected_domain_sha256)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kaggle-transfer-", dir=destination.parent) as temp:
        staging = Path(temp) / "handoff"
        staging.mkdir()
        for name, entry in manifest["files"].items():
            source_file = verify_input_file(source, name, entry)
            target = staging.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source_file.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            verify_input_file(staging, name, entry)
        chain = _verify_trusted_model_chain(staging, expected_domain_sha256)
        (staging / "TRANSFER_COMPLETE.json").write_text(json.dumps({"format": "130u-transfer-installed-v1",
            "transfer_identity": expected_manifest_sha256}, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, destination)
    return {"reused": False, "manifest_sha256": expected_manifest_sha256, "trusted_chain": chain}


def _recovery_path(name, run_version):
    _path(name, {"sft", "live_checkpoints"})
    allowed = (f"sft/{run_version}/report_archive/", f"sft/{run_version}/run/",
               f"live_checkpoints/{run_version}/")
    if not name.startswith(allowed):
        raise ValueError("Recovery contains files outside this run")


def export_recovery(root, output, *, run_version="sft_kaggle_v1"):
    """Save current trusted live checkpoint and reports for user download.

    Kaggle working files may disappear on session termination; creating this
    archive is not proof that the user has downloaded or saved a version.
    """
    if not run_version or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for c in run_version):
        raise ValueError("Invalid run version")
    root, output = Path(root), Path(output)
    live = root / "live_checkpoints" / run_version
    checkpoint = load_checkpoint(live / "latest.pt", allow_previous=False)
    if checkpoint.get("config", {}).get("sft_pilot_binding", {}).get("run_version") != run_version:
        raise ValueError("Live checkpoint does not belong to this SFT run")
    marker = json.loads((root / "TRANSFER_COMPLETE.json").read_text(encoding="utf-8"))
    transfer_sha = marker.get("transfer_identity", marker.get("zip_sha256"))
    if not _digest(transfer_sha):
        raise ValueError("Transfer identity missing")
    roots = (root / "sft" / run_version / "report_archive",
             root / "sft" / run_version / "run", live)
    files = {}
    for directory in roots:
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_symlink():
                    raise ValueError("Recovery source cannot contain symlinks")
                if path.is_file():
                    name = path.relative_to(root).as_posix()
                    _recovery_path(name, run_version)
                    files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if not files or sum(item["bytes"] for item in files.values()) > _MAX_SIZE:
        raise ValueError("Recovery contains no data or exceeds allowed size")
    manifest = {"format": "130u-kaggle-recovery-v1", "run_version": run_version,
                "source_transfer_sha256": transfer_sha,
                "progress": checkpoint.get("progress"), "files": files}
    manifest_raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("Existing recovery or partial archive preserved")
    with partial.open("xb") as handle, zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in sorted(files):
            archive.write(root.joinpath(*PurePosixPath(name).parts), name)
        archive.writestr("recovery_manifest.json", manifest_raw)
    with zipfile.ZipFile(partial) as archive:
        check = _read_manifest(archive, "recovery_manifest.json", "130u-kaggle-recovery-v1", {"sft", "live_checkpoints"})
        for name, item in check["files"].items():
            digest = hashlib.sha256()
            with archive.open(name) as source:
                while block := source.read(1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError("Recovery archive readback failed")
    os.link(partial, output)
    partial.unlink()
    return {"path": str(output), "sha256": sha256_file(output), "bytes": output.stat().st_size,
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "updates": checkpoint["progress"]["updates"], "download_or_save_version_required": True}


def install_recovery_archive(zip_path, root, *, expected_zip_sha256, run_version="sft_kaggle_v1"):
    """Restore one user-saved recovery; never overwrite existing run outputs."""
    zip_path, root = Path(zip_path), Path(root)
    if not _digest(expected_zip_sha256) or sha256_file(zip_path) != expected_zip_sha256:
        raise ValueError("Recovery ZIP SHA256 does not match the saved value")
    marker = json.loads((root / "TRANSFER_COMPLETE.json").read_text(encoding="utf-8"))
    transfer_sha = marker.get("transfer_identity", marker.get("zip_sha256"))
    with zipfile.ZipFile(zip_path) as archive:
        manifest = _read_manifest(archive, "recovery_manifest.json", "130u-kaggle-recovery-v1", {"sft", "live_checkpoints"})
        if manifest.get("run_version") != run_version or manifest.get("source_transfer_sha256") != transfer_sha:
            raise ValueError("Recovery belongs to another transfer or SFT run")
        for name in manifest["files"]:
            _recovery_path(name, run_version)
        for directory in (root / "sft" / run_version, root / "live_checkpoints" / run_version):
            if directory.exists():
                raise FileExistsError("Existing SFT run preserved; do not overlay a recovery archive")
        with tempfile.TemporaryDirectory(prefix=".kaggle-recovery-", dir=root) as temp:
            staging = Path(temp)
            _copy_verified(archive, manifest, staging)
            for part in ("sft", "live_checkpoints"):
                source = staging / part / run_version
                if source.exists():
                    target = root / part / run_version
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(source, target)
    return {"restored": True, "recovery_sha256": expected_zip_sha256, "progress": manifest["progress"]}


def install_recovery_directory(source, root, *, expected_manifest_sha256, run_version="sft_kaggle_v1"):
    """Restore a recovery ZIP that Kaggle has extracted into a private input."""
    source, root = Path(source), Path(root)
    if not _digest(expected_manifest_sha256):
        raise ValueError("Supply the saved recovery manifest SHA256")
    marker = source / "recovery_manifest.json"
    if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Extracted recovery manifest missing or invalid")
    raw = marker.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("Extracted recovery manifest differs from the saved version")
    manifest = json.loads(raw, object_pairs_hook=_object)
    transfer = json.loads((root / "TRANSFER_COMPLETE.json").read_text(encoding="utf-8"))
    transfer_sha = transfer.get("transfer_identity", transfer.get("zip_sha256"))
    if (manifest.get("format") != "130u-kaggle-recovery-v1"
            or manifest.get("run_version") != run_version or manifest.get("source_transfer_sha256") != transfer_sha
            or not isinstance(manifest.get("files"), dict) or not manifest["files"]):
        raise ValueError("Recovery belongs to another transfer or SFT run")
    total = 0
    for name, entry in manifest["files"].items():
        _recovery_path(name, run_version)
        if (not isinstance(entry, dict) or type(entry.get("bytes")) is not int or entry["bytes"] < 0
                or not _digest(entry.get("sha256"))):
            raise ValueError("Invalid extracted recovery file entry")
        total += entry["bytes"]
    if total > _MAX_SIZE:
        raise ValueError("Extracted recovery exceeds allowed size")
    for directory in (root / "sft" / run_version, root / "live_checkpoints" / run_version):
        if directory.exists():
            raise FileExistsError("Existing SFT run preserved; do not overlay a recovery archive")
    with tempfile.TemporaryDirectory(prefix=".kaggle-recovery-", dir=root) as temp:
        staging = Path(temp)
        for name, entry in manifest["files"].items():
            file = source.joinpath(*PurePosixPath(name).parts)
            if (file.is_symlink() or not file.resolve().is_relative_to(source.resolve())
                    or not file.is_file() or file.stat().st_size != entry["bytes"]
                    or sha256_file(file) != entry["sha256"]):
                raise ValueError(f"Extracted recovery content mismatch: {name}")
            target = staging.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with file.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            if target.stat().st_size != entry["bytes"] or sha256_file(target) != entry["sha256"]:
                raise ValueError("Recovery copy changed")
        for part in ("sft", "live_checkpoints"):
            folder = staging / part / run_version
            if folder.exists():
                target = root / part / run_version
                target.parent.mkdir(parents=True, exist_ok=True)
                os.rename(folder, target)
    return {"restored": True, "recovery_manifest_sha256": expected_manifest_sha256,
            "progress": manifest["progress"]}
