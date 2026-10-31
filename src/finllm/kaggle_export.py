"""Create a verified, private, CPU-only handoff of the user's Drive snapshots.

The archive includes trusted model checkpoints and reviewed source headlines.
It must only be attached to a private Kaggle dataset controlled by the user.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

from .checkpoint import sha256_file
from .expansion_runtime import _inventory
from .sft_pilot_runtime import _audit_training_exclusions, audit_task_inputs
from .sft_preflight import audit_domain_parent


SNAPSHOTS = (
    "seeds/seed_v1/raw", "seeds/seed_v1/processed", "seeds/seed_v1/tokenizer",
    "expansions/expansion_v1/prepared", "expansions/expansion_v1/processed",
    "expansions/expansion_v1/parent", "domains/domain_v1/run", "domains/domain_v1/parent",
    "domains/domain_v1/retention/before", "domains/domain_v1/retention/after", "tasks/entity_v1",
)
EXPECTED_DOMAIN_SHA = "2a6585bf18ed4753330221292d89b45ab17a52388ab7a0350eaf5d0dc4255c1f"
EXPECTED_TASK_MANIFEST_SHA = "a9b65b81d5280f43822ebb5456f2b6f6304a70bb84e0f61e284aac0597725840"


def _verified_members(root):
    """Read existing snapshot inventories; exclude unrecorded files."""
    files = {}
    for snapshot in SNAPSHOTS:
        print(f"Verifying snapshot: {snapshot}", flush=True)
        folder = root / snapshot
        entries = _inventory(folder)
        marker = folder / "SNAPSHOT_COMPLETE.json"
        if marker.is_symlink() or not marker.resolve().is_relative_to(root):
            raise ValueError("Snapshot marker escapes the trusted root")
        for entry in entries:
            rel = Path(entry["path"])
            if (rel.is_absolute() or not rel.parts or any(part in (".", "..") for part in rel.parts)
                    or "\\" in entry["path"] or rel.as_posix() != entry["path"]):
                raise ValueError("Unsafe snapshot file name")
            path = folder / rel
            if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
                raise ValueError("Symlink or escaping snapshot file")
            name = (Path(snapshot) / rel).as_posix()
            if name in files:
                raise ValueError("Duplicate snapshot member")
            files[name] = {"bytes": entry["bytes"], "sha256": entry["sha256"]}
        name = (Path(snapshot) / marker.name).as_posix()
        files[name] = {"bytes": marker.stat().st_size, "sha256": sha256_file(marker)}
    return files


def export_kaggle_bundle(drive_root, output, *, expected_domain_sha256=EXPECTED_DOMAIN_SHA,
                         expected_task_manifest_sha256=EXPECTED_TASK_MANIFEST_SHA):
    """Audit actual Drive inputs and stream a non-overwriting ZIP to local disk."""
    root, output = Path(drive_root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError("Project Drive directory does not exist")
    if output == root or output.is_relative_to(root):
        raise ValueError("Write the temporary transfer ZIP outside the Drive project directory")
    if output.suffix.lower() != ".zip":
        raise ValueError("Transfer output must be a ZIP file")
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("Existing ZIP or partial ZIP preserved; use a new output name")
    if (root / "sft/sft_entity_v1").exists() or (root / "live_checkpoints/sft_entity_v1").exists():
        raise ValueError("Existing SFT experiment or live checkpoint found; stop and inspect progress before migration")
    task = root / "tasks/entity_v1"
    task_manifest = task / "task_manifest.json"
    if not task_manifest.is_file() or sha256_file(task_manifest) != expected_task_manifest_sha256:
        raise ValueError("Task manifest differs from the recorded, user-reviewed entity set")
    seed = root / "seeds/seed_v1"
    expansion = root / "expansions/expansion_v1"
    domain = root / "domains/domain_v1"
    tokenizer = seed / "tokenizer"
    print("Verifying the domain parent model, training data, and tokenizer; reading Drive may take several minutes.", flush=True)
    parent_audit = audit_domain_parent(
        domain, tokenizer, expansion / "processed/domain_mix/manifest.json",
        expansion / "processed/finance_validation/manifest.json",
        seed / "processed/general_validation/manifest.json",
        expected_checkpoint_sha256=expected_domain_sha256, expected_updates=74)
    counts = audit_task_inputs(task, tokenizer)
    paths = {"seed_drive": seed, "drive": expansion,
             "seed_processed": seed / "processed", "processed": expansion / "processed"}
    print("Checking whether the actual pretraining exclusion pool covers every task headline.", flush=True)
    exclusion_audit = _audit_training_exclusions(paths, task, domain)
    files = _verified_members(root)
    total = sum(item["bytes"] for item in files.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < total + 128 * 1024 * 1024:
        raise OSError("Not enough local disk space for the uncompressed transfer archive")
    print(f"Snapshots total approximately {total / 1024 ** 3:.2f} GiB. Writing the temporary transfer package.", flush=True)
    manifest = {"format": "130u-kaggle-transfer-v1", "snapshots": list(SNAPSHOTS), "files": files}
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    with zipfile.ZipFile(partial, mode="x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, item in sorted(files.items()):
            digest, size = hashlib.sha256(), 0
            with (root / name).open("rb") as source, archive.open(name, "w", force_zip64=True) as target:
                while chunk := source.read(4 * 1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Snapshot changed during export; partial ZIP preserved: {name}")
        archive.writestr("transfer_manifest.json", manifest_bytes)
    print("Reading back the transfer package and verifying every file.", flush=True)
    with zipfile.ZipFile(partial) as archive:
        if set(archive.namelist()) != set(files) | {"transfer_manifest.json"}:
            raise ValueError("Transfer ZIP entries do not match manifest")
        for name, item in files.items():
            digest, size = hashlib.sha256(), 0
            with archive.open(name) as stream:
                while chunk := stream.read(4 * 1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
            if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Transfer ZIP verification failed: {name}")
    # Non-overwriting publish; link succeeds only when the destination is absent.
    os.link(partial, output)
    partial.unlink()
    return {"output": str(output), "bytes": output.stat().st_size, "sha256": sha256_file(output),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "snapshot_count": len(SNAPSHOTS), "files": len(files), "task_counts": counts,
            "parent_audit": parent_audit, "actual_pretraining_exclusions": exclusion_audit,
            "privacy": "Contains checkpoints and source headlines; attach only to a private Kaggle dataset."}
