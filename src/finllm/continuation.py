"""Bounded continuation of an existing seed run, with persistent report history.

Only project-created checkpoints are accepted. The training CLI remains the
authority for exact data, optimizer, scheduler and runtime resume compatibility.
"""
from __future__ import annotations

import copy
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from .checkpoint import load_checkpoint, sha256_file
from .runtime import restore_seed_checkpoint, sync_snapshot
from .training import _settings


def _snapshot_file(directory: Path, name: str) -> Path:
    """Read one certified file without restoring a stale run over newer work."""
    if (directory / "SNAPSHOT_INCOMPLETE.json").exists():
        raise ValueError(f"Incomplete snapshot: {directory}")
    manifest = json.loads((directory / "SNAPSHOT_COMPLETE.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "finllm-snapshot-v1":
        raise ValueError("Invalid snapshot format")
    entries = [row for row in manifest["files"] if row.get("path") == name]
    path = directory / name
    if (len(entries) != 1 or not path.is_file()
            or path.stat().st_size != entries[0]["bytes"]
            or sha256_file(path) != entries[0]["sha256"]):
        raise ValueError(f"Snapshot file is missing or changed: {path}")
    return path


def _report_matches(report, checkpoint):
    progress = checkpoint["progress"]
    return (report.get("stage") == "pretrain"
            and report.get("mode") == checkpoint["provenance"]["mode"]
            and all(report.get(key) == progress[key] for key in ("updates", "attempted_updates", "tokens"))
            and report.get("loss") == checkpoint.get("last_loss")
            and report.get("validation") == checkpoint.get("last_validation"))


def _write_archive(destination, report_path, metadata):
    """A named archive is immutable; a retry must agree with its existing bytes."""
    destination = Path(destination)
    encoded = json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8")
    if (destination / "SNAPSHOT_COMPLETE.json").exists() and not (destination / "SNAPSHOT_INCOMPLETE.json").exists():
        report = _snapshot_file(destination, "run_summary.json")
        existing = _snapshot_file(destination, "archive.json")
        if report.read_bytes() != Path(report_path).read_bytes() or existing.read_bytes() != encoded:
            raise ValueError(f"Refusing to replace an existing report archive: {destination}")
        (destination / "PENDING_REPORT.json").unlink(missing_ok=True)
        return destination
    # Persist the small report and its segment boundaries before snapshot IO.
    # A later kernel can finish the same archive even if its run summary has
    # already been replaced or only the live checkpoints survived.
    pending = destination / "PENDING_REPORT.json"
    bundle = {"format": "finllm-pending-report-v1", "metadata": metadata,
              "report_bytes": base64.b64encode(Path(report_path).read_bytes()).decode("ascii")}
    body = json.dumps(bundle, sort_keys=True).encode("utf-8")
    envelope = json.dumps({"sha256": hashlib.sha256(body).hexdigest(), "body": body.decode("utf-8")})
    destination.mkdir(parents=True, exist_ok=True)
    if pending.exists():
        if pending.read_text(encoding="utf-8") != envelope:
            raise ValueError(f"Pending report archive has different content: {destination}")
    else:
        temporary = pending.with_suffix(".tmp")
        temporary.write_text(envelope, encoding="utf-8")
        os.replace(temporary, pending)
    with tempfile.TemporaryDirectory(prefix="finllm-report-") as directory:
        source = Path(directory)
        (source / "run_summary.json").write_bytes(Path(report_path).read_bytes())
        (source / "archive.json").write_bytes(encoded)
        sync_snapshot(source, destination)
    _snapshot_file(destination, "run_summary.json")
    _snapshot_file(destination, "archive.json")
    pending.unlink()
    return destination


def _recover_report_archives(seed):
    """Complete pending archive IO, or stop without claiming a safe backup."""
    root = Path(seed) / "report_archive"
    if not root.exists():
        return
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        pending = directory / "PENDING_REPORT.json"
        if pending.exists():
            envelope = json.loads(pending.read_text(encoding="utf-8"))
            body = envelope["body"].encode("utf-8")
            if hashlib.sha256(body).hexdigest() != envelope["sha256"]:
                raise ValueError(f"Pending report integrity check failed: {pending}")
            bundle = json.loads(body)
            if bundle.get("format") != "finllm-pending-report-v1":
                raise ValueError("Invalid pending report archive format")
            with tempfile.TemporaryDirectory(prefix="finllm-report-retry-") as temporary:
                source = Path(temporary) / "run_summary.json"
                source.write_bytes(base64.b64decode(bundle["report_bytes"], validate=True))
                _write_archive(directory, source, bundle["metadata"])
        else:
            _snapshot_file(directory, "run_summary.json")
            _snapshot_file(directory, "archive.json")


def _verify_checkpoint_config(checkpoint, baseline):
    if checkpoint["config"].get("stage") != "pretrain":
        raise ValueError("Seed continuation requires a pretraining checkpoint")
    expected, actual = _settings(baseline), _settings(checkpoint["config"])
    expected.pop("max_updates")
    actual.pop("max_updates")
    if expected != actual:
        raise ValueError("Continuation may change only max_updates; pilot training settings differ")


def prepare_seed_continuation(project, drive_root, seed_version="seed_v1", *,
                              pilot_updates=61, probe_updates=62, target_updates=300):
    """Restore inputs/run and archive the real pilot report before any new step."""
    project, drive_root = Path(project).resolve(), Path(drive_root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", seed_version):
        raise ValueError("Invalid seed version")
    if not (1 <= pilot_updates < probe_updates < target_updates):
        raise ValueError("Expected pilot < probe < target update budgets")
    seed = drive_root / "seeds" / seed_version
    _recover_report_archives(seed)
    # All inputs must be backed up and certified. Never silently replace changed
    # local input files or modify the frozen pilot_config.json.
    for name, destination in (("processed", project / "data/processed" / seed_version),
                              ("tokenizer", project / "tokenizer" / seed_version)):
        source = seed / name
        _snapshot_file(source, "pilot_config.json" if name == "processed" else "tokenizer.json")
        with tempfile.TemporaryDirectory(prefix="finllm-inputs-") as temporary:
            verified = Path(temporary) / name
            sync_snapshot(source, verified, working_copy=True)
            if not destination.exists():
                sync_snapshot(verified, destination, working_copy=True)
            else:
                for path in verified.rglob("*"):
                    if path.is_file():
                        local = destination / path.relative_to(verified)
                        if not local.is_file() or sha256_file(local) != sha256_file(path):
                            raise ValueError(f"Local seed input differs from its verified backup: {local}")
    config_path = project / "data/processed" / seed_version / "pilot_config.json"
    baseline = json.loads(config_path.read_text(encoding="utf-8"))
    if baseline.get("stage") != "pretrain" or baseline["train"]["max_updates"] != pilot_updates:
        raise ValueError("This continuation requires the unchanged original pilot configuration")
    output = (project / baseline["output_dir"]).resolve()
    if output != project / "runs" / (seed_version + "_pretrain"):
        raise ValueError("Pilot output directory does not match this seed version")
    live = drive_root / "live_checkpoints" / (seed_version + "_pretrain")
    snapshot = seed / "run"
    if not list(output.glob("*.pt*")) and snapshot.exists():
        # An interrupted final full snapshot must not hide intact automatic
        # checkpoints. Only fall back when the independent pilot archive is
        # already certified; keep the incomplete snapshot untouched for now.
        try:
            _snapshot_file(snapshot, "run_summary.json")
            sync_snapshot(snapshot, output, working_copy=True)
        except (OSError, ValueError):
            archive = seed / "report_archive" / f"pilot_updates_{pilot_updates:06d}"
            _snapshot_file(archive, "run_summary.json")
            _snapshot_file(archive, "archive.json")
            if restore_seed_checkpoint(output, live) is None:
                raise ValueError("Run snapshot is incomplete and no verified live checkpoint is available")

    # Preserve the measured pilot report before selection of a newer live pair.
    # On later executions the archive is checked against the selected run identity.
    archive = seed / "report_archive" / f"pilot_updates_{pilot_updates:06d}"
    if not archive.exists():
        for directory in (output, snapshot):
            summary = directory / "run_summary.json"
            if not summary.is_file():
                continue
            if directory == snapshot:
                summary = _snapshot_file(snapshot, "run_summary.json")
            report = json.loads(summary.read_text(encoding="utf-8"))
            if report.get("updates") != pilot_updates:
                continue
            checkpoint = load_checkpoint(directory / "latest.pt")
            if not _report_matches(report, checkpoint):
                continue
            _verify_checkpoint_config(checkpoint, baseline)
            _write_archive(archive, summary, {
                "format": "finllm-continuation-report-v1", "kind": "pilot",
                "pilot_config_sha256": sha256_file(config_path),
                "provenance": checkpoint["provenance"],
                # The pilot itself may have resumed after a disconnection;
                # the summary does not record its session start counter.
                "start_updates": None, "end_updates": pilot_updates,
                "checkpoint_sha256": sha256_file(directory / "latest.pt"),
            })
            break
    if not archive.exists():
        raise ValueError("A verified original pilot report is required before continuation; restore the completed 02a backup")
    archived_report = json.loads(_snapshot_file(archive, "run_summary.json").read_text(encoding="utf-8"))
    archived_metadata = json.loads(_snapshot_file(archive, "archive.json").read_text(encoding="utf-8"))
    selected = restore_seed_checkpoint(output, live)
    if selected is None:
        raise ValueError("Continuation requires an existing checkpoint; refusing random initialization")
    checkpoint = load_checkpoint(selected)
    _verify_checkpoint_config(checkpoint, baseline)
    if checkpoint["progress"]["updates"] < pilot_updates:
        raise ValueError("Complete the 02a pilot before continuing")
    if (archived_report.get("updates") != pilot_updates
            or archived_metadata.get("pilot_config_sha256") != sha256_file(config_path)
            or archived_metadata.get("provenance") != checkpoint["provenance"]):
        raise ValueError("Pilot report archive does not belong to this checkpoint/configuration")
    # Live checkpoint pairs contain no session timing report. Recover a matching
    # measured report from the independent archives when the mutable full run
    # snapshot was interrupted, instead of replacing known timing with zeros.
    summary = output / "run_summary.json"
    try:
        local_report = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        local_report = None
    if not isinstance(local_report, dict) or not _report_matches(local_report, checkpoint):
        for directory in (seed / "report_archive").iterdir():
            if not directory.is_dir():
                continue
            candidate = _snapshot_file(directory, "run_summary.json")
            metadata = json.loads(_snapshot_file(directory, "archive.json").read_text(encoding="utf-8"))
            report = json.loads(candidate.read_text(encoding="utf-8"))
            if metadata.get("provenance") == checkpoint["provenance"] and _report_matches(report, checkpoint):
                summary.write_bytes(candidate.read_bytes())
                (output / "report.json").write_bytes(candidate.read_bytes())
                break
    return {"project": str(project), "seed_drive": str(seed), "output_dir": str(output),
            "backup_dir": str(live), "pilot_config": str(config_path),
            "pilot_config_sha256": sha256_file(config_path), "baseline": baseline,
            "pilot_updates": pilot_updates, "probe_updates": probe_updates,
            "target_updates": target_updates, "pilot_archive": str(archive),
            "restored_updates": checkpoint["progress"]["updates"]}


def run_seed_segment(state, phase, runner):
    """Run a mandatory CLI resume, then archive its verified measured report."""
    if phase not in ("probe", "continue"):
        raise ValueError("Unknown continuation phase")
    _recover_report_archives(state["seed_drive"])
    config_path = Path(state["pilot_config"])
    if sha256_file(config_path) != state["pilot_config_sha256"]:
        raise ValueError("Frozen pilot configuration changed")
    checkpoint_path = restore_seed_checkpoint(state["output_dir"], state["backup_dir"])
    if checkpoint_path is None:
        raise ValueError("Continuation checkpoint is missing; refusing random initialization")
    before = load_checkpoint(checkpoint_path)
    _verify_checkpoint_config(before, state["baseline"])
    start = before["progress"]["updates"]
    minimum = state["pilot_updates"] if phase == "probe" else state["probe_updates"]
    if start < minimum:
        raise ValueError("Resume probe must complete before the longer segment")
    target = state["probe_updates"] if phase == "probe" else state["target_updates"]
    config = copy.deepcopy(state["baseline"])
    config["train"]["max_updates"] = target
    config["backup_dir"] = state["backup_dir"]
    runtime = Path(state["project"]) / "configs/runtime/seed_continuation.json"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Even a completed rerun passes through the trainer's strict provenance
    # checks. Never change batch size, precision, scheduler, or runtime silently.
    runner("train", "--config", runtime, "--resume", checkpoint_path)
    after = load_checkpoint(Path(state["output_dir"]) / "latest.pt")
    end = after["progress"]["updates"]
    if end != max(start, target):
        raise ValueError("Training ended before the requested continuation budget")
    if before["provenance"] != after["provenance"]:
        raise ValueError("Run identity changed during continuation")
    summary = Path(state["output_dir"]) / "run_summary.json"
    report = json.loads(summary.read_text(encoding="utf-8"))
    if not _report_matches(report, after):
        raise ValueError("Run summary does not match the completed checkpoint")
    # A no-op has no new measurements. Retain the existing archived reports and
    # do not label a previous session's elapsed time as time spent by this call.
    archive = None
    if end > start:
        archive = Path(state["seed_drive"]) / "report_archive" / f"updates_{start:06d}_to_{end:06d}"
        _write_archive(archive, summary, {
            "format": "finllm-continuation-report-v1", "kind": phase,
            "pilot_config_sha256": state["pilot_config_sha256"],
            "provenance": after["provenance"], "start_updates": start, "end_updates": end,
            "start_tokens": before["progress"]["tokens"], "end_tokens": after["progress"]["tokens"],
        })
    return {"phase": phase, "start_updates": start, "end_updates": end,
            "new_updates": end - start, "tokens": after["progress"]["tokens"],
            "report_archive": str(archive) if archive else None, "summary": report}


def backup_seed_continuation(state):
    """Publish a completed run snapshot only after target and summary checks."""
    output = Path(state["output_dir"])
    _recover_report_archives(state["seed_drive"])
    checkpoint = load_checkpoint(output / "latest.pt")
    report = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    if checkpoint["progress"]["updates"] < state["target_updates"] or not _report_matches(report, checkpoint):
        raise ValueError("Continuation has not completed; refusing final backup")
    # The original pilot measurements are independent of this mutable run snapshot.
    _snapshot_file(Path(state["pilot_archive"]), "run_summary.json")
    _snapshot_file(Path(state["pilot_archive"]), "archive.json")
    result = sync_snapshot(output, Path(state["seed_drive"]) / "run")
    if not result["verified"]:
        raise ValueError("Final run backup did not verify")
    return result
