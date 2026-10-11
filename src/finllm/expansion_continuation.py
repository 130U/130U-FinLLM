"""Strict continuation of a completed expanded-data pretraining checkpoint.

The existing expansion experiment is resumed without a weights-only reset. Its
completed predecessor and original retention measurements are frozen before any
newer live checkpoint is selected. New comparisons have their own namespace.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, sha256_file
from .continuation import _recover_report_archives, _snapshot_file, _write_archive
from .data import TokenBlockDataset, write_json
from .expansion_runtime import (
    _inventory, _matches_expansion_report, _retention_bytes, _retention_inputs,
    _save_retention_snapshot, _validate_retention_result, _verify_run, _version,
    prepare_expansion_training, run_expansion_segment,
)
from .model import FinLLM, ModelConfig
from .runtime import restore_seed_checkpoint, sync_snapshot
from .tokenizer import load_tokenizer, PAD_TOKEN
from .training import _data_digest, _device, _settings, _validate


def _settings_match(checkpoint, config):
    actual, expected = _settings(checkpoint["config"]), _settings(config)
    actual.pop("max_updates")
    expected.pop("max_updates")
    if actual != expected:
        raise ValueError("Strict expansion continuation may change only max_updates; training settings differ")


def _runtime_match(checkpoint):
    settings = _settings(checkpoint["config"])
    device = _device(settings["device"])
    provenance = checkpoint["provenance"]
    if (provenance.get("device_type") != device.type
            or provenance.get("amp") != bool(settings["amp"] and device.type == "cuda")
            or provenance.get("torch_version") != str(torch.__version__)
            or provenance.get("cuda_version") != torch.version.cuda):
        raise ValueError("Continuation runtime differs from the completed checkpoint; preserve its PyTorch/CUDA/precision environment")


def _measured_report(report):
    for key in ("elapsed_seconds", "tokens_per_second"):
        value = report.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            raise ValueError("A genuine positive-time measured training report is required")


def _baseline(directory, completed_updates):
    _inventory(directory)
    checkpoint_path = _snapshot_file(directory, "latest.pt")
    _snapshot_file(directory, "latest.pt.json")
    checkpoint = load_checkpoint(checkpoint_path, allow_previous=False)
    report_path = _snapshot_file(directory, "run_summary.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (checkpoint["config"].get("mode") != "formal"
            or checkpoint["config"].get("stage") != "pretrain"
            or checkpoint["progress"]["updates"] != completed_updates
            or _settings(checkpoint["config"])["max_updates"] != completed_updates
            or not checkpoint["provenance"].get("parent_lineage")
            or not _matches_expansion_report(report, checkpoint)):
        raise ValueError("A verified completed 02c checkpoint and matching measured report are required")
    _measured_report(report)
    _runtime_match(checkpoint)
    return checkpoint_path, checkpoint, report_path, report


def _freeze_predecessor(expansion_drive, continuation_drive, completed_updates, seed_version):
    """Complete this immutable archive before shared latest/live files can advance."""
    archive = continuation_drive / "baseline_run"
    pending = continuation_drive / "predecessor_pending.json"
    if (archive / "SNAPSHOT_COMPLETE.json").exists() and not (archive / "SNAPSHOT_INCOMPLETE.json").exists():
        checkpoint_path, checkpoint, report_path, report = _baseline(archive, completed_updates)
        metadata = json.loads(_snapshot_file(archive, "predecessor.json").read_text(encoding="utf-8"))
        if (metadata.get("format") != "finllm-expansion-continuation-predecessor-v1"
                or metadata.get("completed_updates") != completed_updates
                or metadata.get("source_version") != expansion_drive.name
                or metadata.get("seed_version") != seed_version
                or metadata.get("checkpoint_sha256") != sha256_file(checkpoint_path)
                or metadata.get("report_sha256") != sha256_file(report_path)
                or metadata.get("tokens") != checkpoint["progress"]["tokens"]
                or metadata.get("provenance") != checkpoint["provenance"]
                or metadata.get("training_settings") != _settings(checkpoint["config"])):
            raise ValueError("Immutable predecessor archive identity differs from this continuation")
        for phase in ("before", "after"):
            _snapshot_file(archive, f"original_retention_{phase}.json")
        if pending.exists():
            saved = json.loads(pending.read_text(encoding="utf-8"))
            if saved.get("identity") != metadata:
                raise ValueError("Pending predecessor identity differs from its completed archive")
            pending.unlink()
        return archive, checkpoint, metadata

    source = expansion_drive / "run"
    checkpoint_path, checkpoint, report_path, report = _baseline(source, completed_updates)
    retained = {}
    for phase in ("before", "after"):
        body = _snapshot_file(expansion_drive / "retention" / phase, "result.json").read_bytes()
        result = json.loads(body)
        _validate_retention_result(result, phase, result.get("inputs"))
        retained[phase] = body
    after = json.loads(retained["after"])
    before = json.loads(retained["before"])
    if (after["inputs"].get("checkpoint_sha256") != sha256_file(checkpoint_path)
            or after["inputs"].get("tokenizer_sha256") != checkpoint["tokenizer_hash"]
            or before["inputs"].get("checkpoint_sha256") != checkpoint["provenance"]["parent_lineage"]["parent_checkpoint_sha256"]
            or {k: v for k, v in before["inputs"].items() if k != "checkpoint_sha256"}
            != {k: v for k, v in after["inputs"].items() if k != "checkpoint_sha256"}):
        raise ValueError("Original 02c retention reports do not match the completed checkpoint")
    metadata = {"format": "finllm-expansion-continuation-predecessor-v1",
                "source_version": expansion_drive.name, "seed_version": seed_version,
                "completed_updates": completed_updates, "tokens": checkpoint["progress"]["tokens"],
                "checkpoint_sha256": sha256_file(checkpoint_path), "report_sha256": sha256_file(report_path),
                "provenance": checkpoint["provenance"], "training_settings": _settings(checkpoint["config"])}
    pending_identity = {"identity": metadata, "run_files": _inventory(source),
                        "retention_sha256": {phase: hashlib.sha256(body).hexdigest()
                                              for phase, body in retained.items()}}
    if pending.exists():
        if json.loads(pending.read_text(encoding="utf-8")) != pending_identity:
            raise ValueError("Interrupted predecessor archive has different source bytes; original archive was preserved")
    else:
        continuation_drive.mkdir(parents=True, exist_ok=True)
        temporary = pending.with_suffix(".tmp")
        write_json(temporary, pending_identity)
        os.replace(temporary, pending)
    with tempfile.TemporaryDirectory(prefix="finllm-expansion-baseline-") as temporary:
        folder = Path(temporary)
        for entry in _inventory(source):
            target = folder / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / entry["path"], target)
        for phase, body in retained.items():
            (folder / f"original_retention_{phase}.json").write_bytes(body)
        (folder / "predecessor.json").write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")
        sync_snapshot(folder, archive)
    # The complete branch checks every archived byte and its independent identity.
    return _freeze_predecessor(expansion_drive, continuation_drive, completed_updates, seed_version)


def _retention_state(state):
    result = dict(state)
    result["paths"] = dict(state["paths"], drive=state["continuation_drive"])
    return result


def _continuation_retention_inputs(state, phase):
    checkpoint_path = (Path(state["predecessor_archive"]) / "latest.pt"
                       if phase == "before" else Path(state["config"]["output_dir"]) / "latest.pt")
    checkpoint = load_checkpoint(checkpoint_path, allow_previous=False)
    _verify_run(checkpoint, state)
    _settings_match(checkpoint, state["config"])
    required = state["completed_updates"] if phase == "before" else state["target_updates"]
    if checkpoint["progress"]["updates"] != required:
        raise ValueError("Retention measurement requires the exact completed continuation boundary")
    data = state["paths"]["seed_processed"] / "general_validation/manifest.json"
    expansion = json.loads((state["paths"]["processed"] / "expansion_summary.json").read_text(encoding="utf-8"))
    if expansion["retention_validation"]["general"]["sha256"] != sha256_file(data):
        raise ValueError("Retention manifest differs from the frozen expansion preparation boundary")
    settings = _settings(state["config"])
    device = _device(settings["device"])
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    amp = bool(settings["amp"] and device.type == "cuda")
    key = {"checkpoint_sha256": sha256_file(checkpoint_path), "data_sha256": _data_digest(data),
           "tokenizer_sha256": checkpoint["tokenizer_hash"], "batch_size": settings["batch_size"],
           "seq_len": settings["seq_len"], "max_batches": 50, "requested_device": settings["device"],
           "device": str(device), "device_type": device.type, "amp": amp,
           "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}
    return checkpoint, data, key, device, amp


def _check_baseline_retention(state):
    archive = Path(state["predecessor_archive"])
    before = json.loads(_snapshot_file(archive, "original_retention_before.json").read_text(encoding="utf-8"))
    after = json.loads(_snapshot_file(archive, "original_retention_after.json").read_text(encoding="utf-8"))
    _validate_retention_result(before, "before", _retention_inputs(state, "before")[2])
    _validate_retention_result(after, "after", _continuation_retention_inputs(state, "before")[2])


def _check_current(state):
    selected = restore_seed_checkpoint(state["config"]["output_dir"], state["config"]["backup_dir"])
    if selected is None:
        raise ValueError("Continuation requires an existing expansion checkpoint; refusing fresh initialization")
    checkpoint = load_checkpoint(selected)
    _verify_run(checkpoint, state)
    _settings_match(checkpoint, state["config"])
    progress = checkpoint["progress"]
    if not state["completed_updates"] <= progress["updates"] <= state["target_updates"]:
        raise ValueError("Existing checkpoint is outside the requested continuation boundaries")
    settings = _settings(state["config"])
    data = TokenBlockDataset(state["config"]["data"], settings["seq_len"])
    remaining_blocks = ((state["target_updates"] - progress["updates"])
                        * settings["batch_size"] * settings["grad_accumulation"])
    if progress["epoch"] != 0 or progress["cursor"] + remaining_blocks > len(data):
        raise ValueError("Remaining expanded general data is insufficient without replay; preserve the checkpoint and prepare more data")
    return selected, checkpoint


def _checked_continuation_archives(state):
    _recover_report_archives(state["continuation_drive"])
    root = state["continuation_drive"] / "report_archive"
    for directory in root.glob("*"):
        if not directory.is_dir():
            continue
        metadata = json.loads(_snapshot_file(directory, "archive.json").read_text(encoding="utf-8"))
        report = json.loads(_snapshot_file(directory, "run_summary.json").read_text(encoding="utf-8"))
        if (metadata.get("format") != "finllm-expansion-continuation-report-v1"
                or metadata.get("provenance") != state["expected_provenance"]
                or metadata.get("predecessor_sha256") != state["predecessor"]["checkpoint_sha256"]
                or metadata.get("completed_updates") != state["completed_updates"]
                or metadata.get("end_updates") != report.get("updates")
                or metadata.get("end_tokens") != report.get("tokens")
                or report.get("parent_lineage") != state["expected_provenance"]["parent_lineage"]):
            raise ValueError("Continuation report archive differs from its immutable predecessor")


def prepare_expansion_continuation(project, drive_root, version="expansion_v1", seed_version="seed_v1", *,
                                   completed_updates=610, probe_updates=611, target_updates=1220,
                                   min_parent_updates=300):
    """Freeze a verified completed 02c result, then restore its strict live resume."""
    values = (completed_updates, probe_updates, target_updates, min_parent_updates)
    if (not all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            or not 1 <= completed_updates < probe_updates < target_updates or min_parent_updates < 1):
        raise ValueError("Invalid expansion continuation update budgets")
    version, seed_version = _version(version), _version(seed_version)
    expansion_drive = Path(drive_root).resolve() / "expansions" / version
    continuation_drive = expansion_drive / "continuations" / f"updates_{completed_updates:06d}_to_{target_updates:06d}"
    archive, predecessor, metadata = _freeze_predecessor(expansion_drive, continuation_drive, completed_updates, seed_version)
    state = prepare_expansion_training(project, drive_root, version, seed_version,
                                       min_parent_updates=min_parent_updates,
                                       probe_updates=probe_updates, target_updates=target_updates)
    _verify_run(predecessor, state)
    _settings_match(predecessor, state["config"])
    state.update(completed_updates=completed_updates, predecessor_archive=str(archive),
                 predecessor=metadata, predecessor_tokens=predecessor["progress"]["tokens"],
                 continuation_drive=continuation_drive, before_retention_succeeded=False,
                 after_retention_succeeded=False, continuation_succeeded=False)
    _check_baseline_retention(state)
    _checked_continuation_archives(state)
    _, current = _check_current(state)
    state["restored_updates"] = current["progress"]["updates"]
    return state


def retention_continuation_check(state, phase):
    """Compare the archived 610 model and exact 1220 model on frozen seed text."""
    if phase not in ("before", "after"):
        raise ValueError("Unknown expansion continuation retention phase")
    flag = phase + "_retention_succeeded"
    state[flag] = False
    if phase == "after" and not state.get("continuation_succeeded"):
        raise ValueError("Continuation training did not complete; retention-after is blocked")
    checkpoint, data, key, device, amp = _continuation_retention_inputs(state, phase)
    saved = _retention_bytes(_retention_state(state), phase, key)
    if saved is not None:
        state[flag] = True
        return json.loads(saved)
    settings = _settings(state["config"])
    tokenizer = load_tokenizer(state["config"]["tokenizer"])
    dataset = TokenBlockDataset(data, settings["seq_len"])
    if dataset.manifest.get("tokenizer_sha256") != checkpoint["tokenizer_hash"]:
        raise ValueError("Retention data/tokenizer mismatch")
    rng = capture_rng_state()
    try:
        model = FinLLM(ModelConfig(**checkpoint["model_config"])).float().to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        validation = _validate(model, dataset, "pretrain", settings["batch_size"],
                               tokenizer.token_to_id(PAD_TOKEN), [], device, amp, 50)
        result = {"phase": phase, "inputs": key, "validation": validation,
                  "completed_updates": checkpoint["progress"]["updates"],
                  "scope": "Frozen seed general-validation subset; not a financial stance score"}
        _validate_retention_result(result, phase, key)
    finally:
        restore_rng_state(rng)
    body = (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _save_retention_snapshot(state["continuation_drive"] / "retention", phase, body)
    state[flag] = True
    return result


def run_expansion_continuation(state, phase, runner):
    """Resume mandatory optimizer/RNG/cursor state; only the update cap changes."""
    if phase not in ("probe", "continue"):
        raise ValueError("Unknown expansion continuation phase")
    state["continuation_succeeded"] = False
    state["after_retention_succeeded"] = False
    _checked_continuation_archives(state)
    _check_baseline_retention(state)
    retained = json.loads(_snapshot_file(state["continuation_drive"] / "retention/before", "result.json").read_text(encoding="utf-8"))
    _validate_retention_result(retained, "before", _continuation_retention_inputs(state, "before")[2])
    state["before_retention_succeeded"] = True
    selected, current = _check_current(state)
    if phase == "continue" and current["progress"]["updates"] < state["probe_updates"]:
        raise ValueError("Complete the strict resume probe before longer continuation")
    target = state["probe_updates"] if phase == "probe" else state["target_updates"]
    if current["progress"]["updates"] >= target:
        summary = Path(state["config"]["output_dir"]) / "run_summary.json"
        if not summary.is_file() or not _matches_expansion_report(json.loads(summary.read_text(encoding="utf-8")), current):
            raise ValueError("Completed continuation has no matching measured report; refusing a zero-time replacement")
        _measured_report(json.loads(summary.read_text(encoding="utf-8")))

    def resume_only(*arguments):
        if "--resume" not in arguments:
            raise ValueError("Expansion continuation requires an explicit strict resume")
        # Existing expansion helper writes the config before invoking the CLI.
        # Remove the old weights-transfer entry even though CLI --resume takes
        # precedence: this notebook can never initialize a fresh model.
        runtime_path = Path(arguments[2])
        config = json.loads(runtime_path.read_text(encoding="utf-8"))
        config.pop("init_checkpoint", None)
        runtime_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return runner(*arguments)

    result = run_expansion_segment(state, phase, resume_only)
    state["continuation_succeeded"] = False
    _measured_report(result["summary"])
    output = Path(state["config"]["output_dir"])
    _, after = _check_current(state)
    if result["new_updates"]:
        archive = state["continuation_drive"] / "report_archive" / f"updates_{result['start_updates']:06d}_to_{result['end_updates']:06d}"
        _write_archive(archive, output / "run_summary.json", {
            "format": "finllm-expansion-continuation-report-v1", "kind": phase,
            "completed_updates": state["completed_updates"], "predecessor_sha256": state["predecessor"]["checkpoint_sha256"],
            "provenance": after["provenance"], "start_updates": result["start_updates"],
            "end_updates": result["end_updates"], "start_tokens": current["progress"]["tokens"],
            "end_tokens": after["progress"]["tokens"],
        })
        result["report_archive"] = str(archive)
    result.update(segment_updates=after["progress"]["updates"] - state["completed_updates"],
                  segment_tokens=after["progress"]["tokens"] - state["predecessor_tokens"],
                  new_tokens=after["progress"]["tokens"] - current["progress"]["tokens"],
                  predecessor_tokens=state["predecessor_tokens"], predecessor_archive=state["predecessor_archive"])
    if phase == "continue":
        state["continuation_succeeded"] = True
    return result


def backup_expansion_continuation(state):
    """Certify final run only after strict resume and both scoped comparisons."""
    if not all(state.get(key) for key in ("continuation_succeeded", "before_retention_succeeded", "after_retention_succeeded")):
        raise ValueError("Continuation training/retention did not complete successfully; refusing final backup")
    _baseline(Path(state["predecessor_archive"]), state["completed_updates"])
    _check_baseline_retention(state)
    _checked_continuation_archives(state)
    _, checkpoint = _check_current(state)
    output = Path(state["config"]["output_dir"])
    report = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    _measured_report(report)
    if checkpoint["progress"]["updates"] != state["target_updates"] or not _matches_expansion_report(report, checkpoint):
        raise ValueError("Expansion continuation target or matching report is incomplete")
    for phase in ("before", "after"):
        result = json.loads(_snapshot_file(state["continuation_drive"] / "retention" / phase, "result.json").read_text(encoding="utf-8"))
        _validate_retention_result(result, phase, _continuation_retention_inputs(state, phase)[2])
    result = sync_snapshot(output, state["paths"]["drive"] / "run")
    if not result.get("verified"):
        raise ValueError("Final expansion continuation snapshot did not verify")
    return result
