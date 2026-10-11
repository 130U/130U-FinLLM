"""Versioned Colab handoff for additional data and a weights-only experiment.

Seed snapshots are inputs. New data, parent identity, reports and checkpoints
live under a separate expansion version; none are written back to seed_v1.
"""
from __future__ import annotations

import copy
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import torch

from .checkpoint import load_checkpoint, sha256_file, capture_rng_state, restore_rng_state
from .continuation import _snapshot_file, _write_archive, _recover_report_archives, _report_matches
from .data import TokenBlockDataset, write_json
from .model import FinLLM, ModelConfig
from .runtime import restore_seed_checkpoint, sync_snapshot
from .tokenizer import load_tokenizer, PAD_TOKEN
from .training import _settings, _validate, _data_digest, _device, _audit_formal_data, _pretrain_transition


def _version(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid artifact version")
    return value


def _inventory(directory):
    """Verify a complete snapshot without copying it or loading weights."""
    directory = Path(directory)
    marker = directory / "SNAPSHOT_COMPLETE.json"
    if (directory / "SNAPSHOT_INCOMPLETE.json").exists() or not marker.is_file():
        raise ValueError(f"A complete verified snapshot is required: {directory}")
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    if manifest.get("format") != "finllm-snapshot-v1" or not manifest.get("files"):
        raise ValueError("Invalid snapshot inventory")
    seen = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        target = (directory / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(directory.resolve()) or relative.as_posix() in seen:
            raise ValueError("Invalid snapshot file path")
        seen.add(relative.as_posix())
        _snapshot_file(directory, relative.as_posix())
    return manifest["files"]


def _restore_input(source, destination):
    entries = _inventory(source)
    destination = Path(destination)
    if not destination.exists() or (destination / "SNAPSHOT_INCOMPLETE.json").exists():
        return sync_snapshot(source, destination, working_copy=True)
    for entry in entries:
        path = destination / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Existing input differs from its frozen backup; preserved: {path}")
    return {"files": len(entries), "bytes": sum(x["bytes"] for x in entries),
            "verified": True, "reused": True}


def _backup_input(source, destination):
    destination = Path(destination)
    if (destination / "SNAPSHOT_COMPLETE.json").exists() and not (destination / "SNAPSHOT_INCOMPLETE.json").exists():
        entries = _inventory(destination)
        for entry in entries:
            path = Path(source) / entry["path"]
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise ValueError("Refusing to replace different frozen expansion inputs")
        return {"files": len(entries), "bytes": sum(x["bytes"] for x in entries), "verified": True, "reused": True}
    return sync_snapshot(source, destination)


def restore_expansion_materials(project, drive_root, version="expansion_v1", seed_version="seed_v1", *, restore_raw=True):
    project, drive_root = Path(project).resolve(), Path(drive_root).resolve()
    version, seed_version = _version(version), _version(seed_version)
    seed = drive_root / "seeds" / seed_version
    expansion = drive_root / "expansions" / version
    paths = {"corpora": project / "data/raw" / version / "corpora",
             "prepared": project / "data/raw" / version / "prepared",
             "processed": project / "data/processed" / version,
             "seed_prepared": project / "data/raw" / seed_version / "prepared",
             "seed_processed": project / "data/processed" / seed_version,
             "tokenizer": project / "tokenizer" / seed_version,
             "drive": expansion, "seed_drive": seed, "project": project}
    seed_inputs = [("processed", paths["seed_processed"]), ("tokenizer", paths["tokenizer"])]
    if restore_raw:
        seed_inputs.append(("raw", project / "data/raw" / seed_version))
    for name, destination in seed_inputs:
        _restore_input(seed / name, destination)
    if restore_raw:
        paths["task_pool"] = paths["seed_prepared"] / "frozen_task_exclusions.jsonl"
        if not paths["task_pool"].is_file():
            raise ValueError("Frozen seed task exclusions are missing")
    for name in (("corpora", "prepared", "processed") if restore_raw else ("processed",)):
        if (expansion / name).exists():
            _restore_input(expansion / name, paths[name])
    return paths


def expansion_build_arguments(paths):
    return {key: paths[key] for key in ("corpora", "task_pool", "seed_prepared", "seed_processed", "tokenizer", "prepared", "processed")}


def backup_expansion_data(paths):
    from .expansion import build_expansion_artifacts
    build_expansion_artifacts(**expansion_build_arguments(paths), validate_only=True)
    return {name: _backup_input(paths[name], paths["drive"] / name)
            for name in ("corpora", "prepared", "processed")}


def _freeze_parent(paths, minimum):
    parent_drive = paths["drive"] / "parent"
    local = paths["project"] / "checkpoints" / (paths["drive"].name + "_parent")
    if (parent_drive / "SNAPSHOT_COMPLETE.json").exists() and not (parent_drive / "SNAPSHOT_INCOMPLETE.json").exists():
        _restore_input(parent_drive, local)
    else:
        source = _snapshot_file(paths["seed_drive"] / "run", "latest.pt")
        _snapshot_file(paths["seed_drive"] / "run", "latest.pt.json")
        parent = load_checkpoint(source, allow_previous=False)
        if parent["provenance"]["mode"] != "formal" or parent["config"]["stage"] != "pretrain" or parent["progress"]["updates"] < minimum:
            raise ValueError("A completed formal seed pretraining parent is required")
        identity = {"format": "finllm-expansion-parent-v1", "source_seed": paths["seed_drive"].name,
                    "checkpoint_sha256": sha256_file(source), "updates": parent["progress"]["updates"],
                    "tokens": parent["progress"]["tokens"], "tokenizer_sha256": parent["tokenizer_hash"]}
        paths["drive"].mkdir(parents=True, exist_ok=True)
        pending = paths["drive"] / "parent_identity.json"
        if pending.exists() and json.loads(pending.read_text(encoding="utf-8")) != identity:
            raise ValueError("Frozen parent identity changed; existing experiment was preserved")
        if not pending.exists():
            write_json(pending, identity)
        with tempfile.TemporaryDirectory(prefix="finllm-parent-") as temporary:
            folder = Path(temporary)
            shutil.copy2(source, folder / "latest.pt")
            shutil.copy2(source.with_name("latest.pt.json"), folder / "latest.pt.json")
            write_json(folder / "parent.json", identity)
            sync_snapshot(folder, parent_drive)
        _restore_input(parent_drive, local)
    parent_path = local / "latest.pt"
    identity = json.loads((local / "parent.json").read_text(encoding="utf-8"))
    parent = load_checkpoint(parent_path, allow_previous=False)
    if sha256_file(parent_path) != identity["checkpoint_sha256"] or parent["progress"]["updates"] < minimum:
        raise ValueError("Frozen parent does not match the required seed checkpoint")
    if parent["tokenizer_hash"] != sha256_file(paths["tokenizer"] / "tokenizer.json"):
        raise ValueError("Parent and reused tokenizer differ")
    return parent_path, parent, identity


def _expected_provenance(config):
    """Use the trainer's explicit parent check before selecting a saved run."""
    settings = _settings(config)
    device = _device(settings["device"])
    tokenizer_hash = sha256_file(Path(config["tokenizer"]) / "tokenizer.json")
    _audit_formal_data(config, tokenizer_hash)
    semantic_keys = ("seed", "batch_size", "grad_accumulation", "seq_len", "learning_rate", "min_lr_ratio",
                     "warmup_updates", "weight_decay", "clip_grad", "schedule_updates", "beta")
    provenance = {"mode": "formal", "stage": "pretrain", "model_config": config["model"],
                  "tokenizer_sha256": tokenizer_hash, "data_sha256": _data_digest(config["data"]),
                  "validation_sha256": _data_digest(config["validation_data"]),
                  "training_settings": {key: settings[key] for key in semantic_keys},
                  "device_type": device.type, "amp": bool(settings["amp"] and device.type == "cuda"),
                  "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}
    _, provenance["parent_lineage"] = _pretrain_transition(config, ModelConfig(**config["model"]), tokenizer_hash, provenance)
    return provenance


def _verify_run(checkpoint, state):
    if (checkpoint.get("provenance") != state["expected_provenance"]
            or checkpoint.get("model_config") != state["config"]["model"]
            or checkpoint.get("tokenizer_hash") != state["expected_provenance"]["tokenizer_sha256"]
            or checkpoint.get("config", {}).get("stage") != "pretrain"
            or checkpoint.get("config", {}).get("mode") != "formal"):
        raise ValueError("Expansion checkpoint provenance or parent lineage differs from this experiment")


def _matches_expansion_report(report, checkpoint):
    return (_report_matches(report, checkpoint)
            and report.get("parent_lineage") == checkpoint["provenance"].get("parent_lineage"))


def _checked_archives(state):
    root = state["paths"]["drive"] / "report_archive"
    _recover_report_archives(state["paths"]["drive"])
    archives = []
    for directory in sorted(root.glob("*")):
        if not directory.is_dir():
            continue
        candidate = _snapshot_file(directory, "run_summary.json")
        report = json.loads(candidate.read_text(encoding="utf-8"))
        metadata = json.loads(_snapshot_file(directory, "archive.json").read_text(encoding="utf-8"))
        if (metadata.get("format") != "finllm-expansion-report-v1"
                or metadata.get("provenance") != state["expected_provenance"]
                or metadata.get("parent_sha256") != state["parent"]["checkpoint_sha256"]
                or report.get("parent_lineage") != state["expected_provenance"]["parent_lineage"]
                or metadata.get("end_updates") != report.get("updates")
                or metadata.get("end_tokens") != report.get("tokens")):
            raise ValueError("Expansion report archive provenance or parent lineage differs from this experiment")
        archives.append((candidate, report))
    if archives:
        state["prior_run_evidence"] = True
    return archives


def _select_run(state):
    config = state["config"]
    selected = restore_seed_checkpoint(config["output_dir"], config["backup_dir"])
    if selected is None:
        if ((state["paths"]["drive"] / "run").exists() or state["restored_updates"] > 0
                or state.get("prior_run_evidence") or (Path(config["output_dir"]) / "metrics.jsonl").exists()):
            raise ValueError("An expansion run existed but no verified local/live checkpoint remains; refusing fresh initialization")
    else:
        _verify_run(load_checkpoint(selected), state)
    return selected


def prepare_expansion_training(project, drive_root, version="expansion_v1", seed_version="seed_v1", *,
                               min_parent_updates=300, probe_updates=1, target_updates=610):
    if (not all(isinstance(value, int) and not isinstance(value, bool)
                for value in (probe_updates, target_updates, min_parent_updates))
            or not (1 <= probe_updates < target_updates) or min_parent_updates < 1):
        raise ValueError("Invalid expansion update budgets")
    paths = restore_expansion_materials(project, drive_root, version, seed_version, restore_raw=False)
    _inventory(paths["drive"] / "processed")
    report = json.loads((paths["processed"] / "expansion_summary.json").read_text(encoding="utf-8"))
    parent_path, parent, identity = _freeze_parent(paths, min_parent_updates)
    if (report.get("status") != "expansion_inputs_prepared_training_not_started"
            or (paths["processed"] / "INCOMPLETE").exists()
            or report["tokenizer_sha256"] != identity["tokenizer_sha256"]
            or report.get("tokenizer_vocabulary") != parent["model_config"]["vocab_size"]):
        raise ValueError("Expanded data does not use the frozen parent tokenizer")
    settings = _settings(parent["config"])
    settings.update(max_updates=target_updates, max_tokens=100_000_000, schedule_updates=6104,
                    warmup_updates=50, save_every=10, validation_every=10, max_epochs=None)
    config = {"mode": "formal", "stage": "pretrain", "model": parent["model_config"],
              "tokenizer": str(paths["tokenizer"]),
              "data": str(paths["processed"] / "general/manifest.json"),
              "validation_data": str(paths["processed"] / "general_validation/manifest.json"),
              "output_dir": str(paths["project"] / "runs" / (version + "_pretrain")),
              "backup_dir": str(Path(drive_root) / "live_checkpoints" / (version + "_pretrain")),
              "init_checkpoint": str(parent_path),
              "pretrain_transition": {"kind": "data_expansion", "parent_checkpoint": str(parent_path),
                                      "expected_parent_sha256": identity["checkpoint_sha256"],
                                      "min_parent_updates": min_parent_updates}, "train": settings}
    dataset = TokenBlockDataset(config["data"], settings["seq_len"])
    if len(dataset) < target_updates * settings["batch_size"] * settings["grad_accumulation"]:
        raise ValueError("Expanded general corpus is too small for this budget without replay; prepare more data")
    if dataset.manifest.get("tokenizer_sha256") != identity["tokenizer_sha256"] or dataset.manifest.get("mode") != "formal":
        raise ValueError("Expanded general manifest mode/tokenizer mismatch")
    if report["token_counts"]["general"] != dataset.manifest["token_count"]:
        raise ValueError("Expanded general token count differs from its completed preparation report")
    del dataset
    state = {"paths": paths, "config": config, "parent": identity,
             "probe_updates": probe_updates, "target_updates": target_updates, "restored_updates": 0,
             "expected_provenance": _expected_provenance(config), "continuation_succeeded": False}
    archives = _checked_archives(state)
    state["prior_run_evidence"] = bool(archives)
    output = Path(config["output_dir"])
    snapshot = paths["drive"] / "run"
    if not (output / "latest.pt").exists() and snapshot.exists():
        if (snapshot / "SNAPSHOT_COMPLETE.json").exists() and not (snapshot / "SNAPSHOT_INCOMPLETE.json").exists():
            try:
                sync_snapshot(snapshot, output, working_copy=True)
            except (OSError, ValueError):
                # A damaged full snapshot must not hide independent, verified
                # live checkpoints. Selection below refuses fresh initialization
                # if neither the local pair nor the live pair is recoverable.
                pass
    selected = _select_run(state)
    if selected is not None:
        checkpoint = load_checkpoint(selected)
        # A live-only restore should retain an archived measured summary if available.
        for candidate, archived_report in archives:
            if _matches_expansion_report(archived_report, checkpoint):
                output.mkdir(parents=True, exist_ok=True)
                (output / "run_summary.json").write_bytes(candidate.read_bytes())
                (output / "report.json").write_bytes(candidate.read_bytes())
                break
    state["restored_updates"] = load_checkpoint(selected)["progress"]["updates"] if selected else 0
    return state


def run_expansion_segment(state, phase, runner):
    if phase not in ("probe", "continue"):
        raise ValueError("Unknown expansion phase")
    state["continuation_succeeded"] = False
    config = copy.deepcopy(state["config"])
    archives = _checked_archives(state)
    retention = json.loads(_snapshot_file(state["paths"]["drive"] / "retention/before", "result.json").read_text(encoding="utf-8"))
    _validate_retention_result(retention, "before", _retention_inputs(state, "before")[2])
    selected = _select_run(state)
    before = load_checkpoint(selected) if selected else None
    start = before["progress"]["updates"] if before else 0
    start_tokens = before["progress"]["tokens"] if before else 0
    if phase == "continue" and start < state["probe_updates"]:
        raise ValueError("Complete the weights-transfer probe before longer training")
    target = state["probe_updates"] if phase == "probe" else state["target_updates"]
    measured_body = None
    if before is not None and start >= target:
        for candidate, archived_report in archives:
            if _matches_expansion_report(archived_report, before):
                measured_body = candidate.read_bytes()
                break
        if measured_body is None:
            prior_summary = Path(config["output_dir"]) / "run_summary.json"
            if prior_summary.is_file():
                candidate = prior_summary.read_bytes()
                if _matches_expansion_report(json.loads(candidate), before):
                    measured_body = candidate
    config["train"]["max_updates"] = target
    path = state["paths"]["project"] / "configs/runtime" / (state["paths"]["drive"].name + "_pretrain.json")
    write_json(path, config)
    arguments = ["train", "--config", path]
    if selected:
        arguments += ["--resume", selected]
    runner(*arguments)
    after = load_checkpoint(Path(config["output_dir"]) / "latest.pt")
    end = after["progress"]["updates"]
    summary = Path(config["output_dir"]) / "run_summary.json"
    report = json.loads(summary.read_text(encoding="utf-8"))
    _verify_run(after, state)
    if end != max(start, target) or not _matches_expansion_report(report, after):
        raise ValueError("Expanded-data segment or summary is incomplete")
    state["restored_updates"] = max(state["restored_updates"], end)
    if end == start and measured_body is not None:
        # An equivalent checkpoint can be restored under a different VM path.
        # Preserve the original measured report bytes after the CLI's strict
        # compatibility check; path relocation is not a new timing experiment.
        preserved = json.loads(measured_body)
        if not _matches_expansion_report(preserved, after):
            raise ValueError("Saved measurement no longer matches the restored checkpoint")
        if summary.read_bytes() != measured_body:
            summary.write_bytes(measured_body)
            (Path(config["output_dir"]) / "report.json").write_bytes(measured_body)
        report = preserved
    archive = None
    if end > start:
        archive = state["paths"]["drive"] / "report_archive" / f"updates_{start:06d}_to_{end:06d}"
        _write_archive(archive, summary, {"format": "finllm-expansion-report-v1", "kind": phase,
            "parent_sha256": state["parent"]["checkpoint_sha256"], "provenance": after["provenance"],
            "start_updates": start, "end_updates": end, "start_tokens": start_tokens,
            "end_tokens": after["progress"]["tokens"]})
    if phase == "continue":
        state["continuation_succeeded"] = True
    return {"phase": phase, "start_updates": start, "end_updates": end, "new_updates": end-start,
            "new_experiment_tokens": report["tokens"], "parent_tokens": state["parent"]["tokens"],
            "model_path_total_tokens": state["parent"]["tokens"] + report["tokens"],
            "report_archive": str(archive) if archive else None, "summary": report}


def _retention_inputs(state, phase):
    config = state["config"]
    checkpoint_path = Path(config["init_checkpoint"]) if phase == "before" else Path(config["output_dir"]) / "latest.pt"
    checkpoint = load_checkpoint(checkpoint_path, allow_previous=False)
    if phase == "after" and checkpoint["progress"]["updates"] < state["target_updates"]:
        raise ValueError("Retention-after evaluation requires completed training")
    if phase == "after":
        _verify_run(checkpoint, state)
    elif sha256_file(checkpoint_path) != state["parent"]["checkpoint_sha256"]:
        raise ValueError("Retention-before parent checkpoint differs from the frozen experiment")
    data = state["paths"]["seed_processed"] / "general_validation/manifest.json"
    expansion = json.loads((state["paths"]["processed"] / "expansion_summary.json").read_text(encoding="utf-8"))
    if expansion["retention_validation"]["general"]["sha256"] != sha256_file(data):
        raise ValueError("Retention manifest differs from the frozen expansion preparation boundary")
    settings = _settings(config)
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


def _validate_retention_result(result, phase, key):
    validation = result.get("validation", {})
    if (result.get("phase") != phase or result.get("inputs") != key
            or not isinstance(validation.get("loss"), (int, float))
            or isinstance(validation.get("loss"), bool)
            or not math.isfinite(validation["loss"])
            or not isinstance(validation.get("supervised_tokens"), (int, float))
            or isinstance(validation.get("supervised_tokens"), bool)
            or validation["supervised_tokens"] <= 0):
        raise ValueError("Existing retention measurement has a different identity or invalid metrics")


def _retention_bytes(state, phase, key):
    """Read a real saved measurement; retry snapshot IO without reevaluation."""
    folder = state["paths"]["drive"] / "retention"
    destination = folder / (phase + ".json")
    snapshot = folder / phase
    pending = folder / (phase + ".pending.json")
    complete = (snapshot / "SNAPSHOT_COMPLETE.json").is_file() and not (snapshot / "SNAPSHOT_INCOMPLETE.json").exists()
    if complete:
        body = _snapshot_file(snapshot, "result.json").read_bytes()
        if destination.exists() and destination.read_bytes() != body:
            raise ValueError("Saved retention report differs from its verified snapshot")
        _validate_retention_result(json.loads(body), phase, key)
        if not destination.exists():
            destination.write_bytes(body)
        if pending.exists():
            envelope = json.loads(pending.read_text(encoding="utf-8"))
            if (envelope.get("format") != "finllm-pending-retention-v1"
                    or envelope.get("sha256") != hashlib.sha256(body).hexdigest()
                    or base64.b64decode(envelope["report_bytes"], validate=True) != body):
                raise ValueError("Pending retention report differs from the verified measurement")
            pending.unlink()
        return body
    if pending.exists():
        envelope = json.loads(pending.read_text(encoding="utf-8"))
        body = base64.b64decode(envelope["report_bytes"], validate=True)
        if (envelope.get("format") != "finllm-pending-retention-v1"
                or hashlib.sha256(body).hexdigest() != envelope.get("sha256")):
            raise ValueError("Pending retention measurement integrity check failed")
        if destination.exists() and destination.read_bytes() != body:
            raise ValueError("Saved retention report differs from its pending measurement")
    elif destination.exists():
        # Older unfinished snapshots saved the genuine report before copying.
        body = destination.read_bytes()
    elif snapshot.exists():
        raise ValueError("Retention snapshot is incomplete and no saved measurement remains; refusing a fabricated retry")
    else:
        return None
    _validate_retention_result(json.loads(body), phase, key)
    _save_retention_snapshot(folder, phase, body)
    return body


def _save_retention_snapshot(folder, phase, body):
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / (phase + ".json")
    pending = folder / (phase + ".pending.json")
    envelope = {"format": "finllm-pending-retention-v1", "sha256": hashlib.sha256(body).hexdigest(),
                "report_bytes": base64.b64encode(body).decode("ascii")}
    if not pending.exists():
        temporary = pending.with_suffix(".tmp")
        write_json(temporary, envelope)
        os.replace(temporary, pending)
    if not destination.exists():
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(body)
        os.replace(temporary, destination)
    elif destination.read_bytes() != body:
        raise ValueError("Refusing to replace a saved retention measurement")
    with tempfile.TemporaryDirectory(prefix="finllm-retention-") as temporary:
        path = Path(temporary)
        (path / "result.json").write_bytes(body)
        sync_snapshot(path, folder / phase)
    if _snapshot_file(folder / phase, "result.json").read_bytes() != body:
        raise ValueError("Retention snapshot differs from its saved measurement")
    pending.unlink()


def retention_check(state, phase):
    """Compare the same frozen seed validation positions before/after expansion."""
    if phase not in ("before", "after"):
        raise ValueError("Unknown retention phase")
    checkpoint, data, key, device, amp = _retention_inputs(state, phase)
    saved = _retention_bytes(state, phase, key)
    if saved is not None:
        return json.loads(saved)
    config = state["config"]
    settings = _settings(config)
    tokenizer = load_tokenizer(config["tokenizer"])
    dataset = TokenBlockDataset(data, settings["seq_len"])
    if dataset.manifest.get("tokenizer_sha256") != checkpoint["tokenizer_hash"]:
        raise ValueError("Retention data/tokenizer mismatch")
    rng = capture_rng_state()
    try:
        model = FinLLM(ModelConfig(**checkpoint["model_config"])).float().to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        validation = _validate(model, dataset, "pretrain", settings["batch_size"],
                               tokenizer.token_to_id(PAD_TOKEN), [], device,
                               amp, 50)
        if not math.isfinite(validation["loss"]):
            raise ValueError("Non-finite retention loss")
        result = {"phase": phase, "inputs": key, "validation": validation,
                  "scope": "Frozen seed general-validation subset; not a financial stance score"}
    finally:
        restore_rng_state(rng)
    folder = state["paths"]["drive"] / "retention"
    _save_retention_snapshot(folder, phase, (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return result


def backup_expansion_training(state):
    if not state.get("continuation_succeeded"):
        raise ValueError("Expanded-data continuation did not complete successfully; refusing final backup")
    _checked_archives(state)
    output = Path(state["config"]["output_dir"])
    checkpoint = load_checkpoint(output / "latest.pt")
    report = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    _verify_run(checkpoint, state)
    if checkpoint["progress"]["updates"] < state["target_updates"] or not _matches_expansion_report(report, checkpoint):
        raise ValueError("Expanded-data training has not completed; refusing final backup")
    for phase in ("before", "after"):
        result = json.loads(_snapshot_file(state["paths"]["drive"] / "retention" / phase, "result.json").read_text(encoding="utf-8"))
        _validate_retention_result(result, phase, _retention_inputs(state, phase)[2])
    return sync_snapshot(output, state["paths"]["drive"] / "run")
