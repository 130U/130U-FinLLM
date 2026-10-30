"""Isolated entity-stance SFT pilot from the completed domain checkpoint.

Task data are reviewed headline labels and literal entities, not audited
exchange securities.  Source content remains in private Drive snapshots.
"""
from __future__ import annotations

import copy
import json
import math
import re
import tempfile
import zipfile
from pathlib import Path

from .checkpoint import load_checkpoint, sha256_file
from .continuation import _recover_report_archives, _snapshot_file, _write_archive
from .data import DuplicateIndex, read_jsonl, text_hash, write_json
from .expansion_runtime import _inventory, _restore_input, _version, restore_expansion_materials
from .runtime import restore_seed_checkpoint, sync_snapshot
from .sft_preflight import audit_domain_parent
from .tokenizer import load_tokenizer, encode_prompt
from .training import _audit_formal_data, _data_digest

SPLITS = ("train", "development", "calibration_temperature", "calibration_threshold", "test")
LABELS = {"bearish", "neutral", "bullish"}


def _audit_training_exclusions(paths, task_local, domain_drive):
    """Prove each actual task text was excluded by every upstream data pack.

    Independent builds need not serialize the exclusion lists identically.
    Read only the four verified lists, bind each to the exact pretrained
    manifest/checkpoint and require exact normalized membership for all task
    splits. Missing evidence never permits training.
    """
    seed_parent_dir = paths["drive"] / "parent"
    expansion_parent_dir = Path(domain_drive) / "parent"
    seed_checkpoint_path = _snapshot_file(seed_parent_dir, "latest.pt")
    _snapshot_file(seed_parent_dir, "latest.pt.json")
    expansion_checkpoint_path = _snapshot_file(expansion_parent_dir, "latest.pt")
    _snapshot_file(expansion_parent_dir, "latest.pt.json")
    seed_checkpoint = load_checkpoint(seed_checkpoint_path, allow_previous=False)
    expansion_checkpoint = load_checkpoint(expansion_checkpoint_path, allow_previous=False)
    seed_train = paths["seed_processed"] / "general/manifest.json"
    seed_validation = paths["seed_processed"] / "general_validation/manifest.json"
    expansion_train = paths["processed"] / "general/manifest.json"
    expansion_validation = paths["processed"] / "general_validation/manifest.json"
    domain_mix = paths["processed"] / "domain_mix/manifest.json"
    domain_validation = paths["processed"] / "finance_validation/manifest.json"
    lineage = expansion_checkpoint.get("provenance", {}).get("parent_lineage", {})
    if (lineage.get("kind") != "data_expansion"
            or lineage.get("parent_checkpoint_sha256") != sha256_file(seed_checkpoint_path)
            or seed_checkpoint.get("provenance", {}).get("data_sha256") != _data_digest(seed_train)
            or seed_checkpoint.get("provenance", {}).get("validation_sha256") != _data_digest(seed_validation)
            or expansion_checkpoint.get("provenance", {}).get("data_sha256") != _data_digest(expansion_train)
            or expansion_checkpoint.get("provenance", {}).get("validation_sha256") != _data_digest(expansion_validation)):
        raise ValueError("Pretraining checkpoints do not match the actual seed/expansion data manifests")
    if (json.loads(domain_mix.read_text(encoding="utf-8")).get("eval_exclusion_sha256")
            != json.loads(expansion_train.read_text(encoding="utf-8")).get("eval_exclusion_sha256")
            or json.loads(domain_validation.read_text(encoding="utf-8")).get("eval_exclusion_sha256")
            != json.loads(expansion_validation.read_text(encoding="utf-8")).get("eval_exclusion_sha256")):
        raise ValueError("Domain-stage exclusion files differ from the expansion-stage lineage")
    task_hashes = set()
    task_count = 0
    for split in SPLITS:
        for row in read_jsonl(Path(task_local) / (split + ".jsonl")):
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Task exclusion check requires nonempty task texts")
            task_hashes.add(text_hash(text))
            task_count += 1
    if not task_hashes:
        raise ValueError("Task exclusion check requires nonempty task splits")
    sources = (
        ("seed_train", paths["seed_drive"] / "raw", "prepared/frozen_exclusions.jsonl", seed_train),
        ("seed_validation", paths["seed_drive"] / "raw", "prepared/frozen_task_exclusions.jsonl", seed_validation),
        ("expansion_train", paths["drive"] / "prepared", "frozen_exclusions.jsonl", expansion_train),
        ("expansion_validation", paths["drive"] / "prepared", "frozen_prior_exclusions.jsonl", expansion_validation),
    )
    result = {}
    for name, folder, filename, manifest_path in sources:
        exclusion_path = _snapshot_file(folder, filename)
        actual_hash = sha256_file(exclusion_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("eval_exclusion_sha256") != actual_hash:
            raise ValueError(f"{name} exclusion file does not match the pretrained-data manifest")
        missing = set(task_hashes)
        for row in read_jsonl(exclusion_path):
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid text in {name} exclusion file")
            missing.discard(text_hash(text))
        if missing:
            raise ValueError(f"{name} exclusion file does not contain {len(missing)} distinct task titles")
        result[name] = {"sha256": actual_hash, "task_rows": task_count,
                        "distinct_task_texts": len(task_hashes), "missing": 0}
    return result


def install_task_zip(zip_path, drive_root, expected_sha256, task_version="entity_v1"):
    """Verify and snapshot a user-uploaded task ZIP without extracting paths."""
    task_version = _version(task_version)
    zip_path = Path(zip_path)
    if sha256_file(zip_path) != expected_sha256:
        raise ValueError("Uploaded task ZIP differs from the prepared local artifact")
    destination = Path(drive_root) / "tasks" / task_version
    with zipfile.ZipFile(zip_path) as archive, tempfile.TemporaryDirectory(prefix="finllm-task-upload-") as temp:
        members = archive.infolist()
        names = [item.filename for item in members]
        if (len(names) != len(set(names)) or "task_manifest.json" not in names
                or any(item.is_dir() or Path(item.filename).name != item.filename or item.file_size > 10_000_000 for item in members)
                or sum(item.file_size for item in members) > 50_000_000):
            raise ValueError("Task ZIP has invalid, duplicate or oversized entries")
        manifest = json.loads(archive.read("task_manifest.json"))
        if manifest.get("format") != "130u-entity-pilot-v1" or not isinstance(manifest.get("files"), dict):
            raise ValueError("Unrecognized task package")
        if set(names) != set(manifest["files"]) | {"task_manifest.json"}:
            raise ValueError("Task ZIP entries and manifest differ")
        for name in names:
            raw = archive.read(name)
            if name != "task_manifest.json":
                from hashlib import sha256
                record = manifest["files"][name]
                if record.get("bytes") != len(raw) or record.get("sha256") != sha256(raw).hexdigest():
                    raise ValueError("Task ZIP file content differs from its manifest")
            (Path(temp) / name).write_bytes(raw)
        if destination.exists():
            _inventory(destination)
            for name in names:
                if (destination / name).read_bytes() != (Path(temp) / name).read_bytes():
                    raise ValueError("Existing task backup differs; choose a new task version")
            return {"verified": True, "reused": True, "files": len(names)}
        return sync_snapshot(temp, destination)


def audit_task_inputs(folder, tokenizer, seq_len=512):
    """Validate human labels, literal entities and isolation of all five splits."""
    folder = Path(folder)
    tok = load_tokenizer(tokenizer)
    ids, groups, exact_entities = set(), {}, {}
    duplicates = DuplicateIndex()
    counts = {}
    for split in SPLITS:
        path = folder / (split + ".jsonl")
        rows = list(read_jsonl(path))
        if not rows:
            raise ValueError(f"Missing or empty task split: {split}")
        counts[split] = len(rows)
        labels = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Task row must be an object")
            row_id, text, target = row.get("id"), row.get("text"), row.get("target")
            if not isinstance(row_id, str) or not row_id or row_id in ids:
                raise ValueError("Task row ID missing or duplicated across splits")
            ids.add(row_id)
            if (not isinstance(text, str) or not text.strip() or not isinstance(target, str)
                    or not re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,14}", target)
                    or not re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", text, re.IGNORECASE)):
                raise ValueError("Entity pilot requires the literal target in the headline")
            if (row.get("target_scope") != "literal_entity_not_verified_security"
                    or row.get("asset_identity_scope", "not_verified") != "not_verified"
                    or row.get("asset_verified") is not False or row.get("human_reviewed") is not True
                    or row.get("training_allowed") is not True or not row.get("source") or not row.get("rights_note")
                    or row.get("synthetic") is True or row.get("label") not in LABELS):
                raise ValueError("Task row is not an explicitly reviewed, rights-recorded entity-pilot label")
            expected_split = "calibration" if split.startswith("calibration_") else split
            if row.get("split") != expected_split:
                raise ValueError("Task row is in the wrong split")
            labels.add(row["label"])
            if not isinstance(row.get("group_id"), str) or not row["group_id"]:
                raise ValueError("Task row needs a leakage group")
            for key in ("group_id", "event_id", "source_headline_id"):
                value = row.get(key)
                if value:
                    identity = (key, str(value))
                    if identity in groups and groups[identity] != split:
                        raise ValueError("Task group crosses held-out splits")
                    groups[identity] = split
            for earlier in duplicates.matches(text):
                if exact_entities[earlier] != split:
                    raise ValueError("Task lexical duplicate crosses held-out splits")
            duplicates.add(row_id, text)
            exact_entities[row_id] = split
            encode_prompt(tok, text, target, seq_len)
        if split in ("train", "development") and labels != LABELS:
            raise ValueError(f"{split} needs all three labels")
    if counts["train"] < 32:
        raise ValueError("Need at least 32 reviewed training examples")
    return counts


def _freeze_copy(source, target):
    source, target = Path(source), Path(target)
    entries = _inventory(source)
    if target.exists():
        other = _inventory(target)
        left = {entry["path"]: entry["sha256"] for entry in entries}
        right = {entry["path"]: entry["sha256"] for entry in other}
        if left != right:
            raise ValueError("Existing frozen experiment input differs; preserve it and select a new version")
        return
    sync_snapshot(source, target)


def _check_frozen_working_inputs(state):
    for entry in _inventory(state["stage_drive"] / "task_data"):
        path = Path(state["task_local"]) / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("Working task data differ from the frozen experiment snapshot")
    if sha256_file(state["config"]["init_checkpoint"]) != state["binding"]["parent_sha256"]:
        raise ValueError("Frozen parent changed")


def _budget_complete(progress, state):
    """AMP skips consume examples; either update or bounded epoch cap may stop."""
    return (progress["updates"] >= state["config"]["train"]["max_updates"]
            or progress["tokens"] >= state["counts"]["train"] * state["binding"]["epochs"])


def prepare_sft_pilot(project, drive_root, *, task_version="entity_v1", run_version="sft_entity_v1",
                      expected_checkpoint_sha256="2a6585bf18ed4753330221292d89b45ab17a52388ab7a0350eaf5d0dc4255c1f",
                      expected_task_manifest_sha256="a9b65b81d5280f43822ebb5456f2b6f6304a70bb84e0f61e284aac0597725840",
                      expected_domain_updates=74, epochs=2):
    task_version, run_version = _version(task_version), _version(run_version)
    if type(epochs) is not int or not 1 <= epochs <= 3:
        raise ValueError("The pilot requires one to three epochs")
    project, drive_root = Path(project).resolve(), Path(drive_root).resolve()
    task_drive, domain_drive = drive_root / "tasks" / task_version, drive_root / "domains/domain_v1"
    _inventory(task_drive)
    paths = restore_expansion_materials(project, drive_root, "expansion_v1", "seed_v1", restore_raw=False)
    tok = paths["tokenizer"]
    parent_audit = audit_domain_parent(domain_drive, tok, paths["processed"] / "domain_mix/manifest.json",
        paths["processed"] / "finance_validation/manifest.json", paths["seed_processed"] / "general_validation/manifest.json",
        expected_checkpoint_sha256=expected_checkpoint_sha256, expected_updates=expected_domain_updates)
    task_local = project / "data/task" / task_version
    _restore_input(task_drive, task_local)
    if sha256_file(task_local / "task_manifest.json") != expected_task_manifest_sha256:
        raise ValueError("Task manifest does not match the independently prepared dataset")
    manifest = json.loads((task_local / "task_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "130u-entity-pilot-v1":
        raise ValueError("Invalid task package format")
    for name in (split + ".jsonl" for split in SPLITS):
        item = manifest.get("files", {}).get(name, {})
        path = task_local / name
        if not path.is_file() or item.get("bytes") != path.stat().st_size or item.get("sha256") != sha256_file(path):
            raise ValueError("Task split differs from the recorded data manifest")
    parent = load_checkpoint(domain_drive / "run/latest.pt", allow_previous=False)
    seq_len = min(512, parent["model_config"]["max_position_embeddings"])
    counts = audit_task_inputs(task_local, tok, seq_len)
    exclusion_audit = _audit_training_exclusions(paths, task_local, domain_drive)
    parent_audit["actual_pretraining_exclusions"] = exclusion_audit
    stage_drive = drive_root / "sft" / run_version
    parent_drive, frozen_tasks = stage_drive / "parent", stage_drive / "task_data"
    _freeze_copy(domain_drive / "run", parent_drive)
    _freeze_copy(task_drive, frozen_tasks)
    parent_local = project / "checkpoints" / (run_version + "_parent")
    _restore_input(parent_drive, parent_local)
    parent_path = parent_local / "latest.pt"
    if sha256_file(parent_path) != expected_checkpoint_sha256:
        raise ValueError("Frozen domain parent differs from audited checkpoint")
    parent = load_checkpoint(parent_path, allow_previous=False)
    total_updates = max(2, math.ceil(counts["train"] * epochs / 32))
    binding = {"scope": "literal_entity_pilot_not_verified_security", "parent_sha256": expected_checkpoint_sha256,
               "task_snapshot_sha256": sha256_file(task_drive / "SNAPSHOT_COMPLETE.json"),
               "actual_exclusion_sha256": {name: value["sha256"] for name, value in exclusion_audit.items()},
               "task_version": task_version, "run_version": run_version, "epochs": epochs,
               "code_revision": ((project / "runtime_revision.txt").read_text(encoding="utf-8").strip()
                                 if (project / "runtime_revision.txt").is_file() else None)}
    config = {"mode": "formal", "stage": "sft", "model": parent["model_config"], "tokenizer": str(tok),
              "data": str(task_local / "train.jsonl"), "validation_data": str(task_local / "development.jsonl"),
              "output_dir": str(project / "runs" / run_version),
              "backup_dir": str(drive_root / "live_checkpoints" / run_version),
              "init_checkpoint": str(parent_path), "sft_pilot_binding": binding,
              "train": {"device": "auto", "seed": 130, "batch_size": 4, "grad_accumulation": 8,
                        "seq_len": seq_len, "max_updates": total_updates, "max_epochs": epochs,
                        "schedule_updates": total_updates, "learning_rate": 3e-5, "min_lr_ratio": .1,
                        "warmup_updates": min(5, total_updates // 5), "weight_decay": .1, "clip_grad": 1.0,
                        "save_every": 10, "validation_every": 10, "validation_max_batches": None,
                        "amp": True, "num_workers": 0}}
    _audit_formal_data(config, parent_audit["tokenizer_sha256"])
    _recover_report_archives(stage_drive)
    output = Path(config["output_dir"])
    if not output.exists() and (stage_drive / "run").exists():
        _restore_input(stage_drive / "run", output)
    selected = restore_seed_checkpoint(output, config["backup_dir"])
    if selected:
        previous = load_checkpoint(selected)
        if previous["config"].get("sft_pilot_binding") != binding:
            raise ValueError("Existing SFT run differs from its frozen parent or task set")
    elif (stage_drive / "run").exists() or (Path(config["output_dir"]) / "metrics.jsonl").exists():
        raise ValueError("An SFT run existed but no valid checkpoint remains; refusing restart")
    return {"config": config, "binding": binding, "parent_audit": parent_audit,
            "counts": counts, "task_local": task_local, "stage_drive": stage_drive,
            "selected": selected, "project": project}


def run_sft_pilot(state, phase, runner):
    if phase not in ("probe", "continue"):
        raise ValueError("Unknown SFT phase")
    config = copy.deepcopy(state["config"])
    _check_frozen_working_inputs(state)
    selected = restore_seed_checkpoint(config["output_dir"], config["backup_dir"])
    start = 0
    if selected:
        previous = load_checkpoint(selected)
        if previous["config"].get("sft_pilot_binding") != state["binding"]:
            raise ValueError("Existing SFT checkpoint identity differs")
        start = previous["progress"]["updates"]
        config.pop("init_checkpoint")
    if phase == "continue" and start < 1:
        raise ValueError("Complete the first-step probe before continuing")
    already_complete = selected and (start >= 1 if phase == "probe" else _budget_complete(previous["progress"], state))
    if already_complete:
        checkpoint_hash = sha256_file(selected)
        for archive in (state["stage_drive"] / "report_archive").glob("*"):
            metadata = json.loads(_snapshot_file(archive, "archive.json").read_text(encoding="utf-8"))
            if metadata.get("checkpoint_sha256") == checkpoint_hash and metadata.get("binding") == state["binding"]:
                report_bytes = _snapshot_file(archive, "run_summary.json").read_bytes()
                summary = json.loads(report_bytes)
                output = Path(config["output_dir"])
                for name in ("run_summary.json", "report.json"):
                    (output / name).write_bytes(report_bytes)
                return {"phase": phase, "start_updates": start, "end_updates": start,
                        "scope": state["binding"]["scope"], "summary": summary, "resumed_no_training": True}
        if phase == "probe":
            # A periodic live checkpoint can be newer than the last archived
            # segment report. Probe only verifies that work already exists;
            # the continuation runner performs strict provenance validation.
            # Never present checkpoint-only counters as measured throughput.
            return {"phase": phase, "start_updates": start, "end_updates": start,
                    "scope": state["binding"]["scope"], "summary": None,
                    "checkpoint_progress": dict(previous["progress"]),
                    "report_status": "checkpoint_only_no_matching_measured_report",
                    "resumed_no_training": True}
        # The final checkpoint may have been saved just before a session was
        # interrupted, before its report/archive was written. Let the normal
        # trainer resume it with the same configuration: its budget-complete
        # no-op generates checkpoint-derived metrics with throughput and VRAM
        # unknown, while still enforcing strict full-provenance checks.
    if phase == "probe":
        config["train"]["max_updates"] = 1
    config_path = state["project"] / "configs/runtime" / (state["binding"]["run_version"] + ".json")
    write_json(config_path, config)
    args = ["train", "--config", config_path]
    if selected:
        args.extend(["--resume", selected])
    runner(*args)
    checkpoint = load_checkpoint(Path(config["output_dir"]) / "latest.pt")
    if checkpoint["config"].get("sft_pilot_binding") != state["binding"]:
        raise ValueError("Completed checkpoint lost its frozen experiment identity")
    summary = json.loads((Path(config["output_dir"]) / "run_summary.json").read_text(encoding="utf-8"))
    if any(summary.get(k) != checkpoint["progress"][k] for k in ("updates", "attempted_updates", "tokens")):
        raise ValueError("SFT summary does not match checkpoint")
    if phase == "probe" and checkpoint["progress"]["updates"] < 1:
        raise ValueError("SFT probe did not finish")
    if phase == "continue" and not _budget_complete(checkpoint["progress"], state):
        raise ValueError("SFT training stopped before its declared budget")
    archive = state["stage_drive"] / "report_archive" / f"updates_{start:06d}_to_{checkpoint['progress']['updates']:06d}"
    _write_archive(archive, Path(config["output_dir"]) / "run_summary.json", {"format": "finllm-sft-pilot-report-v1",
        "binding": state["binding"], "checkpoint_sha256": sha256_file(Path(config["output_dir"]) / "latest.pt")})
    return {"phase": phase, "start_updates": start, "end_updates": checkpoint["progress"]["updates"],
            "scope": state["binding"]["scope"], "summary": summary}


def backup_sft_pilot(state):
    _check_frozen_working_inputs(state)
    output = Path(state["config"]["output_dir"])
    checkpoint = load_checkpoint(output / "latest.pt")
    if checkpoint["config"].get("sft_pilot_binding") != state["binding"]:
        raise ValueError("SFT checkpoint identity differs")
    if not _budget_complete(checkpoint["progress"], state):
        raise ValueError("Incomplete SFT cannot be recorded as final")
    report = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    if any(report.get(k) != checkpoint["progress"][k] for k in ("updates", "attempted_updates", "tokens")):
        raise ValueError("SFT final report does not match its checkpoint")
    return sync_snapshot(output, state["stage_drive"] / "run")
