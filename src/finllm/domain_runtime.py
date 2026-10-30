"""Bounded financial-domain training from a certified expanded parent.

The first update transfers weights into a new optimizer/runtime. Later domain
updates restore that experiment exactly; neither seed nor pretrain is rewritten.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import torch

from .checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, sha256_file
from .continuation import _recover_report_archives, _snapshot_file, _write_archive
from .data import TokenBlockDataset, write_json
from .expansion_runtime import (
    _inventory, _matches_expansion_report, _restore_input, _retention_bytes,
    _save_retention_snapshot, _validate_retention_result, _version,
    restore_expansion_materials,
)
from .model import FinLLM, ModelConfig
from .runtime import restore_seed_checkpoint, sync_snapshot
from .tokenizer import PAD_TOKEN, load_tokenizer
from .training import _audit_formal_data, _data_digest, _device, _settings, _validate


def _measured(report):
    for name in ("elapsed_seconds", "tokens_per_second"):
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("A genuine positive-time measured report is required")


def _parent_inputs(paths, minimum, directory, frozen=False):
    _inventory(directory)
    checkpoint_path = _snapshot_file(directory, "latest.pt")
    _snapshot_file(directory, "latest.pt.json")
    checkpoint = load_checkpoint(checkpoint_path, allow_previous=False)
    summary_path = _snapshot_file(directory, "parent_report.json" if frozen else "run_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    settings = _settings(checkpoint["config"])
    if (checkpoint["config"].get("mode") != "formal" or checkpoint["config"].get("stage") != "pretrain"
            or checkpoint["provenance"].get("mode") != "formal"
            or checkpoint["provenance"].get("stage") != "pretrain"
            or checkpoint["progress"]["updates"] < minimum
            or settings["max_updates"] != checkpoint["progress"]["updates"]
            or not _matches_expansion_report(summary, checkpoint)
            or checkpoint["tokenizer_hash"] != sha256_file(paths["tokenizer"] / "tokenizer.json")
            or checkpoint["provenance"].get("tokenizer_sha256") != checkpoint["tokenizer_hash"]
            or checkpoint["provenance"].get("model_config") != checkpoint["model_config"]
            or checkpoint["provenance"].get("data_sha256") != _data_digest(paths["processed"] / "general/manifest.json")
            or checkpoint["provenance"].get("validation_sha256") != _data_digest(paths["processed"] / "general_validation/manifest.json")):
        raise ValueError("Domain training requires the completed formal expansion checkpoint and matching report")
    _measured(summary)
    lineage = checkpoint["provenance"].get("parent_lineage", {})
    seed_path = _snapshot_file(paths["seed_drive"] / "run", "latest.pt")
    seed = load_checkpoint(seed_path, allow_previous=False)
    if (lineage.get("kind") != "data_expansion"
            or seed["config"].get("mode") != "formal" or seed["config"].get("stage") != "pretrain"
            or seed["provenance"].get("mode") != "formal" or seed["provenance"].get("stage") != "pretrain"
            or seed["tokenizer_hash"] != checkpoint["tokenizer_hash"]
            or lineage.get("parent_checkpoint_sha256") != sha256_file(seed_path)
            or lineage.get("parent_updates") != seed["progress"]["updates"]
            or lineage.get("parent_tokens") != seed["progress"]["tokens"]):
        raise ValueError("Expansion parent lineage differs from the preserved seed checkpoint")
    if frozen:
        retention_path = _snapshot_file(directory, "parent_retention_after.json")
        retention = json.loads(retention_path.read_text(encoding="utf-8"))
    else:
        candidates = [paths["expansion_drive"] / "retention/after"]
        candidates += sorted((paths["expansion_drive"] / "continuations").glob("*/retention/after"))
        retention_path = None
        for candidate in candidates:
            if not candidate.exists():
                continue
            path = _snapshot_file(candidate, "result.json")
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("inputs", {}).get("checkpoint_sha256") == sha256_file(checkpoint_path):
                retention_path, retention = path, value
                break
        if retention_path is None:
            raise ValueError("The completed expansion parent retention-after measurement is required")
    key = retention.get("inputs", {})
    _validate_retention_result(retention, "after", key)
    if (key.get("checkpoint_sha256") != sha256_file(checkpoint_path)
            or key.get("tokenizer_sha256") != checkpoint["tokenizer_hash"]
            or key.get("data_sha256") != _data_digest(paths["seed_processed"] / "general_validation/manifest.json")):
        raise ValueError("Expansion retention-after does not match the parent and frozen seed validation")
    identity = {"format": "finllm-domain-parent-v1", "source_expansion": paths["expansion_drive"].name,
                "source_seed": paths["seed_drive"].name, "checkpoint_sha256": sha256_file(checkpoint_path),
                "report_sha256": sha256_file(summary_path), "retention_sha256": sha256_file(retention_path),
                "updates": checkpoint["progress"]["updates"], "tokens": checkpoint["progress"]["tokens"],
                "seed_tokens": lineage["parent_tokens"], "tokenizer_sha256": checkpoint["tokenizer_hash"],
                "parent_provenance": checkpoint["provenance"]}
    return checkpoint_path, checkpoint, summary_path, retention_path, identity


def _freeze_parent(paths, minimum):
    archive = paths["drive"] / "parent"
    pending = paths["drive"] / "parent_identity.json"
    if (archive / "SNAPSHOT_COMPLETE.json").exists() and not (archive / "SNAPSHOT_INCOMPLETE.json").exists():
        path, parent, _, _, identity = _parent_inputs(paths, minimum, archive, frozen=True)
        metadata = json.loads(_snapshot_file(archive, "parent.json").read_text(encoding="utf-8"))
        if metadata != identity or (pending.exists() and json.loads(pending.read_text(encoding="utf-8")) != identity):
            raise ValueError("Frozen financial-domain parent identity changed")
    else:
        path, parent, summary, retention, identity = _parent_inputs(paths, minimum, paths["expansion_drive"] / "run")
        if pending.exists() and json.loads(pending.read_text(encoding="utf-8")) != identity:
            raise ValueError("Interrupted parent copy has different source bytes; preserve the existing domain version")
        if not pending.exists():
            write_json(pending, identity)
        with tempfile.TemporaryDirectory(prefix="finllm-domain-parent-") as temporary:
            folder = Path(temporary)
            for source, name in ((path, "latest.pt"), (path.with_name("latest.pt.json"), "latest.pt.json"),
                                 (summary, "parent_report.json"), (retention, "parent_retention_after.json")):
                shutil.copy2(source, folder / name)
            write_json(folder / "parent.json", identity)
            sync_snapshot(folder, archive)
        return _freeze_parent(paths, minimum)
    local = paths["project"] / "checkpoints" / (paths["drive"].name + "_parent")
    _restore_input(archive, local)
    return local / "latest.pt", parent, identity


def _provenance(config):
    settings = _settings(config)
    device = _device(settings["device"])
    keys = ("seed", "batch_size", "grad_accumulation", "seq_len", "learning_rate", "min_lr_ratio",
            "warmup_updates", "weight_decay", "clip_grad", "schedule_updates", "beta")
    return {"mode": "formal", "stage": "domain", "model_config": config["model"],
            "tokenizer_sha256": sha256_file(Path(config["tokenizer"]) / "tokenizer.json"),
            "data_sha256": _data_digest(config["data"]), "validation_sha256": _data_digest(config["validation_data"]),
            "training_settings": {key: settings[key] for key in keys}, "device_type": device.type,
            "amp": bool(settings["amp"] and device.type == "cuda"),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "forbid_epoch_replay": True}


def _check_transition(state):
    parent = state["parent"]
    expected = {"kind": "financial_domain", "initialization": "weights_only",
                "parent_checkpoint_sha256": parent["checkpoint_sha256"], "parent_updates": parent["updates"],
                "parent_tokens": parent["tokens"], "seed_tokens": parent["seed_tokens"],
                "parent_runtime": {key: parent["parent_provenance"].get(key)
                                   for key in ("device_type", "amp", "torch_version", "cuda_version")}}
    if (state["config"].get("domain_transition") != expected
            or sha256_file(state["config"]["init_checkpoint"]) != parent["checkpoint_sha256"]):
        raise ValueError("Domain transition differs from its frozen parent identity")


def _verify(checkpoint, state):
    _check_transition(state)
    actual, expected = _settings(checkpoint["config"]), _settings(state["config"])
    actual.pop("max_updates")
    expected.pop("max_updates")
    if (checkpoint["provenance"] != state["expected_provenance"]
            or checkpoint["config"].get("stage") != "domain"
            or checkpoint["config"].get("mode") != "formal"
            or checkpoint["model_config"] != state["config"]["model"]
            or checkpoint["tokenizer_hash"] != state["parent"]["tokenizer_sha256"]
            or checkpoint["config"].get("domain_transition") != state["config"]["domain_transition"]
            or checkpoint["config"].get("forbid_epoch_replay") is not True or actual != expected):
        raise ValueError("Domain checkpoint runtime, settings, inputs or parent transition differ")


def _matches(report, checkpoint):
    return (report.get("stage") == "domain" and report.get("mode") == "formal"
            and all(report.get(key) == checkpoint["progress"][key] for key in ("updates", "attempted_updates", "tokens"))
            and report.get("loss") == checkpoint.get("last_loss")
            and report.get("validation") == checkpoint.get("last_validation"))


def _archives(state):
    _recover_report_archives(state["paths"]["drive"])
    result = []
    for folder in sorted((state["paths"]["drive"] / "report_archive").glob("*")):
        if not folder.is_dir():
            continue
        path = _snapshot_file(folder, "run_summary.json")
        metadata = json.loads(_snapshot_file(folder, "archive.json").read_text(encoding="utf-8"))
        report = json.loads(path.read_text(encoding="utf-8"))
        _measured(report)
        if (metadata.get("format") != "finllm-domain-report-v1"
                or metadata.get("parent_sha256") != state["parent"]["checkpoint_sha256"]
                or metadata.get("provenance") != state["expected_provenance"]
                or metadata.get("end_updates") != report.get("updates")
                or metadata.get("end_tokens") != report.get("tokens")):
            raise ValueError("Domain archive differs from its frozen experiment")
        result.append((path, report))
    return result


def _select(state):
    _check_transition(state)
    config = state["config"]
    selected = restore_seed_checkpoint(config["output_dir"], config["backup_dir"])
    if selected is None:
        if (state.get("prior_run_evidence") or state.get("restored_updates", 0) > 0
                or (Path(config["output_dir"]) / "metrics.jsonl").exists()
                or (state["paths"]["drive"] / "run").exists()):
            raise ValueError("A domain experiment existed; refusing fresh initialization without a recoverable checkpoint")
        return None
    checkpoint = load_checkpoint(selected)
    _verify(checkpoint, state)
    settings = _settings(config)
    progress = checkpoint["progress"]
    remaining = (state["target_updates"] - progress["updates"]) * settings["batch_size"] * settings["grad_accumulation"]
    if (not 0 <= progress["updates"] <= state["target_updates"] or progress["epoch"] != 0
            or progress["cursor"] + remaining > len(TokenBlockDataset(config["data"], settings["seq_len"]))):
        raise ValueError("Domain progress is outside the one-pass budget; refusing replay or rollback")
    return selected


def prepare_domain_training(project, drive_root, version="domain_v1", expansion_version="expansion_v1", seed_version="seed_v1", *,
                            min_parent_updates=1220, probe_updates=1, target_updates=74):
    values = (min_parent_updates, probe_updates, target_updates)
    if (not all(isinstance(v, int) and not isinstance(v, bool) for v in values)
            or min_parent_updates < 1 or not 1 <= probe_updates < target_updates):
        raise ValueError("Invalid financial-domain update budgets")
    version, expansion_version, seed_version = map(_version, (version, expansion_version, seed_version))
    paths = restore_expansion_materials(project, drive_root, expansion_version, seed_version, restore_raw=False)
    paths["expansion_drive"] = paths["drive"]
    paths["drive"] = Path(drive_root).resolve() / "domains" / version
    _inventory(paths["expansion_drive"] / "processed")
    report = json.loads((paths["processed"] / "expansion_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((paths["processed"] / "domain_mix/manifest.json").read_text(encoding="utf-8"))
    sources = {name: TokenBlockDataset(paths["processed"] / name / "manifest.json", 1).manifest
               for name in ("general", "finance")}
    source_counts = {name: value["token_count"] for name, value in sources.items()}
    source_hashes = [sha256_file(paths["processed"] / name / "manifest.json") for name in ("general", "finance")]
    unit = min(source_counts["general"], source_counts["finance"] // 4)
    general_validation = json.loads((paths["processed"] / "general_validation/manifest.json").read_text(encoding="utf-8"))
    finance_validation = json.loads((paths["processed"] / "finance_validation/manifest.json").read_text(encoding="utf-8"))
    seed_validation = json.loads((paths["seed_processed"] / "general_validation/manifest.json").read_text(encoding="utf-8"))
    counts = manifest.get("mixture_token_counts", {})
    total = sum(counts.values())
    if (report.get("tokenizer_sha256") != sha256_file(paths["tokenizer"] / "tokenizer.json")
            or report.get("domain_mix", {}).get("repeated_source_tokens") != 0
            or total != manifest["token_count"] or total < 1
            or counts.get("finance", 0) / total != .8
            or report.get("token_counts", {}).get("domain_mix") != total
            or report.get("token_counts", {}).get("finance_validation") is None
            or counts != {"general": unit, "finance": unit * 4}
            or manifest.get("source_manifests") != source_hashes
            or manifest.get("source_sha256") != hashlib.sha256(json.dumps(source_hashes).encode()).hexdigest()
            or manifest.get("source_unique_token_counts") != source_counts
            or source_counts != {name: report["token_counts"].get(name) for name in sources}
            or manifest.get("seed") != report.get("settings", {}).get("seed")
            or any(value.get("mode") != "formal" or value.get("partition") != "train"
                   or value.get("tokenizer_sha256") != report["tokenizer_sha256"]
                   or value.get("vocab_size") != report.get("tokenizer_vocabulary")
                   or value.get("eval_exclusion_sha256") != sources["general"].get("eval_exclusion_sha256")
                   for value in [*sources.values(), manifest])
            or not sources["general"].get("eval_exclusion_sha256")
            or finance_validation.get("eval_exclusion_sha256") != general_validation.get("eval_exclusion_sha256")
            or report.get("seed_provenance", {}).get("task_exclusion_sha256") != seed_validation.get("eval_exclusion_sha256")):
        raise ValueError("A certified non-replayed 80/20 financial-domain mixture is required")
    parent_path, parent, identity = _freeze_parent(paths, min_parent_updates)
    settings = _settings(parent["config"])
    settings.update(max_updates=target_updates, max_tokens=None, max_epochs=None, schedule_updates=74,
                    learning_rate=1e-4, warmup_updates=5, save_every=10, validation_every=10,
                    validation_max_batches=50)
    config = {"mode": "formal", "stage": "domain", "model": parent["model_config"], "train": settings,
              "tokenizer": str(paths["tokenizer"]), "data": str(paths["processed"] / "domain_mix/manifest.json"),
              "validation_data": str(paths["processed"] / "finance_validation/manifest.json"),
              "output_dir": str(paths["project"] / "runs" / version),
              "backup_dir": str(Path(drive_root).resolve() / "live_checkpoints" / version),
              "init_checkpoint": str(parent_path), "forbid_epoch_replay": True,
              "domain_transition": {"kind": "financial_domain", "initialization": "weights_only",
                                    "parent_checkpoint_sha256": identity["checkpoint_sha256"],
                                    "parent_updates": identity["updates"], "parent_tokens": identity["tokens"],
                                    "seed_tokens": identity["seed_tokens"],
                                    "parent_runtime": {key: parent["provenance"].get(key) for key in ("device_type", "amp", "torch_version", "cuda_version")}}}
    _audit_formal_data(config, identity["tokenizer_sha256"])
    dataset = TokenBlockDataset(config["data"], settings["seq_len"])
    validation = TokenBlockDataset(config["validation_data"], settings["seq_len"])
    if (len(dataset) < target_updates * settings["batch_size"] * settings["grad_accumulation"]
            or validation.manifest["token_count"] != report["token_counts"]["finance_validation"]):
        raise ValueError("Financial-domain data is insufficient for the requested one-pass budget")
    state = {"paths": paths, "config": config, "parent": identity, "probe_updates": probe_updates,
             "target_updates": target_updates, "restored_updates": 0, "expected_provenance": _provenance(config),
             "before_retention_succeeded": False, "after_retention_succeeded": False,
             "continuation_succeeded": False, "budget_scope": "bounded financial-domain pilot; not completion of the original 25M-token budget"}
    archives = _archives(state)
    state["prior_run_evidence"] = bool(archives)
    output, snapshot = Path(config["output_dir"]), paths["drive"] / "run"
    if not (output / "latest.pt").exists() and snapshot.exists():
        try:
            _inventory(snapshot)
            sync_snapshot(snapshot, output, working_copy=True)
        except (ValueError, OSError):
            pass  # Independent verified live checkpoints may survive a failed final copy.
    selected = _select(state)
    if selected:
        checkpoint = load_checkpoint(selected)
        for path, report in archives:
            if _matches(report, checkpoint):
                for name in ("run_summary.json", "report.json"):
                    destination = output / name
                    if not destination.is_file() or destination.read_bytes() != path.read_bytes():
                        destination.write_bytes(path.read_bytes())
                break
        state["restored_updates"] = checkpoint["progress"]["updates"]
    return state


def _retention_inputs(state, phase):
    config = state["config"]
    path = Path(config["init_checkpoint"]) if phase == "before" else Path(config["output_dir"]) / "latest.pt"
    checkpoint = load_checkpoint(path, allow_previous=False)
    if phase == "before":
        if sha256_file(path) != state["parent"]["checkpoint_sha256"]:
            raise ValueError("Frozen domain parent changed")
    else:
        _verify(checkpoint, state)
        if checkpoint["progress"]["updates"] != state["target_updates"]:
            raise ValueError("Domain retention-after requires the exact completed target")
    data = state["paths"]["seed_processed"] / "general_validation/manifest.json"
    expansion = json.loads((state["paths"]["processed"] / "expansion_summary.json").read_text(encoding="utf-8"))
    if expansion["retention_validation"]["general"]["sha256"] != sha256_file(data):
        raise ValueError("Domain retention text differs from the frozen seed validation")
    settings = _settings(config)
    device = _device(settings["device"])
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    amp = bool(settings["amp"] and device.type == "cuda")
    key = {"checkpoint_sha256": sha256_file(path), "data_sha256": _data_digest(data),
           "tokenizer_sha256": checkpoint["tokenizer_hash"], "batch_size": settings["batch_size"],
           "seq_len": settings["seq_len"], "max_batches": 50, "device": str(device), "device_type": device.type,
           "amp": amp, "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}
    return checkpoint, data, key, device, amp


def retention_domain_check(state, phase):
    if phase not in ("before", "after"):
        raise ValueError("Unknown domain retention phase")
    state[phase + "_retention_succeeded"] = False
    if phase == "after" and not state.get("continuation_succeeded"):
        raise ValueError("Domain training did not complete; retention-after is blocked")
    checkpoint, data, key, device, amp = _retention_inputs(state, phase)
    body = _retention_bytes(state, phase, key)
    if body is None:
        rng = capture_rng_state()
        try:
            model = FinLLM(ModelConfig(**checkpoint["model_config"])).float().to(device)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            dataset = TokenBlockDataset(data, _settings(state["config"])["seq_len"])
            validation = _validate(model, dataset, "domain", key["batch_size"],
                                   load_tokenizer(state["config"]["tokenizer"]).token_to_id(PAD_TOKEN), [], device, amp, 50)
            result = {"phase": phase, "inputs": key, "validation": validation,
                      "scope": "Frozen seed general-validation subset; not financial stance accuracy"}
            _validate_retention_result(result, phase, key)
        finally:
            restore_rng_state(rng)
        body = (json.dumps(result, indent=2) + "\n").encode("utf-8")
        _save_retention_snapshot(state["paths"]["drive"] / "retention", phase, body)
    state[phase + "_retention_succeeded"] = True
    return json.loads(body)


def run_domain_segment(state, phase, runner):
    if phase not in ("probe", "continue"):
        raise ValueError("Unknown domain phase")
    state["continuation_succeeded"] = False
    state["after_retention_succeeded"] = False
    archives = _archives(state)
    parent_path = Path(state["config"]["init_checkpoint"])
    if sha256_file(parent_path) != state["parent"]["checkpoint_sha256"]:
        raise ValueError("Frozen domain parent changed before training")
    before_retention = json.loads(_snapshot_file(state["paths"]["drive"] / "retention/before", "result.json").read_text(encoding="utf-8"))
    _validate_retention_result(before_retention, "before", _retention_inputs(state, "before")[2])
    state["before_retention_succeeded"] = True
    selected = _select(state)
    before = load_checkpoint(selected) if selected else None
    start = before["progress"]["updates"] if before else 0
    start_tokens = before["progress"]["tokens"] if before else 0
    if phase == "continue" and start < state["probe_updates"]:
        raise ValueError("Complete the domain weights-transfer probe before longer training")
    target = state["probe_updates"] if phase == "probe" else state["target_updates"]
    output = Path(state["config"]["output_dir"])
    measured = None
    if before is not None and start >= target:
        candidates = [path for path, _ in archives] + [output / "run_summary.json"]
        for path in candidates:
            if path.is_file() and _matches(json.loads(path.read_text(encoding="utf-8")), before):
                measured = path.read_bytes()
                _measured(json.loads(measured))
                break
        if measured is None:
            raise ValueError("Completed domain checkpoint has no matching measured report; refusing a fabricated no-op")
    config = copy.deepcopy(state["config"])
    config["train"]["max_updates"] = target
    arguments = ["train", "--config", state["paths"]["project"] / "configs/runtime" / (state["paths"]["drive"].name + ".json")]
    if selected:
        config.pop("init_checkpoint")
        arguments += ["--resume", selected]
    write_json(arguments[2], config)
    runner(*arguments)
    after = load_checkpoint(output / "latest.pt")
    _verify(after, state)
    end = after["progress"]["updates"]
    summary = output / "run_summary.json"
    report = json.loads(summary.read_text(encoding="utf-8"))
    if end != max(start, target) or not _matches(report, after):
        raise ValueError("Domain segment or summary is incomplete")
    if end == start and measured is not None:
        for name in ("run_summary.json", "report.json"):
            path = output / name
            if path.read_bytes() != measured:
                path.write_bytes(measured)
        report = json.loads(measured)
    _measured(report)
    archive = None
    if end > start:
        archive = state["paths"]["drive"] / "report_archive" / f"updates_{start:06d}_to_{end:06d}"
        _write_archive(archive, summary, {"format": "finllm-domain-report-v1", "kind": phase,
            "parent_sha256": state["parent"]["checkpoint_sha256"], "provenance": after["provenance"],
            "start_updates": start, "end_updates": end, "start_tokens": start_tokens, "end_tokens": report["tokens"]})
    state["restored_updates"] = end
    state["continuation_succeeded"] = phase == "continue"
    return {"phase": phase, "start_updates": start, "end_updates": end, "new_updates": end-start,
            "new_experiment_tokens": report["tokens"], "parent_tokens": state["parent"]["tokens"],
            "seed_tokens": state["parent"]["seed_tokens"],
            "model_path_total_tokens": state["parent"]["seed_tokens"] + state["parent"]["tokens"] + report["tokens"],
            "report_archive": str(archive) if archive else None, "budget_scope": state["budget_scope"], "summary": report}


def backup_domain_training(state):
    if not all(state.get(key) for key in ("continuation_succeeded", "before_retention_succeeded", "after_retention_succeeded")):
        raise ValueError("Domain training/retention did not complete; refusing final backup")
    _archives(state)
    _freeze_parent(state["paths"], state["parent"]["updates"])
    selected = _select(state)
    if selected is None:
        raise ValueError("No domain checkpoint exists")
    checkpoint = load_checkpoint(selected)
    report = json.loads((Path(state["config"]["output_dir"]) / "run_summary.json").read_text(encoding="utf-8"))
    if checkpoint["progress"]["updates"] != state["target_updates"] or not _matches(report, checkpoint):
        raise ValueError("Domain target or measured report is incomplete")
    _measured(report)
    for phase in ("before", "after"):
        result = json.loads(_snapshot_file(state["paths"]["drive"] / "retention" / phase, "result.json").read_text(encoding="utf-8"))
        _validate_retention_result(result, phase, _retention_inputs(state, phase)[2])
    result = sync_snapshot(state["config"]["output_dir"], state["paths"]["drive"] / "run")
    if result.get("verified") is not True:
        raise ValueError("Final domain snapshot did not verify")
    return result
