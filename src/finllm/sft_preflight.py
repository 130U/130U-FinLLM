"""Read-only audit of a completed, trusted domain parent before future SFT.

This does not create a trainable dataset, select an SFT model, freeze the
parent, restore a run, or start training. Checkpoints must be trusted project
artifacts: their pickle-based loader is not safe for untrusted files.
"""
from __future__ import annotations

import json
from pathlib import Path

from .checkpoint import load_checkpoint, sha256_file
from .continuation import _snapshot_file
from .data import TokenBlockDataset, tokenizer_file
from .domain_runtime import _matches, _measured
from .expansion_runtime import _inventory, _validate_retention_result
from .training import _data_digest, _settings


def audit_domain_parent(domain_dir, tokenizer, domain_mix, finance_validation,
                        seed_validation, *, expected_checkpoint_sha256, expected_updates):
    """Verify actual snapshot bytes, lineage and compatible source manifests.

    The expected digest and update count must come from a separately recorded
    run, never from the snapshot being checked. No writes are made.
    """
    if (not isinstance(expected_checkpoint_sha256, str) or len(expected_checkpoint_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_checkpoint_sha256)):
        raise ValueError("Expected checkpoint SHA256 must be a lowercase 64-character hex string")
    if type(expected_updates) is not int or expected_updates < 1:
        raise ValueError("Expected update count must be a positive integer")
    root = Path(domain_dir)
    run = root / "run"
    _inventory(run)
    checkpoint_path = _snapshot_file(run, "latest.pt")
    _snapshot_file(run, "latest.pt.json")
    checkpoint_digest = sha256_file(checkpoint_path)
    if checkpoint_digest != expected_checkpoint_sha256:
        raise ValueError("Domain latest checkpoint differs from the separately recorded identity")
    checkpoint = load_checkpoint(checkpoint_path, allow_previous=False)
    report_path = _snapshot_file(run, "run_summary.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = checkpoint["config"]
    provenance = checkpoint["provenance"]
    settings = _settings(config)
    tok_digest = sha256_file(tokenizer_file(tokenizer))
    if (config.get("stage") != "domain" or config.get("mode") != "formal"
            or provenance.get("stage") != "domain" or provenance.get("mode") != "formal"
            or checkpoint["progress"]["updates"] != expected_updates
            or settings["max_updates"] != expected_updates
            or not _matches(report, checkpoint)
            or checkpoint["tokenizer_hash"] != tok_digest
            or provenance.get("tokenizer_sha256") != tok_digest
            or provenance.get("model_config") != checkpoint["model_config"]
            or config.get("model") != checkpoint["model_config"]
            or config.get("forbid_epoch_replay") is not True):
        raise ValueError("Domain checkpoint, measured report, tokenizer or stage disagree")
    _measured(report)
    manifests = []
    for path in (domain_mix, finance_validation, seed_validation):
        manifests.append(TokenBlockDataset(path, 1).manifest)
    for manifest, partition in zip(manifests, ("train", "validation", "validation")):
        if (manifest.get("mode") != "formal" or manifest.get("partition") != partition
                or manifest.get("tokenizer_sha256") != tok_digest):
            raise ValueError("Input token manifest mode, partition or tokenizer differs")
    if (provenance.get("data_sha256") != _data_digest(domain_mix)
            or provenance.get("validation_sha256") != _data_digest(finance_validation)):
        raise ValueError("Domain data or validation differs from checkpoint provenance")
    retention = {}
    for phase in ("before", "after"):
        folder = root / "retention" / phase
        _inventory(folder)
        value = json.loads(_snapshot_file(folder, "result.json").read_text(encoding="utf-8"))
        inputs = value.get("inputs", {})
        _validate_retention_result(value, phase, inputs)
        retention[phase] = value
    before, after = retention["before"]["inputs"], retention["after"]["inputs"]
    transition = config.get("domain_transition", {})
    parent_dir = root / "parent"
    _inventory(parent_dir)
    parent_path = _snapshot_file(parent_dir, "latest.pt")
    _snapshot_file(parent_dir, "latest.pt.json")
    if sha256_file(parent_path) != transition.get("parent_checkpoint_sha256"):
        raise ValueError("Frozen pretraining parent differs from the domain transition")
    parent = load_checkpoint(parent_path, allow_previous=False)
    if (parent["config"].get("stage") != "pretrain" or parent["config"].get("mode") != "formal"
            or parent["progress"]["updates"] != transition.get("parent_updates")
            or parent["progress"]["tokens"] != transition.get("parent_tokens")
            or parent["tokenizer_hash"] != tok_digest):
        raise ValueError("Frozen pretraining checkpoint progress or tokenizer differs from domain lineage")
    if (transition.get("kind") != "financial_domain"
            or transition.get("initialization") != "weights_only"
            or not transition.get("parent_checkpoint_sha256")
            or before.get("checkpoint_sha256") != transition["parent_checkpoint_sha256"]
            or after.get("checkpoint_sha256") != checkpoint_digest
            or before.get("data_sha256") != _data_digest(seed_validation)
            or after.get("data_sha256") != before.get("data_sha256")
            or before.get("tokenizer_sha256") != tok_digest
            or after.get("tokenizer_sha256") != tok_digest
            or {k: v for k, v in before.items() if k != "checkpoint_sha256"}
            != {k: v for k, v in after.items() if k != "checkpoint_sha256"}):
        raise ValueError("Domain retention diagnostics do not match the parent and frozen seed data")
    return {"checkpoint_sha256": checkpoint_digest, "updates": expected_updates,
            "tokens": checkpoint["progress"]["tokens"], "tokenizer_sha256": tok_digest,
            "domain_mix_sha256": provenance["data_sha256"],
            "finance_validation_sha256": provenance["validation_sha256"],
            "seed_retention_before_loss": retention["before"]["validation"]["loss"],
            "seed_retention_after_loss": retention["after"]["validation"]["loss"],
            "trainable_task_data_checked": False,
            "sft_ready": False}
