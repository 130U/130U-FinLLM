"""A genuine tiny domain run exercises read-only pre-SFT checkpoint audit."""
import json
from pathlib import Path

import pytest

from finllm.checkpoint import sha256_file
from finllm.sft_preflight import audit_domain_parent
from test_domain_runtime import _complete, _domain, _domain_fixture, _inventory


@pytest.fixture
def completed_domain(tmp_path):
    import torch
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        project, drive = _domain_fixture(tmp_path)
        state = _domain(project, drive)
        _complete(state)
        root = drive / "domains/domain_v1"
        processed = project / "data/processed/expansion_v1"
        arguments = dict(domain_dir=root, tokenizer=state["paths"]["tokenizer"],
                         domain_mix=processed / "domain_mix/manifest.json",
                         finance_validation=processed / "finance_validation/manifest.json",
                         seed_validation=project / "data/processed/seed_v1/general_validation/manifest.json",
                         expected_checkpoint_sha256=sha256_file(root / "run/latest.pt"), expected_updates=3)
        yield arguments
    finally:
        torch.set_num_threads(old)


def test_complete_domain_snapshot_is_verified_without_writes(completed_domain):
    before = _inventory(completed_domain["domain_dir"])
    result = audit_domain_parent(**completed_domain)
    assert result["updates"] == 3
    assert result["checkpoint_sha256"] == completed_domain["expected_checkpoint_sha256"]
    assert result["trainable_task_data_checked"] is False and result["sft_ready"] is False
    assert _inventory(completed_domain["domain_dir"]) == before


@pytest.mark.parametrize("change", ["wrong_hash", "wrong_updates", "modified_retention", "modified_input", "modified_report", "modified_parent"])
def test_mismatched_identity_or_modified_snapshot_is_rejected(completed_domain, change):
    args = dict(completed_domain)
    if change == "wrong_hash":
        args["expected_checkpoint_sha256"] = "a" * 64
    elif change == "wrong_updates":
        args["expected_updates"] = 2
    elif change == "modified_retention":
        p = args["domain_dir"] / "retention/after/result.json"
        value = json.loads(p.read_text(encoding="utf-8"))
        value["validation"]["loss"] += 1
        p.write_text(json.dumps(value), encoding="utf-8")
    elif change == "modified_input":
        p = args["domain_mix"].parent / "tokens.bin"
        p.write_bytes(bytes(len(p.read_bytes())))
    elif change == "modified_report":
        p = args["domain_dir"] / "run/run_summary.json"
        p.write_text("{}", encoding="utf-8")
    elif change == "modified_parent":
        p = args["domain_dir"] / "parent/latest.pt"
        p.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        audit_domain_parent(**args)


@pytest.mark.parametrize("identity", ["checkpoint_sha256", "data_sha256", "tokenizer_sha256"])
def test_internally_complete_retention_snapshot_with_wrong_identity_is_rejected(completed_domain, identity):
    """A valid inventory is insufficient when diagnostics refer to other bytes."""
    folder = completed_domain["domain_dir"] / "retention/after"
    report_path = folder / "result.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["inputs"][identity] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    marker_path = folder / "SNAPSHOT_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    entry = next(item for item in marker["files"] if item["path"] == "result.json")
    entry.update(bytes=report_path.stat().st_size, sha256=sha256_file(report_path))
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="retention diagnostics"):
        audit_domain_parent(**completed_domain)
