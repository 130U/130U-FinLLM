"""Small, private, weight-only handoff for the previously audited model.

The Kaggle recipient verifies the compact artifact but does not receive or
independently re-audit the historical pretraining data and exclusion pools.
Only load a checkpoint that the owner created and whose SHA is trusted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .checkpoint import load_checkpoint, save_checkpoint, sha256_file
from .sft_pilot_runtime import audit_task_inputs
from .kaggle_sft_runtime import _read_manifest, _copy_verified, _path, _digest
from .continuation import _snapshot_file
from .expansion_runtime import _inventory

DOMAIN_SHA = "2a6585bf18ed4753330221292d89b45ab17a52388ab7a0350eaf5d0dc4255c1f"
TASK_MANIFEST_SHA = "a9b65b81d5280f43822ebb5456f2b6f6304a70bb84e0f61e284aac0597725840"
TOKENIZER_SHA = "3930526cf6d487291cca8ee5f027c0ebeb9a40defb3711110653a36c000ade8a"
POOL_NAMES = ("seed_train", "seed_validation", "expansion_train", "expansion_validation")
ROOTS = {"weights", "tokenizer", "task"}
REPORTED_04B = {
    "seed_train": "0c24d266947f60a11c8871785af1fde6955f0c2e4174942cbb33cac4f685219e",
    "seed_validation": "523110f6a56c15b0c64a32236ca6dee89af8493d4dba1022b23a13d1527ec508",
    "expansion_train": "39026cd8c116c39a2b0477442243cf6ddde833217d267e428e88bf79e96d4a9f",
    "expansion_validation": "d5040a1709321a29b802cef481d881a37ff01a3bcd97f0bf15766f242ecbc278",
}


def _emit_lite(payload, parent, audit_record, output, expected_domain_sha256,
               expected_tokenizer_sha256, expected_task_sha256, expected_task_rows):
    print("正在核对任务分区并导出模型权重。", flush=True)
    tokenizer, task = payload / "tokenizer/tokenizer.json", payload / "task"
    if sha256_file(tokenizer) != expected_tokenizer_sha256 or sha256_file(task / "task_manifest.json") != expected_task_sha256:
        raise ValueError("Task or tokenizer does not match the separately recorded identity")
    counts = audit_task_inputs(task, payload / "tokenizer", parent["model_config"]["max_position_embeddings"])
    if sum(counts.values()) != expected_task_rows:
        raise ValueError("Task row count differs from the previously reported exclusion audit")
    slim = {"config": {"stage": "domain", "mode": "formal"}, "model_config": parent["model_config"],
            "model_state": parent["model_state"], "tokenizer_hash": parent["tokenizer_hash"],
            "progress": parent["progress"], "source_checkpoint_sha256": expected_domain_sha256}
    save_checkpoint(payload / "weights/domain_weights.pt", slim)
    files = {}
    for path in sorted(payload.rglob("*")):
        if path.is_file():
            name = path.relative_to(payload).as_posix()
            _path(name, ROOTS)
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    result = {"format": "130u-kaggle-lite-v1", "files": files,
              "source_checkpoint_sha256": expected_domain_sha256, "tokenizer_sha256": expected_tokenizer_sha256,
              "source_task_manifest_sha256": expected_task_sha256, **audit_record, "task_counts": counts,
              "scope": "weight-only SFT init; Kaggle cannot independently re-audit omitted historical pretraining files"}
    body = json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8")
    partial = output.with_name(output.name + ".partial")
    print("正在生成轻量包并做完整性校验。", flush=True)
    with zipfile.ZipFile(partial, mode="x", compression=zipfile.ZIP_STORED, allowZip64=True) as dest:
        for name in files:
            dest.write(payload / name, name)
        dest.writestr("lite_manifest.json", body)
    with zipfile.ZipFile(partial) as dest:
        check = _read_manifest(dest, "lite_manifest.json", "130u-kaggle-lite-v1", ROOTS)
        for name, record in check["files"].items():
            digest = hashlib.sha256()
            with dest.open(name) as item:
                while block := item.read(4 * 1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError("Lite archive readback failed")
    os.link(partial, output)
    partial.unlink()
    return {"output": str(output), "bytes": output.stat().st_size, "sha256": sha256_file(output),
            "manifest_sha256": hashlib.sha256(body).hexdigest(), "task_counts": counts,
            "scope": result["scope"], "source_checkpoint_sha256": expected_domain_sha256}


def export_lite_from_drive(drive_root, output, *, expected_domain_sha256=DOMAIN_SHA,
                           expected_tokenizer_sha256=TOKENIZER_SHA, expected_task_sha256=TASK_MANIFEST_SHA,
                           expected_task_rows=1008, expected_domain_updates=74):
    """Use intact Drive snapshots even when the old Colab VM has expired.

    Four earlier exclusion outcomes are explicitly historical, user-reported
    04b evidence. The compact archive does not contain those source lists.
    """
    root, output = Path(drive_root).resolve(), Path(output).resolve()
    if not root.is_dir() or output == root or output.is_relative_to(root):
        raise ValueError("Choose an output outside the existing Drive project")
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError("Existing output preserved; choose a new filename")
    if (root / "sft/sft_entity_v1").exists() or (root / "live_checkpoints/sft_entity_v1").exists():
        raise ValueError("Existing Colab SFT progress found; inspect it before starting a separate Kaggle experiment")
    original_folder = root / "domains/domain_v1/run"
    print("正在从 Drive 读取并核验原领域检查点；可能需要一些时间。", flush=True)
    original = _snapshot_file(original_folder, "latest.pt")
    _snapshot_file(original_folder, "latest.pt.json")
    if sha256_file(original) != expected_domain_sha256:
        raise ValueError("Domain checkpoint differs from the reported 74-step result")
    parent = load_checkpoint(original, allow_previous=False)
    if (parent.get("config", {}).get("stage") != "domain" or parent.get("config", {}).get("mode") != "formal"
            or parent.get("progress", {}).get("updates") != expected_domain_updates
            or parent.get("tokenizer_hash") != expected_tokenizer_sha256):
        raise ValueError("Domain checkpoint stage, progress or tokenizer mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="130u-lite-", dir=output.parent) as folder:
        payload = Path(folder) / "payload"
        print("正在核验原词表和已审核任务数据。", flush=True)
        for source, name in ((root / "seeds/seed_v1/tokenizer", "tokenizer"), (root / "tasks/entity_v1", "task")):
            for entry in _inventory(source):
                relative = _path(name + "/" + entry["path"], ROOTS)
                target = payload.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_snapshot_file(source, entry["path"]), target)
        evidence = {name: {"sha256": value, "task_rows": 1008, "distinct_task_texts": 1008, "missing": 0}
                    for name, value in REPORTED_04B.items()}
        record = {"input_kind": "drive_snapshots_direct",
                  "prior_exclusion_audit_reported": evidence,
                  "prior_exclusion_evidence_source": "user-pasted completed 04b Colab output; not re-run here",
                  "source_transfer_sha256_reported": "f534217abfd454c949629913fb0a7772636742d1bb265ea1143358102607e7f1",
                  "source_transfer_manifest_sha256": "e4ba6efefba7d86acdcd569842affce703437d987765117a56417a9d8fbecf11"}
        return _emit_lite(payload, parent, record, output, expected_domain_sha256,
                          expected_tokenizer_sha256, expected_task_sha256, expected_task_rows)


def _check_lite_root(root, manifest, expected_manifest_sha256):
    root = Path(root).resolve()
    if manifest.get("format") != "130u-kaggle-lite-v1" or not isinstance(manifest.get("files"), dict):
        raise ValueError("Invalid lite manifest")
    if (manifest.get("source_checkpoint_sha256") != DOMAIN_SHA or manifest.get("tokenizer_sha256") != TOKENIZER_SHA
            or manifest.get("source_task_manifest_sha256") != TASK_MANIFEST_SHA):
        raise ValueError("Lite bundle does not bind to the expected model, task and tokenizer")
    for name, info in manifest["files"].items():
        relative = _path(name, ROOTS)
        path = root.joinpath(*relative.parts)
        if (not isinstance(info, dict) or type(info.get("bytes")) is not int or not _digest(info.get("sha256"))
                or path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file()
                or path.stat().st_size != info["bytes"] or sha256_file(path) != info["sha256"]):
            raise ValueError("Lite bundle file is absent or does not match its manifest")
    if sha256_file(root / "tokenizer/tokenizer.json") != TOKENIZER_SHA or sha256_file(root / "task/task_manifest.json") != TASK_MANIFEST_SHA:
        raise ValueError("Lite task or tokenizer identity mismatch")
    parent = load_checkpoint(root / "weights/domain_weights.pt", allow_previous=False)
    if (parent.get("source_checkpoint_sha256") != DOMAIN_SHA or parent.get("tokenizer_hash") != TOKENIZER_SHA
            or parent.get("config", {}).get("stage") != "domain"):
        raise ValueError("Lite checkpoint binding mismatch")
    counts = audit_task_inputs(root / "task", root / "tokenizer", parent["model_config"]["max_position_embeddings"])
    if counts != manifest.get("task_counts") or sum(counts.values()) < 1:
        raise ValueError("Lite task counts differ from the transferred report")
    return parent, counts


def install_lite_directory(source, destination, *, expected_manifest_sha256):
    source, destination = Path(source), Path(destination)
    manifest_path = source / "lite_manifest.json"
    if not _digest(expected_manifest_sha256) or sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("Lite manifest differs from the value printed during Colab export")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check_lite_root(source, manifest, expected_manifest_sha256)
    if destination.exists():
        old = destination / "lite_manifest.json"
        if not old.is_file() or sha256_file(old) != expected_manifest_sha256:
            raise FileExistsError("A different or incomplete lite input is preserved")
        _check_lite_root(destination, manifest, expected_manifest_sha256)
        return {"reused": True, "manifest_sha256": expected_manifest_sha256}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lite-install-", dir=destination.parent) as folder:
        stage = Path(folder) / "data"
        stage.mkdir()
        for name in manifest["files"]:
            rel = _path(name, ROOTS)
            path = stage.joinpath(*rel.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.joinpath(*rel.parts), path)
        (stage / "lite_manifest.json").write_bytes(manifest_path.read_bytes())
        _check_lite_root(stage, manifest, expected_manifest_sha256)
        os.rename(stage, destination)
    return {"reused": False, "manifest_sha256": expected_manifest_sha256}


def install_lite_archive(zip_path, destination, *, expected_zip_sha256):
    if not _digest(expected_zip_sha256) or sha256_file(zip_path) != expected_zip_sha256:
        raise ValueError("Lite ZIP SHA mismatch")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive, tempfile.TemporaryDirectory(prefix="lite-unzip-", dir=Path(destination).parent) as temp:
        manifest = _read_manifest(archive, "lite_manifest.json", "130u-kaggle-lite-v1", ROOTS)
        stage = Path(temp) / "contents"
        stage.mkdir()
        _copy_verified(archive, manifest, stage)
        raw = archive.read("lite_manifest.json")
        (stage / "lite_manifest.json").write_bytes(raw)
        return install_lite_directory(stage, destination, expected_manifest_sha256=hashlib.sha256(raw).hexdigest())


def prepare_lite_sft(project, input_dir, run_dir, *, expected_manifest_sha256, epochs=2):
    if type(epochs) is not int or not 1 <= epochs <= 3:
        raise ValueError("Choose 1–3 pilot epochs")
    project, root, output = Path(project).resolve(), Path(input_dir).resolve(), Path(run_dir).resolve()
    raw = (root / "lite_manifest.json").read_bytes()
    if not _digest(expected_manifest_sha256) or hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("Installed lite manifest differs from the separately supplied export hash")
    manifest = json.loads(raw)
    parent, counts = _check_lite_root(root, manifest, hashlib.sha256(raw).hexdigest())
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output must be separate from immutable input")
    total = max(2, math.ceil(counts["train"] * epochs / 32))
    binding = {"scope": "literal_entity_weight_only_exploratory", "lite_manifest_sha256": hashlib.sha256(raw).hexdigest(),
               "parent_sha256": sha256_file(root / "weights/domain_weights.pt"),
               "source_checkpoint_sha256": DOMAIN_SHA, "epochs": epochs,
               "code_revision": ((project / "runtime_revision.txt").read_text(encoding="utf-8").strip()
                                  if (project / "runtime_revision.txt").is_file() else None)}
    config = {"mode": "formal", "stage": "sft", "model": parent["model_config"],
              "tokenizer": str(root / "tokenizer"), "data": str(root / "task/train.jsonl"),
              "validation_data": str(root / "task/development.jsonl"), "output_dir": str(output),
              "backup_dir": str(output.parent / (output.name + "_live")),
              "init_checkpoint": str(root / "weights/domain_weights.pt"), "lite_binding": binding,
              "train": {"device": "auto", "seed": 130, "batch_size": 4, "grad_accumulation": 8,
                        "seq_len": min(512, parent["model_config"]["max_position_embeddings"]),
                        "max_updates": total, "max_epochs": epochs, "schedule_updates": total,
                        "learning_rate": 3e-5, "min_lr_ratio": .1, "warmup_updates": min(5, total // 5),
                        "weight_decay": .1, "clip_grad": 1.0, "save_every": 10, "validation_every": 10,
                        "validation_max_batches": None, "amp": True, "num_workers": 0}}
    return {"config": config, "binding": binding, "counts": counts, "input_dir": str(root),
            "scope": manifest["scope"]}
