"""Full-parameter pretraining, label-only SFT, and single-label DPO.

The model returns unshifted logits. Pretraining data supplies labels already
shifted once; SFT/DPO supervise only the answer position. Checkpoints are written
only after a complete optimizer attempt, including any AMP-skipped attempt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from finllm.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint, sha256_file, snapshot_to_drive
from finllm.data import TokenBlockDataset, TaskDataset, collate_task, read_jsonl
from finllm.model import FinLLM, ModelConfig
from finllm.tokenizer import load_tokenizer


LABEL_TOKENS = ("<|bearish|>", "<|neutral|>", "<|bullish|>")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _tokenizer_file(path: str | Path) -> Path:
    path = Path(path)
    return path / "tokenizer.json" if path.is_dir() else path


def _data_digest(path: str | Path) -> str:
    """Hash source bytes and all referenced token shards using bounded buffers."""
    path = Path(path)
    parts = {"source": sha256_file(path)}
    if path.suffix.lower() == ".json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and "shards" in manifest:
            parts["shards"] = [
                {"path": item["path"], "sha256": sha256_file(path.parent / item["path"])}
                for item in manifest["shards"]
            ]
    return _json_digest(parts)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported training devices are cpu, cuda, and auto.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA runtime is available.")
    return device


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _audit_formal_data(config: dict[str, Any], tokenizer_hash: str) -> None:
    """Refuse known split/rights mistakes before any optimizer work.

    Exact text and group checks here complement the upstream near-duplicate
    audit. They do not claim to detect every paraphrase or shared event.
    """
    if not config.get("validation_data"):
        raise ValueError("Formal training requires a separate validation_data file.")
    training_path, development_path = Path(config["data"]), Path(config["validation_data"])
    if training_path.resolve() == development_path.resolve():
        raise ValueError("Training and validation must be separate files.")
    if config["stage"] in {"pretrain", "domain"}:
        manifests = [json.loads(path.read_text(encoding="utf-8")) for path in (training_path, development_path)]
        for manifest, expected_partition in zip(manifests, ("train", "validation")):
            if manifest.get("mode") != "formal" or manifest.get("tokenizer_sha256") != tokenizer_hash:
                raise ValueError("Formal token manifest mode/tokenizer mismatch.")
            if manifest.get("partition", expected_partition) != expected_partition:
                raise ValueError(f"Expected token manifest partition={expected_partition}.")
            if not manifest.get("eval_exclusion_sha256"):
                raise ValueError("Formal token manifests must record a frozen evaluation-exclusion pool.")
        if manifests[0].get("source_sha256") == manifests[1].get("source_sha256"):
            raise ValueError("Pretraining validation requires a distinct held-out source file.")
        overlap = {item["sha256"] for item in manifests[0]["shards"]} & {item["sha256"] for item in manifests[1]["shards"]}
        if overlap:
            raise ValueError("Identical token shards occur in training and validation.")
        return
    seen_groups: set[str] = set()
    seen_texts: set[str] = set()
    for path, expected_split in ((training_path, "train"), (development_path, "development")):
        for row in read_jsonl(path):
            if row.get("split") != expected_split:
                raise ValueError(f"Expected split={expected_split} in {path}.")
            if row.get("training_allowed") is not True or not row.get("rights_note") or not row.get("source"):
                raise ValueError("Formal task data requires training_allowed=true, source, and rights_note.")
            if row.get("synthetic") is True:
                raise ValueError("Synthetic smoke examples cannot enter formal training or validation.")
            if not row.get("group_id"):
                raise ValueError("Formal task data requires a group_id for leakage checks.")
            if expected_split == "train" and config["stage"] == "dpo" and row.get("preference_reviewed") is not True:
                raise ValueError("Formal DPO pairs require preference_reviewed=true.")
            text_hash = hashlib.sha256(" ".join(row["text"].split()).casefold().encode("utf-8")).hexdigest()
            if expected_split == "train":
                seen_groups.add(str(row["group_id"]))
                seen_texts.add(text_hash)
            elif str(row["group_id"]) in seen_groups or text_hash in seen_texts:
                raise ValueError("Training/development leakage: repeated group or normalized text.")


class DataCursor:
    """Deterministic epoch shuffle independent of global/model RNG.

    ``cursor`` is an example offset in an epoch permutation, not a batch count.
    Reads are synchronous, so there are no prefetched-but-unconsumed samples.
    """

    def __init__(self, size: int, batch_size: int, seed: int, epoch: int = 0, cursor: int = 0):
        if size < 1 or batch_size < 1:
            raise ValueError("Dataset and batch size must be nonempty.")
        if epoch < 0 or not 0 <= cursor <= size:
            raise ValueError("Invalid saved data cursor.")
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.cursor = epoch, cursor
        self._permutation_epoch: int | None = None
        self._permutation: list[int] = []

    def next_indices(self) -> list[int]:
        if self.cursor == self.size:
            self.epoch += 1
            self.cursor = 0
        if self._permutation_epoch != self.epoch:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            self._permutation = torch.randperm(self.size, generator=generator).tolist()
            self._permutation_epoch = self.epoch
        end = min(self.size, self.cursor + self.batch_size)
        indices = self._permutation[self.cursor:end]
        self.cursor = end
        return indices

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor}


def pretraining_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Summed CE with external, exactly-once next-token alignment."""
    if logits.shape[:2] != labels.shape:
        raise ValueError("Pretraining logits and shifted labels have incompatible shapes.")
    count = int(labels.ne(-100).sum().item())
    if count == 0:
        raise ValueError("A pretraining batch has no supervised tokens.")
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum")
    return loss, count


def answer_logits(logits: torch.Tensor, last_positions: torch.Tensor) -> torch.Tensor:
    if last_positions.ndim != 1 or last_positions.shape[0] != logits.shape[0]:
        raise ValueError("One answer position is required for each prompt.")
    if bool(((last_positions < 0) | (last_positions >= logits.shape[1])).any()):
        raise ValueError("Answer position outside the nonempty prompt.")
    return logits[torch.arange(logits.shape[0], device=logits.device), last_positions].float()


def sft_loss(logits: torch.Tensor, last_positions: torch.Tensor, label_token_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    scores = answer_logits(logits, last_positions)
    if label_token_ids.shape != last_positions.shape or label_token_ids.numel() == 0:
        raise ValueError("SFT requires one nonempty label per prompt.")
    return F.cross_entropy(scores, label_token_ids, reduction="sum"), label_token_ids.numel()


def dpo_loss(policy_logits: torch.Tensor, reference_logits: torch.Tensor, last_positions: torch.Tensor,
             chosen_token_ids: torch.Tensor, rejected_token_ids: torch.Tensor, beta: float = 0.1) -> tuple[torch.Tensor, int]:
    if beta <= 0:
        raise ValueError("DPO beta must be positive.")
    if chosen_token_ids.shape != last_positions.shape or rejected_token_ids.shape != last_positions.shape:
        raise ValueError("DPO requires one chosen and one rejected label per prompt.")
    if chosen_token_ids.numel() == 0 or bool(chosen_token_ids.eq(rejected_token_ids).any()):
        raise ValueError("DPO chosen and rejected labels must be distinct and nonempty.")
    policy = answer_logits(policy_logits, last_positions).log_softmax(-1)
    reference = answer_logits(reference_logits.detach(), last_positions).log_softmax(-1)
    row = torch.arange(policy.shape[0], device=policy.device)
    advantage = (policy[row, chosen_token_ids] - policy[row, rejected_token_ids]) - (reference[row, chosen_token_ids] - reference[row, rejected_token_ids])
    return -F.logsigmoid(beta * advantage).sum(), chosen_token_ids.numel()


def _batch(dataset: Any, indices: list[int], stage: str, pad_id: int) -> dict[str, torch.Tensor]:
    items = [dataset[index] for index in indices]
    if stage in {"pretrain", "domain"}:
        return {key: torch.stack([item[key] for item in items]) for key in ("input_ids", "labels")}
    return collate_task(items, pad_id)


def _limit_supervision(batch: dict[str, torch.Tensor], stage: str, remaining: int) -> tuple[dict[str, torch.Tensor], bool]:
    if stage in {"pretrain", "domain"}:
        valid = batch["labels"].ne(-100).flatten().nonzero().flatten()
        if valid.numel() <= remaining:
            return batch, False
        labels = batch["labels"].clone()
        labels.flatten()[valid[remaining:]] = -100
        return {**batch, "labels": labels}, True
    size = batch["input_ids"].shape[0]
    return ({key: value[:remaining] for key, value in batch.items()}, True) if size > remaining else (batch, False)


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _loss(model: nn.Module, batch: dict[str, torch.Tensor], stage: str, reference: nn.Module | None, beta: float) -> tuple[torch.Tensor, int]:
    logits = model(batch["input_ids"], attention_mask=batch.get("attention_mask"))
    if stage in {"pretrain", "domain"}:
        return pretraining_loss(logits, batch["labels"])
    if stage == "sft":
        return sft_loss(logits, batch["last_positions"], batch["label_token_ids"])
    if reference is None:
        raise ValueError("DPO requires a frozen reference model.")
    with torch.no_grad():
        reference_logits = reference(batch["input_ids"], attention_mask=batch.get("attention_mask"))
    return dpo_loss(logits, reference_logits, batch["last_positions"], batch["chosen_token_ids"], batch["rejected_token_ids"], beta)


def _validate(model: nn.Module, dataset: Any, stage: str, batch_size: int, pad_id: int,
              label_ids: list[int], device: torch.device, amp: bool, max_batches: int | None) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss, count = 0.0, 0
    confusion = torch.zeros(3, 3, dtype=torch.int64)
    try:
        with torch.no_grad():
            for index, start in enumerate(range(0, len(dataset), batch_size)):
                if max_batches is not None and index >= max_batches:
                    break
                val_stage = stage if stage in {"pretrain", "domain"} else "sft"
                batch = _move(_batch(dataset, list(range(start, min(len(dataset), start + batch_size))), val_stage, pad_id), device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(batch["input_ids"], attention_mask=batch.get("attention_mask"))
                    if val_stage in {"pretrain", "domain"}:
                        loss, n = pretraining_loss(logits, batch["labels"])
                    else:
                        loss, n = sft_loss(logits, batch["last_positions"], batch["label_token_ids"])
                        predictions = answer_logits(logits, batch["last_positions"])[..., label_ids].argmax(-1).cpu().tolist()
                        truth = [label_ids.index(value) for value in batch["label_token_ids"].cpu().tolist()]
                        for gold, predicted in zip(truth, predictions):
                            confusion[gold, predicted] += 1
                total_loss += float(loss.item())
                count += n
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Validation contains no supervised examples/tokens.")
    result = {"loss": total_loss / count, "supervised_tokens": float(count)}
    if stage not in {"pretrain", "domain"}:
        f1s = []
        for label in range(3):
            tp = int(confusion[label, label])
            denominator = int(confusion[label].sum() + confusion[:, label].sum())
            f1s.append(2 * tp / denominator if denominator else 0.0)
        result["macro_f1"] = sum(f1s) / 3
    return result


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    supplied = config.get("train", {})
    defaults = dict(device="auto", seed=130, batch_size=4, grad_accumulation=8, seq_len=512,
                    max_updates=1000, max_tokens=None, learning_rate=3e-4, min_lr_ratio=0.1,
                    warmup_updates=50, weight_decay=0.1, clip_grad=1.0, save_every=100,
                    amp=True, num_workers=0, max_consecutive_amp_skips=5, validation_every=None,
                    validation_max_batches=None, schedule_updates=None, beta=0.1, max_epochs=None)
    defaults.update(supplied)
    if defaults["schedule_updates"] is None:
        defaults["schedule_updates"] = defaults["max_updates"]
    if defaults["validation_every"] is None:
        defaults["validation_every"] = defaults["save_every"]
    for key in ("batch_size", "grad_accumulation", "seq_len", "max_updates", "save_every", "schedule_updates", "validation_every"):
        if not isinstance(defaults[key], int) or isinstance(defaults[key], bool) or defaults[key] < 1:
            raise ValueError(f"train.{key} must be a positive integer.")
    if defaults["max_tokens"] is not None and (not isinstance(defaults["max_tokens"], int) or defaults["max_tokens"] < 1):
        raise ValueError("train.max_tokens must be a positive integer or null.")
    if defaults["max_epochs"] is not None:
        if not isinstance(defaults["max_epochs"], int) or isinstance(defaults["max_epochs"], bool) or defaults["max_epochs"] < 1:
            raise ValueError("train.max_epochs must be a positive integer or null.")
        if config["stage"] not in {"sft", "dpo"}:
            raise ValueError("train.max_epochs applies only to SFT/DPO; use max_tokens for pretraining.")
    if defaults["num_workers"] != 0:
        raise ValueError("Exact cursor resume currently requires train.num_workers=0.")
    if defaults["learning_rate"] <= 0 or defaults["clip_grad"] <= 0 or defaults["weight_decay"] < 0:
        raise ValueError("Learning rate/gradient clipping must be positive; weight decay must be nonnegative.")
    if not 0 <= defaults["min_lr_ratio"] <= 1 or defaults["warmup_updates"] < 0:
        raise ValueError("Invalid learning-rate schedule.")
    if defaults["validation_max_batches"] is not None and defaults["validation_max_batches"] < 1:
        raise ValueError("validation_max_batches must be positive or null.")
    return defaults


def _pretrain_transition(config: dict[str, Any], model_config: ModelConfig,
                         tokenizer_hash: str, provenance: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a frozen parent for an explicit expanded-data weights-only run.

    The parent is also checked on exact resume. Its content hash, rather than
    its filesystem location, binds the new run to its initialization lineage.
    No parent optimizer, schedule, RNG, counters, or data cursor is restored.
    """
    transition = config["pretrain_transition"]
    if not isinstance(transition, dict) or set(transition) - {
        "kind", "parent_checkpoint", "expected_parent_sha256", "min_parent_updates"
    }:
        raise ValueError("pretrain_transition requires the explicit data_expansion parent checkpoint and hash fields.")
    if config["stage"] != "pretrain" or config["mode"] != "formal" or transition.get("kind") != "data_expansion":
        raise ValueError("data_expansion is a formal pretrain-to-pretrain weights-only transition.")
    parent_path = transition.get("parent_checkpoint")
    expected_hash = transition.get("expected_parent_sha256")
    if not isinstance(parent_path, (str, Path)) or not str(parent_path):
        raise ValueError("pretrain_transition.parent_checkpoint must name a frozen checkpoint pair.")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise ValueError("pretrain_transition.expected_parent_sha256 must be a lowercase SHA256 digest.")
    minimum = transition.get("min_parent_updates", 300)
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise ValueError("pretrain_transition.min_parent_updates must be a positive integer.")
    parent_path = Path(parent_path)
    if not config.get("resume") and not config.get("init_checkpoint"):
        raise ValueError("A new data_expansion run requires init_checkpoint matching its frozen parent.")
    if config.get("init_checkpoint") and Path(config["init_checkpoint"]).resolve() != parent_path.resolve():
        raise ValueError("init_checkpoint must match pretrain_transition.parent_checkpoint.")
    if sha256_file(parent_path) != expected_hash:
        raise ValueError("Data-expansion parent checkpoint SHA256 mismatch.")
    parent = load_checkpoint(parent_path, allow_previous=False)
    # Bind the exact pair, never an automatically selected previous generation.
    if sha256_file(parent_path) != expected_hash:
        raise ValueError("Data-expansion parent checkpoint changed during verification.")
    _check_compatible(parent, model_config, tokenizer_hash, "formal")
    parent_config = parent.get("config", {})
    parent_provenance = parent.get("provenance", {})
    if parent_config.get("stage") != "pretrain" or parent_provenance.get("stage") != "pretrain":
        raise ValueError("A data-expansion parent must be a formal pretrain checkpoint.")
    if (parent_provenance.get("mode") != "formal" or parent_provenance.get("model_config") != asdict(model_config)
            or parent_provenance.get("tokenizer_sha256") != tokenizer_hash):
        raise ValueError("Data-expansion parent provenance is inconsistent with its model or tokenizer.")
    progress = parent.get("progress", {})
    updates, tokens = progress.get("updates"), progress.get("tokens")
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < minimum:
        raise ValueError(f"Data-expansion parent requires at least {minimum} completed optimizer updates.")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
        raise ValueError("Data-expansion parent must record positive supervised-token progress.")
    parent_settings = parent_config.get("train", {})
    update_cap, token_cap = parent_settings.get("max_updates"), parent_settings.get("max_tokens")
    finished = (isinstance(update_cap, int) and not isinstance(update_cap, bool) and updates >= update_cap) or (
        isinstance(token_cap, int) and not isinstance(token_cap, bool) and tokens >= token_cap)
    if not finished:
        raise ValueError("Data-expansion parent has not reached its own completed training budget.")
    parent_data = parent_provenance.get("data_sha256")
    if not isinstance(parent_data, str) or len(parent_data) != 64 or not parent_provenance.get("validation_sha256"):
        raise ValueError("Data-expansion parent must record formal training and validation identities.")
    if parent_data == provenance["data_sha256"]:
        raise ValueError("data_expansion requires a distinct training dataset; use exact resume for unchanged data.")
    protected_directories = {parent_path.parent.resolve()}
    for key in ("output_dir", "backup_dir"):
        if parent_config.get(key):
            protected_directories.add(Path(parent_config[key]).resolve())
    for key in ("output_dir", "backup_dir"):
        if config.get(key):
            target = Path(config[key]).resolve()
            if any(target == old or target.is_relative_to(old) or old.is_relative_to(target) for old in protected_directories):
                raise ValueError("Data-expansion output and backups must use new directories and preserve the parent run.")
    lineage = {
        "kind": "data_expansion", "initialization": "weights_only",
        "parent_checkpoint_sha256": expected_hash,
        "parent_provenance_sha256": _json_digest(parent_provenance),
        "parent_updates": updates, "parent_tokens": tokens,
        "parent_data_sha256": parent_data,
        "parent_validation_sha256": parent_provenance["validation_sha256"],
        "min_parent_updates": minimum,
    }
    return parent, lineage


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """Execute a bounded training stage using a local tokenizer and prepared data.

    ``max_tokens`` counts nonpadding supervised positions, not prompt tokens.
    DPO counts one chosen label per preference pair. Optional task-stage
    ``max_epochs`` caps labels at dataset size times epochs, including across
    accumulated microbatches. ``input_tokens`` separately
    records prompt/sequence work. Extend a run's update stop only with an unchanged
    ``schedule_updates``; exact resume binds the optimizer and dataset settings.
    """
    config = copy.deepcopy(config)
    stage = config.get("stage")
    if stage not in {"pretrain", "domain", "sft", "dpo"}:
        raise ValueError("stage must be pretrain, domain, sft, or dpo.")
    mode = config.get("mode", "formal")
    if mode not in {"smoke", "formal"}:
        raise ValueError("mode must be smoke or formal.")
    settings = _settings(config)
    config["train"] = settings
    config["mode"] = mode
    device = _device(settings["device"])
    amp = bool(settings["amp"] and device.type == "cuda")
    _seed_all(settings["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    tokenizer = load_tokenizer(config["tokenizer"])
    tokenizer_hash = sha256_file(_tokenizer_file(config["tokenizer"]))
    if mode == "formal":
        _audit_formal_data(config, tokenizer_hash)
    label_ids = [tokenizer.token_to_id(token) for token in LABEL_TOKENS]
    if any(value is None for value in label_ids) or len(set(label_ids)) != 3:
        raise ValueError("Tokenizer must reserve three distinct atomic stance tokens.")
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("Tokenizer must define <|pad|>.")
    model_config = ModelConfig(**config["model"])
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise ValueError("Tokenizer vocabulary and model vocabulary must match exactly.")
    if settings["seq_len"] > model_config.max_position_embeddings:
        raise ValueError("seq_len exceeds the model context length.")
    model = FinLLM(model_config).float().to(device)
    if stage in {"pretrain", "domain"}:
        manifest = json.loads(Path(config["data"]).read_text(encoding="utf-8"))
        if manifest.get("mode") != mode or manifest.get("tokenizer_sha256") != tokenizer_hash:
            raise ValueError("Token manifest mode/tokenizer does not match this run.")
        dataset = TokenBlockDataset(config["data"], settings["seq_len"])
    else:
        dataset = TaskDataset(config["data"], tokenizer, settings["seq_len"], preference=stage == "dpo")
    if len(dataset) == 0:
        raise ValueError("Cannot train on an empty dataset.")
    max_tokens = settings["max_tokens"]
    if settings["max_epochs"] is not None:
        epoch_token_cap = len(dataset) * settings["max_epochs"]
        max_tokens = epoch_token_cap if max_tokens is None else min(max_tokens, epoch_token_cap)
    validation = None
    if config.get("validation_data"):
        validation = TokenBlockDataset(config["validation_data"], settings["seq_len"]) if stage in {"pretrain", "domain"} else TaskDataset(config["validation_data"], tokenizer, settings["seq_len"], preference=False)
    if mode == "formal" and settings["validation_max_batches"] is not None and stage in {"sft", "dpo"}:
        raise ValueError("Formal SFT/DPO model selection must evaluate the full development set.")

    semantic_keys = ("seed", "batch_size", "grad_accumulation", "seq_len", "learning_rate", "min_lr_ratio", "warmup_updates", "weight_decay", "clip_grad", "schedule_updates", "beta")
    provenance = {
        "mode": mode, "stage": stage, "model_config": asdict(model_config),
        "tokenizer_sha256": tokenizer_hash, "data_sha256": _data_digest(config["data"]),
        "validation_sha256": _data_digest(config["validation_data"]) if config.get("validation_data") else None,
        "training_settings": {key: settings[key] for key in semantic_keys},
        "device_type": device.type, "amp": amp,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
    }
    init = None
    if config.get("init_checkpoint") and config.get("resume"):
        raise ValueError("Choose init_checkpoint for a new stage or resume, not both.")
    if "pretrain_transition" in config:
        init, lineage = _pretrain_transition(config, model_config, tokenizer_hash, provenance)
        provenance["parent_lineage"] = lineage
        if not config.get("resume"):
            model.load_state_dict(init["model_state"], strict=True)
    elif config.get("init_checkpoint"):
        if stage == "pretrain":
            raise ValueError("Pretraining starts from random initialization; use resume to continue it.")
        init = load_checkpoint(config["init_checkpoint"], allow_previous=False)
        _check_compatible(init, model_config, tokenizer_hash, mode)
        allowed = {"domain": {"pretrain", "domain"}, "sft": {"pretrain", "domain", "sft"}, "dpo": {"sft"}}
        if init["config"]["stage"] not in allowed[stage]:
            raise ValueError(f"Invalid initialization stage for {stage}.")
        model.load_state_dict(init["model_state"], strict=True)
    elif stage != "pretrain" and not config.get("resume"):
        raise ValueError(f"{stage} requires an init_checkpoint or resume.")

    reference = None
    if stage == "dpo":
        reference_path = config.get("reference_checkpoint") or config.get("init_checkpoint")
        if not reference_path:
            raise ValueError("DPO requires its fixed SFT reference_checkpoint, including on resume.")
        reference_payload = load_checkpoint(reference_path, allow_previous=False)
        _check_compatible(reference_payload, model_config, tokenizer_hash, mode)
        if reference_payload["config"]["stage"] != "sft":
            raise ValueError("The DPO reference must be a frozen SFT checkpoint.")
        reference = FinLLM(model_config).float().to(device)
        reference.load_state_dict(reference_payload["model_state"], strict=True)
        reference.requires_grad_(False).eval()
        provenance["reference_sha256"] = sha256_file(reference_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    def lr_factor(step: int) -> float:
        warmup, horizon = settings["warmup_updates"], settings["schedule_updates"]
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        fraction = min(1.0, max(0.0, (step - warmup) / max(1, horizon - warmup)))
        return settings["min_lr_ratio"] + (1 - settings["min_lr_ratio"]) * (1 + math.cos(math.pi * fraction)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    progress: dict[str, Any] = dict(updates=0, attempted_updates=0, tokens=0, successful_tokens=0, input_tokens=0,
                                   epoch=0, cursor=0, best_validation_loss=None, best_macro_f1=None,
                                   consecutive_amp_skips=0, partial_token_batch=False)
    resumed = None
    if config.get("resume"):
        resumed = load_checkpoint(config["resume"])
        _check_compatible(resumed, model_config, tokenizer_hash, mode)
        if resumed["provenance"] != provenance:
            raise ValueError("Resume provenance mismatch: data, configuration, tokenizer, reference, device, AMP, or runtime changed. Keep schedule_updates fixed when extending max_updates.")
        if resumed["progress"].get("partial_token_batch") and (max_tokens is None or max_tokens > resumed["progress"]["tokens"]):
            raise ValueError("Cannot extend an exact-resume run after a partially consumed token-cap batch.")
        model.load_state_dict(resumed["model_state"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state"])
        scheduler.load_state_dict(resumed["scheduler_state"])
        scaler.load_state_dict(resumed["scaler_state"])
        progress.update(resumed["progress"])
        restore_rng_state(resumed["rng_state"])

    cursor = DataCursor(len(dataset), settings["batch_size"], settings["seed"], progress["epoch"], progress["cursor"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    backup_directory = Path(config["backup_dir"]) if config.get("backup_dir") else None
    if backup_directory is not None and backup_directory.resolve() == output.resolve():
        raise ValueError("backup_dir must differ from the working checkpoint output_dir.")
    latest_path, best_path = output / "latest.pt", output / "best.pt"
    if latest_path.exists() and not config.get("resume"):
        raise ValueError(f"Output directory already contains a run; resume it or use a new directory: {output}")
    model.train()
    last_loss: float | None = resumed.get("last_loss") if resumed is not None else None
    last_validation: dict[str, float] | None = resumed.get("last_validation") if resumed is not None else None
    start = time.perf_counter()
    start_tokens = progress["tokens"]
    backed_up_paths: dict[str, str] = {}
    saved_attempts: dict[str, int] = {}

    def session_report(elapsed: float, no_training: bool = False) -> dict[str, Any]:
        report = {
            "stage": stage, "mode": mode, "updates": progress["updates"], "attempted_updates": progress["attempted_updates"],
            "tokens": progress["tokens"], "successful_tokens": progress["successful_tokens"], "input_tokens": progress["input_tokens"],
            "token_definition": "nonpadding supervised positions; one chosen label per DPO pair",
            "latest_checkpoint": str(latest_path), "best_checkpoint": str(best_path) if best_path.exists() else None,
            "loss": last_loss, "validation": last_validation, "elapsed_seconds": elapsed,
            "tokens_per_second": None if no_training else (progress["tokens"] - start_tokens) / max(elapsed, 1e-9),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" and not no_training else None,
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "epoch": cursor.epoch, "cursor": cursor.cursor,
            "completed_epochs": cursor.epoch + int(cursor.cursor == len(dataset)),
            "max_epochs": settings["max_epochs"], "effective_max_tokens": max_tokens,
            "backup_checkpoints": backed_up_paths,
            "resumed_no_training": no_training,
        }
        if "parent_lineage" in provenance:
            report["parent_lineage"] = provenance["parent_lineage"]
        return report

    # Run-all is common in Colab. A verified completed checkpoint is a no-op:
    # do not replace it, upload it again, or erase its measured pilot summary.
    budget_complete = progress["updates"] >= settings["max_updates"] or (max_tokens is not None and progress["tokens"] >= max_tokens)
    if resumed is not None and budget_complete:
        report = session_report(0.0, no_training=True)
        if Path(config["resume"]).resolve() != latest_path.resolve():
            report["latest_checkpoint"] = str(config["resume"])
        try:
            previous_report = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous_report = None
        measured_keys = {"elapsed_seconds", "tokens_per_second", "peak_vram_bytes", "backup_checkpoints", "resumed_no_training"}
        if isinstance(previous_report, dict) and all(previous_report.get(key) == value for key, value in report.items() if key not in measured_keys):
            # Retain the original session's throughput and memory measurements;
            # the return flag distinguishes them from a new training session.
            return {**previous_report, "resumed_no_training": True}
        # Live checkpoint backups need not contain a completed run summary.
        # Checkpoint metrics are known; new-session throughput/VRAM are not.
        report_text = json.dumps(report, indent=2) + "\n"
        (output / "report.json").write_text(report_text, encoding="utf-8")
        (output / "run_summary.json").write_text(report_text, encoding="utf-8")
        return report

    def checkpoint(path: Path) -> None:
        progress.update(cursor.state_dict())
        save_checkpoint(path, {
            "model_config": asdict(model_config), "model_state": model.state_dict(), "tokenizer_hash": tokenizer_hash,
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
            "progress": dict(progress), "rng_state": capture_rng_state(), "provenance": provenance, "config": config,
            "last_loss": last_loss, "last_validation": last_validation,
        })
        if backup_directory is not None:
            backed_up_paths[path.name] = snapshot_to_drive(path, backup_directory)
        saved_attempts[path.name] = progress["attempted_updates"]

    while progress["updates"] < settings["max_updates"] and (max_tokens is None or progress["tokens"] < max_tokens):
        optimizer.zero_grad(set_to_none=True)
        accumulated_count, accumulated_loss, input_count = 0, 0.0, 0
        # Scale before backward to avoid large summed-CE gradients overflowing
        # FP16. Correct to the actual token count after the final microbatch.
        nominal_count = settings["batch_size"] * settings["grad_accumulation"] * (settings["seq_len"] if stage in {"pretrain", "domain"} else 1)
        for _ in range(settings["grad_accumulation"]):
            remaining = None if max_tokens is None else max_tokens - progress["tokens"] - accumulated_count
            if remaining is not None and remaining <= 0:
                break
            batch = _batch(dataset, cursor.next_indices(), stage, pad_id)
            if remaining is not None:
                batch, partial = _limit_supervision(batch, stage, remaining)
                progress["partial_token_batch"] = progress["partial_token_batch"] or partial
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss, count = _loss(model, batch, stage, reference, settings["beta"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite loss; training stopped without saving this incomplete update.")
            scaler.scale(loss / nominal_count).backward()
            accumulated_count += count
            accumulated_loss += float(loss.detach().item())
            input_count += int(batch["attention_mask"].sum().item()) if "attention_mask" in batch else int(batch["input_ids"].numel())
        if accumulated_count == 0:
            raise RuntimeError("No supervised positions accumulated for this optimizer update.")
        scaler.unscale_(optimizer)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(nominal_count / accumulated_count)
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad"], error_if_nonfinite=not amp)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        successful = scaler.get_scale() >= old_scale
        progress["attempted_updates"] += 1
        progress["tokens"] += accumulated_count
        progress["input_tokens"] += input_count
        progress.update(cursor.state_dict())
        last_loss = accumulated_loss / accumulated_count
        if successful:
            progress["updates"] += 1
            progress["successful_tokens"] += accumulated_count
            progress["consecutive_amp_skips"] = 0
            scheduler.step()
        else:
            progress["consecutive_amp_skips"] += 1
        optimizer.zero_grad(set_to_none=True)
        at_end = progress["updates"] >= settings["max_updates"] or (max_tokens is not None and progress["tokens"] >= max_tokens)
        if successful and validation is not None and (progress["updates"] % settings["validation_every"] == 0 or at_end):
            # Validation must not perturb the RNG stream of a resumed training run.
            rng = capture_rng_state()
            try:
                last_validation = _validate(model, validation, stage, settings["batch_size"], pad_id, label_ids, device, amp, settings["validation_max_batches"])
            finally:
                restore_rng_state(rng)
            if not math.isfinite(last_validation["loss"]):
                raise FloatingPointError("Validation loss is non-finite.")
            better = (progress["best_validation_loss"] is None or last_validation["loss"] < progress["best_validation_loss"]) if stage in {"pretrain", "domain"} else (progress["best_macro_f1"] is None or last_validation["macro_f1"] > progress["best_macro_f1"])
            if better:
                progress["best_validation_loss"] = last_validation["loss"]
                progress["best_macro_f1"] = last_validation.get("macro_f1")
                checkpoint(best_path)
        if at_end or progress["attempted_updates"] % settings["save_every"] == 0:
            checkpoint(latest_path)
        event = {"stage": stage, "updates": progress["updates"], "attempted_updates": progress["attempted_updates"], "supervised_tokens": progress["tokens"], "loss": last_loss, "learning_rate": optimizer.param_groups[0]["lr"], "amp_skipped": not successful}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
        if progress["consecutive_amp_skips"] >= settings["max_consecutive_amp_skips"]:
            checkpoint(latest_path)
            raise FloatingPointError("Repeated AMP overflow; saved a complete boundary checkpoint. Lower learning rate or inspect data before continuing.")

    if saved_attempts.get(latest_path.name) != progress["attempted_updates"]:
        checkpoint(latest_path)
    elapsed = time.perf_counter() - start
    report = session_report(elapsed)
    report_text = json.dumps(report, indent=2) + "\n"
    (output / "report.json").write_text(report_text, encoding="utf-8")
    (output / "run_summary.json").write_text(report_text, encoding="utf-8")
    return report


def _check_compatible(payload: dict[str, Any], model_config: ModelConfig, tokenizer_hash: str, mode: str) -> None:
    if payload.get("model_config") != asdict(model_config):
        raise ValueError("Checkpoint architecture does not match the configured model.")
    if payload.get("tokenizer_hash") != tokenizer_hash:
        raise ValueError("Checkpoint tokenizer hash does not match this tokenizer.")
    if payload.get("config", {}).get("mode") != mode:
        raise ValueError("Smoke and formal checkpoints cannot initialize each other.")
