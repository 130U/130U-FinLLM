import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from finllm.checkpoint import load_checkpoint, sha256_file
from finllm.tokenizer import train_tokenizer
from finllm.training import DataCursor, dpo_loss, pretraining_loss, run_training, sft_loss


@pytest.fixture(autouse=True)
def few_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_config(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer = train_tokenizer([
        "$AAPL earnings strong buy bullish. $TSLA weak sell bearish. $MSFT reports neutral.",
        "Text target classify financial stance bearish neutral bullish. " * 10,
    ], tokenizer_dir, 384)
    token_path = tmp_path / "tokens.bin"
    (np.arange(193, dtype=np.uint16) % (tokenizer.get_vocab_size() - 6) + 6).tofile(token_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "format": "finllm-uint16-v1", "mode": "smoke", "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": sha256_file(tokenizer_dir / "tokenizer.json"), "token_count": 193,
        "source_sha256": "test-source", "shards": [{"path": token_path.name, "count": 193, "sha256": sha256_file(token_path)}],
    }), encoding="utf-8")
    return {
        "mode": "smoke", "stage": "pretrain", "tokenizer": str(tokenizer_dir), "data": str(manifest_path),
        "output_dir": str(tmp_path / "run"),
        "model": {"vocab_size": tokenizer.get_vocab_size(), "hidden_size": 16, "num_hidden_layers": 1,
                  "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 32,
                  "max_position_embeddings": 256, "dropout": 0.2},
        "train": {"device": "cpu", "amp": False, "batch_size": 2, "grad_accumulation": 2,
                  "seq_len": 16, "max_updates": 4, "schedule_updates": 4, "warmup_updates": 0,
                  "seed": 17, "save_every": 1, "learning_rate": 0.001},
    }


def test_pretraining_alignment_has_no_second_shift_and_ignores_padding():
    logits = torch.full((1, 3, 5), -8.0, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 2] = 8
        logits[0, 1, 3] = 8
    labels = torch.tensor([[2, 3, -100]])
    loss, count = pretraining_loss(logits, labels)
    assert count == 2 and loss.item() < 1e-4
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 2]).item() == 0


def test_sft_only_answer_position_full_vocabulary_ce():
    logits = torch.zeros((2, 4, 9), requires_grad=True)
    loss, count = sft_loss(logits, torch.tensor([1, 3]), torch.tensor([3, 5]))
    assert count == 2
    assert loss.item() == pytest.approx(2 * np.log(9), rel=1e-6)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 0]).item() == 0
    assert torch.count_nonzero(logits.grad[0, 2:]).item() == 0
    assert torch.count_nonzero(logits.grad[1, :3]).item() == 0
    assert logits.grad[0, 1, 8] > 0  # CE is not restricted to the three class IDs.


def test_dpo_reference_is_detached_and_chosen_margin_is_rewarded():
    policy = torch.zeros((1, 2, 8), requires_grad=True)
    reference = torch.zeros_like(policy, requires_grad=True)
    positions, chosen, rejected = torch.tensor([1]), torch.tensor([5]), torch.tensor([3])
    loss, count = dpo_loss(policy, reference, positions, chosen, rejected)
    assert count == 1 and loss.item() == pytest.approx(np.log(2))
    loss.backward()
    assert reference.grad is None
    assert policy.grad[0, 1, 5] < 0 < policy.grad[0, 1, 3]
    assert torch.count_nonzero(policy.grad[0, 0]) == 0
    with pytest.raises(ValueError, match="distinct"):
        dpo_loss(policy, reference, positions, chosen, chosen)


def test_cursor_resume_next_batch_across_epoch():
    cursor = DataCursor(5, 3, 11)
    cursor.next_indices()
    state = cursor.state_dict()
    restored = DataCursor(5, 3, 11, **state)
    for _ in range(5):
        assert restored.next_indices() == cursor.next_indices()


def test_full_state_resume_matches_uninterrupted_with_dropout(tiny_config, tmp_path):
    uninterrupted = run_training(tiny_config)
    full = load_checkpoint(uninterrupted["latest_checkpoint"])
    interrupted = copy.deepcopy(tiny_config)
    interrupted["output_dir"] = str(tmp_path / "interrupted")
    interrupted["train"]["max_updates"] = 2
    halfway = run_training(interrupted)
    resumed_config = copy.deepcopy(interrupted)
    resumed_config["train"]["max_updates"] = 4
    resumed_config["resume"] = halfway["latest_checkpoint"]
    resumed = load_checkpoint(run_training(resumed_config)["latest_checkpoint"])
    assert full["progress"] == resumed["progress"]
    assert full["scheduler_state"] == resumed["scheduler_state"]
    assert torch.equal(full["rng_state"]["torch"], resumed["rng_state"]["torch"])
    for key in full["model_state"]:
        assert torch.equal(full["model_state"][key], resumed["model_state"][key]), key
    for param_id, state in full["optimizer_state"]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, resumed["optimizer_state"]["state"][param_id][key])
    a = DataCursor(12, 2, 17, full["progress"]["epoch"], full["progress"]["cursor"])
    b = DataCursor(12, 2, 17, resumed["progress"]["epoch"], resumed["progress"]["cursor"])
    assert a.next_indices() == b.next_indices()


def test_exact_token_cap_and_resume_reject_changed_data(tiny_config):
    tiny_config["train"]["max_tokens"] = 37
    result = run_training(tiny_config)
    assert result["tokens"] == 37 and result["updates"] == 1
    resumed = copy.deepcopy(tiny_config)
    resumed["resume"] = result["latest_checkpoint"]
    resumed["train"]["max_tokens"] = 38
    with pytest.raises(ValueError, match="partially consumed"):
        run_training(resumed)
    resumed["train"]["max_tokens"] = 37
    resumed["train"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="provenance mismatch"):
        run_training(resumed)


def test_full_sft_dpo_update_preserves_reference(tiny_config, tmp_path):
    tiny_config["train"]["max_updates"] = 1
    pretrain = run_training(tiny_config)
    rows = [{"text": "$AAPL earnings strong.", "target": "AAPL", "label": "bullish"},
            {"text": "$TSLA margins weak.", "target": "TSLA", "label": "bearish"}]
    training = tmp_path / "task.jsonl"
    training.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    sft = copy.deepcopy(tiny_config)
    sft.update(stage="sft", data=str(training), output_dir=str(tmp_path / "sft"), init_checkpoint=pretrain["latest_checkpoint"])
    sft["train"]["seq_len"] = 256
    sft_result = run_training(sft)
    reference = sft_result["latest_checkpoint"]
    reference_hash = sha256_file(reference)
    preferences = tmp_path / "preferences.jsonl"
    preferences.write_text("".join(json.dumps(dict(row, chosen=row["label"], rejected="neutral")) + "\n" for row in rows), encoding="utf-8")
    dpo = copy.deepcopy(sft)
    dpo.update(stage="dpo", data=str(preferences), output_dir=str(tmp_path / "dpo"), init_checkpoint=reference, reference_checkpoint=reference)
    dpo_result = run_training(dpo)
    assert dpo_result["updates"] == 1
    assert dpo_result["tokens"] == 4  # Two label pairs per batch, two accumulation steps.
    assert sha256_file(reference) == reference_hash
    before, after = load_checkpoint(reference), load_checkpoint(dpo_result["latest_checkpoint"])
    assert any(not torch.equal(before["model_state"][key], after["model_state"][key]) for key in before["model_state"])


def test_formal_run_rejects_smoke_data(tiny_config):
    tiny_config["mode"] = "formal"
    with pytest.raises(ValueError, match="validation_data"):
        run_training(tiny_config)


def test_scheduler_does_not_advance_on_skipped_optimizer_step(tiny_config, monkeypatch):
    import finllm.training as module

    class OneSkippedStep:
        """Simulate scaler overflow once without requiring a CUDA machine."""
        def __init__(self, *args, **kwargs):
            self.attempt = 0
            self.factor = 2.0

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def get_scale(self):
            return self.factor

        def step(self, optimizer):
            if self.attempt > 0:
                optimizer.step()

        def update(self):
            if self.attempt == 0:
                self.factor /= 2
            self.attempt += 1

        def state_dict(self):
            return {"attempt": self.attempt, "factor": self.factor}

    monkeypatch.setattr(module.torch.amp, "GradScaler", OneSkippedStep)
    tiny_config["train"]["max_updates"] = 1
    result = run_training(tiny_config)
    payload = load_checkpoint(result["latest_checkpoint"])
    assert result["updates"] == 1 and result["attempted_updates"] == 2
    assert payload["scheduler_state"]["last_epoch"] == 1
    assert result["tokens"] == 128 and result["successful_tokens"] == 64


def test_task_epoch_cap_stops_exactly_across_accumulation(tiny_config, tmp_path, monkeypatch):
    from collections import Counter
    tiny_config["train"]["max_updates"] = 1
    pretrained = run_training(tiny_config)
    task_path = tmp_path / "three_examples.jsonl"
    task_path.write_text("".join(json.dumps(row) + "\n" for row in [
        {"text": "$AAPL earnings strong.", "target": "AAPL", "label": "bullish"},
        {"text": "$TSLA margins weak.", "target": "TSLA", "label": "bearish"},
        {"text": "$MSFT reports tomorrow.", "target": "MSFT", "label": "neutral"},
    ]), encoding="utf-8")
    config = copy.deepcopy(tiny_config)
    config.update(stage="sft", data=str(task_path), output_dir=str(tmp_path / "three_epochs"),
                  init_checkpoint=pretrained["latest_checkpoint"])
    config["train"].update(seq_len=256, grad_accumulation=3, max_updates=20,
                           schedule_updates=20, max_epochs=3)
    consumed = []
    original = DataCursor.next_indices

    def record(cursor):
        indices = original(cursor)
        consumed.extend(indices)
        return indices

    monkeypatch.setattr(DataCursor, "next_indices", record)
    result = run_training(config)
    assert result["tokens"] == result["effective_max_tokens"] == 9
    assert result["updates"] == 2 and result["completed_epochs"] == 3
    assert result["epoch"] == 2 and result["cursor"] == 3
    assert Counter(consumed) == {0: 3, 1: 3, 2: 3}


def test_periodic_backup_survives_later_training_failure(tiny_config, tmp_path, monkeypatch):
    import finllm.training as module
    tiny_config["train"]["max_updates"] = 2
    tiny_config["backup_dir"] = str(tmp_path / "persistent_drive" / "pretrain")
    original_loss = module._loss
    calls = 0

    def interrupt_second_update(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > tiny_config["train"]["grad_accumulation"]:
            raise RuntimeError("simulated interrupted runtime")
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(module, "_loss", interrupt_second_update)
    with pytest.raises(RuntimeError, match="simulated interrupted runtime"):
        run_training(tiny_config)
    backup = Path(tiny_config["backup_dir"]) / "latest.pt"
    payload = load_checkpoint(backup, allow_previous=False)
    assert payload["progress"]["updates"] == 1
    assert payload["progress"]["tokens"] == 64
    assert not (Path(tiny_config["output_dir"]) / "run_summary.json").exists()
