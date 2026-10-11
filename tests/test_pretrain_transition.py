"""An expanded corpus is a new weights-only experiment, then exact resume."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import finllm.training as training
from finllm.checkpoint import load_checkpoint, marker_path, save_checkpoint, sha256_file
from finllm.tokenizer import train_tokenizer


@pytest.fixture(autouse=True)
def few_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def expansion(tmp_path):
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer = train_tokenizer([
        "$AAPL earnings strong buy bullish. $TSLA weak sell bearish. $MSFT reports neutral.",
        "Text target classify financial stance bearish neutral bullish. " * 10,
    ], tokenizer_directory, 384)
    tokenizer_hash = sha256_file(tokenizer_directory / "tokenizer.json")

    def pack(name, offset, count, partition):
        token_path = tmp_path / (name + ".bin")
        ((np.arange(count, dtype=np.uint16) + offset) % (tokenizer.get_vocab_size() - 6) + 6).tofile(token_path)
        manifest_path = tmp_path / (name + ".json")
        manifest_path.write_text(json.dumps({
            "format": "finllm-uint16-v1", "mode": "formal", "vocab_size": tokenizer.get_vocab_size(),
            "tokenizer_sha256": tokenizer_hash, "token_count": count, "partition": partition,
            "eval_exclusion_sha256": "frozen-test-exclusions", "source_sha256": name,
            "shards": [{"path": token_path.name, "count": count, "sha256": sha256_file(token_path)}],
        }), encoding="utf-8")
        return str(manifest_path)

    parent_config = {
        "mode": "formal", "stage": "pretrain", "tokenizer": str(tokenizer_directory),
        "data": pack("parent_train", 0, 193, "train"),
        "validation_data": pack("parent_validation", 0, 33, "validation"),
        "output_dir": str(tmp_path / "parent_run"),
        "model": {"vocab_size": tokenizer.get_vocab_size(), "hidden_size": 16, "num_hidden_layers": 1,
                  "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 32,
                  "max_position_embeddings": 256, "dropout": 0.2},
        "train": {"device": "cpu", "amp": False, "batch_size": 2, "grad_accumulation": 2,
                  "seq_len": 16, "max_updates": 2, "schedule_updates": 4, "warmup_updates": 0,
                  "seed": 17, "save_every": 1, "learning_rate": 0.001},
    }
    parent_report = training.run_training(parent_config)
    parent_path = parent_report["latest_checkpoint"]
    child = copy.deepcopy(parent_config)
    child.update(data=pack("expanded_train", 1, 513, "train"),
                 validation_data=pack("expanded_validation", 2, 49, "validation"),
                 output_dir=str(tmp_path / "expanded_run"), init_checkpoint=parent_path,
                 pretrain_transition={"kind": "data_expansion", "parent_checkpoint": parent_path,
                                      "expected_parent_sha256": sha256_file(parent_path), "min_parent_updates": 2})
    child["train"].update(max_updates=1, seed=43)
    return child, parent_path


def test_expansion_loads_parent_weights_and_resets_training_state(expansion, monkeypatch):
    config, parent_path = expansion
    parent = load_checkpoint(parent_path)
    parent_hash = sha256_file(parent_path)
    observed = []
    original_loss = training._loss

    def observe_initial_weights(model, *args, **kwargs):
        if not observed:
            observed.append(True)
            for key, tensor in model.state_dict().items():
                assert torch.equal(tensor.cpu(), parent["model_state"][key]), key
        return original_loss(model, *args, **kwargs)

    monkeypatch.setattr(training, "_loss", observe_initial_weights)
    report = training.run_training(config)
    child = load_checkpoint(report["latest_checkpoint"])
    assert observed and report["updates"] == child["progress"]["attempted_updates"] == 1
    assert report["tokens"] == 64 and report["tokens"] < parent["progress"]["tokens"]
    assert child["progress"]["epoch"] == 0 and child["progress"]["cursor"] == 4
    assert child["progress"]["cursor"] != parent["progress"]["cursor"]
    assert child["scheduler_state"]["last_epoch"] == 1
    for state in child["optimizer_state"]["state"].values():
        assert state["step"].item() == 1
    assert not torch.equal(child["rng_state"]["torch"], parent["rng_state"]["torch"])
    assert any(not torch.equal(value, parent["model_state"][key]) for key, value in child["model_state"].items())
    assert child["provenance"]["parent_lineage"] == report["parent_lineage"]
    assert report["parent_lineage"]["initialization"] == "weights_only"
    assert report["parent_lineage"]["parent_updates"] == 2
    assert report["parent_lineage"]["parent_checkpoint_sha256"] == parent_hash
    assert sha256_file(parent_path) == parent_hash


def test_expanded_exact_resume_twice_matches_uninterrupted_and_noop_preserves_summary(expansion, tmp_path, monkeypatch):
    config, parent_path = expansion
    full_config = copy.deepcopy(config)
    full_config["train"]["max_updates"] = 4
    full_config["output_dir"] = str(tmp_path / "uninterrupted")
    full = load_checkpoint(training.run_training(full_config)["latest_checkpoint"])
    first = training.run_training(config)
    resumed = copy.deepcopy(config)
    resumed.pop("init_checkpoint")
    resumed["resume"] = first["latest_checkpoint"]
    resumed["train"]["max_updates"] = 2
    second = training.run_training(resumed)
    assert second["updates"] == 2
    resumed["train"]["max_updates"] = 4
    report = training.run_training(resumed)
    finished = load_checkpoint(report["latest_checkpoint"])
    assert finished["progress"] == full["progress"]
    assert finished["scheduler_state"] == full["scheduler_state"]
    assert finished["provenance"] == full["provenance"]
    assert torch.equal(finished["rng_state"]["torch"], full["rng_state"]["torch"])
    for key, value in full["model_state"].items():
        assert torch.equal(value, finished["model_state"][key]), key
    for param_id, state in full["optimizer_state"]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, finished["optimizer_state"]["state"][param_id][key])
    run = Path(config["output_dir"])
    before = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Completed expanded-data exact resume must not train or overwrite measurements")

    monkeypatch.setattr(training, "_loss", forbidden)
    monkeypatch.setattr(training, "save_checkpoint", forbidden)
    assert training.run_training(resumed) == {**report, "resumed_no_training": True}
    assert before == {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}


@pytest.mark.parametrize("failure", [
    "wrong_hash", "missing_marker", "wrong_architecture", "wrong_tokenizer", "smoke_parent",
    "wrong_stage", "inconsistent_provenance", "insufficient_updates", "unfinished_budget",
    "same_data", "parent_output", "parent_backup", "nested_parent_output", "ancestor_parent_backup",
    "different_init", "missing_init", "smoke_child",
    "wrong_kind", "unknown_field", "invalid_minimum",
])
def test_expansion_refuses_invalid_parent_or_implicit_transition(expansion, tmp_path, failure):
    config, parent_path = expansion
    parent = load_checkpoint(parent_path)
    transition = config["pretrain_transition"]
    changed_parent = False
    if failure == "wrong_hash":
        transition["expected_parent_sha256"] = "0" * 64
    elif failure == "missing_marker":
        marker_path(parent_path).unlink()
    elif failure == "wrong_architecture":
        parent["model_config"]["hidden_size"] *= 2
        changed_parent = True
    elif failure == "wrong_tokenizer":
        parent["tokenizer_hash"] = "wrong-tokenizer"
        changed_parent = True
    elif failure == "smoke_parent":
        parent["config"]["mode"] = "smoke"
        changed_parent = True
    elif failure == "wrong_stage":
        parent["config"]["stage"] = "domain"
        changed_parent = True
    elif failure == "inconsistent_provenance":
        parent["provenance"]["tokenizer_sha256"] = "wrong-tokenizer"
        changed_parent = True
    elif failure == "insufficient_updates":
        transition["min_parent_updates"] = 300
    elif failure == "unfinished_budget":
        parent["config"]["train"]["max_updates"] = 3
        changed_parent = True
    elif failure == "same_data":
        config["data"] = parent["config"]["data"]
    elif failure == "parent_output":
        config["output_dir"] = parent["config"]["output_dir"]
    elif failure == "parent_backup":
        config["backup_dir"] = parent["config"]["output_dir"]
    elif failure == "nested_parent_output":
        config["output_dir"] = str(Path(parent["config"]["output_dir"]) / "expanded")
    elif failure == "ancestor_parent_backup":
        config["backup_dir"] = str(Path(parent["config"]["output_dir"]).parent)
    elif failure == "different_init":
        config["init_checkpoint"] = str(tmp_path / "other.pt")
    elif failure == "missing_init":
        config.pop("init_checkpoint")
    elif failure == "smoke_child":
        config["mode"] = "smoke"
        path = Path(config["data"])
        manifest = json.loads(path.read_text())
        manifest["mode"] = "smoke"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    elif failure == "wrong_kind":
        transition["kind"] = "resume"
    elif failure == "unknown_field":
        transition["expected_parnt_sha256"] = transition["expected_parent_sha256"]
    elif failure == "invalid_minimum":
        transition["min_parent_updates"] = True
    if changed_parent:
        save_checkpoint(parent_path, parent)
        transition["expected_parent_sha256"] = sha256_file(parent_path)
    expected = {
        "wrong_hash": "SHA256 mismatch", "missing_marker": "Incomplete checkpoint",
        "wrong_architecture": "architecture does not match", "wrong_tokenizer": "tokenizer hash does not match",
        "smoke_parent": "Smoke and formal", "wrong_stage": "formal pretrain checkpoint",
        "inconsistent_provenance": "provenance is inconsistent", "insufficient_updates": "at least 300",
        "unfinished_budget": "completed training budget", "same_data": "distinct training dataset",
        "parent_output": "new directories", "parent_backup": "new directories",
        "nested_parent_output": "new directories", "ancestor_parent_backup": "new directories",
        "different_init": "init_checkpoint must match", "missing_init": "requires init_checkpoint",
        "smoke_child": "formal pretrain-to-pretrain", "wrong_kind": "formal pretrain-to-pretrain",
        "unknown_field": "explicit data_expansion", "invalid_minimum": "positive integer",
    }[failure]
    with pytest.raises(ValueError, match=expected):
        training.run_training(config)
    if failure != "parent_output":
        assert not (Path(config["output_dir"]) / "latest.pt").exists()


def test_ordinary_pretrain_init_and_same_stage_resume_contract_remain_strict(expansion):
    config, parent_path = expansion
    implicit = copy.deepcopy(config)
    implicit.pop("pretrain_transition")
    with pytest.raises(ValueError, match="random initialization"):
        training.run_training(implicit)
    conflicting = copy.deepcopy(config)
    conflicting["resume"] = parent_path
    with pytest.raises(ValueError, match="not both"):
        training.run_training(conflicting)
    same_stage_resume = copy.deepcopy(implicit)
    same_stage_resume.pop("init_checkpoint")
    same_stage_resume["resume"] = parent_path
    with pytest.raises(ValueError, match="Resume provenance mismatch"):
        training.run_training(same_stage_resume)


@pytest.mark.parametrize("change", ["missing_lineage", "changed_parent", "changed_data", "changed_schedule"])
def test_expansion_resume_is_bound_to_parent_and_new_data(expansion, change):
    config, parent_path = expansion
    report = training.run_training(config)
    resumed = copy.deepcopy(config)
    resumed.pop("init_checkpoint")
    resumed["resume"] = report["latest_checkpoint"]
    resumed["train"]["max_updates"] = 2
    if change == "missing_lineage":
        resumed.pop("pretrain_transition")
    elif change == "changed_parent":
        parent = load_checkpoint(parent_path)
        parent["model_state"][next(iter(parent["model_state"]))].add_(0.01)
        save_checkpoint(parent_path, parent)
        resumed["pretrain_transition"]["expected_parent_sha256"] = sha256_file(parent_path)
    elif change == "changed_data":
        manifest_path = Path(resumed["data"])
        manifest = json.loads(manifest_path.read_text())
        manifest["source_sha256"] = "different-expanded-source"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        resumed["train"]["schedule_updates"] += 1
    before = sha256_file(report["latest_checkpoint"])
    with pytest.raises(ValueError, match="Resume provenance mismatch"):
        training.run_training(resumed)
    assert sha256_file(report["latest_checkpoint"]) == before
