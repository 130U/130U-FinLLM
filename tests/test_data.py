import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from finllm.data import (
    DuplicateIndex, TokenBlockDataset, file_sha256, pack_corpus, prepare_task,
    read_jsonl, simhash, write_json, write_jsonl,
)
from finllm.tokenizer import EOS_TOKEN, train_tokenizer


def token_manifest(tmp_path, values, shard_sizes, vocab_size=64):
    shards, offset = [], 0
    for index, size in enumerate(shard_sizes):
        path = tmp_path / f"part-{index}.bin"
        np.asarray(values[offset:offset + size], dtype="<u2").tofile(path)
        shards.append({"path": path.name, "count": size, "sha256": file_sha256(path)})
        offset += size
    assert offset == len(values)
    manifest = {"format": "finllm-uint16-v1", "dtype": "<u2", "vocab_size": vocab_size,
                "token_count": len(values), "shards": shards, "mode": "smoke"}
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    return path


def test_cross_shard_next_token_pairs_and_tail_are_not_dropped(tmp_path):
    values = list(range(1, 15))
    dataset = TokenBlockDataset(token_manifest(tmp_path, values, [4, 3, 7]), seq_len=5)
    assert len(dataset) == 3
    inputs, labels = [], []
    for row in dataset:
        valid = row["labels"] != -100
        inputs.extend(row["input_ids"][valid].tolist())
        labels.extend(row["labels"][valid].tolist())
    assert inputs == values[:-1]
    assert labels == values[1:]
    # Token 4 -> 5 and token 7 -> 8 cross physical file boundaries.
    assert list(zip(inputs, labels))[3] == (4, 5)
    assert list(zip(inputs, labels))[6] == (7, 8)
    assert dataset[2]["labels"].tolist() == [12, 13, 14, -100, -100]
    assert dataset[2]["input_ids"].tolist() == [11, 12, 13, 0, 0]
    with pytest.raises(IndexError):
        dataset[len(dataset)]


@pytest.mark.parametrize("change", ["same_size_contents", "truncated"])
def test_changed_shard_is_rejected_before_training(tmp_path, change):
    manifest = token_manifest(tmp_path, [1, 2, 3, 4], [4])
    shard = tmp_path / "part-0.bin"
    shard.write_bytes(b"\x01\x00" * (4 if change == "same_size_contents" else 2))
    with pytest.raises(ValueError, match="Corrupt or changed"):
        TokenBlockDataset(manifest, 2)


def test_manifest_rejects_outside_path_and_invalid_token_id(tmp_path):
    outside = tmp_path / "outside.bin"
    np.asarray([1, 2, 3], dtype="<u2").tofile(outside)
    nested = tmp_path / "nested"
    nested.mkdir()
    manifest = {"format": "finllm-uint16-v1", "vocab_size": 64, "token_count": 3,
                "shards": [{"path": "../outside.bin", "count": 3, "sha256": file_sha256(outside)}]}
    write_json(nested / "manifest.json", manifest)
    with pytest.raises(ValueError, match="escapes"):
        TokenBlockDataset(nested / "manifest.json", 2)
    path = token_manifest(tmp_path, [1, 64, 3], [3], vocab_size=64)
    with pytest.raises(ValueError, match="vocabulary"):
        TokenBlockDataset(path, 2)


def near_duplicate_pair():
    # Many distinct 3-grams make one word change a genuinely near (not exact)
    # duplicate under the recorded SimHash detector. Pick deterministically.
    original = " ".join(f"financialterm{i}" for i in range(180))
    for index in range(1, 100):
        changed = original + f" revision{index}"
        if (simhash(original) ^ simhash(changed)).bit_count() <= 3:
            return original, changed
    raise AssertionError("Could not construct the specified SimHash fixture")


def task_row(index, text=None, **extra):
    date = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
    row = {"id": f"example-{index}", "text": text or f"Unique company topic{index} asset{index} event{index} item{index}.",
           "target": "AAPL", "label": ("bearish", "neutral", "bullish")[index % 3],
           "created_at": date.isoformat(), "source": "test fixture",
           "rights_note": "original synthetic fixture", "training_allowed": True}
    return dict(row, **extra)


def test_grouped_split_keeps_ids_events_and_near_duplicates_together(tmp_path):
    first, second = near_duplicate_pair()
    rows = [task_row(index) for index in range(30)]
    rows.extend([
        task_row(31, "A distinct target interpretation.", id=rows[0]["id"], target="MSFT"),
        task_row(32, "An unrelated wording but same verified event.", event_id="shared-event"),
        task_row(33, first), task_row(34, second),
    ])
    rows[2]["event_id"] = "shared-event"
    source, output = tmp_path / "task.jsonl", tmp_path / "task"
    write_jsonl(source, rows)
    report = prepare_task(source, output)
    combined = [row for name in ("train", "development", "calibration", "test")
                for row in read_jsonl(output / f"{name}.jsonl")]
    assert len(combined) == len(rows)
    assert report["split_method"] == "grouped_time"
    by_text = {row["text"]: row for row in combined}
    for a, b in [(rows[0], rows[30]), (rows[2], rows[31]), (rows[32], rows[33])]:
        assert by_text[a["text"]]["split"] == by_text[b["text"]]["split"]
        assert by_text[a["text"]]["group_id"] == by_text[b["text"]]["group_id"]
    group_splits = {}
    for row in combined:
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    frozen = list(read_jsonl(output / "frozen_eval.jsonl"))
    assert {row["group_id"] for row in frozen}.isdisjoint(
        row["group_id"] for row in combined if row["split"] == "train")
    temperature = {row["group_id"] for row in read_jsonl(output / "calibration_temperature.jsonl")}
    threshold = {row["group_id"] for row in read_jsonl(output / "calibration_threshold.jsonl")}
    assert temperature.isdisjoint(threshold)
    assert temperature | threshold == {row["group_id"] for row in combined if row["split"] == "calibration"}


def test_formal_packing_excludes_exact_and_near_heldout_text(tmp_path):
    original, near = near_duplicate_pair()
    excluded_path = tmp_path / "frozen_eval.jsonl"
    write_jsonl(excluded_path, [{"text": original}])
    kept = "$AAPL publishes its quarterly earnings next Thursday."
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [task_row(0, original), task_row(1, near), task_row(2, kept)])
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer = train_tokenizer([kept] * 4, tokenizer_dir, 320)
    manifest = pack_corpus(source, tokenizer_dir, tmp_path / "packed", shard_tokens=7, exclusions=excluded_path)
    report = json.loads(manifest.read_text(encoding="utf-8"))
    assert report["documents_kept"] == 1
    assert report["documents_excluded"] == 2
    actual = np.concatenate([np.fromfile(manifest.parent / row["path"], dtype="<u2") for row in report["shards"]])
    expected = tokenizer.encode(kept).ids + [tokenizer.token_to_id(EOS_TOKEN)]
    assert actual.tolist() == expected
    assert report["eval_exclusion_sha256"] == file_sha256(excluded_path)


def test_formal_packing_requires_eval_pool_and_training_rights(tmp_path):
    with pytest.raises(ValueError, match="frozen_eval"):
        pack_corpus(tmp_path / "not-read.jsonl", tmp_path / "tokenizer", tmp_path / "output")
    from finllm.data import validate_training_row
    with pytest.raises(ValueError, match="training_allowed"):
        validate_training_row({"text": "A sentence."})


def test_corpus_exact_duplicates_are_removed_and_validation_partition_is_preserved(tmp_path):
    text = "Different companies publish independent quarterly reports."
    source = tmp_path / "validation.jsonl"
    write_jsonl(source, [{"text": text, "split": "validation"},
                         {"text": text.upper(), "split": "validation"}])
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer = train_tokenizer([text] * 3, tokenizer_dir, 320)
    manifest = pack_corpus(source, tokenizer_dir, tmp_path / "packed", smoke=True, partition="validation")
    report = json.loads(manifest.read_text(encoding="utf-8"))
    assert report["partition"] == "validation"
    assert report["documents_kept"] == 1 and report["documents_excluded"] == 1
    assert report["token_count"] == len(tokenizer.encode(text).ids) + 1


def mixture_source(tmp_path, name, tokens):
    directory = tmp_path / name
    directory.mkdir()
    path = token_manifest(directory, tokens, [len(tokens)])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(tokenizer_sha256="same-frozen-tokenizer", partition="train",
                    eval_exclusion_sha256="same-frozen-eval")
    write_json(path, manifest)
    return path


def test_mixture_token_budgets_cycle_without_losing_or_inventing_tokens(tmp_path):
    from finllm.corpus import mix_corpora

    general = mixture_source(tmp_path, "general", [1, 2, 3, 4])
    finance = mixture_source(tmp_path, "finance", [11, 12, 13])
    output = tmp_path / "mixture"
    result = mix_corpora(general, finance, output, general_tokens=1030, finance_tokens=4097, seed=7)
    report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    tokens = np.concatenate([np.fromfile(output / row["path"], dtype="<u2") for row in report["shards"]])
    assert len(tokens) == 5127 == report["token_count"]
    assert int((tokens < 10).sum()) == 1030
    assert int((tokens > 10).sum()) == 4097
    assert tokens[tokens < 10].tolist() == ([1, 2, 3, 4] * 258)[:1030]
    assert tokens[tokens > 10].tolist() == ([11, 12, 13] * 1366)[:4097]
    assert result["mixture_token_counts"] == {"general": 1030, "finance": 4097}
    assert report["source_manifests"] == [file_sha256(general), file_sha256(finance)]


def test_mixture_rejects_source_hash_and_tokenizer_mismatches(tmp_path):
    from finllm.corpus import mix_corpora

    general = mixture_source(tmp_path, "general", [1, 2, 3, 4])
    finance = mixture_source(tmp_path, "finance", [11, 12, 13])
    manifest = json.loads(finance.read_text(encoding="utf-8"))
    manifest["tokenizer_sha256"] = "different-tokenizer"
    write_json(finance, manifest)
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        mix_corpora(general, finance, tmp_path / "bad-tokenizer", 5, 20)
    manifest["tokenizer_sha256"] = "same-frozen-tokenizer"
    write_json(finance, manifest)
    (finance.parent / "part-0.bin").write_bytes(b"\x0b\x00" * 3)
    with pytest.raises(ValueError, match="Corrupt or changed"):
        mix_corpora(general, finance, tmp_path / "bad-shard", 5, 20)
