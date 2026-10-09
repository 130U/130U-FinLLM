import json

import pytest

from finllm.data import read_jsonl, text_hash, write_jsonl
from finllm.preparation import MAX_DOCUMENT_BYTES, prepare_pretraining_inputs


def row(text, source_url=None):
    item = {"text": text, "source": "test fixture", "training_allowed": True,
            "rights_note": "Original test text; not a production corpus"}
    if source_url:
        item["source_url"] = source_url
    return item


def inputs(tmp_path):
    # Deliberately unrelated lexical documents keep the near-duplicate test
    # focused on planted contamination rather than repetitive fixture templates.
    import random
    rng = random.Random(31)
    def sentence():
        return " ".join("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=9)) for _ in range(25))
    general = [row(sentence(), f"https://general.example/doc/{index // 2}") for index in range(160)]
    finance = [row(sentence(), f"https://finance.example/doc/{index // 2}") for index in range(160)]
    heldout = {"text": "Target ACME faces insolvency after lenders rejected its restructuring proposal."}
    general += [row(heldout["text"]), dict(general[0])]
    finance += [dict(general[1])]
    paths = [tmp_path / name for name in ("general.jsonl", "finance.jsonl", "candidates.jsonl")]
    for path, records in zip(paths, (general, finance, [heldout])):
        write_jsonl(path, records)
    return paths, heldout


def test_prepares_frozen_train_only_bounded_inputs(tmp_path):
    paths, heldout = inputs(tmp_path)
    destination = tmp_path / "prepared"
    manifest = prepare_pretraining_inputs(*paths, destination, validation_fraction=.2, tokenizer_max_bytes=9_000)
    assert not (destination / "INCOMPLETE").exists()
    assert not (destination / "preparation.sqlite").exists()
    assert manifest["source_statistics"]["general"]["task_excluded"] == 1
    assert manifest["source_statistics"]["general"]["exact_duplicates"] == 1
    assert manifest["source_statistics"]["finance"]["exact_duplicates"] == 1
    training_hashes = set()
    validation_hashes = set()
    source_splits = {}
    for domain in ("general", "finance"):
        for split in ("train", "validation"):
            rows = list(read_jsonl(destination / f"{domain}_{split}.jsonl"))
            assert rows
            for item in rows:
                assert item["split"] == split
                assert item["text"] != heldout["text"]
                assert "label" not in item
                source_splits.setdefault(item["source_url"], set()).add(split)
                (training_hashes if split == "train" else validation_hashes).add(text_hash(item["text"]))
    assert all(len(splits) == 1 for splits in source_splits.values())
    assert not training_hashes & validation_hashes
    tokenizer_rows = list(read_jsonl(destination / "tokenizer_train.jsonl"))
    assert all(text_hash(item["text"]) in training_hashes and item["split"] == "train" for item in tokenizer_rows)
    assert (destination / "tokenizer_train.jsonl").stat().st_size <= 9_000
    assert all(manifest["tokenizer"]["source_documents"].values())
    frozen = list(read_jsonl(destination / "frozen_exclusions.jsonl"))
    assert {text_hash(item["text"]) for item in frozen} == validation_hashes | {text_hash(heldout["text"])}
    assert list(read_jsonl(destination / "frozen_task_exclusions.jsonl")) == [heldout]
    for name, info in manifest["files"].items():
        assert info["bytes"] == (destination / name).stat().st_size
        assert info["documents"] == len(list(read_jsonl(destination / name)))
    assert json.loads((destination / "preparation_manifest.json").read_text()) == manifest


def test_repeat_is_deterministic_but_existing_output_refused(tmp_path):
    paths, _ = inputs(tmp_path)
    first = prepare_pretraining_inputs(*paths, tmp_path / "one", validation_fraction=.2)
    second = prepare_pretraining_inputs(*paths, tmp_path / "two", validation_fraction=.2)
    assert first == second
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_pretraining_inputs(*paths, tmp_path / "one")
    with pytest.raises(ValueError, match="distinct"):
        prepare_pretraining_inputs(paths[0], paths[0], paths[2], tmp_path / "reused")


def test_near_validation_duplicate_is_removed_from_training(tmp_path):
    paths, _ = inputs(tmp_path)
    first = tmp_path / "initial"
    prepare_pretraining_inputs(*paths, first, validation_fraction=.2)
    train_url = next(read_jsonl(first / "general_train.jsonl"))["source_url"]
    validation_url = next(read_jsonl(first / "general_validation.jsonl"))["source_url"]
    validation_text = " ".join("word" + str(index) for index in range(200))
    almost_same_training_text = validation_text + " appendix"
    with paths[0].open("a", encoding="utf-8") as stream:
        for record in (row(validation_text, validation_url), row(almost_same_training_text, train_url)):
            stream.write(json.dumps(record) + "\n")
    destination = tmp_path / "filtered"
    manifest = prepare_pretraining_inputs(*paths, destination, validation_fraction=.2)
    assert manifest["source_statistics"]["general"]["validation_overlap_removed"] == 1
    assert validation_text in {item["text"] for item in read_jsonl(destination / "general_validation.jsonl")}
    assert almost_same_training_text not in {item["text"] for item in read_jsonl(destination / "general_train.jsonl")}
    assert almost_same_training_text not in {item["text"] for item in read_jsonl(destination / "tokenizer_train.jsonl")}


def test_empty_exclusions_and_budget_fail_explicitly(tmp_path):
    paths, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="budget"):
        prepare_pretraining_inputs(*paths, tmp_path / "small", validation_fraction=.2, tokenizer_max_bytes=10)
    assert (tmp_path / "small" / "INCOMPLETE").exists()
    paths[2].write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Task exclusions"):
        prepare_pretraining_inputs(*paths, tmp_path / "empty")


def test_formal_rights_and_document_bound_are_enforced(tmp_path):
    paths, _ = inputs(tmp_path)
    write_jsonl(paths[0], [{"text": "Reviewed training rights are absent."}])
    with pytest.raises(ValueError, match="training_allowed"):
        prepare_pretraining_inputs(*paths, tmp_path / "rights")
    write_jsonl(paths[0], [row("x" * (MAX_DOCUMENT_BYTES + 1))])
    with pytest.raises(ValueError, match="no truncation"):
        prepare_pretraining_inputs(*paths, tmp_path / "oversized")


def test_single_document_group_cannot_leak_across_splits(tmp_path):
    paths, _ = inputs(tmp_path)
    write_jsonl(paths[0], [row("One unique report has two passages.", "https://example.com/one-report"),
                         row("Another passage gives the separate financial outlook.", "https://example.com/one-report")])
    with pytest.raises(ValueError, match="nonempty train and validation"):
        prepare_pretraining_inputs(*paths, tmp_path / "one_document", validation_fraction=.2)
