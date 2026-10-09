"""Stream reviewed corpora into frozen, leakage-filtered pretraining inputs.

Corpus records live on disk in SQLite. Only the explicitly bounded exclusion
index (task candidates and validation documents) lives in Python memory.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import ExitStack
from pathlib import Path

from .data import DuplicateIndex, file_sha256, text_hash, validate_training_row, write_json


MAX_DOCUMENT_BYTES = 1_000_000
MAX_JSONL_LINE_BYTES = 8_000_000
MAX_EXCLUSION_DOCUMENTS = 100_000
MAX_EXCLUSION_TEXT_BYTES = 100_000_000


def _rows(path):
    """Reject oversized records rather than silently truncating documents."""
    with Path(path).open("rb") as stream:
        line_number = 0
        while line := stream.readline(MAX_JSONL_LINE_BYTES + 1):
            line_number += 1
            if len(line) > MAX_JSONL_LINE_BYTES:
                raise ValueError(f"{path}:{line_number}: JSONL record exceeds byte limit")
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode("utf-8-sig" if line_number == 1 else "utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid UTF-8 JSON") from error
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{path}:{line_number}: expected object with nonempty text")
            if len(row["text"].encode("utf-8")) > MAX_DOCUMENT_BYTES:
                raise ValueError(f"{path}:{line_number}: document exceeds {MAX_DOCUMENT_BYTES} bytes; no truncation applied")
            yield row


def _line(row):
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _digest(seed, value):
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _group(row):
    # A source URL is an article/document identity, not the publisher homepage.
    # Rows without that identity are themselves whole documents.
    if row.get("source_url"):
        if not isinstance(row["source_url"], str):
            raise ValueError("source_url must be a string when supplied")
        return "url:" + row["source_url"].strip()
    if row.get("document_id"):
        return "document:" + str(row["source"]) + ":" + str(row["document_id"])
    return "text:" + text_hash(row["text"])


def prepare_pretraining_inputs(
    general_input, finance_input, task_exclusions_input, output_dir,
    validation_fraction=.05, tokenizer_max_bytes=20_000_000, seed=130,
):
    """Prepare train/validation JSONL without loading either corpus into RAM.

    ``task_exclusions_input`` contains all candidate task evaluation texts; labels
    are optional. Freezing all candidates before task annotation is conservative.
    Corpus rights must already have been reviewed: this function verifies the
    metadata contract, not the underlying legal permission. ``source_url`` must
    identify a document; repeated chunks of that document retain the same split.

    The tokenizer budget counts actual UTF-8 JSONL bytes, including metadata.
    Whole rows are sampled in deterministic hash order, interleaving the sources;
    rows that cannot fit are skipped and never truncated. Both sources must fit.
    An unsuccessful operation leaves an incomplete directory that cannot be reused.
    """
    if not isinstance(validation_fraction, (int, float)) or not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be finite and strictly between 0 and 1")
    if not isinstance(tokenizer_max_bytes, int) or isinstance(tokenizer_max_bytes, bool) or tokenizer_max_bytes <= 0:
        raise ValueError("tokenizer_max_bytes must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    inputs = {"general": Path(general_input), "finance": Path(finance_input),
              "task_exclusions": Path(task_exclusions_input)}
    resolved = [path.resolve() for path in inputs.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("General, finance and task exclusion inputs must be distinct; reused input refused")
    input_hashes = {name: file_sha256(path) for name, path in inputs.items()}
    if len(set(input_hashes.values())) != len(input_hashes):
        raise ValueError("Identical input contents are reused; supply distinct corpora and task exclusions")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Preparation output already exists; choose a new versioned directory")
    output.mkdir(parents=True)
    incomplete = output / "INCOMPLETE"
    incomplete.write_text("Preparation has not completed successfully.\n", encoding="utf-8")
    exclusion_index = DuplicateIndex()
    exclusion_count = exclusion_bytes = 0

    def add_exclusion(row):
        nonlocal exclusion_count, exclusion_bytes
        exclusion_count += 1
        exclusion_bytes += len(row["text"].encode("utf-8"))
        if exclusion_count > MAX_EXCLUSION_DOCUMENTS or exclusion_bytes > MAX_EXCLUSION_TEXT_BYTES:
            raise ValueError("Exclusion index exceeds its documented document/text byte bound")
        exclusion_index.add(exclusion_count, row["text"])

    statistics = {name: {"input": 0, "task_excluded": 0, "exact_duplicates": 0,
                         "validation_overlap_removed": 0, "train": 0, "validation": 0}
                  for name in ("general", "finance")}
    counts = {}
    database_path = output / "preparation.sqlite"
    threshold = int(validation_fraction * (1 << 256))
    with ExitStack() as stack:
        task_frozen = stack.enter_context((output / "frozen_task_exclusions.jsonl").open("wb"))
        all_frozen = stack.enter_context((output / "frozen_exclusions.jsonl").open("wb"))
        for row in _rows(inputs["task_exclusions"]):
            add_exclusion(row)
            encoded = _line(row)
            task_frozen.write(encoded)
            all_frozen.write(encoded)
        task_count = exclusion_count
        if not task_count:
            raise ValueError("Task exclusions must contain at least one nonempty candidate evaluation text")
        connection = sqlite3.connect(database_path)
        stack.callback(connection.close)
        connection.execute("CREATE TABLE documents (hash TEXT PRIMARY KEY, domain TEXT, split TEXT, ordering TEXT, record TEXT)")
        for domain in ("general", "finance"):
            stat = statistics[domain]
            for row in _rows(inputs[domain]):
                stat["input"] += 1
                validate_training_row(row, smoke=False)
                if exclusion_index.matches(row["text"]):
                    stat["task_excluded"] += 1
                    continue
                digest = text_hash(row["text"])
                group = _group(row)
                split = "validation" if int(_digest(seed, group), 16) < threshold else "train"
                prepared = dict(row, split=split, document_hash=digest,
                                preparation_group=hashlib.sha256(group.encode("utf-8")).hexdigest())
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?, ?)",
                    (digest, domain, split, _digest(seed, "order:" + digest), json.dumps(prepared, ensure_ascii=False)),
                )
                if not cursor.rowcount:
                    stat["exact_duplicates"] += 1
                if stat["input"] % 1000 == 0:
                    connection.commit()
            if not stat["input"]:
                raise ValueError(f"{domain} input is empty")
        connection.commit()
        # Freeze validation first, then filter training against the whole frozen
        # pool. Exact duplicate prevention is global across both source corpora.
        for domain in ("general", "finance"):
            with (output / f"{domain}_validation.jsonl").open("wb") as stream:
                for (record,) in connection.execute(
                    "SELECT record FROM documents WHERE domain=? AND split='validation' ORDER BY ordering", (domain,),
                ):
                    row = json.loads(record)
                    add_exclusion(row)
                    encoded = _line(row)
                    stream.write(encoded)
                    all_frozen.write(encoded)
                    statistics[domain]["validation"] += 1
        for domain in ("general", "finance"):
            with (output / f"{domain}_train.jsonl").open("wb") as stream:
                for (record,) in connection.execute(
                    "SELECT record FROM documents WHERE domain=? AND split='train' ORDER BY ordering", (domain,),
                ):
                    row = json.loads(record)
                    if exclusion_index.matches(row["text"]):
                        statistics[domain]["validation_overlap_removed"] += 1
                        continue
                    stream.write(_line(row))
                    statistics[domain]["train"] += 1
            if not statistics[domain]["train"] or not statistics[domain]["validation"]:
                raise ValueError(f"{domain} needs nonempty train and validation after exclusions; provide more independent documents")
        tokenizer_counts = {"general": 0, "finance": 0}
        tokenizer_bytes = 0
        iterators = {domain: iter(_rows(output / f"{domain}_train.jsonl")) for domain in tokenizer_counts}
        with (output / "tokenizer_train.jsonl").open("wb") as stream:
            while iterators:
                for domain in list(iterators):
                    row = next(iterators[domain], None)
                    if row is None:
                        del iterators[domain]
                        continue
                    encoded = _line(row)
                    if tokenizer_bytes + len(encoded) <= tokenizer_max_bytes:
                        stream.write(encoded)
                        tokenizer_bytes += len(encoded)
                        tokenizer_counts[domain] += 1
        if not all(tokenizer_counts.values()):
            raise ValueError("Tokenizer byte budget must fit whole training rows from both general and finance sources")
        counts = {f"{domain}_{split}": statistics[domain][split]
                  for domain in statistics for split in ("train", "validation")}
        counts.update(tokenizer_train=sum(tokenizer_counts.values()), frozen_task_exclusions=task_count,
                      frozen_exclusions=exclusion_count)
    database_path.unlink()
    if input_hashes != {name: file_sha256(path) for name, path in inputs.items()}:
        raise ValueError("An input changed during preparation; frozen provenance cannot be established")
    files = {path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size,
                         "documents": counts[path.stem]}
             for path in sorted(output.glob("*.jsonl"))}
    manifest = {
        "format_version": 1, "mode": "formal", "seed": seed,
        "validation_fraction": validation_fraction,
        "split_method": "seeded document/source_url hash threshold before tokenization; globally exact deduplicated",
        "duplicate_method": "global normalized-exact SQLite; exclusion SimHash64 Hamming<=3",
        "input_sha256": input_hashes, "counts": counts, "source_statistics": statistics,
        "tokenizer": {"max_bytes": tokenizer_max_bytes, "actual_bytes": tokenizer_bytes,
                      "source_documents": tokenizer_counts, "sampling": "interleaved seeded hash order; whole training rows only"},
        "bounds": {"max_document_bytes": MAX_DOCUMENT_BYTES, "max_jsonl_line_bytes": MAX_JSONL_LINE_BYTES,
                   "max_exclusion_documents": MAX_EXCLUSION_DOCUMENTS, "max_exclusion_text_bytes": MAX_EXCLUSION_TEXT_BYTES},
        "files": files,
        "limitations": ["Rights metadata is required but is not independently established by this program.",
                        "Lexical duplicate checks do not detect every semantic paraphrase.",
                        "Hash partitions are non-temporal; held-out proportions are approximate.",
                        "No stance labels are inferred or fabricated; task candidates are excluded regardless of label."],
    }
    write_json(output / "preparation_manifest.json", manifest)
    incomplete.unlink()
    return manifest
