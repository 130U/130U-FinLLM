"""Bounded corpus packing, grouped task splits, and explicit one-shift datasets."""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import EOS_TOKEN, LABELS, encode_prompt, label_token_ids, load_tokenizer


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_file(path):
    path = Path(path)
    return path / "tokenizer.json" if path.is_dir() else path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_text(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def text_hash(text):
    return hashlib.sha256(normalized_text(text).encode()).hexdigest()


def simhash(text):
    words = normalized_text(text).split()
    features = {" ".join(words[i:i + 3]) for i in range(max(1, len(words) - 2))}
    counts = [0] * 64
    for feature in features:
        value = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        for bit in range(64):
            counts[bit] += 1 if value & (1 << bit) else -1
    return sum(1 << bit for bit, count in enumerate(counts) if count > 0)


class DuplicateIndex:
    """Exact normalized hashes plus 64-bit SimHash Hamming<=3 candidates.

    This catches a specified lexical near-duplicate family, not all paraphrases.
    Bounded by the index size, not constant RAM; formal task pools are small.
    """
    def __init__(self):
        self.exact = defaultdict(set)
        self.bands = defaultdict(set)
        self.fingerprints = {}

    def matches(self, text):
        fingerprint = simhash(text)
        candidates = set(self.exact.get(text_hash(text), ()))
        for band in range(4):
            candidates.update(self.bands.get((band, (fingerprint >> (16 * band)) & 65535), ()))
        return {item for item in candidates if
                (fingerprint ^ self.fingerprints[item]).bit_count() <= 3}

    def add(self, item, text):
        fingerprint = simhash(text)
        self.exact[text_hash(text)].add(item)
        self.fingerprints[item] = fingerprint
        for band in range(4):
            self.bands[(band, (fingerprint >> (16 * band)) & 65535)].add(item)


def validate_training_row(row, smoke=False):
    if not isinstance(row.get("text"), str) or not row["text"].strip():
        raise ValueError("Every row needs nonempty text")
    if not smoke:
        if row.get("training_allowed") is not True or not row.get("source") or not row.get("rights_note"):
            raise ValueError("Formal data requires training_allowed=true, source and rights_note after source review")
        if row.get("synthetic") is True:
            raise ValueError("Synthetic samples belong to the separate smoke workflow")


def prepare_task(input_path, output_dir, smoke=False):
    """Group duplicate/ID/event-related examples before 70/10/10/10 splitting."""
    rows = list(read_jsonl(input_path))
    if len(rows) < 10:
        raise ValueError("Need at least 10 rows for grouped splits; tiny examples use smoke")
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    duplicate_index = DuplicateIndex()
    identifiers = {}
    for index, row in enumerate(rows):
        validate_training_row(row, smoke)
        if not row.get("id") or not row.get("target") or row.get("label") not in LABELS:
            raise ValueError("Task rows require id, target and audited bearish/neutral/bullish label")
        for match in duplicate_index.matches(row["text"]):
            union(index, match)
        duplicate_index.add(index, row["text"])
        for field in ("id", "group_id", "event_id"):
            if row.get(field):
                key = (field, str(row[field]))
                if key in identifiers:
                    union(index, identifiers[key])
                identifiers[key] = index
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    if len(groups) < 10:
        raise ValueError("Fewer than 10 independent groups after deduplication; cannot create credible held-out partitions")
    all_dated = all(row.get("created_at") for row in rows)
    if all_dated:
        def timestamp(row):
            date = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            if date.tzinfo is None:
                raise ValueError("created_at needs an explicit UTC offset")
            return date.timestamp()
        ordered = sorted(groups.values(), key=lambda group: max(timestamp(row) for row in group))
    else:
        ordered = sorted(groups.values(), key=lambda group: min(text_hash(row["text"]) for row in group))
    partitions = {name: [] for name in ("train", "development", "calibration", "test")}
    n = len(ordered)
    boundaries = (int(n * .7), int(n * .8), int(n * .9))
    for index, group in enumerate(ordered):
        split = ("train", "development", "calibration", "test")[bisect.bisect_right(boundaries, index)]
        gid = "group-" + min(text_hash(row["text"]) for row in group)[:20]
        for row in group:
            row = dict(row, split=split, group_id=gid)
            partitions[split].append(row)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, subset in partitions.items():
        write_jsonl(output / f"{split}.jsonl", subset)
    cal_groups = sorted({row["group_id"] for row in partitions["calibration"]})
    temp_groups = set(cal_groups[::2])
    write_jsonl(output / "calibration_temperature.jsonl", (row for row in partitions["calibration"] if row["group_id"] in temp_groups))
    write_jsonl(output / "calibration_threshold.jsonl", (row for row in partitions["calibration"] if row["group_id"] not in temp_groups))
    heldout = partitions["development"] + partitions["calibration"] + partitions["test"]
    write_jsonl(output / "frozen_eval.jsonl", heldout)
    summary = {"mode": "smoke" if smoke else "formal", "source_sha256": file_sha256(input_path),
               "split_method": "grouped_time" if all_dated else "grouped_non_temporal",
               "counts": {key: len(value) for key, value in partitions.items()},
               "groups": n, "near_duplicate_method": "normalized-exact + SimHash64 Hamming<=3",
               "frozen_eval_sha256": file_sha256(output / "frozen_eval.jsonl"),
               "warning": "Check class counts and manually audit event/semantic paraphrases; proportions are group-based"}
    write_json(output / "split_manifest.json", summary)
    return summary


def pack_corpus(input_path, tokenizer_path, output_dir, shard_tokens=5_000_000, exclusions=None, smoke=False, partition="train"):
    if shard_tokens < 2:
        raise ValueError("shard_tokens must be >=2")
    if not smoke and not exclusions:
        raise ValueError("Formal packing requires frozen_eval.jsonl via --exclude before tokenizer/pretraining")
    tok = load_tokenizer(tokenizer_path)
    if tok.get_vocab_size() > 65536:
        raise ValueError("uint16 requires <=65536 tokens")
    eos = tok.token_to_id(EOS_TOKEN)
    if eos is None:
        raise ValueError("Tokenizer lacks EOS")
    excluded = DuplicateIndex()
    if exclusions:
        for i, row in enumerate(read_jsonl(exclusions)):
            excluded.add(i, row["text"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Packed output already exists; select a new directory to preserve data version")
    duplicate_db = sqlite3.connect(output / "dedup.sqlite")
    duplicate_db.execute("CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY)")
    if duplicate_db.execute("SELECT COUNT(*) FROM seen").fetchone()[0]:
        duplicate_db.close()
        raise FileExistsError("An interrupted prior packing index exists; choose a new output directory")
    shards, buffer, total, kept, removed = [], [], 0, 0, 0

    def flush():
        nonlocal total
        if not buffer:
            return
        path = output / f"tokens-{len(shards):05d}.bin"
        if path.exists():
            raise FileExistsError(f"Unfinished prior shard exists: {path}; choose a new output directory")
        np.asarray(buffer, dtype="<u2").tofile(path)
        shards.append({"path": path.name, "count": len(buffer), "sha256": file_sha256(path)})
        total += len(buffer)
        buffer.clear()

    try:
        for row in read_jsonl(input_path):
            validate_training_row(row, smoke)
            if row.get("split", partition) != partition:
                raise ValueError(f"Corpus input must contain only split={partition} rows")
            if excluded.matches(row["text"]):
                removed += 1
                continue
            if duplicate_db.execute("INSERT OR IGNORE INTO seen(hash) VALUES(?)", (text_hash(row["text"]),)).rowcount == 0:
                removed += 1
                continue
            ids = tok.encode(row["text"]).ids + [eos]
            kept += 1
            for offset in range(0, len(ids), shard_tokens):
                chunk = ids[offset:offset + shard_tokens]
                while chunk:
                    take = min(shard_tokens - len(buffer), len(chunk))
                    buffer.extend(chunk[:take])
                    chunk = chunk[take:]
                    if len(buffer) == shard_tokens:
                        flush()
        flush()
        duplicate_db.commit()
    finally:
        duplicate_db.close()
    if total < 2:
        raise ValueError("No usable next-token examples after filtering")
    manifest = {"format": "finllm-uint16-v1", "mode": "smoke" if smoke else "formal", "partition": partition, "dtype": "<u2",
                "tokenizer_sha256": file_sha256(tokenizer_file(tokenizer_path)), "vocab_size": tok.get_vocab_size(),
                "token_count": total, "shards": shards, "source_sha256": file_sha256(input_path),
                "eval_exclusion_sha256": file_sha256(exclusions) if exclusions else None,
                "documents_kept": kept, "documents_excluded": removed,
                "attention": "causal across documents with EOS separators", "eos_id": eos}
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"


class TokenBlockDataset(Dataset):
    def __init__(self, manifest_path, seq_len):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "finllm-uint16-v1" or seq_len < 1:
            raise ValueError("Invalid token manifest or sequence length")
        self.seq_len = seq_len
        self.shards, self.ends, total = [], [], 0
        for item in self.manifest["shards"]:
            path = (self.manifest_path.parent / item["path"]).resolve()
            if not path.is_relative_to(self.manifest_path.parent.resolve()):
                raise ValueError("Shard path escapes manifest directory")
            if path.stat().st_size != item["count"] * 2 or file_sha256(path) != item["sha256"]:
                raise ValueError(f"Corrupt or changed shard: {path}")
            array = np.memmap(path, dtype="<u2", mode="r")
            if array.size and int(array.max()) >= self.manifest["vocab_size"]:
                raise ValueError("Token ID exceeds manifest vocabulary")
            self.shards.append(array)
            total += len(array)
            self.ends.append(total)
        if total != self.manifest["token_count"] or total < 2:
            raise ValueError("Token counts do not match")
        self.total = total

    def __len__(self):
        return math.ceil((self.total - 1) / self.seq_len)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.seq_len
        end = min(start + self.seq_len + 1, self.total)
        pieces = []
        while start < end:
            shard = bisect.bisect_right(self.ends, start)
            base = self.ends[shard - 1] if shard else 0
            stop = min(end, self.ends[shard])
            pieces.append(np.asarray(self.shards[shard][start - base:stop - base], dtype=np.int64))
            start = stop
        ids = np.concatenate(pieces)
        x = torch.zeros(self.seq_len, dtype=torch.long)
        y = torch.full((self.seq_len,), -100, dtype=torch.long)
        x[:len(ids) - 1] = torch.from_numpy(ids[:-1].copy())
        y[:len(ids) - 1] = torch.from_numpy(ids[1:].copy())
        return {"input_ids": x, "labels": y}


class TaskDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, preference=False):
        labels = label_token_ids(tokenizer)
        self.rows = []
        for row in read_jsonl(path):
            ids = encode_prompt(tokenizer, row["text"], row["target"], max_length)
            item = {"input_ids": torch.tensor(ids, dtype=torch.long)}
            if preference:
                chosen, rejected = row["chosen"], row["rejected"]
                if chosen not in LABELS or rejected not in LABELS or chosen == rejected:
                    raise ValueError("DPO needs two different valid labels")
                item.update(chosen_token_id=labels[chosen], rejected_token_id=labels[rejected])
            else:
                if row["label"] not in LABELS:
                    raise ValueError("Invalid task label")
                item["label_token_id"] = labels[row["label"]]
            self.rows.append(item)
        if not self.rows:
            raise ValueError(f"Empty task dataset: {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_task(batch, pad_id):
    lengths = torch.tensor([len(row["input_ids"]) for row in batch])
    ids = torch.full((len(batch), int(lengths.max())), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for index, row in enumerate(batch):
        ids[index, :lengths[index]] = row["input_ids"]
        mask[index, :lengths[index]] = True
    result = {"input_ids": ids, "attention_mask": mask, "last_positions": lengths - 1}
    for key in ("label_token_id", "chosen_token_id", "rejected_token_id"):
        if key in batch[0]:
            result[key + "s"] = torch.tensor([row[key] for row in batch], dtype=torch.long)
    return result
