"""Prepare additional corpora without changing a trained seed's tokenizer.

Documents, duplicate fingerprints, and split ordering live in SQLite. Token
packing buffers at most one document and one explicitly bounded shard. Existing
artifacts are only read and validated; failed versioned builds are preserved.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .corpus import mix_corpora
from .data import TokenBlockDataset, file_sha256, simhash, text_hash, validate_training_row, write_json
from .preparation import MAX_DOCUMENT_BYTES, MAX_JSONL_LINE_BYTES, _digest, _group, _line, _rows
from .tokenizer import EOS_TOKEN, load_tokenizer


PREPARED_FILES = {
    "general_train.jsonl", "general_validation.jsonl", "finance_train.jsonl",
    "finance_validation.jsonl", "frozen_prior_exclusions.jsonl", "frozen_exclusions.jsonl",
}
MAX_SHARD_TOKENS = 1_000_000
MAX_PARTIAL_RECOVERIES = 3
PACK_NAMES = {"general", "general_validation", "finance", "finance_validation", "domain_mix"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _verified_file(path, info):
    path = Path(path)
    _check(path.is_file() and path.stat().st_size == info["bytes"]
           and file_sha256(path) == info["sha256"], f"Changed or missing source file: {path}")


def validate_expansion_sources(corpora):
    """Read a completed fixed source snapshot, never a partial fetch."""
    corpora = Path(corpora)
    source = _read(corpora / "manifest.json")
    _check(source.get("status") == "complete", "Expansion corpus download is incomplete")
    for domain in ("general", "finance"):
        _verified_file(corpora / f"{domain}_candidate.jsonl", source[domain])
    _check(source["general"]["sha256"] != source["finance"]["sha256"], "General and finance source inputs must differ")
    return source


def _seed_inputs(seed_prepared, seed_processed, tokenizer, task_pool):
    """Verify the old prepared/packed boundary and unchanged 8K vocabulary."""
    seed_prepared, seed_processed, tokenizer = map(Path, (seed_prepared, seed_processed, tokenizer))
    _check(not (seed_prepared / "INCOMPLETE").exists(), "Seed preparation was interrupted")
    preparation = _read(seed_prepared / "preparation_manifest.json")
    summary = _read(seed_processed / "seed_summary.json")
    _check(summary.get("status") == "seed_inputs_prepared_training_not_started", "Invalid seed completion status")
    _check(preparation.get("mode") == "formal", "Expansion requires a formal seed")
    _check(file_sha256(seed_prepared / "preparation_manifest.json") == summary["preparation_manifest_sha256"],
           "Seed preparation manifest changed")
    for name, info in preparation["files"].items():
        _check(Path(name).name == name, "Seed prepared path escapes its directory")
        _verified_file(seed_prepared / name, info)
    _check(PREPARED_FILES.difference({"frozen_prior_exclusions.jsonl"}).issubset(preparation["files"]),
           "Seed preparation lacks required train/validation files")
    task_sha = file_sha256(task_pool)
    _check(task_sha == file_sha256(seed_prepared / "frozen_task_exclusions.jsonl"),
           "Task exclusions must equal the original frozen seed task pool")
    tok_sha = file_sha256(tokenizer / "tokenizer.json")
    _check(tok_sha == summary["tokenizer_sha256"], "Expansion must reuse the unchanged seed tokenizer bytes")
    vocabulary = load_tokenizer(tokenizer).get_vocab_size()
    _check(tok_sha == summary["tokenizer_sha256"] and vocabulary == summary["tokenizer_vocabulary"] == 8192,
           "Expansion must reuse the unchanged seed 8K tokenizer")
    metadata, provenance = _read(tokenizer / "tokenizer_meta.json"), _read(tokenizer / "provenance.json")
    _check(metadata["sha256"] == tok_sha and metadata["actual_vocab_size"] == vocabulary
           and provenance["tokenizer_sha256"] == tok_sha, "Seed tokenizer metadata or provenance changed")
    counts, manifests = {}, {}
    for domain in ("general", "finance"):
        for partition in ("train", "validation"):
            name = domain if partition == "train" else domain + "_validation"
            path = seed_processed / name / "manifest.json"
            data = TokenBlockDataset(path, 1)
            manifest = data.manifest
            excluded = "frozen_exclusions.jsonl" if partition == "train" else "frozen_task_exclusions.jsonl"
            _check(manifest["mode"] == "formal" and manifest["partition"] == partition
                   and manifest["tokenizer_sha256"] == tok_sha and manifest["vocab_size"] == vocabulary
                   and manifest["source_sha256"] == file_sha256(seed_prepared / f"{domain}_{partition}.jsonl")
                   and manifest["eval_exclusion_sha256"] == file_sha256(seed_prepared / excluded),
                   f"Seed packed provenance changed: {name}")
            counts[name] = manifest["token_count"]
            manifests[name] = {"manifest": path.as_posix(), "sha256": file_sha256(path)}
            del data
    _check(counts == summary["token_counts"], "Seed packed counts changed")
    return {"seed_preparation_manifest_sha256": file_sha256(seed_prepared / "preparation_manifest.json"),
            "seed_summary_sha256": file_sha256(seed_processed / "seed_summary.json"),
            "tokenizer_sha256": tok_sha, "tokenizer_vocabulary": vocabulary,
            "task_exclusion_sha256": task_sha,
            "seed_prepared_sha256": {name: file_sha256(seed_prepared / name) for name in
                                    ("general_train.jsonl", "general_validation.jsonl", "finance_train.jsonl", "finance_validation.jsonl")},
            "retention_validation": {domain: manifests[domain + "_validation"] for domain in ("general", "finance")}}


class _DiskExclusions:
    """SQLite exact plus SimHash64 Hamming<=3 index with a 16MiB page cache."""
    def __init__(self, connection):
        self.connection = connection
        connection.executescript("""
            CREATE TABLE exclusions (hash TEXT PRIMARY KEY, fingerprint TEXT NOT NULL);
            CREATE TABLE exclusion_bands (band INTEGER, value INTEGER, hash TEXT,
                                          PRIMARY KEY(band,value,hash));
            CREATE TABLE protected_groups (name TEXT PRIMARY KEY);
        """)

    def add(self, text, group=None):
        digest, fingerprint = text_hash(text), simhash(text)
        self.connection.execute("INSERT OR IGNORE INTO exclusions VALUES (?,?)", (digest, f"{fingerprint:016x}"))
        self.connection.executemany("INSERT OR IGNORE INTO exclusion_bands VALUES (?,?,?)",
                                    ((band, (fingerprint >> (16 * band)) & 65535, digest) for band in range(4)))
        if group is not None:
            self.connection.execute("INSERT OR IGNORE INTO protected_groups VALUES (?)", (group,))

    def protects_group(self, group):
        return self.connection.execute("SELECT 1 FROM protected_groups WHERE name=?", (group,)).fetchone() is not None

    def matches(self, text):
        digest = text_hash(text)
        if self.connection.execute("SELECT 1 FROM exclusions WHERE hash=?", (digest,)).fetchone():
            return True
        fingerprint = simhash(text)
        # A <=3-bit difference shares at least one of the four 16-bit bands.
        # SQL DISTINCT avoids materializing a Python candidate set.
        values = [(fingerprint >> (16 * band)) & 65535 for band in range(4)]
        for (candidate,) in self.connection.execute("""
                SELECT DISTINCT e.fingerprint FROM exclusions e JOIN exclusion_bands b ON e.hash=b.hash
                WHERE (b.band=0 AND b.value=?) OR (b.band=1 AND b.value=?)
                   OR (b.band=2 AND b.value=?) OR (b.band=3 AND b.value=?)""", values):
            if (fingerprint ^ int(candidate, 16)).bit_count() <= 3:
                return True
        return False


def _prepare(corpora, task_pool, seed_prepared, output, validation_fraction, seed):
    output.mkdir(parents=True)
    (output / "INCOMPLETE").write_text("Expansion preparation has not completed.\n", encoding="utf-8")
    inputs = {domain: Path(corpora) / f"{domain}_candidate.jsonl" for domain in ("general", "finance")}
    input_sha = {name: file_sha256(path) for name, path in inputs.items()}
    statistics = {domain: {"input": 0, "prior_group_excluded": 0, "prior_text_excluded": 0,
                          "exact_duplicates": 0, "validation_overlap_removed": 0, "train": 0, "validation": 0}
                  for domain in inputs}
    database = output / "preparation.sqlite"
    frozen_counts = {"seed_documents": 0, "task_documents": 0, "new_validation_documents": 0}
    with ExitStack() as stack:
        connection = sqlite3.connect(database)
        stack.callback(connection.close)
        connection.execute("PRAGMA cache_size=-16384")
        connection.execute("PRAGMA temp_store=FILE")
        exclusions = _DiskExclusions(connection)
        prior_stream = stack.enter_context((output / "frozen_prior_exclusions.jsonl").open("wb"))
        all_stream = stack.enter_context((output / "frozen_exclusions.jsonl").open("wb"))
        for domain in inputs:
            for partition in ("train", "validation"):
                for row in _rows(Path(seed_prepared) / f"{domain}_{partition}.jsonl"):
                    exclusions.add(row["text"], _group(row))
                    encoded = _line(row)
                    prior_stream.write(encoded)
                    all_stream.write(encoded)
                    frozen_counts["seed_documents"] += 1
        for row in _rows(task_pool):
            exclusions.add(row["text"], _group(row) if row.get("source_url") or row.get("document_id") else None)
            encoded = _line(row)
            prior_stream.write(encoded)
            all_stream.write(encoded)
            frozen_counts["task_documents"] += 1
        _check(frozen_counts["task_documents"] > 0, "Frozen task pool must be nonempty")
        connection.executescript("""
            CREATE TABLE documents (hash TEXT PRIMARY KEY, domain TEXT, split TEXT, ordering TEXT, record TEXT);
        """)
        threshold = int(validation_fraction * (1 << 256))
        for domain in inputs:
            stat = statistics[domain]
            for row in _rows(inputs[domain]):
                stat["input"] += 1
                validate_training_row(row, smoke=False)
                group = _group(row)
                if exclusions.protects_group(group):
                    stat["prior_group_excluded"] += 1
                    continue
                if exclusions.matches(row["text"]):
                    stat["prior_text_excluded"] += 1
                    continue
                digest = text_hash(row["text"])
                split = "validation" if int(_digest(seed, group), 16) < threshold else "train"
                prepared = dict(row, split=split, document_hash=digest, preparation_group=_digest(0, group))
                cursor = connection.execute("INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?)",
                                            (digest, domain, split, _digest(seed, "order:" + digest),
                                             json.dumps(prepared, ensure_ascii=False)))
                stat["exact_duplicates"] += int(not cursor.rowcount)
                if stat["input"] % 1000 == 0:
                    connection.commit()
                    print(json.dumps({"phase": "expansion_prepare", "domain": domain, **stat}), flush=True)
            _check(stat["input"] > 0, f"{domain} candidate input is empty")
            print(json.dumps({"phase": "expansion_candidates_indexed", "domain": domain, **stat}), flush=True)
        connection.commit()
        for domain in inputs:
            with (output / f"{domain}_validation.jsonl").open("wb") as stream:
                for (record,) in connection.execute("SELECT record FROM documents WHERE domain=? AND split='validation' ORDER BY ordering", (domain,)):
                    row = json.loads(record)
                    exclusions.add(row["text"], _group(row))
                    encoded = _line(row)
                    stream.write(encoded)
                    all_stream.write(encoded)
                    statistics[domain]["validation"] += 1
                    frozen_counts["new_validation_documents"] += 1
        for domain in inputs:
            with (output / f"{domain}_train.jsonl").open("wb") as stream:
                for (record,) in connection.execute("SELECT record FROM documents WHERE domain=? AND split='train' ORDER BY ordering", (domain,)):
                    row = json.loads(record)
                    if exclusions.protects_group(_group(row)) or exclusions.matches(row["text"]):
                        statistics[domain]["validation_overlap_removed"] += 1
                        continue
                    stream.write(_line(row))
                    statistics[domain]["train"] += 1
            _check(statistics[domain]["train"] > 0 and statistics[domain]["validation"] > 0,
                   f"{domain} needs independent nonempty train and validation after prior seed/task exclusions")
    database.unlink()
    _check(input_sha == {name: file_sha256(path) for name, path in inputs.items()}, "Source inputs changed during preparation")
    counts = {f"{domain}_{partition}": statistics[domain][partition]
              for domain in inputs for partition in ("train", "validation")}
    counts["frozen_prior_exclusions"] = frozen_counts["seed_documents"] + frozen_counts["task_documents"]
    counts["frozen_exclusions"] = sum(frozen_counts.values())
    manifest = {"format_version": 1, "mode": "formal", "seed": seed, "validation_fraction": validation_fraction,
                "input_sha256": input_sha, "counts": counts, "source_statistics": statistics,
                "prior_exclusions": frozen_counts,
                "split_method": "seeded document/source_url hash; global normalized-exact SQLite deduplication",
                "duplicate_method": "prior seed/task document identity + normalized-exact + SimHash64 Hamming<=3; new validation protected before training",
                "bounds": {"max_document_bytes": MAX_DOCUMENT_BYTES, "max_jsonl_line_bytes": MAX_JSONL_LINE_BYTES,
                           "sqlite_cache_bytes": 16 * 1024 * 1024, "sqlite_temp_store": "FILE"},
                "files": {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path),
                                     "documents": counts[path.stem]} for path in sorted(output.glob("*.jsonl"))},
                "limitations": ["Lexical checks do not detect every semantic paraphrase.",
                                "Source rights metadata is required, not independently adjudicated by this program.",
                                "Holdouts use non-temporal document hashes, not a chronological financial test."]}
    _atomic_json(output / "preparation_manifest.json", manifest)
    (output / "INCOMPLETE").unlink()
    return manifest


def _pack(input_path, tokenizer, output, exclusions, partition, shard_tokens):
    """Pack already verified preparation with one-document/one-shard bounds."""
    output.mkdir(parents=True)
    (output / "INCOMPLETE").write_text("Expansion packing has not completed.\n", encoding="utf-8")
    source_sha, exclusion_sha = file_sha256(input_path), file_sha256(exclusions)
    tok = load_tokenizer(tokenizer)
    eos = tok.token_to_id(EOS_TOKEN)
    buffer, shards, documents, tokens = [], [], 0, 0

    def flush():
        nonlocal tokens
        path = output / f"tokens-{len(shards):05d}.bin"
        np.asarray(buffer, dtype="<u2").tofile(path)
        shards.append({"path": path.name, "count": len(buffer), "sha256": file_sha256(path)})
        tokens += len(buffer)
        buffer.clear()
        print(json.dumps({"phase": "expansion_pack", "output": output.as_posix(), "tokens": tokens,
                          "shards": len(shards), "documents": documents}), flush=True)

    for row in _rows(input_path):
        validate_training_row(row, smoke=False)
        _check(row.get("split") == partition, "Prepared partition changed before packing")
        ids = tok.encode(row["text"]).ids + [eos]
        documents += 1
        for offset in range(0, len(ids), shard_tokens):
            chunk = ids[offset:offset + shard_tokens]
            while chunk:
                take = min(shard_tokens - len(buffer), len(chunk))
                buffer.extend(chunk[:take])
                chunk = chunk[take:]
                if len(buffer) == shard_tokens:
                    flush()
    if buffer:
        flush()
    _check(tokens >= 2, "No additional usable next-token data after exclusions")
    _check(source_sha == file_sha256(input_path) and exclusion_sha == file_sha256(exclusions), "Prepared data changed during packing")
    result = {"format": "finllm-uint16-v1", "mode": "formal", "partition": partition, "dtype": "<u2",
              "tokenizer_sha256": file_sha256(Path(tokenizer) / "tokenizer.json"), "vocab_size": tok.get_vocab_size(),
              "token_count": tokens, "shards": shards, "source_sha256": source_sha,
              "eval_exclusion_sha256": exclusion_sha, "documents_kept": documents, "documents_excluded": 0,
              "attention": "causal across documents with EOS separators", "eos_id": eos,
              "packing_filter": "SQLite-verified expansion preparation; excludes prior seed/task and protects new validation"}
    _atomic_json(output / "manifest.json", result)
    (output / "INCOMPLETE").unlink()
    return result


def _settings(validation_fraction, shard_tokens, seed):
    _check(isinstance(validation_fraction, (int, float)) and not isinstance(validation_fraction, bool)
           and math.isfinite(validation_fraction) and 0 < validation_fraction < 1, "Invalid validation fraction")
    _check(isinstance(shard_tokens, int) and not isinstance(shard_tokens, bool) and 2 <= shard_tokens <= MAX_SHARD_TOKENS,
           f"shard_tokens must be between 2 and {MAX_SHARD_TOKENS}")
    _check(isinstance(seed, int) and not isinstance(seed, bool), "seed must be an integer")
    return {"validation_fraction": validation_fraction, "shard_tokens": shard_tokens, "seed": seed}


def _seed_identity(boundary):
    """Content identity survives a verified copy to a new project directory."""
    identity = dict(boundary)
    identity["retention_validation"] = {domain: {"sha256": info["sha256"]}
                                         for domain, info in boundary["retention_validation"].items()}
    return identity


def _contract_identity(contract):
    return {"settings": contract["settings"], "source_manifest_sha256": contract["source_manifest_sha256"],
            "seed_provenance": _seed_identity(contract["seed_provenance"])}


def _summary_identity(summary):
    identity = {name: value for name, value in summary.items() if name not in {"prepared", "processed"}}
    identity["seed_provenance"] = _seed_identity(summary["seed_provenance"])
    identity["retention_validation"] = {domain: {"sha256": info["sha256"]}
                                         for domain, info in summary["retention_validation"].items()}
    return identity


def _validate_prepared(prepared, source, settings):
    prep = _read(prepared / "preparation_manifest.json")
    _check(prep["mode"] == "formal" and prep["seed"] == settings["seed"]
           and prep["validation_fraction"] == settings["validation_fraction"]
           and set(prep["files"]) == PREPARED_FILES
           and prep["input_sha256"] == {domain: source[domain]["sha256"] for domain in ("general", "finance")},
           "Expansion preparation settings, inputs or file list changed")
    for name, info in prep["files"].items():
        _verified_file(prepared / name, info)
    return prep


def _validate_pack(prepared, processed, name, partition, tokenizer_sha):
    data = TokenBlockDataset(processed / name / "manifest.json", 1)
    manifest = data.manifest
    domain = name.removesuffix("_validation")
    excluded = "frozen_exclusions.jsonl" if partition == "train" else "frozen_prior_exclusions.jsonl"
    _check(manifest["mode"] == "formal" and manifest["partition"] == partition
           and manifest["tokenizer_sha256"] == tokenizer_sha and manifest["vocab_size"] == 8192
           and manifest["source_sha256"] == file_sha256(prepared / f"{domain}_{partition}.jsonl")
           and manifest["eval_exclusion_sha256"] == file_sha256(prepared / excluded),
           f"Expansion packed provenance changed: {name}")
    return manifest


def _validate_mix(processed, counts, boundary, seed):
    mix = TokenBlockDataset(processed / "domain_mix/manifest.json", 1)
    manifest = mix.manifest
    unit = min(counts["general"], counts["finance"] // 4)
    _check(manifest["mode"] == "formal" and manifest["partition"] == "train"
           and manifest["tokenizer_sha256"] == boundary["tokenizer_sha256"] and manifest["vocab_size"] == 8192
           and manifest["mixture_token_counts"] == {"general": unit, "finance": unit * 4}
           and manifest["source_unique_token_counts"] == {name: counts[name] for name in ("general", "finance")}
           and manifest["seed"] == seed, "Domain mixture budget or tokenizer changed")
    _check(manifest["source_manifests"] == [file_sha256(processed / name / "manifest.json") for name in ("general", "finance")],
           "Domain mixture source provenance changed")
    return manifest


def _journal_write(processed, journal):
    _atomic_json(processed / "build_journal.json", journal)


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def _validate_pending_outputs(prepared, processed, journal, source, settings, boundary):
    """Check committed and finished artifacts before changing any journal."""
    _check(set(journal["completed"]).issubset(PACK_NAMES | {"prepare"}), "Unknown completed journal phase")
    prep_exists = (prepared / "preparation_manifest.json").exists()
    if "prepare" in journal["completed"]:
        _check(prep_exists and file_sha256(prepared / "preparation_manifest.json") == journal["completed"]["prepare"],
               "Previously completed preparation is missing or changed; refused rebuilding")
    if prep_exists:
        _validate_prepared(prepared, source, settings)
    elif prepared.exists():
        _check(journal["phase"] == "prepare" and "prepare" not in journal["completed"], "Unowned partial preparation")
        _check_partial_files(prepared, "prepare")
    counts = {}
    for domain in ("general", "finance"):
        for partition in ("train", "validation"):
            name = domain if partition == "train" else domain + "_validation"
            manifest = processed / name / "manifest.json"
            if name in journal["completed"]:
                _check(manifest.is_file() and file_sha256(manifest) == journal["completed"][name],
                       f"Previously completed pack is missing or changed: {name}; refused rebuilding")
            if manifest.exists():
                _check(prep_exists, "Packed artifacts exist without verified prepared documents")
                counts[name] = _validate_pack(prepared, processed, name, partition, boundary["tokenizer_sha256"])["token_count"]
            elif manifest.parent.exists():
                _check(journal["phase"] == name and name not in journal["completed"], f"Unowned partial pack: {name}")
                _check_partial_files(manifest.parent, name)
    manifest = processed / "domain_mix/manifest.json"
    if "domain_mix" in journal["completed"]:
        _check(manifest.is_file() and file_sha256(manifest) == journal["completed"]["domain_mix"],
               "Previously completed mixture is missing or changed; refused rebuilding")
    if manifest.exists():
        _check(set(counts) == PACK_NAMES.difference({"domain_mix"}), "Mixture exists without all verified source packs")
        _validate_mix(processed, counts, boundary, settings["seed"])
    elif manifest.parent.exists():
        _check(journal["phase"] == "domain_mix" and "domain_mix" not in journal["completed"], "Unowned partial mixture")
        _check_partial_files(manifest.parent, "domain_mix")


def _check_partial_files(path, phase):
    _check(path.is_dir() and not path.is_symlink(), f"Unsafe partial output: {path}")
    allowed = PREPARED_FILES | {"INCOMPLETE", "preparation.sqlite", "preparation.sqlite-journal", "preparation_manifest.json.tmp"}
    for item in path.iterdir():
        known = item.name in allowed if phase == "prepare" else item.name in {"INCOMPLETE", "manifest.json.tmp"} or bool(re.fullmatch(r"tokens-\d{5}\.bin", item.name))
        _check(known and item.is_file() and not item.is_symlink(), f"Unknown file in partial {phase}; preserved without moving: {item}")


def _preserve_partial(path, processed, journal, phase):
    """Archive only this journal's unfinished phase, never complete artifacts."""
    _check(journal["phase"] == phase and phase not in journal["completed"],
           f"Cannot recover an unowned or previously completed phase: {phase}")
    _check_partial_files(path, phase)
    archive = processed / "recovery_archive"
    _check(not archive.is_symlink() and archive.resolve().is_relative_to(processed.resolve()), "Recovery archive escapes output directory")
    archive.mkdir(exist_ok=True)
    attempts = len(list(archive.iterdir()))
    _check(attempts < MAX_PARTIAL_RECOVERIES,
           f"Partial recovery limit {MAX_PARTIAL_RECOVERIES} reached; preserved archives need review before another build")
    target = archive / f"{attempts + 1:03d}_{phase}"
    _check(not target.exists(), "Recovery archive path already exists")
    path.rename(target)
    journal["recoveries"].append({"phase": phase, "preserved_at": target.as_posix()})
    _journal_write(processed, journal)
    print(f"Preserved interrupted {phase} files at {target}; rebuilding only that unfinished phase.", flush=True)


def validate_expansion_artifacts(corpora, task_pool, seed_prepared, seed_processed, tokenizer, prepared, processed,
                                 validation_fraction=.05, shard_tokens=MAX_SHARD_TOKENS, seed=130):
    """Read-only complete reuse; interrupted or altered artifacts are preserved."""
    prepared, processed = Path(prepared), Path(processed)
    try:
        settings = _settings(validation_fraction, shard_tokens, seed)
        source = validate_expansion_sources(corpora)
        boundary = _seed_inputs(seed_prepared, seed_processed, tokenizer, task_pool)
        summary = _read(processed / "expansion_summary.json")
        _check(summary.get("status") == "expansion_inputs_prepared_training_not_started", "Expansion completion status missing")
        _check(not (processed / "INCOMPLETE").exists() and not (prepared / "INCOMPLETE").exists(), "Expansion build was interrupted")
        _check(summary["settings"] == settings and _seed_identity(summary["seed_provenance"]) == _seed_identity(boundary),
               "Expansion settings or seed boundary changed")
        _check({domain: info["sha256"] for domain, info in summary["retention_validation"].items()}
               == {domain: info["sha256"] for domain, info in boundary["retention_validation"].items()},
               "Retention validation content identity changed")
        _check(summary["source_manifest_sha256"] == file_sha256(Path(corpora) / "manifest.json"), "Expansion source manifest changed")
        _check(summary["preparation_manifest_sha256"] == file_sha256(prepared / "preparation_manifest.json"), "Expansion preparation manifest changed")
        _validate_prepared(prepared, source, settings)
        counts = {}
        for domain in ("general", "finance"):
            for partition in ("train", "validation"):
                name = domain if partition == "train" else domain + "_validation"
                _check(not (processed / name / "INCOMPLETE").exists(), f"Incomplete pack: {name}")
                manifest = _validate_pack(prepared, processed, name, partition, boundary["tokenizer_sha256"])
                counts[name] = manifest["token_count"]
        manifest = _validate_mix(processed, counts, boundary, seed)
        counts["domain_mix"] = manifest["token_count"]
        _check(summary["token_counts"] == counts and summary["tokenizer_sha256"] == boundary["tokenizer_sha256"]
               and summary["tokenizer_vocabulary"] == 8192, "Expansion counts or tokenizer changed")
        return summary
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Existing expansion is incomplete or changed: {error}. Existing files were preserved. "
                         "Restore a verified backup or choose a NEW EXPANSION_VERSION.") from error


def build_expansion_artifacts(corpora, task_pool, seed_prepared, seed_processed, tokenizer, prepared, processed,
                             validation_fraction=.05, shard_tokens=MAX_SHARD_TOKENS, seed=130, validate_only=False):
    """Build new versioned packs; the original seed/tokenizer remain immutable."""
    prepared, processed = Path(prepared), Path(processed)
    settings = _settings(validation_fraction, shard_tokens, seed)
    arguments = dict(corpora=corpora, task_pool=task_pool, seed_prepared=seed_prepared, seed_processed=seed_processed,
                     tokenizer=tokenizer, prepared=prepared, processed=processed, **settings)
    if validate_only:
        return validate_expansion_artifacts(**arguments)
    existing = prepared.exists() or processed.exists()
    journal_path = processed / "build_journal.json"
    # The verified summary plus the absent INCOMPLETE marker certify readiness.
    # A last journal-write failure must not force changes to completed bytes.
    if existing and (not (processed / "INCOMPLETE").exists() or not journal_path.is_file()):
        return validate_expansion_artifacts(**arguments)
    source = validate_expansion_sources(corpora)
    boundary = _seed_inputs(seed_prepared, seed_processed, tokenizer, task_pool)
    print("Verified seed boundary; preparing additional corpora with the unchanged 8K tokenizer.", flush=True)
    protected = [Path(seed_prepared).resolve(), Path(seed_processed).resolve(), Path(tokenizer).resolve(), Path(corpora).resolve()]
    for target in (prepared.resolve(), processed.resolve()):
        _check(all(target != old and not target.is_relative_to(old) and not old.is_relative_to(target) for old in protected),
               "Expansion outputs must be separate from old seed, tokenizer and source directories")
    _check(prepared.resolve() != processed.resolve() and not prepared.resolve().is_relative_to(processed.resolve())
           and not processed.resolve().is_relative_to(prepared.resolve()), "Prepared and processed outputs must be separate")
    contract = {"settings": settings, "prepared": prepared.as_posix(), "processed": processed.as_posix(),
                "source_manifest_sha256": file_sha256(Path(corpora) / "manifest.json"), "seed_provenance": boundary}
    if existing:
        try:
            journal = _read(journal_path)
            _check(journal["format"] == "finllm-expansion-build-v1" and journal["status"] == "in_progress"
                   and _contract_identity(journal["contract"]) == _contract_identity(contract)
                   and isinstance(journal["completed"], dict)
                   and isinstance(journal["recoveries"], list), "Incomplete build contract changed or is not recoverable")
            allowed = PACK_NAMES | {"INCOMPLETE", "build_journal.json", "build_journal.json.tmp", "recovery_archive", "expansion_summary.json", "expansion_summary.json.tmp"}
            _check(all(item.name in allowed and not item.is_symlink() for item in processed.iterdir()),
                   "Unknown files in incomplete processed output; refused recovery")
            _validate_pending_outputs(prepared, processed, journal, source, settings, boundary)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Existing expansion is incomplete or changed: {error}. Existing files were preserved.") from error
    else:
        processed.mkdir(parents=True)
        (processed / "INCOMPLETE").write_text("Expansion artifact build has not completed.\n", encoding="utf-8")
        journal = {"format": "finllm-expansion-build-v1", "status": "in_progress", "contract": contract,
                   "phase": "prepare", "completed": {}, "recoveries": []}
        _journal_write(processed, journal)
    if (prepared / "preparation_manifest.json").exists():
        prep = _validate_prepared(prepared, source, settings)
        expected = journal["completed"].get("prepare")
        _check(expected is None or expected == file_sha256(prepared / "preparation_manifest.json"),
               "Completed preparation manifest changed; preserved without recovery")
        if (prepared / "INCOMPLETE").exists():
            (prepared / "INCOMPLETE").unlink()
    else:
        if prepared.exists():
            _preserve_partial(prepared, processed, journal, "prepare")
        _check("prepare" not in journal["completed"], "Completed preparation is missing; preserved without recovery")
        journal["phase"] = "prepare"
        _journal_write(processed, journal)
        prep = _prepare(corpora, task_pool, seed_prepared, prepared, validation_fraction, seed)
    journal["completed"]["prepare"] = file_sha256(prepared / "preparation_manifest.json")
    _journal_write(processed, journal)
    counts = {}
    for domain in ("general", "finance"):
        for partition in ("train", "validation"):
            name = domain if partition == "train" else domain + "_validation"
            excluded = "frozen_exclusions.jsonl" if partition == "train" else "frozen_prior_exclusions.jsonl"
            target = processed / name
            if (target / "manifest.json").exists():
                pack = _validate_pack(prepared, processed, name, partition, boundary["tokenizer_sha256"])
                expected = journal["completed"].get(name)
                _check(expected is None or expected == file_sha256(target / "manifest.json"),
                       f"Completed pack manifest changed: {name}; preserved without recovery")
                if (target / "INCOMPLETE").exists():
                    (target / "INCOMPLETE").unlink()
            else:
                _check(name not in journal["completed"], f"Previously completed pack is missing: {name}; refused rebuilding")
                if target.exists():
                    _preserve_partial(target, processed, journal, name)
                journal["phase"] = name
                _journal_write(processed, journal)
                pack = _pack(prepared / f"{domain}_{partition}.jsonl", tokenizer, target,
                             prepared / excluded, partition, shard_tokens)
            counts[name] = pack["token_count"]
            journal["completed"][name] = file_sha256(target / "manifest.json")
            _journal_write(processed, journal)
    unit = min(counts["general"], counts["finance"] // 4)
    _check(unit > 0, "Insufficient additional training tokens for an 80/20 finance/general mixture")
    target = processed / "domain_mix"
    if (target / "manifest.json").exists():
        _validate_mix(processed, counts, boundary, seed)
        expected = journal["completed"].get("domain_mix")
        _check(expected is None or expected == file_sha256(target / "manifest.json"), "Completed mixture manifest changed")
    else:
        _check("domain_mix" not in journal["completed"], "Previously completed domain mixture is missing")
        if target.exists():
            _preserve_partial(target, processed, journal, "domain_mix")
        journal["phase"] = "domain_mix"
        _journal_write(processed, journal)
        mix_corpora(processed / "general/manifest.json", processed / "finance/manifest.json",
                    target, general_tokens=unit, finance_tokens=unit * 4, seed=seed)
    journal["completed"]["domain_mix"] = file_sha256(target / "manifest.json")
    _journal_write(processed, journal)
    counts["domain_mix"] = unit * 5
    _check(boundary == _seed_inputs(seed_prepared, seed_processed, tokenizer, task_pool), "Seed/tokenizer boundary changed during expansion")
    _check(source == validate_expansion_sources(corpora), "Source snapshot changed during expansion")
    summary = {"status": "expansion_inputs_prepared_training_not_started", "format_version": 1,
               "settings": settings, "prepared": prepared.as_posix(), "processed": processed.as_posix(),
               "token_counts": counts, "tokenizer_sha256": boundary["tokenizer_sha256"], "tokenizer_vocabulary": 8192,
               "tokenizer_policy": "Reused exact seed_v1 tokenizer bytes; no tokenizer training performed.",
               "seed_provenance": boundary, "retention_validation": boundary["retention_validation"],
               "source_manifest_sha256": file_sha256(Path(corpora) / "manifest.json"),
               "preparation_manifest_sha256": file_sha256(prepared / "preparation_manifest.json"),
               "source_statistics": prep["source_statistics"], "prior_exclusions": prep["prior_exclusions"],
               "domain_mix": {"finance_tokens": unit * 4, "general_tokens": unit, "finance_fraction": .8,
                              "repeated_source_tokens": 0, "counts_are_not_additional_unique_data": True},
               "limits": ["Counts describe additional documents after seed/task lexical exclusion, not extra training already performed.",
                          "Domain mixture rearranges the additional source tokens and must not be counted again as unique data.",
                          "Financial minutes are not tweet coverage or audited target-asset stance labels.",
                          "No SFT, DPO or financial stance accuracy is established by this build."]}
    if (processed / "expansion_summary.json").exists():
        previous = _read(processed / "expansion_summary.json")
        _check(_summary_identity(previous) == _summary_identity(summary), "Existing complete expansion summary changed; preserved")
        summary = previous
    else:
        _atomic_json(processed / "expansion_summary.json", summary)
    (processed / "INCOMPLETE").unlink()
    journal.update(status="complete", phase="complete")
    _journal_write(processed, journal)
    return validate_expansion_artifacts(**arguments)
