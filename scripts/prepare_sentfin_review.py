"""Freeze a pinned SEntFiN candidate pool and export an unlabelled human review sheet.

Only the Python standard library is required. Source financial sentiment is kept
separate from this project's asset stance labels. No row is human-approved here.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path
import urllib.request


REVISION = "eba43610fe3cb57a3e5773cd6c037da1f8992ad5"
BASE_URL = f"https://raw.githubusercontent.com/pyRis/SEntFiN/{REVISION}"
FILES = {
    "SEntFiN.csv": (1118162, "570af7ba58f724f3fcc414dd5ba80da5b1a3b342134d76c6ebdf53a32c54a92f"),
    "entity_list_comprehensive.csv": (90424, "6ace045edc54c90bc0695ee5a78afec9f7a5294844ccfd0c9e85a14aeea4916b"),
    "LICENSE": (1110, "961d32986ac3fa2fe2e55ebc7bbae1ba8d632412e98493d73bdf59f0ad8fc65f"),
    "README.md": (48, "543e1779ccb4e13b23107b3c4443891f581041d56adf676daa8db173c730e03a"),
}
LABELS = ("negative", "neutral", "positive")
ATTRIBUTION = "Ankur Sinha, Satishwar Kedas, Rishu Kumar, Pekka Malo; SEntFiN 1.0; MIT repository license"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def obtain_sources(source_dir: Path, download: bool = False) -> None:
    """Read only the four pinned files; cap every download at its expected size."""
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, (size, expected_hash) in FILES.items():
        path = source_dir / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; pass --download to fetch the pinned source")
            request = urllib.request.Request(f"{BASE_URL}/{name}", headers={"User-Agent": "130U-FinLLM-research/0.1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read(size + 1)
            if len(content) != size or hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError(f"Pinned source integrity check failed: {name}")
            path.write_bytes(content)
        if path.stat().st_size != size or digest(path) != expected_hash:
            raise ValueError(f"Existing source integrity check failed: {path}; use a new source directory")


def parse_rows(source_file: Path):
    with source_file.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["S No.", "Title", "Decisions", "Words"]:
            raise ValueError("Unexpected SEntFiN schema")
        rows = list(reader)
    pool, quarantined, headlines = [], [], collections.defaultdict(list)
    seen_ids = set()
    for row in rows:
        source_id, title = row["S No."], row["Title"]
        if not source_id or source_id in seen_ids or not title.strip():
            raise ValueError("Source IDs must be unique; IDs and headlines must be nonempty")
        seen_ids.add(source_id)
        headlines[title].append(source_id)
        try:
            decisions = ast.literal_eval(row["Decisions"])
            if not isinstance(decisions, dict) or not decisions:
                raise ValueError("Expected a nonempty entity-sentiment dictionary")
            if any(not isinstance(k, str) or not k.strip() or v not in LABELS for k, v in decisions.items()):
                raise ValueError("Invalid entity or source sentiment")
        except (ValueError, SyntaxError) as error:
            quarantined.append({"source_id": source_id, "reason": type(error).__name__,
                                "source_decisions": row["Decisions"]})
            continue
        for target, label in decisions.items():
            pool.append({"source_id": source_id, "text": title, "source_entity": target,
                         "source_sentiment": label, "source_entity_count": len(decisions),
                         "entity_is_literal_substring": target.casefold() in title.casefold(),
                         "source_entities": list(decisions)})
    return rows, pool, quarantined, headlines


def select_review(pool: list[dict], count: int, seed: int = 130) -> list[dict]:
    if count < 3:
        raise ValueError("Review size must be at least three")
    selected, selected_texts = [], set()

    def order(row):
        return hashlib.sha256(f"{seed}|{row['source_id']}|{row['source_entity']}".encode()).hexdigest()

    for index, label in enumerate(LABELS):
        quota = count // 3 + int(index < count % 3)
        candidates = sorted((r for r in pool if r["source_sentiment"] == label), key=order)
        # Reserve roughly a third of each source-label stratum for multi-entity
        # headlines. Those require explicit target review; no ticker is guessed.
        multi_quota = quota // 3
        for require_multi, cap in ((True, multi_quota), (False, quota)):
            while sum(r["source_sentiment"] == label for r in selected) < cap:
                found = next((r for r in candidates if r["text"] not in selected_texts
                              and (not require_multi or r["source_entity_count"] > 1)), None)
                if found is None:
                    if require_multi:
                        break
                    raise ValueError(f"Insufficient distinct headlines for {label}: requested {quota}")
                selected.append(found)
                selected_texts.add(found["text"])
    return sorted(selected, key=order)


def reuse_complete_output(source_dir: Path, output: Path, count: int, seed: int) -> dict:
    """Reuse a frozen pool; human review files may have been edited since export.

    Legacy schema 1 writes its manifest last, without a status field. Its frozen
    exclusion hash and pinned source metadata remain the completion contract.
    Raw source files may be absent after restoring a Drive snapshot, so their
    recorded provenance is checked and any locally present copy is also checked.
    """
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("status", "complete") != "complete":
            raise ValueError("review export manifest is not complete")
        if manifest["source_repository"] != "https://github.com/pyRis/SEntFiN" or manifest["source_revision"] != REVISION:
            raise ValueError("source repository or pinned revision differs")
        if manifest["review_seed"] != seed or manifest["review_rows"] != count:
            raise ValueError("review count or seed differs from the completed export")
        expected_sources = {name: {"url": f"{BASE_URL}/{name}", "bytes": size, "sha256": sha}
                            for name, (size, sha) in FILES.items()}
        if manifest["source_files"] != expected_sources:
            raise ValueError("pinned source provenance does not match")
        for name, (size, expected_hash) in FILES.items():
            path = source_dir / name
            if path.exists() and (path.stat().st_size != size or digest(path) != expected_hash):
                raise ValueError(f"source integrity check failed: {name}")
        for name in ("task_exclusions.jsonl", "review_200.csv", "SOURCE_LICENSE.txt", "review_instructions.md"):
            if name not in manifest["artifacts"] or not (output / name).is_file():
                raise ValueError(f"review export is incomplete: {name}")
        frozen = output / "task_exclusions.jsonl"
        record = manifest["artifacts"][frozen.name]
        if frozen.stat().st_size != record["bytes"] or digest(frozen) != record["sha256"]:
            raise ValueError("frozen task exclusion integrity check failed")
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Cannot reuse existing review output: {error}. Keep the old directory and use a new seed version; "
            "existing files are never overwritten."
        ) from error


def prepare(source_dir: Path, output: Path, count: int = 200, seed: int = 130, download: bool = False) -> dict:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        return reuse_complete_output(source_dir, output, count, seed)
    obtain_sources(source_dir, download)
    rows, pool, quarantined, headlines = parse_rows(source_dir / "SEntFiN.csv")
    selected = select_review(pool, count, seed)
    output.mkdir(parents=True, exist_ok=True)

    with (output / "task_exclusions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for text in sorted(headlines):
            ids = sorted(headlines[text], key=int)
            stream.write(json.dumps({"id": "sentfin:" + ids[0], "text": text, "source_ids": ids,
                                     "source": f"{BASE_URL}/SEntFiN.csv",
                                     "purpose": "candidate_task_exclusion_only"}, ensure_ascii=False) + "\n")

    fields = ["id", "text", "source_entity", "source_sentiment", "source_entity_count", "source_entities",
              "entity_is_literal_substring", "proposed_stance", "asset_identifier", "human_stance",
              "human_reviewed", "notes"]
    with (output / "review_200.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({"id": f"sentfin:{row['source_id']}:{row['source_entity']}",
                             **{k: row[k] for k in fields if k in row and k != "source_entities"},
                             "source_entities": json.dumps(row["source_entities"], ensure_ascii=False),
                             "proposed_stance": "", "asset_identifier": "", "human_stance": "",
                             "human_reviewed": "false", "notes": ""})

    (output / "SOURCE_LICENSE.txt").write_bytes((source_dir / "LICENSE").read_bytes())
    (output / "review_instructions.md").write_text(
        "# 130U-FinLLM human review worksheet\n\n"
        "This candidate worksheet starts without project-confirmed asset-stance labels. source_sentiment is the original financial entity sentiment label; "
        "it cannot simply be renamed bullish/bearish. proposed_stance is intentionally blank.\n\n"
        "For each row, confirm whether source_entity is an investable asset, company, or applicable asset class. Enter an explicit asset_identifier; "
        "do not guess symbols. Set human_stance to bullish, bearish, neutral, or exclude. "
        "Bullish/bearish must follow the investment direction or outlook expressed for the target; a past price change alone defaults to neutral. "
        "Use exclude with an explanation for regulators, unclear targets, sarcasm, or unclear attribution. Set human_reviewed=true only after review.\n\n"
        "Review each entity separately in multi-entity news. This worksheet is stratified by source labels and includes multi-entity items; it is not a population-random sample, "
        "so do not use it to estimate full-dataset class proportions. Do not label using later market movements. "
        "Record the actual review process; one reviewer does not establish two-reviewer agreement.\n\n"
        "task_exclusions.jsonl freezes all distinct source texts, including items not sampled for review and items whose source fields could not be parsed. "
        "Exclude these texts from tokenizer and pretraining corpora. Isolating the candidate pool does not constitute a completed train/test split.\n\n"
        f"Source: https://github.com/pyRis/SEntFiN/tree/{REVISION}\n\n"
        f"Attribution: {ATTRIBUTION}. See SOURCE_LICENSE.txt.\n", encoding="utf-8")

    manifest = {
        "schema_version": 1, "status": "complete", "source_repository": "https://github.com/pyRis/SEntFiN",
        "source_revision": REVISION, "license": "MIT", "attribution": ATTRIBUTION,
        "paper": "https://arxiv.org/abs/2305.12257", "review_seed": seed,
        "source_headline_rows": len(rows), "unique_exact_headline_texts": len(headlines),
        "source_duplicate_headline_rows": len(rows) - len(headlines),
        "parsed_target_annotations": len(pool), "quarantined_source_rows": quarantined,
        "quarantine_count": len(quarantined),
        "review_rows": len(selected), "review_unique_headlines": len({r["text"] for r in selected}),
        "review_source_sentiment_counts": dict(collections.Counter(r["source_sentiment"] for r in selected)),
        "review_multi_entity_headlines": sum(r["source_entity_count"] > 1 for r in selected),
        "human_stance_labels_created": 0, "human_reviewed_rows": 0,
        "exclusion_scope": "all exact-unique candidate headlines, including quarantined rows",
        "source_files": {name: {"url": f"{BASE_URL}/{name}", "bytes": size, "sha256": sha}
                         for name, (size, sha) in FILES.items()},
        "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                      for p in sorted(output.iterdir()) if p.is_file()},
        "limitations": ["Entity sentiment is not automatically asset stance.",
                        "No verified asset/ticker map or timestamps exist in the source headline CSV.",
                        "No X posts are scraped or hydrated.",
                        "Original author human annotation does not imply local human stance review.",
                        "Malformed source dictionaries are quarantined rather than silently repaired."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/candidates/sentfin"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--download", action="store_true", help="Download missing pinned MIT source files, maximum 1.21 MB total")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=130)
    args = parser.parse_args()
    result = prepare(args.source_dir, args.output, args.count, args.seed, args.download)
    print(json.dumps({k: result[k] for k in ("source_headline_rows", "unique_exact_headline_texts",
                     "quarantine_count", "review_rows", "review_source_sentiment_counts",
                     "review_multi_entity_headlines", "human_stance_labels_created")}, indent=2))


if __name__ == "__main__":
    main()
