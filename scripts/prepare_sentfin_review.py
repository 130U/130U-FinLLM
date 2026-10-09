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


def prepare(source_dir: Path, output: Path, count: int = 200, seed: int = 130, download: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new or empty directory; review files are never overwritten")
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
        "# 130U FinLLM 人工审核表\n\n"
        "这是候选审核表，尚无任何本项目人工确认的资产立场标签。source_sentiment 是原作者的金融实体情绪标签，"
        "不能直接改名为 bullish/bearish。proposed_stance 故意留空。\n\n"
        "逐行确认 source_entity 是可投资资产、公司或适用资产类别；在 asset_identifier 填写明确的资产标识，"
        "不确定时不要猜股票代码。填写 human_stance 为 bullish、bearish、neutral 或 exclude。"
        "bullish/bearish 必须来自本文对指定目标表达的投资方向或前景；单纯已发生的涨跌事实默认 neutral。"
        "监管机构、不明确目标、讽刺或归属不清时填写 exclude 并解释。完成后才将 human_reviewed 改为 true。\n\n"
        "同一新闻的多实体标签要分别判断。本表按原标签分层，并包含多实体新闻；它不是总体随机样本，"
        "不能用它估计完整数据的类别比例。不要依据后来市场涨跌补标签。"
        "一人审核不等于两人独立一致性验证，必须如实记录。\n\n"
        "task_exclusions.jsonl 已冻结全部不同原文，包含未抽中审核以及源字段解析失败的新闻；"
        "这些文本从 tokenizer 和预训练语料排除。整个候选池保持隔离不等于已完成训练/测试划分。\n\n"
        f"Source: https://github.com/pyRis/SEntFiN/tree/{REVISION}\n\n"
        f"Attribution: {ATTRIBUTION}. See SOURCE_LICENSE.txt.\n", encoding="utf-8")

    manifest = {
        "schema_version": 1, "source_repository": "https://github.com/pyRis/SEntFiN",
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
