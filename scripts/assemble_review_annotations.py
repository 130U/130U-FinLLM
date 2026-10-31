"""Join per-record AI decisions to source rows without approving human labels.

Annotation TSV files are actual model review outputs, not source-sentiment maps.
The assembler validates their integrity; it does not establish label accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

from finllm.data import DuplicateIndex


STANCES = {"bullish", "bearish", "neutral", "exclude"}
KINDS = {"company", "index", "commodity", "currency", "sector", "fund", "bond", "regulator", "person", "macro", "unknown"}


def assemble(input_path, annotations, output):
    input_path, output = Path(input_path), Path(output)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    by_number = {row["review_no"]: row for row in rows}
    if len(by_number) != len(rows) or len({r["id"] for r in rows}) != len(rows) or not rows:
        raise ValueError("Source review numbers and IDs must be nonempty and unique")
    if any(r.get("human_reviewed") not in (False, "false", "", None) or r.get("human_stance") or r.get("asset_identifier") for r in rows):
        raise ValueError("Preserve existing human decisions separately; AI assembly cannot approve or replace them")
    decisions, provenance = {}, {}
    for path in map(Path, annotations):
        provenance[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for fields in csv.reader(stream, delimiter="\t"):
                if len(fields) != 7:
                    raise ValueError("AI annotation requires seven TSV columns")
                number = int(fields[0])
                stance, quote, reason, kind, difficulty, priority = fields[1:]
                if number not in by_number or number in decisions:
                    raise ValueError("Unknown or repeated AI review number")
                if (stance not in STANCES or not quote or quote not in by_number[number]["text"]
                        or not reason.strip() or kind not in KINDS
                        or difficulty not in {"easy", "medium", "hard"} or priority not in {"low", "medium", "high"}):
                    raise ValueError(f"Invalid AI decision or nonliteral evidence: {number}")
                decisions[number] = {"ai_stance": stance, "proposed_stance": stance,
                    "ai_rationale": reason, "evidence_quotes": [quote], "asset_kind": kind,
                    "difficulty": difficulty, "priority": priority,
                    "ai_reviewer": "Codex semantic draft; pending user review"}
    if decisions.keys() != by_number.keys():
        raise ValueError("Every source row must have exactly one AI decision")
    parent = {r["review_no"]: r["review_no"] for r in rows}

    def find(number):
        while parent[number] != number:
            parent[number] = parent[parent[number]]
            number = parent[number]
        return number

    duplicates = DuplicateIndex()
    for row in rows:
        for other in duplicates.matches(row["text"]):
            a, b = find(row["review_no"]), find(other)
            parent[max(a, b)] = min(a, b)
        duplicates.add(row["review_no"], row["text"])
    group_sizes = Counter(find(r["review_no"]) for r in rows)
    result = []
    for row in rows:
        decision = decisions[row["review_no"]]
        candidates = row.get("source_symbol_candidates", [])
        group = f"lexical:{find(row['review_no']):04d}"
        draft_asset = " / ".join(candidates) if candidates else "Pending verification: " + row["source_entity"]
        item = {**row, **decision, "ai_asset_candidate": draft_asset,
                "asset_candidate_scope": "Historical author symbol or target name only; market/security identity remains unverified.",
                "group_id": group, "lexical_group_size": group_sizes[find(row["review_no"])],
                "group_scope": "Normalized exact plus SimHash64 Hamming<=3 candidates; not semantic event verification.",
                "human_reviewed": False, "human_stance": "", "asset_identifier": "",
                "human_asset_verified": False, "formal_training_ready": False}
        if item.get("mapping_status") != "unique_historical_candidate":
            item["priority"] = "high"
        result.append(item)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result).encode("utf-8")
    if output.exists() and output.read_bytes() != body:
        raise ValueError("Different existing annotations preserved; choose a new output version")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_bytes(body)
    summary = {"rows": len(result), "unique_normalized_headlines": len({r["text_group_id"] for r in rows}),
        "original_200_rows": sum(bool(r.get("original_200")) for r in rows),
        "ai_stance_counts": dict(Counter(r["ai_stance"] for r in result)),
        "priority_counts": dict(Counter(r["priority"] for r in result)),
        "difficulty_counts": dict(Counter(r["difficulty"] for r in result)),
        "lexical_groups": len(group_sizes), "rows_in_multirow_lexical_groups": sum(n for n in group_sizes.values() if n > 1),
        "human_reviewed_rows": 0, "human_asset_verified_rows": 0, "formal_training_ready": False,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "annotation_files_sha256": provenance, "output_sha256": hashlib.sha256(body).hexdigest(),
        "quality_scope": "AI draft proposals with literal evidence and Chinese reasons; human accuracy and event independence not established."}
    return result, summary


def export_csv(rows, output):
    fields = ["review_no", "batch_id", "priority", "difficulty", "source_entity", "text", "ai_stance",
              "ai_rationale", "evidence_quotes", "ai_asset_candidate", "mapping_status", "asset_kind",
              "secondary_ai_stance", "secondary_ai_reason", "ai_disagreement", "ai_revision_note",
              "human_stance", "human_reviewed", "asset_identifier", "human_asset_verified",
              "human_notes", "event_group_id", "group_id", "id", "source_url", "source_sentiment"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        record = {k: row.get(k, "") for k in fields}
        record["evidence_quotes"] = " | ".join(row["evidence_quotes"])
        for k, value in record.items():
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                record[k] = "'" + value
        writer.writerow(record)
    body = stream.getvalue().encode("utf-8-sig")
    output = Path(output)
    if output.exists() and output.read_bytes() != body:
        raise ValueError("Existing review CSV has changes; preserve it and export to a new path")
    if not output.exists():
        output.write_bytes(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    rows, summary = assemble(args.input, args.annotations, args.output)
    if args.csv:
        export_csv(rows, args.csv)
    if args.summary:
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
