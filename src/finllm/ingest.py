"""Bounded source ingestion. Rights acknowledgement is recorded, not inferred."""
import json
import re
from pathlib import Path

from .data import file_sha256, write_json


def fetch_corpus(args):
    if not args.training_rights_reviewed:
        raise ValueError("Review the original dataset/source terms first, then use --training-rights-reviewed")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError("Use an immutable 40-character dataset commit, not a moving main revision")
    if args.max_documents < 1 or args.max_bytes < 1:
        raise ValueError("Ingestion limits must be positive")
    path = Path(args.output)
    if path.exists():
        raise FileExistsError("Choose a new output file to preserve the recorded source snapshot")
    from datasets import load_dataset
    dataset = load_dataset(args.dataset, name=args.config, split=args.split, revision=args.revision, streaming=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    count, written = 0, 0
    with path.open("w", encoding="utf-8") as stream:
        for source in dataset:
            text = source.get(args.text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            # Oversized source documents require deliberate segmentation before ingestion.
            if len(text.encode("utf-8")) > 1_000_000:
                continue
            row = {"id": str(source.get("id", count)), "text": text, "split": "train",
                   "source": f"https://huggingface.co/datasets/{args.dataset}", "source_revision": args.revision,
                   "rights_note": args.rights_note, "training_allowed": True,
                   "source_url": source.get("url")}
            line = json.dumps(row, ensure_ascii=False) + "\n"
            size = len(line.encode("utf-8"))
            if written + size > args.max_bytes:
                break
            stream.write(line)
            written += size
            count += 1
            if count >= args.max_documents:
                break
    report = {"dataset": args.dataset, "revision": args.revision, "config": args.config, "source_split": args.split,
              "documents": count, "bytes": written, "sha256": file_sha256(path), "rights_note": args.rights_note,
              "note": "Contains only a bounded source subset; not yet deduplicated or checked against frozen evaluation data"}
    write_json(path.with_suffix(path.suffix + ".manifest.json"), report)
    return report
