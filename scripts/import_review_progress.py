"""Import the offline review workspace's progress JSON into a private audit package."""
from __future__ import annotations

import argparse
import json
import sys

from finllm.review_import import import_review_progress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Exact original ai_annotations.jsonl")
    parser.add_argument("--progress", required=True, help="UI review_progress_*.json export")
    parser.add_argument("--output", required=True, help="New private version directory, e.g. runs/review_imports/review_v1")
    args = parser.parse_args(argv)
    try:
        result = import_review_progress(args.source, args.progress, args.output)
    except (ValueError, OSError):
        # Avoid echoing source news, reviewer notes, or malformed input in logs.
        print(json.dumps({"status": "rejected", "formal_training_ready": False,
                          "reason": "Review package validation or output preservation check failed; verify the exact source/progress and use a new directory path that does not yet exist."}),
              file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
