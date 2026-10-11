"""Build additional versioned corpora with the unchanged seed_v1 tokenizer."""
import argparse
import json

from finllm.expansion import build_expansion_artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpora", "task-pool", "seed-prepared", "seed-processed", "tokenizer", "prepared", "processed"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--validation-fraction", type=float, default=.05)
    parser.add_argument("--shard-tokens", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=130)
    parser.add_argument("--validate-only", action="store_true")
    print(json.dumps(build_expansion_artifacts(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
