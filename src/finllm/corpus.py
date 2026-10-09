"""Construct a deterministic mixture with an exact token budget per source."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from .data import TokenBlockDataset, file_sha256, write_json


def mix_corpora(general_manifest, finance_manifest, output, general_tokens, finance_tokens, seed=130):
    if general_tokens < 1 or finance_tokens < 1:
        raise ValueError("Both mixture token budgets must be positive")
    inputs = [TokenBlockDataset(path, 512) for path in (general_manifest, finance_manifest)]
    general, finance = [dataset.manifest for dataset in inputs]
    for field in ("tokenizer_sha256", "vocab_size", "mode", "eval_exclusion_sha256", "partition"):
        if general.get(field) != finance.get(field):
            raise ValueError(f"Mixture sources disagree on {field}")
    if general.get("partition", "train") != "train":
        raise ValueError("The domain mixture must use only training corpora")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Mixture output must be a new empty directory")
    budgets = [general_tokens, finance_tokens]
    remaining = budgets.copy()
    positions = [[0, 0], [0, 0]]
    rng = random.Random(seed)
    buffer, shards = [], []

    def take(source, count):
        pieces = []
        while count:
            shard, offset = positions[source]
            array = inputs[source].shards[shard]
            amount = min(count, len(array) - offset)
            pieces.extend(map(int, array[offset:offset + amount]))
            offset += amount
            count -= amount
            if offset == len(array):
                shard, offset = (shard + 1) % len(inputs[source].shards), 0
            positions[source] = [shard, offset]
        return pieces

    def flush():
        path = root / f"tokens-{len(shards):05d}.bin"
        np.asarray(buffer, dtype="<u2").tofile(path)
        shards.append({"path": path.name, "count": len(buffer), "sha256": file_sha256(path)})
        buffer.clear()

    while sum(remaining):
        selected = 0 if rng.random() * sum(remaining) < remaining[0] else 1
        amount = min(512, remaining[selected], 5_000_000 - len(buffer))
        buffer.extend(take(selected, amount))
        remaining[selected] -= amount
        if len(buffer) == 5_000_000:
            flush()
    if buffer:
        flush()
    source_hashes = [file_sha256(path) for path in (general_manifest, finance_manifest)]
    result = {"format": "finllm-uint16-v1", "mode": general["mode"], "partition": "train", "dtype": "<u2",
              "tokenizer_sha256": general["tokenizer_sha256"], "vocab_size": general["vocab_size"],
              "token_count": sum(budgets), "shards": shards,
              "source_sha256": hashlib.sha256(json.dumps(source_hashes).encode()).hexdigest(),
              "eval_exclusion_sha256": general.get("eval_exclusion_sha256"), "source_manifests": source_hashes,
              "mixture_token_counts": {"general": general_tokens, "finance": finance_tokens},
              "source_unique_token_counts": {"general": general["token_count"], "finance": finance["token_count"]},
              "seed": seed, "note": "512-token interleaving; cycles source when needed. Mixture counts are processed, not unique tokens. Boundary transitions remain ordinary causal attention."}
    write_json(root / "manifest.json", result)
    return {"manifest": str(root / "manifest.json"), "mixture_token_counts": result["mixture_token_counts"],
            "source_unique_token_counts": result["source_unique_token_counts"]}
