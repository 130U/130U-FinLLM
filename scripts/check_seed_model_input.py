"""One CPU forward/backward on real seed tokens; never save trained weights."""
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from finllm.data import TokenBlockDataset, file_sha256, write_json
from finllm.model import FinLLM, ModelConfig


def main():
    torch.set_num_threads(2)
    torch.manual_seed(130)
    model = FinLLM(ModelConfig())
    dataset = TokenBlockDataset("data/processed/seed_v1/general/manifest.json", 128)
    batch = dataset[0]
    start = time.perf_counter()
    x = batch["input_ids"].unsqueeze(0)
    y = batch["labels"].unsqueeze(0)
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    loss.backward()
    if not bool(torch.isfinite(loss)) or not all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise FloatingPointError("Nonfinite model/data integration check")
    report = {"status": "passed", "device": "cpu", "parameters": sum(p.numel() for p in model.parameters()),
              "batch_size": 1, "sequence_length": 128, "loss": float(loss.detach()),
              "elapsed_seconds": time.perf_counter() - start,
              "tokenizer_sha256": file_sha256("tokenizer/seed_v1/tokenizer.json"),
              "data_manifest_sha256": file_sha256("data/processed/seed_v1/general/manifest.json"),
              "optimizer_updates": 0, "saved_weights": False,
              "scope": "Real-token CPU forward/backward only; not the T4 pilot or a finance quality result"}
    write_json("reports/seed_model_input_check.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
