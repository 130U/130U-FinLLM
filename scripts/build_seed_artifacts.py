"""Prepare the reviewed seed corpora, an 8K tokenizer and a separate 24M pilot config."""
import argparse
import json
from pathlib import Path

from finllm.cli import main as finllm
from finllm.data import file_sha256, write_json
from finllm.preparation import prepare_pretraining_inputs


def validate_source_snapshot(corpora):
    corpora = Path(corpora)
    manifest = json.loads((corpora / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Corpus download is incomplete; do not prepare partial source files")
    for name in ("general", "finance"):
        source = corpora / f"{name}_candidate.jsonl"
        expected = manifest[name]
        if source.stat().st_size != expected["bytes"] or file_sha256(source) != expected["sha256"]:
            raise ValueError(f"Source snapshot changed after download: {name}")
    return manifest


def build(corpora, task_pool, prepared, tokenizer, processed, run_output):
    corpora, prepared, tokenizer, processed = map(Path, (corpora, prepared, tokenizer, processed))
    for target in (prepared, tokenizer, processed, Path(run_output)):
        if target.exists():
            raise FileExistsError(f"Use a new version: {target}")
    validate_source_snapshot(corpora)
    prep = prepare_pretraining_inputs(
        corpora / "general_candidate.jsonl", corpora / "finance_candidate.jsonl",
        task_pool, prepared, tokenizer_max_bytes=12_000_000,
    )
    exclusions = prepared / "frozen_exclusions.jsonl"
    finllm(["train-tokenizer", "--input", str(prepared / "tokenizer_train.jsonl"),
            "--output", str(tokenizer), "--vocab-size", "8192", "--exclude", str(exclusions)])
    counts = {}
    for source in ("general", "finance"):
        for partition in ("train", "validation"):
            name = source if partition == "train" else source + "_validation"
            excluded = exclusions if partition == "train" else prepared / "frozen_task_exclusions.jsonl"
            finllm(["pack", "--input", str(prepared / f"{source}_{partition}.jsonl"),
                    "--tokenizer", str(tokenizer), "--output", str(processed / name),
                    "--exclude", str(excluded), "--partition", partition])
            counts[name] = json.loads((processed / name / "manifest.json").read_text())["token_count"]
    config = json.loads(Path("configs/pretrain_pilot.json").read_text())
    config.update(tokenizer=tokenizer.as_posix(),
                  data=(processed / "general/manifest.json").as_posix(),
                  validation_data=(processed / "general_validation/manifest.json").as_posix(),
                  output_dir=Path(run_output).as_posix())
    write_json(processed / "pilot_config.json", config)
    report = {"status": "seed_inputs_prepared_training_not_started", "token_counts": counts,
              "tokenizer_sha256": file_sha256(tokenizer / "tokenizer.json"),
              "tokenizer_vocabulary": 8192,
              "task_exclusion_sha256": file_sha256(task_pool),
              "preparation_manifest_sha256": file_sha256(prepared / "preparation_manifest.json"),
              "pilot_config": (processed / "pilot_config.json").as_posix(),
              "limits": ["Convenience Wikipedia subset and macroeconomic minutes are not tweet coverage.",
                         "All candidate task headlines excluded; stance/asset mapping still requires human review.",
                         "No SFT, DPO or final finance evaluation has run."]}
    write_json(processed / "seed_summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", required=True)
    parser.add_argument("--task-pool", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--run-output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), ensure_ascii=False, indent=2))
