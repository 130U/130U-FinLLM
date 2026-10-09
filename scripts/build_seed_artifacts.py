"""Prepare the reviewed seed corpora, an 8K tokenizer and a separate 24M pilot config."""
import argparse
import json
from pathlib import Path

from finllm.cli import main as finllm
from finllm.data import TokenBlockDataset, file_sha256, write_json
from finllm.preparation import prepare_pretraining_inputs
from finllm.tokenizer import load_tokenizer


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


def pilot_config(tokenizer, processed, run_output):
    config = json.loads(Path("configs/pretrain_pilot.json").read_text(encoding="utf-8"))
    config.update(tokenizer=Path(tokenizer).as_posix(),
                  data=(Path(processed) / "general/manifest.json").as_posix(),
                  validation_data=(Path(processed) / "general_validation/manifest.json").as_posix(),
                  output_dir=Path(run_output).as_posix())
    return config


def validate_existing(corpora, task_pool, prepared, tokenizer, processed, run_output):
    """Read-only validation, including outputs made before rerun support existed."""
    corpora, prepared, tokenizer, processed = map(Path, (corpora, prepared, tokenizer, processed))

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def check(condition, message):
        if not condition:
            raise ValueError(message)

    try:
        source = validate_source_snapshot(corpora)
        report = read(processed / "seed_summary.json")
        check(report["status"] == "seed_inputs_prepared_training_not_started", "Missing completion status")
        check(not (prepared / "INCOMPLETE").exists(), "Preparation was interrupted")
        preparation = read(prepared / "preparation_manifest.json")
        check(file_sha256(prepared / "preparation_manifest.json") == report["preparation_manifest_sha256"],
              "Preparation manifest changed")
        check(preparation["input_sha256"] == {
            "general": source["general"]["sha256"], "finance": source["finance"]["sha256"],
            "task_exclusions": file_sha256(task_pool)}, "Source inputs changed")
        check(report["task_exclusion_sha256"] == file_sha256(task_pool), "Task pool changed")
        check(preparation["mode"] == "formal" and preparation["seed"] == 130
              and preparation["validation_fraction"] == .05
              and preparation["tokenizer"]["max_bytes"] == 12_000_000, "Preparation settings changed")
        expected_files = {"general_train.jsonl", "general_validation.jsonl", "finance_train.jsonl",
                          "finance_validation.jsonl", "frozen_exclusions.jsonl",
                          "frozen_task_exclusions.jsonl", "tokenizer_train.jsonl"}
        check(set(preparation["files"]) == expected_files, "Incomplete preparation file list")
        for name, info in preparation["files"].items():
            path = prepared / name
            check(path.stat().st_size == info["bytes"] and file_sha256(path) == info["sha256"],
                  f"Prepared file changed: {name}")
        tokenizer_sha = file_sha256(tokenizer / "tokenizer.json")
        check(tokenizer_sha == report["tokenizer_sha256"], "Tokenizer changed")
        vocabulary = load_tokenizer(tokenizer).get_vocab_size()
        check(vocabulary == report["tokenizer_vocabulary"] == 8192, "Tokenizer vocabulary changed")
        metadata, provenance = read(tokenizer / "tokenizer_meta.json"), read(tokenizer / "provenance.json")
        check(metadata["sha256"] == tokenizer_sha and metadata["actual_vocab_size"] == vocabulary,
              "Tokenizer metadata changed")
        check(provenance == {"mode": "formal", "vocab_size": vocabulary,
              "sources": [{"sha256": file_sha256(prepared / "tokenizer_train.jsonl")}],
              "eval_exclusion_sha256": file_sha256(prepared / "frozen_exclusions.jsonl"),
              "tokenizer_sha256": tokenizer_sha}, "Tokenizer training provenance changed")
        counts = {}
        for domain in ("general", "finance"):
            for partition in ("train", "validation"):
                name = domain if partition == "train" else domain + "_validation"
                dataset = TokenBlockDataset(processed / name / "manifest.json", seq_len=1)
                manifest = dataset.manifest
                exclusion = "frozen_exclusions.jsonl" if partition == "train" else "frozen_task_exclusions.jsonl"
                check(manifest["mode"] == "formal" and manifest["partition"] == partition
                      and manifest["tokenizer_sha256"] == tokenizer_sha and manifest["vocab_size"] == vocabulary
                      and manifest["source_sha256"] == file_sha256(prepared / f"{domain}_{partition}.jsonl")
                      and manifest["eval_exclusion_sha256"] == file_sha256(prepared / exclusion),
                      f"Packed provenance changed: {name}")
                counts[name] = manifest["token_count"]
                del dataset
        check(counts == report["token_counts"], "Token counts changed")
        check(read(processed / "pilot_config.json") == pilot_config(tokenizer, processed, run_output),
              "Pilot configuration or requested paths changed")
        check(Path(report["pilot_config"]) == processed / "pilot_config.json", "Pilot config path changed")
        return report
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Existing seed is incomplete or changed: {error}. Existing files were preserved. "
            "Restore a verified backup or choose a NEW SEED_VERSION; do not delete the old seed."
        ) from error


def build(corpora, task_pool, prepared, tokenizer, processed, run_output, validate_only=False):
    corpora, prepared, tokenizer, processed = map(Path, (corpora, prepared, tokenizer, processed))
    if validate_only or any(target.exists() for target in (prepared, tokenizer, processed, Path(run_output))):
        report = validate_existing(corpora, task_pool, prepared, tokenizer, processed, run_output)
        print("Existing seed verified; reusing without rebuilding tokenizer or overwriting files.", flush=True)
        return report
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
    write_json(processed / "pilot_config.json", pilot_config(tokenizer, processed, run_output))
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
    parser.add_argument("--validate-only", action="store_true", help="Verify an existing complete seed without writing")
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), ensure_ascii=False, indent=2))
