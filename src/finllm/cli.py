"""Explicit commands; importing the package never downloads or trains a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import (DuplicateIndex, file_sha256, pack_corpus, prepare_task, read_jsonl,
                   tokenizer_file, validate_training_row, write_json, write_jsonl)


def filtered_texts(inputs, exclude, smoke):
    heldout = DuplicateIndex()
    if exclude:
        for index, row in enumerate(read_jsonl(exclude)):
            heldout.add(index, row["text"])
    elif not smoke:
        raise ValueError("Formal tokenizer training requires --exclude frozen_eval.jsonl")
    for path in inputs:
        for row in read_jsonl(path):
            validate_training_row(row, smoke)
            if row.get("split", "train") != "train":
                raise ValueError("Tokenizer input must contain only training rows")
            if not heldout.matches(row["text"]):
                yield row["text"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="130U-FinLLM public Trial: training from random weights")
    commands = parser.add_subparsers(dest="command", required=True)
    env = commands.add_parser("environment")
    env.add_argument("--output")
    smoke = commands.add_parser("smoke", help="Small synthetic engineering check, no real data download")
    smoke.add_argument("--output", default="runs/smoke")
    smoke.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    prep = commands.add_parser("prepare-smoke")
    prep.add_argument("--output", default="data/smoke")
    task = commands.add_parser("prepare-task")
    task.add_argument("--input", required=True)
    task.add_argument("--output", required=True)
    task.add_argument("--smoke", action="store_true")
    tok = commands.add_parser("train-tokenizer")
    tok.add_argument("--input", required=True, action="append")
    tok.add_argument("--output", required=True)
    tok.add_argument("--vocab-size", type=int, default=8192)
    tok.add_argument("--exclude")
    tok.add_argument("--smoke", action="store_true")
    pack = commands.add_parser("pack")
    pack.add_argument("--input", required=True)
    pack.add_argument("--tokenizer", required=True)
    pack.add_argument("--output", required=True)
    pack.add_argument("--shard-tokens", type=int, default=5_000_000)
    pack.add_argument("--exclude")
    pack.add_argument("--smoke", action="store_true")
    pack.add_argument("--partition", choices=["train", "validation"], default="train")
    freeze = commands.add_parser("freeze-exclusions")
    freeze.add_argument("--input", required=True, action="append")
    freeze.add_argument("--output", required=True)
    mix = commands.add_parser("mix-corpora")
    mix.add_argument("--general", required=True)
    mix.add_argument("--finance", required=True)
    mix.add_argument("--general-tokens", type=int, default=5_000_000)
    mix.add_argument("--finance-tokens", type=int, default=20_000_000)
    mix.add_argument("--output", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--resume", help="Trusted local checkpoint, including its .json integrity marker")
    for command in ("evaluate", "predict", "calibrate"):
        action = commands.add_parser(command)
        action.add_argument("--checkpoint", required=True)
        action.add_argument("--tokenizer", required=True)
        action.add_argument("--output", required=True)
        action.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
        action.add_argument("--target-mode", default="ticker", choices=["ticker", "literal_entity"],
                            help="literal_entity is an exploratory headline-entity mode, not security verification")
        if command == "calibrate":
            action.add_argument("--temperature-data", required=True)
            action.add_argument("--threshold-data", required=True)
        else:
            action.add_argument("--input", required=True)
            action.add_argument("--calibration")
    sync = commands.add_parser("sync")
    sync.add_argument("--source", required=True)
    sync.add_argument("--destination", required=True)
    sync.add_argument("--working-copy", action="store_true", help="Restore verified files into a mutable working directory")
    prefs = commands.add_parser("make-preferences", help="Create candidate pairs requiring manual review before formal DPO")
    prefs.add_argument("--input", required=True)
    prefs.add_argument("--output", required=True)
    baseline = commands.add_parser("baseline")
    baseline.add_argument("--train", required=True)
    baseline.add_argument("--input", required=True)
    baseline.add_argument("--output", required=True)
    fetch = commands.add_parser("fetch-corpus", help="Stream a bounded, reviewed HF dataset subset")
    fetch.add_argument("--dataset", required=True)
    fetch.add_argument("--revision", required=True, help="Immutable source revision")
    fetch.add_argument("--config")
    fetch.add_argument("--split", default="train")
    fetch.add_argument("--text-field", default="text")
    fetch.add_argument("--max-documents", type=int, default=1000)
    fetch.add_argument("--max-bytes", type=int, default=20_000_000)
    fetch.add_argument("--rights-note", required=True)
    fetch.add_argument("--training-rights-reviewed", action="store_true")
    fetch.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "environment":
        from .runtime import environment_report
        result = environment_report()
        if args.output:
            write_json(args.output, result)
    elif args.command == "smoke":
        from .smoke import run_smoke
        result = run_smoke(args.output, args.device)
    elif args.command == "prepare-smoke":
        from .smoke import prepare_smoke
        result = prepare_smoke(args.output)
    elif args.command == "prepare-task":
        result = prepare_task(args.input, args.output, args.smoke)
    elif args.command == "train-tokenizer":
        from .tokenizer import train_tokenizer
        target = Path(args.output)
        if (target / "tokenizer.json").exists():
            raise FileExistsError("Tokenizer already exists; use a new directory for a new version")
        tokenizer = train_tokenizer(filtered_texts(args.input, args.exclude, args.smoke), target, args.vocab_size)
        result = {"mode": "smoke" if args.smoke else "formal", "vocab_size": tokenizer.get_vocab_size(),
                  "sources": [{"sha256": file_sha256(path)} for path in args.input],
                  "eval_exclusion_sha256": file_sha256(args.exclude) if args.exclude else None,
                  "tokenizer_sha256": file_sha256(tokenizer_file(target))}
        write_json(target / "provenance.json", result)
        if not args.smoke and tokenizer.get_vocab_size() != args.vocab_size:
            raise ValueError("Corpus too small to learn requested vocabulary; add diverse training text and use a new output directory")
    elif args.command == "pack":
        result = {"manifest": str(pack_corpus(args.input, args.tokenizer, args.output, args.shard_tokens, args.exclude, args.smoke, args.partition))}
    elif args.command == "freeze-exclusions":
        if Path(args.output).exists():
            raise FileExistsError("Frozen exclusion file exists; select a new version")
        write_jsonl(args.output, (row for path in args.input for row in read_jsonl(path)))
        result = {"sha256": file_sha256(args.output), "sources": [file_sha256(path) for path in args.input]}
        write_json(args.output + ".manifest.json", result)
    elif args.command == "mix-corpora":
        from .corpus import mix_corpora
        result = mix_corpora(args.general, args.finance, args.output, args.general_tokens, args.finance_tokens)
    elif args.command == "train":
        from .training import run_training
        config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        if args.resume:
            config["resume"] = args.resume
            config.pop("init_checkpoint", None)
        result = run_training(config)
    elif args.command in ("predict", "evaluate", "calibrate"):
        from .inference import run_prediction
        from .evaluation import run_evaluation, run_calibration
        if args.command == "calibrate":
            result = run_calibration(args.checkpoint, args.tokenizer, args.temperature_data, args.threshold_data,
                                     args.output, device=args.device, target_mode=args.target_mode)
        else:
            function = run_prediction if args.command == "predict" else run_evaluation
            result = function(args.checkpoint, args.tokenizer, args.input, args.output,
                              calibration_path=args.calibration, device=args.device, target_mode=args.target_mode)
    elif args.command == "sync":
        from .runtime import sync_snapshot
        result = sync_snapshot(args.source, args.destination, working_copy=args.working_copy)
    elif args.command == "make-preferences":
        from .tokenizer import LABELS
        candidates = []
        for row in read_jsonl(args.input):
            validate_training_row(row)
            if row.get("split") != "train" or row.get("label") not in LABELS:
                raise ValueError("Preferences must come only from audited labeled training rows")
            alternatives = [label for label in LABELS if label != row["label"]]
            candidates.append(dict(row, chosen=row["label"], rejected=alternatives[len(candidates) % 2], preference_reviewed=False))
        write_jsonl(args.output, candidates)
        result = {"candidate_pairs": len(candidates), "requires_review": True, "output": args.output}
    elif args.command == "baseline":
        from .baselines import run_baseline
        result = run_baseline(args.train, args.input, args.output)
    else:
        from .ingest import fetch_corpus
        result = fetch_corpus(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
