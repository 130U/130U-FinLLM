"""A synthetic engineering exercise; never a financial performance benchmark."""
from pathlib import Path

from .data import pack_corpus, write_json, write_jsonl
from .runtime import environment_report
from .tokenizer import make_prompt, train_tokenizer


def prepare_smoke(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    patterns = {
        "bearish": ["I am bearish on ${asset}; I expect weaker demand.", "I would sell ${asset}; margins look weak."],
        "neutral": ["${asset} reports results on Thursday.", "The ${asset} investor meeting starts tomorrow."],
        "bullish": ["I am bullish on ${asset}; I expect stronger demand.", "I would buy ${asset}; growth looks promising."],
    }
    splits = {name: [] for name in ("train", "development", "calibration_temperature", "calibration_threshold", "test")}
    for split_number, (split, rows) in enumerate(splits.items()):
        for i, asset in enumerate(("AAPL", "MSFT", "TSLA", "SPY")):
            for label, phrases in patterns.items():
                for j, phrase in enumerate(phrases):
                    identifier = f"synthetic-{split}-{i}-{label}-{j}"
                    rows.append({"id": identifier, "group_id": identifier,
                                 "text": phrase.format(asset=asset) + f" Example {split_number}-{i}-{j}.",
                                 "target": asset, "label": label, "split": split, "synthetic": True,
                                 "source": "project-authored synthetic smoke fixtures", "training_allowed": True,
                                 "rights_note": "Project-authored examples, engineering checks only"})
        write_jsonl(root / f"{split}.jsonl", rows)
    corpus = [dict(row, text=make_prompt(row["text"], row["target"])) for row in splits["train"]]
    write_jsonl(root / "corpus.jsonl", corpus)
    prefs = [dict(row, chosen=row["label"], rejected={"bearish": "bullish", "neutral": "bearish", "bullish": "bearish"}[row["label"]])
             for row in splits["train"]]
    write_jsonl(root / "preferences_train.jsonl", prefs)
    return {"directory": str(root), "synthetic_only": True, "rows_per_split": 24}


def run_smoke(output, device="auto"):
    import torch
    from .training import run_training
    from .inference import run_prediction
    from .evaluation import run_evaluation, run_calibration
    root = Path(output).resolve()
    if (root / "summary.json").exists() or (root / "data").exists():
        raise FileExistsError("Smoke output exists. Use a new --output directory so old evidence is preserved.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    prepare_smoke(root / "data")
    from .data import read_jsonl
    tokenizer_dir = root / "tokenizer"
    tok = train_tokenizer((row["text"] for row in read_jsonl(root / "data/corpus.jsonl")), tokenizer_dir, 512)
    packed = pack_corpus(root / "data/corpus.jsonl", tokenizer_dir, root / "packed", 1000, smoke=True)
    model = {"vocab_size": tok.get_vocab_size(), "hidden_size": 64, "num_hidden_layers": 2,
             "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 192,
             "max_position_embeddings": 256}
    common = {"mode": "smoke", "model": model, "tokenizer": str(tokenizer_dir),
              "train": {"device": device, "seed": 130, "batch_size": 2, "grad_accumulation": 2,
                        "seq_len": 256, "max_updates": 2, "schedule_updates": 3,
                        "learning_rate": .001, "warmup_updates": 0, "weight_decay": .01,
                        "clip_grad": 1., "save_every": 1, "amp": True, "num_workers": 0}}
    results, last = {}, None
    for stage in ("pretrain", "domain", "sft", "dpo"):
        config = dict(common, stage=stage, output_dir=str(root / stage))
        config["data"] = str(packed if stage in ("pretrain", "domain") else root / "data" / ("preferences_train.jsonl" if stage == "dpo" else "train.jsonl"))
        if last:
            config["init_checkpoint"] = last
        if stage == "dpo":
            config["reference_checkpoint"] = last
        config["validation_data"] = str(packed if stage in ("pretrain", "domain") else root / "data/development.jsonl")
        write_json(root / f"config_{stage}.json", config)
        results[stage] = run_training(config)
        last = results[stage]["latest_checkpoint"]
    # Exercise full optimizer/data/RNG restore, not just loading inference weights.
    config = dict(config, resume=last)
    config.pop("init_checkpoint", None)
    config["train"] = dict(common["train"], max_updates=3)
    results["resume"] = run_training(config)
    last = results["resume"]["latest_checkpoint"]
    calibration = root / "calibration.json"
    run_calibration(last, str(tokenizer_dir), str(root / "data/calibration_temperature.jsonl"),
                    str(root / "data/calibration_threshold.jsonl"), str(calibration), device=device)
    run_evaluation(last, str(tokenizer_dir), str(root / "data/test.jsonl"), str(root / "metrics.json"),
                   calibration_path=str(calibration), device=device)
    run_prediction(last, str(tokenizer_dir), str(root / "data/test.jsonl"), str(root / "predictions.jsonl"),
                   calibration_path=str(calibration), device=device)
    summary = {"status": "engineering_smoke_completed", "synthetic_only": True,
               "warning": "These tiny runs and overlapping synthetic templates do not measure financial ability.",
               "environment": environment_report(), "stages": results, "model": model}
    write_json(root / "summary.json", summary)
    return summary
