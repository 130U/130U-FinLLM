"""Guarded target-conditioned inference with deterministic JSON formatting."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import torch

from .evaluation import (LABELS, load_records, probabilities_from_logits, sha256_file,
                         tokenizer_file, validate_calibration)
from .tokenizer import is_literal_target


def validate_input(record, target_mode="ticker"):
    if target_mode not in ("ticker", "literal_entity"):
        raise ValueError("Unknown target mode")
    text, target = record.get("text"), record.get("target")
    if not isinstance(text, str) or not text.strip():
        return "invalid_text"
    if target_mode == "ticker":
        valid_target = isinstance(target, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,14}", target)
    else:
        valid_target = is_literal_target(target)
    if not valid_target:
        return "invalid_target"
    language = record.get("language")
    if language is not None and str(language).lower() not in ("en", "eng", "english"):
        return "unsupported_language"
    # This rejects clearly unsupported scripts, not a complete language detector.
    letters = [c for c in text if c.isalpha()]
    if letters and sum("LATIN" not in unicodedata.name(c, "") for c in letters) / len(letters) > 0.25:
        return "unsupported_script"
    if target_mode == "literal_entity":
        # Research pilot only: match the actual textual entity, not a security ID.
        literal = r"(?<!\w)" + re.escape(target) + r"(?!\w)"
        return None if re.search(literal, text, flags=re.IGNORECASE) else "target_not_supported"
    cashtag = r"(?<![\w$])\$" + re.escape(target) + r"(?!\w|[.-]\w)"
    if re.search(cashtag, text, flags=re.IGNORECASE):
        return None
    aliases = record.get("target_aliases", [])
    if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
        return "invalid_target_aliases"
    for alias in aliases:
        # Aliases must be supplied from the user's audited asset map.
        # Bare ticker aliases are deliberately excluded (e.g. ordinary word CAT).
        if len(alias.strip()) >= 2 and alias.casefold() != target.casefold():
            if re.search(r"(?<!\w)" + re.escape(alias.strip()) + r"(?!\w)", text, re.IGNORECASE):
                return None
    return "target_not_supported"


class Predictor:
    def __init__(self, checkpoint, tokenizer_path, calibration_path=None, device="auto", target_mode="ticker"):
        from .checkpoint import load_checkpoint
        from .model import FinLLM, ModelConfig
        from .tokenizer import encode_prompt, label_token_ids, load_tokenizer
        if target_mode not in ("ticker", "literal_entity"):
            raise ValueError("Unknown target mode")
        self.target_mode = target_mode
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        payload = load_checkpoint(checkpoint, map_location="cpu", allow_previous=False)
        recorded_hash = payload.get("tokenizer_hash") or payload.get("provenance", {}).get("tokenizer_sha256")
        actual_hash = sha256_file(tokenizer_file(tokenizer_path))
        if recorded_hash and recorded_hash != actual_hash:
            raise ValueError("Checkpoint tokenizer hash does not match the supplied tokenizer.")
        self.model = FinLLM(ModelConfig(**payload["model_config"]))
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.to(self.device).eval()
        self.max_length = payload["model_config"].get("max_position_embeddings", 512)
        self.tokenizer = load_tokenizer(tokenizer_path)
        ids = label_token_ids(self.tokenizer)
        self.label_ids = [ids[label] for label in LABELS]
        self.encode_prompt = encode_prompt
        self.temperature, self.review_threshold, self.calibrated = 1.0, 0.65, False
        if calibration_path:
            calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
            validate_calibration(calibration, checkpoint, tokenizer_path, target_mode=target_mode)
            self.temperature = float(calibration["temperature"])
            self.review_threshold = float(calibration["review_threshold"])
            self.calibrated = True

    def class_logits(self, record):
        reason = validate_input(record, target_mode=self.target_mode)
        if reason:
            return None, reason
        try:
            ids = self.encode_prompt(self.tokenizer, record["text"], record["target"], self.max_length)
        except ValueError as error:
            return None, "invalid_or_overlength_prompt: " + str(error)
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(tensor)[0, -1, self.label_ids].float().cpu().numpy()
        return logits, None

    def predict(self, record):
        logits, reason = self.class_logits(record)
        result = {"text": record.get("text"), "target": record.get("target"),
                  "label": None, "probabilities": None, "sentiment_score": None,
                  "confidence": None, "needs_review": True, "calibrated": self.calibrated,
                  "target_mode": self.target_mode}
        for field in ("id", "group_id"):
            if field in record:
                result[field] = record[field]
        if reason:
            return {**result, "status": reason.split(":")[0], "review_reason": reason}
        probs = probabilities_from_logits([logits], self.temperature)[0]
        confidence = float(probs.max())
        needs_review = confidence < self.review_threshold
        return {**result, "status": "ok", "label": LABELS[int(probs.argmax())],
                "probabilities": dict(zip(LABELS, map(float, probs))),
                "sentiment_score": float(100 * (probs[2] - probs[0])),
                "confidence": confidence, "needs_review": needs_review,
                "review_reason": "low_confidence" if needs_review else None}


def run_prediction(checkpoint, tokenizer_path, input_path, output_path, calibration_path=None, device="auto", target_mode="ticker"):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Prediction output must not overwrite its input file.")
    predictor = Predictor(checkpoint, tokenizer_path, calibration_path, device, target_mode=target_mode)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = rejected = 0
    # Stream output and inputs; a large prediction file need not occupy RAM.
    with Path(input_path).open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as destination:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Prediction inputs must be JSON objects.")
            result = predictor.predict(record)
            destination.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
            rejected += result["status"] != "ok"
    return {"count": count, "rejected_count": rejected, "output": str(output_path), "calibrated": predictor.calibrated,
            "target_mode": target_mode}
