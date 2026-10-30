"""Held-out calibration and three-class evaluation (never a returns forecast)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

LABELS = ("bearish", "neutral", "bullish")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_file(path):
    path = Path(path)
    return path / "tokenizer.json" if path.is_dir() else path


def probabilities_from_logits(logits, temperature=1.0):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite logits with shape (n, 3).")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    scaled = values / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _labels(labels):
    result = np.asarray([LABELS.index(x) if isinstance(x, str) else x for x in labels])
    if result.ndim != 1 or result.size == 0 or not np.isin(result, [0, 1, 2]).all():
        raise ValueError("Expected nonempty labels in bearish, neutral, bullish.")
    return result.astype(np.int64)


def _validate(labels, probabilities):
    truth = _labels(labels)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (len(truth), 3) or not np.isfinite(probs).all():
        raise ValueError("Probability shape or values are invalid.")
    if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1), 1):
        raise ValueError("Class probabilities must be nonnegative and sum to one.")
    return truth, probs


def compute_metrics(labels, probabilities, bins=10):
    """Macro-F1 always includes all three task classes, including absent classes."""
    truth, probs = _validate(labels, probabilities)
    prediction = probs.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    per_class = {}
    for i, name in enumerate(LABELS):
        tp = int(confusion[i, i])
        precision = tp / int(confusion[:, i].sum()) if confusion[:, i].sum() else 0.0
        recall = tp / int(confusion[i].sum()) if confusion[i].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = dict(precision=precision, recall=recall, f1=f1,
                               support=int(confusion[i].sum()))
    confidence = probs.max(axis=1)
    correct = prediction == truth
    calibration_bins = []
    ece = 0.0
    for i in range(bins):
        lower, upper = i / bins, (i + 1) / bins
        selected = (confidence >= lower) & ((confidence <= upper) if i == bins - 1 else (confidence < upper))
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        mean_confidence = float(confidence[selected].mean()) if count else None
        if count:
            ece += count / len(truth) * abs(accuracy - mean_confidence)
        calibration_bins.append(dict(lower=lower, upper=upper, count=count,
                                     accuracy=accuracy, mean_confidence=mean_confidence))
    return {
        "count": len(truth), "label_order": list(LABELS),
        "accuracy": float(correct.mean()),
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
        "per_class": per_class, "confusion_matrix": confusion.tolist(),
        "confusion_matrix_axes": "rows=true, columns=predicted",
        "nll": float(-np.log(np.clip(probs[np.arange(len(truth)), truth], 1e-15, 1)).mean()),
        "brier_score": float(((probs - np.eye(3)[truth]) ** 2).sum(axis=1).mean()),
        "brier_definition": "mean sum of squared errors over all three classes",
        "ece": float(ece), "ece_bins": calibration_bins,
        "coverage_curve": coverage_curve(truth, probs),
    }


def coverage_curve(labels, probabilities, thresholds=None):
    truth, probs = _validate(labels, probabilities)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == truth
    result = []
    for threshold in thresholds or [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]:
        accepted = confidence >= threshold
        count = int(accepted.sum())
        result.append(dict(threshold=float(threshold), accepted=count,
                           coverage=count / len(truth),
                           accuracy=float(correct[accepted].mean()) if count else None))
    return result


def fit_temperature(logits, labels):
    """Bounded scalar NLL minimization; T=1 is explicitly included as a candidate."""
    truth = _labels(labels)
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (len(truth), 3):
        raise ValueError("Logits and labels have different lengths.")

    def loss(log_temperature):
        scaled = values / math.exp(log_temperature)
        maximum = scaled.max(axis=1)
        logsumexp = maximum + np.log(np.exp(scaled - maximum[:, None]).sum(axis=1))
        return float((logsumexp - scaled[np.arange(len(truth)), truth]).mean())

    probabilities_from_logits(values)
    grid = np.linspace(math.log(0.05), math.log(20.0), 161)
    losses = np.array([loss(x) for x in grid])
    best = int(losses.argmin())
    left, right = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
    for _ in range(80):
        first, second = left + (right - left) / 3, right - (right - left) / 3
        if loss(first) < loss(second):
            right = second
        else:
            left = first
    candidates = [0.0, float(grid[best]), (left + right) / 2]
    selected = min(candidates, key=loss)
    return {"temperature": math.exp(selected), "nll_before": loss(0.0),
            "nll_after": loss(selected), "bounds": [0.05, 20.0], "count": len(truth)}


def select_review_threshold(labels, probabilities, target_accuracy=0.8, min_accepted=5):
    """Maximize held-out coverage subject to an empirical accuracy target.

    This is a tuning rule, not a population accuracy guarantee or confidence bound.
    With fewer than 30 examples use the predeclared 0.65 fallback.
    """
    truth, probs = _validate(labels, probabilities)
    if len(truth) < 30:
        return {"threshold": 0.65, "method": "fixed_small_sample_fallback",
                "count": len(truth), "warning": "Too few threshold examples; threshold was not fitted."}
    candidates = coverage_curve(truth, probs, sorted(set(probs.max(axis=1).tolist())) + [1.01])
    eligible = [x for x in candidates if x["accepted"] >= min_accepted and x["accuracy"] >= target_accuracy]
    choice = max(eligible, key=lambda x: x["coverage"]) if eligible else candidates[-1]
    return {**choice, "method": "held_out_empirical_accuracy", "target_accuracy": target_accuracy,
            "min_accepted": min_accepted, "count": len(truth),
            "warning": "Threshold-set accuracy is not a future accuracy guarantee."}


def load_records(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}.")
                records.append(value)
    if not records:
        raise ValueError(f"No records in {path}.")
    return records


def assert_disjoint_calibration(temperature_records, threshold_records):
    """Check both declared groups and target-independent normalized text hashes."""
    def keys(records):
        result = set()
        for row in records:
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Calibration records need nonempty text.")
            result.add(("text", hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()))
            for field in ("group_id", "duplicate_group", "event_id", "source_id"):
                if row.get(field) is not None:
                    result.add((field, str(row[field])))
        return result
    if keys(temperature_records) & keys(threshold_records):
        raise ValueError("Temperature and threshold calibration partitions overlap by text/group.")


def validate_calibration(calibration, checkpoint_path, tokenizer_path, target_mode="ticker"):
    if target_mode not in ("ticker", "literal_entity"):
        raise ValueError("Unknown target mode")
    if calibration.get("target_mode", "ticker") != target_mode:
        raise ValueError("Calibration belongs to a different target mode.")
    binding = calibration.get("bindings", {})
    if binding.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("Calibration belongs to a different checkpoint; recalibrate the final model.")
    if binding.get("tokenizer_sha256") != sha256_file(tokenizer_file(tokenizer_path)):
        raise ValueError("Calibration belongs to a different tokenizer.")
    if calibration.get("label_order") != list(LABELS):
        raise ValueError("Calibration label order does not match the model task.")
    temperature = calibration.get("temperature", 0)
    threshold = calibration.get("review_threshold", -1)
    if not isinstance(temperature, (float, int)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid calibration temperature.")
    if not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 <= threshold <= 1.01:
        raise ValueError("Invalid review threshold.")


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _collect(predictor, records):
    logits, labels, invalid = [], [], []
    for index, row in enumerate(records):
        if row.get("label") not in LABELS:
            raise ValueError(f"Record {index} has no valid gold label.")
        values, reason = predictor.class_logits(row)
        if values is None:
            invalid.append({"index": index, "id": row.get("id"), "reason": reason})
        else:
            logits.append(values)
            labels.append(row["label"])
    if not logits:
        raise ValueError("No valid target-linked English inputs remain for evaluation.")
    return np.asarray(logits), labels, invalid


def run_calibration(checkpoint, tokenizer_path, temperature_data, threshold_data, output_path, device="auto", target_mode="ticker"):
    from .inference import Predictor
    first, second = load_records(temperature_data), load_records(threshold_data)
    for name, records in (("temperature", first), ("threshold", second)):
        allowed = {"calibration", f"calibration_{name}"}
        if any(row.get("split") is not None and row["split"] not in allowed for row in records):
            raise ValueError(f"The {name} calibration file contains records from another declared split.")
    assert_disjoint_calibration(first, second)
    predictor = Predictor(checkpoint, tokenizer_path, device=device, target_mode=target_mode)
    first_logits, first_labels, first_invalid = _collect(predictor, first)
    second_logits, second_labels, second_invalid = _collect(predictor, second)
    # Silently fitting on a selected subset would hide input/schema problems.
    if first_invalid or second_invalid:
        raise ValueError(f"Repair rejected calibration records before fitting: temperature={first_invalid[:5]}, threshold={second_invalid[:5]}.")
    fitted = fit_temperature(first_logits, first_labels)
    calibrated = probabilities_from_logits(second_logits, fitted["temperature"])
    threshold = select_review_threshold(second_labels, calibrated)
    result = {"format_version": 1, "label_order": list(LABELS), "target_mode": target_mode,
              "temperature": fitted["temperature"], "review_threshold": threshold["threshold"],
              "temperature_fit": fitted, "threshold_fit": threshold,
              "bindings": {"checkpoint_sha256": sha256_file(checkpoint),
                           "tokenizer_sha256": sha256_file(tokenizer_file(tokenizer_path)),
                           "temperature_data_sha256": sha256_file(temperature_data),
                           "threshold_data_sha256": sha256_file(threshold_data)},
              "note": "Use only after selecting the final SFT or DPO checkpoint on development data."}
    _write_json(output_path, result)
    return result


def run_evaluation(checkpoint, tokenizer_path, input_path, output_path, calibration_path=None, device="auto", target_mode="ticker"):
    from .inference import Predictor
    if calibration_path:
        calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        bindings = calibration.get("bindings", {})
        input_hash = sha256_file(input_path)
        if input_hash in (bindings.get("temperature_data_sha256"), bindings.get("threshold_data_sha256")):
            raise ValueError("Use an independent evaluation file, not a calibration fitting partition.")
    predictor = Predictor(checkpoint, tokenizer_path, calibration_path, device, target_mode=target_mode)
    records = load_records(input_path)
    logits, labels, invalid = _collect(predictor, records)
    probs = probabilities_from_logits(logits, predictor.temperature)
    result = {"format_version": 1, "target_mode": target_mode, "input_count": len(records), "valid_count": len(labels),
              "rejected_count": len(invalid), "rejected_records": invalid,
              "input_acceptance_coverage": len(labels) / len(records),
              "metrics": compute_metrics(labels, probs),
              "uncalibrated_metrics": compute_metrics(labels, probabilities_from_logits(logits)),
              "review_threshold": predictor.review_threshold,
              "review_policy": coverage_curve(labels, probs, [predictor.review_threshold])[0],
              "calibrated": predictor.calibrated,
              "bindings": {"checkpoint_sha256": sha256_file(checkpoint),
                           "tokenizer_sha256": sha256_file(tokenizer_file(tokenizer_path)),
                           "evaluation_data_sha256": sha256_file(input_path)},
              "note": "Metrics are on accepted inputs; report input coverage alongside them. Scores are stance, not returns."}
    _write_json(output_path, result)
    return result
