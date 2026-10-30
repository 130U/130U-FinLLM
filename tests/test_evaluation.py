import json

import numpy as np
import pytest

from finllm.evaluation import (assert_disjoint_calibration, compute_metrics, fit_temperature,
                               probabilities_from_logits, select_review_threshold,
                               sha256_file, validate_calibration)
from finllm.inference import validate_input


def test_known_confusion_and_metrics():
    truth = [0, 0, 1, 2]
    probs = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    metrics = compute_metrics(truth, probs)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx(7 / 9)
    assert metrics["brier_score"] == pytest.approx((0.06 + 1.14 + 0.06 + 0.06) / 4)
    assert metrics["coverage_curve"][-1]["accuracy"] is None


def test_softmax_is_stable_and_score_has_semantic_bounds():
    probs = probabilities_from_logits([[10000, 10001, 10002]])
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1)
    assert 0 < 100 * (probs[0, 2] - probs[0, 0]) < 100
    with pytest.raises(ValueError):
        probabilities_from_logits([[1, 2, 3]], 0)


def test_temperature_improves_overconfident_errors_without_changing_argmax():
    logits = np.array([[12, 0, 0], [12, 0, 0], [0, 12, 0], [0, 0, 12]])
    truth = [0, 1, 1, 2]
    fitted = fit_temperature(logits, truth)
    assert fitted["temperature"] > 1
    assert fitted["nll_after"] < fitted["nll_before"]
    assert np.array_equal(logits.argmax(axis=1), probabilities_from_logits(logits, fitted["temperature"]).argmax(axis=1))


def test_target_link_is_not_ordinary_english_ticker_word():
    assert validate_input({"text": "The cat is asleep.", "target": "CAT"}) == "target_not_supported"
    assert validate_input({"text": "Long $CAT.", "target": "CAT"}) is None
    assert validate_input({"text": "$CATER is falling.", "target": "CAT"}) == "target_not_supported"
    assert validate_input({"text": "Caterpillar demand is weak.", "target": "CAT", "target_aliases": ["Caterpillar"]}) is None
    assert validate_input({"text": "$CAT 显著上涨，仍然看涨。", "target": "CAT"}) == "unsupported_script"


def test_explicit_literal_entity_mode_matches_complete_entity_only():
    record = {"text": "Infosys raises revenue guidance.", "target": "INFOSYS"}
    assert validate_input(record) == "target_not_supported"
    assert validate_input(record, target_mode="literal_entity") is None
    assert validate_input({"text": "Infosys raises guidance", "target": "INFO"}, target_mode="literal_entity") == "target_not_supported"
    assert validate_input({"text": "Reliance Industries rises", "target": "Reliance Industries"}, target_mode="literal_entity") == "invalid_target"
    with pytest.raises(ValueError, match="Unknown target mode"):
        validate_input(record, target_mode="unrecognized")


def test_calibration_partitions_cannot_overlap_by_text_or_event():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_calibration([{"text": "Long $AAPL", "target": "AAPL"}],
                                    [{"text": "long   $aapl", "target": "MSFT"}])
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_calibration([{"text": "one", "group_id": "shared"}],
                                    [{"text": "two", "group_id": "shared"}])
    assert_disjoint_calibration([{"text": "one", "group_id": "a"}], [{"text": "two", "group_id": "b"}])


def test_wrong_checkpoint_calibration_is_rejected(tmp_path):
    checkpoint, tokenizer = tmp_path / "model.pt", tmp_path / "tokenizer.json"
    checkpoint.write_bytes(b"model-v1")
    tokenizer.write_text("{}")
    calibration = {"label_order": ["bearish", "neutral", "bullish"], "temperature": 1.2,
                   "review_threshold": 0.65, "bindings": {"checkpoint_sha256": sha256_file(checkpoint),
                   "tokenizer_sha256": sha256_file(tokenizer)}}
    validate_calibration(calibration, checkpoint, tokenizer)
    with pytest.raises(ValueError, match="different target mode"):
        validate_calibration(calibration, checkpoint, tokenizer, target_mode="literal_entity")
    calibration["target_mode"] = "literal_entity"
    validate_calibration(calibration, checkpoint, tokenizer, target_mode="literal_entity")
    with pytest.raises(ValueError, match="different target mode"):
        validate_calibration(calibration, checkpoint, tokenizer)
    calibration["target_mode"] = "ticker"
    checkpoint.write_bytes(b"model-v2")
    with pytest.raises(ValueError, match="different checkpoint"):
        validate_calibration(calibration, checkpoint, tokenizer)


def test_small_threshold_sample_is_not_overfit():
    result = select_review_threshold([0, 1, 2], np.eye(3))
    assert result["threshold"] == 0.65
    assert result["method"] == "fixed_small_sample_fallback"


def test_invalid_probabilities_fail_closed():
    with pytest.raises(ValueError):
        compute_metrics([0], [[0.2, 0.2, 0.2]])


def test_real_checkpoint_inference_rejects_unsupported_and_overlength(tmp_path):
    from dataclasses import asdict
    from finllm.checkpoint import save_checkpoint
    from finllm.inference import Predictor
    from finllm.model import FinLLM, ModelConfig
    from finllm.tokenizer import make_prompt, train_tokenizer

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer = train_tokenizer([make_prompt("Long $AAPL after strong demand.", "AAPL")] * 5,
                                tokenizer_dir, vocab_size=384)
    config = ModelConfig(vocab_size=tokenizer.get_vocab_size(), hidden_size=32, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
                         max_position_embeddings=256)
    checkpoint = tmp_path / "tiny.pt"
    save_checkpoint(checkpoint, {"model_config": asdict(config), "model_state": FinLLM(config).state_dict(),
                                 "tokenizer_hash": sha256_file(tokenizer_dir / "tokenizer.json")})
    predictor = Predictor(checkpoint, tokenizer_dir, device="cpu")
    predicted = predictor.predict({"text": "Long $AAPL after strong demand.", "target": "AAPL"})
    assert predicted["status"] == "ok"
    assert sum(predicted["probabilities"].values()) == pytest.approx(1)
    assert predicted["sentiment_score"] == pytest.approx(100 * (predicted["probabilities"]["bullish"] - predicted["probabilities"]["bearish"]))
    unsupported = predictor.predict({"text": "The cat is asleep.", "target": "CAT"})
    assert unsupported["label"] is None and unsupported["sentiment_score"] is None
    overlength = predictor.predict({"text": "$AAPL " + "unusually " * 500, "target": "AAPL"})
    assert overlength["status"] == "invalid_or_overlength_prompt"
    assert overlength["sentiment_score"] is None
    entity_predictor = Predictor(checkpoint, tokenizer_dir, device="cpu", target_mode="literal_entity")
    entity_result = entity_predictor.predict({"text": "Apple expects strong demand.", "target": "APPLE"})
    assert entity_result["status"] == "ok"
    assert sum(entity_result["probabilities"].values()) == pytest.approx(1)
    assert entity_result["sentiment_score"] == pytest.approx(100 * (entity_result["probabilities"]["bullish"] - entity_result["probabilities"]["bearish"]))
    assert entity_predictor.predict({"text": "Pineapple expects strong demand.", "target": "APPLE"})["status"] == "target_not_supported"

    from finllm.evaluation import run_calibration, run_evaluation
    rows = [{"text": f"Apple trial headline {i}", "target": "APPLE", "label": label,
             "split": "calibration", "group_id": f"group-{i}"}
            for i, label in enumerate(("bearish", "neutral", "bullish", "bearish", "neutral", "bullish"))]
    temperature, threshold, development = (tmp_path / x for x in ("temp.jsonl", "threshold.jsonl", "dev.jsonl"))
    temperature.write_text("".join(json.dumps(r) + "\n" for r in rows[:3]), encoding="utf-8")
    threshold.write_text("".join(json.dumps(r) + "\n" for r in rows[3:]), encoding="utf-8")
    development.write_text(json.dumps({"text": "Apple rises", "target": "APPLE", "label": "bullish"}) + "\n", encoding="utf-8")
    calibration_path = tmp_path / "calibration.json"
    calibration = run_calibration(checkpoint, tokenizer_dir, temperature, threshold, calibration_path,
                                  device="cpu", target_mode="literal_entity")
    assert calibration["target_mode"] == "literal_entity"
    result = run_evaluation(checkpoint, tokenizer_dir, development, tmp_path / "evaluation.json",
                            calibration_path=calibration_path, device="cpu", target_mode="literal_entity")
    assert result["valid_count"] == 1 and result["rejected_count"] == 0
    with pytest.raises(ValueError, match="different target mode"):
        Predictor(checkpoint, tokenizer_dir, calibration_path, "cpu")
