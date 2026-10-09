import pytest

from finllm.tokenizer import (
    BOS_TOKEN, LABELS, LABEL_TOKENS, SPECIAL_TOKENS, encode_prompt,
    label_token_ids, load_tokenizer, make_prompt, tokenizer_hash, train_tokenizer,
)


@pytest.fixture
def tokenizer(tmp_path):
    texts = [
        "$AAPL reports revenue of $23.45B, up 5.2%; I'm not bearish.",
        "Long $MSFT, short $TSLA. Cash-flow: +10%, debt/equity 0.4.",
        "Neutral: earnings arrive on Thursday; no position.",
    ] * 5
    return train_tokenizer(iter(texts), tmp_path, vocab_size=384)


def test_financial_punctuation_and_unicode_roundtrip(tokenizer):
    text = "  $BRK.B isn't cheap: -2.5%, $1,250.00; € and £.\nCafé — 好。"
    ids = tokenizer.encode(text).ids
    assert tokenizer.decode(ids) == text


def test_special_labels_are_atomic_distinct_and_reloadable(tokenizer, tmp_path):
    ids = label_token_ids(tokenizer)
    assert tuple(ids) == LABELS
    assert len(set(ids.values())) == 3
    for label, token_id in ids.items():
        assert tokenizer.encode(LABEL_TOKENS[label]).ids == [token_id]
    reloaded = load_tokenizer(tmp_path)
    assert label_token_ids(reloaded) == ids
    assert len(tokenizer_hash(tmp_path)) == 64
    assert reloaded.get_vocab_size() <= 384
    for token in SPECIAL_TOKENS:
        assert reloaded.token_to_id(token) is not None


def test_prompt_has_bos_and_does_not_silently_truncate(tokenizer):
    ids = encode_prompt(tokenizer, "Long $AAPL.", "aapl", 512)
    assert ids[0] == tokenizer.token_to_id(BOS_TOKEN)
    assert tokenizer.decode(ids[1:]) == make_prompt("Long $AAPL.", "AAPL")
    assert encode_prompt(tokenizer, "Long $AAPL.", "AAPL", len(ids)) == ids
    with pytest.raises(ValueError, match="no truncation"):
        encode_prompt(tokenizer, "Long $AAPL.", "AAPL", len(ids) - 1)
    with pytest.raises(ValueError, match="reserved"):
        encode_prompt(tokenizer, "$AAPL <|bullish|>", "AAPL", 512)


def test_empty_corpus_and_insufficient_byte_vocabulary_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        train_tokenizer(iter(["", " "]), tmp_path)
    with pytest.raises(ValueError, match="256"):
        train_tokenizer(iter(["hello"]), tmp_path, vocab_size=64)
