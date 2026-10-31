"""Train and load the project's own reversible byte-level BPE tokenizer."""

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
LABELS = ("bearish", "neutral", "bullish")
LABEL_TOKENS = {label: f"<|{label}|>" for label in LABELS}
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, *LABEL_TOKENS.values()]


def _tokenizer_path(path) -> Path:
    path = Path(path)
    return path / "tokenizer.json" if path.is_dir() else path


def tokenizer_hash(path) -> str:
    return hashlib.sha256(_tokenizer_path(path).read_bytes()).hexdigest()


def _validate_special_tokens(tokenizer: Tokenizer):
    ids = []
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is None or tokenizer.encode(token, add_special_tokens=False).ids != [token_id]:
            raise ValueError(f"Missing or non-atomic special token: {token}")
        ids.append(token_id)
    if len(set(ids)) != len(ids):
        raise ValueError("Special token IDs must be distinct")


def train_tokenizer(texts: Iterable[str], output_dir, vocab_size: int = 8192) -> Tokenizer:
    """Consume a training-only iterator; no corpus or pretrained files are fetched.

    Iteration bounds input buffering, but BPE's frequency tables still use RAM.
    A small smoke corpus may produce fewer tokens than the requested vocabulary.
    """
    if not isinstance(vocab_size, int) or vocab_size < 256 + len(SPECIAL_TOKENS):
        raise ValueError("vocab_size must fit all 256 byte symbols and special tokens")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2, show_progress=False,
    )
    count = 0

    def checked_texts():
        nonlocal count
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Tokenizer training requires an iterator of strings")
            if text.strip():
                count += 1
                yield text

    tokenizer.train_from_iterator(checked_texts(), trainer=trainer)
    if count == 0:
        raise ValueError("No nonempty training texts were provided")
    _validate_special_tokens(tokenizer)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    metadata = {
        "format_version": 1,
        "algorithm": "ByteLevelBPE",
        "normalizer": None,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "training_documents": count,
        "special_token_ids": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        "sha256": tokenizer_hash(output_dir),
    }
    (output_dir / "tokenizer_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return tokenizer


def load_tokenizer(path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(_tokenizer_path(path)))
    # Persisted HF settings must not silently truncate or pad our task prompts.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    _validate_special_tokens(tokenizer)
    return tokenizer


def label_token_ids(tokenizer: Tokenizer) -> dict[str, int]:
    _validate_special_tokens(tokenizer)
    return {label: tokenizer.token_to_id(token) for label, token in LABEL_TOKENS.items()}


def is_literal_target(target: object) -> bool:
    """Accept a bounded, single-line English literal entity, not an asset map.

    Only ordinary spaces are accepted inside names; this prevents newlines or
    other control characters from introducing additional prompt fields.
    """
    return (isinstance(target, str) and 1 <= len(target) <= 120
            and target == target.strip() and "  " not in target
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 .,'’&()/+-]*", target) is not None)


def make_prompt(text: str, target: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty string")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a nonempty string")
    if any(token in text or token in target for token in SPECIAL_TOKENS):
        raise ValueError("Input contains a reserved special token")
    target = target.strip().removeprefix("$")
    if not is_literal_target(target):
        raise ValueError("target must be a bounded single-line asset identifier or literal entity")
    target = target.upper()
    return (
        "Classify the investment stance toward the target asset.\n"
        "Choose bearish, neutral, or bullish using only the text.\n"
        f"Target: {target}\n"
        f"Text: {json.dumps(text, ensure_ascii=False)}\n"
        "Stance:"
    )


def encode_prompt(tokenizer: Tokenizer, text: str, target: str, max_length: int) -> list[int]:
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError("Tokenizer padding/truncation must be disabled for task prompts")
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    if bos_id is None:
        raise ValueError("Tokenizer is missing the BOS token")
    ids = [bos_id] + tokenizer.encode(make_prompt(text, target), add_special_tokens=False).ids
    if len(ids) > max_length:
        raise ValueError(f"Prompt has {len(ids)} tokens, exceeding max_length={max_length}; no truncation applied")
    return ids
