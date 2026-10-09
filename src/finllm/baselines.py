"""Target-aware sparse baseline fitted on training rows only."""
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import read_jsonl, text_hash, write_json
from .evaluation import compute_metrics
from .tokenizer import LABELS


def target_features(row):
    target = row["target"].lstrip("$").upper()
    text = row["text"]
    pattern = re.compile(r"\$" + re.escape(target) + r"(?![\w.])", re.I)
    text = pattern.sub(" TARGET ", text)
    for alias in sorted(row.get("target_aliases", []), key=len, reverse=True):
        if alias.casefold() != target.casefold():
            text = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", " TARGET ", text, flags=re.I)
    text = re.sub(r"\$[A-Za-z][A-Za-z0-9.]*", " OTHER_ASSET ", text)
    words = text.split()
    windows = [" ".join(words[max(0, index - 5):index + 6]) for index, word in enumerate(words) if word == "TARGET"]
    return text + "\nTARGET_CONTEXT " + " TARGET_CONTEXT ".join(windows)


def run_baseline(train_path, input_path, output_path):
    train, test = list(read_jsonl(train_path)), list(read_jsonl(input_path))
    if not train or not test:
        raise ValueError("Baseline needs nonempty train and evaluation sets")
    train_hashes = {text_hash(row["text"]) for row in train}
    if any(text_hash(row["text"]) in train_hashes for row in test):
        raise ValueError("Exact text leakage between baseline training and evaluation")
    x, y = [target_features(row) for row in train], [LABELS.index(row["label"]) for row in train]
    if len(set(y)) < 2:
        raise ValueError("Need at least two training classes")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50_000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=130)
    clf.fit(vectorizer.fit_transform(x), y)
    raw = clf.predict_proba(vectorizer.transform([target_features(row) for row in test]))
    probabilities = np.zeros((len(test), 3))
    probabilities[:, clf.classes_] = raw
    majority = Counter(y).most_common(1)[0][0]
    majority_probs = np.zeros_like(probabilities)
    majority_probs[:, majority] = 1
    result = {"tfidf": compute_metrics([row["label"] for row in test], probabilities),
              "majority": compute_metrics([row["label"] for row in test], majority_probs),
              "features": "TARGET/OTHER_ASSET, word unigrams+bigrams, +/-5 word target windows"}
    write_json(output_path, result)
    return result
