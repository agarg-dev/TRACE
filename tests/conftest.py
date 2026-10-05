"""Shared fixtures and helpers for the TRACE test suite.

The repository is laid out as a flat source tree that is normally put on the path with
``PYTHONPATH=src`` (see ``scripts/*.sh``). ``pyproject.toml`` configures the same for pytest,
and this module makes the suite importable even when pytest is invoked without it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeTokenizer:
    """Minimal tokenizer stand-in: one character per token, plus a space token.

    ``decode`` accepts the list form used throughout TRACE and the single-token list form
    used by ``_starts_word``, so word-boundary handling can be exercised deterministically.
    """

    def __init__(self, vocabulary=None):
        self.vocabulary = vocabulary if vocabulary is not None else {}

    def decode(self, token_ids):
        return "".join(self.vocabulary.get(int(token_id), chr(int(token_id))) for token_id in token_ids)


class WordTokenizer(FakeTokenizer):
    """Tokenizes whitespace-separated words, one token per word (leading space included)."""

    def __init__(self, words):
        vocabulary = {index: f" {word}" for index, word in enumerate(words)}
        super().__init__(vocabulary)


@pytest.fixture
def word_tokenizer():
    return WordTokenizer(["the", "cat", "sat"])


def make_sequence(idx, label, length, dimension=4, seed=0, token_ids=None):
    """Build a response record shaped like the activation cache output."""
    generator = np.random.RandomState(seed)
    sequence = {
        "idx": idx,
        "label": label,
        "x": generator.standard_normal((length, dimension)).astype(np.float32),
    }
    if token_ids is not None:
        sequence["token_ids"] = token_ids
    return sequence
