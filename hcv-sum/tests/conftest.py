"""Shared fixtures.

Unit tests use deterministic fake models so they test each stage's LOGIC in
milliseconds. They say nothing about the quality of the real models; that is
what the ``slow`` tests in test_end_to_end.py are for.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from hcv_sum.config import load_config
from hcv_sum.models import ModelRegistry
from hcv_sum.scoring import DocumentIndex
from hcv_sum.types import Section

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "was", "were", "be", "by", "for", "with", "at",
    "as", "it", "its", "this", "that", "from", "are", "has", "have", "had", "will", "all", "their", "our", "we",
}
NEGATIONS = {"not", "no", "never", "unrelated", "none", "nothing", "without"}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


class FakeEmbedder:
    """Hashed bag of content words (negations ignored, like real embedders mostly do)."""

    dim = 256

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = content_words(text) - NEGATIONS
            for w in words:
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            if not words:
                out[i, 0] = 1e-3
            out[i] /= np.linalg.norm(out[i])
        return out


class FakeNLI:
    """Rule-based NLI: word-subset => entailment, flipped by a negation-parity mismatch.

    ``table`` maps exact (premise, hypothesis) pairs to [contradiction, entailment, neutral].
    """

    def __init__(self, table: dict[tuple[str, str], list[float]] | None = None):
        self.table = table or {}
        self.calls: list[tuple[str, str]] = []

    def predict(self, pairs):
        self.calls.extend(pairs)
        rows = []
        for premise, hypothesis in pairs:
            if (premise, hypothesis) in self.table:
                rows.append(self.table[(premise, hypothesis)])
                continue
            p, h = content_words(premise), content_words(hypothesis)
            p_core, h_core = p - NEGATIONS, h - NEGATIONS
            negation_mismatch = bool(p & NEGATIONS) != bool(h & NEGATIONS)
            overlap = len(p_core & h_core) / max(len(h_core), 1)
            if h_core and h_core <= p_core:
                rows.append([0.9, 0.05, 0.05] if negation_mismatch else [0.02, 0.95, 0.03])
            elif overlap >= 0.6 and negation_mismatch:
                rows.append([0.85, 0.05, 0.10])
            else:
                rows.append([0.05, 0.05, 0.90])
        return np.asarray(rows, dtype=np.float32)


class FakeSummarizer:
    """Tokens = whitespace words. 'Summary' = leading words of the input, cut at a sentence end."""

    model_max_input = 1_000_000

    def __init__(self, fixed_output: str | None = None):
        self.fixed_output = fixed_output
        self.calls: list[dict] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])

    def summarize_batch(self, texts: list[str], *, min_new_tokens: int, max_new_tokens: int) -> list[str]:
        return [self.summarize(t, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens) for t in texts]

    def summarize(self, text: str, *, min_new_tokens: int, max_new_tokens: int) -> str:
        self.calls.append({"text": text, "min_new_tokens": min_new_tokens, "max_new_tokens": max_new_tokens})
        if self.fixed_output is not None:
            return self.fixed_output
        words = text.split()[:max_new_tokens]
        out = " ".join(words)
        cut = out.rfind(".")
        return out[: cut + 1] if cut > 0 else out


@pytest.fixture
def cfg():
    return load_config()


def make_cfg(*overrides: str):
    return load_config(overrides=list(overrides))


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def nli():
    return FakeNLI()


@pytest.fixture
def summarizer():
    return FakeSummarizer()


@pytest.fixture
def fake_registry(cfg, embedder, nli, summarizer):
    return ModelRegistry(cfg, embedder=embedder, nli=nli, summarizer=summarizer)


def build_sections(spec: list[tuple[str | None, list[str]]]) -> list[Section]:
    sections, offset = [], 0
    for i, (title, sentences) in enumerate(spec):
        sections.append(Section(i, title, list(sentences), offset, "heading"))
        offset += len(sentences)
    return sections


def build_index(spec: list[tuple[str | None, list[str]]], embedder: FakeEmbedder) -> DocumentIndex:
    return DocumentIndex.build(build_sections(spec), embedder)
