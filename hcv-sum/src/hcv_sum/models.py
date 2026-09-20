"""Model wrappers behind small interfaces.

Stages depend only on the three protocols below, so tests can inject cheap,
deterministic fakes while the real pipeline lazily loads each Hugging Face
model once and caches it.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

import numpy as np
import torch

from .config import Config

log = logging.getLogger(__name__)

# Canonical NLI column order used everywhere in the package.
CONTRADICTION, ENTAILMENT, NEUTRAL = 0, 1, 2
NLI_LABELS = ("contradiction", "entailment", "neutral")


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (n, d)."""


class NLIModel(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Return probabilities, shape (n, 3), columns [contradiction, entailment, neutral]."""


class Summarizer(Protocol):
    model_max_input: int          # hard input limit of the model; config budgets are capped to it

    def summarize_batch(self, texts: "list[str]", *, min_new_tokens: int, max_new_tokens: int) -> "list[str]": ...

    def count_tokens(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...

    def summarize(self, text: str, *, min_new_tokens: int, max_new_tokens: int) -> str: ...


class SentenceTransformerEmbedder:
    def __init__(self, name: str, device: str, batch_size: int):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.batch_size = batch_size
        self.model = SentenceTransformer(name, device=device, model_kwargs={"dtype": torch.float32})

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension() or 1), dtype=np.float32)
        return np.asarray(
            self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )


class CrossEncoderNLI:
    def __init__(self, name: str, device: str, batch_size: int):
        from sentence_transformers import CrossEncoder

        self.name = name
        self.batch_size = batch_size
        # float32 is forced: transformers>=5 loads checkpoints in their stored dtype, and fp16 weights
        # (e.g. MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) run ~6x slower on CPU.
        self.model = CrossEncoder(name, device=device, model_kwargs={"dtype": torch.float32})
        self.id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        # Map the checkpoint's own label order onto the canonical order. Never
        # hard-code it: NLI checkpoints disagree on label order.
        order = []
        for target in NLI_LABELS:
            matches = [i for i, label in self.id2label.items() if label.startswith(target)]
            if len(matches) != 1:
                raise ValueError(f"NLI model {name} has unexpected labels {self.id2label}")
            order.append(matches[0])
        self._order = order

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros((0, 3), dtype=np.float32)
        probs = self.model.predict([list(p) for p in pairs], batch_size=self.batch_size, apply_softmax=True,
                                   convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(probs, dtype=np.float32)[:, self._order]


class HFSeq2SeqSummarizer:
    def __init__(self, name: str, device: str, *, num_beams: int, no_repeat_ngram_size: int,
                 length_penalty: float):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.length_penalty = length_penalty
        # BART exposes max_position_embeddings (1024); T5 has relative positions and relies on the
        # tokenizer's model_max_length (512), beyond which quality degrades.
        limit = getattr(self.model.config, "max_position_embeddings", None) or self.tokenizer.model_max_length
        self.model_max_input = int(min(limit, 4096))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate(self, text: str, max_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_tokens:
            return text
        return self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

    def summarize(self, text: str, *, min_new_tokens: int, max_new_tokens: int) -> str:
        return self.summarize_batch([text], min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens)[0]

    def summarize_batch(self, texts: list[str], *, min_new_tokens: int, max_new_tokens: int) -> list[str]:
        """Generate for several sections in one pass.

        Padded batching roughly halves CPU time per section (measured 1.97x on 4 sections). The
        attention mask makes padding invisible to the model, so outputs match one-at-a-time
        generation; tests/test_batching.py checks that on the sample documents.
        """
        if not texts:
            return []
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True,
                                max_length=self.model_max_input)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                num_beams=self.num_beams,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                length_penalty=self.length_penalty,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
            )
        return [self.tokenizer.decode(row, skip_special_tokens=True).strip() for row in out]


class TokenCounter:
    """The summarizer's tokenizer without its weights: enough for the pre-flight estimate."""

    def __init__(self, name: str):
        from transformers import AutoConfig, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        config = AutoConfig.from_pretrained(name)
        limit = getattr(config, "max_position_embeddings", None) or self.tokenizer.model_max_length
        self.model_max_input = int(min(limit, 4096))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])


class ModelRegistry:
    """Lazily loads and caches the three models. Pass instances to override them (tests)."""

    def __init__(self, cfg: Config, *, embedder: Embedder | None = None, nli: NLIModel | None = None,
                 summarizer: Summarizer | None = None, merge_summarizer: Summarizer | None = None):
        self.cfg = cfg
        self._embedder, self._nli, self._summarizer = embedder, nli, summarizer
        self._merge_summarizer = merge_summarizer
        if cfg.models.torch_threads > 0:
            torch.set_num_threads(cfg.models.torch_threads)

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            log.info("loading embedder %s", self.cfg.models.embedder)
            m = self.cfg.models
            self._embedder = SentenceTransformerEmbedder(m.embedder, m.device, m.batch_size)
        return self._embedder

    @property
    def nli(self) -> NLIModel:
        if self._nli is None:
            log.info("loading NLI model %s", self.cfg.models.nli)
            m = self.cfg.models
            self._nli = CrossEncoderNLI(m.nli, m.device, m.batch_size)
        return self._nli

    @property
    def summarizer(self) -> Summarizer:
        if self._summarizer is None:
            log.info("loading summarizer %s", self.cfg.models.summarizer)
            s = self.cfg.summarization
            self._summarizer = HFSeq2SeqSummarizer(
                self.cfg.models.summarizer, self.cfg.models.device, num_beams=s.num_beams,
                no_repeat_ngram_size=s.no_repeat_ngram_size, length_penalty=s.length_penalty)
        return self._summarizer

    @property
    def merge_summarizer(self) -> Summarizer:
        """Stage 4's model: models.merge_summarizer, or the Stage 2 summarizer when that is empty/the same."""
        if self._merge_summarizer is not None:
            return self._merge_summarizer
        name = self.cfg.models.merge_summarizer
        if not name or name == self.cfg.models.summarizer:
            return self.summarizer
        log.info("loading merge summarizer %s", name)
        s = self.cfg.summarization
        self._merge_summarizer = HFSeq2SeqSummarizer(name, self.cfg.models.device, num_beams=s.num_beams,
                                                     no_repeat_ngram_size=s.no_repeat_ngram_size,
                                                     length_penalty=s.length_penalty)
        return self._merge_summarizer


class CountingEmbedder:
    """Wraps an embedder and counts how much work it was asked to do (run instrumentation)."""

    def __init__(self, inner: Embedder):
        self.inner, self.calls, self.texts = inner, 0, 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        self.texts += len(texts)
        return self.inner.encode(texts)


class CountingNLI:
    """Wraps an NLI model and counts sentence-pair scorings -- the pipeline's dominant CPU cost."""

    def __init__(self, inner: NLIModel):
        self.inner, self.calls, self.pairs = inner, 0, 0

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        self.calls += 1
        self.pairs += len(pairs)
        return self.inner.predict(pairs)


class CountingSummarizer:
    """Wraps a summarizer and counts generation calls (the other dominant cost)."""

    def __init__(self, inner: Summarizer):
        self.inner, self.generations = inner, 0
        self.model_max_input = inner.model_max_input

    def count_tokens(self, text: str) -> int:
        return self.inner.count_tokens(text)

    def truncate(self, text: str, max_tokens: int) -> str:
        return self.inner.truncate(text, max_tokens)

    def summarize(self, text: str, *, min_new_tokens: int, max_new_tokens: int) -> str:
        self.generations += 1
        return self.inner.summarize(text, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens)

    def summarize_batch(self, texts: "list[str]", *, min_new_tokens: int, max_new_tokens: int) -> "list[str]":
        self.generations += len(texts)
        return self.inner.summarize_batch(texts, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens)
