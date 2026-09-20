"""Shared retrieval + NLI scoring used by Stage 3 (resolution) and Stage 5 (provenance).

Both stages ask the same question: "how strongly is this claim entailed by some
part of the source?" They answer it identically: embedding similarity picks
candidate source sentences, then NLI is run against each candidate AND against
short windows of consecutive sentences around it, because a summary sentence
often fuses facts from two or three adjacent source sentences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .models import CONTRADICTION, ENTAILMENT, Embedder, NLIModel
from .types import Section


@dataclass
class DocumentIndex:
    """All source sentences of a document with their section ids and embeddings."""

    sections: list[Section]
    sentences: list[str]
    section_of: np.ndarray
    embeddings: np.ndarray

    @classmethod
    def build(cls, sections: list[Section], embedder: Embedder) -> "DocumentIndex":
        sentences = [s for sec in sections for s in sec.sentences]
        section_of = np.array([sec.index for sec in sections for _ in sec.sentences], dtype=int)
        return cls(sections, sentences, section_of, embedder.encode(sentences))

    def section_bounds(self, sentence_id: int) -> tuple[int, int]:
        sec = self.sections[int(self.section_of[sentence_id])]
        return sec.sentence_offset, sec.sentence_offset + len(sec.sentences)


@dataclass
class ScoredPremise:
    sentence_ids: tuple[int, ...]
    text: str
    similarity: float          # similarity of the best-matching sentence inside the premise
    entailment: float
    contradiction: float


def top_candidates(query_emb: np.ndarray, index: DocumentIndex, pool: Sequence[int] | None,
                   top_k: int, min_similarity: float) -> list[tuple[int, float]]:
    ids = np.arange(len(index.sentences)) if pool is None else np.asarray(list(pool), dtype=int)
    if ids.size == 0:
        return []
    sims = index.embeddings[ids] @ query_emb
    order = np.argsort(-sims)[:top_k]
    return [(int(ids[o]), float(sims[o])) for o in order if sims[o] >= min_similarity]


def premise_windows(candidate_ids: Iterable[int], index: DocumentIndex, max_window: int) -> list[tuple[int, ...]]:
    """Every run of 1..max_window consecutive same-section sentences that contains a candidate."""
    windows: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for cid in candidate_ids:
        lo, hi = index.section_bounds(cid)
        for size in range(1, max(1, max_window) + 1):
            for start in range(cid - size + 1, cid + 1):
                if start >= lo and start + size <= hi:
                    w = tuple(range(start, start + size))
                    if w not in seen:
                        seen.add(w)
                        windows.append(w)
    return windows


def score_against_source(hypothesis: str, hyp_emb: np.ndarray, index: DocumentIndex, nli: NLIModel, *,
                         pool: Sequence[int] | None, top_k: int, min_similarity: float,
                         window: int) -> list[ScoredPremise]:
    """NLI-score a hypothesis against its best source premises, sorted by entailment (desc)."""
    candidates = top_candidates(hyp_emb, index, pool, top_k, min_similarity)
    if not candidates:
        return []
    cand_sim = dict(candidates)
    windows = premise_windows(cand_sim, index, window)
    texts = [" ".join(index.sentences[i] for i in w) for w in windows]
    probs = nli.predict([(t, hypothesis) for t in texts])
    scored = [
        ScoredPremise(w, t, max(cand_sim.get(i, float(index.embeddings[i] @ hyp_emb)) for i in w),
                      float(p[ENTAILMENT]), float(p[CONTRADICTION]))
        for w, t, p in zip(windows, texts, probs)
    ]
    scored.sort(key=lambda s: (-s.entailment, len(s.sentence_ids)))
    return scored


def entailed_by_any(hypotheses: Sequence[str], premises: Sequence[str], embedder: Embedder, nli: NLIModel, *,
                    min_similarity: float, min_entailment: float) -> list[bool]:
    """For each hypothesis: is it entailed by at least one premise?

    Similarity only selects which pairs to test. It is never the decision, because
    embeddings barely register negation ("fell because of X" ~ "fell, not because of X").
    """
    if not hypotheses or not premises:
        return [False] * len(hypotheses)
    sims = embedder.encode(list(hypotheses)) @ embedder.encode(list(premises)).T
    pairs = [(i, j) for i in range(len(hypotheses)) for j in range(len(premises)) if sims[i, j] >= min_similarity]
    result = [False] * len(hypotheses)
    if pairs:
        probs = nli.predict([(premises[j], hypotheses[i]) for i, j in pairs])
        for (i, _), p in zip(pairs, probs):
            if p[ENTAILMENT] >= min_entailment:
                result[i] = True
    return result
