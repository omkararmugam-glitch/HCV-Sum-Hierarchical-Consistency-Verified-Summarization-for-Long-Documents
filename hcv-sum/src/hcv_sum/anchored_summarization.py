"""Stage 2 -- context-anchored section summarization.

For each section, retrieve the top-k sentences from OTHER sections that are most
similar to any sentence of this section, then give them to the summarizer as
supporting context (after Ou & Lapata, 2025).

How the context is presented depends on ``summarization.context_mode``:
- ``none``     : ablation, section text only.
- ``append``   : ``<section text>\\n\\n<context sentences>``. Intended for
                 non-instruction models such as DistilBART.
- ``instruct`` : an explicit prompt with SECTION / BACKGROUND fields. Intended
                 for instruction-tuned models such as flan-t5.

FIRST-PASS LOGIC / TO STRENGTHEN LATER:
- A CNN/DailyMail-tuned model (DistilBART) cannot be *told* that the context is
  background, and it has a strong lead bias. In ``append`` mode the context is
  therefore either ignored (most likely) or summarized as if it belonged to the
  section (a new source of cross-section contamination). Leaked sentences are
  detected (matched better by the context than by the section) and, with
  ``drop_context_leaks``, removed. The first failure (context ignored) cannot be
  measured from the output alone. Real anchoring needs a model trained to condition on
  context (e.g. fine-tuned with a context field) or an instruction-following LLM.
- Retrieval is plain cosine top-k with a similarity floor. No reranking, no
  diversity (MMR), no coreference-aware expansion.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .checkpoint import Stage2Checkpoint, generation_key
from .config import SummarizationConfig
from .models import Embedder, Summarizer
from .scoring import DocumentIndex
from .text_utils import drop_incomplete_tail, is_complete_sentence, split_sentences
from .types import RetrievedSentence, Section, SectionSummary

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?%)])")
_SPACE_AFTER_OPEN = re.compile(r"([($])\s+")


# Reported-speech tags in CNN/DailyMail style. The subject must look like a PERSON -- a pronoun, or
# capitalised name/title words -- so "The report says ..." is never touched: stripping that would turn
# a reported claim into a bare assertion.
_SPEECH_VERB = r"(?:says|said|adds|added|notes|noted|insists|insisted|warns|warned|explains|explained)"
_PERSON = r"(?:he|she|they|He|She|They|(?:[A-Z][a-z.]+(?:\s+[A-Z][a-z.]+){0,2})(?:\s+(?:chairman|chairwoman|chair|president|governor|official|spokesman|spokeswoman))?)"
_TRAILING_TAG = re.compile(rf",\s*({_PERSON}\s+{_SPEECH_VERB})\s*([.!?])$")
_LEADING_TAG = re.compile(rf"^({_PERSON}\s+{_SPEECH_VERB})(?:\s+that)?\s+(?=\S)")
_NOT_A_SUBJECT = frozenset({"the", "a", "an", "this", "that", "it", "we", "our", "i", "you"})


def strip_invented_attribution(sentence: str, source_text: str) -> tuple[str, str | None]:
    """Remove a speaker tag that does not occur in the source. Returns (sentence, removed tag or None)."""
    source = source_text.lower()
    m = _TRAILING_TAG.search(sentence)
    if m and m.group(1).lower() not in source and m.group(1).split()[0].lower() not in _NOT_A_SUBJECT:
        return sentence[: m.start()].rstrip() + m.group(2), m.group(1)
    m = _LEADING_TAG.match(sentence)
    if m and m.group(1).lower() not in source and m.group(1).split()[0].lower() not in _NOT_A_SUBJECT:
        rest = sentence[m.end():]
        return rest[:1].upper() + rest[1:], m.group(1)
    return sentence, None


def clean_generated(text: str) -> str:
    """Undo detokenization artifacts of CNN/DM-tuned models ("12 percent ." -> "12 percent.")."""
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return _SPACE_AFTER_OPEN.sub(r"\1", text).strip()


def retrieve_context(section: Section, index: DocumentIndex, cfg: SummarizationConfig) -> list[RetrievedSentence]:
    own = np.array(list(section.sentence_indices), dtype=int)
    others = np.flatnonzero(index.section_of != section.index)
    if own.size == 0 or others.size == 0 or cfg.context_top_k <= 0:
        return []
    other_emb = index.embeddings[others]
    if cfg.retrieval_scoring == "max":
        # A candidate scores by its best match to ANY sentence in the section, so a
        # single shared entity/event (e.g. "the Monterrey shutdown") can pull it in.
        scores = (other_emb @ index.embeddings[own].T).max(axis=1)
    else:
        centroid = index.embeddings[own].mean(axis=0)
        scores = other_emb @ (centroid / (np.linalg.norm(centroid) + 1e-9))
    order = np.argsort(-scores)[: cfg.context_top_k]
    return [
        RetrievedSentence(int(others[o]), int(index.section_of[others[o]]), index.sentences[others[o]], float(scores[o]))
        for o in order
        if scores[o] >= cfg.context_min_similarity
    ]


def _render(section_text: str, context_text: str, cfg: SummarizationConfig) -> str:
    if cfg.context_mode == "instruct":
        rendered = f"{cfg.instruct_prompt}\n\nSECTION: {section_text}"
        return f"{rendered}\n\nBACKGROUND: {context_text}" if context_text else rendered
    if cfg.context_mode == "append" and context_text:
        return f"{section_text}\n\n{context_text}"
    return section_text


def build_model_input(section: Section, context: list[RetrievedSentence], summarizer: Summarizer,
                      cfg: SummarizationConfig) -> tuple[str, bool]:
    """Fit section + context into the input budget. Context is cut first; the section only if it alone overflows."""
    section_text = section.text
    context_text = "" if cfg.context_mode == "none" else " ".join(
        c.text for c in sorted(context, key=lambda c: c.sentence_index))
    overhead = summarizer.count_tokens(_render("", "x", cfg)) + 8
    available = min(cfg.max_input_tokens, summarizer.model_max_input) - overhead
    section_tokens = summarizer.count_tokens(section_text)
    if section_tokens > available:
        return _render(summarizer.truncate(section_text, available), "", cfg), True
    if context_text:
        context_text = summarizer.truncate(context_text, min(cfg.max_context_tokens, available - section_tokens))
    return _render(section_text, context_text, cfg), False


@dataclass
class _Prepared:
    """Everything decided before generation, so sections can be generated in batches."""

    section: Section
    context: list[RetrievedSentence]
    model_input: str
    truncated: bool
    needs_generation: bool
    min_new: int = 0
    max_new: int = 0


def _prepare(section: Section, index: DocumentIndex, summarizer: Summarizer,
             cfg: SummarizationConfig) -> _Prepared:
    context = retrieve_context(section, index, cfg)
    model_input, truncated = build_model_input(section, context, summarizer, cfg)
    n_tokens = summarizer.count_tokens(section.text)
    if n_tokens <= cfg.passthrough_tokens:
        return _Prepared(section, context, model_input, truncated, needs_generation=False)
    max_new = max(cfg.min_new_tokens + 5, min(cfg.max_new_tokens, int(n_tokens * cfg.max_length_ratio)))
    return _Prepared(section, context, model_input, truncated, True, min(cfg.min_new_tokens, max_new // 2), max_new)


def summarize_section(section: Section, index: DocumentIndex, summarizer: Summarizer, embedder: Embedder,
                      cfg: SummarizationConfig) -> SectionSummary:
    prepared = _prepare(section, index, summarizer, cfg)
    generated = None
    if prepared.needs_generation:
        generated = summarizer.summarize(prepared.model_input, min_new_tokens=prepared.min_new,
                                         max_new_tokens=prepared.max_new)
    return _finish(prepared, generated, index, embedder, cfg)


def _finish(prepared: _Prepared, generated: str | None, index: DocumentIndex, embedder: Embedder,
            cfg: SummarizationConfig) -> SectionSummary:
    section, context = prepared.section, prepared.context
    if generated is None:
        # Short section: use it verbatim. Its sentence list is reused as-is rather than
        # re-split from the joined text, which would fuse sentences that lack terminal
        # punctuation (headings, bullet fragments, transcript lines).
        return SectionSummary(section.index, prepared.model_input, section.text, list(section.sentences), context,
                              False, prepared.truncated, [], [], [])

    summary = clean_generated(generated)
    truncated = prepared.truncated
    sentences = split_sentences(summary)
    dropped: list[tuple[str, str]] = []
    notes: list[str] = []
    sentences, cut = drop_incomplete_tail(sentences)
    dropped += [(s, "incomplete sentence (hit max_new_tokens)") for s in cut]
    if len(sentences) == 1 and not is_complete_sentence(sentences[0]):
        # The ONLY generated sentence was cut off mid-clause. drop_incomplete_tail keeps it so a
        # summary is never empty, but a truncated claim must not reach the merge or the evidence
        # panel, so fall back to the section's own first sentence.
        dropped.append((sentences[0], "incomplete sentence (hit max_new_tokens)"))
        sentences = section.sentences[:1]
        notes.append("the only generated sentence was cut off; using the section's first sentence instead")

    if cfg.strip_invented_attribution:
        cleaned = []
        for sentence in sentences:
            new, tag = strip_invented_attribution(sentence, section.text)
            if tag:
                notes.append(f"removed speaker attribution not in the source: '{tag}' (from: {sentence})")
            cleaned.append(new)
        sentences = cleaned

    leaks: list[int] = []
    if context and sentences:
        s_emb = embedder.encode(sentences)
        own_best = (s_emb @ index.embeddings[list(section.sentence_indices)].T).max(axis=1)
        ctx_best = (s_emb @ index.embeddings[[c.sentence_index for c in context]].T).max(axis=1)
        leaks = [i for i in range(len(sentences)) if ctx_best[i] > own_best[i] + cfg.leak_margin]
        if cfg.drop_context_leaks and leaks:
            dropped += [(sentences[i], f"context leak (matches retrieved context {ctx_best[i]:.2f} "
                                       f"> own section {own_best[i]:.2f})") for i in leaks]
            sentences = [s for i, s in enumerate(sentences) if i not in set(leaks)]
            leaks = []
            if not sentences:
                # Everything generated came from the context: fall back to the section's lead sentence
                # rather than silently losing the section.
                sentences = section.sentences[:1]
                notes.append("every generated sentence was a context leak; using the section's first sentence instead")

    return SectionSummary(section.index, prepared.model_input, summary, sentences, context, True, truncated,
                          leaks, dropped, notes)


def _protect_cross_referenced(prepared: "list[_Prepared]", summaries: list[SectionSummary],
                              index: DocumentIndex, embedder: Embedder,
                              cfg: SummarizationConfig) -> None:
    """Re-attach source sentences that another section's retrieval picked out, if compression lost them.

    The signal is already computed and costs nothing extra: when Stage 2 summarises section X it
    retrieves the sentences elsewhere in the document most related to X. A sentence that some other
    section reached for is, by that same measure, load-bearing for cross-section comparison -- and
    it is exactly the kind of sentence Stage 3a needs on BOTH sides to detect a contradiction.

    Measured motivation: in sample 01 both halves of the planted contradiction were retrieved as
    each other's context (cosine 0.71) and both were then compressed away, leaving Stage 3a nothing
    to compare. This restores them verbatim, which makes those summaries longer and more extractive
    -- a deliberate trade of abstraction for verifiability.
    """
    referenced: dict[int, set[int]] = {}
    for p in prepared:
        for c in p.context:
            referenced.setdefault(c.section_index, set()).add(c.sentence_index)

    for summary in summaries:
        wanted = referenced.get(summary.section_index, set())
        if not wanted or not summary.generated:
            continue
        section = index.sections[summary.section_index]
        covered = embedder.encode(summary.sentences) if summary.sentences else None
        for gid in sorted(wanted):
            sentence = index.sentences[gid]
            if covered is not None:
                best = float((covered @ index.embeddings[gid]).max())
                if best >= cfg.protection_coverage_similarity:
                    continue        # the generated summary already says this
            summary.sentences.append(sentence)
            summary.protected.append(sentence)
            local = gid - section.sentence_offset
            summary.notes.append(f"kept source sentence {local} verbatim: another section retrieved it as "
                                 f"context, and the generated summary did not cover it")


def summarize_sections(sections: list[Section], index: DocumentIndex, summarizer: Summarizer, embedder: Embedder,
                       cfg: SummarizationConfig, *, checkpoint: Stage2Checkpoint | None = None,
                       heartbeat: Callable[[str], None] | None = None,
                       heartbeat_every: tuple[int, float] = (10, 30.0)) -> list[SectionSummary]:
    """Summarize every section, generating in batches of ``cfg.batch_size``.

    ``checkpoint``: generations found there are reused, new ones are appended after every batch
    (checkpoint.py). ``heartbeat``: called with a progress line (done/total, s/section, ETA) at least
    every ``heartbeat_every`` = (sections, seconds), so a long run is visibly alive.

    Generation dominates wall-clock on CPU (~2.6 s per section one at a time). Batching padded
    inputs through one generate() call is ~2x faster and leaves the text unchanged, because the
    attention mask hides the padding. Sections short enough to pass through verbatim never reach
    the model, so they are handled first and cost nothing.
    """
    prepared = [_prepare(sec, index, summarizer, cfg) for sec in sections]
    results: dict[int, str] = {}
    keys: dict[int, str] = {}
    if checkpoint is not None:
        for i, p in enumerate(prepared):
            if p.needs_generation:
                keys[i] = generation_key(p.model_input, p.min_new, p.max_new)
                cached = checkpoint.get(keys[i])
                if cached is not None:
                    results[i] = cached

    # Sections are batched ONLY with others that have the same (min_new, max_new) budget, because
    # generate() applies one length limit to the whole batch. Mixing budgets would silently change
    # a section's summary length depending on what it happened to be batched with; grouping by
    # budget keeps every summary byte-identical to generating one at a time (tests/test_batching.py).
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, p in enumerate(prepared):
        if p.needs_generation and i not in results:
            buckets.setdefault((p.min_new, p.max_new), []).append(i)

    to_generate = sum(len(m) for m in buckets.values())
    from_checkpoint = len(results)
    done, t0 = 0, time.perf_counter()
    last_done, last_time = 0, t0
    if heartbeat and (to_generate or from_checkpoint):
        heartbeat(f"Stage 2: {to_generate} sections to generate in {len(buckets)} length buckets"
                  + (f", {from_checkpoint} restored from checkpoint" if from_checkpoint else "")
                  + f" ({len(sections) - to_generate - from_checkpoint} short enough to pass through)")

    batch_size = max(1, cfg.batch_size)
    for (min_new, max_new), members in buckets.items():
        for start in range(0, len(members), batch_size):
            chunk = members[start: start + batch_size]
            outputs = summarizer.summarize_batch([prepared[i].model_input for i in chunk],
                                                 min_new_tokens=min_new, max_new_tokens=max_new)
            results.update(zip(chunk, outputs))
            if checkpoint is not None:
                checkpoint.put_many([(keys[i], sections[i].index, out) for i, out in zip(chunk, outputs)])
            done += len(chunk)
            now = time.perf_counter()
            if heartbeat and (done == to_generate or done - last_done >= heartbeat_every[0]
                              or now - last_time >= heartbeat_every[1]):
                rate = (now - t0) / done
                eta = rate * (to_generate - done)
                heartbeat(f"Stage 2: {done}/{to_generate} sections generated, {rate:.1f} s/section, "
                          f"elapsed {(now - t0) / 60:.1f} min, ETA {eta / 60:.1f} min")
                last_done, last_time = done, now

    summaries = [_finish(prepared[i], results.get(i), index, embedder, cfg) for i in range(len(sections))]
    if cfg.protect_cross_referenced:
        _protect_cross_referenced(prepared, summaries, index, embedder, cfg)
    return summaries
