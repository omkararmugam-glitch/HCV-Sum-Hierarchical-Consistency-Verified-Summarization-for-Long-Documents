"""Stage 3 -- cross-section contradiction detection and resolution.

Source-faithfulness checkers (e.g. SummaC) compare a summary with its source.
They cannot see that two section-summaries, each faithful to its own section,
contradict EACH OTHER. This stage checks for that before merging, with TWO
complementary checks. Neither is "the" mechanism on its own -- measured on the
sample set, each one catches a planted contradiction the other cannot:

  3a  summary claim vs summary claim of a sibling section. Needs BOTH sides to
      survive summarization. Before Stage 2's cross-reference protection it
      caught no planted contradiction in any normal run; with it, it catches
      sample 01 (P=0.996).
  3b  summary claim vs the SOURCE sentences of sibling sections. Catches a
      contradiction whose other side compression dropped (sample 03, P=0.61),
      which 3a structurally cannot see.

Neither check catches sample 04, whose contradiction needs date arithmetic
("24-month term from the Effective Date" vs "expires December 31, 2026"). The
pair IS compared and scores 0.02: sentence-pair NLI detects semantic opposition,
not derived numeric conflict. See FINDINGS.md.

Detection
    Every candidate pair is scored with an NLI cross-encoder in both directions,
    and the two P(contradiction) values are combined with ``direction_aggregation``
    (see AGGREGATIONS). Pre-filters decide which pairs reach NLI: cosine similarity,
    per-claim top-k, a pair budget, comparative framing, and an optional entity gate.

Resolution
    For each flagged pair, each claim is scored for support by ITS OWN section's
    source text (best NLI entailment over the top-k most similar source
    sentences and short windows of them). The better-supported claim is kept.
    If the support gap is below ``resolution_margin`` neither is removed: both
    are flagged "unresolved", because guessing would hide the conflict.
    Pairs are processed from most to least contradictory; a pair whose claim has
    already been rejected is marked "superseded".

FIRST-PASS LOGIC / TO STRENGTHEN LATER:
- Sentence-pair NLI only sees two sentences. Contradictions needing arithmetic
  are MEASURED to be missed (sample 04: 0.02 as written, 0.91 once the end date
  is stated explicitly); multi-sentence reasoning and coreference ("it" vs a
  named entity) are expected to be missed too.
- Precision is the weakest point. After the comparative-framing filter, the
  remaining false positives on the clean sample come from shared numbers with
  different referents ("five retrieved notes" vs "five random seeds") and from
  two steps of one method read as rival claims. No text signal separates these
  from true positives; true and false scores overlap (0.51-0.63 false, 0.61 true).
- Small NLI models over-predict contradiction for pairs that differ only by a
  number or a scope qualifier ("improved on dataset A" vs "did not improve on
  dataset B"). The prefilter does not help there; a calibrated threshold or a
  larger NLI model would.
- Resolution assumes that when the SOURCE itself is inconsistent, the more
  entailed claim is correct. In a genuinely self-contradictory document both
  claims are faithful and this stage can only honestly report "unresolved".
- O(n^2) NLI calls in the number of summary sentences. Fine for summaries,
  expensive if run over raw source sentences (see ``diagnose_source_sections``).
- The non-claim filter (speaker labels stripped, fewer than ``min_claim_words``
  content words skipped) is a heuristic aimed at transcript pleasantries
  ("Thank you, operator."), which NLI models readily call contradictory under
  the same-situation assumption of their training data.

One-sided check (added after run 2)
    Abstractive compression often keeps ONE side of a contradiction and drops the
    other. Summary-vs-summary comparison then cannot see it, and Stage 5 marks the
    surviving claim "supported" because its own section does support it. So each
    summary sentence is also compared with the SOURCE sentences of sibling
    sections (skipping source sentences that survived near-verbatim into their own
    summary; the summary-vs-summary check covers those). Resolution follows the
    same rule: if the summary claim is clearly less supported by its own section
    than the sibling source sentence is by its section, the summary claim is
    rejected; otherwise it is kept and flagged "unresolved".
"""

from __future__ import annotations

import dataclasses
import logging

import time
from typing import Callable

import numpy as np

from .config import ContradictionConfig
from .models import CONTRADICTION, Embedder, NLIModel
from .pair_selection import select_pairs_chunked
from .scoring import DocumentIndex, score_against_source
from .text_utils import (comparative_framing, content_tokens, content_word_count, entity_tokens,
                         has_verb_or_number, strip_reporting_frame, strip_speaker_label)
from .types import ClaimRef, Contradiction, ContradictionReport, PairStats, SectionSummary

log = logging.getLogger(__name__)


def use_chunked(n_rows: int, n_cols: int, cfg: ContradictionConfig) -> bool:
    """Large comparisons use the memory-bounded selection (pair_selection.py); small ones the dense code."""
    return cfg.dense_pair_limit > 0 and n_rows * n_cols > cfg.dense_pair_limit


def _stats_from(sel: dict) -> PairStats:
    return PairStats(sel["total"], len(sel["checked"]), sel["skipped_top_k"], sel["skipped_budget"],
                     sel["skipped_entity"], sel["skipped_comparative"], sel["skipped_topic"])


# Contradiction is logically symmetric, but NLI scores are not: a premise with extra content
# often scores lower in one direction. max = most recall, min = strictly symmetric, mean = compromise.
AGGREGATIONS = {
    "max": np.maximum,
    "min": np.minimum,
    "mean": lambda a, b: (a + b) / 2,
}


def select_pairs(sims: np.ndarray, eligible: "list[list[int]]", cfg: ContradictionConfig,
                 symmetric: bool) -> tuple[list[tuple[int, int]], int, int]:
    """Pick which (row, column) pairs to send to NLI, and report what was skipped.

    ``sims`` is a row-by-column cosine matrix; ``eligible[i]`` lists the columns that may be
    compared with row ``i`` (cross-section only). Three filters, in this order:

    1. ``pair_min_similarity`` -- a pair must share some topic to be worth an NLI call.
    2. ``candidate_top_k``     -- keep only each row's k most similar candidates. Without it the
       work is O(n^2) in the number of claims, which is untenable on a long document.
    3. ``max_pairs``           -- a hard ceiling on NLI work, keeping the highest-similarity pairs.

    Returns (pairs, skipped_by_top_k, skipped_by_budget). Both skip counts are reported rather
    than hidden: they are the contradictions this run did not even look for.
    """
    def dedupe(items: "list[tuple[float, int, int]]") -> "list[tuple[float, int, int]]":
        """For a symmetric comparison, i-vs-j and j-vs-i are one pair; keep the better-scoring copy."""
        if not symmetric:
            return items
        best: dict[tuple[int, int], tuple[float, int, int]] = {}
        for sim, i, j in items:
            key = (i, j) if i < j else (j, i)
            if sim > best.get(key, (-2.0, 0, 0))[0]:
                best[key] = (sim, i, j)
        return list(best.values())

    # Counts are computed on DEDUPED sets at each step, so every pair is attributed to exactly one
    # outcome. Halving a per-row skip count double-counts pairs that one row dropped and the other
    # kept, which made the reported numbers exceed the number of pairs that exist.
    above_floor: list[tuple[float, int, int]] = []
    within_top_k: list[tuple[float, int, int]] = []
    for i, columns in enumerate(eligible):
        row = [(float(sims[i, j]), i, j) for j in columns if sims[i, j] >= cfg.pair_min_similarity]
        above_floor.extend(row)
        if cfg.candidate_top_k > 0 and len(row) > cfg.candidate_top_k:
            row = sorted(row, key=lambda t: -t[0])[: cfg.candidate_top_k]
        within_top_k.extend(row)

    above_floor = dedupe(above_floor)
    scored = dedupe(within_top_k)
    skipped_top_k = len(above_floor) - len(scored)

    skipped_budget = 0
    if cfg.max_pairs > 0 and len(scored) > cfg.max_pairs:
        scored.sort(key=lambda t: -t[0])
        skipped_budget = len(scored) - cfg.max_pairs
        scored = scored[: cfg.max_pairs]

    return [(i, j) for _, i, j in scored], skipped_top_k, skipped_budget


def filter_by_shared_entity(eligible: "list[list[int]]", row_texts: "list[str]", col_texts: "list[str]",
                           symmetric: bool) -> tuple["list[list[int]]", int]:
    """Keep only pairs that share an entity-like token (name, acronym or figure).

    MEASURED CONSEQUENCE, which is why this is off by default: sample 03's planted contradiction is
    between "Every enterprise customer is now running on the new Helix Core platform" and a sentence
    whose entity set is empty (all lower case, "roughly sixty" spelled out). The filter drops that
    pair, taking the catch rate on our test set from 2/3 to 1/3. It removes real contradictions, not
    just unrelated ones, so it is opt-in and documented as such.
    """
    row_entities = [entity_tokens(t) for t in row_texts]
    col_entities = row_entities if row_texts is col_texts else [entity_tokens(t) for t in col_texts]
    kept, skipped = [], 0
    for i, columns in enumerate(eligible):
        keep = [j for j in columns if row_entities[i] & col_entities[j]]
        skipped += len(columns) - len(keep)
        kept.append(keep)
    return kept, (skipped // 2 if symmetric else skipped)


def filter_by_shared_content_word(eligible: "list[list[int]]", row_texts: "list[str]",
                                  col_texts: "list[str]", symmetric: bool) -> tuple["list[list[int]]", int]:
    """Keep only pairs that share at least one content word (topic overlap).

    Aimed at the dominant false flag on real documents (FINDINGS 10.5): two sentences with the same
    grammatical subject describing different things ("We conduct an experiment with the ChaosNLI
    dataset" vs "We construct the RefNLI benchmark"), which NLI reads as rival claims about one
    situation. Unlike the entity gate it keeps ordinary nouns, so "the labor market" still matches.
    Limitation: a contradiction worded with no word in common ("the deal closed" vs "the
    acquisition was abandoned") is skipped. The skip is counted and reported.
    """
    row_tokens = [content_tokens(t) for t in row_texts]
    col_tokens = row_tokens if row_texts is col_texts else [content_tokens(t) for t in col_texts]
    kept, skipped = [], 0
    for i, columns in enumerate(eligible):
        keep = [j for j in columns if row_tokens[i] & col_tokens[j]]
        skipped += len(columns) - len(keep)
        kept.append(keep)
    return kept, (skipped // 2 if symmetric else skipped)


def filter_comparative_pairs(eligible: "list[list[int]]", row_texts: "list[str]", col_texts: "list[str]",
                             symmetric: bool) -> tuple["list[list[int]]", int]:
    """Drop pairs where either sentence is explicitly framed as a comparison.

    What this is and is not: 4 of the 5 false positives on the contradiction-free sample paired a
    description of the proposed system with "The baseline is the same six-layer classifier
    fine-tuned WITHOUT retrieval". Those sentences describe two DIFFERENT systems on purpose. NLI
    scores them as contradictory because it assumes both sentences share one context -- the
    reference-determinacy failure documented by Chen et al. (2025). Comparison wording is one
    visible cue of that mismatch, so this is a NARROW HEURISTIC for a symptom, not a fix for the
    cause: mismatches without such wording ("five retrieved notes" vs "five random seeds") pass
    straight through. Measured on 1,143 RefNLI pairs it removes 7 of 282 false flags and one true
    contradiction, so its effect at scale is close to none. The shared-entity gate does better there
    (-45% false flags, -2 of 57 true) but blocked both true positives on our samples, because those
    name their referents with common nouns rather than proper names; see FINDINGS.md 9.4 and 9.5.

    Limitation: a sentence can be both comparative and genuinely contradictory ("revenue rose faster
    than last year" vs "revenue fell"), and this filter would skip that pair. It is reported as a
    skip count, and it is a config switch.
    """
    row_flags = [comparative_framing(t) for t in row_texts]
    col_flags = row_flags if row_texts is col_texts else [comparative_framing(t) for t in col_texts]
    kept, skipped = [], 0
    for i, columns in enumerate(eligible):
        keep = [j for j in columns if not (row_flags[i] or col_flags[j])]
        skipped += len(columns) - len(keep)
        kept.append(keep)
    return kept, (skipped // 2 if symmetric else skipped)


def is_claim(text: str, cfg: ContradictionConfig) -> bool:
    stripped = strip_speaker_label(text)
    if content_word_count(stripped) < cfg.min_claim_words:
        return False
    if not cfg.require_verb_or_number or len(stripped.split()) >= cfg.verbless_fragment_max_words:
        return True
    return has_verb_or_number(stripped)


def nli_text(text: str, cfg: ContradictionConfig) -> str:
    """The form of a claim the NLI model sees. Speaker labels ("Jane Doe, CFO: ...") are always stripped:
    different speakers are not a contradiction. Reporting frames ("We also note that ...") are stripped
    when ``strip_reporting_frames`` is on (FINDINGS 12.1). Selection (embeddings) is unaffected."""
    text = strip_speaker_label(text)
    if cfg.strip_attribution_frames:
        from .dialogue import strip_attribution_frame
        text = strip_attribution_frame(text)
    return strip_reporting_frame(text) if cfg.strip_reporting_frames else text


def _score_pairs(pairs: list[tuple[str, str]], nli: NLIModel, cfg: ContradictionConfig,
                 progress: "Callable[[int, int], None] | None" = None, chunk: int = 500):
    """Score every pair in both directions. Returns (forward, backward, aggregated).

    With ``progress``, the pairs are scored in blocks of ``chunk`` and ``progress(done, total)`` is called
    after each block, so a long Stage 3 can report how far it has got. This is reporting only: each pair is
    scored independently of the others, so blocking changes no score, and without ``progress`` the calls are
    exactly the two whole-list calls they always were.
    """
    stripped = [(nli_text(a, cfg), nli_text(b, cfg)) for a, b in pairs]
    if progress is None:
        forward = nli.predict(stripped)[:, CONTRADICTION]
        backward = nli.predict([(b, a) for a, b in stripped])[:, CONTRADICTION]
        return forward, backward, AGGREGATIONS[cfg.direction_aggregation](forward, backward)
    forward_blocks, backward_blocks = [], []
    for start in range(0, len(stripped), max(1, chunk)):
        block = stripped[start: start + max(1, chunk)]
        forward_blocks.append(nli.predict(block)[:, CONTRADICTION])
        backward_blocks.append(nli.predict([(b, a) for a, b in block])[:, CONTRADICTION])
        progress(min(start + len(block), len(stripped)), len(stripped))
    empty = np.zeros(0, dtype=np.float32)
    forward = np.concatenate(forward_blocks) if forward_blocks else empty
    backward = np.concatenate(backward_blocks) if backward_blocks else empty
    return forward, backward, AGGREGATIONS[cfg.direction_aggregation](forward, backward)


def detect_contradictions(claims: list[list[str]], embedder: Embedder, nli: NLIModel,
                          cfg: ContradictionConfig, progress: "Callable[[int, int], None] | None" = None,
                          progress_chunk: int = 500) -> tuple[PairStats, list[Contradiction]]:
    """Return (pair statistics, flagged contradictions sorted by score desc).

    ``claims[s]`` is the list of claim sentences belonging to section ``s``. Sentences failing
    ``is_claim`` are never compared and are not counted in either total.
    """
    flat = [ClaimRef(s, j, text) for s, sentences in enumerate(claims) for j, text in enumerate(sentences)
            if is_claim(text, cfg)]
    if len(flat) < 2:
        return PairStats(0, 0, 0, 0), []
    emb = embedder.encode([strip_speaker_label(c.text) for c in flat])
    if use_chunked(len(flat), len(flat), cfg):
        texts = [c.text for c in flat]
        sections = np.array([c.section_index for c in flat])
        sel = select_pairs_chunked(emb, emb, sections, sections, texts, texts, cfg, symmetric=True)
        stats, checked = _stats_from(sel), sel["checked"]
        if not checked:
            return stats, []
        forward, backward, score = _score_pairs([(flat[i].text, flat[j].text) for i, j in checked], nli, cfg,
                                                progress, progress_chunk)
        found = [Contradiction(flat[i], flat[j], float(s), float(f), float(b))
                 for (i, j), f, b, s in zip(checked, forward, backward, score) if s >= cfg.threshold]
        found.sort(key=lambda c: -c.score)
        return stats, found
    sims = emb @ emb.T

    total = sum(1 for i in range(len(flat)) for j in range(i + 1, len(flat))
                if flat[i].section_index != flat[j].section_index)
    eligible = [[j for j in range(len(flat)) if flat[j].section_index != flat[i].section_index]
                for i in range(len(flat))]
    texts = [c.text for c in flat]
    skipped_entity = skipped_comparative = skipped_topic = 0
    if cfg.skip_comparative_framing:
        eligible, skipped_comparative = filter_comparative_pairs(eligible, texts, texts, symmetric=True)
    if cfg.require_shared_entity:
        eligible, skipped_entity = filter_by_shared_entity(eligible, texts, texts, symmetric=True)
    if cfg.require_shared_content_word:
        eligible, skipped_topic = filter_by_shared_content_word(eligible, texts, texts, symmetric=True)
    checked, skipped_top_k, skipped_budget = select_pairs(sims, eligible, cfg, symmetric=True)
    if not checked:
        return PairStats(total, 0, skipped_top_k, skipped_budget, skipped_entity, skipped_comparative,
                         skipped_topic), []

    forward, backward, score = _score_pairs([(flat[i].text, flat[j].text) for i, j in checked], nli, cfg,
                                            progress, progress_chunk)
    found = [
        Contradiction(flat[i], flat[j], float(s), float(f), float(b))
        for (i, j), f, b, s in zip(checked, forward, backward, score)
        if s >= cfg.threshold
    ]
    found.sort(key=lambda c: -c.score)
    return PairStats(total, len(checked), skipped_top_k, skipped_budget, skipped_entity, skipped_comparative,
                     skipped_topic), found


def detect_one_sided(claims: list[list[str]], skip: set[str], index: DocumentIndex, embedder: Embedder,
                     nli: NLIModel, cfg: ContradictionConfig,
                     progress: "Callable[[int, int], None] | None" = None,
                     progress_chunk: int = 500) -> tuple[PairStats, list[Contradiction]]:
    """Compare each summary claim with the source sentences of SIBLING sections.

    ``claim_a`` is the summary claim; ``claim_b`` is a source sentence whose ``sentence_index`` is
    its position inside its own section. Returns (pairs_checked, flagged sorted by score desc).
    """
    summary = [ClaimRef(s, j, t) for s, sentences in enumerate(claims) for j, t in enumerate(sentences)
               if f"{s}:{j}" not in skip and is_claim(t, cfg)]
    if not summary:
        return PairStats(0, 0, 0, 0), []
    sum_emb = embedder.encode([strip_speaker_label(c.text) for c in summary])

    sources: list[ClaimRef] = []
    for sec in index.sections:
        own_summary = claims[sec.index] if sec.index < len(claims) else []
        own_emb = embedder.encode(own_summary) if own_summary else None
        for k, (gid, text) in enumerate(zip(sec.sentence_indices, sec.sentences)):
            if not is_claim(text, cfg):
                continue
            if own_emb is not None and float((own_emb @ index.embeddings[gid]).max()) >= cfg.sibling_source_skip_similarity:
                continue   # survived into its own summary: the summary-vs-summary check covers it
            sources.append(ClaimRef(sec.index, k, text))
    if not sources:
        return PairStats(0, 0, 0, 0), []
    src_emb = embedder.encode([strip_speaker_label(c.text) for c in sources])
    if use_chunked(len(summary), len(sources), cfg):
        sel = select_pairs_chunked(sum_emb, src_emb, np.array([c.section_index for c in summary]),
                                   np.array([c.section_index for c in sources]), [c.text for c in summary],
                                   [c.text for c in sources], cfg, symmetric=False)
        stats, checked = _stats_from(sel), sel["checked"]
        if not checked:
            return stats, []
        forward, backward, score = _score_pairs([(summary[i].text, sources[j].text) for i, j in checked], nli,
                                                cfg, progress, progress_chunk)
        found = [Contradiction(summary[i], sources[j], float(s), float(f), float(b), kind="one_sided")
                 for (i, j), f, b, s in zip(checked, forward, backward, score) if s >= cfg.threshold]
        found.sort(key=lambda c: -c.score)
        return stats, found
    sims = sum_emb @ src_emb.T

    eligible = [[j for j in range(len(sources)) if sources[j].section_index != summary[i].section_index]
                for i in range(len(summary))]
    total = sum(len(columns) for columns in eligible)
    skipped_entity = skipped_comparative = skipped_topic = 0
    summary_texts, source_texts = [c.text for c in summary], [c.text for c in sources]
    if cfg.skip_comparative_framing:
        eligible, skipped_comparative = filter_comparative_pairs(eligible, summary_texts, source_texts,
                                                                 symmetric=False)
    if cfg.require_shared_entity:
        eligible, skipped_entity = filter_by_shared_entity(eligible, summary_texts, source_texts, symmetric=False)
    if cfg.require_shared_content_word:
        eligible, skipped_topic = filter_by_shared_content_word(eligible, summary_texts, source_texts,
                                                                symmetric=False)
    checked, skipped_top_k, skipped_budget = select_pairs(sims, eligible, cfg, symmetric=False)
    if not checked:
        return PairStats(total, 0, skipped_top_k, skipped_budget, skipped_entity, skipped_comparative,
                         skipped_topic), []
    forward, backward, score = _score_pairs([(summary[i].text, sources[j].text) for i, j in checked], nli,
                                            cfg, progress, progress_chunk)
    found = [
        Contradiction(summary[i], sources[j], float(s), float(f), float(b), kind="one_sided")
        for (i, j), f, b, s in zip(checked, forward, backward, score)
        if s >= cfg.threshold
    ]
    found.sort(key=lambda c: -c.score)
    return PairStats(total, len(checked), skipped_top_k, skipped_budget, skipped_entity, skipped_comparative,
                     skipped_topic), found


def _key(ref: ClaimRef) -> str:
    return f"{ref.section_index}:{ref.sentence_index}"


def _add_flag(flags: dict[str, str], key: str, note: str) -> None:
    """Accumulate notes: a claim can be contradicted by more than one sibling, and an auditor
    needs to see all of them (this used to overwrite, leaving only the last one)."""
    existing = flags.get(key)
    if not existing:
        flags[key] = note
    elif note not in existing:
        flags[key] = f"{existing} | {note}"


class SupportScorer:
    """Entailment of a claim by its OWN section's source text (cached)."""

    def __init__(self, index: DocumentIndex, embedder: Embedder, nli: NLIModel, cfg: ContradictionConfig):
        self.index, self.embedder, self.nli, self.cfg = index, embedder, nli, cfg
        self._cache: dict[tuple[int, str], tuple[float, str]] = {}

    def __call__(self, ref: ClaimRef) -> tuple[float, str]:
        key = (ref.section_index, ref.text)
        if key not in self._cache:
            sec = self.index.sections[ref.section_index]
            premises = score_against_source(
                ref.text, self.embedder.encode([ref.text])[0], self.index, self.nli, pool=list(sec.sentence_indices),
                top_k=self.cfg.support_top_k, min_similarity=-1.0, window=self.cfg.support_window)
            self._cache[key] = (premises[0].entailment, premises[0].text) if premises else (0.0, "")
        return self._cache[key]


def resolution_budget(cfg: ContradictionConfig, n_sections: int) -> int:
    """max(max_resolutions, max_resolutions_per_section x sections); 0 = unlimited.

    A fixed 200 gave a 500-page document the same budget as a 5-page one. The floor keeps small
    documents exactly where they were (the per-section term only exceeds 200 above 100 sections).
    """
    if cfg.max_resolutions <= 0:
        return 0
    return max(cfg.max_resolutions, cfg.max_resolutions_per_section * n_sections)


def _apply_resolution_budget(contradictions: list[Contradiction], budget: int) -> int:
    """Mark contradictions beyond the resolution budget as unscored and return how many.

    They stay in the report as flags -- nothing is removed from the summary on their account,
    which is the same outcome as "unresolved". Only the support scoring is skipped.
    """
    if budget <= 0 or len(contradictions) <= budget:
        return 0
    for c in contradictions[budget:]:
        c.resolution = "unscored"
        c.reason = (f"flagged but not resolved: more than {budget} contradictions were flagged (resolution budget "
                    f"= max(max_resolutions, max_resolutions_per_section x sections)), so support scoring stopped "
                    f"after the {budget} most contradictory. Both claims are kept.")
    return len(contradictions) - budget


def _resolve_pairwise(contradictions: list[Contradiction], support: SupportScorer, cfg: ContradictionConfig,
                      rejected: set[str], flags: dict[str, str]) -> None:
    for c in contradictions:
        if c.resolution == "unscored":
            continue
        c.support_a, c.evidence_a = support(c.claim_a)
        c.support_b, c.evidence_b = support(c.claim_b)
        if _key(c.claim_a) in rejected or _key(c.claim_b) in rejected:
            c.resolution = "superseded"
            c.reason = "one of the claims was already rejected by a stronger contradiction"
            continue
        gap = c.support_a - c.support_b
        if abs(gap) < cfg.resolution_margin:
            c.resolution = "unresolved"
            c.reason = (f"support gap {abs(gap):.2f} < margin {cfg.resolution_margin:.2f} "
                        f"(A={c.support_a:.2f}, B={c.support_b:.2f}); both kept and flagged")
            for ref, other in ((c.claim_a, c.claim_b), (c.claim_b, c.claim_a)):
                _add_flag(flags, _key(ref),
                          f"UNRESOLVED contradiction with section {other.section_index}: \"{other.text}\"")
            continue
        c.resolution = "kept_a" if gap > 0 else "kept_b"
        kept, lost = c.kept, c.rejected
        kept_s, lost_s = (c.support_a, c.support_b) if gap > 0 else (c.support_b, c.support_a)
        c.reason = (f"kept section {kept.section_index} claim: entailed by its own source at {kept_s:.2f} vs "
                    f"{lost_s:.2f} for the section {lost.section_index} claim (margin {abs(gap):.2f})")
        rejected.add(_key(lost))
        flags.pop(_key(lost), None)   # a later, decisive rejection replaces an earlier "unresolved" note
        if cfg.action == "flag":
            _add_flag(flags, _key(lost), f"REJECTED in favour of section {kept.section_index}: \"{kept.text}\"")


def _resolve_one_sided(contradictions: list[Contradiction], support: SupportScorer, cfg: ContradictionConfig,
                       rejected: set[str], flags: dict[str, str]) -> None:
    for c in contradictions:
        if c.resolution == "unscored":
            continue
        c.support_a, c.evidence_a = support(c.claim_a)
        c.support_b, c.evidence_b = support(c.claim_b)
        if _key(c.claim_a) in rejected:
            c.resolution = "superseded"
            c.reason = "the summary claim was already rejected"
            continue
        gap = c.support_a - c.support_b
        if gap <= -cfg.resolution_margin:
            c.resolution = "kept_b"
            c.reason = (f"summary claim of section {c.claim_a.section_index} is entailed by its own source at only "
                        f"{c.support_a:.2f}, while the contradicting section {c.claim_b.section_index} source sentence "
                        f"is supported at {c.support_b:.2f}; summary claim rejected")
            rejected.add(_key(c.claim_a))
            flags.pop(_key(c.claim_a), None)
            if cfg.action == "flag":
                _add_flag(flags, _key(c.claim_a),
                          f"REJECTED: contradicted by section {c.claim_b.section_index} source")
        else:
            c.resolution = "unresolved"
            c.reason = (f"both are supported by their own sections (summary claim {c.support_a:.2f}, sibling "
                        f"source {c.support_b:.2f}): the DOCUMENT is inconsistent; summary claim kept and flagged")
            _add_flag(flags, _key(c.claim_a), f"CONTRADICTED by section {c.claim_b.section_index} source "
                                             f"(not in its summary): \"{c.claim_b.text}\"")


def _apply(claims: list[list[str]], rejected: set[str], cfg: ContradictionConfig) -> list[list[str]]:
    return [[text for j, text in enumerate(sentences) if cfg.action == "flag" or f"{s}:{j}" not in rejected]
            for s, sentences in enumerate(claims)]


def resolve_contradictions(contradictions: list[Contradiction], claims: list[list[str]], index: DocumentIndex,
                           embedder: Embedder, nli: NLIModel, cfg: ContradictionConfig,
                           ) -> tuple[list[list[str]], dict[str, str]]:
    """Fill in support scores + resolution on summary-vs-summary contradictions."""
    rejected: set[str] = set()
    flags: dict[str, str] = {}
    _resolve_pairwise(contradictions, SupportScorer(index, embedder, nli, cfg), cfg, rejected, flags)
    return _apply(claims, rejected, cfg), flags


def _pair_reporter(label: str, heartbeat: "Callable[[str], None] | None",
                   every: "tuple[int, float]") -> "Callable[[int, int], None] | None":
    """Progress callback for one Stage 3 check: a line every ``every[0]`` pairs or ``every[1]`` seconds.

    Same shape as the Stage 2 heartbeat, so a long run reads the same way:
      Stage 3a: 1,200/4,697 pairs compared, 0.11 s/pair, elapsed 2.2 min, ETA 6.4 min
    """
    if heartbeat is None:
        return None
    start = time.perf_counter()
    state = {"pairs": 0, "time": start}

    def report(done: int, total: int) -> None:
        now = time.perf_counter()
        if not (done >= total or done - state["pairs"] >= every[0] or now - state["time"] >= every[1]):
            return
        rate = (now - start) / max(1, done)
        heartbeat(f"{label}: {done:,}/{total:,} pairs compared, {rate:.2f} s/pair, "
                  f"elapsed {(now - start) / 60:.1f} min, ETA {rate * (total - done) / 60:.1f} min")
        state["pairs"], state["time"] = done, now

    return report


def check_section_summaries(summaries: list[SectionSummary], index: DocumentIndex, embedder: Embedder,
                            nli: NLIModel, cfg: ContradictionConfig,
                            heartbeat: "Callable[[str], None] | None" = None,
                            heartbeat_every: "tuple[int, float]" = (500, 30.0)) -> ContradictionReport:
    """``heartbeat`` receives progress lines while the pairs are being compared (see _pair_reporter).

    It is reporting only: the pairs compared, the scores and the decisions are the same with and without it.
    """
    claims = [s.sentences for s in summaries]
    if not cfg.enabled:
        return ContradictionReport(0, 0, [], [list(c) for c in claims])

    support = SupportScorer(index, embedder, nli, cfg)
    rejected: set[str] = set()
    flags: dict[str, str] = {}
    say = heartbeat or (lambda _message: None)

    budget = resolution_budget(cfg, len(claims))
    stats, found = detect_contradictions(claims, embedder, nli, cfg,
                                         _pair_reporter("Stage 3a", heartbeat, heartbeat_every),
                                         max(1, heartbeat_every[0]))
    unscored = _apply_resolution_budget(found, budget)
    if found:
        say(f"Stage 3a: {len(found)} pairs flagged from {stats.checked:,} compared; resolving up to {budget}")
    _resolve_pairwise(found, support, cfg, rejected, flags)

    one_sided_stats, one_sided = PairStats(0, 0, 0, 0), []
    if cfg.check_sibling_sources:
        one_sided_stats, one_sided = detect_one_sided(claims, rejected, index, embedder, nli, cfg,
                                                      _pair_reporter("Stage 3b", heartbeat, heartbeat_every),
                                                      max(1, heartbeat_every[0]))
        unscored += _apply_resolution_budget(one_sided, budget)
        if one_sided:
            say(f"Stage 3b: {len(one_sided)} pairs flagged from {one_sided_stats.checked:,} compared; resolving")
        _resolve_one_sided(one_sided, support, cfg, rejected, flags)

    return ContradictionReport(stats.total, stats.checked, found, _apply(claims, rejected, cfg), flags,
                               one_sided=one_sided, one_sided_checked=one_sided_stats.checked,
                               pair_stats=stats, one_sided_stats=one_sided_stats, unscored=unscored,
                               resolution_budget=budget)


def diagnose_source_sections(index: DocumentIndex, embedder: Embedder, nli: NLIModel,
                             cfg: ContradictionConfig) -> ContradictionReport:
    """Diagnostic: run the SAME detector over raw source sentences of each section.

    If a planted contradiction is found here but not in the summaries, the miss is
    the summarizer's (claim dropped/softened), not the detector's. Nothing is
    resolved or removed.
    """
    claims = [sec.sentences for sec in index.sections]
    diag_cfg = dataclasses.replace(cfg, max_pairs=cfg.diagnostic_max_pairs)
    stats, found = detect_contradictions(claims, embedder, nli, diag_cfg)
    if stats.skipped_budget:
        log.warning("source diagnostic hit its comparison ceiling (diagnostic_max_pairs=%d): %d candidate pairs "
                    "were NOT compared; the diagnostic is incomplete", cfg.diagnostic_max_pairs, stats.skipped_budget)
    for c in found:
        c.resolution = "diagnostic"
        c.kind = "diagnostic"
        c.reason = "source-level diagnostic (not resolved)"
    return ContradictionReport(stats.total, stats.checked, found, [list(c) for c in claims], pair_stats=stats)
