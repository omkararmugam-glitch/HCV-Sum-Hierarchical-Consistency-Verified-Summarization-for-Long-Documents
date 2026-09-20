"""Stage 5 -- sentence-level provenance tagging (the evidence panel).

For every final summary sentence:
1. Embedding filter: top-k source sentences with cosine >= ``min_candidate_similarity``.
   If none pass, the sentence is "unsupported" (nothing in the source is even on topic).
2. NLI confirmation: each candidate AND each window of up to ``window`` consecutive
   same-section sentences containing it is used as a premise. Similarity alone is
   not trusted: "revenue fell because of X" and "revenue fell, unrelated to X" are
   highly similar.
3. Status from the best entailment: supported / weakly_supported / unsupported. A
   sentence that Stage 3 left "unresolved" also gets a DISPUTED flag: it is faithful
   to its own section, yet another section contradicts it.
   A ``contradicted_by_source`` flag is added when a candidate premise
   contradicts the sentence strongly and nothing entails it.
4. Lexical backstop: tokens of the sentence that occur nowhere in the source are
   flagged. Entailment alone missed a summarizer corruption ("twenty-four (24)
   months" -> "twenty-24 months") that this catches.
5. Stage 3 linkage: if the sentence carries a claim that Stage 3 kept over a
   rival (or left unresolved, or flagged as rejected), the decision and its
   reason are attached so the audit trail survives abstractive merging.

FIRST-PASS LOGIC / TO STRENGTHEN LATER:
- Linking a rewritten final sentence back to a Stage 3 claim uses embedding
  similarity only; heavy paraphrase can break the link.
- Windows never cross section boundaries, so a summary sentence fusing facts
  from two sections is scored against each half separately and may come out
  "weakly_supported" even though every fact is in the source.
- Thresholds are uncalibrated. Calibrating them on a small labelled set (e.g.
  AggreFact / SummaC benchmark) is the obvious next step.
"""

from __future__ import annotations

import numpy as np

from .config import ProvenanceConfig
from .models import Embedder, NLIModel
from .scoring import DocumentIndex, entailed_by_any, score_against_source
from .text_utils import novel_tokens, surface_tokens
from .types import Citation, ContradictionReport, MergeResult, ProvenanceRecord


def _citation(premise, index: DocumentIndex) -> Citation:
    sec = index.sections[int(index.section_of[premise.sentence_ids[0]])]
    return Citation(list(premise.sentence_ids), sec.index, sec.title, premise.text,
                    premise.similarity, premise.entailment, premise.contradiction)


def _claim_embeddings(report: ContradictionReport | None, embedder: Embedder) -> dict[str, "np.ndarray"]:
    """Embed every claim text once (was: re-encoded for every final sentence)."""
    if report is None:
        return {}
    texts = sorted({c.text for x in report.all_contradictions for c in (x.claim_a, x.claim_b)})
    return dict(zip(texts, embedder.encode(texts))) if texts else {}


def _closer_side(contradiction, sent_emb, claim_emb: dict, cfg: ProvenanceConfig):
    """Which side of a flagged pair this final sentence came from: the more similar of the two, or (None, None).

    Both sides of a contradiction say nearly the same thing -- they differ by a negation, a changed number or
    a reversed direction, all of which sentence embeddings largely ignore. So a final sentence taken from one
    side is usually above ``link_similarity`` to BOTH, and reporting every side above the threshold gave one
    conflict two notes under the same sentence: identical prefix, identical reason and score, with the quoted
    claims differing by a word, the second naming the sentence's own claim as the thing contradicting it.
    Returning the best match means one conflict is reported once, from the perspective of the sentence at hand.
    """
    ranked = sorted(((float(claim_emb[this.text] @ sent_emb), this, other)
                     for this, other in ((contradiction.claim_a, contradiction.claim_b),
                                         (contradiction.claim_b, contradiction.claim_a))),
                    key=lambda item: -item[0])
    similarity, this, other = ranked[0]
    return (this, other) if similarity >= cfg.link_similarity else (None, None)


def _stage3_notes(sentence: str, sent_emb, report: ContradictionReport | None, embedder: Embedder, nli: NLIModel,
                  cfg: ProvenanceConfig, claim_emb: dict) -> list[str]:
    if report is None or not report.all_contradictions:
        return []
    # (key, note) per conflict. The key identifies the underlying conflict as this sentence sees it -- the
    # kind of decision plus the opposing claim -- so one real conflict is reported once however many
    # Contradiction objects carry it, while genuinely different opposing claims stay separate. See the
    # grouping note at the end of the function.
    notes: list[tuple[tuple, str]] = []
    surviving = [s for sec in report.corrected_sentences for s in sec]
    for c in report.all_contradictions:
        if c.resolution in ("kept_a", "kept_b"):
            kept, lost = c.kept, c.rejected
            if float(claim_emb[kept.text] @ sent_emb) >= cfg.link_similarity:
                notes.append((("kept", lost.text),
                              f"Stage 3 kept this claim (section {kept.section_index}) over the contradicting "
                              f"claim \"{lost.text}\" (section {lost.section_index}). Reason: {c.reason}."))
            elif (float(claim_emb[lost.text] @ sent_emb) >= cfg.link_similarity
                  and not entailed_by_any([sentence], surviving, embedder, nli, min_similarity=cfg.link_similarity,
                                          min_entailment=cfg.supported_entailment)[0]):
                # Only warn if no surviving (post-Stage-3) claim entails the sentence.
                notes.append((("rejected", lost.text),
                              f"WARNING: resembles the claim Stage 3 REJECTED: \"{lost.text}\" "
                              f"(section {lost.section_index}). Reason: {c.reason}."))
        elif c.resolution == "unscored":
            if float(claim_emb[c.claim_a.text] @ sent_emb) >= cfg.link_similarity:
                notes.append((("unscored", c.claim_b.text),
                              f"Stage 3 flagged a contradiction with \"{c.claim_b.text}\" (section "
                              f"{c.claim_b.section_index}) but did not resolve it: {c.reason}"))
        elif c.resolution == "possibly_superseded":
            _this, other = _closer_side(c, sent_emb, claim_emb, cfg)
            if other is not None:
                notes.append((("possibly_superseded", other.text),
                              f"Stage 3 POSSIBLY SUPERSEDED: differs from \"{other.text}\" (section "
                              f"{other.section_index}), but the documents cover different periods. {c.reason}."))
        elif c.resolution == "unresolved" and c.kind == "one_sided":
            if float(claim_emb[c.claim_a.text] @ sent_emb) >= cfg.link_similarity:
                notes.append((("one_sided", c.claim_b.text),
                              f"Stage 3 UNRESOLVED (one-sided): this claim is contradicted by section "
                              f"{c.claim_b.section_index} of the source, which its section summary dropped: "
                              f"\"{c.claim_b.text}\". {c.reason}."))
        elif c.resolution == "unresolved":
            _this, other = _closer_side(c, sent_emb, claim_emb, cfg)
            if other is not None:
                notes.append((("unresolved", other.text),
                              f"Stage 3 UNRESOLVED: contradicts \"{other.text}\" "
                              f"(section {other.section_index}). {c.reason}."))
    # One note per underlying conflict, in the order they were found.
    #
    # Why one conflict arrives several times: a claim is a (section, sentence, text) reference, so a claim
    # TEXT that survives into several sections' summaries is several ClaimRefs, and a single real conflict is
    # then flagged as one Contradiction per section holding it. Those objects are NOT interchangeable -- their
    # reasons quote per-section support scores, so the rendered notes can differ in their numbers -- but from
    # this sentence's point of view they all say the same thing: it is contradicted by that one opposing claim.
    # Grouping therefore keys on the decision and the opposing claim's TEXT, not on the rendered note, and
    # keeps the first note of each group (its section reference is the first one found).
    #
    # What is deliberately NOT collapsed: two different opposing claims give two different keys, so a sentence
    # contradicted by several distinct claims still lists each of them. A pair whose two sides both resemble
    # the sentence is handled earlier, by _closer_side.
    grouped: dict[tuple, str] = {}
    for key, note in notes:
        grouped.setdefault(key, note)
    return list(grouped.values())


def tag_provenance(merge: MergeResult, index: DocumentIndex, embedder: Embedder, nli: NLIModel,
                   cfg: ProvenanceConfig, report: ContradictionReport | None = None) -> list[ProvenanceRecord]:
    records: list[ProvenanceRecord] = []
    if not merge.sentences:
        return records
    embeddings = embedder.encode(merge.sentences)
    vocabulary = {t for s in index.sentences for t in surface_tokens(s)}
    claim_emb = _claim_embeddings(report, embedder)
    resurrected = {s: claim for s, claim, _ in merge.resurrected}

    for i, (sentence, emb) in enumerate(zip(merge.sentences, embeddings)):
        best_similarity = float((index.embeddings @ emb).max()) if len(index.sentences) else 0.0
        premises = score_against_source(sentence, emb, index, nli, pool=None, top_k=cfg.top_k,
                                        min_similarity=cfg.min_candidate_similarity, window=cfg.window)
        flags: list[str] = []
        if not premises:
            status, entailment, citation = "unsupported", 0.0, None
            flags.append(f"no source sentence above similarity {cfg.min_candidate_similarity:.2f} "
                         f"(best {best_similarity:.2f})")
        else:
            best = premises[0]
            entailment, citation = best.entailment, _citation(best, index)
            if entailment >= cfg.supported_entailment:
                status = "supported"
            elif entailment >= cfg.weak_entailment:
                status = "weakly_supported"
            else:
                status = "unsupported"
            worst = max(premises, key=lambda p: p.contradiction)
            if status != "supported" and worst.contradiction >= cfg.source_contradiction_flag:
                flags.append(f"contradicted_by_source ({worst.contradiction:.2f}): \"{worst.text}\"")
        if cfg.flag_novel_tokens:
            novel = novel_tokens(sentence, vocabulary, cfg.novel_token_min_length)
            if novel:
                # NLI entailment is generous with corrupted numbers ("twenty-24 months" scored 0.87),
                # so a purely lexical check backs it up.
                flags.append(f"not in source text: {', '.join(repr(t) for t in novel)}")
        if sentence in resurrected:
            flags.append(f"resurrected a claim rejected in Stage 3: \"{resurrected[sentence]}\"")

        notes = _stage3_notes(sentence, emb, report, embedder, nli, cfg, claim_emb)
        if any(n.startswith("Stage 3 UNRESOLVED") for n in notes):
            # "supported" alone would mislead: the sentence is faithful to its own section but
            # another part of the document contradicts it.
            flags.append("DISPUTED: another section of the source contradicts this sentence "
                         "(see the Stage 3 note below)")

        records.append(ProvenanceRecord(
            index=i, sentence=sentence, status=status, similarity=best_similarity, entailment=entailment,
            citation=citation, candidates=[_citation(p, index) for p in premises[:5]], flags=flags,
            stage3_notes=notes))
    return records
