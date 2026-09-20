"""Stage 4 -- merge the Stage-3-corrected section-summaries into one summary.

Because contradictions were resolved BEFORE this stage, the merger never sees
both sides of a detected conflict.

Steps
1. Drop sentences that are near-duplicates (cosine >= ``dedup_similarity``) of an
   earlier kept sentence. Sibling summaries often repeat the same headline fact.
2. ``extractive`` mode: join the remaining sentences in document order.
   ``abstractive`` mode: pack the per-section texts into groups that fit the
   summarizer input, summarize each group, and repeat on the outputs until a
   single group remains (classic hierarchical merging).
3. Re-introduction check: abstractive fusion can re-hallucinate a claim that
   Stage 3 rejected. Each final sentence that is similar to a rejected claim
   AND entailed by it is reported as "resurrected".
4. Between-round check (abstractive only): contradictions BETWEEN the blocks a
   round produced are detected and reported as ``introduced_by_merge``. See
   ``_check_round`` for why these are detected but never resolved.

FIRST-PASS LOGIC / TO STRENGTHEN LATER:
- ``extractive`` is the default because it was measurably better on the samples:
  DistilBART-CNN is lead-biased, so abstractive fusion cited only 2-3 of 5-8
  sections (dropping the later half of each document) while extractive cited
  5/5, 6/6, 5/5 and 7/8. A ``max_group_sections`` fan-in of 2 was worse still.
  The cost of extractive is fluency: it concatenates corrected claims instead of
  rewriting them. A merger that preserves coverage needs an instruction-following
  model or a coverage-aware decoding constraint. Section coverage of the final
  summary should be checked (the evidence panel shows which section each final
  sentence traces to). Extractive mode avoids that loss at the cost of fluency.
- Resurrected claims are reported, not automatically removed.
- No ordering or discourse model; merging relies on document order.
"""

from __future__ import annotations

import numpy as np

from .config import MergingConfig
from .models import ENTAILMENT, Embedder, NLIModel, Summarizer
from .anchored_summarization import clean_generated
from .scoring import DocumentIndex, entailed_by_any
from .text_utils import drop_incomplete_tail, split_sentences
from .types import MergeResult, MergeRound


def deduplicate(sections: list[list[str]], embedder: Embedder, threshold: float,
                ) -> tuple[list[list[str]], list[tuple[str, str, float]]]:
    flat = [s for sec in sections for s in sec]
    if not flat:
        return [list(sec) for sec in sections], []
    emb = embedder.encode(flat)
    kept_ids: list[int] = []
    removed: list[tuple[str, str, float]] = []
    keep_mask = []
    # Kept embeddings live in a preallocated buffer. Indexing emb[kept_ids] instead copied every kept
    # row for every sentence: quadratic memory traffic on a long extractive summary. Same rows, same
    # order, same dot products, so the result is unchanged.
    kept = np.empty_like(emb)
    for i in range(len(flat)):
        if kept_ids:
            sims = kept[: len(kept_ids)] @ emb[i]
            best = int(sims.argmax())
            if sims[best] >= threshold:
                removed.append((flat[i], flat[kept_ids[best]], float(sims[best])))
                keep_mask.append(False)
                continue
        kept[len(kept_ids)] = emb[i]
        kept_ids.append(i)
        keep_mask.append(True)
    out, k = [], 0
    for sec in sections:
        out.append([s for s, keep in zip(sec, keep_mask[k:k + len(sec)]) if keep])
        k += len(sec)
    return out, removed


def central_sentences(sentences: list[str], embedder: Embedder, k: int) -> list[str]:
    """The k sentences most similar on average to the rest of their section, kept in document order."""
    if k <= 0 or len(sentences) <= k:
        return list(sentences)
    emb = embedder.encode(sentences)
    centrality = (emb @ emb.T).mean(axis=1)
    keep = sorted(np.argsort(-centrality, kind="stable")[:k].tolist())
    return [sentences[i] for i in keep]


def guard_coverage(block_texts: list[str], output: str, embedder: Embedder, threshold: float) -> tuple[str, list[str]]:
    """Re-add a block's most central sentence if the merged output no longer covers that block at all.

    Addresses the measured failure of abstractive merging: DistilBART's lead bias summarises the first
    blocks of a group and silently drops the rest. A block counts as covered when some output sentence
    has cosine >= ``threshold`` with some sentence of the block. Costs length, which the sweep in
    SCALING.md measures.
    """
    out_sents = split_sentences(output)
    added: list[str] = []
    out_emb = embedder.encode(out_sents) if out_sents else None
    for block in block_texts:
        sents = split_sentences(block)
        if not sents:
            continue
        emb = embedder.encode(sents)
        if out_emb is not None and float((emb @ out_emb.T).max()) >= threshold:
            continue
        added.append(central_sentences(sents, embedder, 1)[0])
    return (" ".join([output] + added).strip() if added else output), added


def pack_groups_members(blocks: list[str], summarizer: Summarizer, budget: int, max_blocks: int = 0) -> list[list[str]]:
    """Like pack_groups, but returns the member blocks of each group."""
    groups: list[list[str]] = []
    current: list[str] = []
    for block in blocks:
        candidate = " ".join(current + [block])
        if current and (summarizer.count_tokens(candidate) > budget or 0 < max_blocks <= len(current)):
            groups.append(current)
            current = [block]
        else:
            current.append(block)
    if current:
        groups.append(current)
    return groups


def retrieve_merge_context(member_blocks: list[str], index: DocumentIndex, embedder: Embedder,
                           cfg: MergingConfig, count_tokens) -> list[str]:
    """Source sentences to give the merger alongside the blocks it fuses (after Ou & Lapata, 2025).

    Retrieval is PER BLOCK: each member block contributes its own ``context_top_k_per_block`` most
    similar source sentences (max cosine to any of the block's sentences, over the whole document),
    so every block being merged -- not only the first, which DistilBART already favours -- is anchored
    in its source. Same limits as Stage 3's candidate selection: a similarity floor
    (``context_min_similarity``) and a per-row top-k; the score matrix per block is
    (source sentences x block sentences), linear in document size. Sentences the block already states
    near-verbatim (cosine >= 0.95) are skipped, since repeating them adds no information. The result
    is cut to ``context_max_tokens`` in block order, whole sentences only.
    """
    chosen: list[str] = []
    seen: set[int] = set()
    for block in member_blocks:
        sents = split_sentences(block)
        if not sents:
            continue
        scores = index.embeddings @ embedder.encode(sents).T          # (n_source, n_block_sentences)
        best = scores.max(axis=1)
        taken = 0
        for gid in np.argsort(-best, kind="stable"):
            if taken >= cfg.context_top_k_per_block or best[gid] < cfg.context_min_similarity:
                break
            if best[gid] >= 0.95 or int(gid) in seen:
                continue
            seen.add(int(gid))
            chosen.append(index.sentences[int(gid)])
            taken += 1
    out, used = [], 0
    for sentence in chosen:
        n = count_tokens(sentence)
        if used + n > cfg.context_max_tokens:
            continue
        out.append(sentence)
        used += n
    return out


def render_merge_input(group: str, context: list[str], cfg: MergingConfig) -> str:
    """The text handed to the merge model for one group.

    ``merge_prompt: none`` -- the blocks, then any context after a blank line (for DistilBART-style
    models, which cannot be instructed). ``instruct`` -- an explicit instruction naming the summaries and,
    when context is present, what the context is for (for an instruction-tuned merge model).
    """
    if cfg.merge_prompt == "instruct":
        note = cfg.merge_instruct_context if context else ""
        block = f"\n\nCONTEXT: {' '.join(context)}" if context else ""
        return cfg.merge_instruct_prompt.format(context_note=note, summaries=group, context_block=block)
    return f"{group}\n\n{' '.join(context)}" if context else group


def pack_groups(blocks: list[str], summarizer: Summarizer, budget: int, max_blocks: int = 0) -> list[str]:
    """Greedily pack consecutive blocks into groups under the token budget (and max_blocks, if > 0)."""
    groups: list[str] = []
    current: list[str] = []
    for block in blocks:
        candidate = " ".join(current + [block])
        if current and (summarizer.count_tokens(candidate) > budget or 0 < max_blocks <= len(current)):
            groups.append(" ".join(current))
            current = [block]
        else:
            current.append(block)
    if current:
        groups.append(" ".join(current))
    return groups


def _lengths(n_tokens: int, cfg: MergingConfig) -> tuple[int, int]:
    max_new = min(cfg.max_new_tokens, max(cfg.min_new_tokens + 10, int(n_tokens * cfg.max_length_ratio)))
    min_new = min(max_new - 10, max(cfg.min_new_tokens, int(n_tokens * cfg.min_length_ratio)))
    return max(min_new, 1), max_new


def _check_round(inputs: list[str], outputs: list[str], embedder: Embedder, nli: NLIModel,
                 contradiction_cfg) -> list:
    """Detect contradictions BETWEEN the blocks a merge round produced.

    Why here and not only at the leaves: Stage 3 compares every pair of leaf claims before merging,
    so a conflict that exists in the source is already resolved by then. What it cannot see is a
    conflict the MERGER invents -- abstractive fusion rewrites text and can state something that
    contradicts another branch. Those are detected, never resolved: resolution compares a claim with
    its own section's source, and a merged block no longer belongs to a single section, so there is
    no honest support score to decide with. They are reported for a human instead.
    """
    if contradiction_cfg is None or not contradiction_cfg.enabled or len(outputs) < 2:
        return []
    from .contradiction import detect_contradictions   # local import: contradiction.py is upstream

    claims = [split_sentences(o) for o in outputs]
    _, found = detect_contradictions(claims, embedder, nli, contradiction_cfg)
    previous = " ".join(inputs)
    fresh = []
    for c in found:
        # Only report it if neither side is a near-verbatim copy of the round's input: if the text
        # was already there, the leaf-level check owned it.
        if c.claim_a.text not in previous and c.claim_b.text not in previous:
            c.kind = "introduced_by_merge"
            c.resolution = "unresolved"
            c.reason = ("introduced by the merge round: neither claim is a verbatim leaf claim, so no "
                        "per-section support score exists to resolve it; reported for review")
            fresh.append(c)
    return fresh


def merge_summaries(corrected_sections: list[list[str]], rejected_claims: list[str], summarizer: Summarizer,
                    embedder: Embedder, nli: NLIModel, cfg: MergingConfig,
                    contradiction_cfg=None, mode: str | None = None,
                    index: DocumentIndex | None = None) -> MergeResult:
    """``mode`` overrides ``cfg.mode`` (the pipeline passes the size-based choice, config ``scale``).

    ``index`` (the document's source sentences) is required when ``cfg.context_anchoring`` is on:
    abstractive merging then gives the merger relevant source sentences alongside the blocks it
    fuses. Extractive merging makes no model call, so the setting does not apply to it.
    """
    mode = mode or cfg.mode
    anchored = cfg.context_anchoring and mode == "abstractive"
    if anchored and index is None:
        raise ValueError("merging.context_anchoring needs the document index (merge_summaries(..., index=...))")
    input_sentences = [s for sec in corrected_sections for s in sec]
    deduped, removed = deduplicate(corrected_sections, embedder, cfg.dedup_similarity)
    blocks = [" ".join(sec) for sec in deduped if sec]
    rounds: list[MergeRound] = []

    if mode == "extractive" and cfg.extractive_max_per_section > 0:
        # Length-capped extractive: the most central sentences of each section, so output length is
        # bounded by sections x cap instead of growing with every section's full summary.
        deduped = [central_sentences(sec, embedder, cfg.extractive_max_per_section) for sec in deduped]
    kept_sentences = [s for sec in deduped for s in sec]
    if mode == "extractive" or not blocks:
        # Keep the sentence list as-is. Re-splitting the joined text would merge adjacent
        # sentences that have no terminal punctuation (headings, bullet fragments, transcripts).
        return MergeResult(mode, input_sentences, removed, [], " ".join(kept_sentences),
                           kept_sentences, [])
    else:
        # Recursive batched merge: at most `max_group_sections` blocks are fused per summarizer
        # call, and the outputs are merged again, round after round, until one block remains.
        # Merging everything in a single pass instead would hand a long document's entire set of
        # section summaries to one generate() call, where DistilBART's lead bias drops the tail.
        budget = min(cfg.max_input_tokens, summarizer.model_max_input) - 8
        if anchored:
            budget -= cfg.context_max_tokens + 2      # room for the context; only when the feature is on
        if cfg.merge_prompt == "instruct":             # room for the instruction text itself
            budget -= summarizer.count_tokens(render_merge_input("", ["x"] if anchored else [], cfg)) + 2
        for round_number in range(1, cfg.max_rounds + 1):
            members = pack_groups_members(blocks, summarizer, budget, cfg.max_group_sections)
            groups = [" ".join(m) for m in members]
            outputs, contexts = [], []
            for group, member_blocks in zip(groups, members):
                # Output length follows the blocks being merged, not the added context.
                min_new, max_new = _lengths(summarizer.count_tokens(group), cfg)
                context = (retrieve_merge_context(member_blocks, index, embedder, cfg, summarizer.count_tokens)
                           if anchored else [])
                contexts.append(context)
                model_input = render_merge_input(group, context, cfg)
                output = clean_generated(summarizer.summarize(model_input, min_new_tokens=min_new,
                                                              max_new_tokens=max_new))
                if cfg.coverage_guard and len(member_blocks) > 1:
                    output, _ = guard_coverage(member_blocks, output, embedder, cfg.coverage_guard_similarity)
                outputs.append(output)
            introduced = (_check_round(blocks, outputs, embedder, nli, contradiction_cfg)
                          if cfg.check_between_rounds and len(outputs) > 1 else [])
            rounds.append(MergeRound(groups, outputs, round_number, introduced, contexts if anchored else []))
            blocks = outputs
            if len(groups) == 1:
                break
        summary = " ".join(blocks)

    sentences = split_sentences(summary)
    if mode == "abstractive":
        sentences, _ = drop_incomplete_tail(sentences)
        summary = " ".join(sentences)
    # Extractive mode inherits Stage 2's sentences, which are already complete by construction.
    resurrected: list[tuple[str, str, float]] = []
    surviving = [s for sec in deduped for s in sec]
    if rejected_claims and surviving:
        # A rejected claim whose content ALSO survives elsewhere (e.g. a leaked duplicate was rejected
        # but the original was kept) is not "resurrected" when it shows up again. Decided by NLI.
        explained = entailed_by_any(rejected_claims, surviving, embedder, nli,
                                    min_similarity=cfg.resurrection_similarity,
                                    min_entailment=cfg.resurrection_entailment)
        rejected_claims = [c for c, e in zip(rejected_claims, explained) if not e]
    if rejected_claims and sentences and mode == "abstractive":
        sims = embedder.encode(sentences) @ embedder.encode(rejected_claims).T
        pairs = [(i, j) for i in range(len(sentences)) for j in range(len(rejected_claims))
                 if sims[i, j] >= cfg.resurrection_similarity]
        if pairs:
            probs = nli.predict([(rejected_claims[j], sentences[i]) for i, j in pairs])
            resurrected = [(sentences[i], rejected_claims[j], float(p[ENTAILMENT]))
                           for (i, j), p in zip(pairs, probs) if p[ENTAILMENT] >= cfg.resurrection_entailment]

    return MergeResult(mode, input_sentences, removed, rounds, summary, sentences, resurrected)
