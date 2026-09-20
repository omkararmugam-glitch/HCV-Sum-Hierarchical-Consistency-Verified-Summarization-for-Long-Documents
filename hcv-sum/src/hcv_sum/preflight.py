"""Pre-flight check: estimate what a document will do to the pipeline BEFORE any model runs.

Needs only sentence splitting and the summarizer's tokenizer (no model inference), so it takes
seconds even on a 500-page document. It answers: how many sections will Stage 1 produce, how big
will they be, will the section ceiling or the summarizer's token budget force emergency splits or
truncation, and roughly how long will the run take.

The section count is a RANGE, because the embedding fallback's boundaries depend on the embeddings:
- low:  every section filled up to ``max_section_tokens`` (the fewest sections the budget allows);
- high: every allowed boundary taken (one section per ``minimum segment`` of sentences).
For documents with headings the heading blocks are known exactly; only the splitting of blocks over
the token budget is estimated.

Runtime is a rough projection from per-unit costs measured on this project's CPU (config ``scale``):
seconds per generated section and milliseconds per NLI pair. It is a planning number, not a promise;
real prose with long sentences ran slower than the synthetic benchmark (FINDINGS.md section 10.5).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable

from .config import Config
from .segmentation import _structured_blocks, minimum_segment_size
from .text_utils import split_sentences


@dataclass
class Preflight:
    sentences: int
    tokens: int
    structured: bool
    heading_blocks: int
    sections_low: int
    sections_high: int
    min_segment_sentences: int          # effective minimum, after the section ceiling
    section_ceiling_binding: bool       # max_sections forced the minimum segment up
    avg_tokens_per_sentence: float
    projected_section_tokens: float     # average tokens per section at the midpoint estimate
    largest_block_tokens: int
    blocks_over_budget: int             # heading blocks / minimum segments that must be split again
    summarizer_input_budget: int
    large_document: bool
    est_stage2_minutes: float
    est_stage3_minutes: float
    est_total_minutes: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def run_preflight(text: str, count_tokens: Callable[[str], int], cfg: Config,
                  model_max_input: int = 1024) -> Preflight:
    seg, summ, con, merge, scale = cfg.segmentation, cfg.summarization, cfg.contradiction, cfg.merging, cfg.scale
    budget = seg.max_section_tokens
    input_budget = min(summ.max_input_tokens, model_max_input)
    warnings: list[str] = []

    blocks, n_headings = _structured_blocks(text, seg) if seg.use_structure else ([], 0)
    structured = seg.use_structure and n_headings >= seg.min_headings
    if structured:
        block_sentences = [split_sentences(body) for _, body in blocks]
        block_sentences = [b for b in block_sentences if b]
    else:
        block_sentences = [split_sentences(text)]
    all_sentences = [s for b in block_sentences for s in b]
    n = len(all_sentences)
    sentence_tokens = [count_tokens(s) for s in all_sentences]
    tokens = sum(sentence_tokens)
    avg_sentence = tokens / n if n else 0.0

    min_seg = minimum_segment_size(n, seg)
    ceiling_binding = seg.max_sections > 0 and min_seg > seg.min_segment_sentences
    if structured:
        k = 0
        low = high = over = largest = 0
        for b in block_sentences:
            t = sum(sentence_tokens[k:k + len(b)])
            k += len(b)
            largest = max(largest, t)
            if t > budget:
                over += 1
                low += math.ceil(t / budget)
                high += max(math.ceil(t / budget), len(b) // max(seg.min_segment_sentences, 1))
            else:
                low += 1
                high += 1
    else:
        largest = int(min_seg * avg_sentence)
        low = max(1, math.ceil(tokens / budget)) if n else 0
        high = max(low, n // max(min_seg, 1))
        over = 0
        if min_seg * avg_sentence > budget:
            over = high   # every minimum-size segment is already over budget
            warnings.append(
                f"section ceiling max_sections={seg.max_sections} forces segments of at least {min_seg} sentences "
                f"(~{min_seg * avg_sentence:.0f} tokens), over max_section_tokens={budget}: nearly every section "
                f"will need an emergency split. Raise segmentation.max_sections.")
    if ceiling_binding:
        warnings.append(f"section ceiling reached: {n} sentences > max_sections={seg.max_sections} x "
                        f"{seg.min_segment_sentences}; minimum segment raised to {min_seg} sentences")
    if structured and over:
        warnings.append(f"{over} of {len(block_sentences)} heading sections exceed max_section_tokens={budget} "
                        f"and will be split at semantic valleys (largest: {largest} tokens)")
    if min(largest, budget) > input_budget - summ.max_context_tokens:
        warnings.append(f"sections of up to {min(largest, budget)} tokens + max_context_tokens="
                        f"{summ.max_context_tokens} exceed the summarizer input budget ({input_budget}); retrieved "
                        f"context is cut first, so the largest sections get little or none")
    mid = (low + high) / 2 if n else 0
    projected = tokens / mid if mid else 0.0
    if projected > input_budget:
        warnings.append(f"projected average section size {projected:.0f} tokens exceeds the summarizer input "
                        f"budget ({input_budget}): section text will be TRUNCATED")

    large = high >= scale.large_document_sections
    generated = min(high, mid if mid else 0)
    stage2 = generated * scale.est_seconds_per_section / 60
    claims = mid * scale.est_claims_per_section
    top_k = con.candidate_top_k or claims
    pairs = min(claims * top_k / 2, con.max_pairs or math.inf) + min(claims * top_k, con.max_pairs or math.inf)
    stage3 = pairs * 2 * scale.est_ms_per_nli_pair / 60000
    merge_calls = (mid / 4) if (large and scale.large_document_merge_mode == "abstractive") or merge.mode == "abstractive" else 0
    total = stage2 + stage3 + merge_calls * scale.est_seconds_per_section / 60
    if large:
        warnings.append(f"large document (up to ~{high} sections >= scale.large_document_sections="
                        f"{scale.large_document_sections}): Stage 2 checkpointing, progress lines and "
                        f"'{scale.large_document_merge_mode}' merging switch on")
    return Preflight(n, tokens, structured, len(block_sentences) if structured else 0, low, high, min_seg,
                     ceiling_binding, round(avg_sentence, 1), round(projected, 1), int(largest), over, input_budget,
                     large, round(stage2, 1), round(stage3, 1), round(total, 1), warnings)


def render_preflight(p: Preflight, name: str = "document") -> str:
    lines = [f"PRE-FLIGHT: {name}",
             f"  {p.sentences} sentences, {p.tokens} tokens (~{p.avg_tokens_per_sentence} per sentence), "
             f"{'heading-structured, ' + str(p.heading_blocks) + ' heading blocks' if p.structured else 'no headings (embedding segmentation)'}",
             f"  projected sections: {p.sections_low}-{p.sections_high} (minimum segment {p.min_segment_sentences} "
             f"sentences), ~{p.projected_section_tokens:.0f} tokens per section on average, summarizer input "
             f"budget {p.summarizer_input_budget}",
             f"  rough runtime on this CPU: Stage 2 ~{p.est_stage2_minutes} min, Stage 3 ~{p.est_stage3_minutes} min, "
             f"total ~{p.est_total_minutes} min (planning estimate; real prose has run slower)"]
    lines += [f"  WARNING: {w}" for w in p.warnings]
    return "\n".join(lines)
