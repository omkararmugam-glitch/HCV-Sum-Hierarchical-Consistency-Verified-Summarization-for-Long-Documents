"""Stage 1 -- structure-aware segmentation.

1. If the document has explicit structure (markdown headings, numbered section
   titles, ALL-CAPS header lines), split on it. Author structure is treated as
   authoritative.
2. Otherwise fall back to TextTiling-style topic segmentation over sentence
   embeddings: compare the averaged embeddings of the ``window`` sentences on
   each side of every gap, compute a depth score for each valley in that
   similarity curve, and cut at valleys that are deep relative to the document.
3. Any section that is too long for the summarizer is split again at its
   deepest semantic valley (never at a fixed token offset).

FIRST-PASS LOGIC / TO STRENGTHEN LATER:
- Heading detection is regex heuristics. It will miss bold-only headings and
  can mistake short ALL-CAPS lines (e.g. table headers) for headings.
- The fallback ignores paragraph and speaker-turn boundaries, which are strong
  cues in transcripts; snapping boundaries to them is an obvious improvement.
- The depth cutoff (mean + k*std) is relative to each document, so it always
  finds *some* boundaries in a long enough document even when the topic never
  changes. A supervised segmenter or an absolute similarity floor would fix it.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np

from .config import SegmentationConfig
from .models import Embedder
from .text_utils import split_sentences
from .types import Section

_MARKDOWN = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*#*\s*$")
_NUMBERED = re.compile(
    r"^\s*(?:(?:section|article|part|chapter)\s+)?"
    r"(?:\d{1,3}(?:\.\d{1,3})*[.):]?|[IVXLC]{1,6}[.):])\s+(?P<title>[A-Z].*?)\s*$"
)
_KEYWORD_ONLY = re.compile(r"^\s*(?:section|article|part|chapter)\s+(?:\d{1,3}|[IVXLC]{1,6})\s*[.:]?\s*$",
                           re.IGNORECASE)
_TERMINAL_PUNCT = re.compile(r"[.;,:!?]$")


def detect_heading(line: str, cfg: SegmentationConfig) -> str | None:
    """Return the heading title if ``line`` looks like a section heading, else None."""
    stripped = line.strip()
    if not stripped or len(stripped.split()) > cfg.max_heading_words:
        return None
    if cfg.detect_markdown and (m := _MARKDOWN.match(stripped)):
        return m.group("title").strip()
    if cfg.detect_numbered:
        if _KEYWORD_ONLY.match(stripped):
            return stripped.rstrip(".:")
        m = _NUMBERED.match(stripped)
        if m and not _TERMINAL_PUNCT.search(m.group("title")):
            return stripped
    if cfg.detect_allcaps:
        letters = [c for c in stripped if c.isalpha()]
        # Lines ending in ':' are excluded: in transcripts those are speaker labels, not topics.
        if len(letters) >= 3 and all(c.isupper() for c in letters) and not _TERMINAL_PUNCT.search(stripped):
            return stripped
    return None


def _structured_blocks(text: str, cfg: SegmentationConfig) -> tuple[list[tuple[str | None, str]], int]:
    blocks: list[tuple[str | None, list[str]]] = [(None, [])]
    n_headings = 0
    for line in text.splitlines():
        title = detect_heading(line, cfg)
        if title is None:
            blocks[-1][1].append(line)
            continue
        n_headings += 1
        prev_title, prev_body = blocks[-1]
        if not "".join(prev_body).strip() and prev_title is not None:
            # A heading with no body (e.g. a document title directly followed by a
            # section heading) is folded into the next heading's title.
            blocks[-1] = (f"{prev_title} > {title}", [])
        else:
            blocks.append((title, []))
    return [(t, "\n".join(b)) for t, b in blocks], n_headings


def gap_similarities(embeddings: np.ndarray, window: int) -> np.ndarray:
    """Cosine similarity across each gap i|i+1 between the mean of ``window`` sentences on each side."""
    n = len(embeddings)
    sims = np.zeros(max(n - 1, 0), dtype=np.float32)
    for g in range(n - 1):
        left = embeddings[max(0, g - window + 1): g + 1].mean(axis=0)
        right = embeddings[g + 1: min(n, g + 1 + window)].mean(axis=0)
        sims[g] = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))
    return sims


def depth_scores(sims: np.ndarray) -> np.ndarray:
    """TextTiling depth: how far a gap's similarity sits below the nearest peaks on both sides."""
    depths = np.zeros_like(sims)
    for i, s in enumerate(sims):
        left = s
        for j in range(i - 1, -1, -1):
            if sims[j] < left:
                break
            left = sims[j]
        right = s
        for j in range(i + 1, len(sims)):
            if sims[j] < right:
                break
            right = sims[j]
        depths[i] = (left - s) + (right - s)
    return depths


def minimum_segment_size(n_sentences: int, cfg: SegmentationConfig) -> int:
    """Grow the minimum segment length so a very long document cannot explode into micro-sections."""
    if cfg.max_sections > 0:
        return max(cfg.min_segment_sentences, -(-n_sentences // cfg.max_sections))
    return cfg.min_segment_sentences


def choose_boundaries(sims: np.ndarray, cfg: SegmentationConfig, n_sentences: int) -> list[int]:
    """Return sorted gap indices g (boundary between sentence g and g+1)."""
    min_segment = minimum_segment_size(n_sentences, cfg)
    if n_sentences < 2 * min_segment or len(sims) == 0:
        return []
    depths = depth_scores(sims)
    cutoff = max(cfg.min_depth, float(depths.mean() + cfg.depth_std_factor * depths.std()))
    chosen: list[int] = []
    for g in sorted(range(len(sims)), key=lambda i: -depths[i]):
        if depths[g] < cutoff:
            break
        cuts = sorted(chosen + [g])
        edges = [0] + [c + 1 for c in cuts] + [n_sentences]
        if all(b - a >= min_segment for a, b in zip(edges, edges[1:])):
            chosen.append(g)
    return sorted(chosen)


def _split_on_gaps(items: list, gaps: list[int]) -> list[list]:
    edges = [0] + [g + 1 for g in gaps] + [len(items)]
    return [items[a:b] for a, b in zip(edges, edges[1:])]


def _split_oversize(sentences: list[str], embeddings: np.ndarray, count_tokens: Callable[[str], int],
                    cfg: SegmentationConfig) -> list[list[str]]:
    """Recursively split at the deepest semantic valley until every part fits the token budget."""
    if len(sentences) < 2 or count_tokens(" ".join(sentences)) <= cfg.max_section_tokens:
        return [sentences]
    sims = gap_similarities(embeddings, cfg.window)
    depths = depth_scores(sims)
    min_part = min(cfg.min_segment_sentences, len(sentences) // 2)
    valid = [g for g in range(len(sims)) if g + 1 >= max(min_part, 1) and len(sentences) - (g + 1) >= max(min_part, 1)]
    # Deepest valley; ties (e.g. a flat curve) broken by lowest raw similarity.
    g = max(valid, key=lambda i: (depths[i], -sims[i]))
    return (_split_oversize(sentences[: g + 1], embeddings[: g + 1], count_tokens, cfg)
            + _split_oversize(sentences[g + 1:], embeddings[g + 1:], count_tokens, cfg))


def segment_document(text: str, embedder: Embedder, count_tokens: Callable[[str], int],
                     cfg: SegmentationConfig) -> list[Section]:
    raw: list[tuple[str | None, list[str], str]] = []   # (title, sentences, origin)

    blocks, n_headings = _structured_blocks(text, cfg) if cfg.use_structure else ([], 0)
    if cfg.use_structure and n_headings >= cfg.min_headings:
        for title, body in blocks:
            sentences = split_sentences(body)
            if sentences:
                raw.append((title, sentences, "heading"))
    else:
        sentences = split_sentences(text)
        if not sentences:
            return []
        gaps = choose_boundaries(gap_similarities(embedder.encode(sentences), cfg.window), cfg, len(sentences))
        origin = "embedding" if gaps else "single"
        raw.extend((None, part, origin) for part in _split_on_gaps(sentences, gaps))

    sections: list[Section] = []
    offset = 0
    for title, sentences, origin in raw:
        parts = [sentences]
        if count_tokens(" ".join(sentences)) > cfg.max_section_tokens:
            parts = _split_oversize(sentences, embedder.encode(sentences), count_tokens, cfg)
        for k, part in enumerate(parts):
            part_title = title if len(parts) == 1 or title is None else f"{title} (part {k + 1})"
            sections.append(Section(len(sections), part_title, part, offset,
                                    origin if len(parts) == 1 else f"{origin}+split"))
            offset += len(part)
    return sections
