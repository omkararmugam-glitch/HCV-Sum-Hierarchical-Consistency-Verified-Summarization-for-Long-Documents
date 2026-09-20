import numpy as np
import pytest

from hcv_sum.segmentation import choose_boundaries, depth_scores, detect_heading, segment_document

from conftest import make_cfg

CATS = [
    "The cat sleeps on the warm mat.",
    "A cat grooms its fur on the mat.",
    "The cat chases a toy mouse near the mat.",
    "Cat fur covers the mat every evening.",
    "The mat is where the cat naps.",
    "The cat purrs on the soft mat.",
]
STOCKS = [
    "Stock markets rallied as bank shares rose.",
    "Bank shares led the stock market gains.",
    "Investors bought bank stock as markets climbed.",
    "Stock market volume in bank shares doubled.",
]


def words(text):
    return len(text.split())


@pytest.mark.parametrize("line,expected", [
    ("## Revenue and Sales", "Revenue and Sales"),
    ("# Title", "Title"),
    ("2. TERM AND RENEWAL", "2. TERM AND RENEWAL"),
    ("3.1 Scope of Services", "3.1 Scope of Services"),
    ("Section 4", "Section 4"),
    ("IV. Results", "IV. Results"),
    ("LIMITATION OF LIABILITY", "LIMITATION OF LIABILITY"),
])
def test_detect_heading_positive(line, expected, cfg):
    assert detect_heading(line, cfg.segmentation) == expected


@pytest.mark.parametrize("line", [
    "The company reported strong results this quarter.",
    "1. The supplier shall deliver the goods.",       # numbered list item that is a sentence
    "OPERATOR:",                                       # transcript speaker label
    "I think we did well",                             # roman-numeral lookalike
    "2026 was a record year for the company and its many long-standing customers overall",  # too long
    "",
    "---",
])
def test_detect_heading_negative(line, cfg):
    assert detect_heading(line, cfg.segmentation) is None


def test_structured_document_uses_headings_and_folds_title(cfg, embedder):
    doc = "# Report\n\n## Cats\n" + " ".join(CATS[:2]) + "\n\n## Stocks\n" + " ".join(STOCKS[:2])
    sections = segment_document(doc, embedder, words, cfg.segmentation)
    assert [s.title for s in sections] == ["Report > Cats", "Stocks"]
    assert all(s.origin == "heading" for s in sections)
    assert sections[1].sentence_offset == 2


def test_too_few_headings_falls_back_to_embeddings(embedder):
    cfg = make_cfg("segmentation.min_segment_sentences=3", "segmentation.window=2")
    doc = "## Only heading\n" + " ".join(CATS + STOCKS)
    sections = segment_document(doc, embedder, words, cfg.segmentation)
    assert all(s.origin == "embedding" for s in sections)


def test_depth_scores_valley():
    sims = np.array([0.9, 0.2, 0.9], dtype=np.float32)
    assert depth_scores(sims)[1] == pytest.approx(1.4)
    assert depth_scores(sims)[0] == pytest.approx(0.0)


def test_choose_boundaries_respects_min_segment(cfg):
    cfg2 = make_cfg("segmentation.min_segment_sentences=3").segmentation
    sims = np.array([0.1, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)   # deepest valley right after sentence 0
    assert 0 not in choose_boundaries(sims, cfg2, n_sentences=7)


def test_unstructured_topic_shift_is_found(embedder):
    cfg = make_cfg("segmentation.min_segment_sentences=3", "segmentation.window=2")
    sections = segment_document(" ".join(CATS + STOCKS), embedder, words, cfg.segmentation)
    assert len(sections) == 2
    assert sections[0].sentences == CATS
    assert sections[1].sentences == STOCKS


def test_single_topic_stays_single(embedder):
    cfg = make_cfg("segmentation.min_segment_sentences=3", "segmentation.window=2")
    sections = segment_document(" ".join(CATS), embedder, words, cfg.segmentation)
    assert len(sections) == 1 and sections[0].origin == "single"


def test_oversize_section_split_at_semantic_boundary_not_midpoint(embedder):
    # 6 cat + 4 stock sentences under one heading; a fixed-length split would cut at 5|5.
    cfg = make_cfg("segmentation.max_section_tokens=60", "segmentation.min_segment_sentences=2",
                   "segmentation.window=2")
    doc = "## A\n" + " ".join(CATS + STOCKS) + "\n## B\nShort closing section here."
    sections = segment_document(doc, embedder, words, cfg.segmentation)
    first_two = sections[:2]
    assert first_two[0].sentences == CATS and first_two[1].sentences == STOCKS
    assert first_two[0].origin == "heading+split"
    assert first_two[0].title == "A (part 1)"
    # global sentence offsets stay contiguous
    assert [s.sentence_offset for s in sections] == [0, 6, 10]
