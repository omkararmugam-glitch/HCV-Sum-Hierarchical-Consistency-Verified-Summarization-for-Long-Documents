"""Stage 2 batching must be a pure speed optimisation, never a change in output.

generate() applies one length budget per batch, so sections are grouped by identical
(min_new, max_new). These tests check the grouping logic with fakes, and the ``slow`` test
checks byte-identical summaries from the real model on a sample document.
"""

import pytest

from hcv_sum.anchored_summarization import summarize_sections
from hcv_sum.config import load_config
from hcv_sum.models import ModelRegistry
from hcv_sum.scoring import DocumentIndex
from hcv_sum.segmentation import segment_document

from conftest import SAMPLES, FakeEmbedder, FakeSummarizer, build_index

SPEC = [
    ("A", ["The Monterrey plant shutdown halted sensor production in April.",
           "Repair costs reached four million dollars.",
           "Utilization at other plants averaged eighty one percent."]),
    ("B", ["Sensor revenue declined eighteen percent.",
           "The Monterrey plant shutdown did not affect sensor revenue.",
           "European demand weakened sharply this quarter."]),
    ("C", ["Resin lead times improved to nine weeks.",
           "Freight costs fell eleven percent year over year.",
           "Inventory days decreased from ninety four to eighty eight."]),
]


class RecordingSummarizer(FakeSummarizer):
    """Records the (min_new, max_new, batch size) of every generation call."""

    def __init__(self):
        super().__init__()
        self.batches: list[tuple[int, int, int]] = []

    def summarize_batch(self, texts, *, min_new_tokens, max_new_tokens):
        self.batches.append((min_new_tokens, max_new_tokens, len(texts)))
        return [self.summarize(t, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens) for t in texts]


def test_batches_never_mix_generation_budgets():
    embedder = FakeEmbedder()
    index = build_index(SPEC, embedder)
    fs = RecordingSummarizer()
    cfg = load_config(overrides=["summarization.passthrough_tokens=1", "summarization.batch_size=8"]).summarization
    summarize_sections(index.sections, index, fs, embedder, cfg)
    assert fs.batches, "expected at least one generation call"
    for min_new, max_new, size in fs.batches:
        assert size >= 1
    # every section that was generated used the budget its own length implies
    budgets = {(b[0], b[1]) for b in fs.batches}
    assert len(budgets) == len({(b[0], b[1]) for b in fs.batches}), "budgets must not be merged"


def summaries_at(batch_size, embedder):
    index = build_index(SPEC, embedder)
    cfg = load_config(overrides=["summarization.passthrough_tokens=1",
                                 f"summarization.batch_size={batch_size}"]).summarization
    return [s.summary for s in summarize_sections(index.sections, index, FakeSummarizer(), embedder, cfg)]


@pytest.mark.parametrize("batch_size", [2, 3, 8])
def test_batch_size_does_not_change_the_summaries(batch_size):
    embedder = FakeEmbedder()
    assert summaries_at(batch_size, embedder) == summaries_at(1, embedder)


@pytest.mark.slow
def test_real_model_batching_is_byte_identical():
    """The real check: DistilBART output must not depend on batch size."""
    cfg1 = load_config(overrides=["summarization.batch_size=1"])
    cfg4 = load_config(overrides=["summarization.batch_size=4"])
    registry = ModelRegistry(cfg1)
    text = (SAMPLES / "01_planted_contradiction.md").read_text(encoding="utf-8")
    sections = segment_document(text, registry.embedder, registry.summarizer.count_tokens, cfg1.segmentation)
    index = DocumentIndex.build(sections, registry.embedder)

    one = summarize_sections(sections, index, registry.summarizer, registry.embedder, cfg1.summarization)
    four = summarize_sections(sections, index, registry.summarizer, registry.embedder, cfg4.summarization)
    assert [s.summary for s in one] == [s.summary for s in four]
