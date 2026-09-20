"""Context-anchored abstractive merging (merging.context_anchoring, after Ou & Lapata 2025; FINDINGS 14)."""

import pytest

from hcv_sum.merging import merge_summaries, retrieve_merge_context
from hcv_sum.models import ModelRegistry
from hcv_sum.pipeline import HCVSumPipeline

from conftest import FakeEmbedder, FakeNLI, FakeSummarizer, build_index, make_cfg

SPEC = [
    ("Plant", ["The Monterrey plant closed for three weeks after a fire.",
               "Transformer repairs at the Monterrey plant cost four million dollars.",
               "Monterrey plant output resumed in May."]),
    ("Freight", ["Freight costs from Osaka rose eleven percent.",
                 "Osaka freight contracts were renegotiated in June.",
                 "Shipping delays from Osaka eased by August."]),
    ("Hiring", ["Engineering headcount in Lyon grew by forty people.",
                "Hiring focused on firmware engineers in Lyon.",
                "Lyon hiring will slow next quarter."]),
]
# Stage 2-style section summaries: the first sentence of each section only.
SUMMARIES = [[sentences[0]] for _, sentences in SPEC]
ON = ("merging.context_anchoring=true", "merging.max_group_sections=3", "merging.check_between_rounds=false")


def merge(*overrides, mode="abstractive", index=True, summarizer=None):
    embedder = FakeEmbedder()
    idx = build_index(SPEC, embedder) if index else None
    summarizer = summarizer or FakeSummarizer()
    result = merge_summaries(SUMMARIES, [], summarizer, embedder, FakeNLI(), make_cfg(*overrides).merging,
                             mode=mode, index=idx)
    return result, summarizer


def test_off_by_default_and_identical_to_no_index(cfg):
    assert cfg.merging.context_anchoring is False
    with_index, s1 = merge("merging.max_group_sections=3")
    without, s2 = merge("merging.max_group_sections=3", index=False)
    assert with_index.summary == without.summary and s1.calls == s2.calls
    assert all(r.contexts == [] for r in with_index.rounds)


def test_on_gives_the_merger_source_sentences_the_blocks_do_not_contain():
    result, summarizer = merge(*ON)
    first_input = summarizer.calls[0]["text"]
    group, _, context = first_input.partition("\n\n")
    assert context, "the merger's input must carry a context part after the blocks"
    source = {s for _, sentences in SPEC for s in sentences}
    blocks = {s[0] for s in SUMMARIES}
    given = result.rounds[0].contexts[0]
    assert given and all(s in source and s not in blocks for s in given)
    assert all(s in context for s in given)


def test_retrieval_is_per_block_so_every_block_is_anchored():
    embedder = FakeEmbedder()
    index = build_index(SPEC, embedder)
    cfg = make_cfg("merging.context_top_k_per_block=1", "merging.context_min_similarity=0.0").merging
    context = retrieve_merge_context([" ".join(s) for s in SUMMARIES], index, embedder, cfg, lambda t: len(t.split()))
    sections_hit = {next(i for i, (_, sents) in enumerate(SPEC) if c in sents) for c in context}
    assert sections_hit == {0, 1, 2}, "each block contributes context from its own section"


def test_context_respects_its_token_budget_and_the_group_still_fits():
    result, summarizer = merge(*ON, "merging.context_max_tokens=12", "merging.max_input_tokens=60")
    for call in summarizer.calls:
        group, _, context = call["text"].partition("\n\n")
        assert len(context.split()) <= 12
        assert len(call["text"].split()) <= 60
    assert result.summary


def test_extractive_mode_is_unaffected():
    on, s_on = merge(*ON, mode="extractive")
    off, s_off = merge("merging.max_group_sections=3", mode="extractive")
    assert on.summary == off.summary and s_on.calls == s_off.calls == []


def test_missing_index_is_a_clear_error():
    with pytest.raises(ValueError, match="needs the document index"):
        merge(*ON, index=False)


def test_pipeline_passes_the_index_through():
    doc = "\n\n".join(f"## {title}\n\n" + " ".join(sentences) for title, sentences in SPEC)
    cfg = make_cfg("merging.mode=abstractive", *ON, "summarization.passthrough_tokens=1000")
    registry = ModelRegistry(cfg, embedder=FakeEmbedder(), nli=FakeNLI(), summarizer=FakeSummarizer())
    result = HCVSumPipeline(cfg, registry).run(doc, "doc.md")
    assert result.merge.mode == "abstractive"
    first = result.merge.rounds[0]
    # Context was computed and recorded per group (recorded only on the anchored path). It is empty
    # here by design: passthrough makes every summary verbatim, so nothing new is left to retrieve.
    assert len(first.contexts) == len(first.inputs)
