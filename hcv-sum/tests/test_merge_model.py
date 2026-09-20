"""Task 1: a separately configurable Stage 4 model and the instruction prompt (FINDINGS 15)."""

from hcv_sum.config import load_config
from hcv_sum.merging import merge_summaries, render_merge_input
from hcv_sum.models import ModelRegistry
from hcv_sum.pipeline import HCVSumPipeline

from conftest import FakeEmbedder, FakeNLI, FakeSummarizer, build_index, make_cfg

SPEC = [
    ("Plant", ["The Monterrey plant closed for three weeks after a fire.", "Transformer repairs cost four million dollars.",
               "Monterrey plant output resumed in May."]),
    ("Freight", ["Freight costs from Osaka rose eleven percent.", "Osaka freight contracts were renegotiated in June.",
                 "Shipping delays from Osaka eased by August."]),
]
DOC = "\n\n".join(f"## {t}\n\n" + " ".join(s) for t, s in SPEC)


def test_merge_model_defaults_to_the_stage2_model(cfg):
    assert cfg.models.merge_summarizer == ""
    fake = FakeSummarizer()
    reg = ModelRegistry(cfg, embedder=FakeEmbedder(), nli=FakeNLI(), summarizer=fake)
    assert reg.merge_summarizer is fake


def test_pipeline_sends_merging_to_the_merge_model_only():
    cfg = make_cfg("merging.mode=abstractive", "merging.check_between_rounds=false")
    stage2, stage4 = FakeSummarizer(), FakeSummarizer()
    reg = ModelRegistry(cfg, embedder=FakeEmbedder(), nli=FakeNLI(), summarizer=stage2, merge_summarizer=stage4)
    result = HCVSumPipeline(cfg, reg).run(DOC, "doc.md")
    assert result.merge.mode == "abstractive" and stage4.calls, "the merge model must do the merging"
    merged_inputs = {c["text"] for c in stage4.calls}
    assert not any(c["text"] in merged_inputs for c in stage2.calls), "Stage 2 model must not be asked to merge"


def test_instruct_prompt_names_the_parts_and_what_context_is_for():
    m = make_cfg("merging.merge_prompt=instruct").merging
    with_ctx = render_merge_input("A happened. B happened.", ["C is true."], m)
    without = render_merge_input("A happened. B happened.", [], m)
    assert "SUMMARIES: A happened. B happened." in with_ctx and "CONTEXT: C is true." in with_ctx
    assert "do not contradict" in with_ctx.lower() or "does not contradict" in with_ctx.lower()
    assert "CONTEXT" not in without and "SUMMARIES: A happened." in without


def test_default_prompt_is_unchanged_plain_text(cfg):
    assert cfg.merging.merge_prompt == "none"
    assert render_merge_input("A. B.", [], cfg.merging) == "A. B."
    assert render_merge_input("A. B.", ["C."], cfg.merging) == "A. B.\n\nC."


def test_instruct_prompt_fits_the_input_budget():
    """The instruction text is taken out of the group budget, so every merge input fits the model."""
    embedder = FakeEmbedder()
    index = build_index(SPEC, embedder)
    small = FakeSummarizer()
    small.model_max_input = 200
    m = make_cfg("merging.merge_prompt=instruct", "merging.context_anchoring=true", "merging.context_max_tokens=20",
                 "merging.context_min_similarity=0.0", "merging.check_between_rounds=false").merging
    result = merge_summaries([s[:2] for _, s in SPEC], [], small, embedder, FakeNLI(), m, mode="abstractive", index=index)
    assert result.summary and small.calls
    assert all(len(c["text"].split()) <= 200 for c in small.calls)
    with_context = [c["text"] for c in small.calls if "CONTEXT:" in c["text"]]
    assert with_context
    assert all(t.index("SUMMARIES:") < t.index("CONTEXT:") for t in with_context),         "context last, so truncation never cuts summaries"
    assert load_config().merging.merge_prompt == "none"
