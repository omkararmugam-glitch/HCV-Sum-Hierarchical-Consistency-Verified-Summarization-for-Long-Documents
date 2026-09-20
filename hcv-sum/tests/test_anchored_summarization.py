from hcv_sum.anchored_summarization import (build_model_input, clean_generated, retrieve_context,
                                           summarize_section)

from conftest import FakeSummarizer, build_index, make_cfg

SPEC = [
    ("Ops", ["The Monterrey plant shutdown halted sensor production in April.",
             "Repair costs reached four million dollars.",
             "Utilization at other plants averaged eighty one percent."]),
    ("Revenue", ["Sensor revenue declined eighteen percent.",
                 "The Monterrey plant shutdown did not affect sensor revenue."]),
    ("Supply", ["Resin lead times improved to nine weeks."]),
]


def test_retrieval_excludes_own_section_and_ranks_by_similarity(embedder):
    index = build_index(SPEC, embedder)
    cfg = make_cfg("summarization.context_top_k=2", "summarization.context_min_similarity=0.1").summarization
    ctx = retrieve_context(index.sections[0], index, cfg)
    assert ctx, "expected related sentences from other sections"
    assert all(c.section_index != 0 for c in ctx)
    assert ctx[0].text == "The Monterrey plant shutdown did not affect sensor revenue."
    assert [c.similarity for c in ctx] == sorted((c.similarity for c in ctx), reverse=True)


def test_retrieval_similarity_floor(embedder):
    index = build_index(SPEC, embedder)
    cfg = make_cfg("summarization.context_min_similarity=0.99").summarization
    assert retrieve_context(index.sections[2], index, cfg) == []


def test_mean_scoring_mode_runs(embedder):
    index = build_index(SPEC, embedder)
    cfg = make_cfg("summarization.retrieval_scoring=mean", "summarization.context_min_similarity=0.0").summarization
    assert len(retrieve_context(index.sections[1], index, cfg)) == cfg.context_top_k


def test_context_modes_render_differently(embedder):
    index = build_index(SPEC, embedder)
    fs = FakeSummarizer()
    sec = index.sections[0]
    base = make_cfg("summarization.context_min_similarity=0.1").summarization
    ctx = retrieve_context(sec, index, base)

    none_input, _ = build_model_input(sec, ctx, fs, make_cfg("summarization.context_mode=none").summarization)
    append_input, _ = build_model_input(sec, ctx, fs, base)
    instruct_input, _ = build_model_input(sec, ctx, fs, make_cfg("summarization.context_mode=instruct").summarization)

    assert none_input == sec.text
    assert append_input.startswith(sec.text) and ctx[0].text.split()[0] in append_input[len(sec.text):]
    assert "SECTION:" in instruct_input and "BACKGROUND:" in instruct_input


def test_context_is_truncated_before_section(embedder):
    index = build_index(SPEC, embedder)
    fs = FakeSummarizer()
    sec = index.sections[0]
    n = fs.count_tokens(sec.text)
    cfg = make_cfg(f"summarization.max_input_tokens={n + 12}", "summarization.context_min_similarity=0.0",
                   "summarization.context_top_k=5").summarization
    text, truncated = build_model_input(sec, retrieve_context(sec, index, cfg), fs, cfg)
    assert not truncated
    assert text.startswith(sec.text)
    assert fs.count_tokens(text) <= cfg.max_input_tokens


def test_section_truncated_only_when_it_alone_overflows(embedder):
    index = build_index(SPEC, embedder)
    fs = FakeSummarizer()
    cfg = make_cfg("summarization.max_input_tokens=15", "summarization.context_min_similarity=0.0").summarization
    text, truncated = build_model_input(index.sections[0], retrieve_context(index.sections[0], index, cfg), fs, cfg)
    assert truncated
    assert "Resin" not in text and "affect" not in text


def test_short_section_passes_through_without_generation(embedder):
    index = build_index(SPEC, embedder)
    fs = FakeSummarizer()
    cfg = make_cfg("summarization.passthrough_tokens=100").summarization
    result = summarize_section(index.sections[2], index, fs, embedder, cfg)
    assert not result.generated and fs.calls == []
    assert result.sentences == SPEC[2][1]


def test_generation_lengths_follow_config(embedder):
    index = build_index(SPEC, embedder)
    fs = FakeSummarizer()
    cfg = make_cfg("summarization.passthrough_tokens=5", "summarization.min_new_tokens=4",
                   "summarization.max_new_tokens=8").summarization
    result = summarize_section(index.sections[0], index, fs, embedder, cfg)
    assert result.generated
    assert fs.calls[0]["max_new_tokens"] <= 9 and fs.calls[0]["min_new_tokens"] <= 4


LEAKED = "The Monterrey plant shutdown did not affect sensor revenue."   # a Revenue-section sentence


def test_context_leak_is_detected_when_guard_off(embedder):
    index = build_index(SPEC, embedder)
    fs = FakeSummarizer(fixed_output="Repair costs reached four million dollars. " + LEAKED)
    cfg = make_cfg("summarization.passthrough_tokens=1", "summarization.context_min_similarity=0.1",
                   "summarization.drop_context_leaks=false").summarization
    result = summarize_section(index.sections[0], index, fs, embedder, cfg)
    assert result.context_leaks == [1]
    assert LEAKED in result.sentences


def test_context_leak_is_dropped_when_guard_on(embedder):
    index = build_index(SPEC, embedder)
    fs = FakeSummarizer(fixed_output="Repair costs reached four million dollars. " + LEAKED)
    cfg = make_cfg("summarization.passthrough_tokens=1", "summarization.context_min_similarity=0.1",
                   "summarization.drop_context_leaks=true").summarization
    result = summarize_section(index.sections[0], index, fs, embedder, cfg)
    assert result.sentences == ["Repair costs reached four million dollars."]
    assert result.dropped[0][0] == LEAKED and "context leak" in result.dropped[0][1]


def test_all_leaks_fall_back_to_section_lead(embedder):
    index = build_index(SPEC, embedder)
    fs = FakeSummarizer(fixed_output=LEAKED)
    cfg = make_cfg("summarization.passthrough_tokens=1", "summarization.context_min_similarity=0.1").summarization
    result = summarize_section(index.sections[0], index, fs, embedder, cfg)
    assert result.sentences == SPEC[0][1][:1]
    assert result.dropped[0][0] == LEAKED and "context leak" in result.dropped[0][1]
    assert any("context leak" in n and "first sentence instead" in n for n in result.notes)


def test_incomplete_final_sentence_is_dropped(embedder):
    index = build_index(SPEC, embedder)
    fs = FakeSummarizer(fixed_output="Repair costs reached four million dollars. Utilization at other plants averaged")
    cfg = make_cfg("summarization.passthrough_tokens=1").summarization
    result = summarize_section(index.sections[0], index, fs, embedder, cfg)
    assert result.sentences == ["Repair costs reached four million dollars."]
    assert "incomplete" in result.dropped[0][1]


def test_clean_generated_fixes_cnn_spacing():
    assert clean_generated("Revenue fell 18 % . Costs rose ( slightly ) .") == "Revenue fell 18%. Costs rose (slightly)."


def test_single_incomplete_sentence_falls_back_to_section_lead(embedder):
    index = build_index(SPEC, embedder)
    fs = FakeSummarizer(fixed_output="Each party's total aggregate liability shall not exceed the Fees paid in the")
    cfg = make_cfg("summarization.passthrough_tokens=1", "summarization.context_min_similarity=0.99").summarization
    result = summarize_section(index.sections[0], index, fs, embedder, cfg)
    assert result.sentences == SPEC[0][1][:1]
    assert "incomplete sentence" in result.dropped[0][1]
    assert any("cut off" in n and "first sentence instead" in n for n in result.notes)
    # the truncated text must NOT survive as a kept sentence
    assert all("shall not exceed the Fees paid in the" not in s for s in result.sentences)
