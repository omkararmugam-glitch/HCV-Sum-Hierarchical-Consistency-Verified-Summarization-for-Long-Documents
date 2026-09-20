"""Large-document readiness (SCALING.md): pre-flight estimate, Stage 2 checkpoint/resume, the size-based
switch to abstractive merging, progress lines, and the run-limits report. Fake models throughout."""

import json
import random

import pytest

from hcv_sum.checkpoint import Stage2Checkpoint
from hcv_sum.models import ModelRegistry
from hcv_sum.pipeline import HCVSumPipeline
from hcv_sum.preflight import run_preflight
from hcv_sum.report import render_limits
from hcv_sum.segmentation import segment_document

from conftest import FakeEmbedder, FakeNLI, FakeSummarizer, make_cfg

TOPICS = ["revenue", "margin", "plant", "contract", "renewal", "payment", "inflation", "labor", "policy",
          "sensor", "europe", "automotive", "notice", "invoice", "committee", "market", "shipping", "audit"]


def synthetic_document(n_sections: int, sentences_per_section: int = 6, seed: int = 0, headings: bool = True) -> str:
    rng = random.Random(seed)
    parts = []
    for s in range(n_sections):
        topic = rng.sample(TOPICS, 3)
        body = " ".join(
            f"The {topic[k % 3]} {rng.choice(TOPICS)} figure for unit {s} changed by {rng.randint(1, 90)} percent "
            f"in period {k} according to the {rng.choice(TOPICS)} report." for k in range(sentences_per_section))
        parts.append((f"## Section {s} {topic[0].title()}\n\n" if headings else "") + body)
    return "\n\n".join(parts)


def pipeline(*overrides, summarizer=None):
    cfg = make_cfg(*overrides)
    return HCVSumPipeline(cfg, ModelRegistry(cfg, embedder=FakeEmbedder(), nli=FakeNLI(),
                                             summarizer=summarizer or FakeSummarizer()))


def words(text: str) -> int:
    return len(text.split())


# ------------------------------------------------------------------ pre-flight estimator
def test_preflight_structured_document_is_exact_when_sections_fit(cfg):
    text = synthetic_document(12)
    p = run_preflight(text, words, cfg, 1024)
    actual = segment_document(text, FakeEmbedder(), words, cfg.segmentation)
    assert p.structured and p.sections_low == p.sections_high == len(actual) == 12
    assert p.blocks_over_budget == 0 and not p.large_document and not p.warnings


def test_preflight_range_contains_actual_sections_when_blocks_are_split():
    cfg = make_cfg("segmentation.max_section_tokens=60")
    text = synthetic_document(8, sentences_per_section=10)
    p = run_preflight(text, words, cfg, 1024)
    actual = segment_document(text, FakeEmbedder(), words, cfg.segmentation)
    assert p.blocks_over_budget == 8
    assert p.sections_low <= len(actual) <= p.sections_high
    assert any("exceed max_section_tokens" in w for w in p.warnings)


def test_preflight_warns_when_section_ceiling_forces_oversize_segments():
    """The failure this work was about: a ceiling that forces segments bigger than the token budget."""
    cfg = make_cfg("segmentation.max_sections=5", "segmentation.max_section_tokens=100")
    text = synthetic_document(40, headings=False)            # 240 sentences, no headings
    p = run_preflight(text, words, cfg, 1024)
    assert p.section_ceiling_binding and p.min_segment_sentences == 48
    assert any("emergency split" in w for w in p.warnings)
    assert any("section ceiling reached" in w for w in p.warnings)


def test_preflight_warns_when_projected_sections_exceed_the_summarizer_budget():
    cfg = make_cfg("segmentation.max_section_tokens=5000", "summarization.max_input_tokens=300")
    p = run_preflight(synthetic_document(3, sentences_per_section=40), words, cfg, 1024)
    assert p.projected_section_tokens > p.summarizer_input_budget == 300
    assert any("TRUNCATED" in w for w in p.warnings)


def test_preflight_flags_large_documents_and_estimates_runtime(cfg):
    p = run_preflight(synthetic_document(60), words, cfg, 1024)
    assert p.large_document and p.sections_high >= cfg.scale.large_document_sections
    assert p.est_stage2_minutes > 0 and p.est_total_minutes >= p.est_stage2_minutes
    assert any("large document" in w for w in p.warnings)


def test_default_ceiling_leaves_small_documents_alone(cfg):
    """Raising max_sections 250 -> 2000 changes nothing below 1000 sentences (the minimum stays 4)."""
    from hcv_sum.segmentation import minimum_segment_size
    for n in (50, 500, 999):
        assert minimum_segment_size(n, cfg.segmentation) == cfg.segmentation.min_segment_sentences
    assert minimum_segment_size(25_000, cfg.segmentation) == 13


# ------------------------------------------------------------------ checkpoint / resume
class CrashingSummarizer(FakeSummarizer):
    def __init__(self, crash_after_batches: int):
        super().__init__()
        self.batches, self.crash_after = 0, crash_after_batches

    def summarize_batch(self, texts, *, min_new_tokens, max_new_tokens):
        if self.batches >= self.crash_after:
            raise KeyboardInterrupt("simulated interruption")
        self.batches += 1
        return super().summarize_batch(texts, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens)


def test_resume_after_crash_regenerates_only_missing_sections_and_matches(tmp_path):
    text = synthetic_document(20, sentences_per_section=12)
    overrides = ("summarization.batch_size=2",)
    reference = pipeline(*overrides).run(text, "doc", checkpoint_mode="never")
    total_generations = reference.stats["summarizer_generations"]
    assert total_generations > 6

    with pytest.raises(KeyboardInterrupt):
        pipeline(*overrides, summarizer=CrashingSummarizer(3)).run(
            text, "doc", checkpoint_mode="always", checkpoint_dir=str(tmp_path))
    files = list(tmp_path.glob("stage2_*.jsonl"))
    assert len(files) == 1
    saved = len(files[0].read_text(encoding="utf-8").splitlines())
    assert 0 < saved < total_generations

    resumed_summarizer = FakeSummarizer()
    resumed = pipeline(*overrides, summarizer=resumed_summarizer).run(
        text, "doc", checkpoint_mode="always", checkpoint_dir=str(tmp_path))
    assert resumed.stats["checkpoint"]["reused"] == saved
    assert len(resumed_summarizer.calls) == total_generations - saved
    assert resumed.merge.summary == reference.merge.summary
    assert [s.sentences for s in resumed.section_summaries] == [s.sentences for s in reference.section_summaries]


def test_checkpoint_is_not_reused_for_a_different_config_or_document(tmp_path):
    text = synthetic_document(10, sentences_per_section=12)
    pipeline().run(text, "doc", checkpoint_mode="always", checkpoint_dir=str(tmp_path))
    other = pipeline("summarization.max_new_tokens=60").run(text, "doc", checkpoint_mode="always",
                                                             checkpoint_dir=str(tmp_path))
    assert other.stats["checkpoint"]["reused"] == 0
    changed = pipeline().run(text + " One more sentence about the audit.", "doc", checkpoint_mode="always",
                             checkpoint_dir=str(tmp_path))
    assert changed.stats["checkpoint"]["reused"] == 0
    assert len(list(tmp_path.glob("stage2_*.jsonl"))) == 3


def test_truncated_checkpoint_line_is_skipped(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"key": "a", "section_index": 0, "generated": "x."}) + "\n{\"key\": \"b\", \"gen",
                    encoding="utf-8")
    cp = Stage2Checkpoint(path)
    assert cp.loaded == 1 and cp.corrupt_lines == 1 and cp.get("a") == "x." and cp.get("b") is None


def test_checkpoint_auto_only_for_large_documents(tmp_path):
    small = pipeline(f"scale.checkpoint_dir={tmp_path.as_posix()}").run(synthetic_document(10), "small")
    assert small.stats["checkpoint"] is None and not list(tmp_path.iterdir())
    large = pipeline(f"scale.checkpoint_dir={tmp_path.as_posix()}", "scale.large_document_sections=10").run(
        synthetic_document(12), "large")
    assert large.stats["checkpoint"] is not None and list(tmp_path.glob("stage2_*.jsonl"))


# ------------------------------------------------------------------ size-based merge switch
def test_small_document_keeps_configured_merge_mode(cfg):
    result = pipeline().run(synthetic_document(10), "small")
    assert result.merge.mode == cfg.merging.mode == "extractive"
    assert result.stats["large_document"] is False
    switch = [x for x in result.stats["limits"] if x["limit"] == "scale.large_document_merge_mode"][0]
    assert not switch["hit"]


def test_large_document_switches_to_abstractive_merge(tmp_path):
    result = pipeline("scale.large_document_sections=12", f"scale.checkpoint_dir={tmp_path.as_posix()}").run(
        synthetic_document(15), "large")
    assert result.stats["large_document"] is True
    assert result.merge.mode == "abstractive" and result.merge.rounds
    assert "large_document_sections=12" in result.stats["merge_mode"]["reason"]
    limits = {x["limit"]: x for x in result.stats["limits"]}
    assert limits["scale.large_document_merge_mode"]["hit"]
    assert "abstractive coverage" in limits                      # coverage is always reported in this mode


def test_large_document_mode_can_stay_extractive(tmp_path):
    result = pipeline("scale.large_document_sections=12", "scale.large_document_merge_mode=extractive",
                      f"scale.checkpoint_dir={tmp_path.as_posix()}").run(synthetic_document(15), "large")
    assert result.merge.mode == "extractive"


# ------------------------------------------------------------------ progress + limits reporting
def test_progress_lines_only_for_large_documents(tmp_path):
    lines: list[str] = []
    pipeline(f"scale.checkpoint_dir={tmp_path.as_posix()}").run(synthetic_document(10, 12), "small",
                                                               heartbeat=lines.append)
    assert lines == []
    pipeline("scale.large_document_sections=10", "scale.progress_every_sections=3",
             f"scale.checkpoint_dir={tmp_path.as_posix()}").run(synthetic_document(12, 12), "large",
                                                               heartbeat=lines.append)
    stage2 = [line for line in lines if line.startswith("Stage 2:")]
    assert stage2 and "ETA" in stage2[-1] and "PRE-FLIGHT" in lines[0]


def test_limits_report_every_cap_that_was_hit():
    result = pipeline("contradiction.candidate_top_k=1", "contradiction.pair_min_similarity=0.0").run(
        synthetic_document(10), "doc")
    limits = {(x["stage"], x["limit"]): x for x in result.stats["limits"]}
    assert limits[("stage3a", "contradiction.candidate_top_k")]["hit"]
    assert not limits[("stage2", "summarization.max_input_tokens")]["hit"]
    text = render_limits(result, only_hits=True)
    assert "candidate_top_k" in text and "max_input_tokens" not in text


def test_per_stage_memory_and_time_are_recorded():
    result = pipeline().run(synthetic_document(6), "doc")
    for name, stage in result.stats["per_stage"].items():
        assert stage["peak_rss_mb"] > 0 and stage["seconds"] >= 0, name


# ------------------------------------------------------------------ Stage 4 options used by the sweep
def test_capped_extractive_bounds_length_and_keeps_every_section():
    from hcv_sum.merging import merge_summaries
    sections = [[f"Unit {s} revenue grew {k} percent in the audit period." for k in range(5)] for s in range(8)]
    cfg = make_cfg("merging.extractive_max_per_section=1", "merging.dedup_similarity=1.01").merging
    merge = merge_summaries(sections, [], FakeSummarizer(), FakeEmbedder(), FakeNLI(), cfg, mode="extractive")
    assert len(merge.sentences) == 8
    assert [int(s.split()[1]) for s in merge.sentences] == list(range(8))


def test_coverage_guard_restores_a_block_the_merger_dropped():
    from hcv_sum.merging import guard_coverage
    blocks = ["Plant revenue in Monterrey grew sharply this year.", "Shipping costs from Osaka doubled after the strike."]
    output, added = guard_coverage(blocks, "Plant revenue in Monterrey grew sharply this year.", FakeEmbedder(), 0.6)
    assert added == ["Shipping costs from Osaka doubled after the strike."]
    assert output.endswith("doubled after the strike.")
    same, none = guard_coverage(blocks, " ".join(blocks), FakeEmbedder(), 0.6)
    assert none == [] and same == " ".join(blocks)


def test_merge_options_are_off_by_default(cfg):
    assert cfg.merging.coverage_guard is False and cfg.merging.extractive_max_per_section == 0


# ------------------------------------------------------------------ Stage 3 live progress
def test_stage3_reports_pair_progress_on_a_large_document(tmp_path):
    """Stage 3 was silent, so a long run could not be told from a hang. It now reports like Stage 2 does."""
    lines: list[str] = []
    pipe = pipeline("scale.large_document_sections=10", "scale.progress_every_pairs=25",
                    "scale.progress_every_seconds=0", "contradiction.pair_min_similarity=0.0",
                    f"scale.checkpoint_dir={tmp_path.as_posix()}")
    pipe.run(synthetic_document(12, 8), "large", heartbeat=lines.append)
    stage3 = [line for line in lines if line.startswith("Stage 3")]
    assert stage3, f"no Stage 3 progress lines in: {lines}"
    compared = [line for line in stage3 if "pairs compared" in line]
    assert compared and "ETA" in compared[-1] and "elapsed" in compared[-1]
    assert compared[-1].split("/")[1].split()[0].replace(",", "").isdigit(), compared[-1]


def test_stage3_progress_is_quiet_on_a_small_document(tmp_path):
    """Same rule as Stage 2: no progress chatter below the large-document threshold."""
    lines: list[str] = []
    pipeline(f"scale.checkpoint_dir={tmp_path.as_posix()}").run(synthetic_document(6, 6), "small",
                                                                heartbeat=lines.append)
    assert [line for line in lines if line.startswith("Stage 3")] == []


def test_stage3_progress_does_not_change_what_stage3_computes(tmp_path):
    """The pairs are scored in blocks when reporting; scores and decisions must be bit-identical."""
    from hcv_sum.contradiction import check_section_summaries
    from hcv_sum.scoring import DocumentIndex
    from hcv_sum.segmentation import segment_document
    from hcv_sum.anchored_summarization import summarize_sections

    cfg = make_cfg("contradiction.pair_min_similarity=0.0")
    embedder, nli, summarizer = FakeEmbedder(), FakeNLI(), FakeSummarizer()
    sections = segment_document(synthetic_document(8, 6), embedder, summarizer.count_tokens, cfg.segmentation)
    index = DocumentIndex.build(sections, embedder)
    summaries = summarize_sections(sections, index, summarizer, embedder, cfg.summarization)

    quiet = check_section_summaries(summaries, index, embedder, nli, cfg.contradiction)
    noisy = check_section_summaries(summaries, index, embedder, nli, cfg.contradiction,
                                    heartbeat=lambda _m: None, heartbeat_every=(7, 0.0))
    assert (quiet.pairs_checked, quiet.one_sided_checked) == (noisy.pairs_checked, noisy.one_sided_checked)
    assert [(c.claim_a.text, c.claim_b.text, c.score) for c in quiet.all_contradictions] == \
           [(c.claim_a.text, c.claim_b.text, c.score) for c in noisy.all_contradictions]
    assert quiet.corrected_sentences == noisy.corrected_sentences and quiet.flagged == noisy.flagged
