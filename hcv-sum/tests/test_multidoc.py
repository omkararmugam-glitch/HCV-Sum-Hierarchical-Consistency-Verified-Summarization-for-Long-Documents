"""Task 3: unsupervised multi-document mode (src/hcv_sum/multidoc.py, periods.py, centrality.py; FINDINGS 17).

Fake models: FakeNLI flags a pair when one sentence repeats the other's words with a negation added."""

import re

from hcv_sum import cli, multidoc as multidoc_module
from hcv_sum.models import ModelRegistry
from hcv_sum.multidoc import MultiDocPipeline, select_salient
from hcv_sum.periods import claim_period, describe, document_period, markers

from conftest import FakeEmbedder, FakeNLI, FakeSummarizer, make_cfg

Q1 = """# Acme First Quarter 2026 Update

## Results
Sensor revenue in the first quarter fell because of the plant shutdown.
The Lyon warehouse was expanded during the spring months.

## Staff
Engineering headcount grew by forty people in Lyon.
"""
Q2 = """# Acme Second Quarter 2026 Update

## Results
Sensor revenue in the first quarter fell not because of the plant shutdown.
The Lyon warehouse was not expanded during the spring months.

## Staff
Hiring focused on firmware engineers in Lyon.
"""


def run(*docs, overrides=()):
    cfg = make_cfg("summarization.passthrough_tokens=1000", *overrides)
    reg = ModelRegistry(cfg, embedder=FakeEmbedder(), nli=FakeNLI(), summarizer=FakeSummarizer())
    return MultiDocPipeline(cfg, reg).run(list(docs))


def test_period_markers():
    assert markers("Revenue for Q2 2026 rose.") == [("Q", 2026, 2)]
    assert markers("In the first quarter, revenue fell.", default_year=2026) == [("Q", 2026, 1)]
    assert markers("fiscal 2025 and Amendment No. 3") == [("FY", 2025, None), ("V", "3", None)]
    assert markers("We have never paid a dividend.") == []
    assert describe(document_period(Q2)) == "Q2 2026"
    assert claim_period("Revenue in the first quarter was 104 million.", ("Q", 2026, 2)) == ("Q", 2026, 1)
    assert claim_period("Headcount was 1,350.", ("Q", 2026, 2)) == ("Q", 2026, 2)


def test_sections_are_pooled_across_documents_with_document_titles():
    result = run(("q1.md", Q1), ("q2.md", Q2))
    assert [d["period"] for d in result.stats["documents"]] == ["Q1 2026", "Q2 2026"]
    assert result.stats["section_documents"] == [0, 0, 1, 1]
    assert all(s.title.startswith(("[q1.md]", "[q2.md]")) for s in result.sections)


def test_cross_document_conflict_about_the_same_period_stays_a_contradiction():
    result = run(("q1.md", Q1), ("q2.md", Q2))
    flags = result.contradictions.all_contradictions
    same_period = [c for c in flags if "first quarter" in c.claim_a.text and "first quarter" in c.claim_b.text]
    assert same_period and all(c.resolution != "possibly_superseded" for c in same_period)
    assert "both claims about Q1 2026" in same_period[0].reason


def test_cross_document_difference_without_a_shared_period_is_possibly_superseded():
    result = run(("q1.md", Q1), ("q2.md", Q2))
    warehouse = [c for c in result.contradictions.all_contradictions if "warehouse" in c.claim_a.text]
    assert warehouse and all(c.resolution == "possibly_superseded" for c in warehouse)
    assert "Q1 2026" in warehouse[0].reason and "Q2 2026" in warehouse[0].reason
    assert result.stats["flags"]["possibly_superseded"] == len(warehouse)


def test_supersession_can_be_switched_off():
    result = run(("q1.md", Q1), ("q2.md", Q2), overrides=("multidoc.supersede_by_period=false",))
    assert all(c.resolution != "possibly_superseded" for c in result.contradictions.all_contradictions)


def test_documents_without_period_markers_are_never_downgraded():
    result = run(("a.md", Q1.replace("First Quarter 2026", "Update").replace(" in the first quarter", "")),
                 ("b.md", Q2.replace("Second Quarter 2026", "Update").replace(" in the first quarter", "")))
    assert result.contradictions.all_contradictions
    assert all(c.resolution != "possibly_superseded" for c in result.contradictions.all_contradictions)


def test_salience_keeps_the_most_central_sentences_in_document_order():
    sections = [[f"Revenue from sensors grew in region {i}." for i in range(4)], ["The cafeteria menu changed."],
                [f"Sensor revenue in region {i} grew strongly." for i in range(4)]]
    kept, _, pool = select_salient(sections, FakeEmbedder(), 4, "pagerank", 3)
    assert len(kept) == 4 and "The cafeteria menu changed." not in kept
    assert kept == [s for s in pool if s in kept], "document order preserved"


def test_final_summary_is_capped_and_multi_document_brief_line(tmp_path, monkeypatch, capsys):
    result = run(("q1.md", Q1), ("q2.md", Q2), overrides=("multidoc.max_summary_sentences=3",))
    assert result.merge.mode == "salience" and len(result.merge.sentences) <= 3
    monkeypatch.setattr(multidoc_module, "ModelRegistry",
                        lambda cfg: ModelRegistry(cfg, embedder=FakeEmbedder(), nli=FakeNLI(), summarizer=FakeSummarizer()))
    a, b = tmp_path / "q1.md", tmp_path / "q2.md"
    a.write_text(Q1, encoding="utf-8")
    b.write_text(Q2, encoding="utf-8")
    assert cli.main([str(a), str(b), "--brief", "--set", "summarization.passthrough_tokens=1000"]) == 0
    out = capsys.readouterr().out
    # --brief is plain language for both paths: no stage numbers, and the pooled run says how many documents.
    assert out.startswith("Split the 2 documents into 4 sections")
    assert not re.search(r"(?i)stage\s*[1-5]", out.split("\n\n")[0])
    assert "possibly superseded (different periods)" in out and "most central sentences across documents" in out


# ------------------------------------------------------------------ scale-review regressions (final validation)
def test_verbose_report_renders_for_multi_document_runs():
    """Regression: render_result(show_stages=True) raised KeyError('embedded_texts_total') on a multi-document
    result, so `hcv-sum a.md b.md --verbose` crashed after the whole run had finished."""
    from hcv_sum.report import render_result
    text = render_result(run(("q1.md", Q1), ("q2.md", Q2)), show_stages=True)
    assert "RUN STATS" in text and "RUN LIMITS" in text and "stage3_contradictions" in text


def test_multi_document_runs_report_every_cap_they_hit():
    """Regression: multi-document runs produced no run-limits list, so a cap hit on a large pooled set
    (candidate_top_k, max_pairs, truncation, resolution budget) went unreported."""
    result = run(("q1.md", Q1), ("q2.md", Q2), overrides=("contradiction.candidate_top_k=1",
                                                          "contradiction.pair_min_similarity=0.0"))
    limits = {(x["stage"], x["limit"]): x for x in result.stats["limits"]}
    assert limits[("stage3a", "contradiction.candidate_top_k")]["hit"]
    assert not limits[("stage2", "summarization.max_input_tokens")]["hit"]
    for stage in result.stats["per_stage"].values():
        assert "peak_rss_mb" in stage and "seconds" in stage


def test_progress_lines_for_a_large_pooled_set():
    lines: list[str] = []
    cfg = make_cfg("summarization.passthrough_tokens=1000", "scale.large_document_sections=3")
    reg = ModelRegistry(cfg, embedder=FakeEmbedder(), nli=FakeNLI(), summarizer=FakeSummarizer())
    MultiDocPipeline(cfg, reg).run([("q1.md", Q1), ("q2.md", Q2)], heartbeat=lines.append)
    assert lines and lines[0].startswith("multi-document: 2 documents, 4 pooled sections")


def test_many_documents_pool_without_a_fixed_limit():
    docs = [(f"q{i}.md", Q1.replace("First Quarter 2026", f"Update {i}")) for i in range(5)]
    result = run(*docs)
    assert len(result.stats["documents"]) == 5
    assert result.stats["section_documents"] == [d for d in range(5) for _ in range(2)]
    assert len(result.sections) == 10


def test_cli_notes_single_document_flags_it_ignores(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(multidoc_module, "ModelRegistry",
                        lambda cfg: ModelRegistry(cfg, embedder=FakeEmbedder(), nli=FakeNLI(), summarizer=FakeSummarizer()))
    a, b = tmp_path / "q1.md", tmp_path / "q2.md"
    a.write_text(Q1, encoding="utf-8")
    b.write_text(Q2, encoding="utf-8")
    assert cli.main([str(a), str(b), "--diagnose-sources", "--set", "summarization.passthrough_tokens=1000"]) == 0
    assert "--diagnose-sources is a single-document option" in capsys.readouterr().err
