"""End-to-end tests.

Fast tests run the full pipeline and the CLI with fake models (wiring, data flow,
report rendering, JSON export). Tests marked ``slow`` load the real models and
check behaviour on the sample documents; run them with ``pytest -m slow``.
"""

import json

from hcv_sum import cli, pipeline as pipeline_module
from hcv_sum.config import load_config
from hcv_sum.models import ModelRegistry
from hcv_sum.pipeline import HCVSumPipeline
from hcv_sum.report import render_result

from conftest import FakeEmbedder, FakeNLI, FakeSummarizer

DOC = """# Plant report

## Operations
The Monterrey plant closed for three weeks after a fire.
Sensor revenue fell because of the plant shutdown.
Repair costs were four million dollars.

## Revenue
Automotive revenue grew four percent.
European demand weakened sharply.
Sensor revenue fell not because of the plant shutdown.

## Outlook
The company expects revenue growth next quarter.
Capital spending will focus on automation.
"""


def fake_registry(cfg):
    return ModelRegistry(cfg, embedder=FakeEmbedder(), nli=FakeNLI(), summarizer=FakeSummarizer())


def test_pipeline_runs_all_stages_with_fake_models():
    cfg = load_config(overrides=["summarization.passthrough_tokens=1000"])   # verbatim sections: deterministic
    result = HCVSumPipeline(cfg, fake_registry(cfg)).run(DOC, "doc.md", diagnose_sources=True)

    assert [s.title for s in result.sections] == ["Plant report > Operations", "Revenue", "Outlook"]
    assert len(result.section_summaries) == 3
    flagged = result.contradictions.contradictions
    assert len(flagged) == 1
    assert {flagged[0].claim_a.section_index, flagged[0].claim_b.section_index} == {0, 1}
    # Both claims are verbatim source sentences, so both are fully supported: must stay unresolved.
    assert flagged[0].resolution == "unresolved"
    assert result.source_diagnostic is not None and len(result.source_diagnostic.contradictions) == 1
    assert result.merge.summary
    assert len(result.provenance) == len(result.merge.sentences)
    assert all(r.status == "supported" for r in result.provenance)
    assert set(result.timings) >= {"stage1_segmentation", "stage2_anchored_summarization", "stage3_contradictions",
                                   "stage4_merge", "stage5_provenance"}


def test_report_and_json_export(tmp_path):
    cfg = load_config(overrides=["summarization.passthrough_tokens=1000"])
    result = HCVSumPipeline(cfg, fake_registry(cfg)).run(DOC, "doc.md")
    text = render_result(result)
    for heading in ("STAGE 1", "STAGE 2", "STAGE 3", "STAGE 4", "FINAL SUMMARY", "EVIDENCE PANEL"):
        assert heading in text
    data = result.to_dict()
    json.dumps(data)  # must be serialisable
    assert data["contradictions"]["contradictions"][0]["resolution"] == "unresolved"


VERBATIM = ["summarization.passthrough_tokens=1000"]


def run_cli(tmp_path, monkeypatch, capsys, *flags):
    monkeypatch.setattr(pipeline_module, "ModelRegistry", fake_registry)
    doc = tmp_path / "doc.md"
    doc.write_text(DOC, encoding="utf-8")
    code = cli.main([str(doc), *flags, "--set", *VERBATIM])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_default_prints_only_the_summary(tmp_path, monkeypatch, capsys):
    """With no flags: the final summary and nothing else -- no stages, no evidence panel, no progress."""
    code, out, err = run_cli(tmp_path, monkeypatch, capsys)
    assert code == 0
    assert "STAGE" not in out and "EVIDENCE PANEL" not in out
    assert "=====" not in out and "HCV-Sum run" not in out
    assert err == "", f"stderr must stay empty by default, got: {err!r}"
    assert out.strip().startswith("The Monterrey plant closed for three weeks after a fire.")
    assert len(out.strip().splitlines()) == 1, "the summary is printed as one unwrapped block"


def test_cli_default_output_matches_render_summary(tmp_path, monkeypatch, capsys):
    cfg = load_config(overrides=VERBATIM)
    expected = HCVSumPipeline(cfg, fake_registry(cfg)).run(DOC, "doc.md").merge.summary
    _, out, _ = run_cli(tmp_path, monkeypatch, capsys)
    assert out.strip() == expected.strip()


def test_cli_json_flag_still_writes_the_full_run(tmp_path, monkeypatch, capsys):
    """--json is unchanged: the file holds every stage even though the terminal shows only the summary."""
    out_json = tmp_path / "run.json"
    code, out, _ = run_cli(tmp_path, monkeypatch, capsys, "--json", str(out_json))
    assert code == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["document_name"] == "doc.md"
    assert data["sections"] and data["section_summaries"] and data["provenance"]
    assert "STAGE" not in out


def test_cli_evidence_flag_adds_the_evidence_panel(tmp_path, monkeypatch, capsys):
    code, out, _ = run_cli(tmp_path, monkeypatch, capsys, "--evidence")
    assert code == 0
    assert "FINAL SUMMARY" in out and "EVIDENCE PANEL" in out
    assert "STAGE 1" not in out and "STAGE 3a" not in out


def test_cli_run_limits_block_appears_exactly_once(tmp_path, monkeypatch, capsys):
    """Regression: --evidence printed RUN LIMITS twice (render_result() and the CLI both added it)."""
    for flags in (("--evidence",), ("--verbose",), ("--brief", "--evidence")):
        code, out, _ = run_cli(tmp_path, monkeypatch, capsys, *flags)
        assert code == 0
        assert out.count("RUN LIMITS:") == 1, f"{flags}: RUN LIMITS printed {out.count('RUN LIMITS:')} times"
    for flags in ((), ("--brief",)):
        _, out, _ = run_cli(tmp_path, monkeypatch, capsys, *flags)
        assert "RUN LIMITS" not in out, f"{flags}: stdout must not carry the limits block"


def test_cli_verbose_prints_every_stage(tmp_path, monkeypatch, capsys):
    code, out, err = run_cli(tmp_path, monkeypatch, capsys, "--verbose")
    assert code == 0
    for heading in ("STAGE 1", "STAGE 2", "STAGE 3a", "STAGE 3b", "STAGE 4", "FINAL SUMMARY", "EVIDENCE PANEL"):
        assert heading in out, f"missing {heading} in verbose output"
    assert "stage1_segmentation" in err, "verbose must report progress on stderr"


def test_cli_diagnose_sources_still_works(tmp_path, monkeypatch, capsys):
    out_json = tmp_path / "run.json"
    code, out, _ = run_cli(tmp_path, monkeypatch, capsys, "--diagnose-sources", "--json", str(out_json))
    assert code == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["source_diagnostic"] is not None
    assert "STAGE" not in out


def test_cli_rejects_bad_override(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text(DOC, encoding="utf-8")
    assert cli.main([str(doc), "--set", "contradiction.nope=1"]) == 2


def test_cli_missing_file(tmp_path):
    assert cli.main([str(tmp_path / "missing.txt")]) == 2


def test_cli_strips_utf8_bom(tmp_path, monkeypatch, capsys):
    """Regression: a BOM became part of the first heading and broke Stage 1 detection."""
    monkeypatch.setattr(pipeline_module, "ModelRegistry", fake_registry)
    doc = tmp_path / "bom.md"
    doc.write_text(DOC, encoding="utf-8-sig")
    out_json = tmp_path / "bom.json"
    assert cli.main([str(doc), "--json", str(out_json), "--set", "summarization.passthrough_tokens=1000"]) == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["sections"][0]["title"] == "Plant report > Operations"
    assert "﻿" not in json.dumps(data)


def test_cli_reports_non_utf8_file_cleanly(tmp_path, capsys):
    doc = tmp_path / "latin.md"
    doc.write_bytes("# Titél\n\nRevenue grew.\n".encode("latin-1"))
    assert cli.main([str(doc)]) == 2
    assert "not UTF-8" in capsys.readouterr().err


def check_invariants(result):
    """Structural invariants that must hold for ANY document. Wrong indices here would mean
    the evidence panel cites the wrong source sentence, which is the whole point of the pipeline."""
    flat, sec_of, offset = [], [], 0
    for sec in result.sections:
        assert sec.sentence_offset == offset, "section offsets must be contiguous"
        offset += len(sec.sentences)
        flat += sec.sentences
        sec_of += [sec.index] * len(sec.sentences)

    for record in result.provenance:
        c = record.citation
        if c is None:
            continue
        assert {sec_of[i] for i in c.sentence_indices} == {c.section_index}, "citation crosses sections"
        assert c.text == " ".join(flat[i] for i in c.sentence_indices), "citation text must match its ids"
        assert 0.0 <= c.entailment <= 1.0 and -1.01 <= c.similarity <= 1.01

    for x in result.contradictions.contradictions:
        for claim in (x.claim_a, x.claim_b):
            assert result.section_summaries[claim.section_index].sentences[claim.sentence_index] == claim.text
    for x in result.contradictions.one_sided:
        a, b = x.claim_a, x.claim_b
        assert result.section_summaries[a.section_index].sentences[a.sentence_index] == a.text
        assert result.sections[b.section_index].sentences[b.sentence_index] == b.text

    assert len(result.provenance) == len(result.merge.sentences)
    if result.merge.mode == "extractive":
        removed = {r[0] for r in result.merge.removed_duplicates}
        corrected = [s for sec in result.contradictions.corrected_sentences for s in sec]
        assert result.merge.sentences == [s for s in corrected if s not in removed]
    for x in result.contradictions.all_contradictions:
        if x.rejected is not None:
            assert x.rejected.text not in result.merge.sentences, "a rejected claim reached the summary"


def test_pipeline_invariants_hold():
    cfg = load_config(overrides=["summarization.passthrough_tokens=1000"])
    result = HCVSumPipeline(cfg, fake_registry(cfg)).run(DOC, "doc.md", diagnose_sources=True)
    check_invariants(result)


def test_pipeline_invariants_hold_in_flag_mode():
    cfg = load_config(overrides=["summarization.passthrough_tokens=1000", "contradiction.action=flag",
                                 "merging.mode=abstractive"])
    result = HCVSumPipeline(cfg, fake_registry(cfg)).run(DOC, "doc.md")
    assert len(result.provenance) == len(result.merge.sentences)
    offset = 0
    for sec in result.sections:
        assert sec.sentence_offset == offset
        offset += len(sec.sentences)
    # In flag mode nothing is deleted, so every corrected claim is still present.
    corrected = [s for sec in result.contradictions.corrected_sentences for s in sec]
    assert len(corrected) == sum(len(ss.sentences) for ss in result.section_summaries)


def test_pipeline_invariants_on_awkward_input():
    """No headings, no terminal punctuation, duplicate and boilerplate sentences."""
    doc = ("revenue grew four percent this quarter\n\n"
           "costs fell three percent overall\n\n"
           "Thank you.\n\n"
           "the plant closed for three weeks after a fire\n")
    cfg = load_config(overrides=["summarization.passthrough_tokens=1000"])
    result = HCVSumPipeline(cfg, fake_registry(cfg)).run(doc, "awkward.txt")
    check_invariants(result)
    # sentences without terminal punctuation must survive as separate claims
    assert len(result.merge.sentences) >= 3


# ------------------------------------------------------------------ evidence panel: one note per conflict
REPEATED = """# Plant report

## Operations
Sensor revenue fell because of the plant shutdown.
Repair costs were four million dollars.

## Regional operations
Sensor revenue fell because of the plant shutdown.
The Lyon warehouse was expanded during the spring.

## Revenue
Sensor revenue fell not because of the plant shutdown.
Automotive revenue grew four percent.
"""


def test_evidence_panel_lists_each_contradiction_once_per_sentence():
    """Regression: the panel printed the same Stage 3 note twice in a row under one sentence.

    The claim "Sensor revenue fell because of the plant shutdown." is in TWO sections' summaries here, so
    its single conflict with the Revenue section is flagged as two Contradiction objects, which used to
    render as two identical notes under the surviving final sentence (as seen on S4/S8 of the
    01_planted_contradiction run).
    """
    cfg = load_config(overrides=["summarization.passthrough_tokens=1000"])
    result = HCVSumPipeline(cfg, fake_registry(cfg)).run(REPEATED, "doc.md")
    flags = [c for c in result.contradictions.all_contradictions if "plant shutdown" in c.claim_a.text
             or "plant shutdown" in c.claim_b.text]
    assert len(flags) >= 2, "setup: the same conflict must be flagged from both sections holding the claim"

    for record in result.provenance:
        assert len(record.stage3_notes) == len(set(record.stage3_notes)), \
            f"S{record.index} repeats a Stage 3 note: {record.stage3_notes}"

    # The conflicting claim must be named once: not once per section that happened to hold the claim, and
    # not once per side of the pair (both sides resemble the sentence, since the fakes ignore the negation).
    disputed = [r for r in result.provenance if "plant shutdown" in r.sentence and "not because" not in r.sentence]
    assert disputed, "the disputed sentence survived into the final summary"
    notes = [n for n in disputed[0].stage3_notes if n.startswith("Stage 3 UNRESOLVED")]
    assert len(notes) == 1, f"one conflict, one note, got: {notes}"
    assert "fell not because of the plant shutdown" in notes[0], "the note names the other side"

    panel = render_result(result, show_stages=False)
    blocks = panel.split("\n  S")[1:]          # one block per final sentence
    assert blocks, "the evidence panel lists the final sentences"
    for block in blocks:
        lines = [line for line in block.splitlines() if "STAGE 3:" in line]
        assert len(lines) == len(set(lines)), f"a sentence lists the same note twice:\n{block}"
    shutdown = [b for b in blocks if "not because" not in b.split("\n")[1] and "plant shutdown" in b.split("\n")[1]]
    assert shutdown and shutdown[0].count("STAGE 3:") == 1, f"expected exactly one note:\n{shutdown}"


def test_wrapped_panel_labels_appear_once_per_entry():
    """Regression: a wrapped flag repeated its label, so "...(see the Stage 3 note" / "FLAG: below)" read
    as two flags, and a three-line Stage 3 note read as three notes."""
    from hcv_sum.report import _wrap
    lines = _wrap("DISPUTED: another section of the source contradicts this sentence " * 3, "    FLAG: ").splitlines()
    assert len(lines) > 1, "setup: the text must be long enough to wrap"
    assert lines[0].startswith("    FLAG: ")
    assert all(line.startswith(" " * len("    FLAG: ")) and "FLAG:" not in line for line in lines[1:])
