"""--brief: a few plain-language lines on what the run did, then the final summary, nothing else.

The wording is deliberately free of the pipeline's internal stage numbers (report.render_stage_lines);
--verbose keeps them. The numbers behind each line are unchanged, so these tests derive what they expect
from the finished run rather than hardcoding a phrase."""

import json
import re

import pytest

from hcv_sum import cli
from hcv_sum.config import load_config
from hcv_sum.pipeline import HCVSumPipeline
from hcv_sum.report import render_brief, render_stage_lines, render_summary

from test_end_to_end import DOC, VERBATIM, fake_registry, run_cli

# Two planted contradictions in different section pairs (the fake NLI flags a negation mismatch).
TWO = """# Report

## Operations
Sensor revenue fell because of the plant shutdown.
The Lyon warehouse was expanded during the spring.

## Revenue
Sensor revenue fell not because of the plant shutdown.
Automotive revenue grew four percent.

## Logistics
The Lyon warehouse was not expanded during the spring.
Freight costs were renegotiated.
"""
LINES = 4          # segmentation+summarization, contradictions, merge, provenance


def run(text, *overrides):
    cfg = load_config(overrides=[*VERBATIM, *overrides])
    return HCVSumPipeline(cfg, fake_registry(cfg)).run(text, "doc.md")


def split_brief(out: str):
    lines = out.rstrip("\n").split("\n")
    return lines[:LINES], lines[LINES], "\n".join(lines[LINES + 1:])


def test_brief_is_plain_lines_then_the_summary_and_nothing_else(tmp_path, monkeypatch, capsys):
    code, out, err = run_cli(tmp_path, monkeypatch, capsys, "--brief")
    _, default_out, _ = run_cli(tmp_path, monkeypatch, capsys)
    assert code == 0 and err == ""
    stages, blank, summary = split_brief(out)
    assert len(stages) == LINES
    assert blank == ""
    assert summary == default_out.rstrip("\n"), "after the plain lines, --brief prints exactly the default output"


def test_brief_never_mentions_stage_numbers_and_stays_short(tmp_path, monkeypatch, capsys):
    """The whole point of the wording: someone reading the result is not told about the 5-stage pipeline."""
    _, out, _ = run_cli(tmp_path, monkeypatch, capsys, "--brief")
    stages, _, _ = split_brief(out)
    for line in stages:
        assert not re.search(r"(?i)stage\s*[1-5]", line), line
        assert line.endswith("."), f"each line is a sentence: {line!r}"
        assert len(line) <= 160, f"still scannable, not a paragraph: {line!r}"
    assert stages[0].startswith("Split ") or stages[0].startswith("Summarized ")


def test_brief_output_equals_render_brief(tmp_path, monkeypatch, capsys):
    _, out, _ = run_cli(tmp_path, monkeypatch, capsys, "--brief")
    assert out.rstrip("\n") == render_brief(run(DOC))


def test_lines_report_the_runs_actual_counts():
    result = run(TWO)
    flags = result.contradictions.all_contradictions
    assert len(flags) == 2, "setup: two planted contradictions"
    sections, contradictions, merge, provenance = render_stage_lines(result)
    assert sections.startswith(f"Split the document into {len(result.sections)} sections")
    assert "summarized each one" in sections

    compared = result.contradictions.pairs_checked + result.contradictions.one_sided_checked
    assert f"Compared {compared} claim pairs across sections" in contradictions
    unresolved = sum(c.resolution == "unresolved" for c in flags)
    if unresolved == len(flags):        # one status category: it reads as an adjective
        assert f"found {unresolved} unresolved contradictions" in contradictions
    else:
        assert f"found {len(flags)} contradictions" in contradictions
        assert f"{unresolved} unresolved" in contradictions
    assert "kept in the summary and flagged below for review" in contradictions   # default action=flag

    # The dedup clause is part of the correct output whenever Stage 4 dropped near-duplicates.
    duplicates = len(result.merge.removed_duplicates)
    expected = f"Combined into a final summary of {len(result.merge.sentences)} sentences"
    if duplicates:
        expected += f"; {duplicates} near-duplicate{'' if duplicates == 1 else 's'} removed"
    assert merge == expected + "."

    supported = sum(r.status == "supported" for r in result.provenance)
    if supported == len(result.provenance):
        assert provenance.startswith("Every sentence traces back to the source")
    else:
        assert provenance.startswith(f"{supported} of {len(result.provenance)} sentences trace back to the source")
    disputed = sum(any(f.startswith("DISPUTED") for f in r.flags) for r in result.provenance)
    assert ("disputed by the contradictions above" if disputed else "none are disputed") in provenance


def test_lines_follow_the_data_not_a_template():
    result = run(DOC)
    flags = result.contradictions.all_contradictions
    assert len(flags) == 1, "setup: one planted contradiction"
    one = render_stage_lines(result)[1]
    unresolved = sum(c.resolution == "unresolved" for c in flags)
    assert (f"found {unresolved} unresolved contradiction" if unresolved == len(flags)
            else f"found {len(flags)} contradiction") in one

    none = render_stage_lines(run(DOC.replace(" not because", " because")))[1]
    assert "no contradictions were found" in none

    disabled = render_stage_lines(run(DOC, "contradiction.enabled=false"))
    assert len(disabled) == LINES and disabled[1] == "Contradiction checking was skipped."


def test_removal_mode_says_what_was_removed():
    line = render_stage_lines(run(TWO, "contradiction.action=remove"))[1]
    assert "removed as unsupported" in line and "kept in the summary" not in line


def test_brief_contains_no_sentences_scores_citations_or_timings(tmp_path, monkeypatch, capsys):
    _, out, _ = run_cli(tmp_path, monkeypatch, capsys, "--brief")
    stages, _, _ = split_brief(out)
    stage_text = "\n".join(stages)
    source_sentences = [line.strip() for line in DOC.splitlines() if line.strip() and not line.startswith("#")]
    assert not any(s in stage_text for s in source_sentences), "no source or summary sentence in the plain lines"
    assert not re.search(r"\d\.\d", stage_text), "no scores, probabilities or timings (decimals)"
    for marker in ("P(contradiction)", "entailment", "cites", "source:", "EVIDENCE PANEL", "STAGE", "[section",
                   "seconds", "RUN LIMITS", "support("):
        assert marker not in out


def test_brief_with_json_still_writes_the_complete_run(tmp_path, monkeypatch, capsys):
    brief_json, default_json = tmp_path / "brief.json", tmp_path / "default.json"
    code, out, _ = run_cli(tmp_path, monkeypatch, capsys, "--brief", "--json", str(brief_json))
    run_cli(tmp_path, monkeypatch, capsys, "--json", str(default_json))
    assert code == 0 and out.startswith("Split the document into")
    brief_run = json.loads(brief_json.read_text(encoding="utf-8"))
    default_run = json.loads(default_json.read_text(encoding="utf-8"))
    for key in ("sections", "section_summaries", "contradictions", "merge", "provenance", "stats"):
        assert brief_run[key], key
    strip = lambda d: {k: v for k, v in d.items() if k not in ("timings", "stats")}   # noqa: E731
    assert strip(brief_run) == strip(default_run), "the JSON is the same full run in either mode"


def test_brief_and_verbose_are_rejected_together(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text(DOC, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.main([str(doc), "--brief", "--verbose"])
    assert exc.value.code == 2
    assert "not allowed with argument --brief" in capsys.readouterr().err


def test_brief_with_evidence_prefixes_the_plain_lines(tmp_path, monkeypatch, capsys):
    _, both, _ = run_cli(tmp_path, monkeypatch, capsys, "--brief", "--evidence")
    _, evidence, _ = run_cli(tmp_path, monkeypatch, capsys, "--evidence")
    assert both.startswith("Split the document into")
    assert both.endswith(evidence), "--evidence output itself is unchanged"


def test_other_modes_are_unchanged(tmp_path, monkeypatch, capsys):
    """--brief is additive: default output is still render_summary, and no other mode gains the plain lines."""
    _, default_out, _ = run_cli(tmp_path, monkeypatch, capsys)
    assert default_out.strip() == render_summary(run(DOC)).strip()
    first_line = render_stage_lines(run(DOC))[0]
    for flags in ((), ("--evidence",), ("--verbose",)):
        _, out, _ = run_cli(tmp_path, monkeypatch, capsys, *flags)
        assert first_line not in out


def test_verbose_still_uses_the_stage_structure(tmp_path, monkeypatch, capsys):
    """Requirement: --verbose is for debugging and keeps its stage headings."""
    _, out, _ = run_cli(tmp_path, monkeypatch, capsys, "--verbose")
    assert "STAGE 1 - SEGMENTATION" in out and "STAGE 3a - SIBLING CONTRADICTION CHECK" in out
