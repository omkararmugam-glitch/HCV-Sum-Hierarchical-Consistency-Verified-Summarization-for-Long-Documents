"""Plain-text rendering of a pipeline run (every stage + evidence panel) and JSON export."""

from __future__ import annotations

import json
import textwrap
from collections import Counter
from pathlib import Path

from .types import ContradictionReport, PipelineResult

WIDTH = 100


def _wrap(text: str, indent: str = "    ") -> str:
    """Wrap ``text`` at WIDTH. A labelled indent ("    FLAG: ") labels the FIRST line only.

    Continuation lines are padded to the same width with spaces. Repeating the label instead, as this used
    to, reads as several entries where there is one: a two-line flag came out as "FLAG: ... (see the Stage 3
    note" / "FLAG: below)", and a three-line Stage 3 note looked like three separate notes.
    """
    return textwrap.fill(text, width=WIDTH, initial_indent=indent, subsequent_indent=" " * len(indent),
                         break_long_words=False, break_on_hyphens=False)


def _rule(title: str, char: str = "=") -> str:
    return f"\n{char * WIDTH}\n{title}\n{char * WIDTH}"


def _contradiction_lines(report: ContradictionReport, titles: dict[int, str], one_sided: bool = False) -> list[str]:
    if one_sided:
        items = report.one_sided
        lines = ["  (summary claim A vs SOURCE sentence B of a sibling section that is not in B's summary)",
                 f"  pairs checked after similarity prefilter: {report.one_sided_checked} | flagged: {len(items)}"]
    else:
        items = report.contradictions
        lines = [f"  pairs across sections: {report.pairs_total} | checked after similarity prefilter: "
                 f"{report.pairs_checked} | flagged: {len(items)}"]
    for k, c in enumerate(items, 1):
        lines.append(f"\n  [{k}] P(contradiction)={c.score:.3f}  (A->B {c.score_ab:.3f}, B->A {c.score_ba:.3f})")
        lines.append(_wrap(f"A [section {c.claim_a.section_index}: {titles[c.claim_a.section_index]}] {c.claim_a.text}"))
        lines.append(_wrap(f"B [section {c.claim_b.section_index}: {titles[c.claim_b.section_index]}] {c.claim_b.text}"))
        if c.resolution != "diagnostic":
            lines.append(_wrap(f"support(A | own source) = {c.support_a:.3f}   evidence: {c.evidence_a}", "      "))
            lines.append(_wrap(f"support(B | own source) = {c.support_b:.3f}   evidence: {c.evidence_b}", "      "))
        lines.append(_wrap(f"=> {c.resolution.upper()}: {c.reason}", "    "))
    return lines


def render_limits(result: PipelineResult, only_hits: bool = False) -> str:
    """Which caps, ceilings and size-based switches this run hit, so a constrained result is visible."""
    limits = (result.stats or {}).get("limits", [])
    shown = [x for x in limits if x["hit"] or not only_hits]
    s = result.stats or {}
    head = (f"RUN LIMITS: {s.get('sections', '?')} sections, {s.get('source_sentences', '?')} source sentences, "
            f"{s.get('final_sentences', '?')} final sentences, {s.get('seconds_total', '?')} s, peak memory "
            f"{s.get('process_peak_rss_mb') or max((v.get('peak_rss_mb', 0) for v in s.get('per_stage', {}).values()), default=0)} MB")
    hits = sum(x["hit"] for x in limits)
    lines = [head, f"  {hits} of {len(limits)} limits hit" + (" -- the result is COMPLETE within every limit" if not hits else "")]
    for x in shown:
        lines.append(_wrap(f"[{'HIT' if x['hit'] else 'ok '}] {x['stage']:<17} {x['limit']}: {x['detail']}", "  "))
    return "\n".join(lines)


def render_summary(result: PipelineResult) -> str:
    """Just the final summary text: the default CLI output, safe to pipe into another tool."""
    return result.merge.summary


def _n(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _was(count: int) -> str:
    return "was" if count == 1 else "were"


def _is(count: int) -> str:
    return "is" if count == 1 else "are"


def render_stage_lines(result: PipelineResult) -> list[str]:
    """What the run did, in four plain sentences, built only from the finished run's data (--brief).

    Counts only: no sentences, scores, citations or timings. Every number here is the one the pipeline
    recorded; the phrasing deliberately avoids the internal stage numbers, because --brief is read by
    someone who wants the result. --verbose keeps the stage structure, which is what it is for.
    Adaptive throughout: a step that did nothing, or was switched off, says so.
    """
    cfg = result.config or {}
    lines = []

    # 1. Segmentation + section summarization.
    sections = result.sections
    split = sum(1 for s in sections if s.origin.endswith("+split"))
    origins = {s.origin.split("+")[0] for s in sections}
    docs = (result.stats or {}).get("documents")
    subject = f"the {_n(len(docs), 'document')}" if docs else "the document"
    generated = sum(1 for s in result.section_summaries if s.generated)
    verbatim = len(result.section_summaries) - generated
    truncated = sum(1 for s in result.section_summaries if s.input_truncated)
    if origins == {"single"}:
        line = f"Summarized {subject} as a single section"
    else:
        how = (" along the headings" if origins == {"heading"} else
               " where the topic shifts" if origins == {"embedding"} else
               "" if origins == {"single"} else " using headings and topic shifts")
        line = f"Split {subject} into {_n(len(sections), 'section')}{how} and summarized each one"
    if split:
        line += f"; {split} of them {_was(split)} split further for length"
    if verbatim:
        line += f"; {verbatim} {_was(verbatim)} short enough to keep unchanged"
    if truncated:
        line += f"; {truncated} {_was(truncated)} too long for the model and had to be cut"
    lines.append(line + ".")

    # 2. Cross-section contradiction checking.
    report = result.contradictions
    if not cfg.get("contradiction", {}).get("enabled", True):
        lines.append("Contradiction checking was skipped.")
    else:
        flags = report.all_contradictions
        compared = report.pairs_checked + report.one_sided_checked
        opening = f"Compared {_n(compared, 'claim pair')} across sections"
        if not flags:
            line = f"{opening}; no contradictions were found"
        else:
            status = Counter(c.resolution for c in flags)
            # (count, adjective, phrase): the adjective, where one reads naturally, goes straight in front
            # of "contradiction"; otherwise the phrase follows the count.
            found = [(status["unresolved"], "unresolved", "unresolved"),
                     (status["kept_a"] + status["kept_b"], None, "decided in favour of one side"),
                     (status["superseded"], "superseded", "superseded"),
                     (status["unscored"], None, "left unscored (the resolution budget ran out)"),
                     (status["possibly_superseded"], None, "possibly superseded (different periods)")]
            found = [item for item in found if item[0]]
            if len(found) == 1:
                count, adjective, phrase = found[0]
                line = (f"{opening} and found {_n(count, f'{adjective} contradiction')}" if adjective
                        else f"{opening} and found {_n(count, 'contradiction')}, all {phrase}")
            else:
                line = (f"{opening} and found {_n(len(flags), 'contradiction')}: "
                        + ", ".join(f"{n} {phrase}" for n, _adjective, phrase in found))
            if cfg.get("contradiction", {}).get("action") == "flag":
                line += "; kept in the summary and flagged below for review"
            else:
                removed = sum(1 for c in flags if c.rejected is not None)
                line += f"; {_n(removed, 'claim')} {_was(removed)} removed as unsupported"
        if result.source_diagnostic is not None:
            line += (f"; a further check of the raw source text flagged "
                     f"{len(result.source_diagnostic.contradictions)}")
        lines.append(line + ".")

    # 3. Merging into the final summary.
    merge = result.merge
    dup = len(merge.removed_duplicates)
    line = f"Combined into a final summary of {_n(len(merge.sentences), 'sentence')}"
    if merge.mode == "abstractive":
        line += f", rewritten in {_n(len(merge.rounds), 'round')}"
    elif merge.mode == "salience":
        line += ", the most central sentences across documents"
    if dup:
        line += f"; {_n(dup, 'near-duplicate')} removed"
    lines.append(line + ".")

    # 4. Sentence-level provenance.
    prov = result.provenance
    if not prov:
        lines.append("There were no final sentences to trace back to the source.")
    else:
        status = Counter(r.status for r in prov)
        disputed = sum(any(f.startswith("DISPUTED") for f in r.flags) for r in prov)
        if status["supported"] == len(prov):
            line = "Every sentence traces back to the source"
        else:
            line = f"{status['supported']} of {len(prov)} sentences trace back to the source"
        if status["weakly_supported"]:
            line += f", {status['weakly_supported']} only weakly"
        if status["unsupported"]:
            line += f", {status['unsupported']} not at all"
        line += (f"; {disputed} {_is(disputed)} disputed by the contradictions above" if disputed
                 else "; none are disputed")
        lines.append(line + ".")
    return lines


def render_brief(result: PipelineResult) -> str:
    """--brief: a few plain lines saying what happened, a blank line, then the final summary. Nothing else."""
    return "\n".join(render_stage_lines(result)) + "\n\n" + render_summary(result)


def render_result(result: PipelineResult, show_stages: bool = True) -> str:
    out: list[str] = []
    titles = {s.index: (s.title or "(untitled)") for s in result.sections}
    out.append(_rule(f"HCV-Sum run: {result.document_name}"))

    if show_stages:
        out.append(_rule("STAGE 1 - SEGMENTATION", "-"))
        origins = Counter(s.origin for s in result.sections)
        out.append(f"  {len(result.sections)} sections, origin: {dict(origins)}")
        for s in result.sections:
            out.append(f"\n  [section {s.index}] {titles[s.index]}  "
                       f"(origin={s.origin}, sentences {s.sentence_offset}-{s.sentence_offset + len(s.sentences) - 1})")
            for j, sent in enumerate(s.sentences):
                out.append(_wrap(f"{s.sentence_offset + j:>3}: {sent}"))

        out.append(_rule("STAGE 2 - CONTEXT-ANCHORED SECTION SUMMARIES", "-"))
        for ss in result.section_summaries:
            tag = "generated" if ss.generated else "passthrough (section too short to abstract)"
            out.append(f"\n  [section {ss.section_index}] {titles[ss.section_index]}  -- {tag}"
                       f"{'  [INPUT TRUNCATED]' if ss.input_truncated else ''}")
            if ss.context:
                out.append("    retrieved context:")
                for c in ss.context:
                    out.append(_wrap(f"(sim {c.similarity:.2f}, from section {c.section_index}) {c.text}", "      - "))
            else:
                out.append("    retrieved context: none above similarity floor")
            for sent, why in ss.dropped:
                out.append(_wrap(f"DROPPED [{why}]: {sent}", "    x "))
            for note in ss.notes:
                out.append(_wrap(note, "    ! "))
            out.append("    summary:")
            for j, sent in enumerate(ss.sentences):
                leak = "  [CONTEXT LEAK: matches retrieved context better than own section]" if j in ss.context_leaks else ""
                out.append(_wrap(f"{j}: {sent}{leak}", "      "))

        out.append(_rule("STAGE 3a - SIBLING CONTRADICTION CHECK (section summary vs section summary)", "-"))
        out.extend(_contradiction_lines(result.contradictions, titles))
        out.append(_rule("STAGE 3b - ONE-SIDED CHECK (section summary vs sibling section SOURCE)", "-"))
        out.extend(_contradiction_lines(result.contradictions, titles, one_sided=True))
        if result.contradictions.flagged:
            out.append("\n  flagged claims kept in the summary:")
            for key, note in result.contradictions.flagged.items():
                out.append(_wrap(f"{key}: {note}", "    "))

        if result.source_diagnostic is not None:
            out.append(_rule("STAGE 3 DIAGNOSTIC - same detector over raw SOURCE sentences (nothing removed)", "-"))
            out.extend(_contradiction_lines(result.source_diagnostic, titles))

        out.append(_rule("STAGE 4 - MERGE", "-"))
        m = result.merge
        out.append(f"  mode={m.mode} | input sentences={len(m.input_sentences)} | "
                   f"removed near-duplicates={len(m.removed_duplicates)} | merge rounds={len(m.rounds)}")
        for removed, dup, sim in m.removed_duplicates:
            out.append(_wrap(f"dropped (sim {sim:.2f}): {removed}  ==  {dup}", "    - "))
        for rnd in m.rounds:
            out.append(f"  round {rnd.round_number}: {len(rnd.inputs)} group(s) -> {len(rnd.outputs)} output(s)"
                       + (f", {len(rnd.introduced)} contradiction(s) introduced by this round"
                          if rnd.introduced else ""))
            for c in rnd.introduced:
                out.append(_wrap(f"INTRODUCED BY MERGE (P={c.score:.2f}): \"{c.claim_a.text}\" vs "
                                 f"\"{c.claim_b.text}\"", "    ! "))
        for claim_final, claim_rejected, ent in m.resurrected:
            out.append(_wrap(f"RESURRECTED rejected claim (entail {ent:.2f}): {claim_final}  <=  {claim_rejected}",
                             "    ! "))

    out.append(_rule("FINAL SUMMARY"))
    out.append(_wrap(result.merge.summary, "  "))

    out.append(_rule("EVIDENCE PANEL"))
    counts = Counter(r.status for r in result.provenance)
    disputed = sum(any(f.startswith("DISPUTED") for f in r.flags) for r in result.provenance)
    out.append(f"  {dict(counts)}" + (f" | disputed by another section: {disputed}" if disputed else ""))
    for r in result.provenance:
        out.append(f"\n  S{r.index} [{r.status.upper()}]  entailment={r.entailment:.3f}  best similarity={r.similarity:.3f}")
        out.append(_wrap(r.sentence, "    \""))
        if r.citation:
            c = r.citation
            ids = ",".join(map(str, c.sentence_indices))
            out.append(_wrap(f"cites source sentence(s) {ids} in section {c.section_index} ({c.section_title or 'untitled'}): "
                             f"{c.text}", "    source: "))
        for flag in r.flags:
            out.append(_wrap(flag, "    FLAG: "))
        for note in r.stage3_notes:
            out.append(_wrap(note, "    STAGE 3: "))

    rejected = [c for c in result.contradictions.all_contradictions if c.rejected is not None]
    if rejected:
        out.append("\n  Claims rejected in Stage 3 (not in the summary):")
        for c in rejected:
            out.append(_wrap(f"\"{c.rejected.text}\" (section {c.rejected.section_index}) -- {c.reason}", "    - "))

    if show_stages and result.stats:
        out.append(_rule("RUN STATS", "-"))
        s = result.stats
        out.append(f"  {s['sections']} sections | {s['source_sentences']} source sentences | "
                   f"{s['summary_claims']} summary claims | {s['final_sentences']} final sentences")
        out.append(f"  work: {s['summarizer_generations']} summarizer generations, "
                   f"{s['nli_pairs_total']} NLI sentence pairs, {s['embedded_texts_total']} texts embedded")
        for name, counts in s["per_stage"].items():
            out.append(f"    {name:<30} {counts['seconds']:>7.2f}s  nli_pairs={counts['nli_pairs']:<7}"
                       f" generations={counts['generations']}")
        for key, label in (("stage3a_pairs", "Stage 3a (summary vs summary)"),
                           ("stage3b_pairs", "Stage 3b (summary vs sibling source)"),
                           ("stage3_diagnostic_pairs", "Stage 3 source diagnostic")):
            p = s.get(key)
            if not p:
                continue
            out.append(f"    {label}: compared {p['compared']} of {p['possible']} possible pairs "
                       f"({p['nli_calls']} NLI calls)")
            if key == "stage3a_pairs" and result.contradictions.unscored:
                out.append(f"      {result.contradictions.unscored} flagged pairs left UNSCORED "
                           f"(max_resolutions budget); they stay flagged and nothing was removed for them")
            out.append(_wrap(f"not compared: {p['skipped_low_similarity']} below similarity floor, "
                             f"{p['skipped_candidate_top_k']} beyond candidate_top_k, "
                             f"{p['skipped_max_pairs_budget']} over the max_pairs budget, "
                             f"{p.get('skipped_comparative_framing', 0)} framed as a comparison, "
                             f"{p.get('skipped_no_shared_entity', 0)} without a shared entity, "
                             f"{p.get('skipped_no_shared_content_word', 0)} without a shared content word",
                             "      "))

    if result.stats and result.stats.get("limits"):
        out.append(_rule("RUN LIMITS", "-"))
        out.append(render_limits(result))
    out.append(f"\n  timings (s): {result.timings}")
    return "\n".join(out)


def save_json(result: PipelineResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)
